"""Leakage-safe modern baselines and their shared protocol contracts.

The TimeMixer++ baseline in this module is a paper-derived mechanism
reproduction built from local PyTorch modules. It is not official TimeMixer++
source code. Timer-XL adapts an identity-audited official OpenLTM backbone;
RQR-TFT is a paper-derived protocol reproduction built from local TFT-style
modules and does not use official implementation code.
"""

from __future__ import annotations

import _imp
import hashlib
import importlib.util
import io
import math
import re
import subprocess
import sys
import tarfile
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from racepv.losses import trapezoidal_qcrps

from .neural import OrderedQuantileProjection

MODERN_BASELINES = ("timemixerpp_qr", "timer_xl_qr", "rqr_tft")
REGISTERED_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
_FULL_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GITHUB_REMOTE = re.compile(
    r"^(?:https://github\.com/|ssh://git@github\.com/|git@github\.com:)(?P<path>.+)$"
)
_OPENLTM_TIMER_XL_PATH = Path("models/timer_xl.py")
_OPENLTM_NAME = "timer_xl_qr"
_OPENLTM_REPOSITORY_URL = "https://github.com/thuml/OpenLTM.git"
_OPENLTM_IMPLEMENTATION_KIND = "official_backbone_protocol_adaptation"
_OPENLTM_LICENSE_NAMES = ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "COPYING.md")
_OPENLTM_IMPORT_LOCK = threading.RLock()


@dataclass(frozen=True)
class ModernBaselineSpec:
    name: str
    implementation_kind: str
    actual_backbone: str
    source_name: str
    official_code_used: bool


@dataclass(frozen=True)
class RampClassWeight:
    """Training-label counts and the resulting RQR positive-class weight."""

    stable_count: int
    ramp_count: int
    pos_weight: float

    def as_metadata(self) -> dict[str, int | float]:
        return {
            "stable_count": self.stable_count,
            "ramp_count": self.ramp_count,
            "ramp_pos_weight": self.pos_weight,
        }


def ramp_class_weight_from_training_labels(
    state_labels: Tensor,
    *,
    daylight_mask: Tensor | None = None,
) -> RampClassWeight:
    """Compute ``sqrt(N_stable / N_ramp)`` from training daylight labels.

    Labels outside ``{0, 1, 2}`` are unlabelled.  A training sample with no
    ramp labels receives the finite neutral weight 1.0; the returned counts
    preserve that fact for run metadata and review.
    """

    if daylight_mask is not None and daylight_mask.shape != state_labels.shape:
        raise ValueError("daylight_mask must match state_labels")
    eligible = torch.ones_like(state_labels, dtype=torch.bool)
    if daylight_mask is not None:
        eligible &= daylight_mask.bool()
    stable_count = int((eligible & (state_labels == 0)).sum().item())
    ramp_count = int(
        (eligible & torch.isin(state_labels, state_labels.new_tensor((1, 2)))).sum().item()
    )
    pos_weight = math.sqrt(stable_count / ramp_count) if ramp_count > 0 else 1.0
    return RampClassWeight(stable_count, ramp_count, pos_weight)


class ModernCovariateTokenBuilder(nn.Module):
    """Build a leakage-safe history-plus-future token sequence.

    Each token has ``[masked_history, history_mask, future_known, site]``
    fields.  In the history region the future fields are zero; in the forecast
    region both history fields are zero.  Future observations are not accepted
    by this interface and therefore cannot enter the tokens.
    """

    def __init__(
        self,
        input_dim: int,
        future_dim: int,
        site_dim: int,
        *,
        horizon: int = 24,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.future_dim = int(future_dim)
        self.site_dim = int(site_dim)
        self.horizon = int(horizon)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.future_dim <= 0:
            raise ValueError("future_dim must be positive")
        if self.site_dim < 0:
            raise ValueError("site_dim cannot be negative")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        self.token_dim = 2 * self.input_dim + self.future_dim + self.site_dim

    def forward(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor | None = None,
    ) -> Tensor:
        if history_values.ndim != 3 or history_values.shape[-1] != self.input_dim:
            raise ValueError("history_values must have shape [batch, history, input_dim]")
        if history_mask.shape != history_values.shape:
            raise ValueError("history_mask must have the same shape as history_values")
        batch_size, history_length, _ = history_values.shape
        if (
            future_known.ndim != 3
            or future_known.shape[0] != batch_size
            or future_known.shape[1] != self.horizon
            or future_known.shape[2] != self.future_dim
        ):
            raise ValueError(
                "future_known must have shape "
                f"[batch, horizon={self.horizon}, future_dim={self.future_dim}]"
            )
        if self.site_dim:
            if site_metadata is None or site_metadata.shape != (batch_size, self.site_dim):
                raise ValueError(f"site_metadata must have shape [batch, site_dim={self.site_dim}]")
            site = site_metadata.to(
                dtype=history_values.dtype,
                device=history_values.device,
            )
        else:
            if site_metadata is not None and site_metadata.shape != (batch_size, 0):
                raise ValueError("site_metadata must be absent when site_dim is zero")
            site = history_values.new_zeros(batch_size, 0)
        if future_known.device != history_values.device:
            raise ValueError("future_known and history_values must be on the same device")

        valid_history = history_mask.bool()
        masked_history = torch.where(
            valid_history,
            history_values,
            torch.zeros_like(history_values),
        )
        history_tokens = torch.cat(
            (
                masked_history,
                valid_history.to(history_values.dtype),
                history_values.new_zeros(batch_size, history_length, self.future_dim),
                site[:, None, :].expand(-1, history_length, -1),
            ),
            dim=-1,
        )
        future_tokens = torch.cat(
            (
                history_values.new_zeros(
                    batch_size,
                    self.horizon,
                    2 * self.input_dim,
                ),
                future_known.to(dtype=history_values.dtype),
                site[:, None, :].expand(-1, self.horizon, -1),
            ),
            dim=-1,
        )
        return torch.cat((history_tokens, future_tokens), dim=1)


class ModernQuantilePostprocessor(nn.Module):
    """Apply the one shared ordering, physical-range, and night-zero rule."""

    def __init__(
        self,
        quantiles: Sequence[float] = REGISTERED_LEVELS,
        *,
        horizon: int = 24,
        clear_sky_feature_index: int | None = 0,
        daylight_feature_index: int | None = 1,
        maximum_power: float = 1.05,
    ) -> None:
        super().__init__()
        self.quantiles = tuple(float(level) for level in quantiles)
        self.horizon = int(horizon)
        self.clear_sky_feature_index = clear_sky_feature_index
        self.daylight_feature_index = daylight_feature_index
        self.maximum_power = float(maximum_power)
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if not math.isfinite(self.maximum_power) or not (0 < self.maximum_power <= 1.05):
            raise ValueError("maximum_power must be finite and within (0, 1.05]")
        if clear_sky_feature_index is None and daylight_feature_index is None:
            raise ValueError("night zeroing requires clear-sky or daylight information")
        self.ordered = OrderedQuantileProjection(self.quantiles)

    def forward(self, raw: Tensor, *, future_known: Tensor) -> Tensor:
        if future_known.ndim != 3:
            raise ValueError("future_known must be three-dimensional [batch, horizon, features]")
        expected = (future_known.shape[0], self.horizon, len(self.quantiles))
        if raw.ndim != 3 or raw.shape != expected:
            raise ValueError(
                "raw quantiles must have shape "
                f"[batch, horizon={self.horizon}, quantiles={len(self.quantiles)}]"
            )
        if future_known.shape[1] != self.horizon:
            raise ValueError(
                f"future_known must have shape [batch, horizon={self.horizon}, features]"
            )
        if raw.device != future_known.device:
            raise ValueError("raw and future_known must be on the same device")

        daylight = torch.ones(
            future_known.shape[:2],
            dtype=torch.bool,
            device=future_known.device,
        )
        for name, index in (
            ("clear_sky_feature_index", self.clear_sky_feature_index),
            ("daylight_feature_index", self.daylight_feature_index),
        ):
            if index is None:
                continue
            if index < 0 or index >= future_known.shape[-1]:
                raise ValueError(f"{name} is outside the future_known feature dimension")
            signal = future_known[..., index]
            daylight &= torch.isfinite(signal) & (signal > 0)

        finite_raw = torch.nan_to_num(
            raw,
            nan=0.0,
            posinf=self.maximum_power,
            neginf=0.0,
        )
        ordered = self.ordered(finite_raw).clamp(0.0, self.maximum_power)
        return torch.where(daylight[..., None], ordered, torch.zeros_like(ordered))


class MultiScaleProjection(nn.Module):
    """Project one Conv1d-downsampled history scale into model tokens."""

    def __init__(self, input_dim: int, d_model: int, factor: int) -> None:
        super().__init__()
        self.factor = int(factor)
        self.projection = nn.Conv1d(
            input_dim,
            d_model,
            kernel_size=self.factor,
            stride=self.factor,
        )
        self.normalization = nn.LayerNorm(d_model)

    def forward(self, inputs: Tensor) -> Tensor:
        projected = self.projection(inputs.transpose(1, 2)).transpose(1, 2)
        return self.normalization(F.gelu(projected))


class TimeImageResolutionBranch(nn.Module):
    """Mix one period grid with seasonal and trend attention on opposite axes."""

    def __init__(self, d_model: int, period: int, dropout: float) -> None:
        super().__init__()
        self.period = int(period)
        attention_heads = 4 if d_model % 4 == 0 else 1
        self.season_attention = nn.MultiheadAttention(
            d_model,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.trend_attention = nn.MultiheadAttention(
            d_model,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.residual_normalization = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        batch_size, token_count, d_model = tokens.shape
        padding = (-token_count) % self.period
        padded = F.pad(tokens, (0, 0, 0, padding))
        cycles = padded.shape[1] // self.period
        grid = padded.reshape(batch_size, cycles, self.period, d_model)

        trend = grid.mean(dim=2, keepdim=True).expand(-1, -1, self.period, -1)
        season = grid - trend
        season_axis = season.reshape(batch_size * cycles, self.period, d_model)
        season_mixed, _ = self.season_attention(
            season_axis,
            season_axis,
            season_axis,
            need_weights=False,
        )
        season_mixed = season_mixed.reshape(batch_size, cycles, self.period, d_model)

        trend_axis = trend.permute(0, 2, 1, 3).reshape(
            batch_size * self.period,
            cycles,
            d_model,
        )
        trend_mixed, _ = self.trend_attention(
            trend_axis,
            trend_axis,
            trend_axis,
            need_weights=False,
        )
        trend_mixed = trend_mixed.reshape(
            batch_size,
            self.period,
            cycles,
            d_model,
        ).permute(0, 2, 1, 3)

        mixed = self.residual_normalization(grid + self.dropout(season_mixed + trend_mixed))
        return mixed.reshape(batch_size, padded.shape[1], d_model)[:, :token_count]


class TimeImageMixer(nn.Module):
    """Apply and learnably aggregate every configured time-image resolution."""

    def __init__(self, d_model: int, periods: Sequence[int], dropout: float) -> None:
        super().__init__()
        self.resolution_branches = nn.ModuleList(
            TimeImageResolutionBranch(d_model, period, dropout) for period in periods
        )
        self.resolution_logits = nn.Parameter(torch.zeros(len(periods)))
        self.dropout = nn.Dropout(dropout)
        self.output_normalization = nn.LayerNorm(d_model)

    def forward(self, tokens: Sequence[Tensor]) -> list[Tensor]:
        weights = self.resolution_logits.softmax(dim=0)
        mixed_scales = []
        for token in tokens:
            branches = torch.stack([branch(token) for branch in self.resolution_branches])
            mixed = (branches * weights[:, None, None, None]).sum(dim=0)
            mixed_scales.append(self.output_normalization(token + self.dropout(mixed)))
        return mixed_scales


class FutureQuantileDecoder(nn.Module):
    """Decode scale-aggregated context with known future and site covariates."""

    def __init__(
        self,
        d_model: int,
        future_dim: int,
        site_dim: int,
        horizon: int,
        quantiles: Sequence[float],
    ) -> None:
        super().__init__()
        self.site_dim = int(site_dim)
        self.horizon = int(horizon)
        self.quantiles = tuple(float(level) for level in quantiles)
        self.future_projection = nn.Linear(future_dim, d_model, bias=False)
        self.site_projection = (
            nn.Linear(self.site_dim, d_model, bias=False) if self.site_dim else None
        )
        self.horizon_embedding = nn.Parameter(torch.empty(1, self.horizon, d_model))
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )
        self.output_projection = nn.Linear(d_model, len(self.quantiles))
        self.postprocess = ModernQuantilePostprocessor(
            self.quantiles,
            horizon=self.horizon,
            clear_sky_feature_index=0,
            daylight_feature_index=1,
        )
        nn.init.normal_(self.horizon_embedding, std=1.0 / math.sqrt(d_model))
        nn.init.normal_(self.output_projection.weight, std=0.02)
        nn.init.constant_(self.output_projection.bias, -3.0)
        median_index = self.quantiles.index(0.5)
        with torch.no_grad():
            self.output_projection.bias[median_index] = 0.5

    def forward(
        self,
        context: Tensor,
        future_known: Tensor,
        site_metadata: Tensor | None,
    ) -> Tensor:
        if context.ndim == 2:
            context_tokens = context[:, None, :]
        elif context.ndim == 3 and context.shape[1] == self.horizon:
            context_tokens = context
        else:
            raise ValueError(
                "context must have shape [batch, d_model] or "
                f"[batch, horizon={self.horizon}, d_model]"
            )
        decoded = context_tokens + self.future_projection(future_known)
        decoded = decoded + self.horizon_embedding
        if self.site_projection is not None:
            if site_metadata is None:
                raise ValueError("site_metadata is required for the configured site dimension")
            decoded = decoded + self.site_projection(site_metadata)[:, None, :]
        raw = self.output_projection(self.fusion(decoded))
        return self.postprocess(raw, future_known=future_known)


class TimeMixerPPQuantile(nn.Module):
    """Paper-derived TimeMixer++ mechanism reproduction for quantile forecasting."""

    spec = ModernBaselineSpec(
        name="timemixerpp_qr",
        implementation_kind="paper_derived_protocol_reproduction",
        actual_backbone="timemixerpp_paper_reproduction",
        source_name="timemixerpp_qr",
        official_code_used=False,
    )

    def __init__(
        self,
        input_dim: int,
        future_dim: int,
        site_dim: int,
        history_length: int,
        horizon: int,
        quantiles: Sequence[float],
        *,
        d_model: int,
        layers: int,
        downsampling_scales: Sequence[int],
        periods: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        history_length = int(history_length)
        horizon = int(horizon)
        d_model = int(d_model)
        layers = int(layers)
        scales = self._validated_integer_sequence(
            "downsampling_scales",
            downsampling_scales,
        )
        period_values = self._validated_integer_sequence("periods", periods)
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if layers <= 0:
            raise ValueError("layers must be positive")
        if not math.isfinite(dropout) or not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be finite and within [0, 1)")
        if future_dim < 2:
            raise ValueError("future_dim must include clear-sky and daylight features")
        if tuple(float(level) for level in quantiles) != REGISTERED_LEVELS:
            raise ValueError("TimeMixer++-QR requires the registered seven quantiles")
        if any(scale > history_length for scale in scales):
            raise ValueError("downsampling_scales cannot exceed history_length")
        projected_lengths = tuple(history_length // scale for scale in scales)
        if any(period > min(projected_lengths) for period in period_values):
            raise ValueError("periods must produce a non-empty grid at every scale")

        self.input_dim = int(input_dim)
        self.history_length = history_length
        self.token_builder = ModernCovariateTokenBuilder(
            input_dim,
            future_dim,
            site_dim,
            horizon=horizon,
        )
        self.scales = nn.ModuleList(
            MultiScaleProjection(2 * input_dim, d_model, factor) for factor in scales
        )
        self.mixers = nn.ModuleList(
            TimeImageMixer(d_model, period_values, dropout) for _ in range(layers)
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(scales)))
        self.decoder = FutureQuantileDecoder(
            d_model,
            future_dim,
            site_dim,
            horizon,
            quantiles,
        )

    @staticmethod
    def _validated_integer_sequence(name: str, values: Sequence[int]) -> tuple[int, ...]:
        result = tuple(values)
        if not result:
            raise ValueError(f"{name} cannot be empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in result
        ):
            raise ValueError(f"{name} must contain positive integers")
        if len(set(result)) != len(result):
            raise ValueError(f"{name} must not contain duplicates")
        return result

    def forward(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if history_values.ndim != 3 or history_values.shape[1] != self.history_length:
            raise ValueError(f"history_values must have history_length={self.history_length}")
        shared_tokens = self.token_builder(
            history_values,
            history_mask,
            future_known,
            site_metadata,
        )
        encoded = shared_tokens[:, : self.history_length, : 2 * self.input_dim]
        tokens = [projection(encoded) for projection in self.scales]
        for mixer in self.mixers:
            tokens = mixer(tokens)
        scale_contexts = torch.stack([token.mean(dim=1) for token in tokens])
        weights = self.scale_logits.softmax(dim=0)
        context = (scale_contexts * weights[:, None, None]).sum(dim=0)
        quantiles = self.decoder(context, future_known, site_metadata)
        return {"quantiles": quantiles}


class TFTGatedResidualNetwork(nn.Module):
    """TFT-style nonlinear transform with a gated residual skip connection."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.hidden_projection = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gate_projection = nn.Linear(hidden_size, 2 * output_size)
        self.skip_projection = (
            nn.Identity() if input_size == output_size else nn.Linear(input_size, output_size)
        )
        self.normalization = nn.LayerNorm(output_size)

    def forward(self, inputs: Tensor, residual: Tensor | None = None) -> Tensor:
        transformed = F.elu(self.input_projection(inputs))
        transformed = self.dropout(self.hidden_projection(transformed))
        gated = F.glu(self.gate_projection(transformed), dim=-1)
        skip = self.skip_projection(inputs) if residual is None else residual
        if skip.shape != gated.shape:
            raise ValueError("explicit gated-residual skip must match the transformed shape")
        return self.normalization(skip + gated)


class TFTVariableSelectionNetwork(nn.Module):
    """Select scalar covariates with learned, time-varying TFT-style weights."""

    def __init__(self, variable_count: int, hidden_size: int, *, dropout: float) -> None:
        super().__init__()
        self.variable_count = int(variable_count)
        self.variable_transforms = nn.ModuleList(
            TFTGatedResidualNetwork(1, hidden_size, hidden_size, dropout=dropout)
            for _ in range(self.variable_count)
        )
        self.selection_weights = TFTGatedResidualNetwork(
            self.variable_count,
            hidden_size,
            self.variable_count,
            dropout=dropout,
        )

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.variable_count:
            raise ValueError(
                "variable-selection inputs must have shape "
                f"[batch, time, variables={self.variable_count}]"
            )
        weights = self.selection_weights(inputs).softmax(dim=-1)
        transformed = torch.stack(
            [
                transform(inputs[..., index : index + 1])
                for index, transform in enumerate(self.variable_transforms)
            ],
            dim=-2,
        )
        selected = (transformed * weights[..., None]).sum(dim=-2)
        return selected, weights


class InterpretableMultiHeadTemporalAttention(nn.Module):
    """TFT interpretable attention with one shared value projection across heads."""

    def __init__(self, hidden_size: int, attention_heads: int, *, dropout: float) -> None:
        super().__init__()
        if hidden_size % attention_heads != 0:
            raise ValueError("hidden_size must be divisible by attention_heads")
        self.hidden_size = int(hidden_size)
        self.attention_heads = int(attention_heads)
        self.head_size = self.hidden_size // self.attention_heads
        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.shared_value_projection = nn.Linear(hidden_size, self.head_size, bias=False)
        self.output_projection = nn.Linear(self.head_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        if query.ndim != 3 or memory.ndim != 3:
            raise ValueError("temporal attention expects three-dimensional query and memory")
        if query.shape[0] != memory.shape[0]:
            raise ValueError("temporal attention query and memory batches must match")
        if query.shape[-1] != self.hidden_size or memory.shape[-1] != self.hidden_size:
            raise ValueError("temporal attention features must match hidden_size")
        batch_size, query_length, _ = query.shape
        memory_length = memory.shape[1]
        projected_query = self.query_projection(query).reshape(
            batch_size,
            query_length,
            self.attention_heads,
            self.head_size,
        )
        projected_key = self.key_projection(memory).reshape(
            batch_size,
            memory_length,
            self.attention_heads,
            self.head_size,
        )
        projected_query = projected_query.transpose(1, 2)
        projected_key = projected_key.transpose(1, 2)
        scores = torch.matmul(projected_query, projected_key.transpose(-2, -1))
        weights = self.dropout((scores / math.sqrt(self.head_size)).softmax(dim=-1))
        shared_values = self.shared_value_projection(memory)
        attended_heads = torch.einsum("bhqt,btd->bhqd", weights, shared_values)
        attended = attended_heads.mean(dim=1)
        return self.output_projection(attended), weights


class RQRTFTQuantile(nn.Module):
    """Paper-derived RQR-TFT protocol reproduction with a shared TFT encoder."""

    spec = ModernBaselineSpec(
        name="rqr_tft",
        implementation_kind="paper_derived_protocol_reproduction",
        actual_backbone="rqr_tft_paper_reproduction",
        source_name="rqr_tft",
        official_code_used=False,
    )

    def __init__(
        self,
        input_dim: int,
        future_dim: int,
        site_dim: int,
        quantiles: Sequence[float],
        *,
        hidden_size: int,
        attention_heads: int,
        lstm_layers: int,
        dropout: float,
        ramp_loss_weight: float,
        ramp_weighted_quantile_multiplier: float,
        ramp_decision_threshold: float,
        ramp_pos_weight: float = 1.0,
        history_length: int = 192,
        horizon: int = 24,
    ) -> None:
        super().__init__()
        hidden_size = int(hidden_size)
        attention_heads = int(attention_heads)
        lstm_layers = int(lstm_layers)
        history_length = int(history_length)
        horizon = int(horizon)
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        if horizon != 24:
            raise ValueError("RQR-TFT requires the registered 24-step horizon")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if attention_heads <= 0:
            raise ValueError("attention_heads must be positive")
        if hidden_size % attention_heads != 0:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if lstm_layers <= 0:
            raise ValueError("lstm_layers must be positive")
        if not math.isfinite(dropout) or not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be finite and within [0, 1)")
        if future_dim < 2:
            raise ValueError("future_dim must include clear-sky and daylight features")
        if tuple(float(level) for level in quantiles) != REGISTERED_LEVELS:
            raise ValueError("RQR-TFT requires the registered seven quantiles")
        if not math.isfinite(ramp_loss_weight) or ramp_loss_weight < 0:
            raise ValueError("ramp_loss_weight must be finite and non-negative")
        if (
            not math.isfinite(ramp_weighted_quantile_multiplier)
            or ramp_weighted_quantile_multiplier <= 0
        ):
            raise ValueError("ramp_weighted_quantile_multiplier must be finite and positive")
        if not math.isfinite(ramp_decision_threshold) or not (0.0 < ramp_decision_threshold < 1.0):
            raise ValueError("ramp_decision_threshold must be finite and inside (0, 1)")
        if not math.isfinite(ramp_pos_weight) or ramp_pos_weight < 0:
            raise ValueError("ramp_pos_weight must be finite and non-negative")

        self.history_length = history_length
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.ramp_loss_weight = float(ramp_loss_weight)
        self.ramp_weighted_quantile_multiplier = float(ramp_weighted_quantile_multiplier)
        # This threshold is diagnostic metadata only. Forward and training loss
        # deliberately never read it.
        self.ramp_decision_threshold = float(ramp_decision_threshold)
        self.register_buffer("ramp_pos_weight", torch.tensor(float(ramp_pos_weight)))

        self.token_builder = ModernCovariateTokenBuilder(
            input_dim,
            future_dim,
            site_dim,
            horizon=self.horizon,
        )
        self.historical_variable_selection = TFTVariableSelectionNetwork(
            self.token_builder.token_dim,
            hidden_size,
            dropout=dropout,
        )
        self.future_variable_selection = TFTVariableSelectionNetwork(
            self.token_builder.token_dim,
            hidden_size,
            dropout=dropout,
        )
        self.historical_lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.future_gated_residual = TFTGatedResidualNetwork(
            hidden_size,
            hidden_size,
            hidden_size,
            dropout=dropout,
        )
        self.temporal_attention = InterpretableMultiHeadTemporalAttention(
            hidden_size,
            attention_heads,
            dropout=dropout,
        )
        self.decoder_gated_residual = TFTGatedResidualNetwork(
            hidden_size,
            hidden_size,
            hidden_size,
            dropout=dropout,
        )
        self.quantile_head = nn.Linear(hidden_size, len(quantiles))
        self.ramp_head = nn.Linear(hidden_size, 1)
        self.postprocess = ModernQuantilePostprocessor(
            quantiles,
            horizon=self.horizon,
            clear_sky_feature_index=0,
            daylight_feature_index=1,
        )
        nn.init.normal_(self.quantile_head.weight, std=0.02)
        nn.init.constant_(self.quantile_head.bias, -3.0)
        with torch.no_grad():
            self.quantile_head.bias[REGISTERED_LEVELS.index(0.5)] = 0.5

    def implementation_metadata(self) -> dict[str, object]:
        return {
            "actual_backbone": self.spec.actual_backbone,
            "official_code_used": self.spec.official_code_used,
            "implementation_kind": self.spec.implementation_kind,
            "shared_encoder": "tft_style_variable_selection_lstm_temporal_attention",
            "ramp_loss_weight": self.ramp_loss_weight,
            "ramp_weighted_quantile_multiplier": self.ramp_weighted_quantile_multiplier,
            "ramp_decision_threshold": self.ramp_decision_threshold,
            "ramp_decision_threshold_semantics": "diagnostic_only",
            "ramp_pos_weight": float(self.ramp_pos_weight.item()),
        }

    def forward(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if history_values.ndim != 3 or history_values.shape[1] != self.history_length:
            raise ValueError(f"history_values must have history_length={self.history_length}")
        shared_tokens = self.token_builder(
            history_values,
            history_mask,
            future_known,
            site_metadata,
        )
        historical_tokens = shared_tokens[:, : self.history_length]
        future_tokens = shared_tokens[:, self.history_length :]
        selected_history, _ = self.historical_variable_selection(historical_tokens)
        selected_future, _ = self.future_variable_selection(future_tokens)
        encoded_history, (hidden_state, _) = self.historical_lstm(selected_history)
        history_context = hidden_state[-1, :, None, :]
        future_context = self.future_gated_residual(selected_future + history_context)
        memory = torch.cat((encoded_history, future_context), dim=1)
        attended_future, _ = self.temporal_attention(future_context, memory)
        shared_horizon = self.decoder_gated_residual(
            attended_future,
            residual=future_context,
        )
        raw_quantiles = self.quantile_head(shared_horizon)
        ramp_logits = self.ramp_head(shared_horizon).squeeze(-1)
        return {
            "quantiles": self.postprocess(raw_quantiles, future_known=future_known),
            "ramp_logits": ramp_logits,
            "ramp_probs": ramp_logits.sigmoid(),
            "ramp_pos_weight": self.ramp_pos_weight,
        }


def _git_output(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "--no-replace-objects", "-C", str(root), *args],
            check=False,
            text=True,
            capture_output=True,
            shell=False,
        )
    except OSError as exc:
        raise ValueError("Git is required to verify the audited OpenLTM source") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"cannot inspect the OpenLTM source root {root}: {detail}")
    return completed.stdout.strip()


def _git_archive(root: Path, commit: str) -> bytes:
    try:
        completed = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "-C",
                str(root),
                "archive",
                "--format=tar",
                commit,
            ],
            check=False,
            capture_output=True,
            shell=False,
        )
    except OSError as exc:
        raise ValueError("Git is required to snapshot the audited OpenLTM source") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        raise ValueError(f"cannot archive audited OpenLTM commit {commit}: {detail}")
    return completed.stdout


def _canonical_github_remote(url: str) -> str:
    match = _GITHUB_REMOTE.fullmatch(url.strip().rstrip("/"))
    if match is None:
        raise ValueError(f"unsupported GitHub remote URL: {url!r}")
    repository_path = match.group("path").removesuffix(".git")
    if not repository_path or "/" not in repository_path:
        raise ValueError(f"invalid GitHub repository URL: {url!r}")
    return f"https://github.com/{repository_path}.git"


def _validated_openltm_source_identity(
    openltm_root: Path,
    source_identity: Mapping[str, object] | None,
) -> dict[str, object]:
    root = Path(openltm_root)
    if not root.is_dir():
        raise ValueError(f"OpenLTM source root is not a directory: {root}")
    required_file = root / _OPENLTM_TIMER_XL_PATH
    if not required_file.is_file():
        raise ValueError(f"missing required OpenLTM file: {required_file}")
    if source_identity is None:
        raise ValueError("Timer-XL requires an audited source_identity mapping")

    name = source_identity.get("name")
    if name != _OPENLTM_NAME:
        raise ValueError(f"source_identity name must be {_OPENLTM_NAME!r}")
    repository_url = source_identity.get("repository_url")
    if repository_url != _OPENLTM_REPOSITORY_URL:
        raise ValueError(
            "source_identity repository_url must identify the canonical OpenLTM repository URL "
            f"{_OPENLTM_REPOSITORY_URL!r}"
        )
    clean_tree = source_identity.get("clean_tree")
    if clean_tree is not True:
        raise ValueError("source_identity clean_tree must be true")
    commit = source_identity.get("commit")
    if not isinstance(commit, str) or _FULL_GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError("source_identity commit must be a full 40-character lowercase Git SHA-1")
    license_sha256 = source_identity.get("license_sha256")
    if not isinstance(license_sha256, str) or _FULL_SHA256.fullmatch(license_sha256) is None:
        raise ValueError("source_identity license_sha256 must be a full lowercase SHA-256")
    required_file_sha256 = source_identity.get("required_file_sha256")
    if (
        not isinstance(required_file_sha256, str)
        or _FULL_SHA256.fullmatch(required_file_sha256) is None
    ):
        raise ValueError("source_identity required_file_sha256 must be a full lowercase SHA-256")
    implementation_kind = source_identity.get("implementation_kind")
    if implementation_kind != _OPENLTM_IMPLEMENTATION_KIND:
        raise ValueError(
            f"source_identity implementation_kind must be {_OPENLTM_IMPLEMENTATION_KIND!r}"
        )
    required_path = source_identity.get("required_path")
    if required_path is not None and required_path != _OPENLTM_TIMER_XL_PATH.as_posix():
        raise ValueError(
            "source_identity required_path must be "
            f"{_OPENLTM_TIMER_XL_PATH.as_posix()!r} when present"
        )

    repository_root = Path(_git_output(root, "rev-parse", "--show-toplevel")).resolve()
    if repository_root != root.resolve():
        raise ValueError(f"OpenLTM source root is not the Git repository root: {root}")
    actual_remote = _canonical_github_remote(_git_output(root, "remote", "get-url", "origin"))
    if actual_remote != _OPENLTM_REPOSITORY_URL:
        raise ValueError(
            "source root is not the canonical OpenLTM repository: "
            f"{actual_remote!r} != {_OPENLTM_REPOSITORY_URL!r}"
        )
    if _git_output(root, "status", "--porcelain"):
        raise ValueError(f"OpenLTM source has a dirty working tree: {root}")
    current_commit = _git_output(root, "rev-parse", "HEAD")
    if current_commit != commit:
        raise ValueError(
            f"audited OpenLTM commit does not match the source root: {commit} != {current_commit}"
        )
    return {
        "name": name,
        "repository_url": repository_url,
        "clean_tree": clean_tree,
        "commit": commit,
        "license_sha256": license_sha256,
        "required_file_sha256": required_file_sha256,
        "implementation_kind": implementation_kind,
    }


def _safe_extract_git_archive(archive_bytes: bytes, destination: Path) -> None:
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
            for member in archive.getmembers():
                relative = PurePosixPath(member.name)
                if (
                    not member.name
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or relative == PurePosixPath(".")
                ):
                    raise ValueError(f"unsafe Git archive member path: {member.name!r}")
                if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                    raise ValueError(f"unsafe Git archive member type: {member.name!r}")
                target = destination.joinpath(*relative.parts).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError(f"unsafe Git archive member escape: {member.name!r}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise ValueError(f"duplicate Git archive member: {member.name!r}")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read Git archive member: {member.name!r}")
                with target.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
    except (tarfile.TarError, OSError) as exc:
        raise ValueError("cannot safely extract the audited OpenLTM Git archive") from exc


def _openltm_license_file(root: Path) -> Path:
    for name in _OPENLTM_LICENSE_NAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise ValueError(f"archived OpenLTM source has no recognized license file: {root}")


@contextmanager
def _openltm_commit_snapshot(
    openltm_root: Path,
    audited_identity: Mapping[str, object],
) -> Iterator[Path]:
    commit = str(audited_identity["commit"])
    archive_bytes = _git_archive(Path(openltm_root), commit)
    with TemporaryDirectory(prefix="racepv-openltm-") as temporary_directory:
        snapshot_root = Path(temporary_directory)
        _safe_extract_git_archive(archive_bytes, snapshot_root)
        required_file = snapshot_root / _OPENLTM_TIMER_XL_PATH
        if not required_file.is_file():
            raise ValueError(f"archived OpenLTM source is missing {required_file}")
        required_file_sha256 = hashlib.sha256(required_file.read_bytes()).hexdigest()
        if required_file_sha256 != audited_identity["required_file_sha256"]:
            raise ValueError(
                "archived OpenLTM required-file SHA-256 does not match the audited identity"
            )
        license_file = _openltm_license_file(snapshot_root)
        license_sha256 = hashlib.sha256(license_file.read_bytes()).hexdigest()
        if license_sha256 != audited_identity["license_sha256"]:
            raise ValueError("archived OpenLTM license SHA-256 does not match the audited identity")
        yield snapshot_root


def _openltm_top_level_import_names(root: Path) -> set[str]:
    names = {path.stem for path in root.glob("*.py")}
    names.update(path.name for path in root.iterdir() if path.is_dir() and any(path.rglob("*.py")))
    return names


def load_openltm_timer_xl_class(openltm_root: Path) -> type[nn.Module]:
    """Load official ``Model`` from an immutable snapshot without alias leakage."""

    root = Path(openltm_root).resolve()
    required_file = root / _OPENLTM_TIMER_XL_PATH
    if not required_file.is_file():
        raise ValueError(f"missing required OpenLTM file: {required_file}")
    digest = hashlib.sha256(required_file.read_bytes()).hexdigest()
    module_name = f"_racepv_openltm_timer_xl_{digest[:16]}"
    top_level_names = _openltm_top_level_import_names(root)

    def is_scoped_name(name: str) -> bool:
        return name == module_name or any(
            name == top_level or name.startswith(f"{top_level}.") for top_level in top_level_names
        )

    _imp.acquire_lock()
    try:
        with _OPENLTM_IMPORT_LOCK:
            path_snapshot = list(sys.path)
            module_snapshot = {
                name: module for name, module in sys.modules.items() if is_scoped_name(name)
            }
            for name in tuple(sys.modules):
                if is_scoped_name(name):
                    del sys.modules[name]

            model_class: object | None = None
            try:
                sys.path.insert(0, str(root))
                spec = importlib.util.spec_from_file_location(module_name, required_file)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot create an import specification for {required_file}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                model_class = getattr(module, "Model", None)
            finally:
                sys.path[:] = path_snapshot
                for name in tuple(sys.modules):
                    if is_scoped_name(name):
                        del sys.modules[name]
                sys.modules.update(module_snapshot)
    finally:
        _imp.release_lock()

    if not isinstance(model_class, type) or not issubclass(model_class, nn.Module):
        raise ImportError(
            f"official OpenLTM file does not define a torch.nn.Module Model: {required_file}"
        )
    return model_class


def build_timer_xl_config(
    *,
    input_dim: int,
    patch_len: int,
    d_model: int,
    n_heads: int,
    e_layers: int,
    dropout: float,
) -> SimpleNamespace:
    """Map the registered protocol values onto the verified OpenLTM API."""

    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if patch_len <= 0:
        raise ValueError("patch_len must be positive")
    if d_model <= 0:
        raise ValueError("d_model must be positive")
    if n_heads <= 0 or d_model % n_heads:
        raise ValueError("n_heads must be positive and divide d_model")
    if e_layers <= 0:
        raise ValueError("e_layers must be positive")
    if not math.isfinite(dropout) or not (0.0 <= dropout < 1.0):
        raise ValueError("dropout must be finite and within [0, 1)")
    return SimpleNamespace(
        input_token_len=int(patch_len),
        output_token_len=int(patch_len),
        d_model=int(d_model),
        n_heads=int(n_heads),
        e_layers=int(e_layers),
        d_ff=4 * int(d_model),
        dropout=float(dropout),
        activation="gelu",
        covariate=True,
        flash_attention=False,
        output_attention=False,
        use_norm=True,
    )


class TimerXLQuantileAdapter(nn.Module):
    """Quantile adapter around an identity-audited official OpenLTM Timer-XL."""

    spec = ModernBaselineSpec(
        name="timer_xl_qr",
        implementation_kind="official_backbone_protocol_adaptation",
        actual_backbone="openltm_timer_xl",
        source_name="OpenLTM",
        official_code_used=True,
    )

    def __init__(
        self,
        openltm_root: Path,
        input_dim: int,
        future_dim: int,
        site_dim: int,
        quantiles: Sequence[float],
        *,
        patch_len: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        dropout: float,
        source_identity: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        if future_dim < 2:
            raise ValueError("future_dim must include clear-sky and daylight features")
        if tuple(float(level) for level in quantiles) != REGISTERED_LEVELS:
            raise ValueError("Timer-XL-QR requires the registered seven quantiles")
        audited_identity = _validated_openltm_source_identity(openltm_root, source_identity)
        self.patch_len = int(patch_len)
        self.horizon = 24
        self.token_builder = ModernCovariateTokenBuilder(
            input_dim,
            future_dim,
            site_dim,
            horizon=self.horizon,
        )
        config = build_timer_xl_config(
            input_dim=self.token_builder.token_dim,
            patch_len=self.patch_len,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            dropout=dropout,
        )
        with _openltm_commit_snapshot(openltm_root, audited_identity) as snapshot_root:
            model_class = load_openltm_timer_xl_class(snapshot_root)
            self.backbone = model_class(config)
        self.backbone_projection = nn.Linear(self.token_builder.token_dim, d_model)
        self.decoder = FutureQuantileDecoder(
            d_model,
            future_dim,
            site_dim,
            self.horizon,
            quantiles,
        )
        self._implementation_metadata = {
            "actual_backbone": "openltm_timer_xl",
            "official_code_used": True,
            "pretrained_checkpoint_used": False,
            "adaptation": "task_specific_from_scratch",
            "implementation_kind": "official_backbone_protocol_adaptation",
            "commit": audited_identity["commit"],
            "required_file_sha256": audited_identity["required_file_sha256"],
            "backbone_class": type(self.backbone).__qualname__,
            "backbone_module": type(self.backbone).__module__,
        }

    def implementation_metadata(self) -> dict[str, object]:
        return dict(self._implementation_metadata)

    def forward(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor | None = None,
    ) -> dict[str, Tensor]:
        shared_tokens = self.token_builder(
            history_values,
            history_mask,
            future_known,
            site_metadata,
        )
        padding = (-shared_tokens.shape[1]) % self.patch_len
        padded_tokens = F.pad(shared_tokens, (0, 0, 0, padding))
        batch_size = padded_tokens.shape[0]
        x_mark = padded_tokens.new_zeros(batch_size, padded_tokens.shape[1], 4)
        y_mark = padded_tokens.new_zeros(batch_size, self.horizon, 4)
        backbone_output = self.backbone(padded_tokens, x_mark, y_mark)

        if isinstance(backbone_output, tuple):
            raise TypeError(
                "Timer-XL backbone must return a Tensor [batch, time, channels], not tuple"
            )
        if not isinstance(backbone_output, Tensor):
            raise TypeError(
                "Timer-XL backbone must return a Tensor [batch, time, channels], "
                f"got {type(backbone_output).__name__}"
            )
        if backbone_output.ndim != 3:
            raise ValueError("Timer-XL backbone output must have shape [batch, time, channels]")
        if backbone_output.shape[0] != batch_size:
            raise ValueError(
                "Timer-XL backbone batch dimension does not match its input: "
                f"{backbone_output.shape[0]} != {batch_size}"
            )
        if backbone_output.shape[1] < self.horizon:
            raise ValueError(
                "Timer-XL backbone time dimension must be at least 24, "
                f"got {backbone_output.shape[1]}"
            )
        if backbone_output.shape[2] != self.token_builder.token_dim:
            raise ValueError(
                "Timer-XL backbone channel dimension must match the shared token sequence: "
                f"{backbone_output.shape[2]} != {self.token_builder.token_dim}"
            )
        official_tail = backbone_output[:, -self.horizon :, :]
        context = self.backbone_projection(official_tail)
        quantiles = self.decoder(context, future_known, site_metadata)
        return {"quantiles": quantiles}


class ModernBaselineLoss(nn.Module):
    """Comparable daylight qCRPS, plus the optional RQR training terms."""

    selection_metric = "daylight_qcrps"

    def __init__(
        self,
        levels: Sequence[float] = REGISTERED_LEVELS,
        *,
        ramp_pos_weight: float = 1.0,
        ramp_weighted_quantile_multiplier: float = 6.0,
        ramp_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.levels = tuple(float(level) for level in levels)
        if ramp_pos_weight < 0:
            raise ValueError("ramp_pos_weight cannot be negative")
        if ramp_weighted_quantile_multiplier <= 0:
            raise ValueError("ramp_weighted_quantile_multiplier must be positive")
        if ramp_loss_weight < 0:
            raise ValueError("ramp_loss_weight cannot be negative")
        self.register_buffer("ramp_pos_weight", torch.tensor(float(ramp_pos_weight)))
        self.ramp_weighted_quantile_multiplier = float(ramp_weighted_quantile_multiplier)
        self.ramp_loss_weight = float(ramp_loss_weight)

    def forward(
        self,
        outputs: Mapping[str, Tensor],
        targets: Tensor,
        state_labels: Tensor,
        *,
        daylight_mask: Tensor | None = None,
    ) -> Mapping[str, Tensor]:
        quantiles = outputs["quantiles"]
        if state_labels.shape != targets.shape:
            raise ValueError("state_labels must match targets")
        if daylight_mask is not None and daylight_mask.shape != targets.shape:
            raise ValueError("daylight_mask must match targets")
        mask = torch.isfinite(targets)
        if daylight_mask is not None:
            mask &= daylight_mask.bool()
        daylight_qcrps = trapezoidal_qcrps(
            quantiles,
            targets,
            self.levels,
            mask=mask,
        )
        result = {
            "loss": daylight_qcrps,
            "daylight_qcrps": daylight_qcrps,
        }
        if "ramp_logits" not in outputs:
            return result

        ramp_logits = outputs["ramp_logits"]
        if ramp_logits.shape != targets.shape:
            raise ValueError("ramp_logits must match targets")
        ramp = torch.isin(state_labels, state_labels.new_tensor((1, 2)))
        labelled = mask & torch.isin(
            state_labels,
            state_labels.new_tensor((0, 1, 2)),
        )
        if self.ramp_weighted_quantile_multiplier == 1.0:
            weighted_forecast = daylight_qcrps
        else:
            daylight_ramp = mask & ramp
            cell_weights = torch.where(
                daylight_ramp,
                targets.new_full((), self.ramp_weighted_quantile_multiplier),
                targets.new_ones(()),
            )
            per_cell_qcrps = trapezoidal_qcrps(
                quantiles,
                targets,
                self.levels,
                mask=mask,
                reduction="none",
            )
            weighted_forecast = (per_cell_qcrps * cell_weights).sum() / mask.sum().clamp_min(1)

        if bool(labelled.any()):
            ramp_target = ramp.to(ramp_logits.dtype)
            ramp_bce = F.binary_cross_entropy_with_logits(
                ramp_logits[labelled],
                ramp_target[labelled],
                pos_weight=self.ramp_pos_weight.to(
                    dtype=ramp_logits.dtype,
                    device=ramp_logits.device,
                ),
            )
        else:
            ramp_bce = ramp_logits[labelled].sum()
        result["forecast_loss"] = weighted_forecast
        result["ramp_bce"] = ramp_bce
        result["loss"] = weighted_forecast + self.ramp_loss_weight * ramp_bce
        return result


_MODERN_BASELINE_FACTORIES: dict[str, type[nn.Module]] = {
    "timemixerpp_qr": TimeMixerPPQuantile,
    "timer_xl_qr": TimerXLQuantileAdapter,
    "rqr_tft": RQRTFTQuantile,
}


def build_modern_baseline(
    name: str,
    *,
    input_dim: int,
    future_dim: int,
    site_dim: int,
    quantiles: Sequence[float],
    hyperparameters: Mapping[str, object],
    openltm_root: Path | None = None,
    source_identity: Mapping[str, object] | None = None,
) -> tuple[nn.Module, ModernBaselineSpec]:
    """Build an audited modern baseline, failing closed when unavailable."""

    if name not in MODERN_BASELINES:
        raise KeyError(f"unknown modern baseline: {name}")
    if name == "timer_xl_qr" and openltm_root is None:
        raise ValueError("Timer-XL requires an audited OpenLTM source root")
    if name != "timer_xl_qr" and source_identity is not None:
        raise ValueError("source_identity is only valid for the Timer-XL baseline")
    factory = _MODERN_BASELINE_FACTORIES.get(name)
    if factory is None:
        raise NotImplementedError(f"registered modern baseline {name} is not yet implemented")
    kwargs = dict(hyperparameters)
    if name == "timemixerpp_qr":
        implementation_kind = kwargs.pop(
            "implementation_kind",
            "paper_derived_protocol_reproduction",
        )
        if implementation_kind != "paper_derived_protocol_reproduction":
            raise ValueError("timemixerpp_qr implementation_kind does not match its audited spec")
        kwargs.setdefault("history_length", 192)
        kwargs.setdefault("horizon", 24)
    elif name == "timer_xl_qr":
        implementation_kind = kwargs.pop(
            "implementation_kind",
            "official_backbone_protocol_adaptation",
        )
        if implementation_kind != "official_backbone_protocol_adaptation":
            raise ValueError("timer_xl_qr implementation_kind does not match its audited spec")
        initialization = kwargs.pop("initialization", "task_specific_from_scratch")
        if initialization != "task_specific_from_scratch":
            raise ValueError("timer_xl_qr initialization must be task_specific_from_scratch")
        kwargs["openltm_root"] = openltm_root
        kwargs["source_identity"] = source_identity
    elif name == "rqr_tft":
        implementation_kind = kwargs.pop(
            "implementation_kind",
            "paper_derived_protocol_reproduction",
        )
        if implementation_kind != "paper_derived_protocol_reproduction":
            raise ValueError("rqr_tft implementation_kind does not match its audited spec")
        kwargs.setdefault("history_length", 192)
        kwargs.setdefault("horizon", 24)
    model = factory(
        input_dim=input_dim,
        future_dim=future_dim,
        site_dim=site_dim,
        quantiles=quantiles,
        **kwargs,
    )
    spec = getattr(model, "spec", None)
    if not isinstance(spec, ModernBaselineSpec):
        raise TypeError(f"{name} implementation must expose a ModernBaselineSpec as .spec")
    return model, spec


__all__ = [
    "MODERN_BASELINES",
    "REGISTERED_LEVELS",
    "ModernBaselineLoss",
    "ModernBaselineSpec",
    "ModernCovariateTokenBuilder",
    "ModernQuantilePostprocessor",
    "InterpretableMultiHeadTemporalAttention",
    "RQRTFTQuantile",
    "RampClassWeight",
    "TFTGatedResidualNetwork",
    "TFTVariableSelectionNetwork",
    "TimerXLQuantileAdapter",
    "TimeMixerPPQuantile",
    "build_timer_xl_config",
    "build_modern_baseline",
    "load_openltm_timer_xl_class",
    "ramp_class_weight_from_training_labels",
]
