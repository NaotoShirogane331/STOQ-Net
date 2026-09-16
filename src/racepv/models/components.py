"""Reusable neural components used by RaCE-PV."""

from __future__ import annotations

import math
import warnings

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:  # The official package is an optional CUDA-oriented dependency.
    from mamba_ssm import Mamba as _OfficialMamba
except Exception as exc:  # pragma: no cover - depends on the host runtime
    _OfficialMamba = None
    _MAMBA_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - exercised only with mamba_ssm installed
    _MAMBA_IMPORT_ERROR = None


class CausalConv1d(nn.Module):
    """One-dimensional convolution with left-only padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            groups=groups,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(F.pad(inputs, (self.left_padding, 0)))


class CausalConvEncoder(nn.Module):
    """Small causal encoder followed by a validity-weighted history pool."""

    def __init__(self, input_size: int, hidden_size: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(input_size, hidden_size, kernel_size)
        self.conv2 = CausalConv1d(hidden_size, hidden_size, kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        # Convolution modules use channel-first layout; all padding is causal.
        hidden = inputs.transpose(1, 2)
        hidden = self.dropout(F.gelu(self.conv1(hidden)))
        hidden = self.dropout(F.gelu(self.conv2(hidden))).transpose(1, 2)
        hidden = self.norm(hidden)

        weights = valid_mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return hidden, pooled


class MaskedFutureWeatherEncoder(nn.Module):
    """Encode per-horizon issued weather while preserving explicit missingness."""

    def __init__(self, feature_count: int, hidden_size: int) -> None:
        super().__init__()
        if feature_count <= 0 or hidden_size <= 0:
            raise ValueError("weather feature_count and hidden_size must be positive")
        self.feature_count = int(feature_count)
        self.hidden_size = int(hidden_size)
        self.network = nn.Sequential(
            nn.Linear(2 * self.feature_count, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        if values.ndim != 3 or values.shape[-1] != self.feature_count:
            raise ValueError(
                "future_weather must have shape [B,H,"
                f"{self.feature_count}]"
            )
        if mask.shape != values.shape:
            raise ValueError("future_weather_mask must match future_weather")
        if not values.is_floating_point() or not mask.is_floating_point():
            raise TypeError("future weather values and mask must be floating point")
        if not torch.isfinite(mask).all() or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("future_weather_mask must be finite and binary")
        available = mask.bool()
        clean_values = torch.where(
            available & torch.isfinite(values),
            values,
            torch.zeros_like(values),
        )
        encoded = self.network(torch.cat((clean_values, mask), dim=-1))
        any_available = available.any(dim=-1, keepdim=True).to(encoded.dtype)
        return encoded * any_available


class OrderedQuantileHead(nn.Module):
    """Generate non-crossing quantiles around a supplied physical location.

    Positive adjacent gaps are constructed with ``softplus``.  The location is
    allowed a learned correction and represents the median; lower and upper
    levels are then accumulated outwards.  Consequently monotonicity holds for
    every possible parameter value.
    """

    def __init__(
        self,
        hidden_size: int,
        quantiles: tuple[float, ...],
        gap_scale: float,
    ) -> None:
        super().__init__()
        self.quantiles = quantiles
        self.median_index = quantiles.index(0.5)
        self.gap_scale = gap_scale
        self.projection = nn.Linear(hidden_size, len(quantiles))
        with torch.no_grad():
            # Initial softplus gaps are modest relative to normalized PV power.
            self.projection.bias[1:].fill_(-2.0)

    def forward(self, hidden: Tensor, location: Tensor) -> Tensor:
        raw = self.projection(hidden)
        median = location + raw[..., :1].squeeze(-1)
        gaps = F.softplus(raw[..., 1:]) * self.gap_scale

        left_count = self.median_index
        left_near_to_far = gaps[..., :left_count]
        right_near_to_far = gaps[..., left_count:]

        if left_count:
            left_distance = torch.cumsum(left_near_to_far, dim=-1).flip(-1)
            lower = median.unsqueeze(-1) - left_distance
        else:  # pragma: no cover - the registered levels include lower tails
            lower = median.new_empty((*median.shape, 0))
        if right_near_to_far.shape[-1]:
            upper = median.unsqueeze(-1) + torch.cumsum(right_near_to_far, dim=-1)
        else:  # pragma: no cover - the registered levels include upper tails
            upper = median.new_empty((*median.shape, 0))
        return torch.cat((lower, median.unsqueeze(-1), upper), dim=-1)


class CausalTCNBlock(nn.Module):
    """Residual depthwise causal TCN block with Mamba-matched width."""

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dropout: float,
        expand: int,
    ) -> None:
        super().__init__()
        hidden_size = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.depthwise = CausalConv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            groups=d_model,
        )
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(d_model, hidden_size, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, d_model, kernel_size=1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.norm(inputs).transpose(1, 2)
        hidden = F.gelu(self.depthwise(hidden))
        hidden = self.channel_mixer(hidden).transpose(1, 2)
        return inputs + self.dropout(hidden)


class CausalSequenceBackbone(nn.Module):
    """Auditable Mamba, causal TCN, or causal Conv-GRU sequence backbone.

    ``backend_name`` is ``"mamba_ssm"`` only when official ``mamba_ssm.Mamba``
    blocks were actually constructed.  The fallback is deliberately reported
    as ``"causal_conv_gru"`` and is never described as Mamba.
    """

    def __init__(
        self,
        *,
        d_model: int,
        d_state: int,
        d_conv: int,
        num_layers: int,
        dropout: float,
        backend: str,
        expand: int,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers: nn.ModuleList | None = None
        self.layer_norms: nn.ModuleList | None = None
        self.local_conv: CausalConv1d | None = None
        self.gru: nn.GRU | None = None
        self.output_norm: nn.LayerNorm | None = None
        self.tcn_layers: nn.ModuleList | None = None

        if backend == "tcn":
            self.tcn_layers = nn.ModuleList(
                [
                    CausalTCNBlock(
                        d_model=d_model,
                        kernel_size=d_conv,
                        dropout=dropout,
                        expand=expand,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.output_norm = nn.LayerNorm(d_model)
            self.backend_name = "causal_tcn"
            return

        if backend in {"auto", "mamba"} and _OfficialMamba is not None:
            try:
                self.layers = nn.ModuleList(
                    [
                        _OfficialMamba(
                            d_model=d_model,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand,
                        )
                        for _ in range(num_layers)
                    ]
                )
                self.layer_norms = nn.ModuleList(
                    [nn.LayerNorm(d_model) for _ in range(num_layers)]
                )
                self.backend_name = "mamba_ssm"
                return
            except Exception as exc:  # pragma: no cover - runtime-specific package failure
                if backend == "mamba":
                    raise RuntimeError("official mamba_ssm blocks could not be constructed") from exc
                warnings.warn(
                    "mamba_ssm was importable but unusable; selecting the explicit "
                    "causal Conv-GRU fallback (this is not Mamba).",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if backend == "mamba":
            raise ImportError(
                "backbone='mamba' requires the official mamba_ssm package"
            ) from _MAMBA_IMPORT_ERROR
        if backend == "auto" and _OfficialMamba is None:
            warnings.warn(
                "mamba_ssm is unavailable; selecting the explicit causal Conv-GRU "
                "fallback (this is not Mamba).",
                RuntimeWarning,
                stacklevel=2,
            )

        self.backend_name = "causal_conv_gru"
        self.local_conv = CausalConv1d(
            d_model,
            d_model,
            kernel_size=d_conv,
            groups=d_model,
        )
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.backend_name == "mamba_ssm":  # pragma: no cover - optional CUDA dependency
            if self.layers is None or self.layer_norms is None:
                raise RuntimeError("mamba backbone was not initialized correctly")
            hidden = inputs
            for norm, layer in zip(self.layer_norms, self.layers, strict=True):
                hidden = hidden + self.dropout(layer(norm(hidden)))
            return hidden

        if self.backend_name == "causal_tcn":
            if self.tcn_layers is None or self.output_norm is None:
                raise RuntimeError("TCN backbone was not initialized correctly")
            hidden = inputs
            for layer in self.tcn_layers:
                hidden = layer(hidden)
            return self.output_norm(hidden)

        if self.local_conv is None or self.gru is None or self.output_norm is None:
            raise RuntimeError("GRU fallback was not initialized correctly")
        convolved = self.local_conv(inputs.transpose(1, 2)).transpose(1, 2)
        hidden = inputs + self.dropout(F.gelu(convolved))
        hidden, _ = self.gru(hidden)
        return self.output_norm(hidden)


def make_harmonic_basis(
    physical_time_hours: Tensor,
    lead_time_hours: Tensor,
    periods_hours: tuple[float, ...],
) -> Tensor:
    """Create a length-independent continuous-time harmonic design matrix."""

    horizon_scale = lead_time_hours.amax(dim=1, keepdim=True).clamp_min(1e-6)
    trend = lead_time_hours / horizon_scale
    columns = [torch.ones_like(physical_time_hours), trend]
    for period in periods_hours:
        phase = (2.0 * math.pi / period) * physical_time_hours
        columns.extend((torch.sin(phase), torch.cos(phase)))
    return torch.stack(columns, dim=-1)


def cyclic_hour_to_hours(hour_sin: Tensor, hour_cos: Tensor) -> Tensor:
    """Recover local clock hour in ``[0, 24)`` from audited cyclic features."""

    if hour_sin.shape != hour_cos.shape:
        raise ValueError("hour_sin and hour_cos must have identical shapes")
    phase = torch.atan2(hour_sin, hour_cos)
    return torch.remainder(phase, 2.0 * math.pi) * (24.0 / (2.0 * math.pi))


def causal_rolling_std(values: Tensor, window: int) -> Tensor:
    """Differentiable trailing standard deviation for ``[B, L, 1]`` values."""

    window = max(1, min(window, values.shape[1]))
    channel_first = values.transpose(1, 2)
    padded = F.pad(channel_first, (window - 1, 0), mode="replicate")
    mean = F.avg_pool1d(padded, kernel_size=window, stride=1)
    mean_sq = F.avg_pool1d(padded.square(), kernel_size=window, stride=1)
    variance = (mean_sq - mean.square()).clamp_min(0.0)
    return variance.sqrt().transpose(1, 2)


def normalized_spectral_entropy(values: Tensor, window: int, eps: float = 1e-8) -> Tensor:
    """Normalized entropy of the non-DC periodogram over a trailing window."""

    window = max(2, min(window, values.shape[1]))
    segment = values[:, -window:].float()
    segment = segment - segment.mean(dim=1, keepdim=True)
    spectrum = torch.fft.rfft(segment, dim=1).abs().square()
    if spectrum.shape[1] > 1:
        spectrum = spectrum[:, 1:]
    bins = spectrum.shape[1]
    if bins <= 1:
        return values.new_zeros((values.shape[0], 1))
    probabilities = spectrum / spectrum.sum(dim=1, keepdim=True).clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=1)
    entropy = entropy / math.log(bins)
    return entropy.to(values.dtype)
