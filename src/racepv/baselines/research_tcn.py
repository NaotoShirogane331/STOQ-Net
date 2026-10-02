"""Research-only TCN matching the registered B3 experiment protocol."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .neural import _DirectQuantileDecoder

_DILATIONS = (1, 2, 4, 8)
_KERNEL_SIZE = 3
_CONVOLUTIONS_PER_BLOCK = 2


class _ResearchCausalBlock(nn.Module):
    """Two causal convolutions with per-timestep channel normalization."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = (_KERNEL_SIZE - 1) * dilation
        self.convolutions = nn.ModuleList(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=_KERNEL_SIZE,
                dilation=dilation,
            )
            for _ in range(_CONVOLUTIONS_PER_BLOCK)
        )
        self.normalizations = nn.ModuleList(
            nn.LayerNorm(channels) for _ in range(_CONVOLUTIONS_PER_BLOCK)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        encoded = inputs
        for convolution, normalization in zip(
            self.convolutions, self.normalizations, strict=True
        ):
            encoded = nn.functional.pad(encoded, (self.left_padding, 0))
            encoded = convolution(encoded)
            encoded = normalization(encoded.transpose(1, 2)).transpose(1, 2)
            encoded = nn.functional.gelu(encoded)
        return inputs + encoded


class ResearchTCNQuantile(nn.Module):
    """Protocol B3 TCN with causal encoding and full-history mean pooling."""

    def __init__(
        self,
        input_dim: int,
        future_dim: int,
        site_dim: int = 0,
        hidden_dim: int = 64,
        quantiles: Sequence[float] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.future_dim = int(future_dim)
        self.site_dim = int(site_dim)
        self.hidden_dim = int(hidden_dim)
        self.quantiles = tuple(float(quantile) for quantile in quantiles)
        self.input_projection = nn.Conv1d(2 * self.input_dim, self.hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            _ResearchCausalBlock(self.hidden_dim, dilation) for dilation in _DILATIONS
        )
        self.decoder = _DirectQuantileDecoder(
            self.hidden_dim,
            self.future_dim,
            self.site_dim,
            self.hidden_dim,
            self.quantiles,
        )

    def encode_history(self, history_values: Tensor, history_mask: Tensor) -> Tensor:
        if history_values.ndim != 3 or history_values.shape[-1] != self.input_dim:
            raise ValueError("history_values must be BxLxinput_dim")
        if history_mask.shape != history_values.shape or history_mask.dtype != torch.bool:
            raise ValueError("history_mask must be boolean and match history_values")
        observed = torch.where(history_mask, history_values, torch.zeros_like(history_values))
        encoder_inputs = torch.cat(
            (observed, history_mask.to(history_values.dtype)), dim=-1
        ).transpose(1, 2)
        encoded = self.input_projection(encoder_inputs)
        for block in self.blocks:
            encoded = block(encoded)
        return encoded.transpose(1, 2)

    def resolved_identity(self) -> dict[str, object]:
        return {
            "schema_version": "racepv.research-tcn.v1",
            "input_dim": self.input_dim,
            "future_dim": self.future_dim,
            "site_dim": self.site_dim,
            "hidden_dim": self.hidden_dim,
            "quantiles": list(self.quantiles),
            "history_missingness": "zero_maskfalse_values",
            "architecture": {
                "input_encoding": "masked_values_concatenated_with_boolean_mask",
                "input_projection": "conv1d_kernel1",
                "residual_blocks": len(self.blocks),
                "convolutions_per_block": _CONVOLUTIONS_PER_BLOCK,
                "kernel_size": _KERNEL_SIZE,
                "dilations": list(_DILATIONS),
                "padding": "left_causal",
                "normalization": {
                    "kind": "layer_norm",
                    "normalized_axis": "per_timestep_channel",
                    "cross_time": False,
                },
                "activation": "gelu",
                "residual_position": "after_two_convolutions",
                "encoder_dropout": 0.0,
                "history_pooling": "full_history_mean",
                "decoder": "racepv.baselines.neural._DirectQuantileDecoder",
            },
        }

    def forward(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor | None = None,
    ) -> dict[str, Tensor]:
        encoded = self.encode_history(history_values, history_mask)
        context = encoded.mean(dim=1)
        return {"quantiles": self.decoder(context, future_known, site_metadata)}
