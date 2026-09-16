"""Compact quantile neural baselines with a common tensor interface."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


class OrderedQuantileProjection(nn.Module):
    """Turn unconstrained outputs into ordered quantiles around the median."""

    def __init__(self, quantiles: Sequence[float]) -> None:
        super().__init__()
        self.quantiles = tuple(float(q) for q in quantiles)
        if 0.5 not in self.quantiles:
            raise ValueError("quantiles must include 0.5")
        self.median_index = self.quantiles.index(0.5)

    def forward(self, raw: Tensor) -> Tensor:
        if raw.shape[-1] != len(self.quantiles):
            raise ValueError("last dimension must equal number of quantiles")
        median = raw[..., self.median_index : self.median_index + 1]
        lower_gaps = torch.nn.functional.softplus(raw[..., : self.median_index])
        upper_gaps = torch.nn.functional.softplus(raw[..., self.median_index + 1 :])

        lower = []
        cursor = median
        for index in range(self.median_index - 1, -1, -1):
            cursor = cursor - lower_gaps[..., index : index + 1]
            lower.append(cursor)
        lower.reverse()

        upper = []
        cursor = median
        for index in range(upper_gaps.shape[-1]):
            cursor = cursor + upper_gaps[..., index : index + 1]
            upper.append(cursor)
        return torch.cat([*lower, median, *upper], dim=-1)


class _DirectQuantileDecoder(nn.Module):
    def __init__(
        self,
        context_dim: int,
        future_dim: int,
        site_dim: int,
        hidden_dim: int,
        quantiles: Sequence[float],
    ) -> None:
        super().__init__()
        self.site_dim = int(site_dim)
        self.net = nn.Sequential(
            nn.Linear(context_dim + future_dim + self.site_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, len(quantiles)),
        )
        self.ordered = OrderedQuantileProjection(quantiles)

    def forward(
        self,
        context: Tensor,
        future_known: Tensor,
        site_metadata: Tensor | None = None,
    ) -> Tensor:
        repeated = context[:, None, :].expand(-1, future_known.shape[1], -1)
        pieces = [repeated, future_known]
        if self.site_dim:
            if site_metadata is None or site_metadata.shape != (context.shape[0], self.site_dim):
                raise ValueError("site_metadata does not match the configured site dimension")
            pieces.append(site_metadata[:, None, :].expand(-1, future_known.shape[1], -1))
        return self.ordered(self.net(torch.cat(pieces, dim=-1)))


class LSTMQuantile(nn.Module):
    def __init__(
        self,
        input_dim: int,
        future_dim: int,
        site_dim: int = 0,
        hidden_dim: int = 64,
        num_layers: int = 2,
        quantiles: Sequence[float] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            2 * input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=0.1 if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder = _DirectQuantileDecoder(
            hidden_dim, future_dim, site_dim, hidden_dim, quantiles
        )

    def forward(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor | None = None,
    ) -> dict[str, Tensor]:
        encoded_inputs = torch.cat(
            (history_values * history_mask, history_mask.to(history_values.dtype)), dim=-1
        )
        encoded, _ = self.encoder(encoded_inputs)
        quantiles = self.decoder(encoded[:, -1], future_known, site_metadata)
        return {"quantiles": quantiles}


class _CausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = 2 * dilation
        self.padding = padding
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=padding)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, inputs: Tensor) -> Tensor:
        convolved = self.conv(inputs)
        if self.padding:
            convolved = convolved[..., : -self.padding]
        return torch.nn.functional.gelu(self.norm(convolved)) + inputs


class TCNQuantile(nn.Module):
    def __init__(
        self,
        input_dim: int,
        future_dim: int,
        site_dim: int = 0,
        hidden_dim: int = 64,
        quantiles: Sequence[float] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(2 * input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(*[_CausalBlock(hidden_dim, dilation) for dilation in (1, 2, 4, 8)])
        self.decoder = _DirectQuantileDecoder(
            hidden_dim, future_dim, site_dim, hidden_dim, quantiles
        )

    def forward(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor | None = None,
    ) -> dict[str, Tensor]:
        values = torch.cat(
            (history_values * history_mask, history_mask.to(history_values.dtype)), dim=-1
        ).transpose(1, 2)
        encoded = self.blocks(self.input_projection(values)).transpose(1, 2)
        return {"quantiles": self.decoder(encoded[:, -1], future_known, site_metadata)}


class PatchTSTQuantile(nn.Module):
    """Parameter-controlled PatchTST-style encoder for a fair QR baseline."""

    def __init__(
        self,
        input_dim: int,
        future_dim: int,
        site_dim: int = 0,
        history_length: int = 192,
        patch_length: int = 16,
        patch_stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        quantiles: Sequence[float] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
    ) -> None:
        super().__init__()
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        n_patches = 1 + (history_length - patch_length) // patch_stride
        self.patch_projection = nn.Linear(2 * input_dim * patch_length, d_model)
        self.position = nn.Parameter(torch.zeros(1, n_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.decoder = _DirectQuantileDecoder(
            d_model, future_dim, site_dim, d_model, quantiles
        )
        nn.init.trunc_normal_(self.position, std=1.0 / math.sqrt(d_model))

    def forward(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor | None = None,
    ) -> dict[str, Tensor]:
        values = torch.cat(
            (history_values * history_mask, history_mask.to(history_values.dtype)), dim=-1
        )
        patches = values.unfold(1, self.patch_length, self.patch_stride)
        patches = patches.permute(0, 1, 3, 2).reshape(values.shape[0], patches.shape[1], -1)
        tokens = self.patch_projection(patches)
        tokens = tokens + self.position[:, : tokens.shape[1]]
        context = self.encoder(tokens).mean(dim=1)
        return {"quantiles": self.decoder(context, future_known, site_metadata)}
