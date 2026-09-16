"""Quantile adaptation of the published Simple-Mamba baseline.

This module intentionally requires the official CUDA Mamba package. A recurrent
fallback would no longer be S-Mamba and is therefore not silently substituted in
the reported baseline.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .neural import OrderedQuantileProjection


def _mamba_layer(d_model: int, d_state: int, d_conv: int):
    try:
        from mamba_ssm import Mamba
    except ImportError as exc:  # pragma: no cover - CUDA-only optional dependency
        raise ImportError(
            "S-Mamba-QR requires the official mamba-ssm package on Linux/CUDA; "
            "install race-pv[mamba] after installing a matching CUDA PyTorch build"
        ) from exc
    return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)


class SMambaQuantile(nn.Module):
    """Variate tokens, bidirectional Mamba mixing, and direct quantile decoding."""

    def __init__(
        self,
        input_dim: int,
        future_dim: int,
        site_dim: int = 0,
        history_length: int = 192,
        horizon: int = 24,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        quantiles: Sequence[float] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.quantiles = tuple(quantiles)
        self.variate_embedding = nn.Linear(history_length, d_model)
        self.forward_mamba = _mamba_layer(d_model, d_state, d_conv)
        self.reverse_mamba = _mamba_layer(d_model, d_state, d_conv)
        self.norm = nn.LayerNorm(d_model)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * d_model, d_model),
        )
        self.future_projection = nn.Sequential(nn.Linear(future_dim, d_model), nn.GELU())
        self.site_dim = int(site_dim)
        self.site_projection = (
            nn.Sequential(nn.Linear(self.site_dim, d_model), nn.GELU())
            if self.site_dim
            else None
        )
        decoder_width = 2 * d_model + (d_model if self.site_dim else 0)
        self.decoder = nn.Linear(decoder_width, len(self.quantiles))
        self.ordered = OrderedQuantileProjection(self.quantiles)
        self.input_dim = input_dim

    def forward(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if history_values.shape[-1] != self.input_dim:
            raise ValueError("history input dimension does not match configured input_dim")
        if future_known.shape[1] != self.horizon:
            raise ValueError("future horizon does not match configured horizon")
        values_and_mask = torch.cat(
            (history_values * history_mask, history_mask.to(history_values.dtype)), dim=-1
        )
        tokens = self.variate_embedding(values_and_mask.transpose(1, 2))
        forward = self.forward_mamba(tokens)
        reverse = torch.flip(self.reverse_mamba(torch.flip(tokens, dims=(1,))), dims=(1,))
        mixed = self.norm(tokens + 0.5 * (forward + reverse))
        context = (mixed + self.temporal_ffn(mixed)).mean(dim=1)
        future = self.future_projection(future_known)
        pieces = [context[:, None, :].expand(-1, self.horizon, -1), future]
        if self.site_projection is not None:
            if site_metadata is None or site_metadata.shape != (
                history_values.shape[0],
                self.site_dim,
            ):
                raise ValueError("site_metadata does not match the configured site dimension")
            pieces.append(
                self.site_projection(site_metadata)[:, None, :].expand(-1, self.horizon, -1)
            )
        raw = self.decoder(torch.cat(pieces, dim=-1))
        return {"quantiles": self.ordered(raw)}
