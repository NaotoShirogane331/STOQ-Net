"""Sparse bounded correction for rare forecast-tail errors."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import RaCEPVConfig


class SparseTailResidualHead(nn.Module):
    """Predict a detached, bounded signed correction for each horizon cell."""

    def __init__(self, config: RaCEPVConfig) -> None:
        super().__init__()
        input_size = (
            3 * config.d_model
            + 3
            + 3
            + config.future_features
            + config.site_features
            + 3
        )
        self.network = nn.Sequential(
            nn.Linear(input_size, config.tail_residual_hidden_size),
            nn.LayerNorm(config.tail_residual_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.tail_residual_hidden_size, 2),
        )
        self.cap = float(config.tail_residual_cap)

    def forward(
        self,
        *,
        stable_hidden: Tensor,
        up_hidden: Tensor,
        down_hidden: Tensor,
        expert_quantiles: Tensor,
        gate_weights: Tensor,
        future_known: Tensor,
        site_metadata: Tensor,
        normalized_lead: Tensor,
        clock_sin: Tensor,
        clock_cos: Tensor,
        median_index: int,
    ) -> tuple[Tensor, Tensor]:
        horizon = future_known.shape[1]
        site = site_metadata.unsqueeze(1).expand(-1, horizon, -1)
        pieces = (
            stable_hidden.detach(),
            up_hidden.detach(),
            down_hidden.detach(),
            expert_quantiles[..., median_index].detach(),
            gate_weights.detach(),
            future_known.detach(),
            site.detach(),
            normalized_lead.detach().unsqueeze(-1),
            clock_sin.detach().unsqueeze(-1),
            clock_cos.detach().unsqueeze(-1),
        )
        raw = self.network(torch.cat(pieces, dim=-1))
        trigger = torch.sigmoid(raw[..., 0])
        correction = self.cap * trigger * torch.tanh(raw[..., 1])
        return trigger, correction
