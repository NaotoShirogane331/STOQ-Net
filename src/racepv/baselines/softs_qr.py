"""SOFTS official-backbone adaptation with seven direct quantiles."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import Tensor, nn

from racepv.baselines.extension_registry import SupervisedQRAdapterBase


class SOFTSQuantileAdapter(SupervisedQRAdapterBase):
    """Preserve the official STAR encoder and replace its point head."""

    def __init__(
        self,
        resolved_config: dict[str, object],
        upstream_model_class: type[nn.Module],
    ) -> None:
        super().__init__(resolved_config)
        defaults = resolved_config["backbone_defaults"]
        config = SimpleNamespace(
            seq_len=self.history_length + self.horizon,
            pred_len=self.horizon,
            d_model=defaults["d_model"],
            d_core=defaults["d_core"],
            e_layers=defaults["e_layers"],
            d_ff=defaults["d_ff"],
            dropout=defaults["dropout"],
            activation=defaults["activation"],
            use_norm=defaults["use_norm"],
        )
        self.backbone = upstream_model_class(config)
        self.backbone.projection = nn.Identity()
        self.quantile_head = nn.Linear(
            int(defaults["d_model"]),
            self.horizon * len(self.quantiles),
        )
        nn.init.normal_(self.quantile_head.weight, std=0.02)
        nn.init.constant_(self.quantile_head.bias, -3.0)
        median = self.quantiles.index(0.5)
        with torch.no_grad():
            for step in range(self.horizon):
                self.quantile_head.bias[step * len(self.quantiles) + median] = 0.5

    def backbone_features(self, tokens: Tensor) -> Tensor:
        encoded = self.normalize_tokens(tokens)
        encoded = self.backbone.enc_embedding(encoded, None)
        encoded, _ = self.backbone.encoder(encoded, attn_mask=None)
        return encoded

    def raw_quantiles(self, features: Tensor) -> Tensor:
        projected = self.quantile_head(features)
        target = projected[:, 0, :]
        return target.reshape(target.shape[0], self.horizon, len(self.quantiles))
