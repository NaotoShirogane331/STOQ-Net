"""FACT official-backbone adaptation with the common seven-quantile head."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import Tensor, nn

from racepv.baselines.extension_registry import SupervisedQRAdapterBase


class FACTQuantileAdapter(SupervisedQRAdapterBase):
    """Replace FACT's point projection while preserving its fixed backbone."""

    def __init__(
        self,
        resolved_config: dict[str, object],
        upstream_model_class: type[nn.Module],
    ) -> None:
        super().__init__(resolved_config)
        defaults = resolved_config["backbone_defaults"]
        config = SimpleNamespace(
            task_name="long_term_forecast",
            pred_len=self.horizon,
            seq_len=self.history_length + self.horizon,
            enc_in=self.token_builder.token_dim,
            use_norm=int(defaults["use_norm"]),
            freq=defaults["freq"],
            d_model=defaults["d_model"],
            dilation=list(defaults["dilation"]),
            num_kernels=defaults["num_kernels"],
            core=defaults["core"],
            d_ff=defaults["d_ff"],
            dropout=defaults["dropout"],
        )
        self.backbone = upstream_model_class(config)
        self.backbone.projection = nn.Identity()
        self.quantile_head = nn.Conv1d(
            int(defaults["d_model"]),
            self.horizon * len(self.quantiles),
            kernel_size=1,
        )
        nn.init.normal_(self.quantile_head.weight, std=0.02)
        nn.init.constant_(self.quantile_head.bias, -3.0)
        median = self.quantiles.index(0.5)
        with torch.no_grad():
            for step in range(self.horizon):
                self.quantile_head.bias[step * len(self.quantiles) + median] = 0.5

    def backbone_features(self, tokens: Tensor) -> Tensor:
        encoded = self.normalize_tokens(tokens)
        encoded = self.backbone.emb(encoded)

        frequency = torch.fft.rfft(encoded, dim=1)
        amplitude = torch.abs(frequency)
        phase = torch.angle(frequency)
        for amplitude_block, phase_block in zip(
            self.backbone.conv_amp,
            self.backbone.conv_phase,
            strict=True,
        ):
            amplitude = amplitude_block(amplitude)
            phase = phase_block(phase)
        phase = phase - 2 * torch.pi * torch.round(phase / (2 * torch.pi))
        frequency = torch.complex(amplitude * torch.cos(phase), amplitude * torch.sin(phase))
        frequency_features = torch.fft.irfft(frequency, dim=1).to(torch.float32)

        time_features = encoded
        for block in self.backbone.conv_time:
            time_features = block(time_features)
        return self.backbone.alpha * time_features + (1 - self.backbone.alpha) * frequency_features

    def raw_quantiles(self, features: Tensor) -> Tensor:
        projected = self.quantile_head(features)
        target = projected[:, :, 0]
        return target.reshape(target.shape[0], self.horizon, len(self.quantiles))
