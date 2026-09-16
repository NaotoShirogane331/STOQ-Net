"""Run a small forward/backward contract check without downloading data."""

from __future__ import annotations

import argparse
import json

import torch

from racepv.environment import environment_report
from racepv.losses import RaCEPVLoss
from racepv.models import RaCEPV, RaCEPVConfig
from racepv.utils import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=("auto", "mamba", "gru", "tcn"), default="gru")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    device = torch.device(select_device(args.device))
    config = RaCEPVConfig(
        history_features=14,
        future_features=9,
        site_features=6,
        backbone=args.backbone,
        future_hour_sin_index=4,
        future_hour_cos_index=5,
    )
    model = RaCEPV(config)
    if model.backbone_name == "mamba_ssm" and device.type != "cuda":
        raise RuntimeError(
            "official mamba_ssm smoke tests require a CUDA device in WSL/Linux"
        )
    model = model.to(device)
    batch = args.batch_size
    history = torch.rand(batch, 192, 14, device=device)
    mask = torch.ones_like(history)
    future = torch.rand(batch, 24, 9, device=device)
    future[..., 0] = torch.linspace(0.0, 1.0, 24, device=device)
    future[:, :2, 0] = 0.0
    local_hour = torch.arange(1, 25, device=device) * 0.25 + 6.0
    clock_phase = 2.0 * torch.pi * local_hour / 24.0
    future[..., 4] = torch.sin(clock_phase)
    future[..., 5] = torch.cos(clock_phase)
    future[..., 8] = torch.arange(1, 25, device=device) * 0.25
    metadata = torch.zeros(batch, 6, device=device)
    targets = torch.rand(batch, 24, device=device)
    targets[:, :2] = 0.0
    labels = torch.randint(0, 3, (batch, 24), device=device)
    labels[:, :2] = -100
    daylight = future[..., 0] >= 0.01

    outputs = model(history, mask, future, metadata)
    if not torch.all(outputs["quantiles"][..., 1:] >= outputs["quantiles"][..., :-1]):
        raise RuntimeError("quantile monotonicity contract failed")
    if not torch.allclose(outputs["quantiles"][:, :2], torch.zeros_like(outputs["quantiles"][:, :2])):
        raise RuntimeError("night-zero contract failed")
    objective = RaCEPVLoss()(outputs, targets, labels, daylight_mask=daylight)
    objective["loss"].backward()
    print(
        json.dumps(
            {
                "status": "ok",
                "loss": float(objective["loss"].detach().cpu()),
                "model_metadata": model.model_metadata,
                "environment": environment_report(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
