"""Random-initialization synthetic STOQ-Net interface demonstration.

This example uses the existing M1 factory; it reads no data or weights and
does not train, score, calibrate, or reproduce manuscript performance.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from racepv.models.config import RaCEPVConfig
from racepv.models.research_variants import build_research_candidate

MODEL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "stoq-paper-model.json"
TUPLE_FIELDS = (
    "quantiles",
    "harmonic_periods_hours",
    "active_experts",
    "clear_sky_history_feature_indices",
    "clear_sky_future_feature_indices",
)


def load_model_config(path: Path = MODEL_CONFIG) -> RaCEPVConfig:
    """Restore the original tuple schema without changing model fields."""
    values = json.loads(path.read_text(encoding="utf-8"))
    for name in TUPLE_FIELDS:
        values[name] = tuple(values[name])
    return RaCEPVConfig(**values)


def build_demo_model(config: RaCEPVConfig, backbone: str):
    """Use an explicit backend; Mamba errors propagate without fallback."""
    if backbone not in {"gru", "mamba"}:
        raise ValueError("backbone must be explicitly 'gru' or 'mamba'")
    model, _objective = build_research_candidate("M1", replace(config, backbone=backbone))
    return model


def make_synthetic_inputs(config: RaCEPVConfig, device: str = "cpu"):
    """Make a small artificial batch, including masked power and one night lead."""
    batch = 2
    history = torch.rand(batch, config.history_length, config.history_features, device=device)
    mask = torch.ones_like(history, dtype=torch.bool)
    history[:, 32:36, config.power_index] = float("nan")
    mask[:, 32:36, config.power_index] = False
    future = torch.zeros(batch, config.horizon, config.future_features, device=device)
    future[..., config.future_clear_sky_index] = 0.6
    future[:, 0, config.future_clear_sky_index] = 0.0
    lead = torch.arange(1, config.horizon + 1, device=device) * config.sampling_minutes / 60
    clock = 2 * math.pi * (10 + lead) / 24
    future[..., config.future_hour_sin_index] = clock.sin()
    future[..., config.future_hour_cos_index] = clock.cos()
    future[..., 1] = 45.0  # synthetic solar elevation, degrees
    future[..., 3] = 1.0  # synthetic azimuth cosine
    future[..., 7] = 1.0  # synthetic day-of-year cosine
    future[..., 8] = lead
    static = torch.zeros(batch, config.site_features, device=device)
    static[..., -1] = 1.0
    return history, mask, future, static


def run_demo(backbone: str = "gru", device: str = "cpu"):
    """Run only a random synthetic forward and return outputs plus identity."""
    if backbone == "mamba" and torch.device(device).type != "cuda":
        raise ValueError("The native Mamba demo requires --device cuda and a qualified environment")
    config = load_model_config()
    torch.manual_seed(13)
    model = build_demo_model(config, backbone).to(device)
    model.eval()
    inputs = make_synthetic_inputs(config, device)
    with torch.inference_mode():
        output = model(*inputs)
    q = output["quantiles"]
    if tuple(q.shape) != (2, 24, 7):
        raise RuntimeError("unexpected quantile output shape")
    if not torch.isfinite(q).all() or not (q.diff(dim=-1) >= 0).all():
        raise RuntimeError("quantiles must be finite and ordered")
    if not ((q >= 0) & (q <= config.power_upper_bound)).all():
        raise RuntimeError("quantiles exceed the configured output bounds")
    if torch.count_nonzero(q[:, 0]) != 0:
        raise RuntimeError("synthetic night targets were not zeroed")
    metadata = model.model_metadata
    summary = {
        "display_name": "STOQ-Net",
        "internal_model_id": "M1",
        "model_class": type(model).__name__,
        "model_revision": metadata["model_revision"],
        "requested_backbone": backbone,
        "actual_backbone": metadata["actual_backbone"],
        "device": str(q.device),
        "output_shape": list(q.shape),
        "quantile_levels": list(config.quantiles),
        "mode": "random_initialization_synthetic_interface_demo",
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=("gru", "mamba"), default="gru")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    _output, summary = run_demo(args.backbone, args.device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
