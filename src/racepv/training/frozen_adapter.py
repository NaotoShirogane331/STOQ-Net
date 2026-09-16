"""Fail-closed initialization for the frozen-core issued-weather adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from racepv.audit import sha256_file
from racepv.models import RaCEPV

from .checkpoint import load_racepv_checkpoint

FROZEN_WEATHER_ADAPTER_SCHEMA = "frozen_weather_adapter_v1"
FROZEN_WEATHER_ADAPTER_PREFIXES = (
    "weather_encoder.",
    "transient_expert.weather_projection.",
)


def _require_equal(
    metadata: Mapping[str, object],
    key: str,
    expected: object,
    label: str,
) -> None:
    if metadata.get(key) != expected:
        raise ValueError(f"source checkpoint {label} mismatch")


def initialize_frozen_weather_adapter(
    model: RaCEPV,
    checkpoint_path: str | Path,
    *,
    expected_seed: int,
    expected_train_content_sha256: str,
    expected_validation_content_sha256: str,
    expected_weather_store_sha256: str,
    require_official_mamba: bool = True,
) -> dict[str, object]:
    """Load an accepted W00 checkpoint and expose only its weather adapter."""

    if model.config.forecast_weather_mode != "transient_isolated":
        raise ValueError("frozen weather adapter requires transient_isolated mode")
    source_path = Path(checkpoint_path)
    payload = load_racepv_checkpoint(model, source_path)
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):  # guarded by the shared loader
        raise TypeError("source checkpoint metadata must be a mapping")
    weather = metadata.get("weather_protocol")
    if not isinstance(weather, Mapping):
        raise ValueError("source checkpoint lacks weather protocol provenance")
    source_config = metadata.get("model_config")
    if not isinstance(source_config, Mapping):
        raise ValueError("source checkpoint lacks model configuration provenance")
    candidate_config = asdict(model.config)
    mismatched_config = sorted(
        key
        for key, candidate_value in candidate_config.items()
        if key != "forecast_weather_mode" and source_config.get(key) != candidate_value
    )
    if mismatched_config:
        raise ValueError(
            "source checkpoint model configuration mismatch: "
            f"{mismatched_config}"
        )

    if metadata.get("forecast_weather_mode") != "off":
        raise ValueError("source checkpoint is not W00/off")
    _require_equal(metadata, "seed", int(expected_seed), "seed")
    source_uses_official_mamba = metadata.get("uses_official_mamba") is True
    if require_official_mamba and not source_uses_official_mamba:
        raise ValueError("source checkpoint is not official Mamba")
    _require_equal(
        metadata,
        "train_window_content_sha256",
        expected_train_content_sha256,
        "train content hash",
    )
    _require_equal(
        metadata,
        "validation_window_content_sha256",
        expected_validation_content_sha256,
        "validation content hash",
    )
    if weather.get("forecast_weather_mode") != "off":
        raise ValueError("source checkpoint weather protocol is not W00/off")
    if weather.get("weather_store_sha256") != expected_weather_store_sha256:
        raise ValueError("source checkpoint weather store hash mismatch")

    weather_encoder = model.weather_encoder
    weather_projection = model.transient_expert.weather_projection
    if not isinstance(weather_encoder, nn.Module) or not isinstance(
        weather_projection, nn.Module
    ):
        raise ValueError("frozen weather adapter modules are missing")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (weather_encoder, weather_projection):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    with torch.no_grad():
        for parameter in weather_projection.parameters():
            parameter.zero_()

    trainable = sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if not trainable:
        raise RuntimeError("frozen weather adapter has no trainable parameters")
    if any(
        not name.startswith(FROZEN_WEATHER_ADAPTER_PREFIXES) for name in trainable
    ):
        raise RuntimeError("adapter trainable-parameter whitelist violated")

    return {
        "training_protocol": FROZEN_WEATHER_ADAPTER_SCHEMA,
        "optimizer_schema": FROZEN_WEATHER_ADAPTER_SCHEMA,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256_file(source_path),
        "source_optimizer_schema": metadata.get("optimizer_schema"),
        "source_forecast_weather_mode": "off",
        "source_uses_official_mamba": source_uses_official_mamba,
        "development_backbone_smoke": not bool(require_official_mamba),
        "trainable_parameter_names": trainable,
        "state_optimizer_enabled": False,
        "state_loss_gradient_target": "none_frozen",
        "frozen_core": True,
        "zero_initialized_weather_projection": True,
    }


__all__ = [
    "FROZEN_WEATHER_ADAPTER_PREFIXES",
    "FROZEN_WEATHER_ADAPTER_SCHEMA",
    "initialize_frozen_weather_adapter",
]
