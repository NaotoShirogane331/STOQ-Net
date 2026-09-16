"""Version-gated RaCE-PV checkpoint loading without legacy weight migration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from racepv.models import RaCEPV


def load_racepv_checkpoint(
    model: RaCEPV,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load only a checkpoint whose architecture exactly matches the v2 model."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("RaCE-PV checkpoint must contain a dictionary payload")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("RaCE-PV checkpoint metadata must be a mapping")
    raw_version = metadata.get(
        "architecture_version", payload.get("architecture_version", 1)
    )
    try:
        checkpoint_version = int(raw_version)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("checkpoint architecture_version is invalid") from exc
    expected_version = model.config.architecture_version
    if checkpoint_version != expected_version:
        raise RuntimeError(
            f"checkpoint architecture_version={checkpoint_version} cannot load into "
            f"RaCE-PV architecture_version={expected_version}; legacy weights are not migrated"
        )
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise TypeError("RaCE-PV checkpoint is missing model_state_dict")
    model.load_state_dict(state, strict=True)
    return payload


__all__ = ["load_racepv_checkpoint"]
