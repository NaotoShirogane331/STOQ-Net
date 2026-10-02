"""Version-gated RaCE-PV checkpoint loading without legacy weight migration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from racepv.models import RaCEPV


def _candidate_checkpoint_metadata(payload, expected):
    """Require every supplied identity copy to agree before touching weights."""
    containers = []
    for key in ("metadata", "model_metadata"):
        if key in payload:
            if not isinstance(payload[key], Mapping):
                raise TypeError(f"checkpoint {key} must be a mapping")
            containers.append(payload[key])
    nested = payload.get("metadata", {}).get("model_metadata")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise TypeError("checkpoint nested model_metadata must be a mapping")
        containers.append(nested)
    fields = ("architecture_version", "model_revision", "objective_version")
    if not containers:
        raise RuntimeError("candidate checkpoint is missing model/objective identity")
    for metadata in containers:
        for key in fields:
            if metadata.get(key) != expected[key]:
                raise RuntimeError(f"checkpoint {key} does not match candidate identity")
    for key in fields:
        if key in payload and payload[key] != expected[key]:
            raise RuntimeError(f"checkpoint top-level {key} conflicts with candidate identity")
    return containers[0]


def load_racepv_checkpoint(
    model: RaCEPV,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a matching legacy architecture or exact candidate/objective identity."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("RaCE-PV checkpoint must contain a dictionary payload")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("RaCE-PV checkpoint metadata must be a mapping")
    expected_metadata = model.model_metadata
    if isinstance(expected_metadata["architecture_version"], str):
        _candidate_checkpoint_metadata(payload, expected_metadata)
    else:
        raw_version = metadata.get("architecture_version", payload.get("architecture_version", 1))
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
