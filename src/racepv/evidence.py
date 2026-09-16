"""Fail-closed admission guard for artifacts entering registered evidence flows."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class RegisteredEvidenceGuardError(ValueError):
    """Raised when an artifact is valid only outside registered evidence flows."""


def _is_true(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.casefold() == "true")


def _is_false(value: Any) -> bool:
    return value is False or (isinstance(value, str) and value.casefold() == "false")


def _unsafe_marker(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "evidence_status" and item == "hypothetical_oracle_conditioned":
                return "evidence_status=hypothetical_oracle_conditioned"
            if key == "registered_evidence_eligible" and _is_false(item):
                return "registered_evidence_eligible=false"
            if key == "oracle_conditioned" and _is_true(item):
                return "oracle_conditioned=true"
            nested = _unsafe_marker(item)
            if nested is not None:
                return nested
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            nested = _unsafe_marker(item)
            if nested is not None:
                return nested
    return None


def _has_scenarios_component(path: Path) -> bool:
    return any(part.casefold() == "scenarios" for part in path.parts)


def _reject(path: Path, resolved: Path, marker: str) -> None:
    resolution = f" (resolved to {resolved})" if resolved != path else ""
    raise RegisteredEvidenceGuardError(
        f"registered evidence input rejected: {path}{resolution}; unsafe marker {marker}. "
        "This artifact remains valid for scenario analysis."
    )


def require_registered_input(
    path: str | Path, *, metadata: Mapping[str, Any] | None = None
) -> None:
    """Reject scenario paths or metadata that cannot enter registered evidence."""

    supplied = Path(path).absolute()
    resolved = supplied.resolve(strict=False)
    if _has_scenarios_component(supplied) or _has_scenarios_component(resolved):
        _reject(supplied, resolved, "path_component=scenarios")
    if metadata is not None:
        marker = _unsafe_marker(metadata)
        if marker is not None:
            _reject(supplied, resolved, marker)


def require_registered_json_input(path: str | Path) -> dict[str, Any]:
    """Load a JSON object only after its path and recursive markers are admitted."""

    candidate = Path(path)
    require_registered_input(candidate)
    with candidate.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{candidate} does not contain a JSON object")
    require_registered_input(candidate, metadata=payload)
    return payload


def require_registered_prediction_input(path: str | Path) -> None:
    """Check a prediction path and its JSON sidecar before array loading."""

    candidate = Path(path)
    require_registered_input(candidate)
    sidecar = candidate.with_suffix(".json")
    if sidecar.is_file():
        require_registered_input(sidecar)
        with sidecar.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise TypeError(f"{sidecar} does not contain a JSON object")
        require_registered_input(candidate, metadata=metadata)


__all__ = [
    "RegisteredEvidenceGuardError",
    "require_registered_input",
    "require_registered_json_input",
    "require_registered_prediction_input",
]
