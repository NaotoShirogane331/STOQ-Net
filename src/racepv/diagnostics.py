"""Fail-closed identities for validation-only RaCE-PV diagnostics."""

from __future__ import annotations

SPECIALIST_DIAGNOSTIC_WEIGHTS = {
    "no_specialist_loss": 0.0,
    "specialist_weight_0_15": 0.15,
    "specialist_weight_0_45": 0.45,
}

SPECIALIST_WEIGHT_SWEEP = {
    name: weight
    for name, weight in SPECIALIST_DIAGNOSTIC_WEIGHTS.items()
    if name.startswith("specialist_weight_")
}


__all__ = ["SPECIALIST_DIAGNOSTIC_WEIGHTS", "SPECIALIST_WEIGHT_SWEEP"]
