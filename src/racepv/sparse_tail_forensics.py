"""Post-hoc diagnostics for sparse tail residual activation masks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

import numpy as np

from racepv.validation_screening import qcrps_cells


def build_sparse_tail_mask(
    arrays: Mapping[str, np.ndarray],
    *,
    max_horizon_index: int,
    min_ramp_up_probability: float,
    min_trigger: float,
    min_abs_correction: float,
    positive_only: bool,
) -> np.ndarray:
    """Build one inference-available mask without using targets or labels."""

    correction = np.asarray(arrays["tail_correction"], dtype=float)
    trigger = np.asarray(arrays["tail_trigger"], dtype=float)
    state_probs = np.asarray(arrays["state_probs"], dtype=float)
    daylight = np.asarray(arrays["daylight_mask"], dtype=bool)
    if correction.ndim != 2:
        raise ValueError("tail_correction must have shape [N,H]")
    if trigger.shape != correction.shape or daylight.shape != correction.shape:
        raise ValueError("trigger, correction, and daylight mask must align")
    if state_probs.shape != correction.shape + (3,):
        raise ValueError("state_probs must have shape [N,H,3]")
    if not 0 <= max_horizon_index < correction.shape[1]:
        raise ValueError("max_horizon_index is outside the forecast horizon")
    horizons = np.arange(correction.shape[1])[None, :]
    mask = (
        daylight
        & (horizons <= max_horizon_index)
        & (state_probs[..., 1] >= min_ramp_up_probability)
        & (trigger >= min_trigger)
        & (np.abs(correction) >= min_abs_correction)
    )
    if positive_only:
        mask &= correction > 0.0
    return mask


def apply_sparse_tail_mask(
    arrays: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    upper_bound: float = 1.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a scalar residual only where selected, with an exact zero fallback."""

    base = np.asarray(arrays["base_quantiles"], dtype=float)
    correction = np.asarray(arrays["tail_correction"], dtype=float)
    selected = np.asarray(mask, dtype=bool)
    if base.ndim != 3 or base.shape[:-1] != correction.shape:
        raise ValueError("base_quantiles must have shape [N,H,Q]")
    if selected.shape != correction.shape:
        raise ValueError("mask and tail_correction must align")
    applied = np.where(selected, correction, 0.0)
    quantiles = np.clip(base + applied[..., None], 0.0, upper_bound)
    return quantiles, applied


def _point_metrics(
    errors: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    selected = np.asarray(errors, dtype=float)[mask]
    if selected.size == 0:
        return {"count": 0, "nmae": float("nan"), "nrmse": float("nan")}
    return {
        "count": int(selected.size),
        "nmae": float(np.mean(np.abs(selected))),
        "nrmse": float(np.sqrt(np.mean(np.square(selected)))),
    }


def scan_sparse_tail_masks(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    quantiles: Sequence[float],
    max_horizon_indices: Sequence[int],
    ramp_up_probability_thresholds: Sequence[float],
    trigger_thresholds: Sequence[float],
    correction_thresholds: Sequence[float],
    positive_only_options: Sequence[bool],
) -> dict[str, Any]:
    """Rank deterministic masks as validation-only post-hoc diagnostics."""

    levels = np.asarray(tuple(quantiles), dtype=float)
    median_positions = np.flatnonzero(np.isclose(levels, 0.5, atol=1e-12))
    if median_positions.size != 1:
        raise ValueError("quantiles must contain exactly one median")
    targets = np.asarray(reference["targets"], dtype=float)
    labels = np.asarray(reference["state_labels"])
    daylight = np.asarray(reference["daylight_mask"], dtype=bool)
    valid = daylight & np.isfinite(targets) & np.isin(labels, (0, 1, 2))
    ramp = valid & np.isin(labels, (1, 2))
    ramp_up = valid & (labels == 1)
    ramp_down = valid & (labels == 2)
    base = np.asarray(candidate["base_quantiles"], dtype=float)
    correction = np.asarray(candidate["tail_correction"], dtype=float)
    corrected = np.clip(base + correction[..., None], 0.0, 1.05)
    base_losses = qcrps_cells(targets, base, levels, mask=valid)
    corrected_losses = qcrps_cells(targets, corrected, levels, mask=valid)
    median_index = int(median_positions[0])
    base_errors = base[..., median_index] - targets
    corrected_errors = corrected[..., median_index] - targets
    rows: list[dict[str, Any]] = []
    grid = product(
        max_horizon_indices,
        ramp_up_probability_thresholds,
        trigger_thresholds,
        correction_thresholds,
        positive_only_options,
    )
    for max_horizon, min_up, min_trigger, min_correction, positive_only in grid:
        mask = build_sparse_tail_mask(
            candidate,
            max_horizon_index=int(max_horizon),
            min_ramp_up_probability=float(min_up),
            min_trigger=float(min_trigger),
            min_abs_correction=float(min_correction),
            positive_only=bool(positive_only),
        )
        losses = np.where(mask, corrected_losses, base_losses)
        errors = np.where(mask, corrected_errors, base_errors)
        applied = np.where(mask, correction, 0.0)
        rows.append(
            {
                "max_horizon_index": int(max_horizon),
                "min_ramp_up_probability": float(min_up),
                "min_trigger": float(min_trigger),
                "min_abs_correction": float(min_correction),
                "positive_only": bool(positive_only),
                "daylight_qcrps": float(np.mean(losses[valid])),
                "activation_share": float(np.count_nonzero(applied[valid]) / np.sum(valid)),
                "overall": _point_metrics(errors, valid),
                "ramp": _point_metrics(errors, ramp),
                "ramp_up": _point_metrics(errors, ramp_up),
                "ramp_down": _point_metrics(errors, ramp_down),
            }
        )
    rows.sort(key=lambda row: (row["daylight_qcrps"], row["activation_share"]))
    return {
        "diagnostic_only": True,
        "selection_boundary": "validation_posthoc_diagnostic",
        "uses_targets_for_mask": False,
        "rows": rows,
    }
