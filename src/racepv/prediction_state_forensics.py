"""Paired validation diagnostics for prediction/state coupling experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_ALIGNED_FIELDS = (
    "targets",
    "state_labels",
    "daylight_mask",
    "timestamps",
    "site_ids",
)


def _metric_summary(
    reference_error: np.ndarray,
    candidate_error: np.ndarray,
) -> dict[str, float | int | None]:
    cells = int(reference_error.size)
    if cells == 0:
        return {
            "cells": 0,
            "reference_rmse": None,
            "candidate_rmse": None,
            "reference_mae": None,
            "candidate_mae": None,
            "mean_squared_error_gain": None,
            "sum_squared_error_gain": None,
            "mean_absolute_error_gain": None,
            "candidate_better_share": None,
        }
    reference_squared = np.square(reference_error)
    candidate_squared = np.square(candidate_error)
    squared_gain = reference_squared - candidate_squared
    return {
        "cells": cells,
        "reference_rmse": float(np.sqrt(np.mean(reference_squared))),
        "candidate_rmse": float(np.sqrt(np.mean(candidate_squared))),
        "reference_mae": float(np.mean(np.abs(reference_error))),
        "candidate_mae": float(np.mean(np.abs(candidate_error))),
        "mean_squared_error_gain": float(np.mean(squared_gain)),
        "sum_squared_error_gain": float(np.sum(squared_gain)),
        "mean_absolute_error_gain": float(
            np.mean(np.abs(reference_error) - np.abs(candidate_error))
        ),
        "candidate_better_share": float(np.mean(candidate_squared < reference_squared)),
    }


def _group_summaries(
    values: np.ndarray,
    reference_error: np.ndarray,
    candidate_error: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    return {
        str(value): _metric_summary(
            reference_error[values == value],
            candidate_error[values == value],
        )
        for value in np.unique(values)
    }


def _ranked_severity_summaries(
    reference_error: np.ndarray,
    candidate_error: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    order = np.argsort(np.square(reference_error))[::-1]
    size = order.size
    top_1 = max(1, int(np.ceil(0.01 * size)))
    top_5 = max(top_1, int(np.ceil(0.05 * size)))
    top_10 = max(top_5, int(np.ceil(0.10 * size)))
    slices = {
        "top_1pct": order[:top_1],
        "p95_to_p99": order[top_1:top_5],
        "p90_to_p95": order[top_5:top_10],
        "bottom_90pct": order[top_10:],
    }
    return {
        name: _metric_summary(reference_error[index], candidate_error[index])
        for name, index in slices.items()
    }


def _quartile_summaries(
    values: np.ndarray,
    reference_error: np.ndarray,
    candidate_error: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    boundaries = np.quantile(values, (0.25, 0.5, 0.75))
    bins = np.digitize(values, boundaries, right=True)
    return {
        f"q{index + 1}": {
            "value_min": float(np.min(values[bins == index]))
            if bool(np.any(bins == index))
            else None,
            "value_max": float(np.max(values[bins == index]))
            if bool(np.any(bins == index))
            else None,
            **_metric_summary(
                reference_error[bins == index],
                candidate_error[bins == index],
            ),
        }
        for index in range(4)
    }


def paired_ramp_forensics(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    quantile_levels: Sequence[float],
) -> dict[str, Any]:
    """Compare paired ramp-cell point errors without changing either archive."""

    for name in _ALIGNED_FIELDS:
        if name not in reference or name not in candidate:
            raise ValueError(f"missing aligned field {name}")
        if not np.array_equal(reference[name], candidate[name]):
            raise ValueError(f"unaligned {name}")

    targets = np.asarray(reference["targets"], dtype=float)
    state_labels = np.asarray(reference["state_labels"])
    daylight = np.asarray(reference["daylight_mask"]) > 0.5
    if targets.ndim != 2:
        raise ValueError("targets must have shape [origins, horizon]")
    ramp_mask = daylight & np.isin(state_labels, (1, 2))
    if not bool(np.any(ramp_mask)):
        raise ValueError("no daylight ramp cells")

    levels = np.asarray(quantile_levels, dtype=float)
    median_matches = np.flatnonzero(np.isclose(levels, 0.5, atol=1e-12, rtol=0.0))
    if median_matches.size != 1:
        raise ValueError("quantile levels must contain one median")
    median_index = int(median_matches[0])
    reference_quantiles = np.asarray(reference["quantiles"], dtype=float)
    candidate_quantiles = np.asarray(candidate["quantiles"], dtype=float)
    expected_quantile_shape = (*targets.shape, levels.size)
    if reference_quantiles.shape != expected_quantile_shape:
        raise ValueError("reference quantiles have an unexpected shape")
    if candidate_quantiles.shape != expected_quantile_shape:
        raise ValueError("candidate quantiles have an unexpected shape")

    reference_error_grid = reference_quantiles[..., median_index] - targets
    candidate_error_grid = candidate_quantiles[..., median_index] - targets
    if not bool(
        np.all(np.isfinite(reference_error_grid[ramp_mask]))
        and np.all(np.isfinite(candidate_error_grid[ramp_mask]))
    ):
        raise ValueError("paired ramp errors must be finite")
    reference_error = reference_error_grid[ramp_mask]
    candidate_error = candidate_error_grid[ramp_mask]

    horizon_grid = np.broadcast_to(np.arange(targets.shape[1]), targets.shape)
    sites = np.asarray(reference["site_ids"]).reshape(-1)
    if sites.size != targets.shape[0]:
        raise ValueError("site_ids must have one value per forecast origin")
    site_grid = np.broadcast_to(sites[:, None], targets.shape)

    state_names = {1: "ramp_up", 2: "ramp_down"}
    by_state = {
        state_names.get(int(value), f"state_{int(value)}"): _metric_summary(
            reference_error_grid[ramp_mask & (state_labels == value)],
            candidate_error_grid[ramp_mask & (state_labels == value)],
        )
        for value in np.unique(state_labels[ramp_mask])
    }

    expert_spread: np.ndarray | None = None
    if "expert_quantiles" in reference:
        experts = np.asarray(reference["expert_quantiles"], dtype=float)
        if experts.shape[:2] != targets.shape or experts.shape[-1] != levels.size:
            raise ValueError("reference expert quantiles have an unexpected shape")
        expert_spread = np.ptp(experts[..., median_index], axis=-1)[ramp_mask]

    gate_shift: np.ndarray | None = None
    if "gate_weights" in reference and "gate_weights" in candidate:
        reference_gate = np.asarray(reference["gate_weights"], dtype=float)
        candidate_gate = np.asarray(candidate["gate_weights"], dtype=float)
        if reference_gate.shape != candidate_gate.shape or reference_gate.shape[:2] != targets.shape:
            raise ValueError("paired gate weights have an unexpected shape")
        gate_shift = np.sum(np.abs(candidate_gate - reference_gate), axis=-1)[ramp_mask]

    return {
        "overall": _metric_summary(reference_error, candidate_error),
        "by_state": by_state,
        "by_horizon": _group_summaries(
            horizon_grid[ramp_mask], reference_error, candidate_error
        ),
        "by_site": _group_summaries(site_grid[ramp_mask], reference_error, candidate_error),
        "reference_error_severity": _ranked_severity_summaries(
            reference_error, candidate_error
        ),
        "expert_spread_quartiles": _quartile_summaries(
            expert_spread, reference_error, candidate_error
        )
        if expert_spread is not None
        else {},
        "gate_shift_quartiles": _quartile_summaries(
            gate_shift, reference_error, candidate_error
        )
        if gate_shift is not None
        else {},
        "diagnostics": {
            "paired_alignment_pass": True,
            "expert_spread_available": expert_spread is not None,
            "gate_shift_available": gate_shift is not None,
        },
    }
