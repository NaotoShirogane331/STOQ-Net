"""Paired validation analysis for the sealed sparse tail residual experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from racepv.validation_screening import qcrps_cells

REGISTERED_TAIL_SEEDS = (13, 29, 47, 71, 101)
TAIL_CELLS = ("R00", "R02")

_REQUIRED_ARRAYS = {
    "targets",
    "quantiles",
    "base_quantiles",
    "tail_trigger",
    "tail_correction",
    "state_labels",
    "daylight_mask",
    "timestamps",
    "site_ids",
    "state_probs",
    "state_logits",
    "gate_weights",
    "fusion_logits",
    "expert_quantiles",
}


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def _relative_change(candidate: float, reference: float) -> float:
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("relative change requires a finite positive reference")
    return float(candidate / reference - 1.0)


def _rmse(errors: np.ndarray) -> float:
    values = np.asarray(errors, dtype=float)
    if values.size == 0 or not bool(np.all(np.isfinite(values))):
        raise ValueError("RMSE requires a non-empty finite population")
    return float(np.sqrt(np.mean(np.square(values))))


def _mae(errors: np.ndarray) -> float:
    values = np.asarray(errors, dtype=float)
    if values.size == 0 or not bool(np.all(np.isfinite(values))):
        raise ValueError("MAE requires a non-empty finite population")
    return float(np.mean(np.abs(values)))


def _mean_finite(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values, dtype=float)[mask]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        raise ValueError("metric population is empty")
    return float(np.mean(selected))


def _validate_shapes(
    arrays: Mapping[str, np.ndarray],
    levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing = _REQUIRED_ARRAYS.difference(arrays)
    if missing:
        raise KeyError(f"sparse tail archive is missing: {sorted(missing)}")
    targets = np.asarray(arrays["targets"], dtype=float)
    if targets.ndim != 2:
        raise ValueError("targets must have shape [N,H]")
    n, horizon = targets.shape
    quantiles = np.asarray(arrays["quantiles"], dtype=float)
    base = np.asarray(arrays["base_quantiles"], dtype=float)
    expected_quantiles = (n, horizon, levels.size)
    if quantiles.shape != expected_quantiles or base.shape != expected_quantiles:
        raise ValueError("formal and base quantiles must have shape [N,H,Q]")
    for name in ("tail_trigger", "tail_correction", "state_labels", "daylight_mask"):
        if np.asarray(arrays[name]).shape != targets.shape:
            raise ValueError(f"{name} must have shape [N,H]")
    for name in ("state_probs", "state_logits", "gate_weights", "fusion_logits"):
        if np.asarray(arrays[name]).shape != (n, horizon, 3):
            raise ValueError(f"{name} must have shape [N,H,3]")
    if np.asarray(arrays["expert_quantiles"]).shape != (n, horizon, 3, levels.size):
        raise ValueError("expert_quantiles must have shape [N,H,3,Q]")
    timestamps = np.asarray(arrays["timestamps"])
    if timestamps.shape != targets.shape:
        raise ValueError("timestamps must have shape [N,H]")
    site_ids = np.asarray(arrays["site_ids"])
    if site_ids.shape not in {(n,), (n, horizon)}:
        raise ValueError("site_ids must have shape [N] or [N,H]")
    labels = np.asarray(arrays["state_labels"])
    daylight = np.asarray(arrays["daylight_mask"], dtype=bool)
    return targets, quantiles, labels, daylight


def _require_alignment(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
) -> None:
    for name in ("targets", "state_labels", "daylight_mask", "timestamps", "site_ids"):
        left = np.asarray(reference[name])
        right = np.asarray(candidate[name])
        equal = left.shape == right.shape and bool(
            np.array_equal(left, right, equal_nan=True)
        )
        if not equal:
            raise ValueError(f"paired sparse tail archives are not aligned on {name}")


def _point_report(
    reference_error: np.ndarray,
    candidate_error: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    reference = reference_error[mask]
    candidate = candidate_error[mask]
    reference_rmse = _rmse(reference)
    candidate_rmse = _rmse(candidate)
    reference_mae = _mae(reference)
    candidate_mae = _mae(candidate)
    return {
        "count": int(reference.size),
        "reference_nrmse": reference_rmse,
        "candidate_nrmse": candidate_rmse,
        "candidate_nrmse_change": _relative_change(candidate_rmse, reference_rmse),
        "reference_nmae": reference_mae,
        "candidate_nmae": candidate_mae,
        "candidate_nmae_change": _relative_change(candidate_mae, reference_mae),
    }


def _severity_report(
    reference_error: np.ndarray,
    candidate_error: np.ndarray,
    ramp_mask: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    flat_indices = np.flatnonzero(ramp_mask.reshape(-1))
    if flat_indices.size == 0:
        raise ValueError("severity analysis requires ramp cells")
    flat_reference = reference_error.reshape(-1)
    flat_candidate = candidate_error.reshape(-1)
    ordered = flat_indices[
        np.argsort(np.abs(flat_reference[flat_indices]), kind="stable")[::-1]
    ]
    top_1_end = max(1, int(np.ceil(0.01 * ordered.size)))
    top_5_end = max(top_1_end, int(np.ceil(0.05 * ordered.size)))
    top_10_end = max(top_5_end, int(np.ceil(0.10 * ordered.size)))
    buckets = {
        "top_1pct": ordered[:top_1_end],
        "pct_1_5": ordered[top_1_end:top_5_end],
        "pct_5_10": ordered[top_5_end:top_10_end],
        "bottom_90pct": ordered[top_10_end:],
    }
    report: dict[str, dict[str, float | int]] = {}
    for name, indices in buckets.items():
        if indices.size == 0:
            report[name] = {"count": 0}
            continue
        reference_rmse = _rmse(flat_reference[indices])
        candidate_rmse = _rmse(flat_candidate[indices])
        report[name] = {
            "count": int(indices.size),
            "reference_rmse": reference_rmse,
            "candidate_rmse": candidate_rmse,
            "candidate_rmse_change": _relative_change(
                candidate_rmse,
                reference_rmse,
            ),
        }
    return report


def _grouped_ramp_report(
    reference_error: np.ndarray,
    candidate_error: np.ndarray,
    ramp_mask: np.ndarray,
    labels: np.ndarray,
    site_ids: np.ndarray,
) -> dict[str, dict[str, dict[str, float | int]]]:
    sites = np.asarray(site_ids)
    if sites.ndim == 1:
        sites = np.broadcast_to(sites[:, None], ramp_mask.shape)
    horizons = np.broadcast_to(np.arange(ramp_mask.shape[1]), ramp_mask.shape)

    def collect(group_values: np.ndarray) -> dict[str, dict[str, float | int]]:
        report: dict[str, dict[str, float | int]] = {}
        for value in np.unique(group_values[ramp_mask]):
            selected = ramp_mask & (group_values == value)
            report[str(value)] = _point_report(
                reference_error,
                candidate_error,
                selected,
            )
        return report

    return {
        "by_site": collect(sites),
        "by_horizon": collect(horizons),
        "by_direction": {
            "ramp_up": _point_report(
                reference_error,
                candidate_error,
                ramp_mask & (labels == 1),
            ),
            "ramp_down": _point_report(
                reference_error,
                candidate_error,
                ramp_mask & (labels == 2),
            ),
        },
    }


def _invariants(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_quantiles: np.ndarray,
) -> dict[str, Any]:
    quantiles = np.asarray(arrays["quantiles"], dtype=float)
    base = np.asarray(arrays["base_quantiles"], dtype=float)
    trigger = np.asarray(arrays["tail_trigger"], dtype=float)
    correction = np.asarray(arrays["tail_correction"], dtype=float)
    daylight = np.asarray(arrays["daylight_mask"], dtype=bool)
    predicted_arrays = (
        quantiles,
        base,
        trigger,
        correction,
        np.asarray(arrays["state_probs"], dtype=float),
        np.asarray(arrays["state_logits"], dtype=float),
        np.asarray(arrays["gate_weights"], dtype=float),
        np.asarray(arrays["fusion_logits"], dtype=float),
        np.asarray(arrays["expert_quantiles"], dtype=float),
    )
    finite = bool(all(np.all(np.isfinite(values)) for values in predicted_arrays))
    crossings = int(np.count_nonzero(np.diff(quantiles, axis=-1) < -1e-7))
    night_nonzero = int(
        np.count_nonzero(quantiles[~daylight])
        + np.count_nonzero(trigger[~daylight])
        + np.count_nonzero(correction[~daylight])
    )
    max_abs_correction = float(np.max(np.abs(correction)))
    state_consistent = bool(
        np.allclose(
            np.asarray(arrays["state_probs"]),
            _softmax(np.asarray(arrays["state_logits"], dtype=float)),
            rtol=1e-5,
            atol=1e-6,
        )
    )
    gate_consistent = bool(
        np.allclose(
            np.asarray(arrays["gate_weights"]),
            _softmax(np.asarray(arrays["fusion_logits"], dtype=float)),
            rtol=1e-5,
            atol=1e-6,
        )
    )
    correction_consistent = bool(
        np.allclose(quantiles, expected_quantiles, rtol=1e-6, atol=1e-7)
    )
    bounded = bool(np.all((quantiles >= -1e-7) & (quantiles <= 1.05 + 1e-7)))
    passed = bool(
        finite
        and crossings == 0
        and night_nonzero == 0
        and max_abs_correction <= 0.08 + 1e-12
        and state_consistent
        and gate_consistent
        and correction_consistent
        and bounded
    )
    return {
        "outputs_finite": finite,
        "quantile_crossings": crossings,
        "night_nonzero_count": night_nonzero,
        "max_abs_correction": max_abs_correction,
        "state_probability_consistent": state_consistent,
        "gate_probability_consistent": gate_consistent,
        "correction_consistent": correction_consistent,
        "bounded_quantiles": bounded,
        "passed": passed,
    }


def paired_sparse_tail_metrics(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    quantiles: Sequence[float],
) -> dict[str, Any]:
    """Compute paired R00/R02 metrics on one exactly aligned validation archive."""

    levels = np.asarray(tuple(quantiles), dtype=float)
    if levels.ndim != 1 or levels.size < 3 or not np.all(np.diff(levels) > 0):
        raise ValueError("quantiles must be a strictly increasing one-dimensional sequence")
    reference_targets, reference_quantiles, labels, daylight = _validate_shapes(
        reference,
        levels,
    )
    candidate_targets, candidate_quantiles, _, _ = _validate_shapes(candidate, levels)
    del candidate_targets
    _require_alignment(reference, candidate)
    median_positions = np.flatnonzero(np.isclose(levels, 0.5, atol=1e-12))
    if median_positions.size != 1:
        raise ValueError("quantiles must contain exactly one median")
    median_index = int(median_positions[0])
    valid = daylight & np.isfinite(reference_targets) & np.isin(labels, (0, 1, 2))
    activation_population = daylight & np.isfinite(reference_targets)
    ramp_up = valid & (labels == 1)
    ramp_down = valid & (labels == 2)
    ramp = valid & np.isin(labels, (1, 2))
    if not bool(valid.any()) or not bool(ramp.any()):
        raise ValueError("paired analysis requires daylight and ramp cells")

    reference_loss = qcrps_cells(reference_targets, reference_quantiles, levels, mask=valid)
    candidate_loss = qcrps_cells(reference_targets, candidate_quantiles, levels, mask=valid)
    reference_error = reference_quantiles[..., median_index] - reference_targets
    candidate_error = candidate_quantiles[..., median_index] - reference_targets
    correction = np.asarray(candidate["tail_correction"], dtype=float)
    active = activation_population & (np.abs(correction) > 0.005)
    expected_quantiles = np.clip(
        np.asarray(candidate["base_quantiles"], dtype=float) + correction[..., None],
        0.0,
        1.05,
    )

    qcrps_reference = _mean_finite(reference_loss, valid)
    qcrps_candidate = _mean_finite(candidate_loss, valid)
    point = {
        "overall": _point_report(reference_error, candidate_error, valid),
        "ramp": _point_report(reference_error, candidate_error, ramp),
        "ramp_up": _point_report(reference_error, candidate_error, ramp_up),
        "ramp_down": _point_report(reference_error, candidate_error, ramp_down),
    }
    reference_correction = np.asarray(reference["tail_correction"], dtype=float)
    reference_expected = np.clip(
        np.asarray(reference["base_quantiles"], dtype=float)
        + reference_correction[..., None],
        0.0,
        1.05,
    )
    reference_invariants = _invariants(
        reference,
        expected_quantiles=reference_expected,
    )
    candidate_invariants = _invariants(candidate, expected_quantiles=expected_quantiles)
    return {
        "cells": {
            "daylight": int(np.sum(valid)),
            "ramp": int(np.sum(ramp)),
            "ramp_up": int(np.sum(ramp_up)),
            "ramp_down": int(np.sum(ramp_down)),
        },
        "qcrps": {
            "reference": qcrps_reference,
            "candidate": qcrps_candidate,
            "candidate_change": _relative_change(qcrps_candidate, qcrps_reference),
        },
        "point": point,
        "activation": {
            "share_abs_gt_0_005": float(
                np.sum(active) / np.sum(activation_population)
            ),
            "positive_share": float(
                np.sum(activation_population & (correction > 0.0))
                / np.sum(activation_population)
            ),
            "negative_share": float(
                np.sum(activation_population & (correction < 0.0))
                / np.sum(activation_population)
            ),
            "mean_abs_correction": float(
                np.mean(np.abs(correction[activation_population]))
            ),
            "max_abs_correction": float(
                np.max(np.abs(correction[activation_population]))
            ),
        },
        "reference_error_severity": _severity_report(
            reference_error,
            candidate_error,
            ramp,
        ),
        "grouped_ramp_benefit": _grouped_ramp_report(
            reference_error,
            candidate_error,
            ramp,
            labels,
            np.asarray(reference["site_ids"]),
        ),
        "reference_invariants": reference_invariants,
        "candidate_invariants": candidate_invariants,
        "invariants": candidate_invariants,
        "invariants_pass": bool(
            reference_invariants["passed"] and candidate_invariants["passed"]
        ),
    }


def paired_site_day_bootstrap(
    reference_losses: np.ndarray,
    candidate_losses: np.ndarray,
    group_ids: np.ndarray,
    *,
    repetitions: int = 500,
    seed: int = 20260828,
) -> dict[str, float | int]:
    """Bootstrap paired site-day means with shared group selections."""

    reference = np.asarray(reference_losses, dtype=float).reshape(-1)
    candidate = np.asarray(candidate_losses, dtype=float).reshape(-1)
    groups = np.asarray(group_ids).reshape(-1)
    if reference.shape != candidate.shape or reference.shape != groups.shape:
        raise ValueError("paired losses and group ids must have identical shapes")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    finite = np.isfinite(reference) & np.isfinite(candidate)
    if not bool(finite.any()):
        raise ValueError("bootstrap requires finite paired observations")
    reference = reference[finite]
    candidate = candidate[finite]
    groups = groups[finite]
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    reference_group = np.bincount(inverse, weights=reference) / counts
    candidate_group = np.bincount(inverse, weights=candidate) / counts
    improvements = reference_group - candidate_group
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, unique.size, size=(repetitions, unique.size))
    distribution = np.mean(improvements[draws], axis=1)
    return {
        "mean_improvement": float(np.mean(improvements)),
        "ci_low": float(np.quantile(distribution, 0.025)),
        "ci_high": float(np.quantile(distribution, 0.975)),
        "groups": int(unique.size),
        "repetitions": int(repetitions),
        "seed": int(seed),
    }


def paired_site_day_rmse_bootstrap(
    reference_errors: np.ndarray,
    candidate_errors: np.ndarray,
    group_ids: np.ndarray,
    *,
    repetitions: int = 500,
    seed: int = 20260828,
) -> dict[str, float | int | str]:
    """Bootstrap paired pooled nRMSE differences over shared site-day blocks."""

    reference = np.asarray(reference_errors, dtype=float).reshape(-1)
    candidate = np.asarray(candidate_errors, dtype=float).reshape(-1)
    groups = np.asarray(group_ids).reshape(-1)
    if reference.shape != candidate.shape or reference.shape != groups.shape:
        raise ValueError("paired errors and group ids must have identical shapes")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    finite = np.isfinite(reference) & np.isfinite(candidate)
    if not bool(finite.any()):
        raise ValueError("RMSE bootstrap requires finite paired observations")
    reference = reference[finite]
    candidate = candidate[finite]
    groups = groups[finite]
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    reference_squares = np.bincount(inverse, weights=np.square(reference))
    candidate_squares = np.bincount(inverse, weights=np.square(candidate))
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, unique.size, size=(repetitions, unique.size))
    drawn_counts = np.sum(counts[draws], axis=1)
    reference_rmse = np.sqrt(
        np.sum(reference_squares[draws], axis=1) / drawn_counts
    )
    candidate_rmse = np.sqrt(
        np.sum(candidate_squares[draws], axis=1) / drawn_counts
    )
    distribution = reference_rmse - candidate_rmse
    observed_reference = _rmse(reference)
    observed_candidate = _rmse(candidate)
    return {
        "metric": "reference_nrmse_minus_candidate_nrmse",
        "observed_reference_nrmse": observed_reference,
        "observed_candidate_nrmse": observed_candidate,
        "observed_improvement": observed_reference - observed_candidate,
        "bootstrap_mean_improvement": float(np.mean(distribution)),
        "ci_low": float(np.quantile(distribution, 0.025)),
        "ci_high": float(np.quantile(distribution, 0.975)),
        "groups": int(unique.size),
        "repetitions": int(repetitions),
        "seed": int(seed),
    }


def _rows_by_seed(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("cell")), int(row.get("seed", -1)))
        if key in indexed:
            raise ValueError(f"duplicate sparse tail result: {key}")
        indexed[key] = row
    return indexed


def _pair_failures(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    bottom_90_change: float,
) -> list[str]:
    failures: list[str] = []
    required_numeric = (
        "daylight_qcrps",
        "ramp_nrmse",
        "ramp_up_nrmse",
        "ramp_down_nrmse",
        "ramp_nmae",
    )
    values = np.asarray(
        [reference.get(name, np.nan) for name in required_numeric]
        + [candidate.get(name, np.nan) for name in required_numeric]
        + [candidate.get("activation_share", np.nan), bottom_90_change],
        dtype=float,
    )
    if not bool(np.all(np.isfinite(values))):
        return ["nonfinite_metric"]
    if reference.get("invariants_pass") is not True or candidate.get("invariants_pass") is not True:
        failures.append("invariant_failure")
    checks = (
        (
            _relative_change(
                float(candidate["daylight_qcrps"]),
                float(reference["daylight_qcrps"]),
            )
            <= 0.005 + 1e-12,
            "daylight_qcrps_degraded_more_than_0.5pct",
        ),
        (
            _relative_change(
                float(candidate["ramp_nrmse"]),
                float(reference["ramp_nrmse"]),
            )
            <= -0.02 + 1e-12,
            "ramp_nrmse_improvement_below_2pct",
        ),
        (
            _relative_change(
                float(candidate["ramp_up_nrmse"]),
                float(reference["ramp_up_nrmse"]),
            )
            <= -0.05 + 1e-12,
            "ramp_up_nrmse_improvement_below_5pct",
        ),
        (
            _relative_change(
                float(candidate["ramp_nmae"]),
                float(reference["ramp_nmae"]),
            )
            <= 0.01 + 1e-12,
            "ramp_nmae_degraded_more_than_1pct",
        ),
        (
            _relative_change(
                float(candidate["ramp_down_nrmse"]),
                float(reference["ramp_down_nrmse"]),
            )
            <= 0.01 + 1e-12,
            "ramp_down_nrmse_degraded_more_than_1pct",
        ),
        (
            bottom_90_change <= 0.01 + 1e-12,
            "bottom_90pct_rmse_degraded_more_than_1pct",
        ),
        (
            0.01 - 1e-12
            <= float(candidate["activation_share"])
            <= 0.20 + 1e-12,
            "activation_share_outside_0.01_0.20",
        ),
    )
    failures.extend(message for passed, message in checks if not passed)
    return failures


def classify_sparse_tail_stage(
    rows: Sequence[Mapping[str, Any]],
    pooled: Mapping[str, Any],
    *,
    multiseed: bool,
) -> dict[str, Any]:
    """Apply the frozen seed-13 or five-seed sparse-tail acceptance gate."""

    indexed = _rows_by_seed(rows)
    required_seeds = REGISTERED_TAIL_SEEDS if multiseed else (13,)
    expected = {(cell, seed) for cell in TAIL_CELLS for seed in required_seeds}
    if set(indexed) != expected:
        return {
            "selection_boundary": "validation_only",
            "evaluation_boundary": "unseen_site_spatial_validation",
            "historical_extrapolation": False,
            "calendar_overlap_policy": "allowed_audited_not_historical",
            "decision": "inconclusive" if multiseed else "keep_current",
            "continue_multiseed": False,
            "failure_reasons": ["incomplete_or_unregistered_matrix"],
        }
    bottom_90_change = float(pooled.get("bottom_90pct_rmse_change", np.nan))
    failures: list[str] = []
    for seed in required_seeds:
        failures.extend(
            _pair_failures(
                indexed[("R00", seed)],
                indexed[("R02", seed)],
                bottom_90_change=bottom_90_change,
            )
        )
    failures = list(dict.fromkeys(failures))
    if not multiseed:
        return {
            "selection_boundary": "validation_only",
            "evaluation_boundary": "unseen_site_spatial_validation",
            "historical_extrapolation": False,
            "calendar_overlap_policy": "allowed_audited_not_historical",
            "decision": None if not failures else "keep_current",
            "continue_multiseed": not failures,
            "failure_reasons": failures,
        }

    if pooled.get("bootstrap_complete") is not True:
        failures.append("bootstrap_incomplete")
    for metric, threshold, message in (
        ("daylight_qcrps", 0.005, "mean_daylight_qcrps_degraded_more_than_0.5pct"),
        ("ramp_nmae", 0.01, "mean_ramp_nmae_degraded_more_than_1pct"),
        ("ramp_down_nrmse", 0.01, "mean_ramp_down_nrmse_degraded_more_than_1pct"),
    ):
        reference_mean = float(
            np.mean([float(indexed[("R00", seed)][metric]) for seed in required_seeds])
        )
        candidate_mean = float(
            np.mean([float(indexed[("R02", seed)][metric]) for seed in required_seeds])
        )
        if _relative_change(candidate_mean, reference_mean) > threshold + 1e-12:
            failures.append(message)
    for metric, threshold, message in (
        ("ramp_nrmse", -0.02, "mean_ramp_nrmse_improvement_below_2pct"),
        ("ramp_up_nrmse", -0.05, "mean_ramp_up_nrmse_improvement_below_5pct"),
    ):
        reference_values = np.asarray(
            [float(indexed[("R00", seed)][metric]) for seed in required_seeds]
        )
        candidate_values = np.asarray(
            [float(indexed[("R02", seed)][metric]) for seed in required_seeds]
        )
        if _relative_change(float(candidate_values.mean()), float(reference_values.mean())) > (
            threshold + 1e-12
        ):
            failures.append(message)
        if int(np.sum(candidate_values < reference_values)) < 4:
            failures.append(f"{metric}_wins_below_4_of_5")
    failures = list(dict.fromkeys(failures))
    return {
        "selection_boundary": "validation_only",
        "evaluation_boundary": "unseen_site_spatial_validation",
        "historical_extrapolation": False,
        "calendar_overlap_policy": "allowed_audited_not_historical",
        "decision": "tail_adjustment_accepted" if not failures else "keep_current",
        "continue_multiseed": False,
        "failure_reasons": failures,
    }


__all__ = [
    "REGISTERED_TAIL_SEEDS",
    "TAIL_CELLS",
    "classify_sparse_tail_stage",
    "paired_site_day_bootstrap",
    "paired_site_day_rmse_bootstrap",
    "paired_sparse_tail_metrics",
]
