"""Validation-only state-head error diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from racepv.metrics import average_precision

STATE_NAMES = ("stable", "ramp_up", "ramp_down")


def _binary_report(
    selected: np.ndarray,
    *,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | int]:
    mask = np.asarray(selected, dtype=bool)
    truth = actual[mask]
    labels = predicted[mask]
    tp = int(np.sum(truth & labels))
    fp = int(np.sum(~truth & labels))
    fn = int(np.sum(truth & ~labels))
    tn = int(np.sum(~truth & ~labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    false_negative_rate = fn / (fn + tp) if fn + tp else 0.0
    return {
        "count": int(np.sum(mask)),
        "ramp_count": int(np.sum(truth)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "error_rate": float((fp + fn) / truth.size) if truth.size else 0.0,
    }


def _reliability_report(
    score: np.ndarray,
    actual: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(score[valid], dtype=float)
    truth = np.asarray(actual[valid], dtype=float)
    edges = np.linspace(0.0, 1.0, 11)
    rows: list[dict[str, Any]] = []
    weighted_gap = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        if index == len(edges) - 2:
            selected = (values >= lower) & (values <= upper)
        else:
            selected = (values >= lower) & (values < upper)
        count = int(np.sum(selected))
        mean_score = float(np.mean(values[selected])) if count else None
        observed_rate = float(np.mean(truth[selected])) if count else None
        gap = abs(mean_score - observed_rate) if count else None
        if count:
            assert gap is not None
            weighted_gap += count * gap
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_score": mean_score,
                "observed_ramp_rate": observed_rate,
                "absolute_gap": gap,
            }
        )
    auprc = float(average_precision(truth.astype(bool), values))
    return {
        "count": int(values.size),
        "brier": float(np.mean(np.square(values - truth))),
        "auprc": auprc if np.isfinite(auprc) else None,
        "ece": float(weighted_gap / values.size),
        "bins": rows,
    }


def _precision_recall_curve(
    score: np.ndarray,
    actual: np.ndarray,
    valid: np.ndarray,
) -> list[dict[str, float | int]]:
    rows = []
    for threshold in np.round(np.arange(0.05, 0.8001, 0.01), 2):
        rows.append(
            {
                "threshold": float(threshold),
                **_binary_report(
                    valid,
                    actual=actual,
                    predicted=score >= threshold,
                ),
            }
        )
    return rows


def _phase_masks(labels: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    masks = {
        "first_lead": np.zeros_like(valid),
        "onset_from_stable": np.zeros_like(valid),
        "ramp_continuation": np.zeros_like(valid),
        "direction_reversal": np.zeros_like(valid),
        "post_ramp_stable": np.zeros_like(valid),
        "stable_continuation": np.zeros_like(valid),
    }
    masks["first_lead"][:, 0] = valid[:, 0]
    if labels.shape[1] == 1:
        return masks
    previous = labels[:, :-1]
    current = labels[:, 1:]
    pair_valid = valid[:, 1:] & np.isin(previous, (0, 1, 2))
    current_ramp = np.isin(current, (1, 2))
    previous_ramp = np.isin(previous, (1, 2))
    masks["onset_from_stable"][:, 1:] = pair_valid & current_ramp & (previous == 0)
    masks["ramp_continuation"][:, 1:] = pair_valid & current_ramp & (previous == current)
    masks["direction_reversal"][:, 1:] = (
        pair_valid & current_ramp & previous_ramp & (previous != current)
    )
    masks["post_ramp_stable"][:, 1:] = pair_valid & (current == 0) & previous_ramp
    masks["stable_continuation"][:, 1:] = pair_valid & (current == 0) & (previous == 0)
    return masks


def _fixed_band_reports(
    values: np.ndarray,
    valid: np.ndarray,
    definitions: tuple[tuple[str, float, float], ...],
    *,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    reports = {}
    finite = np.isfinite(values)
    for name, lower, upper in definitions:
        selected = valid & finite & (values >= lower) & (values < upper)
        reports[name] = _binary_report(selected, actual=actual, predicted=predicted)
    return reports


def _expanded_sites(site_ids: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(site_ids)
    if values.shape == (shape[0],):
        return np.broadcast_to(values[:, None], shape)
    if values.shape == (shape[0], 1):
        return np.broadcast_to(values, shape)
    if values.shape == shape:
        return values
    raise ValueError("site_ids must have shape [N], [N,1], or [N,H]")


def _site_name(raw_site: Any, site_id_map: Mapping[str, str] | None) -> tuple[str, str]:
    if isinstance(raw_site, np.generic):
        raw_site = raw_site.item()
    site_key = str(raw_site)
    return site_key, str(site_id_map.get(site_key, site_key)) if site_id_map else site_key


def _utc_hours(timestamps: np.ndarray) -> np.ndarray | None:
    values = np.asarray(timestamps)
    try:
        if np.issubdtype(values.dtype, np.datetime64):
            hours = values.astype("datetime64[h]").astype(np.int64)
        elif np.issubdtype(values.dtype, np.integer):
            hours = values.astype("datetime64[ns]").astype("datetime64[h]").astype(np.int64)
        else:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return np.mod(hours, 24).astype(np.int64)


def _rank_error_slices(
    dimensions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    error_name: str,
    limit: int = 8,
) -> list[dict[str, Any]]:
    rows = []
    for dimension, slices in dimensions.items():
        for name, report in slices.items():
            if int(report.get("count", 0)) == 0:
                continue
            rows.append({"dimension": dimension, "name": name, **report})
    return sorted(
        rows,
        key=lambda row: (int(row[error_name]), float(row["error_rate"])),
        reverse=True,
    )[:limit]


def analyze_state_errors(
    arrays: Mapping[str, np.ndarray],
    *,
    threshold: float,
    site_id_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Decompose fixed-threshold state errors without selecting or training a model."""

    required = {
        "state_labels",
        "state_probs",
        "daylight_mask",
        "targets",
        "timestamps",
        "site_ids",
    }
    missing = required.difference(arrays)
    if missing:
        raise KeyError(f"state error diagnostics are missing: {sorted(missing)}")
    if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be finite and lie inside (0, 1]")
    labels = np.asarray(arrays["state_labels"])
    if labels.ndim != 2:
        raise ValueError("state_labels must have shape [N,H]")
    n, horizon = labels.shape
    probabilities = np.asarray(arrays["state_probs"], dtype=float)
    if probabilities.shape != (n, horizon, 3):
        raise ValueError("state_probs must have shape [N,H,3]")
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities < -1e-8)
        or not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-6)
    ):
        raise ValueError("state_probs must form a finite probability simplex")
    daylight = np.asarray(arrays["daylight_mask"], dtype=bool)
    targets = np.asarray(arrays["targets"], dtype=float)
    timestamps = np.asarray(arrays["timestamps"])
    for name, values in (
        ("daylight_mask", daylight),
        ("targets", targets),
        ("timestamps", timestamps),
    ):
        if values.shape != (n, horizon):
            raise ValueError(f"{name} must have shape [N,H]")
    sites = _expanded_sites(np.asarray(arrays["site_ids"]), labels.shape)
    score = probabilities[..., 1:].sum(axis=-1)
    valid = daylight & np.isin(labels, (0, 1, 2)) & np.isfinite(targets)
    if not np.any(valid):
        raise ValueError("state error diagnostics require labelled daylight cells")
    actual = labels != 0
    predicted = score >= threshold

    by_lead = []
    for lead in range(horizon):
        selected = np.zeros_like(valid)
        selected[:, lead] = valid[:, lead]
        by_lead.append(
            {
                "lead": lead + 1,
                **_binary_report(selected, actual=actual, predicted=predicted),
            }
        )
    by_site: dict[str, dict[str, Any]] = {}
    for raw_site in np.unique(sites):
        site_key, site_name = _site_name(raw_site, site_id_map)
        by_site[site_name] = {
            "site_id": site_key,
            **_binary_report(valid & (sites == raw_site), actual=actual, predicted=predicted),
        }
    by_true_state = {
        STATE_NAMES[state]: _binary_report(
            valid & (labels == state), actual=actual, predicted=predicted
        )
        for state in range(3)
    }
    phase_masks = _phase_masks(labels, valid)
    by_phase = {
        name: _binary_report(mask, actual=actual, predicted=predicted)
        for name, mask in phase_masks.items()
    }
    target_delta = np.full_like(targets, np.nan, dtype=float)
    if horizon > 1:
        target_delta[:, 1:] = np.round(np.abs(targets[:, 1:] - targets[:, :-1]), decimals=12)
    magnitude_definitions = (
        ("lt_0.025", 0.0, 0.025),
        ("0.025_to_0.05", 0.025, 0.05),
        ("0.05_to_0.10", 0.05, 0.10),
        ("0.10_to_0.20", 0.10, 0.20),
        ("ge_0.20", 0.20, float("inf")),
    )
    by_magnitude = _fixed_band_reports(
        target_delta,
        valid,
        magnitude_definitions,
        actual=actual,
        predicted=predicted,
    )
    by_power = _fixed_band_reports(
        targets,
        valid,
        (
            ("0_to_0.10", 0.0, 0.10),
            ("0.10_to_0.30", 0.10, 0.30),
            ("0.30_to_0.60", 0.30, 0.60),
            ("ge_0.60", 0.60, float("inf")),
        ),
        actual=actual,
        predicted=predicted,
    )
    solar_values = arrays.get("solar_elevation")
    if solar_values is None:
        solar_report: dict[str, Any] = {"available": False, "reason": "not_archived"}
    else:
        solar = np.asarray(solar_values, dtype=float)
        if solar.shape != (n, horizon):
            raise ValueError("solar_elevation must have shape [N,H]")
        solar_report = {
            "available": True,
            "by_band": _fixed_band_reports(
                solar,
                valid,
                (
                    ("lt_0deg", float("-inf"), 0.0),
                    ("0_to_15deg", 0.0, 15.0),
                    ("15_to_30deg", 15.0, 30.0),
                    ("30_to_60deg", 30.0, 60.0),
                    ("ge_60deg", 60.0, float("inf")),
                ),
                actual=actual,
                predicted=predicted,
            ),
        }
    hours = _utc_hours(timestamps)
    by_utc_hour = (
        {
            f"{hour:02d}": _binary_report(
                valid & (hours == hour), actual=actual, predicted=predicted
            )
            for hour in range(24)
        }
        if hours is not None
        else {}
    )
    predicted_states = np.argmax(probabilities, axis=-1)
    direction_confusion = {
        STATE_NAMES[true_state]: {
            STATE_NAMES[predicted_state]: int(
                np.sum(valid & (labels == true_state) & (predicted_states == predicted_state))
            )
            for predicted_state in range(3)
        }
        for true_state in range(3)
    }
    ranking_dimensions = {
        "site": by_site,
        "lead": {f"lead_{row['lead']}": row for row in by_lead},
        "phase": by_phase,
        "magnitude": by_magnitude,
        "target_power": by_power,
    }
    return {
        "schema_version": 1,
        "threshold": float(threshold),
        "population": {
            "origins": n,
            "horizon": horizon,
            "valid_daylight_cells": int(np.sum(valid)),
            "ramp_cells": int(np.sum(valid & actual)),
        },
        "overall": _binary_report(valid, actual=actual, predicted=predicted),
        "probability_reliability": _reliability_report(score, actual, valid),
        "precision_recall_curve": _precision_recall_curve(score, actual, valid),
        "direction_confusion": direction_confusion,
        "by_lead": by_lead,
        "by_site": by_site,
        "by_true_state": by_true_state,
        "by_phase": by_phase,
        "by_target_step_magnitude": by_magnitude,
        "by_target_power": by_power,
        "by_utc_hour": by_utc_hour,
        "solar_elevation": solar_report,
        "largest_false_positive_slices": _rank_error_slices(ranking_dimensions, error_name="fp"),
        "largest_false_negative_slices": _rank_error_slices(ranking_dimensions, error_name="fn"),
    }
