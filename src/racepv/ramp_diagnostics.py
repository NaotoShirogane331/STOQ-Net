"""Fail-closed diagnostics for ramp experts, fusion gates, and error tails."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from racepv.validation_screening import qcrps_cells

EXPERT_NAMES = ("stable", "ramp_up", "ramp_down")


def _point_metrics(errors: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "nmae": float("nan"), "nrmse": float("nan")}
    return {
        "count": int(values.size),
        "nmae": float(np.mean(np.abs(values))),
        "nrmse": float(np.sqrt(np.mean(np.square(values)))),
    }


def _probabilistic_metrics(losses: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(losses, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "qcrps": float(np.mean(values)) if values.size else float("nan"),
    }


def _error_tail(errors: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(errors, dtype=float))
    absolute = absolute[np.isfinite(absolute)]
    if absolute.size == 0:
        raise ValueError("error-tail diagnostics require finite errors")
    squared = np.square(absolute)
    total = float(np.sum(squared))

    def share(fraction: float) -> float:
        count = max(1, int(np.ceil(fraction * absolute.size)))
        selected = np.partition(squared, squared.size - count)[-count:]
        return float(np.sum(selected) / total) if total > 0 else 0.0

    return {
        "p50_absolute_error": float(np.quantile(absolute, 0.50)),
        "p90_absolute_error": float(np.quantile(absolute, 0.90)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "p99_absolute_error": float(np.quantile(absolute, 0.99)),
        "top_1pct_squared_error_share": share(0.01),
        "top_5pct_squared_error_share": share(0.05),
    }


def _matching_expert_values(
    experts: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    indices = np.clip(labels, 0, len(EXPERT_NAMES) - 1)[..., None, None]
    return np.take_along_axis(experts, indices, axis=2).squeeze(axis=2)


def _masked_report(
    mask: np.ndarray,
    *,
    fused_error: np.ndarray,
    matching_error: np.ndarray,
    oracle_error: np.ndarray,
    fused_losses: np.ndarray,
    matching_losses: np.ndarray,
    matching_gate: np.ndarray,
    hard_matching: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    count = int(np.sum(selected))
    if count == 0:
        return {"count": 0}
    return {
        "count": count,
        "fused": {
            **_point_metrics(fused_error[selected]),
            **_probabilistic_metrics(fused_losses[selected]),
        },
        "matching_expert": {
            **_point_metrics(matching_error[selected]),
            **_probabilistic_metrics(matching_losses[selected]),
        },
        "oracle_expert_median": _point_metrics(oracle_error[selected]),
        "matching_gate": {
            "mean_weight": float(np.mean(matching_gate[selected])),
            "hard_routing_share": float(np.mean(hard_matching[selected])),
        },
    }


def analyze_ramp_mechanism(
    arrays: Mapping[str, np.ndarray],
    levels: Sequence[float],
    *,
    site_id_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Disentangle expert, fusion, tail, horizon, phase, and site failure modes."""

    required = {
        "targets",
        "quantiles",
        "expert_quantiles",
        "gate_weights",
        "state_labels",
        "daylight_mask",
        "timestamps",
        "site_ids",
    }
    missing = required.difference(arrays)
    if missing:
        raise KeyError(f"ramp diagnostics are missing: {sorted(missing)}")
    targets = np.asarray(arrays["targets"], dtype=float)
    quantiles = np.asarray(arrays["quantiles"], dtype=float)
    experts = np.asarray(arrays["expert_quantiles"], dtype=float)
    gates = np.asarray(arrays["gate_weights"], dtype=float)
    labels = np.asarray(arrays["state_labels"])
    daylight = np.asarray(arrays["daylight_mask"], dtype=bool)
    timestamps = np.asarray(arrays["timestamps"])
    site_ids = np.asarray(arrays["site_ids"])
    if targets.ndim != 2:
        raise ValueError("targets must have shape [N,H]")
    n, horizon = targets.shape
    level_values = np.asarray(tuple(levels), dtype=float)
    median_positions = np.flatnonzero(np.isclose(level_values, 0.5, atol=1e-12))
    if median_positions.size != 1:
        raise ValueError("ramp diagnostics require exactly one median quantile")
    quantile_shape = (n, horizon, level_values.size)
    if quantiles.shape != quantile_shape:
        raise ValueError(f"quantiles must have shape {quantile_shape}")
    if experts.shape != (n, horizon, len(EXPERT_NAMES), level_values.size):
        raise ValueError("expert_quantiles must have shape [N,H,3,Q]")
    if gates.shape != (n, horizon, len(EXPERT_NAMES)):
        raise ValueError("gate_weights must have shape [N,H,3]")
    for name, values in (
        ("state_labels", labels),
        ("daylight_mask", daylight),
        ("timestamps", timestamps),
    ):
        if values.shape != (n, horizon):
            raise ValueError(f"{name} must have shape [N,H]")
    if site_ids.shape == (n,):
        expanded_sites = np.broadcast_to(site_ids[:, None], (n, horizon))
    elif site_ids.shape == (n, horizon):
        expanded_sites = site_ids
    else:
        raise ValueError("site_ids must have shape [N] or [N,H]")
    finite_outputs = all(
        np.all(np.isfinite(values)) for values in (quantiles, experts, gates)
    )
    if not finite_outputs:
        raise ValueError("ramp diagnostic predictions must be finite")
    if np.any(gates < -1e-7) or not np.allclose(gates.sum(axis=-1), 1.0, atol=1e-6):
        raise ValueError("gate_weights must form a finite probability simplex")

    valid = daylight & np.isfinite(targets) & np.isin(labels, (0, 1, 2))
    ramp = valid & np.isin(labels, (1, 2))
    if not np.any(ramp):
        raise ValueError("ramp diagnostics require daylight ramp cells")
    median_index = int(median_positions[0])
    fused_median = quantiles[..., median_index]
    matching_quantiles = _matching_expert_values(experts, labels)
    matching_median = matching_quantiles[..., median_index]
    expert_medians = experts[..., median_index]
    fused_error = fused_median - targets
    matching_error = matching_median - targets
    all_expert_error = expert_medians - targets[..., None]
    oracle_indices = np.argmin(np.abs(all_expert_error), axis=-1)
    oracle_error = np.take_along_axis(
        all_expert_error, oracle_indices[..., None], axis=-1
    ).squeeze(axis=-1)
    fused_losses = qcrps_cells(targets, quantiles, level_values, mask=valid)
    matching_losses = qcrps_cells(
        targets, matching_quantiles, level_values, mask=valid
    )
    label_indices = np.clip(labels, 0, len(EXPERT_NAMES) - 1)
    matching_gate = np.take_along_axis(
        gates, label_indices[..., None], axis=-1
    ).squeeze(axis=-1)
    hard_matching = np.argmax(gates, axis=-1) == label_indices

    common = {
        "fused_error": fused_error,
        "matching_error": matching_error,
        "oracle_error": oracle_error,
        "fused_losses": fused_losses,
        "matching_losses": matching_losses,
        "matching_gate": matching_gate,
        "hard_matching": hard_matching,
    }
    ramp_report = _masked_report(ramp, **common)
    ramp_report["fused_error_tail"] = _error_tail(fused_error[ramp])
    ramp_report["matching_expert_error_tail"] = _error_tail(matching_error[ramp])

    by_state = {
        EXPERT_NAMES[state]: _masked_report(valid & (labels == state), **common)
        for state in (1, 2)
    }
    by_lead = []
    for lead in range(horizon):
        lead_mask = np.zeros_like(ramp)
        lead_mask[:, lead] = ramp[:, lead]
        by_lead.append(
            {
                "lead": lead + 1,
                "ramp_count": int(np.sum(lead_mask)),
                **_masked_report(lead_mask, **common),
            }
        )

    phase_masks = {
        "boundary_ramp": np.zeros_like(valid),
        "onset_from_stable": np.zeros_like(valid),
        "continuation": np.zeros_like(valid),
        "direction_reversal": np.zeros_like(valid),
        "post_ramp_stable": np.zeros_like(valid),
    }
    phase_masks["boundary_ramp"][:, 0] = ramp[:, 0]
    if horizon > 1:
        previous = labels[:, :-1]
        current = labels[:, 1:]
        current_valid = valid[:, 1:]
        phase_masks["onset_from_stable"][:, 1:] = (
            current_valid & np.isin(current, (1, 2)) & (previous == 0)
        )
        phase_masks["continuation"][:, 1:] = (
            current_valid & np.isin(current, (1, 2)) & (previous == current)
        )
        phase_masks["direction_reversal"][:, 1:] = (
            current_valid
            & np.isin(current, (1, 2))
            & np.isin(previous, (1, 2))
            & (previous != current)
        )
        phase_masks["post_ramp_stable"][:, 1:] = (
            current_valid & (current == 0) & np.isin(previous, (1, 2))
        )
    by_phase = {
        name: _masked_report(mask, **common) for name, mask in phase_masks.items()
    }

    by_site: dict[str, Any] = {}
    for raw_site in np.unique(expanded_sites):
        site_mask = ramp & (expanded_sites == raw_site)
        site_key = str(int(raw_site)) if np.issubdtype(type(raw_site), np.integer) else str(raw_site)
        site_name = str(site_id_map.get(site_key, site_key)) if site_id_map else site_key
        by_site[site_name] = {
            "site_id": site_key,
            "ramp_count": int(np.sum(site_mask)),
            **_masked_report(site_mask, **common),
        }

    horizon_bands: dict[str, Any] = {}
    for name, start, stop in (("short_1_4", 0, 4), ("middle_5_12", 4, 12), ("long_13_24", 12, 24)):
        band_mask = np.zeros_like(ramp)
        band_mask[:, start : min(stop, horizon)] = ramp[:, start : min(stop, horizon)]
        horizon_bands[name] = _masked_report(band_mask, **common)

    return {
        "schema_version": 1,
        "population": {
            "origins": n,
            "horizon": horizon,
            "daylight_cells": int(np.sum(valid)),
            "ramp_cells": int(np.sum(ramp)),
            "timestamps_present": bool(timestamps.size),
        },
        "ramp": ramp_report,
        "by_state": by_state,
        "by_lead": by_lead,
        "by_horizon_band": horizon_bands,
        "by_phase": by_phase,
        "by_site": by_site,
    }


def recommend_next_mechanism(report: Mapping[str, Any]) -> dict[str, Any]:
    """Choose a bounded next diagnostic from measured, not assumed, headroom."""

    ramp = report["ramp"]
    fused = float(ramp["fused"]["nrmse"])
    matching = float(ramp["matching_expert"]["nrmse"])
    oracle = float(ramp["oracle_expert_median"]["nrmse"])
    if fused <= 0 or matching < 0 or oracle < 0:
        raise ValueError("recommendation requires valid positive reference errors")
    matching_gain = 1.0 - matching / fused
    oracle_gain = 1.0 - oracle / fused
    hard_matching = float(ramp["matching_gate"]["hard_routing_share"])
    tail_share = float(ramp["fused_error_tail"]["top_5pct_squared_error_share"])
    if matching_gain >= 0.02:
        diagnosis = "fusion_bottleneck"
        cells = ["G00", "G02"]
        reason = "matching expert has at least 2% nRMSE headroom over fused output"
    elif oracle_gain >= 0.02:
        diagnosis = "expert_tail_bottleneck"
        cells = ["T00", "T02"]
        reason = "cell-wise expert oracle has headroom but named matching expert does not"
    else:
        diagnosis = "no_registered_headroom"
        cells = []
        reason = "neither matching nor oracle experts provide 2% nRMSE headroom"
    return {
        "diagnosis": diagnosis,
        "candidate_cells": cells,
        "requires_gpu": bool(cells),
        "observed": {
            "matching_expert_nrmse_gain": matching_gain,
            "oracle_expert_nrmse_gain": oracle_gain,
            "matching_gate_hard_share": hard_matching,
            "top_5pct_squared_error_share": tail_share,
        },
        "reason": reason,
        "selection_rule": (
            "fusion if matching expert gain>=2%; else expert-tail if oracle gain>=2%; "
            "otherwise stop"
        ),
    }
