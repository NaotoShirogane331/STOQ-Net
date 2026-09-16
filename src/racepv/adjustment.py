"""Registered validation-only selection rules for the RaCE-PV adjustment study."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from racepv.audit import sha256_file
from racepv.config import load_config
from racepv.metrics import average_precision
from racepv.validation_screening import (
    expert_gate_diagnostics,
    paired_site_day_bootstrap,
    qcrps_cells,
    require_aligned_predictions,
)

ADJUSTMENT_SEEDS = (13, 29, 47, 71, 101)


def _binary_counts(
    actual: np.ndarray, predicted: np.ndarray
) -> tuple[int, int, int, int, float, float, float, float]:
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    tn = int(np.sum(~actual & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2_denominator = 4.0 * precision + recall
    f2 = 5.0 * precision * recall / f2_denominator if f2_denominator else 0.0
    return tp, fp, fn, tn, precision, recall, f1, f2


def select_ramp_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    daylight: np.ndarray,
    *,
    thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Select the highest-tie F2 threshold under the frozen precision/recall gate."""

    truth = np.asarray(labels)
    values = np.asarray(probabilities, dtype=float)
    if values.shape == truth.shape + (3,):
        scores = values[..., 1:].sum(axis=-1)
    elif values.shape == truth.shape:
        scores = values
    else:
        raise ValueError("probabilities must be ramp scores or labels.shape + [3]")
    try:
        selected = np.broadcast_to(np.asarray(daylight, dtype=bool), truth.shape).copy()
    except ValueError as exc:
        raise ValueError("daylight is not broadcastable to labels") from exc
    selected &= np.isin(truth, (0, 1, 2)) & np.isfinite(scores)
    if not np.any(selected):
        raise ValueError("threshold selection requires labelled daylight cells")
    scores = scores[selected]
    actual = truth[selected] != 0
    grid = (
        np.round(np.arange(0.05, 0.8001, 0.01), 2)
        if thresholds is None
        else np.asarray(tuple(thresholds), dtype=float)
    )
    if grid.ndim != 1 or grid.size == 0 or np.any(~np.isfinite(grid)) or np.any(
        (grid < 0.0) | (grid > 1.0)
    ):
        raise ValueError("thresholds must be finite values inside [0, 1]")

    rows: list[dict[str, Any]] = []
    for threshold in grid:
        tp, fp, fn, tn, precision, recall, f1, f2 = _binary_counts(
            actual, scores >= threshold
        )
        rows.append(
            {
                "threshold": float(threshold),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f2": f2,
                "gate_pass": precision >= 0.45 and recall >= 0.40,
            }
        )
    eligible = [row for row in rows if row["gate_pass"]]
    pool = eligible or rows
    winner = max(pool, key=lambda row: (float(row["f2"]), float(row["threshold"])))
    passed = bool(eligible)
    return {
        **winner,
        "passed": passed,
        "failure_reasons": [] if passed else ["no_threshold_meets_precision_recall_gate"],
        "selection_rule": (
            "maximize F2 subject to precision>=0.45 and recall>=0.40; "
            "higher threshold breaks exact ties"
        ),
        "grid": rows,
    }


def evaluate_seed13_candidate(
    *,
    base_qcrps: float,
    candidate_qcrps: float,
    precision: float,
    recall: float,
    base_ramp_nrmse: float,
    candidate_ramp_nrmse: float,
    base_ramp_nmae: float,
    candidate_ramp_nmae: float,
    base_ramp_auprc: float,
    candidate_ramp_auprc: float,
    minimum_active_soft_weight: float,
    minimum_active_hard_share: float,
    outputs_finite: bool,
) -> dict[str, Any]:
    """Apply every frozen seed-13 advancement gate without short-circuiting."""

    numeric = np.asarray(
        [
            base_qcrps,
            candidate_qcrps,
            precision,
            recall,
            base_ramp_nrmse,
            candidate_ramp_nrmse,
            base_ramp_nmae,
            candidate_ramp_nmae,
            base_ramp_auprc,
            candidate_ramp_auprc,
            minimum_active_soft_weight,
            minimum_active_hard_share,
        ],
        dtype=float,
    )
    finite = bool(outputs_finite and np.all(np.isfinite(numeric)))
    if base_qcrps <= 0 or base_ramp_nrmse <= 0 or base_ramp_nmae <= 0:
        raise ValueError("base forecast metrics must be positive")
    qcrps_degradation = candidate_qcrps / base_qcrps - 1.0
    ramp_nrmse_improvement = 1.0 - candidate_ramp_nrmse / base_ramp_nrmse
    ramp_nmae_degradation = candidate_ramp_nmae / base_ramp_nmae - 1.0
    ramp_auprc_drop = base_ramp_auprc - candidate_ramp_auprc
    failed: list[str] = []
    if not finite:
        failed.append("nonfinite_output_or_metric")
    if qcrps_degradation > 0.005:
        failed.append("qcrps_degradation_above_0_5pct")
    if precision < 0.45:
        failed.append("ramp_precision_below_0_45")
    if recall < 0.40:
        failed.append("ramp_recall_below_0_40")
    if ramp_nrmse_improvement < 0.02:
        failed.append("ramp_nrmse_improvement_below_2pct")
    if ramp_nmae_degradation > 0.01:
        failed.append("ramp_nmae_degradation_above_1pct")
    if ramp_auprc_drop > 0.01:
        failed.append("ramp_auprc_drop_above_0_01")
    if minimum_active_soft_weight < 0.01 or minimum_active_hard_share < 0.01:
        failed.append("active_expert_utilization_below_1pct")
    return {
        "passed": not failed,
        "failure_reasons": failed,
        "observed": {
            "qcrps_degradation": qcrps_degradation,
            "precision": float(precision),
            "recall": float(recall),
            "ramp_nrmse_improvement": ramp_nrmse_improvement,
            "ramp_nmae_degradation": ramp_nmae_degradation,
            "ramp_auprc_drop": ramp_auprc_drop,
            "minimum_active_soft_weight": float(minimum_active_soft_weight),
            "minimum_active_hard_share": float(minimum_active_hard_share),
        },
    }


def analyze_adjustment_predictions(
    arrays: Mapping[str, np.ndarray],
    levels: Sequence[float],
    *,
    active_experts: Sequence[str],
    threshold: float | None = None,
) -> dict[str, Any]:
    """Summarize one aligned validation prediction archive for adjustment gates."""

    required = {
        "targets",
        "quantiles",
        "state_labels",
        "state_probs",
        "daylight_mask",
        "expert_quantiles",
        "gate_weights",
    }
    missing = required.difference(arrays)
    if missing:
        raise KeyError(f"adjustment predictions are missing: {sorted(missing)}")
    targets = np.asarray(arrays["targets"], dtype=float)
    quantiles = np.asarray(arrays["quantiles"], dtype=float)
    labels = np.asarray(arrays["state_labels"])
    probabilities = np.asarray(arrays["state_probs"], dtype=float)
    daylight = np.asarray(arrays["daylight_mask"], dtype=bool)
    level_values = np.asarray(tuple(levels), dtype=float)
    median_positions = np.flatnonzero(np.isclose(level_values, 0.5, atol=1e-12))
    if median_positions.size != 1:
        raise ValueError("adjustment analysis requires exactly one median quantile")
    median = quantiles[..., int(median_positions[0])]
    ramp = daylight & np.isin(labels, (1, 2)) & np.isfinite(targets) & np.isfinite(median)
    if not np.any(ramp):
        raise ValueError("adjustment analysis requires daylight ramp cells")
    error = median[ramp] - targets[ramp]
    losses = qcrps_cells(targets, quantiles, levels, mask=daylight)
    ramp_losses = losses[ramp & np.isfinite(losses)]
    selected_losses = losses[daylight & np.isfinite(losses)]
    threshold_report = select_ramp_threshold(
        labels,
        probabilities,
        daylight,
        thresholds=None if threshold is None else (threshold,),
    )
    valid_states = daylight & np.isin(labels, (0, 1, 2))
    actual_ramp = labels[valid_states] != 0
    ramp_score = probabilities[valid_states, 1:].sum(axis=-1)
    threshold_report["auprc"] = float(average_precision(actual_ramp, ramp_score))
    expert_report = expert_gate_diagnostics(
        experts=np.asarray(arrays["expert_quantiles"], dtype=float),
        gates=np.asarray(arrays["gate_weights"], dtype=float),
        targets=targets,
        labels=labels,
        daylight=daylight,
        levels=levels,
        active_experts=active_experts,
    )
    utilization = expert_report["overall_utilization"]
    soft = utilization["mean_gate_weight"]
    hard = utilization["hard_routing_share"]
    active = tuple(str(name) for name in active_experts)
    finite_names = required | {"fusion_logits", "state_logits", "router_logits"}
    outputs_finite = all(
        np.all(np.isfinite(np.asarray(arrays[name])))
        for name in finite_names
        if name in arrays
    )
    return {
        "qcrps": {
            "overall": float(np.mean(selected_losses)),
            "ramp": float(np.mean(ramp_losses)),
        },
        "ramp_point": {
            "nmae": float(np.mean(np.abs(error))),
            "nrmse": float(np.sqrt(np.mean(np.square(error)))),
        },
        "ramp_classification": threshold_report,
        "expert_utilization": {
            "minimum_active_soft_weight": float(min(float(soft[name]) for name in active)),
            "minimum_active_hard_share": float(min(float(hard[name]) for name in active)),
            "gate_pass": bool(expert_report["gate_utilization_pass"]),
            "specialization_pass": bool(expert_report["expert_specialization_pass"]),
        },
        "outputs_finite": outputs_finite,
    }


def classify_multiseed_adjustment(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold_only_available: bool,
) -> dict[str, Any]:
    """Apply the frozen five-seed acceptance rule to one selected treatment."""

    by_seed = {int(row["seed"]): row for row in rows}
    missing = [seed for seed in ADJUSTMENT_SEEDS if seed not in by_seed]
    if missing:
        return {
            "status": "inconclusive",
            "failure_reasons": ["missing_seeds:" + ",".join(map(str, missing))],
            "observed": {"missing_seeds": missing},
        }
    selected = [by_seed[seed] for seed in ADJUSTMENT_SEEDS]

    def mean(name: str) -> float:
        return float(np.mean([float(row[name]) for row in selected]))

    precision_recall_seed_passes = sum(
        float(row["precision"]) >= 0.45 and float(row["recall"]) >= 0.40
        for row in selected
    )
    ramp_nrmse_seed_wins = sum(float(row["ramp_nrmse_improvement"]) > 0.0 for row in selected)
    all_finite = all(bool(row["finite"]) for row in selected)
    experts_pass = all(bool(row["expert_ok"]) for row in selected)
    deep_pass = all(bool(row["beats_deep_reference"]) for row in selected)
    raw_metric_names = {
        "candidate_qcrps",
        "reference_qcrps",
        "candidate_ramp_nrmse",
        "reference_ramp_nrmse",
        "candidate_ramp_nmae",
        "reference_ramp_nmae",
    }
    has_raw_metrics = all(raw_metric_names.issubset(row) for row in selected)
    if has_raw_metrics:
        mean_qcrps_degradation = (
            mean("candidate_qcrps") / mean("reference_qcrps") - 1.0
        )
        mean_ramp_nrmse_improvement = (
            1.0 - mean("candidate_ramp_nrmse") / mean("reference_ramp_nrmse")
        )
        mean_ramp_nmae_degradation = (
            mean("candidate_ramp_nmae") / mean("reference_ramp_nmae") - 1.0
        )
    else:
        mean_qcrps_degradation = mean("qcrps_degradation")
        mean_ramp_nrmse_improvement = mean("ramp_nrmse_improvement")
        mean_ramp_nmae_degradation = mean("ramp_nmae_degradation")
    observed = {
        "mean_qcrps_degradation": mean_qcrps_degradation,
        "mean_precision": mean("precision"),
        "mean_recall": mean("recall"),
        "mean_ramp_nrmse_improvement": mean_ramp_nrmse_improvement,
        "mean_ramp_nmae_degradation": mean_ramp_nmae_degradation,
        "precision_recall_seed_passes": precision_recall_seed_passes,
        "ramp_nrmse_seed_wins": ramp_nrmse_seed_wins,
        "all_outputs_finite": all_finite,
        "experts_pass": experts_pass,
        "beats_deep_reference": deep_pass,
    }
    failed: list[str] = []
    if observed["mean_qcrps_degradation"] > 0.005:
        failed.append("mean_qcrps_degradation_above_0_5pct")
    if observed["mean_precision"] < 0.45 or observed["mean_recall"] < 0.40:
        failed.append("mean_precision_or_recall_below_gate")
    if precision_recall_seed_passes < 4:
        failed.append("fewer_than_4_of_5_precision_recall_seed_passes")
    if observed["mean_ramp_nrmse_improvement"] < 0.02:
        failed.append("mean_ramp_nrmse_improvement_below_2pct")
    if ramp_nrmse_seed_wins < 3:
        failed.append("fewer_than_3_of_5_ramp_nrmse_wins")
    if observed["mean_ramp_nmae_degradation"] > 0.01:
        failed.append("mean_ramp_nmae_degradation_above_1pct")
    if not all_finite:
        failed.append("nonfinite_seed_output")
    if not experts_pass:
        failed.append("expert_collapse_or_specialization_failure")
    if not deep_pass:
        failed.append("does_not_beat_frozen_deep_reference")
    if not failed:
        status = "adjustment_accepted"
    elif threshold_only_available:
        status = "threshold_only"
    else:
        status = "keep_current"
    return {"status": status, "failure_reasons": failed, "observed": observed}


def select_factor_candidate(
    reports: Mapping[str, Sequence[Mapping[str, Any]]],
    passing_cells: Sequence[str],
) -> dict[str, Any]:
    """Select the factor winner, requiring seed 29 for a close seed-13 pair."""

    passing = [str(cell) for cell in passing_cells]
    if not passing:
        return {"selected_cell": None, "tie_break_cells": []}
    missing = [cell for cell in passing if not reports.get(cell)]
    if missing:
        raise ValueError(f"passing factor cells have no reports: {missing}")

    def means(cell: str) -> tuple[float, float, float, tuple[int, int]]:
        rows = reports[cell]
        qcrps = float(np.mean([float(row["qcrps"]["overall"]) for row in rows]))
        ramp_nrmse = float(
            np.mean([float(row["ramp_point"]["nrmse"]) for row in rows])
        )
        f2 = float(
            np.mean([float(row["ramp_classification"]["f2"]) for row in rows])
        )
        complexity = {
            "F00": (0, 0),
            "F01": (1, 0),
            "F10": (1, 1),
            "F11": (2, 1),
        }.get(cell, (99, 99))
        return qcrps, ramp_nrmse, -f2, complexity

    seed13_order = sorted(passing, key=means)
    if len(seed13_order) == 1:
        return {"selected_cell": seed13_order[0], "tie_break_cells": []}
    first, second = seed13_order[:2]
    first_score = float(reports[first][0]["qcrps"]["overall"])
    second_score = float(reports[second][0]["qcrps"]["overall"])
    close = abs(first_score - second_score) / min(first_score, second_score) < 0.002
    if close and (len(reports[first]) < 2 or len(reports[second]) < 2):
        return {"selected_cell": None, "tie_break_cells": [first, second]}
    contenders = (
        [first, second]
        if close
        else seed13_order
    )
    selected = min(contenders, key=means)
    return {"selected_cell": selected, "tie_break_cells": []}


def analyze_adjustment_manifest(
    manifest_path: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Analyze the frozen adjustment matrix without opening calibration or test data."""

    from racepv.cli.evaluate import _load_metadata, _load_npz, _validate_prediction_archive

    manifest_file = Path(manifest_path)
    with manifest_file.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise TypeError("adjustment manifest must contain a JSON object")
    if manifest.get("schema_version") != 1:
        raise ValueError("adjustment manifest must use schema_version=1")
    if manifest.get("purpose") != "racepv_one_day_adjustment":
        raise ValueError("adjustment manifest purpose is invalid")
    if manifest.get("selection_boundary") != "validation_only":
        raise ValueError("adjustment manifest must declare validation_only")
    if manifest.get("bootstrap_replicates") != 500:
        raise ValueError("adjustment manifest fixes bootstrap_replicates at 500")
    raw_runs = manifest.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("adjustment manifest runs must be a non-empty list")

    config = load_config(Path(config_path))
    levels = tuple(float(value) for value in config.quantiles)
    loaded: list[dict[str, Any]] = []
    identities: set[tuple[str, str, int]] = set()
    hashes: list[dict[str, Any]] = []
    for raw in raw_runs:
        if not isinstance(raw, dict):
            raise TypeError("every adjustment run must be a JSON object")
        role = str(raw.get("role", ""))
        if role not in {"factor", "historical_current", "deep_reference"}:
            raise ValueError(f"unsupported adjustment run role: {role!r}")
        if isinstance(raw.get("seed"), bool):
            raise ValueError("adjustment run seed must be an integer")
        try:
            seed = int(raw["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("adjustment run seed must be an integer") from exc
        name = str(raw.get("cell" if role == "factor" else "experiment", ""))
        if role == "factor" and name not in {"F00", "F10", "F01", "F11"}:
            raise ValueError(f"factor run has unsupported cell: {name!r}")
        if not name:
            raise ValueError("adjustment run identity must be non-empty")
        identity = (role, name, seed)
        if identity in identities:
            raise ValueError(f"duplicate adjustment run: {identity}")
        identities.add(identity)
        path_value = raw.get("predictions")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"adjustment run {identity} has no predictions path")
        path = Path(path_value)
        if not path.is_absolute():
            path = manifest_file.parent / path
        path = path.resolve()
        if path.name != "validation_predictions.npz":
            raise ValueError("adjustment analysis accepts validation_predictions.npz only")
        metadata_path = path.with_suffix(".json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"adjustment prediction or sidecar is missing: {path}")
        arrays = _load_npz(path)
        metadata = _load_metadata(path)
        if metadata.get("split") != "validation":
            raise ValueError("adjustment analysis requires metadata split='validation'")
        if role == "factor":
            if metadata.get("selection_boundary") != "validation_only":
                raise ValueError("factor archives must declare validation_only")
            if metadata.get("adjustment_cell") != name:
                raise ValueError("factor metadata adjustment_cell differs from manifest")
            if metadata.get("diagnostic_only") is not True:
                raise ValueError("factor archives must declare diagnostic_only=true")
        if int(metadata.get("seed", -1)) != seed:
            raise ValueError("adjustment manifest seed differs from prediction metadata")
        _validate_prediction_archive(
            arrays,
            metadata,
            split="validation",
            levels=levels,
            horizon=config.data.horizon,
            power_upper_bound=config.data.power_upper_bound,
        )
        loaded.append(
            {
                "role": role,
                "name": name,
                "seed": seed,
                "arrays": arrays,
                "metadata": metadata,
                "path": path,
            }
        )
        hashes.append(
            {
                "role": role,
                "name": name,
                "seed": seed,
                "predictions": str(path),
                "prediction_sha256": sha256_file(path),
                "metadata_sha256": sha256_file(metadata_path),
            }
        )

    factors = {(item["name"], item["seed"]): item for item in loaded if item["role"] == "factor"}
    required_factor = [(cell, 13) for cell in ("F00", "F10", "F01", "F11")]
    missing_factor = [cell for cell in required_factor if cell not in factors]
    if missing_factor:
        raise ValueError(f"seed-13 factor matrix is incomplete: {missing_factor}")
    base = factors[("F00", 13)]
    for item in loaded:
        require_aligned_predictions(base["arrays"], item["arrays"])

    historical = [item for item in loaded if item["role"] == "historical_current"]
    historical_threshold: dict[str, Any] | None = None
    if historical:
        historical_threshold = select_ramp_threshold(
            np.concatenate([item["arrays"]["state_labels"] for item in historical]),
            np.concatenate([item["arrays"]["state_probs"] for item in historical]),
            np.concatenate([item["arrays"]["daylight_mask"] for item in historical]),
        )

    flat_threshold = select_ramp_threshold(
        factors[("F00", 13)]["arrays"]["state_labels"],
        factors[("F00", 13)]["arrays"]["state_probs"],
        factors[("F00", 13)]["arrays"]["daylight_mask"],
    )
    hierarchical_threshold = select_ramp_threshold(
        factors[("F10", 13)]["arrays"]["state_labels"],
        factors[("F10", 13)]["arrays"]["state_probs"],
        factors[("F10", 13)]["arrays"]["daylight_mask"],
    )
    family_threshold = {
        "F00": float(flat_threshold["threshold"]),
        "F01": float(flat_threshold["threshold"]),
        "F10": float(hierarchical_threshold["threshold"]),
        "F11": float(hierarchical_threshold["threshold"]),
    }
    factor_reports = {
        cell: analyze_adjustment_predictions(
            factors[(cell, 13)]["arrays"],
            levels,
            active_experts=factors[(cell, 13)]["metadata"].get(
                "active_experts", ("stable", "ramp_up", "ramp_down")
            ),
            threshold=family_threshold[cell],
        )
        for cell in ("F00", "F10", "F01", "F11")
    }
    base_report = factor_reports["F00"]
    candidate_gates: dict[str, Any] = {}
    for cell in ("F10", "F01", "F11"):
        report = factor_reports[cell]
        candidate_gates[cell] = evaluate_seed13_candidate(
            base_qcrps=base_report["qcrps"]["overall"],
            candidate_qcrps=report["qcrps"]["overall"],
            precision=report["ramp_classification"]["precision"],
            recall=report["ramp_classification"]["recall"],
            base_ramp_nrmse=base_report["ramp_point"]["nrmse"],
            candidate_ramp_nrmse=report["ramp_point"]["nrmse"],
            base_ramp_nmae=base_report["ramp_point"]["nmae"],
            candidate_ramp_nmae=report["ramp_point"]["nmae"],
            base_ramp_auprc=base_report["ramp_classification"]["auprc"],
            candidate_ramp_auprc=report["ramp_classification"]["auprc"],
            minimum_active_soft_weight=report["expert_utilization"][
                "minimum_active_soft_weight"
            ],
            minimum_active_hard_share=report["expert_utilization"][
                "minimum_active_hard_share"
            ],
            outputs_finite=report["outputs_finite"],
        )
    passing = [cell for cell, result in candidate_gates.items() if result["passed"]]
    reports_for_selection: dict[str, list[dict[str, Any]]] = {
        cell: [factor_reports[cell]] for cell in passing
    }
    tie_break_seed29_reports: dict[str, dict[str, Any]] = {}
    for cell in passing:
        tie_run = factors.get((cell, 29))
        if tie_run is None:
            continue
        tie_report = analyze_adjustment_predictions(
            tie_run["arrays"],
            levels,
            active_experts=tie_run["metadata"].get(
                "active_experts", ("stable", "ramp_up", "ramp_down")
            ),
            threshold=family_threshold[cell],
        )
        reports_for_selection[cell].append(tie_report)
        tie_break_seed29_reports[cell] = tie_report
    factor_selection = select_factor_candidate(reports_for_selection, passing)
    selected_cell = str(manifest.get("selected_cell") or "") or None
    if selected_cell is not None and selected_cell not in passing:
        raise ValueError("manifest selected_cell did not pass the seed-13 gates")
    if selected_cell is None:
        selected_cell = factor_selection["selected_cell"]
    tie_break_cells = (
        [] if selected_cell is not None else factor_selection["tie_break_cells"]
    )

    stage2: dict[str, Any] = {
        "status": "pending" if selected_cell or tie_break_cells else "not_started",
        "selected_cell": selected_cell,
        "tie_break_cells": tie_break_cells,
    }
    conclusion = (
        "threshold_only"
        if not passing and historical_threshold is not None and historical_threshold["passed"]
        else "keep_current"
        if not passing
        else "inconclusive"
    )
    if selected_cell is not None:
        candidate_by_seed = {
            item["seed"]: item
            for item in loaded
            if item["role"] == "factor" and item["name"] == selected_cell
        }
        historical_by_seed = {item["seed"]: item for item in historical}
        deep_by_seed = {
            item["seed"]: item for item in loaded if item["role"] == "deep_reference"
        }
        if all(seed in candidate_by_seed and seed in historical_by_seed for seed in ADJUSTMENT_SEEDS):
            rows: list[dict[str, Any]] = []
            bootstraps: dict[str, Any] = {}
            candidate_loss_stack: list[np.ndarray] = []
            reference_loss_stack: list[np.ndarray] = []
            bootstrap_arrays: Mapping[str, np.ndarray] | None = None
            for seed in ADJUSTMENT_SEEDS:
                candidate = candidate_by_seed[seed]
                reference = historical_by_seed[seed]
                require_aligned_predictions(candidate["arrays"], reference["arrays"])
                candidate_report = analyze_adjustment_predictions(
                    candidate["arrays"],
                    levels,
                    active_experts=candidate["metadata"].get(
                        "active_experts", ("stable", "ramp_up", "ramp_down")
                    ),
                    threshold=family_threshold[selected_cell],
                )
                reference_report = analyze_adjustment_predictions(
                    reference["arrays"],
                    levels,
                    active_experts=reference["metadata"].get(
                        "active_experts", ("stable", "ramp_up", "ramp_down")
                    ),
                    threshold=(
                        float(historical_threshold["threshold"])
                        if historical_threshold is not None
                        else 0.5
                    ),
                )
                candidate_score = candidate_report["qcrps"]["overall"]
                reference_score = reference_report["qcrps"]["overall"]
                deep = deep_by_seed.get(seed)
                if deep is None:
                    deep_score = None
                else:
                    deep_loss = qcrps_cells(
                        deep["arrays"]["targets"],
                        deep["arrays"]["quantiles"],
                        levels,
                        mask=deep["arrays"]["daylight_mask"],
                    )
                    deep_score = float(np.nanmean(deep_loss))
                rows.append(
                    {
                        "seed": seed,
                        "qcrps_degradation": candidate_score / reference_score - 1.0,
                        "reference_qcrps": reference_score,
                        "precision": candidate_report["ramp_classification"]["precision"],
                        "recall": candidate_report["ramp_classification"]["recall"],
                        "ramp_threshold": family_threshold[selected_cell],
                        "ramp_nrmse_improvement": 1.0
                        - candidate_report["ramp_point"]["nrmse"]
                        / reference_report["ramp_point"]["nrmse"],
                        "candidate_ramp_nrmse": candidate_report["ramp_point"]["nrmse"],
                        "reference_ramp_nrmse": reference_report["ramp_point"]["nrmse"],
                        "ramp_nmae_degradation": candidate_report["ramp_point"]["nmae"]
                        / reference_report["ramp_point"]["nmae"]
                        - 1.0,
                        "candidate_ramp_nmae": candidate_report["ramp_point"]["nmae"],
                        "reference_ramp_nmae": reference_report["ramp_point"]["nmae"],
                        "expert_ok": bool(
                            candidate_report["expert_utilization"]["gate_pass"]
                            and candidate_report["expert_utilization"]["specialization_pass"]
                        ),
                        "finite": bool(candidate_report["outputs_finite"]),
                        "candidate_qcrps": candidate_score,
                        "deep_reference_qcrps": deep_score,
                        "beats_deep_reference": False,
                    }
                )
                candidate_losses = qcrps_cells(
                    candidate["arrays"]["targets"],
                    candidate["arrays"]["quantiles"],
                    levels,
                    mask=candidate["arrays"]["daylight_mask"],
                )
                reference_losses = qcrps_cells(
                    reference["arrays"]["targets"],
                    reference["arrays"]["quantiles"],
                    levels,
                    mask=reference["arrays"]["daylight_mask"],
                )
                candidate_loss_stack.append(candidate_losses)
                reference_loss_stack.append(reference_losses)
                bootstrap_arrays = candidate["arrays"]
                bootstrap = paired_site_day_bootstrap(
                    candidate_losses,
                    reference_losses,
                    candidate["arrays"]["site_ids"],
                    candidate["arrays"]["day_ids"],
                    candidate["arrays"]["daylight_mask"],
                    n_bootstrap=500,
                    seed=30_000 + seed,
                )
                bootstrap.pop("distribution", None)
                bootstraps[str(seed)] = bootstrap
            valid_deep = all(row["deep_reference_qcrps"] is not None for row in rows)
            beats_deep = bool(
                valid_deep
                and np.mean([row["candidate_qcrps"] for row in rows])
                < np.mean([row["deep_reference_qcrps"] for row in rows])
            )
            for row in rows:
                row["beats_deep_reference"] = beats_deep
            assert bootstrap_arrays is not None
            pooled_bootstrap = paired_site_day_bootstrap(
                np.mean(np.stack(candidate_loss_stack), axis=0),
                np.mean(np.stack(reference_loss_stack), axis=0),
                bootstrap_arrays["site_ids"],
                bootstrap_arrays["day_ids"],
                bootstrap_arrays["daylight_mask"],
                n_bootstrap=500,
                seed=40_000,
            )
            pooled_bootstrap.pop("distribution", None)
            decision = classify_multiseed_adjustment(
                rows,
                threshold_only_available=bool(
                    historical_threshold is not None and historical_threshold["passed"]
                ),
            )
            stage2 = {
                "status": "completed",
                "selected_cell": selected_cell,
                "per_seed": rows,
                "paired_site_day_bootstrap": {
                    "per_seed": bootstraps,
                    "pooled_five_seed_mean_loss": pooled_bootstrap,
                },
                "decision": decision,
            }
            conclusion = decision["status"]

    return {
        "schema_version": 1,
        "purpose": "racepv_one_day_adjustment_analysis",
        "selection_boundary": "validation_only",
        "conclusion": conclusion,
        "stage0_historical_threshold": historical_threshold,
        "stage1": {
            "flat_threshold": flat_threshold,
            "hierarchical_threshold": hierarchical_threshold,
            "factor_reports": factor_reports,
            "candidate_gates": candidate_gates,
            "passing_cells": passing,
            "selected_cell": selected_cell,
            "tie_break_cells": tie_break_cells,
            "tie_break_seed29_reports": tie_break_seed29_reports,
        },
        "stage2": stage2,
        "artifact_hashes": hashes,
    }


__all__ = [
    "ADJUSTMENT_SEEDS",
    "analyze_adjustment_predictions",
    "analyze_adjustment_manifest",
    "classify_multiseed_adjustment",
    "evaluate_seed13_candidate",
    "select_factor_candidate",
    "select_ramp_threshold",
]
