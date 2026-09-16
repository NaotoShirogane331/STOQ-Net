"""Validation-only diagnostics for the one-day RaCE-PV screening experiment."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from racepv.audit import sha256_file
from racepv.config import load_config
from racepv.diagnostics import (
    SPECIALIST_DIAGNOSTIC_WEIGHTS,
    SPECIALIST_WEIGHT_SWEEP,
)
from racepv.metrics import average_precision

STATE_NAMES = ("stable", "ramp_up", "ramp_down")
CORE_ABLATIONS = ("equal_fusion", "stable_only", "no_specialist_loss")


def qcrps_cells(
    targets: np.ndarray,
    predictions: np.ndarray,
    levels: Sequence[float],
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return trapezoidal qCRPS for every forecast cell."""

    truth = np.asarray(targets, dtype=float)
    forecast = np.asarray(predictions, dtype=float)
    quantiles = np.asarray(tuple(levels), dtype=float)
    if quantiles.ndim != 1 or np.any((quantiles <= 0.0) | (quantiles >= 1.0)):
        raise ValueError("quantile levels must be one-dimensional inside (0, 1)")
    if np.any(np.diff(quantiles) <= 0.0):
        raise ValueError("quantile levels must be strictly increasing")
    if forecast.shape != truth.shape + (quantiles.size,):
        raise ValueError("predictions must have shape targets.shape + [Q]")
    valid = np.isfinite(truth) & np.all(np.isfinite(forecast), axis=-1)
    if mask is not None:
        try:
            valid &= np.broadcast_to(np.asarray(mask, dtype=bool), truth.shape)
        except ValueError as error:
            raise ValueError("mask is not broadcastable to target shape") from error
    residual = truth[..., None] - forecast
    pinball = np.maximum(quantiles * residual, (quantiles - 1.0) * residual)
    integration_levels = np.concatenate(([0.0], quantiles, [1.0]))
    endpoint = np.zeros(truth.shape + (1,), dtype=float)
    integration_values = np.concatenate((endpoint, pinball, endpoint), axis=-1)
    integrated = np.sum(
        (integration_values[..., 1:] + integration_values[..., :-1])
        * np.diff(integration_levels)
        * 0.5,
        axis=-1,
    ) * 2.0
    return np.where(valid, integrated, np.nan)


def state_head_diagnostics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    daylight: np.ndarray,
) -> dict[str, Any]:
    """Audit physical-state probabilities without consulting forecast gates."""

    truth = np.asarray(labels)
    probs = np.asarray(probabilities, dtype=float)
    if probs.shape != truth.shape + (3,):
        raise ValueError("state probabilities must have shape labels.shape + [3]")
    if np.any(~np.isfinite(probs)) or np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("state probabilities must be finite values inside [0, 1]")
    if not np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5):
        raise ValueError("state probabilities must lie on the simplex")
    try:
        valid = np.broadcast_to(np.asarray(daylight, dtype=bool), truth.shape).copy()
    except ValueError as error:
        raise ValueError("daylight mask is not broadcastable to labels") from error
    valid &= np.isin(truth, (0, 1, 2))
    if not np.any(valid):
        raise ValueError("state diagnostics require labelled daylight cells")
    selected_truth = truth[valid].astype(int)
    selected_probs = probs[valid]
    expected = np.eye(3, dtype=float)[selected_truth]
    brier = float(np.mean(np.sum(np.square(selected_probs - expected), axis=-1)))
    chosen = selected_probs[np.arange(selected_truth.size), selected_truth]
    nll = float(np.mean(-np.log(np.clip(chosen, np.finfo(float).tiny, 1.0))))

    per_class: dict[str, dict[str, float | int]] = {}
    failures: list[str] = []
    auprcs: list[float] = []
    for index, name in enumerate(STATE_NAMES):
        actual = selected_truth == index
        support = int(np.sum(actual))
        prevalence = float(np.mean(actual))
        auprc = average_precision(actual, selected_probs[:, index])
        per_class[name] = {
            "support": support,
            "prevalence": prevalence,
            "auprc": float(auprc),
        }
        auprcs.append(float(auprc))
        if support == 0:
            failures.append(f"missing_state_support:{name}")
        elif not np.isfinite(auprc) or auprc <= prevalence:
            failures.append(f"auprc_not_above_prevalence:{name}")
    return {
        "count": int(selected_truth.size),
        "brier": brier,
        "nll": nll,
        "macro_auprc": float(np.mean(auprcs)) if np.all(np.isfinite(auprcs)) else float("nan"),
        "per_class": per_class,
        "pass_rule": "every class has support and AUPRC above its prevalence",
        "passed": not failures,
        "failure_reasons": failures,
    }


def expert_gate_diagnostics(
    *,
    experts: np.ndarray,
    gates: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    daylight: np.ndarray,
    levels: Sequence[float],
    active_experts: Sequence[str],
) -> dict[str, Any]:
    """Measure expert specialization and gate use as separate forecast diagnostics."""

    target = np.asarray(targets, dtype=float)
    state = np.asarray(labels)
    expert_values = np.asarray(experts, dtype=float)
    gate_values = np.asarray(gates, dtype=float)
    if expert_values.shape != target.shape + (3, len(tuple(levels))):
        raise ValueError("expert quantiles have the wrong shape")
    if gate_values.shape != target.shape + (3,):
        raise ValueError("gate weights have the wrong shape")
    if np.any(~np.isfinite(gate_values)) or not np.allclose(
        gate_values.sum(axis=-1), 1.0, atol=1e-5
    ):
        raise ValueError("gate weights must be finite convex weights")
    active = tuple(str(value) for value in active_experts)
    if not active or len(set(active)) != len(active):
        raise ValueError("active experts must be a non-empty unique subset")
    unknown = sorted(set(active).difference(STATE_NAMES))
    if unknown:
        raise ValueError(f"unknown active experts: {unknown}")
    valid_daylight = np.broadcast_to(np.asarray(daylight, dtype=bool), target.shape)
    if not np.any(valid_daylight):
        raise ValueError("expert diagnostics require daylight forecast cells")

    state_performance: dict[str, Any] = {}
    specialization_failures: list[str] = []
    active_indices = [STATE_NAMES.index(name) for name in active]
    for state_index, state_name in enumerate(STATE_NAMES):
        selected = valid_daylight & (state == state_index)
        if not np.any(selected):
            specialization_failures.append(f"missing_state_support:{state_name}")
            state_performance[state_name] = {"count": 0, "expert_qcrps": {}}
            continue
        losses = {
            expert_name: float(
                np.nanmean(
                    qcrps_cells(
                        target,
                        expert_values[..., expert_index, :],
                        levels,
                        mask=selected,
                    )
                )
            )
            for expert_index, expert_name in enumerate(STATE_NAMES)
        }
        best_active = min(active, key=lambda name: (losses[name], name))
        state_performance[state_name] = {
            "count": int(np.sum(selected)),
            "expert_qcrps": losses,
            "best_active_expert": best_active,
        }
        if state_name in active and best_active != state_name:
            specialization_failures.append(
                f"matching_expert_not_best:{state_name}:best={best_active}"
            )

    selected_gates = gate_values[valid_daylight]
    means = selected_gates.mean(axis=0)
    hard = np.argmax(selected_gates, axis=-1)
    shares = np.asarray([np.mean(hard == index) for index in range(3)], dtype=float)
    collapsed = bool(
        np.any(means[active_indices] < 0.01) or np.any(shares[active_indices] < 0.01)
    )
    inactive_indices = [index for index in range(3) if index not in active_indices]
    inactive_nonzero = bool(
        inactive_indices
        and np.any(np.abs(selected_gates[:, inactive_indices]) > 1e-7)
    )

    diversity: dict[str, float] = {}
    for left in range(3):
        for right in range(left + 1, 3):
            name = f"{STATE_NAMES[left]}__vs__{STATE_NAMES[right]}"
            diversity[name] = float(
                np.mean(
                    np.abs(
                        expert_values[..., left, :] - expert_values[..., right, :]
                    )[valid_daylight]
                )
            )
    if len(active_indices) > 1:
        active_pairs = [
            value
            for name, value in diversity.items()
            if all(part in active for part in name.split("__vs__"))
        ]
        diversity_pass = bool(active_pairs and all(value > 1e-6 for value in active_pairs))
    else:
        diversity_pass = True
    if not diversity_pass:
        specialization_failures.append("active_expert_predictions_not_diverse")

    return {
        "expert_order": list(STATE_NAMES),
        "active_experts": list(active),
        "performance_by_true_state": state_performance,
        "expert_prediction_diversity": diversity,
        "expert_specialization_pass": not specialization_failures,
        "expert_specialization_failure_reasons": specialization_failures,
        "overall_utilization": {
            "mean_gate_weight": {
                name: float(means[index]) for index, name in enumerate(STATE_NAMES)
            },
            "hard_routing_share": {
                name: float(shares[index]) for index, name in enumerate(STATE_NAMES)
            },
            "collapse_threshold": 0.01,
            "collapsed_active_expert": collapsed,
            "inactive_expert_has_nonzero_weight": inactive_nonzero,
        },
        "gate_utilization_pass": not collapsed and not inactive_nonzero,
        "interpretation_boundary": (
            "fusion gates optimize forecasts and are not state probabilities"
        ),
    }


def paired_site_day_bootstrap(
    model_losses: np.ndarray,
    reference_losses: np.ndarray,
    site_ids: np.ndarray,
    day_ids: np.ndarray,
    mask: np.ndarray,
    *,
    n_bootstrap: int = 500,
    seed: int = 13,
) -> dict[str, Any]:
    """Paired, site-stratified bootstrap over observed site-day units."""

    model = np.asarray(model_losses, dtype=float)
    reference = np.asarray(reference_losses, dtype=float)
    days = np.asarray(day_ids)
    if model.shape != reference.shape or days.shape != model.shape:
        raise ValueError("losses and day_ids must share shape [N,H]")
    sites = np.asarray(site_ids).reshape(-1)
    if model.ndim != 2 or sites.size != model.shape[0]:
        raise ValueError("site_ids must provide one site for each forecast origin")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    valid = (
        np.broadcast_to(np.asarray(mask, dtype=bool), model.shape)
        & np.isfinite(model)
        & np.isfinite(reference)
    )
    if not np.any(valid):
        raise ValueError("paired bootstrap has no finite selected cells")
    site_grid = np.broadcast_to(sites[:, None], model.shape)
    difference = reference - model
    strata: list[tuple[np.ndarray, np.ndarray]] = []
    for site in np.unique(site_grid[valid]):
        site_selected = valid & (site_grid == site)
        sums: list[float] = []
        counts: list[int] = []
        for day in np.unique(days[site_selected]):
            selected = site_selected & (days == day)
            sums.append(float(np.sum(difference[selected])))
            counts.append(int(np.sum(selected)))
        strata.append((np.asarray(sums, dtype=float), np.asarray(counts, dtype=int)))

    estimate = float(np.sum(difference[valid]) / np.sum(valid))
    rng = np.random.default_rng(seed)
    distribution = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        total = 0.0
        count = 0
        for sums, counts in strata:
            draws = rng.integers(0, len(sums), size=len(sums))
            total += float(np.sum(sums[draws]))
            count += int(np.sum(counts[draws]))
        distribution[iteration] = total / count
    low, high = np.quantile(distribution, (0.025, 0.975))
    return {
        "unit": "site_day",
        "method": "site-stratified paired percentile bootstrap",
        "paired_resampling": True,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "site_count": len(strata),
        "site_day_count": int(sum(len(sums) for sums, _ in strata)),
        "selected_cell_count": int(np.sum(valid)),
        "estimate_reference_minus_model": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "distribution": distribution,
    }


def classify_screening(
    *,
    seed_improvements: Mapping[int, float],
    mean_improvement: float,
    bootstrap_ci_low: float,
    bootstrap_ci_high: float,
    naive_baselines_pass: bool,
    worse_than_deep_by_more_than_two_percent: bool,
    core_ablation_effects: Mapping[str, float],
    state_head_pass: bool,
    expert_specialization_pass: bool,
    gate_utilization_pass: bool,
    expected_seeds: Sequence[int] = (13, 29, 47),
    required_inputs_complete: bool = True,
    incomplete_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    """Apply the preregistered four-way screening decision."""

    seeds = {int(seed): float(value) for seed, value in seed_improvements.items()}
    expected = tuple(int(seed) for seed in expected_seeds)
    missing = [seed for seed in expected if seed not in seeds]
    wins = sum(seeds.get(seed, -np.inf) > 0.0 for seed in expected)
    failed: list[str] = []

    if not required_inputs_complete:
        failed.extend(str(reason) for reason in incomplete_reasons)
        if not failed:
            failed.append("required_stage_one_runs_incomplete")
        status = "inconclusive"
    elif not naive_baselines_pass:
        failed.append("does_not_beat_both_naive_baselines")
        status = "redesign"
    elif worse_than_deep_by_more_than_two_percent:
        failed.append("more_than_2pct_worse_than_strongest_deep_baseline")
        status = "redesign"
    elif missing:
        failed.append("missing_required_seeds:" + ",".join(str(seed) for seed in missing))
        status = "inconclusive"
    else:
        forecast_failures: list[str] = []
        if mean_improvement < 0.02:
            forecast_failures.append("mean_qcrps_improvement_below_2pct")
        if wins < 2:
            forecast_failures.append("fewer_than_2_of_3_seed_wins")
        if not np.isfinite(bootstrap_ci_low) or bootstrap_ci_low <= 0.0:
            forecast_failures.append("paired_bootstrap_ci_crosses_zero")
        if forecast_failures:
            failed.extend(forecast_failures)
            status = "inconclusive"
        else:
            mechanism_failures: list[str] = []
            for name in CORE_ABLATIONS:
                effect = core_ablation_effects.get(name)
                if effect is None:
                    mechanism_failures.append(f"missing_core_ablation:{name}")
                elif not np.isfinite(effect) or effect < 0.01:
                    mechanism_failures.append(f"core_ablation_below_1pct:{name}")
            if not state_head_pass:
                mechanism_failures.append("state_head_diagnostic_failed")
            if not expert_specialization_pass:
                mechanism_failures.append("expert_specialization_failed")
            if not gate_utilization_pass:
                mechanism_failures.append("gate_utilization_failed")
            failed.extend(mechanism_failures)
            status = "forecast_only" if mechanism_failures else "mechanism_supported"

    return {
        "status": status,
        "failed_reasons": failed,
        "thresholds": {
            "deep_baseline_relative_qcrps_improvement": 0.02,
            "minimum_seed_wins": 2,
            "core_ablation_relative_effect": 0.01,
            "bootstrap_confidence": 0.95,
        },
        "observed": {
            "seed_improvements": seeds,
            "mean_improvement": float(mean_improvement),
            "seed_wins": int(wins),
            "bootstrap_ci": [float(bootstrap_ci_low), float(bootstrap_ci_high)],
        },
    }


def classify_specialist_weight_candidate(
    *,
    seed_improvements: Mapping[int, float],
    mean_improvement: float,
    bootstrap_ci_low: float,
    ramp_mean_improvement: float,
    diagnostics_pass: bool,
    expected_seeds: Sequence[int] = (13, 29, 47),
) -> dict[str, Any]:
    """Apply the preregistered pilot and multiseed gates for one weight."""

    seeds = {int(seed): float(value) for seed, value in seed_improvements.items()}
    expected = tuple(int(seed) for seed in expected_seeds)
    missing = [seed for seed in expected if seed not in seeds]
    wins = sum(seeds.get(seed, -np.inf) > 0.0 for seed in expected)
    failed: list[str] = []

    if 13 not in seeds:
        failed.append("missing_selection_seed:13")
        status = "inconclusive"
    else:
        if seeds[13] < 0.005:
            failed.append("seed13_improvement_below_0_5pct")
        if ramp_mean_improvement < 0.0:
            failed.append("ramp_qcrps_worsened")
        if not diagnostics_pass:
            failed.append("expert_or_gate_diagnostic_failed")
        if failed:
            status = "reject"
        elif missing:
            status = "advance_to_multiseed"
        else:
            if mean_improvement < 0.005:
                failed.append("mean_improvement_below_0_5pct")
            if wins < 2:
                failed.append("fewer_than_2_of_3_seed_wins")
            if not np.isfinite(bootstrap_ci_low) or bootstrap_ci_low <= 0.0:
                failed.append("paired_bootstrap_ci_crosses_zero")
            status = "reject" if failed else "accept"

    return {
        "status": status,
        "failed_reasons": failed,
        "thresholds": {
            "seed13_relative_qcrps_improvement": 0.005,
            "multiseed_mean_relative_qcrps_improvement": 0.005,
            "minimum_seed_wins": 2,
            "ramp_mean_improvement": 0.0,
            "bootstrap_confidence": 0.95,
        },
        "observed": {
            "seed_improvements": seeds,
            "mean_improvement": float(mean_improvement),
            "seed_wins": int(wins),
            "bootstrap_ci_low": float(bootstrap_ci_low),
            "ramp_mean_improvement": float(ramp_mean_improvement),
            "diagnostics_pass": bool(diagnostics_pass),
            "missing_seeds": missing,
        },
    }


def require_validation_only_metadata(metadata: Mapping[str, Any]) -> None:
    """Fail closed unless an archive explicitly declares the validation boundary."""

    if metadata.get("split") != "validation":
        raise ValueError("validation analysis requires metadata split='validation'")
    if metadata.get("selection_boundary") != "validation_only":
        raise ValueError(
            "validation analysis requires selection_boundary='validation_only'"
        )


def validate_run_metadata_contract(
    run: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    """Validate one manifest row against its prediction sidecar."""

    require_validation_only_metadata(metadata)
    experiment = str(run.get("experiment", ""))
    role = str(run.get("role", ""))
    if role not in {
        "full",
        "naive_baseline",
        "deep_baseline",
        "core_ablation",
        "secondary_ablation",
    }:
        raise ValueError(f"unknown validation run role: {role!r}")
    if isinstance(run.get("seed"), bool):
        raise ValueError("run seed must be an integer")
    try:
        seed = int(run["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("run seed must be an integer") from error
    if metadata.get("seed") != seed:
        raise ValueError(f"manifest and prediction seed differ for {experiment}")

    racepv_role = role in {"full", "core_ablation", "secondary_ablation"}
    if racepv_role:
        if not isinstance(metadata.get("run_name"), str):
            raise ValueError(f"{experiment} lacks a RaCE-PV run identity")
        weights = metadata.get("loss_weights")
        if not isinstance(weights, Mapping):
            raise ValueError(f"{experiment} lacks loss weight metadata")
        try:
            state_weight = float(weights["state"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{experiment} lacks a numeric state loss weight") from error
        if state_weight != 1.0:
            raise ValueError(f"{experiment} must keep state cross-entropy weight 1.0")
        if experiment in SPECIALIST_DIAGNOSTIC_WEIGHTS:
            expected_specialist_weight = SPECIALIST_DIAGNOSTIC_WEIGHTS[experiment]
            if (
                metadata.get("diagnostic_only") is not True
                or metadata.get("diagnostic_ablation") != experiment
                or abs(
                    float(weights.get("specialist", -1.0))
                    - expected_specialist_weight
                )
                > 1e-12
            ):
                raise ValueError(
                    f"{experiment} requires diagnostic_only=true, matching "
                    f"diagnostic metadata, and specialist weight "
                    f"{expected_specialist_weight}"
                )
        elif (
            metadata.get("diagnostic_only") is not False
            or metadata.get("diagnostic_ablation") is not None
            or abs(float(weights.get("specialist", -1.0)) - 0.3) > 1e-12
        ):
            raise ValueError(
                f"{experiment} must use the ordinary non-diagnostic specialist weight 0.3"
            )
        if role == "full" and (
            metadata.get("uses_official_mamba") is not True
            or metadata.get("actual_backbone") != "mamba_ssm"
        ):
            raise ValueError("full RaCE-PV run did not use the official Mamba backbone")
    else:
        if metadata.get("baseline") != experiment:
            raise ValueError(f"baseline metadata identity differs for {experiment}")


def forecast_diagnostics(
    arrays: Mapping[str, np.ndarray], levels: Sequence[float]
) -> tuple[dict[str, Any], np.ndarray]:
    """Return reported forecast metrics and reusable daylight cell losses."""

    targets = np.asarray(arrays["targets"], dtype=float)
    quantiles = np.asarray(arrays["quantiles"], dtype=float)
    labels = np.asarray(arrays["state_labels"])
    daylight = np.asarray(arrays["daylight_mask"], dtype=bool)
    losses = qcrps_cells(targets, quantiles, levels, mask=daylight)

    def selected_mean(selection: np.ndarray) -> float:
        values = losses[selection & np.isfinite(losses)]
        return float(np.mean(values)) if values.size else float("nan")

    levels_array = np.asarray(tuple(levels), dtype=float)
    median_positions = np.flatnonzero(np.isclose(levels_array, 0.5, atol=1e-12))
    if median_positions.size != 1:
        raise ValueError("validation analysis requires exactly one median quantile")
    median = quantiles[..., int(median_positions[0])]
    point_errors = median[daylight] - targets[daylight]
    point = {
        "daylight_nmae": float(np.mean(np.abs(point_errors))),
        "daylight_nrmse": float(np.sqrt(np.mean(np.square(point_errors)))),
    }
    return (
        {
            "daylight_count": int(np.sum(daylight)),
            "qcrps": {
                "overall": selected_mean(daylight),
                "stable": selected_mean(daylight & (labels == 0)),
                "ramp": selected_mean(daylight & np.isin(labels, (1, 2))),
            },
            "point": point,
        },
        losses,
    )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def require_aligned_predictions(
    first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]
) -> None:
    for name in (
        "targets",
        "state_labels",
        "daylight_mask",
        "timestamps",
        "day_ids",
        "site_ids",
    ):
        if not np.array_equal(first[name], second[name]):
            raise ValueError(f"paired validation predictions differ at {name}")


def analyze_manifest(manifest_path: Path, config_path: Path) -> dict[str, Any]:
    """Analyze one reviewed run manifest without opening calibration or test data."""

    from racepv.cli.evaluate import (
        _load_metadata,
        _load_npz,
        _validate_prediction_archive,
    )

    manifest_path = Path(manifest_path)
    config_path = Path(config_path)
    manifest = _load_json_object(manifest_path, "validation run manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("validation run manifest must use schema_version=1")
    if manifest.get("purpose") != "racepv_day1_validation_screening":
        raise ValueError("manifest purpose is not the one-day RaCE-PV screening")
    if manifest.get("selection_boundary") != "validation_only":
        raise ValueError("manifest must declare selection_boundary='validation_only'")
    if manifest.get("bootstrap_replicates") != 500:
        raise ValueError("the one-day screen fixes bootstrap_replicates at 500")
    try:
        selection_seed = int(manifest["selection_seed"])
        expected_seeds = tuple(int(value) for value in manifest["expected_seeds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("manifest seeds must be explicit integers") from error
    if selection_seed != 13 or expected_seeds != (13, 29, 47):
        raise ValueError("the one-day screen fixes selection seed 13 and seeds 13/29/47")
    raw_runs = manifest.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("manifest runs must be a non-empty list")

    config = load_config(config_path)
    levels = tuple(float(value) for value in config.quantiles)
    loaded: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    artifact_rows: list[dict[str, Any]] = []
    for raw in raw_runs:
        if not isinstance(raw, dict):
            raise TypeError("every manifest run must be a JSON object")
        experiment = str(raw.get("experiment", ""))
        if not experiment:
            raise ValueError("manifest run experiment must be non-empty")
        if isinstance(raw.get("seed"), bool):
            raise ValueError("manifest run seed must be an integer")
        try:
            seed = int(raw["seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("manifest run seed must be an integer") from error
        identity = (experiment, seed)
        if identity in seen:
            raise ValueError(f"duplicate manifest run: {experiment}/seed-{seed}")
        seen.add(identity)
        path_value = raw.get("predictions")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"{experiment} has no prediction path")
        path = Path(path_value)
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        metadata_path = path.with_suffix(".json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"prediction archive or sidecar is missing: {path}")
        arrays = _load_npz(path)
        metadata = _load_metadata(path)
        validate_run_metadata_contract(raw, metadata)
        _validate_prediction_archive(
            arrays,
            metadata,
            split="validation",
            levels=levels,
            horizon=config.data.horizon,
            power_upper_bound=config.data.power_upper_bound,
        )
        forecast, losses = forecast_diagnostics(arrays, levels)
        state_report = state_head_diagnostics(
            arrays["state_labels"], arrays["state_probs"], arrays["daylight_mask"]
        )
        expert_report = None
        if "expert_quantiles" in arrays:
            expert_report = expert_gate_diagnostics(
                experts=arrays["expert_quantiles"],
                gates=arrays["gate_weights"],
                targets=arrays["targets"],
                labels=arrays["state_labels"],
                daylight=arrays["daylight_mask"],
                levels=levels,
                active_experts=metadata.get("active_experts", STATE_NAMES),
            )
        run_report: dict[str, Any] = {
            "experiment": experiment,
            "role": str(raw["role"]),
            "seed": seed,
            "forecast": forecast,
            "state_head": state_report,
        }
        if expert_report is not None:
            run_report["experts"] = {
                "performance_by_true_state": expert_report[
                    "performance_by_true_state"
                ],
                "prediction_diversity": expert_report[
                    "expert_prediction_diversity"
                ],
                "specialization_pass": expert_report[
                    "expert_specialization_pass"
                ],
                "failure_reasons": expert_report[
                    "expert_specialization_failure_reasons"
                ],
            }
            run_report["fusion_gate"] = {
                **expert_report["overall_utilization"],
                "passed": expert_report["gate_utilization_pass"],
                "interpretation_boundary": expert_report[
                    "interpretation_boundary"
                ],
            }
        artifact_rows.append(
            {
                "experiment": experiment,
                "seed": seed,
                "path": str(path),
                "prediction_sha256": sha256_file(path),
                "metadata_sha256": sha256_file(metadata_path),
            }
        )
        loaded.append(
            {
                "entry": raw,
                "arrays": arrays,
                "metadata": metadata,
                "losses": losses,
                "report": run_report,
            }
        )

    def select(*, role: str, seed: int | None = None, experiment: str | None = None):
        return [
            item
            for item in loaded
            if item["entry"]["role"] == role
            and (seed is None or int(item["entry"]["seed"]) == seed)
            and (experiment is None or item["entry"]["experiment"] == experiment)
        ]

    selection_full = select(role="full", seed=selection_seed)
    if len(selection_full) != 1:
        raise ValueError("manifest requires exactly one full run for selection seed 13")
    full_at_selection = selection_full[0]
    deep_candidates = select(role="deep_baseline", seed=selection_seed)
    if not deep_candidates:
        raise ValueError("manifest requires a deep baseline at selection seed 13")
    strongest = min(
        deep_candidates,
        key=lambda item: (
            item["report"]["forecast"]["qcrps"]["overall"],
            item["entry"]["experiment"],
        ),
    )
    strongest_name = str(strongest["entry"]["experiment"])

    full_by_seed = {
        int(item["entry"]["seed"]): item for item in select(role="full")
    }
    deep_by_seed = {
        int(item["entry"]["seed"]): item
        for item in select(role="deep_baseline", experiment=strongest_name)
    }
    for item in loaded:
        seed = int(item["entry"]["seed"])
        full = full_by_seed.get(seed)
        if full is not None and item is not full:
            require_aligned_predictions(full["arrays"], item["arrays"])
    per_seed_comparison: dict[str, Any] = {}
    seed_improvements: dict[int, float] = {}
    bootstrap_distributions: list[np.ndarray] = []
    bootstrap_estimates: list[float] = []
    for seed in sorted(set(full_by_seed).intersection(deep_by_seed)):
        full = full_by_seed[seed]
        deep = deep_by_seed[seed]
        require_aligned_predictions(full["arrays"], deep["arrays"])
        full_score = float(full["report"]["forecast"]["qcrps"]["overall"])
        deep_score = float(deep["report"]["forecast"]["qcrps"]["overall"])
        improvement = 1.0 - full_score / deep_score
        seed_improvements[seed] = improvement
        bootstrap = paired_site_day_bootstrap(
            full["losses"],
            deep["losses"],
            full["arrays"]["site_ids"],
            full["arrays"]["day_ids"],
            full["arrays"]["daylight_mask"],
            n_bootstrap=500,
            seed=10_000 + seed,
        )
        bootstrap_distributions.append(np.asarray(bootstrap["distribution"], dtype=float))
        bootstrap_estimates.append(float(bootstrap["estimate_reference_minus_model"]))
        per_seed_comparison[str(seed)] = {
            "full_qcrps": full_score,
            "deep_baseline_qcrps": deep_score,
            "relative_improvement": improvement,
            "bootstrap": bootstrap,
        }
    if bootstrap_distributions:
        pooled_distribution = np.mean(np.stack(bootstrap_distributions), axis=0)
        pooled_low, pooled_high = np.quantile(pooled_distribution, (0.025, 0.975))
        pooled_bootstrap: dict[str, Any] = {
            "method": "equal-seed mean of paired site-day bootstrap replicates",
            "n_bootstrap": 500,
            "seed_count": len(bootstrap_distributions),
            "estimate_reference_minus_model": float(np.mean(bootstrap_estimates)),
            "ci_low": float(pooled_low),
            "ci_high": float(pooled_high),
            "distribution": pooled_distribution,
        }
    else:  # pragma: no cover - selection seed guarantees one comparison
        pooled_bootstrap = {
            "n_bootstrap": 500,
            "seed_count": 0,
            "estimate_reference_minus_model": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "distribution": np.full(500, np.nan),
        }

    naive_names = {"smart_persistence", "clear_sky_persistence"}
    naive_at_selection = {
        str(item["entry"]["experiment"]): item
        for item in select(role="naive_baseline", seed=selection_seed)
    }
    naive_complete = naive_names.issubset(naive_at_selection)
    missing_naive = sorted(naive_names.difference(naive_at_selection))
    full_selection_score = float(
        full_at_selection["report"]["forecast"]["qcrps"]["overall"]
    )
    naive_pass = naive_complete and all(
        full_selection_score
        < float(naive_at_selection[name]["report"]["forecast"]["qcrps"]["overall"])
        for name in naive_names
    )

    core_effects: dict[str, float] = {}
    for ablation_name in CORE_ABLATIONS:
        effects = []
        for ablation in select(role="core_ablation", experiment=ablation_name):
            seed = int(ablation["entry"]["seed"])
            full = full_by_seed.get(seed)
            if full is None:
                continue
            ablation_score = float(
                ablation["report"]["forecast"]["qcrps"]["overall"]
            )
            full_score = float(full["report"]["forecast"]["qcrps"]["overall"])
            effects.append(1.0 - full_score / ablation_score)
        if effects:
            core_effects[ablation_name] = float(np.mean(effects))

    specialist_weight_sweep: dict[str, Any] = {}
    for diagnostic_name, specialist_weight in SPECIALIST_WEIGHT_SWEEP.items():
        candidates = select(
            role="secondary_ablation",
            experiment=diagnostic_name,
        )
        if not candidates:
            continue
        per_seed: dict[str, Any] = {}
        seed_effects: dict[int, float] = {}
        ramp_effects: list[float] = []
        distributions: list[np.ndarray] = []
        estimates: list[float] = []
        diagnostics_pass = True
        for candidate in candidates:
            seed = int(candidate["entry"]["seed"])
            full = full_by_seed.get(seed)
            if full is None:
                continue
            require_aligned_predictions(full["arrays"], candidate["arrays"])
            candidate_score = float(
                candidate["report"]["forecast"]["qcrps"]["overall"]
            )
            full_score = float(full["report"]["forecast"]["qcrps"]["overall"])
            relative_improvement = 1.0 - candidate_score / full_score
            candidate_ramp = float(
                candidate["report"]["forecast"]["qcrps"]["ramp"]
            )
            full_ramp = float(full["report"]["forecast"]["qcrps"]["ramp"])
            ramp_improvement = 1.0 - candidate_ramp / full_ramp
            bootstrap = paired_site_day_bootstrap(
                candidate["losses"],
                full["losses"],
                full["arrays"]["site_ids"],
                full["arrays"]["day_ids"],
                full["arrays"]["daylight_mask"],
                n_bootstrap=500,
                seed=20_000 + seed + int(round(specialist_weight * 100)),
            )
            seed_effects[seed] = relative_improvement
            ramp_effects.append(ramp_improvement)
            distributions.append(np.asarray(bootstrap["distribution"], dtype=float))
            estimates.append(float(bootstrap["estimate_reference_minus_model"]))
            candidate_diagnostics_pass = bool(
                candidate["report"].get("experts", {}).get("specialization_pass")
                is True
                and candidate["report"].get("fusion_gate", {}).get("passed") is True
            )
            diagnostics_pass = diagnostics_pass and candidate_diagnostics_pass
            per_seed[str(seed)] = {
                "candidate_qcrps": candidate_score,
                "registered_full_qcrps": full_score,
                "relative_improvement": relative_improvement,
                "candidate_ramp_qcrps": candidate_ramp,
                "registered_full_ramp_qcrps": full_ramp,
                "ramp_relative_improvement": ramp_improvement,
                "diagnostics_pass": candidate_diagnostics_pass,
                "bootstrap": bootstrap,
            }
        if not seed_effects:
            continue
        pooled_distribution = np.mean(np.stack(distributions), axis=0)
        pooled_low, pooled_high = np.quantile(pooled_distribution, (0.025, 0.975))
        mean_improvement = float(np.mean(tuple(seed_effects.values())))
        ramp_mean_improvement = float(np.mean(ramp_effects))
        decision = classify_specialist_weight_candidate(
            seed_improvements=seed_effects,
            mean_improvement=mean_improvement,
            bootstrap_ci_low=float(pooled_low),
            ramp_mean_improvement=ramp_mean_improvement,
            diagnostics_pass=diagnostics_pass,
            expected_seeds=expected_seeds,
        )
        specialist_weight_sweep[diagnostic_name] = {
            "specialist_weight": float(specialist_weight),
            "per_seed": per_seed,
            "equal_seed_mean_relative_improvement": mean_improvement,
            "ramp_mean_relative_improvement": ramp_mean_improvement,
            "pooled_bootstrap": {
                "method": "equal-seed mean of paired site-day bootstrap replicates",
                "n_bootstrap": 500,
                "seed_count": len(distributions),
                "estimate_registered_full_minus_candidate": float(
                    np.mean(estimates)
                ),
                "ci_low": float(pooled_low),
                "ci_high": float(pooled_high),
                "distribution": pooled_distribution,
            },
            "decision": decision,
        }

    full_state_pass = bool(
        full_by_seed
        and all(item["report"]["state_head"]["passed"] for item in full_by_seed.values())
    )
    full_expert_pass = bool(
        full_by_seed
        and all(
            item["report"].get("experts", {}).get("specialization_pass") is True
            for item in full_by_seed.values()
        )
    )
    full_gate_pass = bool(
        full_by_seed
        and all(
            item["report"].get("fusion_gate", {}).get("passed") is True
            for item in full_by_seed.values()
        )
    )
    mean_improvement = float(np.mean(list(seed_improvements.values())))
    selection_improvement = seed_improvements[selection_seed]
    decision = classify_screening(
        seed_improvements=seed_improvements,
        mean_improvement=mean_improvement,
        bootstrap_ci_low=float(pooled_bootstrap["ci_low"]),
        bootstrap_ci_high=float(pooled_bootstrap["ci_high"]),
        naive_baselines_pass=naive_pass,
        worse_than_deep_by_more_than_two_percent=selection_improvement < -0.02,
        core_ablation_effects=core_effects,
        state_head_pass=full_state_pass,
        expert_specialization_pass=full_expert_pass,
        gate_utilization_pass=full_gate_pass,
        expected_seeds=expected_seeds,
        required_inputs_complete=naive_complete,
        incomplete_reasons=tuple(
            f"missing_naive_baseline:{name}" for name in missing_naive
        ),
    )
    report = {
        "schema_version": 1,
        "purpose": "racepv_day1_validation_screening_analysis",
        "selection_boundary": "validation_only",
        "selection_seed": selection_seed,
        "expected_seeds": list(expected_seeds),
        "strongest_deep_baseline": {
            "experiment": strongest_name,
            "selection_seed": selection_seed,
            "selection_daylight_qcrps": strongest["report"]["forecast"]["qcrps"][
                "overall"
            ],
            "selection_rule": "minimum seed-13 overall daylight qCRPS; lexical exact tie-break",
        },
        "runs": [item["report"] for item in loaded],
        "comparisons": {
            "full_vs_strongest_deep": {
                "per_seed": per_seed_comparison,
                "equal_seed_mean_relative_improvement": mean_improvement,
                "pooled_bootstrap": pooled_bootstrap,
            },
            "core_ablation_relative_effects": core_effects,
            "specialist_weight_sweep": specialist_weight_sweep,
            "naive_baselines_complete": naive_complete,
            "beats_both_naive_baselines_at_seed_13": naive_pass,
        },
        "decision": decision,
        "artifacts": {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "predictions": artifact_rows,
        },
        "interpretation_boundary": (
            "validation-only screening evidence; calibration and test were not read"
        ),
    }
    return _json_ready(report)
