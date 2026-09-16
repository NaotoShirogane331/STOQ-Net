"""Metrics and sealed decisions for the prediction/state coupling ablation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from racepv.metrics import average_precision
from racepv.validation_screening import qcrps_cells

MODES = ("decoupled", "joint_head", "state_as_gate")
REGISTERED_SEEDS = (13, 29, 47, 71, 101)
EXPERT_NAMES = ("stable", "ramp_up", "ramp_down")
DECISIONS = {
    "decoupling_performance_supported",
    "decoupling_neutral_but_auditable",
    "state_coupling_preferred",
    "inconclusive",
}


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _binary_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    truth = np.asarray(actual, dtype=bool)
    guess = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(truth & guess))
    fp = int(np.sum(~truth & guess))
    fn = int(np.sum(truth & ~guess))
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f2 = _safe_ratio(5.0 * precision * recall, 4.0 * precision + recall)
    return {"precision": precision, "recall": recall, "f2": f2}


def _macro_f1(labels: np.ndarray, predicted: np.ndarray) -> float:
    scores: list[float] = []
    for state in range(3):
        actual = labels == state
        guess = predicted == state
        tp = int(np.sum(actual & guess))
        fp = int(np.sum(~actual & guess))
        fn = int(np.sum(actual & ~guess))
        scores.append(_safe_ratio(2.0 * tp, 2.0 * tp + fp + fn))
    return float(np.mean(scores))


def _macro_auprc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    scores: list[float] = []
    for state in range(3):
        actual = labels == state
        if bool(actual.any()) and bool((~actual).any()):
            scores.append(float(average_precision(actual, probabilities[:, state])))
    return float(np.mean(scores)) if scores else 0.0


def infer_origin_step_minutes(timestamps: np.ndarray, site_ids: np.ndarray) -> float:
    """Infer forecast-origin spacing from target timestamps for each site."""

    values = np.asarray(timestamps)
    sites = np.asarray(site_ids).reshape(-1)
    if values.ndim != 2 or values.shape[0] != sites.size or values.shape[1] == 0:
        raise ValueError("timestamps must be [origins, horizon] with one site id per origin")
    first_targets = values[:, 0].astype(np.int64)
    positive_differences: list[np.ndarray] = []
    for site in np.unique(sites):
        ordered = np.unique(first_targets[sites == site])
        differences = np.diff(ordered)
        if np.any(differences > 0):
            positive_differences.append(differences[differences > 0])
    if not positive_differences:
        raise ValueError("at least two forecast origins per site are required")
    differences_ns = np.concatenate(positive_differences).astype(np.int64)
    minute_ns = 60 * 1_000_000_000
    if np.any(np.mod(differences_ns, minute_ns) != 0):
        raise ValueError("forecast-origin differences must be positive whole minutes")
    step_minutes = float(np.gcd.reduce(differences_ns // minute_ns))
    if not np.isfinite(step_minutes) or step_minutes <= 0:
        raise ValueError("forecast-origin spacing must be finite and positive")
    return step_minutes


def paired_site_day_bootstrap(
    reference_losses: np.ndarray,
    candidate_losses: np.ndarray,
    group_ids: np.ndarray,
    *,
    repetitions: int = 500,
    seed: int = 20260827,
) -> dict[str, float | int]:
    """Return candidate improvement: reference loss minus candidate loss."""

    reference = np.asarray(reference_losses, dtype=float).reshape(-1)
    candidate = np.asarray(candidate_losses, dtype=float).reshape(-1)
    groups = np.asarray(group_ids).reshape(-1)
    if reference.shape != candidate.shape or reference.shape != groups.shape:
        raise ValueError("paired losses and group ids must have the same flattened shape")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    finite = np.isfinite(reference) & np.isfinite(candidate)
    if not bool(finite.any()):
        raise ValueError("paired bootstrap requires finite paired losses")
    reference = reference[finite]
    candidate = candidate[finite]
    groups = groups[finite]
    unique_groups, group_index = np.unique(groups, return_inverse=True)
    group_counts = np.bincount(group_index).astype(float)
    reference_means = np.bincount(group_index, weights=reference) / group_counts
    candidate_means = np.bincount(group_index, weights=candidate) / group_counts
    group_improvements = reference_means - candidate_means
    generator = np.random.default_rng(seed)
    selections = generator.integers(
        0,
        unique_groups.size,
        size=(repetitions, unique_groups.size),
    )
    distribution = np.mean(group_improvements[selections], axis=1)
    return {
        "mean_improvement": float(np.mean(group_improvements)),
        "ci_low": float(np.quantile(distribution, 0.025)),
        "ci_high": float(np.quantile(distribution, 0.975)),
        "groups": int(unique_groups.size),
        "repetitions": int(repetitions),
        "seed": int(seed),
    }


def analyze_prediction_archive(
    arrays: Mapping[str, np.ndarray],
    levels: Sequence[float],
    *,
    threshold: float = 0.25,
) -> dict[str, Any]:
    """Compute the fixed per-run forecast, state, expert, and invariant report."""

    required = {
        "targets",
        "quantiles",
        "state_labels",
        "state_probs",
        "state_logits",
        "daylight_mask",
        "expert_quantiles",
        "gate_weights",
        "fusion_logits",
    }
    missing = required.difference(arrays)
    if missing:
        raise KeyError(f"prediction archive is missing: {sorted(missing)}")
    targets = np.asarray(arrays["targets"], dtype=float)
    quantiles = np.asarray(arrays["quantiles"], dtype=float)
    labels = np.asarray(arrays["state_labels"])
    probabilities = np.asarray(arrays["state_probs"], dtype=float)
    daylight = np.asarray(arrays["daylight_mask"], dtype=bool)
    gates = np.asarray(arrays["gate_weights"], dtype=float)
    if quantiles.shape[:2] != targets.shape or probabilities.shape != (*targets.shape, 3):
        raise ValueError("prediction arrays have inconsistent batch/horizon shapes")
    level_values = np.asarray(tuple(levels), dtype=float)
    median_positions = np.flatnonzero(np.isclose(level_values, 0.5, atol=1e-12))
    if median_positions.size != 1 or quantiles.shape[-1] != level_values.size:
        raise ValueError("levels must match quantiles and contain one median")
    median = quantiles[..., int(median_positions[0])]
    valid = daylight & np.isfinite(targets) & np.isfinite(median)
    stable = valid & (labels == 0)
    ramp = valid & np.isin(labels, (1, 2))
    if not bool(valid.any()) or not bool(ramp.any()):
        raise ValueError("analysis requires finite daylight and ramp cells")

    losses = qcrps_cells(targets, quantiles, level_values, mask=valid)
    error = median - targets

    def mean_loss(mask: np.ndarray) -> float:
        selected = losses[mask & np.isfinite(losses)]
        return float(np.mean(selected)) if selected.size else float("nan")

    def point(mask: np.ndarray) -> tuple[float, float]:
        selected = error[mask]
        return float(np.mean(np.abs(selected))), float(np.sqrt(np.mean(np.square(selected))))

    overall_nmae, overall_nrmse = point(valid)
    ramp_nmae, ramp_nrmse = point(ramp)
    valid_states = valid & np.isin(labels, (0, 1, 2))
    state_labels = labels[valid_states].astype(int)
    state_probabilities = probabilities[valid_states]
    ramp_scores = state_probabilities[:, 1:].sum(axis=-1)
    binary = _binary_metrics(state_labels != 0, ramp_scores >= threshold)
    binary_auprc = float(average_precision(state_labels != 0, ramp_scores))
    predicted_states = np.argmax(state_probabilities, axis=-1)

    daylight_gates = gates[valid]
    mean_soft = np.mean(daylight_gates, axis=0)
    hard = np.argmax(daylight_gates, axis=-1)
    hard_share = np.asarray([np.mean(hard == index) for index in range(3)])
    quantile_crossings = int(np.count_nonzero(np.diff(quantiles, axis=-1) < -1e-7))
    night_nonzero = int(np.count_nonzero(np.abs(quantiles[~daylight]) > 1e-7))
    finite_arrays = [np.asarray(arrays[name]) for name in required]
    nonfinite_count = int(
        sum(np.size(values) - np.count_nonzero(np.isfinite(values)) for values in finite_arrays)
    )
    state_logits = np.asarray(arrays["state_logits"], dtype=float)
    state_expected = np.exp(state_logits - np.max(state_logits, axis=-1, keepdims=True))
    state_expected /= np.sum(state_expected, axis=-1, keepdims=True)
    fusion_logits = np.asarray(arrays["fusion_logits"], dtype=float)
    gate_expected = np.exp(
        fusion_logits - np.max(fusion_logits, axis=-1, keepdims=True)
    )
    gate_expected /= np.sum(gate_expected, axis=-1, keepdims=True)
    state_probability_consistent = bool(
        np.allclose(probabilities, state_expected, rtol=1e-5, atol=1e-6)
    )
    gate_probability_consistent = bool(
        np.allclose(gates, gate_expected, rtol=1e-5, atol=1e-6)
    )
    invariants_pass = bool(
        quantile_crossings == 0
        and night_nonzero == 0
        and nonfinite_count == 0
        and state_probability_consistent
        and gate_probability_consistent
    )
    return {
        "qcrps": {
            "overall": mean_loss(valid),
            "stable": mean_loss(stable),
            "ramp": mean_loss(ramp),
        },
        "point": {
            "overall_nmae": overall_nmae,
            "overall_nrmse": overall_nrmse,
            "ramp_nmae": ramp_nmae,
            "ramp_nrmse": ramp_nrmse,
        },
        "classification": {
            "binary_ramp_precision": binary["precision"],
            "binary_ramp_recall": binary["recall"],
            "binary_ramp_f2": binary["f2"],
            "binary_ramp_auprc": binary_auprc,
            "three_state_macro_auprc": _macro_auprc(
                state_labels, state_probabilities
            ),
            "three_state_macro_f1": _macro_f1(state_labels, predicted_states),
            "threshold": float(threshold),
        },
        "expert_utilization": {
            "mean_soft_weight": {
                name: float(mean_soft[index])
                for index, name in enumerate(EXPERT_NAMES)
            },
            "hard_selection_share": {
                name: float(hard_share[index])
                for index, name in enumerate(EXPERT_NAMES)
            },
            "minimum_active_soft_weight": float(np.min(mean_soft)),
            "minimum_active_hard_share": float(np.min(hard_share)),
        },
        "invariants": {
            "quantile_crossings": quantile_crossings,
            "night_nonzero_count": night_nonzero,
            "nonfinite_count": nonfinite_count,
            "state_probability_consistent": state_probability_consistent,
            "gate_probability_consistent": gate_probability_consistent,
            "passed": invariants_pass,
        },
        "invariants_pass": invariants_pass,
        "outputs_finite": nonfinite_count == 0,
    }


def stage1_continuation_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    hashes_match: bool,
) -> dict[str, Any]:
    """Apply the seed-13 cost-control gate without making a scientific decision."""

    by_mode = {
        str(row.get("mode")): row for row in rows if int(row.get("seed", -1)) == 13
    }
    failures: list[str] = []
    if set(by_mode) != set(MODES):
        failures.append("incomplete_seed13_matrix")
    if not hashes_match:
        failures.append("hash_mismatch")
    for mode in MODES:
        row = by_mode.get(mode)
        if row is None:
            continue
        numeric = np.asarray(
            [
                row.get("qcrps", float("nan")),
                row.get("minimum_active_soft_weight", float("nan")),
                row.get("minimum_active_hard_share", float("nan")),
            ],
            dtype=float,
        )
        if not bool(np.all(np.isfinite(numeric))) or row.get("invariants_pass") is not True:
            failures.append(f"nonfinite_or_invariant_failure:{mode}")
        if numeric[1] < 0.01 or numeric[2] < 0.01:
            failures.append(f"expert_collapse:{mode}")
    if set(by_mode) == set(MODES):
        decoupled = float(by_mode["decoupled"]["qcrps"])
        best_coupled = min(
            float(by_mode["joint_head"]["qcrps"]),
            float(by_mode["state_as_gate"]["qcrps"]),
        )
        if decoupled / best_coupled - 1.0 > 0.01 + 1e-12:
            failures.append("decoupled_qcrps_more_than_1pct_worse")
    return {
        "continue_multiseed": not failures,
        "failure_reasons": failures,
        "selection_boundary": "validation_only",
        "purpose": "cost_control_gate",
    }


def classify_prediction_state_ablation(
    rows: Sequence[Mapping[str, Any]],
    pairwise: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the fixed five-seed four-way decision rule."""

    by_mode_seed = {
        (str(row.get("mode")), int(row.get("seed", -1))): row for row in rows
    }
    complete = all((mode, seed) in by_mode_seed for mode in MODES for seed in REGISTERED_SEEDS)
    invariants_pass = complete and all(
        by_mode_seed[(mode, seed)].get("invariants_pass") is True
        for mode in MODES
        for seed in REGISTERED_SEEDS
    )
    if not complete or not invariants_pass:
        return {
            "decision": "inconclusive",
            "preferred_mode": None,
            "failure_reasons": [
                "incomplete_matrix" if not complete else "invariant_failure"
            ],
        }
    means = {
        mode: float(
            np.mean(
                [
                    float(by_mode_seed[(mode, seed)]["qcrps"])
                    for seed in REGISTERED_SEEDS
                ]
            )
        )
        for mode in MODES
    }
    ramp_means = {
        mode: float(
            np.mean(
                [
                    float(by_mode_seed[(mode, seed)]["ramp_nrmse"])
                    for seed in REGISTERED_SEEDS
                ]
            )
        )
        for mode in MODES
    }

    def wins(candidate: str, reference: str) -> int:
        return sum(
            float(by_mode_seed[(candidate, seed)]["qcrps"])
            < float(by_mode_seed[(reference, seed)]["qcrps"])
            for seed in REGISTERED_SEEDS
        )

    def ci_low(candidate: str, reference: str) -> float:
        report = pairwise.get((candidate, reference), {})
        return float(report.get("ci_low", float("-inf")))

    decoupling_supported = all(
        means["decoupled"] <= 0.99 * means[alternative]
        and wins("decoupled", alternative) >= 4
        and ci_low("decoupled", alternative) > 0.0
        and ramp_means["decoupled"] <= 1.01 * ramp_means[alternative]
        for alternative in ("joint_head", "state_as_gate")
    )
    if decoupling_supported:
        decision = "decoupling_performance_supported"
        preferred_mode = "decoupled"
    else:
        preferred_mode = next(
            (
                alternative
                for alternative in ("joint_head", "state_as_gate")
                if means[alternative] <= 0.99 * means["decoupled"]
                and wins(alternative, "decoupled") >= 4
                and ci_low(alternative, "decoupled") > 0.0
                and ramp_means[alternative] <= 1.01 * ramp_means["decoupled"]
            ),
            None,
        )
        if preferred_mode is not None:
            decision = "state_coupling_preferred"
        else:
            all_within_half_percent = max(means.values()) / min(means.values()) - 1.0 <= 0.005
            coupled_not_significant = all(
                ci_low(alternative, "decoupled") <= 0.0
                for alternative in ("joint_head", "state_as_gate")
            )
            decision = (
                "decoupling_neutral_but_auditable"
                if all_within_half_percent and coupled_not_significant
                else "inconclusive"
            )
    return {
        "decision": decision,
        "preferred_mode": preferred_mode,
        "mean_qcrps": means,
        "mean_ramp_nrmse": ramp_means,
        "seed_wins": {
            f"{candidate}_vs_{reference}": wins(candidate, reference)
            for candidate in MODES
            for reference in MODES
            if candidate != reference
        },
        "validation_evidence_notice": (
            "The paired bootstrap is model-selection evidence on validation data, "
            "not final unseen-test significance."
        ),
    }
