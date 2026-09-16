"""In-memory dual-protocol screening for realism-v2 candidates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from racepv.metrics import (
    binary_ramp_metrics,
    normalized_rmse,
    prediction_interval_coverage,
    quantile_crps,
)

from .constants import BASELINES, QUANTILES, RACE_PV, SEEDS
from .realism import RealismCandidate, canonical_config_sha256
from .scenario_manifest import cohort_sha256
from .synthesis import SynthesizedPrediction, synthesize_predictions

VALIDATION_PROTOCOLS = ("dkasc-yulara", "nrel-transfer")
EXPECTED_VALIDATION_SEEDS = (13, 29, 47, 71, 101)
VALIDATION_PRECHECK_RANGES = {
    "daylight_nrmse_improvement_pct": (4.0, 8.0),
    "mean_quantile_crps_improvement_pct": (5.0, 10.0),
    "ramp_nrmse_improvement_pct": (8.0, 14.0),
    "ramp_qcrps_improvement_pct": (9.0, 16.0),
    "ramp_aucpr_absolute_gain": (0.04, 0.08),
    "stable_picp90": (0.87, 0.94),
    "ramp_picp90": (0.85, 0.93),
    "neural_seed_relative_sd": (0.012, 0.045),
    "correct_expert_mean_gate": (0.52, 0.73),
    "normalized_gate_entropy": (0.57, 0.85),
    "maximum_expert_global_share": (0.0, 0.85),
}

_DEEP_BASELINES = tuple(profile for profile in BASELINES if profile.family == "neural")
_MEDIAN_INDEX = tuple(QUANTILES).index(0.5)


def _strict_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class CandidateGateResult:
    """Validation-only outcome and provenance for one complete candidate."""

    candidate_id: str
    config: dict[str, Any]
    config_sha256: str
    scenario_seed: int
    passed: bool
    metrics: dict[str, float]
    reasons: tuple[str, ...]
    selected_validation_reference: dict[str, Any]
    cohort_hashes: dict[str, str]
    protocols: tuple[str, ...]
    seeds: tuple[int, ...]
    distance_to_range_midpoints: float

    @property
    def status(self) -> str:
        return "accepted" if self.passed else "rejected"

    def to_payload(self, *, deterministic_rank: int) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "cohort_hashes": dict(self.cohort_hashes),
            "config": dict(self.config),
            "config_sha256": self.config_sha256,
            "deterministic_rank": deterministic_rank,
            "distance_to_range_midpoints": (
                self.distance_to_range_midpoints
                if np.isfinite(self.distance_to_range_midpoints)
                else None
            ),
            "metrics": {
                name: value if np.isfinite(value) else None
                for name, value in self.metrics.items()
            },
            "protocols": list(self.protocols),
            "reasons": list(self.reasons),
            "scenario_seed": self.scenario_seed,
            "seeds": list(self.seeds),
            "selected_validation_reference": _strict_json_value(
                self.selected_validation_reference
            ),
            "status": self.status,
        }


@dataclass(frozen=True)
class CandidateScreeningReport:
    """Deterministically ranked candidate outcomes without prediction arrays."""

    candidates: tuple[CandidateGateResult, ...]
    cohort_hashes: dict[str, str]
    protocol_cohort_sizes: dict[str, dict[str, Any]]

    @property
    def accepted(self) -> tuple[CandidateGateResult, ...]:
        return tuple(result for result in self.candidates if result.passed)

    @property
    def rejected(self) -> tuple[CandidateGateResult, ...]:
        return tuple(result for result in self.candidates if not result.passed)

    @property
    def selected_candidate_id(self) -> str | None:
        return self.accepted[0].candidate_id if self.accepted else None

    def to_payload(self) -> dict[str, Any]:
        from .constants import EVIDENCE_METADATA
        from .scenario_artifacts import DISCLOSURE

        return {
            **dict(EVIDENCE_METADATA),
            "accepted_count": len(self.accepted),
            "candidate_count": len(self.candidates),
            "candidates": [
                result.to_payload(deterministic_rank=rank)
                for rank, result in enumerate(self.candidates, start=1)
            ],
            "cohort_hashes": dict(self.cohort_hashes),
            "disclosure": DISCLOSURE,
            "formal_57_site_result": False,
            "precheck_ranges": {
                name: {"minimum": bounds[0], "maximum": bounds[1]}
                for name, bounds in VALIDATION_PRECHECK_RANGES.items()
            },
            "protocol_cohort_sizes": dict(self.protocol_cohort_sizes),
            "protocols": list(VALIDATION_PROTOCOLS),
            "registered_replicate_seeds": list(EXPECTED_VALIDATION_SEEDS),
            "rejected_count": len(self.rejected),
            "schema": "realism_candidate_screening_v1",
            "screening_scope": "validation_precheck_only",
            "selected_candidate_id": self.selected_candidate_id,
            "selection_order": "accepted:(distance_to_range_midpoints,candidate_id);rejected:candidate_id",
            "status": "passed" if self.accepted else "failed",
        }


@dataclass(frozen=True)
class _BaselineSeedMetrics:
    """Scalar losses retained from one validation baseline synthesis."""

    nrmse: float
    qcrps: float
    ramp_nrmse: float
    ramp_qcrps: float
    ramp_auprc: float


@dataclass(frozen=True, order=True)
class _BaselineEffectiveKey:
    """Every candidate input that changes a stochastic baseline array."""

    scenario_seed: int
    stochastic_residual_weight: float
    spread_scale: float
    ramp_spread_multiplier: float
    router_noise_scale: float


@dataclass(frozen=True)
class _ValidationBaselineCache:
    """Call-local scalar baseline summaries; never prediction tensors."""

    by_model: dict[str, dict[str, dict[int, _BaselineSeedMetrics]]]
    candidate_scores: dict[str, float]
    protocol_scores: dict[str, dict[str, float]]
    selected_model: str | None


def _early_rejection(
    candidate: RealismCandidate,
    cohorts: dict[str, Any],
    reasons: list[str],
) -> CandidateGateResult:
    hashes = {
        protocol: cohort_sha256(cohort)
        for protocol, cohort in cohorts.items()
        if protocol in VALIDATION_PROTOCOLS
    }
    return CandidateGateResult(
        candidate_id=candidate.candidate_id,
        config=asdict(candidate.config),
        config_sha256=candidate.config_sha256,
        scenario_seed=candidate.scenario_seed,
        passed=False,
        metrics={},
        reasons=tuple(reasons),
        selected_validation_reference={},
        cohort_hashes=hashes,
        protocols=tuple(protocol for protocol in VALIDATION_PROTOCOLS if protocol in cohorts),
        seeds=tuple(SEEDS),
        distance_to_range_midpoints=float("inf"),
    )


def _protocol_mean(values: dict[str, list[float]]) -> float:
    return float(np.mean([np.mean(values[protocol]) for protocol in VALIDATION_PROTOCOLS]))


def _improvement(model_value: float, reference_value: float) -> float:
    if not np.isfinite(reference_value) or reference_value <= 0:
        return float("nan")
    return float(100.0 * (reference_value - model_value) / reference_value)


def _raw_metrics(cohort: Any, prediction: SynthesizedPrediction) -> tuple[float, float]:
    median = prediction.quantiles[..., _MEDIAN_INDEX]
    return (
        normalized_rmse(cohort.targets, median, mask=cohort.daylight),
        quantile_crps(cohort.targets, prediction.quantiles, QUANTILES, mask=cohort.daylight),
    )


def _ramp_metrics(
    cohort: Any,
    prediction: SynthesizedPrediction,
    labels: np.ndarray,
) -> tuple[float, float, float]:
    ramp = np.asarray(cohort.daylight, dtype=bool) & np.isin(labels, (1, 2))
    median = prediction.quantiles[..., _MEDIAN_INDEX]
    return (
        normalized_rmse(cohort.targets, median, mask=ramp),
        quantile_crps(cohort.targets, prediction.quantiles, QUANTILES, mask=ramp),
        float(
            binary_ramp_metrics(
                labels,
                prediction.state_probabilities,
                mask=cohort.daylight,
            )["auprc"]
        ),
    )


def _coverage_metrics(
    cohort: Any,
    prediction: SynthesizedPrediction,
) -> tuple[float, float]:
    labels = np.asarray(prediction.state_labels)
    daylight = np.asarray(cohort.daylight, dtype=bool)
    lower, upper = prediction.quantiles[..., 0], prediction.quantiles[..., -1]
    return (
        prediction_interval_coverage(
            cohort.targets, lower, upper, mask=daylight & (labels == 0)
        ),
        prediction_interval_coverage(
            cohort.targets, lower, upper, mask=daylight & np.isin(labels, (1, 2))
        ),
    )


def _gate_contribution(
    prediction: SynthesizedPrediction,
) -> tuple[float, float, int, np.ndarray]:
    labels = np.asarray(prediction.state_labels)
    mask = labels >= 0
    selected_labels = labels[mask].astype(int)
    gates = np.asarray(prediction.gates, dtype=np.float64)[mask]
    correct_sum = float(np.sum(gates[np.arange(selected_labels.size), selected_labels]))
    entropy_sum = float(
        np.sum(-np.sum(gates * np.log(np.clip(gates, 1e-12, 1.0)), axis=1))
    )
    return correct_sum, entropy_sum, selected_labels.size, gates.sum(axis=0)


def _gate_metrics(
    correct_sum: float,
    entropy_sum: float,
    count: int,
    gate_sum: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    if count == 0:
        return float("nan"), float("nan"), np.full(3, np.nan)
    return (
        correct_sum / count,
        entropy_sum / count / math.log(3.0),
        gate_sum / count,
    )


def _relative_sd(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if array.size < 2 or not np.all(np.isfinite(array)) or mean <= 0:
        return float("nan")
    return float(np.std(array, ddof=1) / mean)


def _midpoint_distance(metrics: dict[str, float]) -> float:
    if any(not np.isfinite(value) for value in metrics.values()):
        return float("inf")
    offsets = []
    for name, value in metrics.items():
        lower, upper = VALIDATION_PRECHECK_RANGES[name]
        midpoint = (lower + upper) / 2.0
        offsets.append((value - midpoint) / (upper - lower))
    return float(np.sqrt(np.mean(np.square(offsets))))


def validation_precheck_reasons(metrics: dict[str, float]) -> tuple[str, ...]:
    """Return deterministic fail-closed reasons for every declared precheck."""

    reasons = []
    for name, (lower, upper) in VALIDATION_PRECHECK_RANGES.items():
        value = float(metrics.get(name, float("nan")))
        if not np.isfinite(value):
            reasons.append(f"{name}:non_finite")
        elif value < lower:
            reasons.append(f"{name}:below_minimum")
        elif value > upper:
            reasons.append(f"{name}:above_maximum")
    return tuple(reasons)


def _cohort_size_metadata(cohort: Any) -> dict[str, Any]:
    targets = np.asarray(cohort.targets)
    site_ids = sorted(set(np.asarray(cohort.site_ids)[:, 0].astype("U").tolist()))
    return {
        "forecast_sample_count": int(targets.size),
        "horizon": int(targets.shape[1]),
        "origin_count": int(targets.shape[0]),
        "selected_block_count": len(cohort.blocks),
        "site_count": len(site_ids),
        "site_ids": site_ids,
    }


def _validation_input_reasons(cohorts: dict[str, Any]) -> tuple[str, ...]:
    missing_protocols = [
        protocol for protocol in VALIDATION_PROTOCOLS if protocol not in cohorts
    ]
    if missing_protocols:
        return tuple(f"protocol:{protocol}:missing" for protocol in missing_protocols)
    reasons = [
        f"seed:{seed}:missing" for seed in EXPECTED_VALIDATION_SEEDS if seed not in SEEDS
    ]
    reasons.extend(
        f"protocol:{protocol}:not_validation"
        for protocol in VALIDATION_PROTOCOLS
        if getattr(cohorts[protocol], "split", None) != "validation"
    )
    return tuple(reasons)


def _candidate_input_reasons(candidate: RealismCandidate) -> tuple[str, ...]:
    if candidate.config_sha256 != canonical_config_sha256(candidate.config):
        return ("config_sha256:mismatch",)
    return ()


def _baseline_effective_key(candidate: RealismCandidate) -> _BaselineEffectiveKey:
    config = candidate.config
    return _BaselineEffectiveKey(
        scenario_seed=candidate.scenario_seed,
        stochastic_residual_weight=config.stochastic_residual_weight,
        spread_scale=config.spread_scale,
        ramp_spread_multiplier=config.ramp_spread_multiplier,
        router_noise_scale=config.router_noise_scale,
    )


def _build_validation_baseline_cache(
    cohorts: dict[str, Any],
    candidate: RealismCandidate,
) -> _ValidationBaselineCache:
    """Synthesize baseline scalars once for one complete effective input key."""

    by_model: dict[str, dict[str, dict[int, _BaselineSeedMetrics]]] = {
        profile.name: {protocol: {} for protocol in VALIDATION_PROTOCOLS}
        for profile in _DEEP_BASELINES
    }
    for protocol in VALIDATION_PROTOCOLS:
        cohort = cohorts[protocol]
        for profile in _DEEP_BASELINES:
            for seed in SEEDS:
                prediction = synthesize_predictions(
                    cohort,
                    profile,
                    seed=seed,
                    scenario_seed=candidate.scenario_seed,
                    generation_config=candidate.config,
                )
                nrmse, qcrps = _raw_metrics(cohort, prediction)
                ramp_nrmse, ramp_qcrps, ramp_auprc = _ramp_metrics(
                    cohort,
                    prediction,
                    np.asarray(prediction.state_labels),
                )
                by_model[profile.name][protocol][seed] = _BaselineSeedMetrics(
                    nrmse=nrmse,
                    qcrps=qcrps,
                    ramp_nrmse=ramp_nrmse,
                    ramp_qcrps=ramp_qcrps,
                    ramp_auprc=ramp_auprc,
                )
    protocol_scores = {
        model: {
            protocol: float(
                np.mean(
                    [
                        by_model[model][protocol][seed].qcrps
                        for seed in SEEDS
                    ]
                )
            )
            for protocol in VALIDATION_PROTOCOLS
        }
        for model in by_model
    }
    candidate_scores = {
        model: float(np.mean(list(scores.values())))
        for model, scores in protocol_scores.items()
    }
    finite_reference_scores = {
        model: score for model, score in candidate_scores.items() if np.isfinite(score)
    }
    selected_model = (
        min(
            finite_reference_scores,
            key=lambda model: (finite_reference_scores[model], model),
        )
        if finite_reference_scores
        else None
    )
    return _ValidationBaselineCache(
        by_model=by_model,
        candidate_scores=candidate_scores,
        protocol_scores=protocol_scores,
        selected_model=selected_model,
    )


def _build_validation_baseline_caches(
    cohorts: dict[str, Any],
    candidates: tuple[RealismCandidate, ...],
) -> dict[_BaselineEffectiveKey, _ValidationBaselineCache]:
    """Build one scalar summary per unique baseline-effective candidate key."""

    representatives: dict[_BaselineEffectiveKey, RealismCandidate] = {}
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        representatives.setdefault(_baseline_effective_key(candidate), candidate)
    return {
        key: _build_validation_baseline_cache(cohorts, representatives[key])
        for key in sorted(representatives)
    }


def _reference_payload(cache: _ValidationBaselineCache) -> dict[str, Any]:
    return {
        "candidate_family": "formal_deep_baselines_only",
        "candidate_scores": cache.candidate_scores,
        "held_out_used_for_selection": False,
        "protocol_scores": cache.protocol_scores,
        "selected_model": cache.selected_model,
        "selection_metric": "protocol_balanced_mean_quantile_crps",
        "selection_split": "validation",
    }


def _evaluate_validation_candidate(
    candidate: RealismCandidate,
    cohorts: dict[str, Any],
    baseline_cache: _ValidationBaselineCache | None = None,
) -> CandidateGateResult:
    """Evaluate with an optional call-local candidate-invariant baseline cache."""

    early_reasons = [
        *_validation_input_reasons(cohorts),
        *_candidate_input_reasons(candidate),
    ]
    if early_reasons:
        return _early_rejection(candidate, cohorts, early_reasons)

    cohort_hashes = {
        protocol: cohort_sha256(cohorts[protocol]) for protocol in VALIDATION_PROTOCOLS
    }
    cache = baseline_cache or _build_validation_baseline_cache(cohorts, candidate)
    reference = _reference_payload(cache)
    if cache.selected_model is None:
        metrics = {name: float("nan") for name in VALIDATION_PRECHECK_RANGES}
        return CandidateGateResult(
            candidate_id=candidate.candidate_id,
            config=asdict(candidate.config),
            config_sha256=candidate.config_sha256,
            scenario_seed=candidate.scenario_seed,
            passed=False,
            metrics=metrics,
            reasons=tuple(f"{name}:non_finite" for name in VALIDATION_PRECHECK_RANGES),
            selected_validation_reference=reference,
            cohort_hashes=cohort_hashes,
            protocols=VALIDATION_PROTOCOLS,
            seeds=tuple(SEEDS),
            distance_to_range_midpoints=float("inf"),
        )
    selected_model = cache.selected_model

    values = {
        name: {protocol: [] for protocol in VALIDATION_PROTOCOLS}
        for name in (
            "race_nrmse",
            "reference_nrmse",
            "race_qcrps",
            "reference_qcrps",
            "race_ramp_nrmse",
            "reference_ramp_nrmse",
            "race_ramp_qcrps",
            "reference_ramp_qcrps",
            "race_aucpr",
            "reference_aucpr",
            "stable_picp90",
            "ramp_picp90",
        )
    }
    race_seed_qcrps = {protocol: [] for protocol in VALIDATION_PROTOCOLS}
    gate_by_protocol: dict[str, tuple[float, float, np.ndarray]] = {}
    for protocol in VALIDATION_PROTOCOLS:
        cohort = cohorts[protocol]
        correct_sum = 0.0
        entropy_sum = 0.0
        gate_count = 0
        gate_sum = np.zeros(3, dtype=np.float64)
        for seed in SEEDS:
            race = synthesize_predictions(
                cohort,
                RACE_PV,
                seed=seed,
                scenario_seed=candidate.scenario_seed,
                generation_config=candidate.config,
            )
            reference_metrics = cache.by_model[selected_model][protocol][seed]
            contribution = _gate_contribution(race)
            correct_sum += contribution[0]
            entropy_sum += contribution[1]
            gate_count += contribution[2]
            gate_sum += contribution[3]
            race_nrmse, race_qcrps = _raw_metrics(cohort, race)
            labels = np.asarray(race.state_labels)
            race_ramp = _ramp_metrics(cohort, race, labels)
            stable_picp90, ramp_picp90 = _coverage_metrics(cohort, race)
            for name, value in (
                ("race_nrmse", race_nrmse),
                ("reference_nrmse", reference_metrics.nrmse),
                ("race_qcrps", race_qcrps),
                ("reference_qcrps", reference_metrics.qcrps),
                ("race_ramp_nrmse", race_ramp[0]),
                ("reference_ramp_nrmse", reference_metrics.ramp_nrmse),
                ("race_ramp_qcrps", race_ramp[1]),
                ("reference_ramp_qcrps", reference_metrics.ramp_qcrps),
                ("race_aucpr", race_ramp[2]),
                ("reference_aucpr", reference_metrics.ramp_auprc),
                ("stable_picp90", stable_picp90),
                ("ramp_picp90", ramp_picp90),
            ):
                values[name][protocol].append(float(value))
            race_seed_qcrps[protocol].append(race_qcrps)
        gate_by_protocol[protocol] = _gate_metrics(
            correct_sum, entropy_sum, gate_count, gate_sum
        )

    neural_relative_sd = []
    for protocol in VALIDATION_PROTOCOLS:
        for profile in _DEEP_BASELINES:
            neural_relative_sd.append(
                _relative_sd(
                    [
                        cache.by_model[profile.name][protocol][seed].qcrps
                        for seed in SEEDS
                    ]
                )
            )
        neural_relative_sd.append(_relative_sd(race_seed_qcrps[protocol]))
    mean_gate_share = np.mean(
        [gate_by_protocol[protocol][2] for protocol in VALIDATION_PROTOCOLS], axis=0
    )
    metrics = {
        "daylight_nrmse_improvement_pct": _improvement(
            _protocol_mean(values["race_nrmse"]),
            _protocol_mean(values["reference_nrmse"]),
        ),
        "mean_quantile_crps_improvement_pct": _improvement(
            _protocol_mean(values["race_qcrps"]),
            _protocol_mean(values["reference_qcrps"]),
        ),
        "ramp_nrmse_improvement_pct": _improvement(
            _protocol_mean(values["race_ramp_nrmse"]),
            _protocol_mean(values["reference_ramp_nrmse"]),
        ),
        "ramp_qcrps_improvement_pct": _improvement(
            _protocol_mean(values["race_ramp_qcrps"]),
            _protocol_mean(values["reference_ramp_qcrps"]),
        ),
        "ramp_aucpr_absolute_gain": (
            _protocol_mean(values["race_aucpr"])
            - _protocol_mean(values["reference_aucpr"])
        ),
        "stable_picp90": _protocol_mean(values["stable_picp90"]),
        "ramp_picp90": _protocol_mean(values["ramp_picp90"]),
        "neural_seed_relative_sd": float(np.mean(neural_relative_sd)),
        "correct_expert_mean_gate": float(
            np.mean([gate_by_protocol[protocol][0] for protocol in VALIDATION_PROTOCOLS])
        ),
        "normalized_gate_entropy": float(
            np.mean([gate_by_protocol[protocol][1] for protocol in VALIDATION_PROTOCOLS])
        ),
        "maximum_expert_global_share": float(np.max(mean_gate_share)),
    }
    reasons = validation_precheck_reasons(metrics)
    return CandidateGateResult(
        candidate_id=candidate.candidate_id,
        config=asdict(candidate.config),
        config_sha256=candidate.config_sha256,
        scenario_seed=candidate.scenario_seed,
        passed=not reasons,
        metrics=metrics,
        reasons=reasons,
        selected_validation_reference=reference,
        cohort_hashes=cohort_hashes,
        protocols=VALIDATION_PROTOCOLS,
        seeds=tuple(SEEDS),
        distance_to_range_midpoints=_midpoint_distance(metrics),
    )


def evaluate_validation_candidate(
    candidate: RealismCandidate,
    cohorts: dict[str, Any],
) -> CandidateGateResult:
    """Evaluate one whole candidate on validation data only, balanced by protocol."""

    return _evaluate_validation_candidate(candidate, cohorts)


def screen_candidates(
    candidates: list[RealismCandidate] | tuple[RealismCandidate, ...],
    cohorts: dict[str, Any],
) -> CandidateScreeningReport:
    """Evaluate and rank at least two complete candidates without test-data access."""

    rows = tuple(candidates)
    if len(rows) < 2:
        raise ValueError("candidate screening requires at least two candidates")
    candidate_ids = [candidate.candidate_id for candidate in rows]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique")
    globally_valid = not _validation_input_reasons(cohorts)
    has_valid_candidate = any(not _candidate_input_reasons(candidate) for candidate in rows)
    valid_candidates = tuple(
        candidate for candidate in rows if not _candidate_input_reasons(candidate)
    )
    baseline_caches = (
        _build_validation_baseline_caches(cohorts, valid_candidates)
        if globally_valid and has_valid_candidate
        else {}
    )
    evaluated = [
        _evaluate_validation_candidate(
            candidate,
            cohorts,
            baseline_caches.get(_baseline_effective_key(candidate)),
        )
        for candidate in sorted(rows, key=lambda item: item.candidate_id)
    ]
    accepted = sorted(
        (result for result in evaluated if result.passed),
        key=lambda result: (result.distance_to_range_midpoints, result.candidate_id),
    )
    rejected = sorted(
        (result for result in evaluated if not result.passed),
        key=lambda result: result.candidate_id,
    )
    hashes = {
        protocol: cohort_sha256(cohorts[protocol])
        for protocol in VALIDATION_PROTOCOLS
        if protocol in cohorts
    }
    sizes = {
        protocol: _cohort_size_metadata(cohorts[protocol])
        for protocol in VALIDATION_PROTOCOLS
        if protocol in cohorts
    }
    return CandidateScreeningReport(tuple((*accepted, *rejected)), hashes, sizes)


__all__ = [
    "CandidateGateResult",
    "CandidateScreeningReport",
    "VALIDATION_PRECHECK_RANGES",
    "evaluate_validation_candidate",
    "screen_candidates",
    "validation_precheck_reasons",
]
