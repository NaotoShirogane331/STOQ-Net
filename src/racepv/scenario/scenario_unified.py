"""Dual-protocol phase-4D evaluation and numerical acceptance reporting."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from racepv.calibration.cqr import lead_band_names
from racepv.metrics import (
    binary_ramp_metrics,
    interval_score,
    normalized_rmse,
    prediction_interval_coverage,
    quantile_crps,
)

from .compact_storage import load_prediction_arrays, prediction_entries
from .constants import (
    ABLATIONS,
    BASELINES,
    EVIDENCE_METADATA,
    QUANTILES,
    RACE_PV,
    REPORT_LEADS,
    SEEDS,
    VALIDATION_SENSITIVITIES,
)
from .evaluation import benjamini_hochberg, diebold_mariano_test, forecast_skill
from .realism import ScenarioGenerationConfig, canonical_config_sha256
from .scenario_artifacts import DISCLOSURE, artifact_root_digest, sha256_file, write_json
from .scenario_manifest import ACCEPTANCE_RANGES

PROTOCOLS = ("dkasc-yulara", "nrel-transfer")
ADDITIONAL_ACCEPTANCE_CHECKS = (
    "state_lead_width_not_abnormal",
    "all_ablations_degrade_a_primary_metric",
    "held_out_not_used_for_reference_selection",
)
_DERIVED_ACCEPTANCE_PATHS = (
    "02-predictions/dkasc-yulara/manifest.json",
    "02-predictions/dkasc-yulara/evaluations.json",
    "02-predictions/dkasc-yulara/selection.json",
    "02-predictions/dkasc-yulara/aggregate.json",
    "02-predictions/dkasc-yulara/artifact-index.json",
    "02-predictions/dkasc-yulara/cohort-index.json",
    "02-predictions/nrel-transfer/manifest.json",
    "02-predictions/nrel-transfer/evaluations.json",
    "02-predictions/nrel-transfer/selection.json",
    "02-predictions/nrel-transfer/aggregate.json",
    "02-predictions/nrel-transfer/artifact-index.json",
    "02-predictions/nrel-transfer/cohort-index.json",
    "03-evaluations/unified-evaluation.json",
    "04-aggregates/unified-aggregate.json",
    "04-aggregates/acceptance.json",
    "04-aggregates/unified-artifact-index.json",
    "09-logs/materialization-journal.json",
)


@dataclass(frozen=True)
class CandidateAcceptance:
    """Recomputed 16+3 decision with an audit of the derived inputs used."""

    passed: bool
    reasons: tuple[str, ...]
    checks: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def unmet_checks(self) -> tuple[str, ...]:
        return self.reasons


def _real_finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float, np.integer, np.floating))
        and math.isfinite(float(value))
    )


def _contains_non_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_non_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_non_finite(item) for item in value)
    return isinstance(value, (float, np.floating)) and not math.isfinite(float(value))


def _valid_reference_selection(value: Any) -> bool:
    formal_deep_models = {
        profile.name for profile in BASELINES if profile.family == "neural"
    }
    return bool(
        isinstance(value, Mapping)
        and isinstance(value.get("selected_model"), str)
        and value.get("selected_model") in formal_deep_models
        and value.get("selection_split") == "validation"
        and value.get("selection_metric") == "protocol_balanced_mean_quantile_crps"
        and value.get("candidate_family") == "formal_deep_baselines_only"
        and value.get("held_out_used_for_selection") is False
    )


def evaluate_candidate_acceptance_payload(
    acceptance: Mapping[str, Any],
) -> CandidateAcceptance:
    """Fail closed unless the payload exactly satisfies every declared 16+3 check."""

    if not isinstance(acceptance, Mapping):
        return CandidateAcceptance(False, ("acceptance_payload_malformed",))
    reasons: list[str] = []
    checks = acceptance.get("checks")
    if not isinstance(checks, Mapping):
        checks = {}
    for name, bounds in ACCEPTANCE_RANGES.items():
        check = checks.get(name)
        valid = isinstance(check, Mapping)
        if valid:
            value = check.get("value")
            minimum = check.get("minimum")
            maximum = check.get("maximum")
            valid = bool(
                _real_finite(value)
                and _real_finite(minimum)
                and _real_finite(maximum)
                and float(minimum) == float(bounds["minimum"])
                and float(maximum) == float(bounds["maximum"])
                and float(minimum) <= float(value) <= float(maximum)
                and check.get("passed") is True
            )
        if not valid:
            reasons.append(name)
    reasons.extend(
        f"unexpected_check:{name}"
        for name in sorted(set(checks).difference(ACCEPTANCE_RANGES))
    )

    additional = acceptance.get("additional_checks")
    if not isinstance(additional, Mapping):
        additional = {}
    width = additional.get("state_lead_width_not_abnormal")
    if not (
        isinstance(width, Mapping)
        and _real_finite(width.get("value"))
        and _real_finite(width.get("maximum"))
        and float(width["maximum"]) == 1.10
        and float(width["value"]) <= 1.10
        and width.get("passed") is True
    ):
        reasons.append("state_lead_width_not_abnormal")
    ablations = additional.get("all_ablations_degrade_a_primary_metric")
    if not (
        isinstance(ablations, Mapping)
        and isinstance(ablations.get("value"), int)
        and not isinstance(ablations.get("value"), bool)
        and ablations.get("value") == len(ABLATIONS)
        and ablations.get("required") == len(ABLATIONS)
        and ablations.get("passed") is True
    ):
        reasons.append("all_ablations_degrade_a_primary_metric")
    held_out = additional.get("held_out_not_used_for_reference_selection")
    if not (
        isinstance(held_out, Mapping)
        and held_out.get("value") is False
        and held_out.get("required") is False
        and held_out.get("passed") is True
        and _valid_reference_selection(acceptance.get("reference_selection"))
    ):
        reasons.append("held_out_not_used_for_reference_selection")
    reasons.extend(
        f"unexpected_additional_check:{name}"
        for name in sorted(set(additional).difference(ADDITIONAL_ACCEPTANCE_CHECKS))
    )
    return CandidateAcceptance(
        not reasons,
        tuple(reasons),
        {
            "checks": dict(checks),
            "additional_checks": dict(additional),
        },
    )


def _load_derived_payload(root: Path, relative: str) -> tuple[dict[str, Any] | None, str | None]:
    path = root / relative
    if not path.is_file():
        return None, None
    digest = sha256_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, digest
    return (payload if isinstance(payload, dict) else None), digest


def _finite_rows(rows: Any, fields: tuple[str, ...]) -> bool:
    return bool(
        isinstance(rows, list)
        and all(
            isinstance(row, Mapping) and all(_real_finite(row.get(field)) for field in fields)
            for row in rows
        )
    )


def _recompute_reference(payloads: Mapping[str, dict[str, Any]]) -> dict[str, Any] | None:
    deep_models = tuple(profile.name for profile in BASELINES if profile.family == "neural")
    protocol_scores: dict[str, dict[str, float]] = {model: {} for model in deep_models}
    for protocol in PROTOCOLS:
        evaluations = payloads.get(f"02-predictions/{protocol}/evaluations.json", {})
        runs = evaluations.get("runs")
        if not isinstance(runs, list):
            return None
        for model in deep_models:
            rows = [
                row for row in runs
                if row.get("model") == model and row.get("split") == "validation"
            ]
            if {row.get("seed") for row in rows} != set(SEEDS) or len(rows) != len(SEEDS):
                return None
            values = [row.get("metrics", {}).get("quantile_crps") for row in rows]
            if not all(_real_finite(value) for value in values):
                return None
            protocol_scores[model][protocol] = float(np.mean(values))
    candidate_scores = {
        model: float(np.mean([protocol_scores[model][protocol] for protocol in PROTOCOLS]))
        for model in deep_models
    }
    selected = min(candidate_scores, key=lambda model: (candidate_scores[model], model))
    return {
        "selected_model": selected,
        "selection_split": "validation",
        "selection_metric": "protocol_balanced_mean_quantile_crps",
        "candidate_family": "formal_deep_baselines_only",
        "candidate_scores": candidate_scores,
        "protocol_scores": protocol_scores,
        "held_out_used_for_selection": False,
    }


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def evaluate_candidate_acceptance(root: Path) -> CandidateAcceptance:
    """Re-evaluate one candidate from JSON aggregates/evaluations, never prediction arrays."""

    candidate_root = Path(root).expanduser().resolve()
    payloads: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    reasons: list[str] = []
    for relative in _DERIVED_ACCEPTANCE_PATHS:
        payload, digest = _load_derived_payload(candidate_root, relative)
        if digest is None:
            reasons.append(f"derived_artifact_missing:{relative}")
        elif payload is None:
            reasons.append(f"derived_artifact_malformed:{relative}")
            hashes[relative] = digest
        else:
            payloads[relative] = payload
            hashes[relative] = digest
            if _contains_non_finite(payload):
                reasons.append(f"non_finite_json:{relative}")

    acceptance = payloads.get("04-aggregates/acceptance.json", {})
    payload_result = evaluate_candidate_acceptance_payload(acceptance)
    reasons = [*payload_result.reasons, *reasons]
    aggregate = payloads.get("04-aggregates/unified-aggregate.json")
    evaluation = payloads.get("03-evaluations/unified-evaluation.json")

    profiles = (*BASELINES, RACE_PV, *ABLATIONS)
    profiles = (*profiles, *VALIDATION_SENSITIVITIES)
    expected_model_splits = {
        (profile.name, split) for profile in profiles for split in profile.applies_to
    }
    protocol_sites: dict[str, set[str]] = {}
    protocol_index_roots: dict[str, str] = {}
    config_hashes: set[str] = set()
    generation_configs: list[dict[str, Any]] = []
    validation_cohort_hashes: dict[str, str] = {}
    for protocol in PROTOCOLS:
        prefix = f"02-predictions/{protocol}"
        manifest = payloads.get(f"{prefix}/manifest.json", {})
        evaluations = payloads.get(f"{prefix}/evaluations.json", {})
        protocol_aggregate = payloads.get(f"{prefix}/aggregate.json", {})
        cohort_index = payloads.get(f"{prefix}/cohort-index.json", {})
        identities = manifest.get("run_identities")
        expected_identities = {
            (protocol, model, seed, split)
            for model, split in expected_model_splits
            for seed in SEEDS
        }
        actual_identities = {
            (row.get("protocol"), row.get("model"), row.get("seed"), row.get("split"))
            for row in identities
        } if isinstance(identities, list) else set()
        evaluation_rows = evaluations.get("runs")
        evaluation_identities = {
            (row.get("protocol", protocol), row.get("model"), row.get("seed"), row.get("split"))
            for row in evaluation_rows
        } if isinstance(evaluation_rows, list) else set()
        aggregate_identities = {
            (row.get("model"), row.get("split"))
            for row in protocol_aggregate.get("rows", ())
        }
        if (
            actual_identities != expected_identities
            or len(identities or ()) != len(expected_identities)
            or evaluation_identities != expected_identities
            or len(evaluation_rows or ()) != len(expected_identities)
            or aggregate_identities != expected_model_splits
        ):
            reasons.append(f"protocol_identity_set_mismatch:{protocol}")
        indexed_cohorts = cohort_index.get("cohorts")
        test_cohort = (
            indexed_cohorts.get("test", {})
            if isinstance(indexed_cohorts, Mapping)
            else {}
        )
        indexed_sites = (
            test_cohort.get("site_ids") if isinstance(test_cohort, Mapping) else None
        )
        sites_valid = bool(
            isinstance(indexed_sites, list)
            and indexed_sites
            and all(isinstance(site, str) and site for site in indexed_sites)
            and len(indexed_sites) == len(set(indexed_sites))
        )
        expected_sites = set(indexed_sites) if sites_valid else set()
        protocol_sites[protocol] = expected_sites
        race_test_rows = [
            row
            for row in (evaluation_rows or ())
            if row.get("model") == RACE_PV.name and row.get("split") == "test"
        ]
        expected_site_leads = {
            (site, lead) for site in expected_sites for lead in REPORT_LEADS
        }
        if not sites_valid or any(
            not isinstance(row.get("site_lead"), list)
            or len(row["site_lead"]) != len(expected_site_leads)
            or {
                (item.get("site_id"), item.get("lead"))
                for item in row["site_lead"]
                if isinstance(item, Mapping)
            }
            != expected_site_leads
            for row in race_test_rows
        ):
            reasons.append(f"race_test_site_identity_mismatch:{protocol}")
        config_hash = manifest.get("generation_config_sha256")
        generation_config = manifest.get("generation_config")
        try:
            resolved_config = (
                ScenarioGenerationConfig(**generation_config)
                if isinstance(generation_config, Mapping)
                else None
            )
        except (TypeError, ValueError):
            resolved_config = None
        if (
            resolved_config is None
            or not _valid_sha256(config_hash)
            or canonical_config_sha256(resolved_config) != config_hash
        ):
            reasons.append(f"generation_config_provenance_mismatch:{protocol}")
        else:
            config_hashes.add(config_hash)
            generation_configs.append(dict(generation_config))
        cohorts = manifest.get("cohorts")
        cohort_hashes_match = bool(
            isinstance(cohorts, Mapping)
            and isinstance(indexed_cohorts, Mapping)
            and manifest.get("cohort_index_path") == "cohort-index.json"
            and manifest.get("cohort_index_sha256")
            == hashes.get(f"{prefix}/cohort-index.json")
            and all(
                isinstance(cohorts.get(split), Mapping)
                and isinstance(indexed_cohorts.get(split), Mapping)
                and _valid_sha256(cohorts.get(split, {}).get("cohort_sha256"))
                and cohorts.get(split, {}).get("cohort_sha256")
                == indexed_cohorts.get(split, {}).get("cohort_sha256")
                for split in ("calibration", "validation", "test")
            )
        )
        if not cohort_hashes_match:
            reasons.append(f"cohort_hash_provenance_mismatch:{protocol}")
        else:
            validation_cohort_hashes[protocol] = cohorts["validation"]["cohort_sha256"]
        index = payloads.get(f"{prefix}/artifact-index.json", {})
        artifacts = index.get("artifacts")
        valid_index = isinstance(artifacts, Mapping)
        if valid_index:
            path_entries = {entry.get("path"): entry for entry in artifacts.values()}
            for name in (
                "manifest.json",
                "cohort-index.json",
                "evaluations.json",
                "selection.json",
                "aggregate.json",
            ):
                entry = path_entries.get(name, {})
                valid_index = valid_index and entry.get("sha256") == hashes.get(f"{prefix}/{name}")
                valid_index = valid_index and all(source in artifacts for source in entry.get("source_artifact_ids", ()))
        if not valid_index:
            reasons.append(f"protocol_artifact_index_mismatch:{protocol}")
        else:
            protocol_index_roots[protocol] = artifact_root_digest(index)
    if len(config_hashes) != 1 or len(generation_configs) != len(PROTOCOLS) or any(
        config != generation_configs[0] for config in generation_configs[1:]
    ):
        reasons.append("generation_config_provenance_mismatch")

    journal = payloads.get("09-logs/materialization-journal.json", {})
    journal_sources = journal.get("materialization_source_hashes")
    candidate_config_hash = next(iter(config_hashes), None)
    if not (
        isinstance(journal.get("candidate_id"), str)
        and journal.get("candidate_id")
        and journal.get("candidate_config_sha256") == candidate_config_hash
        and _valid_sha256(journal.get("screening_report_sha256"))
    ):
        reasons.append("generation_config_provenance_mismatch")
    if not isinstance(journal_sources, Mapping) or any(
        journal_sources.get(f"validation_cohort:{protocol}")
        != validation_cohort_hashes.get(protocol)
        for protocol in PROTOCOLS
    ):
        reasons.append("cohort_source_provenance_mismatch")

    recomputed_reference = _recompute_reference(payloads)
    persisted_references = [
        payload.get("reference_selection")
        for payload in (acceptance, aggregate, evaluation)
        if isinstance(payload, Mapping)
    ]
    if recomputed_reference is None or any(
        reference != recomputed_reference for reference in persisted_references
    ):
        reasons.append("reference_selection_recomputed_mismatch")

    unified_index = payloads.get("04-aggregates/unified-artifact-index.json", {})
    expected_outputs = {
        relative: hashes.get(relative)
        for relative in (
            "03-evaluations/unified-evaluation.json",
            "04-aggregates/unified-aggregate.json",
            "04-aggregates/acceptance.json",
        )
    }
    if not (
        unified_index.get("stage") == "4D_unified_evaluation"
        and unified_index.get("outputs") == expected_outputs
        and unified_index.get("sources") == acceptance.get("source_hashes")
        and unified_index.get("artifact_index_roots") == protocol_index_roots
    ):
        reasons.append("unified_artifact_index_mismatch")

    if aggregate is not None and evaluation is not None:
        expected_path = "03-evaluations/unified-evaluation.json"
        if aggregate.get("evaluation_path") != expected_path:
            reasons.append("unified_evaluation_path_mismatch")
        if aggregate.get("evaluation_sha256") != hashes.get(expected_path):
            reasons.append("unified_evaluation_sha256_mismatch")
    if acceptance and aggregate is not None:
        expected_path = "04-aggregates/unified-aggregate.json"
        if acceptance.get("unified_aggregate_path") != expected_path:
            reasons.append("unified_aggregate_path_mismatch")
        if acceptance.get("unified_aggregate_sha256") != hashes.get(expected_path):
            reasons.append("unified_aggregate_sha256_mismatch")

    if aggregate is not None:
        values = aggregate.get("values")
        for name in ACCEPTANCE_RANGES:
            check = acceptance.get("checks", {}).get(name, {})
            if not isinstance(values, Mapping) or values.get(name) != check.get("value"):
                reasons.append(f"aggregate_value_mismatch:{name}")
        if aggregate.get("dm_bh_hypotheses") != 150:
            reasons.append("dm_bh_hypotheses:expected_150")
        if aggregate.get("forecast_skill_sites") != 25:
            reasons.append("forecast_skill_sites:expected_25")

    if evaluation is not None:
        dm_rows = evaluation.get("site_lead_dm_bh")
        actual = len(dm_rows) if isinstance(dm_rows, list) else 0
        if actual != 150:
            reasons.append(f"site_lead_dm_bh_rows:expected_150:actual_{actual}")
        elif not _finite_rows(
            dm_rows,
            (
                "statistic",
                "pvalue",
                "mean_loss_difference",
                "bh_adjusted_pvalue",
                "n_observations",
            ),
        ) or any(not isinstance(row.get("bh_rejected"), bool) for row in dm_rows):
            reasons.append("site_lead_dm_bh_rows:non_finite_or_malformed")
        elif len({(row.get("protocol"), row.get("site_id"), row.get("lead")) for row in dm_rows}) != 150:
            reasons.append("site_lead_dm_bh_rows:duplicate_or_invalid_identities")
        elif {
            (row.get("protocol"), row.get("site_id"), row.get("lead")) for row in dm_rows
        } != {
            (protocol, site, lead)
            for protocol, sites in protocol_sites.items()
            for site in sites
            for lead in REPORT_LEADS
        }:
            reasons.append("site_lead_dm_bh_rows:cross_table_identity_mismatch")
        skill_rows = evaluation.get("forecast_skill")
        actual = len(skill_rows) if isinstance(skill_rows, list) else 0
        if actual != 25:
            reasons.append(f"forecast_skill_rows:expected_25:actual_{actual}")
        elif not _finite_rows(
            skill_rows, ("race_pv_nrmse", "reference_nrmse", "forecast_skill")
        ):
            reasons.append("forecast_skill_rows:non_finite_or_malformed")
        elif len({(row.get("protocol"), row.get("site_id")) for row in skill_rows}) != 25:
            reasons.append("forecast_skill_rows:duplicate_or_invalid_identities")
        elif {(row.get("protocol"), row.get("site_id")) for row in skill_rows} != {
            (protocol, site) for protocol, sites in protocol_sites.items() for site in sites
        }:
            reasons.append("forecast_skill_rows:cross_table_identity_mismatch")
        calibration = evaluation.get("calibration")
        calibration_rows = calibration.get("rows") if isinstance(calibration, Mapping) else None
        actual = len(calibration_rows) if isinstance(calibration_rows, list) else 0
        if actual != 10:
            reasons.append(f"calibration_rows:expected_10:actual_{actual}")
        elif not _finite_rows(
            calibration_rows,
            (
                "stable_picp",
                "ramp_picp",
                "lead_interval_score",
                "state_interval_score",
                "lead_width",
                "state_width",
            ),
        ):
            reasons.append("calibration_rows:non_finite_or_malformed")
        elif {(row.get("protocol"), row.get("seed")) for row in calibration_rows} != {
            (protocol, seed) for protocol in PROTOCOLS for seed in SEEDS
        }:
            reasons.append("calibration_rows:duplicate_or_invalid_identities")
        calibration_summary = evaluation.get("calibration", {})
        width = calibration_summary.get("state_lead_to_lead_only_width_ratio")
        if not _real_finite(width) or float(width) > 1.10:
            reasons.append("state_lead_width_not_abnormal")
        expected_ablations = {profile.name for profile in ABLATIONS}
        ablations = evaluation.get("ablations")
        if not (
            isinstance(ablations, list)
            and len(ablations) == len(expected_ablations)
            and {row.get("model") for row in ablations} == expected_ablations
            and all(
                row.get("passed") is True
                and isinstance(row.get("degraded_primary_metrics"), list)
                and row["degraded_primary_metrics"]
                for row in ablations
            )
        ):
            reasons.append("all_ablations_degrade_a_primary_metric")

    bootstrap_rows: list[Any] = []
    for protocol in PROTOCOLS:
        payload = payloads.get(f"02-predictions/{protocol}/aggregate.json")
        rows = payload.get("rows") if isinstance(payload, Mapping) else None
        if isinstance(rows, list):
            bootstrap_rows.extend(rows)
    if len(bootstrap_rows) != 102:
        reasons.append(f"bootstrap_aggregate_rows:expected_102:actual_{len(bootstrap_rows)}")
    elif not _finite_rows(
        bootstrap_rows,
        ("bootstrap_estimate", "bootstrap_ci_low", "bootstrap_ci_high"),
    ):
        reasons.append("bootstrap_aggregate_rows:non_finite_or_malformed")
    elif any(
        len(
            {
                (row.get("model"), row.get("split"))
                for row in payloads[f"02-predictions/{protocol}/aggregate.json"]["rows"]
            }
        )
        != 51
        for protocol in PROTOCOLS
    ):
        reasons.append("bootstrap_aggregate_rows:duplicate_or_invalid_identities")

    source_maps = [
        payload.get("source_hashes")
        for payload in (acceptance, aggregate, evaluation)
        if isinstance(payload, Mapping)
    ]
    if len(source_maps) != 3 or not all(isinstance(item, Mapping) for item in source_maps):
        reasons.append("source_hash_provenance_missing")
    elif not (source_maps[0] == source_maps[1] == source_maps[2]):
        reasons.append("source_hash_provenance_mismatch")
    for protocol in PROTOCOLS:
        relative = f"02-predictions/{protocol}/aggregate.json"
        bootstrap_source = (
            aggregate.get("bootstrap_sources", {}).get(protocol, {}) if aggregate else {}
        )
        mapped = (
            source_maps[0].get(protocol, {})
            if source_maps and isinstance(source_maps[0], Mapping)
            else {}
        )
        if not isinstance(mapped, Mapping):
            mapped = {}
        if not (
            bootstrap_source.get("path") == relative
            and bootstrap_source.get("sha256") == hashes.get(relative)
            and mapped.get("aggregate.json") == hashes.get(relative)
        ):
            reasons.append(f"protocol_aggregate_provenance_mismatch:{protocol}")
        expected_source_names = {
            "manifest.json",
            "evaluations.json",
            "selection.json",
            "aggregate.json",
            "artifact-index.json",
        }
        if set(mapped) != expected_source_names:
            reasons.append(f"protocol_source_hash_set_mismatch:{protocol}")
        for name in sorted(expected_source_names):
            if mapped.get(name) != hashes.get(f"02-predictions/{protocol}/{name}"):
                reasons.append(f"protocol_source_hash_mismatch:{protocol}:{name}")

    references = [
        payload.get("reference_selection")
        for payload in (acceptance, aggregate, evaluation)
        if isinstance(payload, Mapping)
    ]
    if len(references) != 3 or not all(_valid_reference_selection(item) for item in references):
        reasons.append("reference_selection_not_validation_only")
    elif not (references[0] == references[1] == references[2]):
        reasons.append("reference_selection_provenance_mismatch")

    for relative, payload in payloads.items():
        if any(payload.get(key) != value for key, value in EVIDENCE_METADATA.items()):
            reasons.append(f"evidence_metadata_mismatch:{relative}")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return CandidateAcceptance(
        not unique_reasons,
        unique_reasons,
        payload_result.checks,
        {
            **dict(EVIDENCE_METADATA),
            "candidate_root": str(candidate_root),
            "derived_artifact_sha256": hashes,
            "reference_selection": acceptance.get("reference_selection", {}),
            "bootstrap_aggregate_rows": len(bootstrap_rows),
            "site_lead_dm_bh_rows": (
                len(evaluation.get("site_lead_dm_bh", [])) if evaluation else 0
            ),
            "forecast_skill_rows": (
                len(evaluation.get("forecast_skill", [])) if evaluation else 0
            ),
            "calibration_rows": (
                len(evaluation.get("calibration", {}).get("rows", [])) if evaluation else 0
            ),
        },
    )


def acceptance_check(value: float, bounds: dict[str, float]) -> dict[str, Any]:
    numeric = float(value)
    minimum, maximum = float(bounds["minimum"]), float(bounds["maximum"])
    return {
        "value": numeric,
        "minimum": minimum,
        "maximum": maximum,
        "passed": bool(np.isfinite(numeric) and minimum <= numeric <= maximum),
    }


def select_strongest_deep_baseline(
    records_by_protocol: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    deep_models = tuple(profile.name for profile in BASELINES if profile.family == "neural")
    scores: dict[str, float] = {}
    protocol_scores: dict[str, dict[str, float]] = {}
    for model in deep_models:
        by_protocol: dict[str, float] = {}
        for protocol, records in records_by_protocol.items():
            values = [
                float(record["metrics"]["quantile_crps"])
                for record in records
                if record["model"] == model and record["split"] == "validation"
            ]
            if values:
                by_protocol[protocol] = float(np.mean(values))
        if by_protocol:
            protocol_scores[model] = by_protocol
            scores[model] = float(np.mean(list(by_protocol.values())))
    if not scores:
        raise ValueError("unified reference selection found no deep validation records")
    selected = min(scores, key=lambda model: (scores[model], model))
    return {
        "selected_model": selected,
        "selection_split": "validation",
        "selection_metric": "protocol_balanced_mean_quantile_crps",
        "candidate_family": "formal_deep_baselines_only",
        "candidate_scores": scores,
        "protocol_scores": protocol_scores,
        "held_out_used_for_selection": False,
    }


def calibrated_interval_from_report(
    quantiles: np.ndarray,
    state_probabilities: np.ndarray,
    parameter_rows: list[dict[str, Any]],
    *,
    control: str,
    daylight: np.ndarray,
    ramp_threshold: float = 0.0,
) -> np.ndarray:
    forecasts = np.asarray(quantiles, dtype=np.float64)
    probabilities = np.asarray(state_probabilities, dtype=np.float64)
    day = np.asarray(daylight, dtype=bool)
    if forecasts.ndim != 3 or forecasts.shape[-1] != len(QUANTILES):
        raise ValueError("quantiles must have shape [N,H,7]")
    if probabilities.shape != (*forecasts.shape[:2], 3) or day.shape != forecasts.shape[:2]:
        raise ValueError("state probabilities/daylight shape mismatch")
    if control not in {"global", "lead_only", "state_lead"}:
        raise ValueError("control must be global, lead_only, or state_lead")
    bands = lead_band_names(forecasts.shape[1])
    ramp_score = probabilities[..., 1:].sum(axis=-1)
    state = np.where(ramp_score < float(ramp_threshold), 0, np.argmax(probabilities, axis=-1))
    corrections = np.zeros(forecasts.shape[:2], dtype=np.float64)
    lookup = {
        (str(row["state_group"]), str(row["lead_band"])): float(row["correction"])
        for row in parameter_rows
    }
    for lead, band in enumerate(bands):
        if control == "global":
            correction = lookup.get(("global", "all"))
            if correction is None:
                raise ValueError("global calibration correction is missing")
            corrections[:, lead] = correction
        elif control == "lead_only":
            correction = lookup.get(("all-states", str(band)))
            if correction is None:
                raise ValueError(f"lead-only calibration correction is missing: {band}")
            corrections[:, lead] = correction
        else:
            for label, group in ((0, "stable"), (1, "ramp"), (2, "ramp")):
                correction = lookup.get((group, str(band)))
                if correction is None:
                    raise ValueError(f"state-lead calibration correction is missing: {group}/{band}")
                corrections[state[:, lead] == label, lead] = correction
    lower = np.clip(forecasts[..., 0] - corrections, 0.0, 1.05)
    upper = np.clip(forecasts[..., -1] + corrections, 0.0, 1.05)
    interval = np.stack((lower, upper), axis=-1)
    interval[~day] = 0.0
    return interval


def _record_mean(
    records_by_protocol: dict[str, list[dict[str, Any]]],
    model: str,
    split: str,
    field: str,
) -> float:
    protocol_values = []
    for records in records_by_protocol.values():
        selected = [record for record in records if record["model"] == model and record["split"] == split]
        if field == "quantile_crps":
            values = [float(record["metrics"]["quantile_crps"]) for record in selected]
        elif field == "daylight_nrmse":
            values = [float(record["metrics"]["raw"]["daylight"]["nrmse"]) for record in selected]
        else:
            raise ValueError(f"unknown record field: {field}")
        protocol_values.append(float(np.mean(values)))
    return float(np.mean(protocol_values))


def _improvement(model_value: float, baseline_value: float) -> float:
    return float((baseline_value - model_value) / baseline_value * 100.0)


def _finite_mean(values: np.ndarray, *, axis: int) -> np.ndarray:
    """Average finite values without warning for structurally empty slices."""
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    counts = np.sum(finite, axis=axis)
    sums = np.sum(np.where(finite, array, 0.0), axis=axis)
    result = np.full(np.shape(sums), np.nan, dtype=np.float64)
    np.divide(sums, counts, out=result, where=counts > 0)
    return result


def _prediction_map(root: Path) -> dict[str, dict[str, Any]]:
    return {logical["run_id"]: source for _path, logical, source in prediction_entries(root)}


def _source_hashes(roots: dict[str, Path]) -> dict[str, dict[str, str]]:
    names = ("manifest.json", "evaluations.json", "selection.json", "aggregate.json", "artifact-index.json")
    return {
        protocol: {name: sha256_file(root / name) for name in names}
        for protocol, root in roots.items()
    }


def run_unified_4d(scenario_root: Path) -> dict[str, Any]:
    scenario_root = Path(scenario_root).expanduser().resolve()
    roots = {protocol: scenario_root / "02-predictions" / protocol for protocol in PROTOCOLS}
    required = ("manifest.json", "evaluations.json", "aggregate.json", "selection.json")
    for protocol, root in roots.items():
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise ValueError(f"4D requires completed aggregate for {protocol}: {', '.join(missing)}")
    records_by_protocol = {
        protocol: json.loads((root / "evaluations.json").read_text(encoding="utf-8"))["runs"]
        for protocol, root in roots.items()
    }
    if any(len(records) != 255 for records in records_by_protocol.values()):
        raise ValueError("4D requires exactly 255 evaluation identities per protocol")
    reference = select_strongest_deep_baseline(records_by_protocol)
    baseline = str(reference["selected_model"])
    maps = {protocol: _prediction_map(root) for protocol, root in roots.items()}
    cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    def arrays(protocol: str, model: str, seed: int, split: str = "test"):
        identity = f"{protocol}/{model}/{seed}/{split}"
        key = (protocol, identity)
        if key not in cache:
            cache[key] = load_prediction_arrays(roots[protocol], maps[protocol][identity])
        return cache[key]

    daylight_nrmse = _improvement(
        _record_mean(records_by_protocol, RACE_PV.name, "test", "daylight_nrmse"),
        _record_mean(records_by_protocol, baseline, "test", "daylight_nrmse"),
    )
    qcrps = _improvement(
        _record_mean(records_by_protocol, RACE_PV.name, "test", "quantile_crps"),
        _record_mean(records_by_protocol, baseline, "test", "quantile_crps"),
    )

    ramp_values: dict[str, list[float]] = {
        "race_nrmse": [],
        "baseline_nrmse": [],
        "race_qcrps": [],
        "baseline_qcrps": [],
        "race_aucpr": [],
        "baseline_aucpr": [],
    }
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            race, base = arrays(protocol, RACE_PV.name, seed), arrays(protocol, baseline, seed)
            labels = np.asarray(race["state_labels"])
            ramp_mask = np.asarray(race["daylight"], dtype=bool) & np.isin(labels, (1, 2))
            for name, item in (("race", race), ("baseline", base)):
                median = item["quantiles"][..., list(QUANTILES).index(0.5)]
                ramp_values[f"{name}_nrmse"].append(
                    normalized_rmse(item["targets"], median, mask=ramp_mask)
                )
                ramp_values[f"{name}_qcrps"].append(
                    quantile_crps(item["targets"], item["quantiles"], QUANTILES, mask=ramp_mask)
                )
                ramp_values[f"{name}_aucpr"].append(
                    float(
                        binary_ramp_metrics(
                            labels,
                            item["state_probabilities"],
                            mask=item["daylight"],
                        )["auprc"]
                    )
                )
    ramp_nrmse = _improvement(
        float(np.mean(ramp_values["race_nrmse"])),
        float(np.mean(ramp_values["baseline_nrmse"])),
    )
    ramp_qcrps = _improvement(
        float(np.mean(ramp_values["race_qcrps"])),
        float(np.mean(ramp_values["baseline_qcrps"])),
    )
    ramp_aucpr_gain = float(
        np.mean(ramp_values["race_aucpr"]) - np.mean(ramp_values["baseline_aucpr"])
    )

    coverage_rows: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        race_records = {
            int(record["seed"]): record
            for record in records_by_protocol[protocol]
            if record["model"] == RACE_PV.name and record["split"] == "test"
        }
        for seed in SEEDS:
            item = arrays(protocol, RACE_PV.name, seed)
            record = race_records[seed]
            output = json.loads(
                (roots[protocol] / record["calibration_output_artifact"]["path"]).read_text(
                    encoding="utf-8"
                )
            )
            config = output["control_config"]
            lead = calibrated_interval_from_report(
                item["quantiles"],
                item["state_probabilities"],
                output["control_parameters"]["lead_only"],
                control="lead_only",
                daylight=item["daylight"],
                ramp_threshold=float(config["ramp_threshold"]),
            )
            state = calibrated_interval_from_report(
                item["quantiles"],
                item["state_probabilities"],
                output["control_parameters"]["state_lead"],
                control="state_lead",
                daylight=item["daylight"],
                ramp_threshold=float(config["ramp_threshold"]),
            )
            labels = np.asarray(item["state_labels"])
            daylight = np.asarray(item["daylight"], dtype=bool)
            stable, ramp = daylight & (labels == 0), daylight & np.isin(labels, (1, 2))
            coverage_rows.append(
                {
                    "protocol": protocol,
                    "seed": seed,
                    "stable_picp": prediction_interval_coverage(
                        item["targets"], state[..., 0], state[..., 1], mask=stable
                    ),
                    "ramp_picp": prediction_interval_coverage(
                        item["targets"], state[..., 0], state[..., 1], mask=ramp
                    ),
                    "lead_interval_score": interval_score(
                        item["targets"], lead[..., 0], lead[..., 1], coverage=0.90, mask=daylight
                    ),
                    "state_interval_score": interval_score(
                        item["targets"], state[..., 0], state[..., 1], coverage=0.90, mask=daylight
                    ),
                    "lead_width": float(np.mean((lead[..., 1] - lead[..., 0])[daylight])),
                    "state_width": float(np.mean((state[..., 1] - state[..., 0])[daylight])),
                }
            )
    stable_picp = float(np.mean([row["stable_picp"] for row in coverage_rows]))
    ramp_picp = float(np.mean([row["ramp_picp"] for row in coverage_rows]))
    lead_score = float(np.mean([row["lead_interval_score"] for row in coverage_rows]))
    state_score = float(np.mean([row["state_interval_score"] for row in coverage_rows]))
    interval_improvement = _improvement(state_score, lead_score)
    width_ratio = float(
        np.mean([row["state_width"] for row in coverage_rows])
        / np.mean([row["lead_width"] for row in coverage_rows])
    )

    site_skill_rows: list[dict[str, Any]] = []
    dm_rows: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        exemplar = arrays(protocol, RACE_PV.name, SEEDS[0])
        sites = sorted(set(map(str, exemplar["site_ids"][:, 0])))
        for site in sites:
            race_sse = base_sse = 0.0
            count = 0
            for seed in SEEDS:
                race, base = arrays(protocol, RACE_PV.name, seed), arrays(protocol, baseline, seed)
                mask = (race["site_ids"] == site) & race["daylight"]
                race_median = race["quantiles"][..., 3]
                base_median = base["quantiles"][..., 3]
                race_sse += float(np.sum(np.square(race_median[mask] - race["targets"][mask])))
                base_sse += float(np.sum(np.square(base_median[mask] - base["targets"][mask])))
                count += int(np.sum(mask))
            race_rmse, base_rmse = np.sqrt(race_sse / count), np.sqrt(base_sse / count)
            site_skill_rows.append(
                {
                    "protocol": protocol,
                    "site_id": site,
                    "reference_model": baseline,
                    "race_pv_nrmse": float(race_rmse),
                    "reference_nrmse": float(base_rmse),
                    "forecast_skill": forecast_skill(race_rmse, base_rmse),
                }
            )
            for lead in REPORT_LEADS:
                index = lead - 1
                seed_race, seed_base = [], []
                valid = None
                for seed in SEEDS:
                    race, base = arrays(protocol, RACE_PV.name, seed), arrays(protocol, baseline, seed)
                    site_mask = race["site_ids"][:, index] == site
                    valid = site_mask & race["daylight"][:, index]
                    seed_race.append(
                        np.where(
                            valid,
                            np.square(race["quantiles"][:, index, 3] - race["targets"][:, index]),
                            np.nan,
                        )
                    )
                    seed_base.append(
                        np.where(
                            valid,
                            np.square(base["quantiles"][:, index, 3] - base["targets"][:, index]),
                            np.nan,
                        )
                    )
                race_loss = _finite_mean(np.stack(seed_race), axis=0)
                base_loss = _finite_mean(np.stack(seed_base), axis=0)
                paired = np.isfinite(race_loss) & np.isfinite(base_loss)
                observation_count = int(np.sum(paired))
                if observation_count >= 2:
                    result = diebold_mariano_test(race_loss, base_loss, horizon=lead)
                    statistic = result.statistic
                    pvalue = result.pvalue
                    mean_difference = result.mean_loss_difference
                else:
                    statistic = float("nan")
                    pvalue = 1.0
                    mean_difference = (
                        float(np.mean(race_loss[paired] - base_loss[paired]))
                        if observation_count
                        else float("nan")
                    )
                dm_rows.append(
                    {
                        "protocol": protocol,
                        "site_id": site,
                        "lead": lead,
                        "reference_model": baseline,
                        "statistic": statistic,
                        "pvalue": pvalue,
                        "mean_loss_difference": mean_difference,
                        "n_observations": observation_count,
                    }
                )
    adjusted, rejected = benjamini_hochberg([row["pvalue"] for row in dm_rows])
    for row, qvalue, significant in zip(dm_rows, adjusted, rejected, strict=True):
        row["bh_adjusted_pvalue"] = float(qvalue)
        row["bh_rejected"] = bool(significant)
    significant_fraction = float(np.mean(rejected))
    near_flat_leads = 0
    lead_diagnostics = []
    for lead in REPORT_LEADS:
        selected = [row for row in dm_rows if row["lead"] == lead]
        share = float(np.mean([row["bh_rejected"] for row in selected]))
        mean_difference = float(np.mean([row["mean_loss_difference"] for row in selected]))
        close = share < 0.5 or mean_difference >= 0.0
        near_flat_leads += int(close)
        lead_diagnostics.append(
            {
                "lead": lead,
                "bh_significant_site_fraction": share,
                "mean_loss_difference": mean_difference,
                "near_flat_or_nonsignificant": close,
            }
        )

    neural_models = [profile.name for profile in BASELINES if profile.family == "neural"] + [
        RACE_PV.name
    ]
    seed_cv_rows = []
    for protocol, records in records_by_protocol.items():
        for model in neural_models:
            values = [
                float(record["metrics"]["quantile_crps"])
                for record in records
                if record["model"] == model and record["split"] == "test"
            ]
            seed_cv_rows.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "relative_sd": float(np.std(values, ddof=1) / np.mean(values)),
                }
            )
    neural_relative_sd = float(np.mean([row["relative_sd"] for row in seed_cv_rows]))

    correct_sum = entropy_sum = count = 0.0
    gate_sum = np.zeros(3, dtype=np.float64)
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            item = arrays(protocol, RACE_PV.name, seed)
            mask = np.asarray(item["daylight"], dtype=bool)
            labels = np.asarray(item["state_labels"])[mask].astype(int)
            probabilities = np.asarray(item["gates"], dtype=np.float64)[mask]
            correct_sum += float(np.sum(probabilities[np.arange(labels.size), labels]))
            entropy_sum += float(
                np.sum(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1))
            )
            gate_sum += probabilities.sum(axis=0)
            count += labels.size
    correct_gate = correct_sum / count
    normalized_entropy = entropy_sum / count / np.log(3.0)
    gate_shares = gate_sum / count
    maximum_share = float(np.max(gate_shares))

    race_validation = {
        "quantile_crps": _record_mean(records_by_protocol, RACE_PV.name, "validation", "quantile_crps"),
        "daylight_nrmse": _record_mean(records_by_protocol, RACE_PV.name, "validation", "daylight_nrmse"),
    }
    ablation_rows = []
    for profile in ABLATIONS:
        metrics = {
            "quantile_crps": _record_mean(records_by_protocol, profile.name, "validation", "quantile_crps"),
            "daylight_nrmse": _record_mean(records_by_protocol, profile.name, "validation", "daylight_nrmse"),
        }
        degraded = [name for name, value in metrics.items() if value > race_validation[name]]
        ablation_rows.append(
            {"model": profile.name, "metrics": metrics, "degraded_primary_metrics": degraded, "passed": bool(degraded)}
        )

    yulara_positive = sum(
        row["forecast_skill"] > 0 for row in site_skill_rows if row["protocol"] == "dkasc-yulara"
    )
    arizona_positive = sum(
        row["forecast_skill"] > 0 for row in site_skill_rows if row["protocol"] == "nrel-transfer"
    )
    values = {
        "daylight_nrmse_improvement_pct": daylight_nrmse,
        "mean_quantile_crps_improvement_pct": qcrps,
        "ramp_nrmse_improvement_pct": ramp_nrmse,
        "ramp_qcrps_improvement_pct": ramp_qcrps,
        "ramp_aucpr_absolute_gain": ramp_aucpr_gain,
        "stable_picp90": stable_picp,
        "ramp_picp90": ramp_picp,
        "state_lead_interval_score_improvement_pct": interval_improvement,
        "yulara_positive_skill_sites": float(yulara_positive),
        "arizona_positive_skill_sites": float(arizona_positive),
        "near_flat_report_leads": float(near_flat_leads),
        "bh_significant_site_lead_fraction": significant_fraction,
        "neural_seed_relative_sd": neural_relative_sd,
        "correct_expert_mean_gate": correct_gate,
        "normalized_gate_entropy": normalized_entropy,
        "maximum_expert_global_share": maximum_share,
    }
    checks = {name: acceptance_check(values[name], bounds) for name, bounds in ACCEPTANCE_RANGES.items()}
    additional_checks = {
        "state_lead_width_not_abnormal": {
            "value": width_ratio,
            "maximum": 1.10,
            "passed": bool(np.isfinite(width_ratio) and width_ratio <= 1.10),
        },
        "all_ablations_degrade_a_primary_metric": {
            "value": int(sum(row["passed"] for row in ablation_rows)),
            "required": len(ABLATIONS),
            "passed": all(row["passed"] for row in ablation_rows),
        },
        "held_out_not_used_for_reference_selection": {
            "value": reference["held_out_used_for_selection"],
            "required": False,
            "passed": reference["held_out_used_for_selection"] is False,
        },
    }
    passed = all(check["passed"] for check in checks.values()) and all(
        check["passed"] for check in additional_checks.values()
    )
    source_hashes = _source_hashes(roots)
    evaluation_report = {
        **dict(EVIDENCE_METADATA),
        "disclosure": DISCLOSURE,
        "protocols": list(PROTOCOLS),
        "reference_selection": reference,
        "performance": {
            "daylight_nrmse_improvement_pct": daylight_nrmse,
            "mean_quantile_crps_improvement_pct": qcrps,
            "ramp_nrmse_improvement_pct": ramp_nrmse,
            "ramp_qcrps_improvement_pct": ramp_qcrps,
            "ramp_aucpr_absolute_gain": ramp_aucpr_gain,
        },
        "calibration": {
            "rows": coverage_rows,
            "stable_picp90": stable_picp,
            "ramp_picp90": ramp_picp,
            "lead_only_interval_score": lead_score,
            "state_lead_interval_score": state_score,
            "state_lead_interval_score_improvement_pct": interval_improvement,
            "state_lead_to_lead_only_width_ratio": width_ratio,
        },
        "forecast_skill": site_skill_rows,
        "site_lead_dm_bh": dm_rows,
        "lead_diagnostics": lead_diagnostics,
        "seed_relative_sd": seed_cv_rows,
        "gate_diagnostics": {
            "correct_expert_mean_gate": correct_gate,
            "normalized_gate_entropy": normalized_entropy,
            "global_expert_shares": gate_shares,
            "maximum_expert_global_share": maximum_share,
        },
        "ablations": ablation_rows,
        "source_hashes": source_hashes,
    }
    evaluation_path = scenario_root / "03-evaluations" / "unified-evaluation.json"
    write_json(evaluation_path, evaluation_report)
    aggregate_report = {
        **dict(EVIDENCE_METADATA),
        "disclosure": DISCLOSURE,
        "protocols": list(PROTOCOLS),
        "reference_selection": reference,
        "values": values,
        "bootstrap_sources": {
            protocol: {
                "path": str((root / "aggregate.json").relative_to(scenario_root)),
                "sha256": source_hashes[protocol]["aggregate.json"],
                "replicates": sorted(
                    {
                        int(row["bootstrap_replicates"])
                        for row in json.loads((root / "aggregate.json").read_text())["rows"]
                    }
                ),
            }
            for protocol, root in roots.items()
        },
        "dm_bh_hypotheses": len(dm_rows),
        "forecast_skill_sites": len(site_skill_rows),
        "evaluation_path": str(evaluation_path.relative_to(scenario_root)),
        "evaluation_sha256": sha256_file(evaluation_path),
        "source_hashes": source_hashes,
    }
    aggregate_path = scenario_root / "04-aggregates" / "unified-aggregate.json"
    write_json(aggregate_path, aggregate_report)
    acceptance = {
        **dict(EVIDENCE_METADATA),
        "disclosure": DISCLOSURE,
        "status": "passed" if passed else "failed_numerical_acceptance",
        "passed": passed,
        "reference_selection": reference,
        "checks": checks,
        "additional_checks": additional_checks,
        "unmet_checks": [name for name, check in checks.items() if not check["passed"]]
        + [name for name, check in additional_checks.items() if not check["passed"]],
        "unified_aggregate_path": str(aggregate_path.relative_to(scenario_root)),
        "unified_aggregate_sha256": sha256_file(aggregate_path),
        "source_hashes": source_hashes,
    }
    acceptance_path = scenario_root / "04-aggregates" / "acceptance.json"
    write_json(acceptance_path, acceptance)
    artifact_index = {
        **dict(EVIDENCE_METADATA),
        "disclosure": DISCLOSURE,
        "stage": "4D_unified_evaluation",
        "outputs": {
            str(path.relative_to(scenario_root)): sha256_file(path)
            for path in (evaluation_path, aggregate_path, acceptance_path)
        },
        "sources": source_hashes,
        "artifact_index_roots": {
            protocol: artifact_root_digest(
                json.loads((root / "artifact-index.json").read_text(encoding="utf-8"))
            )
            for protocol, root in roots.items()
        },
    }
    index_path = scenario_root / "04-aggregates" / "unified-artifact-index.json"
    write_json(index_path, artifact_index)
    return {
        "protocols": list(PROTOCOLS),
        "passed": passed,
        "unmet_checks": acceptance["unmet_checks"],
        "reference_model": baseline,
        "evaluation_path": str(evaluation_path),
        "aggregate_path": str(aggregate_path),
        "acceptance_path": str(acceptance_path),
        "artifact_index_path": str(index_path),
    }


__all__ = [
    "ADDITIONAL_ACCEPTANCE_CHECKS",
    "CandidateAcceptance",
    "acceptance_check",
    "calibrated_interval_from_report",
    "evaluate_candidate_acceptance",
    "evaluate_candidate_acceptance_payload",
    "run_unified_4d",
    "select_strongest_deep_baseline",
]
