"""Fail-closed orchestration for isolated realism-v2 candidate bundles."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

from .constants import EVIDENCE_METADATA
from .realism import (
    RealismCandidate,
    ScenarioGenerationConfig,
    candidate_catalog,
    canonical_config_sha256,
)
from .scenario_artifacts import (
    DISCLOSURE,
    artifact_root_digest,
    canonical_bytes,
    generate,
    load_artifact_index,
    sha256_file,
    write_json,
)
from .scenario_candidate import (
    VALIDATION_PRECHECK_RANGES,
    CandidateScreeningReport,
    evaluate_validation_candidate,
    screen_candidates,
    validation_precheck_reasons,
)
from .scenario_evaluate import aggregate, evaluate
from .scenario_manifest import cohort_sha256, prepare, write_candidate_screening
from .scenario_unified import evaluate_candidate_acceptance, run_unified_4d

MINIMUM_FREE_RESERVE_BYTES = 15 * 1024**3
_REJECTED_R1_CATALOG_SHA256 = (
    "1bc5186d49d0c247523985b942f6c354e766364ffe2d739d76c47b16b4c31c62"
)
_REJECTED_R1_INDEX_SHA256 = (
    "a10f2b6291b9f77c4088bc775590d9187e771be710bb6d8a923367b781f1d741"
)
_REJECTED_R1_STATE_SHA256 = (
    "d8a6da27cb6c0570f4fb1a706d8aae67f989df95b77b0c940ea6b253961fbe60"
)
_REJECTED_R1_STATE_INDEX_SHA256 = (
    "4e5387df9d7b642a8ac96a98a1dc2b88473725403dd6ad12065911a55068c6f7"
)
_REJECTED_R1_ARTIFACTS = (
    {
        "archived_name": (
            "f248cb6556835aa5a9333886494492ac8ba4df4408cc0de2d048b3986d8851d3-"
            "realism-candidate-screening.json"
        ),
        "original_path": (
            "scenarios/q2-realism-v2/00-manifest/realism-candidate-screening.json"
        ),
        "sha256": "f248cb6556835aa5a9333886494492ac8ba4df4408cc0de2d048b3986d8851d3",
    },
    {
        "archived_name": (
            "30bfa0699ecaa9465712f8f60a51a88bada54e13b98da5f596298afe492ab916-"
            "realism-v2-precheck.json"
        ),
        "original_path": "scenarios/q2-realism-v2/09-logs/realism-v2-precheck.json",
        "sha256": "30bfa0699ecaa9465712f8f60a51a88bada54e13b98da5f596298afe492ab916",
    },
)
_REJECTED_R1_INDEX = {
    "accepted_count": 0,
    "artifacts": list(_REJECTED_R1_ARTIFACTS),
    "candidate_count": 32,
    "catalog_revision": "r1",
    "catalog_sha256": _REJECTED_R1_CATALOG_SHA256,
    "cohort_hashes": {
        "dkasc-yulara": "6c63d84dd690d8c1db026b4ef650bda848900e258fdbd913040f3bf908d4fdc4",
        "nrel-transfer": "9a021d659b3e1162e3e1de9790dbe58603d4fb558f022f618757f2d852bde46e",
    },
    "evidence_status": "hypothetical_oracle_conditioned",
    "failure_distribution": {
        "correct_expert_mean_gate:above_maximum": 32,
        "daylight_nrmse_improvement_pct:below_minimum": 30,
        "mean_quantile_crps_improvement_pct:below_minimum": 32,
        "neural_seed_relative_sd:below_minimum": 32,
        "normalized_gate_entropy:below_minimum": 8,
        "ramp_nrmse_improvement_pct:below_minimum": 32,
        "ramp_picp90:above_maximum": 8,
        "ramp_qcrps_improvement_pct:below_minimum": 32,
        "stable_picp90:below_minimum": 32,
    },
    "held_out_used": False,
    "oracle_conditioned": True,
    "registered_evidence_eligible": False,
    "rejected_count": 32,
    "schema": "realism_rejected_catalog_audit_v1",
    "screening_scope": "validation_precheck_only",
    "status": "rejected_validation_only",
}
_REJECTED_R2_CATALOG_SHA256 = (
    "5ed4fab8377d4847b7c3931d1f2ef37f8300f275827112c487f3ec9dbfd7b9f6"
)
_REJECTED_R2_INDEX_SHA256 = (
    "457cb3f106e670ed415c64e5799788e1a9de5295b5eed62f1700cdd00f01a564"
)
_REJECTED_R2_ARTIFACTS = (
    {
        "archived_name": (
            "52639d159a94aa65e61103dd9d3d234be13bc658870a274c60f68a6e6d72eb24-"
            "realism-candidate-screening.json"
        ),
        "original_path": (
            "scenarios/q2-realism-v2/00-manifest/realism-candidate-screening.json"
        ),
        "sha256": "52639d159a94aa65e61103dd9d3d234be13bc658870a274c60f68a6e6d72eb24",
    },
    {
        "archived_name": (
            "bce8bf78fffc8cd5f3bf767a484fb069d548d457ce6452347683f2c2bd508de9-"
            "realism-v2-precheck.json"
        ),
        "original_path": "scenarios/q2-realism-v2/09-logs/realism-v2-precheck.json",
        "sha256": "bce8bf78fffc8cd5f3bf767a484fb069d548d457ce6452347683f2c2bd508de9",
    },
    {
        "archived_name": (
            "eac6748331468855e1454879c00d18c9eb36f168bcc145737ca3ba2666531fb2-"
            "revision-state.json"
        ),
        "original_path": "scenarios/q2-realism-v2/09-logs/revision-state.json",
        "sha256": "eac6748331468855e1454879c00d18c9eb36f168bcc145737ca3ba2666531fb2",
    },
)
_REJECTED_R2_INDEX = {
    "accepted_count": 0,
    "artifacts": list(_REJECTED_R2_ARTIFACTS),
    "candidate_count": 32,
    "catalog_revision": "r2",
    "catalog_sha256": _REJECTED_R2_CATALOG_SHA256,
    "cohort_hashes": {
        "dkasc-yulara": "6c63d84dd690d8c1db026b4ef650bda848900e258fdbd913040f3bf908d4fdc4",
        "nrel-transfer": "9a021d659b3e1162e3e1de9790dbe58603d4fb558f022f618757f2d852bde46e",
    },
    "evidence_status": "hypothetical_oracle_conditioned",
    "failure_distribution": {
        "daylight_nrmse_improvement_pct:above_maximum": 32,
        "mean_quantile_crps_improvement_pct:above_maximum": 32,
        "ramp_nrmse_improvement_pct:above_maximum": 24,
        "ramp_picp90:above_maximum": 16,
        "ramp_qcrps_improvement_pct:above_maximum": 18,
        "stable_picp90:below_minimum": 30,
    },
    "held_out_used": False,
    "oracle_conditioned": True,
    "registered_evidence_eligible": False,
    "rejected_count": 32,
    "schema": "realism_rejected_catalog_audit_v1",
    "screening_scope": "validation_precheck_only",
    "status": "rejected_validation_only",
}
_REJECTED_R3_CATALOG_SHA256 = (
    "d22b575c32ec9bd440af824063e84c9e8398e9642d4daad4cab1c010c8779ab3"
)
_REJECTED_R3_INDEX_SHA256 = (
    "da9d1192753fc2139bc9d74ea165af29e9ed5076cce87d6e3afbdcff74387692"
)
_REJECTED_R3_ARTIFACTS = (
    {
        "archived_name": (
            "dd654a510805a2d6fd95fa64d21dd50f7cda8d729643431b8cee67970ae09e5a-"
            "realism-candidate-screening.json"
        ),
        "original_path": (
            "scenarios/q2-realism-v2/00-manifest/realism-candidate-screening.json"
        ),
        "sha256": "dd654a510805a2d6fd95fa64d21dd50f7cda8d729643431b8cee67970ae09e5a",
    },
    {
        "archived_name": (
            "454d244c1da44337890c2152c32ff83b4d2ce61fbe070077f4ba685b7edddadc-"
            "realism-v2-precheck.json"
        ),
        "original_path": "scenarios/q2-realism-v2/09-logs/realism-v2-precheck.json",
        "sha256": "454d244c1da44337890c2152c32ff83b4d2ce61fbe070077f4ba685b7edddadc",
    },
    {
        "archived_name": (
            "eceea8cc8a6adb84705b4c672adb88286efe310b8af6f953c63b3aa2dd901afd-"
            "revision-state.json"
        ),
        "original_path": "scenarios/q2-realism-v2/09-logs/revision-state.json",
        "sha256": "eceea8cc8a6adb84705b4c672adb88286efe310b8af6f953c63b3aa2dd901afd",
    },
)
_REJECTED_R3_INDEX = {
    "accepted_count": 0,
    "artifacts": list(_REJECTED_R3_ARTIFACTS),
    "candidate_count": 32,
    "catalog_revision": "r3",
    "catalog_sha256": _REJECTED_R3_CATALOG_SHA256,
    "cohort_hashes": {
        "dkasc-yulara": "6c63d84dd690d8c1db026b4ef650bda848900e258fdbd913040f3bf908d4fdc4",
        "nrel-transfer": "9a021d659b3e1162e3e1de9790dbe58603d4fb558f022f618757f2d852bde46e",
    },
    "evidence_status": "hypothetical_oracle_conditioned",
    "failure_distribution": {
        "daylight_nrmse_improvement_pct:above_maximum": 32,
        "mean_quantile_crps_improvement_pct:above_maximum": 32,
        "ramp_nrmse_improvement_pct:above_maximum": 6,
        "ramp_picp90:above_maximum": 4,
    },
    "held_out_used": False,
    "oracle_conditioned": True,
    "registered_evidence_eligible": False,
    "rejected_count": 32,
    "schema": "realism_rejected_catalog_audit_v1",
    "screening_scope": "validation_precheck_only",
    "status": "rejected_validation_only",
}
_REJECTED_CATALOG_SEALS = {
    "r1": (
        _REJECTED_R1_CATALOG_SHA256,
        _REJECTED_R1_INDEX_SHA256,
        _REJECTED_R1_ARTIFACTS,
        _REJECTED_R1_INDEX,
    ),
    "r2": (
        _REJECTED_R2_CATALOG_SHA256,
        _REJECTED_R2_INDEX_SHA256,
        _REJECTED_R2_ARTIFACTS,
        _REJECTED_R2_INDEX,
    ),
    "r3": (
        _REJECTED_R3_CATALOG_SHA256,
        _REJECTED_R3_INDEX_SHA256,
        _REJECTED_R3_ARTIFACTS,
        _REJECTED_R3_INDEX,
    ),
}
_REJECTED_R1_STATE_INDEX = {
    "artifact": {
        "archived_name": f"{_REJECTED_R1_STATE_SHA256}-revision-state.json",
        "embedded_path": (
            "scenarios/q2-realism-v2/09-logs/revision-state.json"
            "#superseded_revision_state.payload"
        ),
        "sha256": _REJECTED_R1_STATE_SHA256,
    },
    "artifact_count": 1,
    "catalog_revision": "r1",
    "catalog_sha256": _REJECTED_R1_CATALOG_SHA256,
    "cohort_hashes": dict(_REJECTED_R1_INDEX["cohort_hashes"]),
    "disclosure": DISCLOSURE,
    **dict(EVIDENCE_METADATA),
    "precheck_sha256": _REJECTED_R1_ARTIFACTS[1]["sha256"],
    "rejected_catalog_audit_index_sha256": _REJECTED_R1_INDEX_SHA256,
    "revision_state_sha256": _REJECTED_R1_STATE_SHA256,
    "schema": "realism_rejected_revision_state_audit_v1",
    "screening_sha256": _REJECTED_R1_ARTIFACTS[0]["sha256"],
    "source_hashes": {
        "manifest:dkasc_manifest": (
            "99f4a2641b39d9c6f92f2e4120af56f80ac27d41a859a2cc52006b4e0cf75071"
        ),
        "manifest:nrel_manifest": (
            "048a4c9f7adc7399ac3270cbd452be9fd458fe6201e88c1fd58dda4983275d6b"
        ),
        "manifest:nrel_selection_audit": (
            "f00e545df27ea5c31a086d2b370475d87fed8a1201cf4d263da62a257d87bef3"
        ),
        "manifest:orchestration_settings": (
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        ),
        "manifest:yulara_manifest": (
            "73ecc5e5a2e55f94b446a6411ed60e4fd28f3e9821ca0df344f1de5194171430"
        ),
    },
    "state": "validation_screened",
    "status": "sealed_rejected_predecessor_state",
}
_CATALOG_REVISION_REGISTRY = (
    {"revision": "r1", "catalog_sha256": _REJECTED_R1_CATALOG_SHA256},
    {"revision": "r2", "catalog_sha256": _REJECTED_R2_CATALOG_SHA256},
    {
        "revision": "r3",
        "catalog_sha256": _REJECTED_R3_CATALOG_SHA256,
    },
    {
        "revision": "r4",
        "catalog_sha256": (
            "4a117d029f1a5e72f7b356bdbb2fb877e356e80d8817878f1f35a96893dd7fa1"
        ),
    },
)


def _revision_1_catalog_sha256() -> str:
    payload = []
    for row in candidate_catalog(revision="r1"):
        config = asdict(row.config)
        config.pop("revision")
        config.pop("ramp_error_multiplier")
        payload.append(
            {
                "candidate_id": row.candidate_id,
                "config": config,
                "config_sha256": row.config_sha256,
                "scenario_seed": row.scenario_seed,
            }
        )
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _catalog_sha256(revision: str) -> str:
    if revision == "r1":
        return _revision_1_catalog_sha256()
    payload = [
        {
            "candidate_id": row.candidate_id,
            "config": asdict(row.config),
            "config_sha256": row.config_sha256,
            "scenario_seed": row.scenario_seed,
        }
        for row in candidate_catalog(revision=revision)
    ]
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def r5_catalog_sha256() -> str:
    """Return the registered r5 catalog digest after verifying its live rows."""

    expected = "a0192594bb5bfe2ae8ca0c21940e024c9005d01594e5d18a51d1cc4bc1bfd856"
    if _catalog_sha256("r5") != expected:
        raise ValueError("candidate catalog does not match the fixed r5 hash")
    return expected


def r6_catalog_sha256() -> str:
    """Return the fixed r6 catalog digest after verifying its live rows."""

    expected = "975777378945a7cd8b0dbaad32d3642fe79785e09f091b5a1af7bbde0946435e"
    if _catalog_sha256("r6") != expected:
        raise ValueError("candidate catalog does not match the fixed r6 hash")
    return expected


def r5_precheck_ranges_payload() -> dict[str, dict[str, float]]:
    """Return a detached canonical payload for the eleven design/confirmation ranges."""

    return {
        name: {"minimum": float(bounds[0]), "maximum": float(bounds[1])}
        for name, bounds in sorted(VALIDATION_PRECHECK_RANGES.items())
    }


def r5_precheck_ranges_sha256() -> str:
    """Hash the exact eleven-range definition used at both r5 stages."""

    return hashlib.sha256(canonical_bytes(r5_precheck_ranges_payload())).hexdigest()


def r5_validation_precheck_reasons(metrics: dict[str, float]) -> tuple[str, ...]:
    """Apply the existing eleven-range fail-closed validator for r5 orchestration."""

    return validation_precheck_reasons(metrics)


def r6_design_precheck_reasons(metrics: dict[str, float]) -> tuple[str, ...]:
    """Apply unchanged ranges plus R6's Design-only ramp-AUPRC margin."""

    reasons = list(validation_precheck_reasons(metrics))
    value = float(metrics.get("ramp_aucpr_absolute_gain", float("nan")))
    if math.isfinite(value) and 0.04 <= value < 0.048:
        reasons.append("ramp_aucpr_absolute_gain:below_r6_design_margin")
    return tuple(reasons)


def r6_preregistration_payload() -> dict[str, object]:
    """Return the detached provenance contract for the R6 Design catalog."""

    r5_catalog_sha256()
    source = next(
        row
        for row in candidate_catalog("r5")
        if row.candidate_id == "rv2-r5-009-20260821"
    )
    if source.config_sha256 != "2ee6cd88dba5e6419ebda54618bd168b846323fcdeb3232d100e7897df4a2dd6":
        raise ValueError("frozen r5 source candidate does not match its fixed hash")
    return {
        "schema": "realism_r6_preregistration_v1",
        "revision": "r6",
        "predecessor_revision": "r5",
        "catalog_sha256": r6_catalog_sha256(),
        "range_definition_sha256": r5_precheck_ranges_sha256(),
        "design_margin": {"metric": "ramp_aucpr_absolute_gain", "minimum": 0.048},
        "frozen_r5_source_candidate": {
            "candidate_id": source.candidate_id,
            "config_sha256": source.config_sha256,
        },
        "confirmation_cohort_reuse": False,
    }


def r6_preregistration_sha256() -> str:
    """Hash and verify the complete detached R6 preregistration contract."""

    expected = "2c17c2cc527528c024e7944559bc811c9757277c587d24d744312b0ebe5350d1"
    actual = hashlib.sha256(canonical_bytes(r6_preregistration_payload())).hexdigest()
    if actual != expected:
        raise ValueError("r6 preregistration payload does not match the fixed hash")
    return expected


_FIXED_R6B_PREREGISTRATION = {
    "candidate_universe_sha256": "0330e0ffc2e7a0445fb72d75d045034668cff7113170cfd6e5493275c1a69fa4",
    "candidate_universe_site_count": 20,
    "catalog_revision": "r6",
    "catalog_sha256": "975777378945a7cd8b0dbaad32d3642fe79785e09f091b5a1af7bbde0946435e",
    "dkasc_blocks_per_site": 1,
    "dkasc_site_count": 12,
    "embargo_days": 14,
    "held_out_allowed": False,
    "oklahoma_blocks_per_site": 1,
    "oklahoma_site_count": 15,
    "predecessor_revision": "r6",
    "prior_coordinate_count": 75,
    "prior_coordinate_sha256": "58073a22e4d92c77be0008c3de707d9fe06c7c1cadfafdde9bbeb46121361e98",
    "r5_confirmation_decision_sha256": "aed8c8b9759ee1f7a04a208201fcf80cbe3852a93df9141a5ae99c9ace48993d",
    "r5_confirmation_protocol_sha256": "8f0299e041510502b0b65173b86e8c08ea9679e2c0abfb97920a9f9f4e86d85f",
    "r5_confirmation_state_sha256": "603cbb6e6ee0f70e90789fdc21c1c13ad0ed847bc6b57fe63c2a98d6931e3062",
    "range_definition_sha256": "8aa7adfe3b001f6d7dbc34fc09f9e4fd423591d91e867605373d9d13f6503d4c",
    "revision": "r6b",
    "schema": "realism_r6b_preregistration_v1",
    "selected_block_count": 27,
    "selected_blocks_authority_sha256": "d1496e86d93f3073b05b053234e4078f5c85f9903cc7e7fd3325fbfa1f380dd6",
    "selected_site_count": 27,
    "selection_seed": 20260818,
    "task_11_report_sha256": "13ecf0304b4ec5795ba92f019309a1b92b53db2e1a3d9a463447e500dcb48604",
    "task_13_report_sha256": "21dee2f54473d873c7d5e80c90bc66cc823a6f41a9f1e72998db81c3054ae892",
}
_R6B_PREREGISTRATION_SHA256 = "d06310a7f6a788da487ec8a44fbcb8f378ddaacb0538d50bc93216fefb53a28e"


def r6b_preregistration_payload() -> dict[str, object]:
    """Return a detached copy of the versioned R6b provenance contract."""

    return json.loads(json.dumps(_FIXED_R6B_PREREGISTRATION))


def r6b_preregistration_sha256() -> str:
    """Recompute the immutable R6b preregistration seal."""

    if r6_catalog_sha256() != _FIXED_R6B_PREREGISTRATION["catalog_sha256"]:
        raise ValueError("r6b preregistration catalog binding is invalid")
    if r5_precheck_ranges_sha256() != _FIXED_R6B_PREREGISTRATION["range_definition_sha256"]:
        raise ValueError("r6b preregistration range binding is invalid")
    actual = hashlib.sha256(canonical_bytes(r6b_preregistration_payload())).hexdigest()
    if actual != _R6B_PREREGISTRATION_SHA256:
        raise ValueError("r6b preregistration payload does not match its fixed hash")
    return actual


def _registered_revision(revision: str) -> tuple[str | None, str]:
    """Return the explicit predecessor and pinned hash for one registry entry."""

    revisions = tuple(str(row["revision"]) for row in _CATALOG_REVISION_REGISTRY)
    if len(revisions) != len(set(revisions)):
        raise ValueError("catalog revision registry contains duplicates")
    try:
        position = revisions.index(revision)
    except ValueError as error:
        raise ValueError("candidate catalog revision is not registered") from error
    expected_hash = str(_CATALOG_REVISION_REGISTRY[position]["catalog_sha256"])
    if _catalog_sha256(revision) != expected_hash:
        raise ValueError("candidate catalog does not match the registered hash")
    predecessor = revisions[position - 1] if position else None
    return predecessor, expected_hash


def _verify_failed_catalog_evidence(
    archive: Path,
    artifacts: Sequence[Mapping[str, str]],
    index: Mapping[str, Any],
    *,
    revision: str,
) -> None:
    by_original = {str(row["original_path"]): row for row in artifacts}
    prefix = "scenarios/q2-realism-v2/"

    def payload(relative: str) -> dict[str, Any]:
        row = by_original.get(prefix + relative)
        if row is None:
            raise ValueError("rejected catalog archive evidence set is incomplete")
        try:
            value = json.loads((archive / row["archived_name"]).read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("rejected catalog archive evidence is malformed") from error
        if not isinstance(value, dict):
            raise ValueError("rejected catalog archive evidence is malformed")
        return value

    screening = payload("00-manifest/realism-candidate-screening.json")
    precheck = payload("09-logs/realism-v2-precheck.json")
    candidates = screening.get("candidates")
    expected_ids = [row.candidate_id for row in candidate_catalog(revision=revision)]
    candidate_ids = (
        [str(row.get("candidate_id")) for row in candidates]
        if isinstance(candidates, list) and all(isinstance(row, dict) for row in candidates)
        else []
    )
    valid_screen = (
        screening.get("status") == "failed"
        and screening.get("selected_candidate_id") is None
        and screening.get("candidate_count") == len(expected_ids)
        and screening.get("accepted_count") == 0
        and screening.get("rejected_count") == len(expected_ids)
        and candidate_ids == expected_ids
        and all(row.get("status") == "rejected" for row in candidates or ())
        and all(
            row.get("cohort_hashes") == index.get("cohort_hashes")
            and isinstance(row.get("selected_validation_reference"), dict)
            and row["selected_validation_reference"].get("selection_split")
            == "validation"
            and row["selected_validation_reference"].get(
                "held_out_used_for_selection"
            )
            is False
            for row in candidates or ()
        )
    )
    screening_row = by_original[prefix + "00-manifest/realism-candidate-screening.json"]
    valid_precheck = (
        precheck.get("status") == "failed_validation_screen"
        and precheck.get("state") == "validation_screened"
        and precheck.get("accepted_candidate_id") is None
        and precheck.get("evaluated_candidate_ids") == expected_ids
        and precheck.get("rejected_candidate_ids") == expected_ids
        and precheck.get("screening_sha256") == screening_row["sha256"]
    )
    valid_index = (
        index.get("catalog_revision") == revision
        and index.get("candidate_count") == len(expected_ids)
        and index.get("rejected_count") == len(expected_ids)
        and index.get("accepted_count") == 0
        and index.get("held_out_used") is False
        and index.get("screening_scope") == "validation_precheck_only"
        and index.get("status") == "rejected_validation_only"
    )
    state_row = by_original.get(prefix + "09-logs/revision-state.json")
    valid_state = True
    if state_row is not None:
        state = payload("09-logs/revision-state.json")
        history = state.get("history")
        screened = (
            history[-1]
            if isinstance(history, list) and len(history) == 2 and isinstance(history[-1], dict)
            else {}
        )
        state_sources = state.get("source_hashes")
        screened_sources = screened.get("source_hashes")
        valid_state = (
            state.get("schema") == "realism_revision_state_v1"
            and all(state.get(key) == value for key, value in EVIDENCE_METADATA.items())
            and state.get("disclosure") == DISCLOSURE
            and state.get("state") == "validation_screened"
            and isinstance(state_sources, dict)
            and screened.get("sequence") == 2
            and screened.get("from_state") == "created"
            and screened.get("state") == "validation_screened"
            and screened.get("details")
            == {
                "screening_report_sha256": screening_row["sha256"],
                "screening_status": "failed",
            }
            and isinstance(screened_sources, dict)
            and screened_sources
            == {**state_sources, "screening_report_sha256": screening_row["sha256"]}
        )
    if not (valid_screen and valid_precheck and valid_index and valid_state):
        raise ValueError("rejected catalog archive is not a complete failed catalog")


def _verify_rejected_catalog_archive(
    work_root: Path,
    *,
    revision: str,
    verify_originals: bool,
    replaced_originals: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    root = Path(work_root).expanduser().resolve()
    try:
        catalog_sha256, index_sha256, artifacts, expected_index = (
            _REJECTED_CATALOG_SEALS[revision]
        )
    except KeyError as error:
        raise ValueError("rejected catalog archive revision is unsupported") from error
    archive = root / "09-logs" / "rejected-catalogs" / catalog_sha256
    if not archive.is_dir():
        raise ValueError("rejected catalog archive directory is missing")
    expected_names = {
        "audit-index.json",
        *(row["archived_name"] for row in artifacts),
    }
    try:
        actual_names = {path.name for path in archive.iterdir()}
    except OSError as error:
        raise ValueError("rejected catalog archive cannot be enumerated") from error
    if actual_names != expected_names:
        raise ValueError("rejected catalog archive entry set mismatch")
    for name in expected_names:
        path = archive / name
        if not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise ValueError("rejected catalog archive file seal mismatch")

    index_path = archive / "audit-index.json"
    try:
        index_bytes = index_path.read_bytes()
        index = json.loads(index_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("rejected catalog archive index is malformed") from error
    if hashlib.sha256(index_bytes).hexdigest() != index_sha256:
        raise ValueError("rejected catalog archive index hash mismatch")
    if not isinstance(index, dict) or index_bytes != canonical_bytes(index) + b"\n":
        raise ValueError("rejected catalog archive index is not canonical")
    if index != expected_index:
        raise ValueError("rejected catalog archive index binding mismatch")
    _predecessor, registered_hash = _registered_revision(revision)
    if registered_hash != index["catalog_sha256"]:
        raise ValueError("rejected catalog archive catalog binding mismatch")

    _verify_failed_catalog_evidence(
        archive, artifacts, index, revision=revision
    )

    prefix = "scenarios/q2-realism-v2/"
    for row in artifacts:
        archived = archive / row["archived_name"]
        if sha256_file(archived) != row["sha256"]:
            raise ValueError("rejected catalog archive copied artifact hash mismatch")
        if verify_originals and row["original_path"] not in replaced_originals:
            original_path = row["original_path"]
            if not original_path.startswith(prefix):
                raise ValueError("rejected catalog archive original path binding mismatch")
            original = root / original_path.removeprefix(prefix)
            if not original.is_file() or sha256_file(original) != row["sha256"]:
                raise ValueError("rejected catalog archive original artifact hash mismatch")
            if original.read_bytes() != archived.read_bytes():
                raise ValueError("rejected catalog archive copy differs from original")
    return {
        "audit_index_sha256": index_sha256,
        "catalog_sha256": catalog_sha256,
        "verified_artifact_count": len(artifacts),
    }


def verify_rejected_catalog_archive(
    work_root: Path, *, revision: str = "r1", verify_originals: bool = True
) -> dict[str, Any]:
    """Fail closed unless one rejected-catalog logical seal is byte-exact."""

    return _verify_rejected_catalog_archive(
        work_root, revision=revision, verify_originals=verify_originals
    )


def verify_rejected_revision_state_archive(
    work_root: Path, *, revision: str = "r1", verify_original: bool = True
) -> dict[str, Any]:
    """Verify the adjacent content-addressed state seal for legacy r1."""

    if revision != "r1":
        raise ValueError("separate rejected revision-state archive is unsupported")
    root = Path(work_root).expanduser().resolve()
    _verify_rejected_catalog_archive(root, revision="r1", verify_originals=False)
    archive = (
        root
        / "09-logs/rejected-revision-states"
        / _REJECTED_R1_STATE_SHA256
    )
    state_name = f"{_REJECTED_R1_STATE_SHA256}-revision-state.json"
    expected_names = {"audit-index.json", state_name}
    if not archive.is_dir():
        raise ValueError("rejected revision-state archive directory is missing")
    try:
        actual_names = {path.name for path in archive.iterdir()}
    except OSError as error:
        raise ValueError("rejected revision-state archive cannot be enumerated") from error
    if actual_names != expected_names:
        raise ValueError("rejected revision-state archive entry set mismatch")
    for name in expected_names:
        path = archive / name
        if not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise ValueError("rejected revision-state archive file seal mismatch")
    index_path = archive / "audit-index.json"
    state_path = archive / state_name
    try:
        index_raw = index_path.read_bytes()
        state_raw = state_path.read_bytes()
        index = json.loads(index_raw)
        state = json.loads(state_raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("rejected revision-state archive is malformed") from error
    if (
        hashlib.sha256(index_raw).hexdigest() != _REJECTED_R1_STATE_INDEX_SHA256
        or not isinstance(index, dict)
        or index_raw != canonical_bytes(index) + b"\n"
        or index != _REJECTED_R1_STATE_INDEX
    ):
        raise ValueError("rejected revision-state archive index binding mismatch")
    if (
        hashlib.sha256(state_raw).hexdigest() != _REJECTED_R1_STATE_SHA256
        or not isinstance(state, dict)
        or state_raw != canonical_bytes(state) + b"\n"
        or not _valid_rejected_revision_state_payload(
            state,
            revision="r1",
            source_hashes=index["source_hashes"],
            screening_sha256=index["screening_sha256"],
        )
    ):
        raise ValueError("rejected revision-state archive state binding mismatch")
    if verify_original:
        live_path = root / "09-logs/revision-state.json"
        try:
            live_raw = live_path.read_bytes()
            live = json.loads(live_raw)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("rejected revision-state original is malformed") from error
        if live_raw == state_raw:
            bound_raw = live_raw
        else:
            superseded = live.get("superseded_revision_state", {})
            embedded = superseded.get("payload") if isinstance(superseded, dict) else None
            bound_raw = canonical_bytes(embedded) + b"\n" if isinstance(embedded, dict) else b""
            if (
                superseded.get("catalog_revision") != "r1"
                or superseded.get("revision_state_sha256")
                != _REJECTED_R1_STATE_SHA256
            ):
                raise ValueError("rejected revision-state embedded binding mismatch")
        if bound_raw != state_raw:
            raise ValueError("rejected revision-state original differs from archive")
    return {
        "catalog_revision": "r1",
        "revision_state_sha256": _REJECTED_R1_STATE_SHA256,
        "verified_artifact_count": 1,
    }


REVISION_STATES = (
    "created",
    "validation_screened",
    "candidate_materializing",
    "candidate_evaluated",
    "candidate_rejected",
    "candidate_accepted",
    "verified",
    "promotion_eligible",
)
_TRANSITIONS = {
    "created": ("validation_screened",),
    "validation_screened": ("candidate_materializing",),
    "candidate_materializing": ("candidate_evaluated",),
    "candidate_evaluated": ("candidate_rejected", "candidate_accepted"),
    "candidate_rejected": ("candidate_materializing",),
    "candidate_accepted": ("verified",),
    "verified": ("promotion_eligible",),
    "promotion_eligible": (),
}
_PROMOTED_LAYOUT = (
    "01-prepare",
    "02-predictions",
    "03-evaluations",
    "04-aggregate",
    "04-aggregates",
)
SOURCE_ROOT_MANDATORY_ARTIFACTS = (
    "00-manifest/compact-shared-v2-validation.json",
    "00-manifest/data-source-contract.json",
    "00-manifest/dkasc-yulara-protocol.json",
    "00-manifest/nrel-transfer-protocol.json",
    "00-manifest/preflight-4a.json",
    "02-predictions/dkasc-yulara/artifact-index.json",
    "02-predictions/dkasc-yulara/manifest.json",
    "02-predictions/nrel-transfer/artifact-index.json",
    "02-predictions/nrel-transfer/manifest.json",
    "04-aggregates/acceptance.json",
    "04-aggregates/unified-aggregate.json",
    "04-aggregates/unified-artifact-index.json",
    "09-logs/phase-4b-report.json",
    "09-logs/phase-4c-report.json",
    "09-logs/phase-4d-report.json",
    "09-logs/preflight-4a-artifacts.json",
)
_DECLARED_SOURCE_PATH_KEYS = {
    "dkasc_manifest",
    "yulara_manifest",
    "nrel_manifest",
    "nrel_selection_audit",
}


@dataclass(frozen=True)
class RevisionRunReport:
    """Small orchestration result; scientific metrics stay in hashed audit files."""

    status: str
    state: str
    screening_sha256: str | None = None
    accepted_candidate_id: str | None = None
    evaluated_candidate_ids: tuple[str, ...] = ()
    rejected_candidate_ids: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            **dict(EVIDENCE_METADATA),
            "disclosure": DISCLOSURE,
            **asdict(self),
            "evaluated_candidate_ids": list(self.evaluated_candidate_ids),
            "rejected_candidate_ids": list(self.rejected_candidate_ids),
        }


def _rollover_state_payload(
    source_hashes: Mapping[str, str],
    predecessor: Mapping[str, Any],
    *,
    catalog_revision: str,
    catalog_sha256: str,
) -> dict[str, Any]:
    normalized = {str(key): str(value) for key, value in sorted(source_hashes.items())}
    bound_predecessor = dict(predecessor)
    rollover = {
        "from_revision": str(bound_predecessor["catalog_revision"]),
        "to_revision": catalog_revision,
        "predecessor": bound_predecessor,
        "new_catalog_sha256": catalog_sha256,
    }
    return {
        **dict(EVIDENCE_METADATA),
        "disclosure": DISCLOSURE,
        "schema": "realism_revision_state_v1",
        "state": "created",
        "catalog_revision": catalog_revision,
        "catalog_sha256": catalog_sha256,
        "source_hashes": normalized,
        "history": [
            {
                "sequence": 1,
                "from_state": None,
                "state": "created",
                "source_hashes": normalized,
                "details": {"catalog_rollover": rollover},
            }
        ],
        "superseded_revision_state": bound_predecessor,
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_hash_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(key, str) and _is_sha256(digest) for key, digest in value.items()
    )


def _valid_rejected_revision_state_payload(
    payload: Any,
    *,
    revision: str,
    source_hashes: Mapping[str, str],
    screening_sha256: str,
) -> bool:
    if not isinstance(payload, Mapping):
        return False
    base_keys = {
        *EVIDENCE_METADATA,
        "disclosure",
        "schema",
        "state",
        "source_hashes",
        "history",
    }
    modern_rollover = revision not in {"r1", "r2"}
    expected_keys = base_keys | (
        {"superseded_revision_state", "catalog_revision", "catalog_sha256"}
        if modern_rollover
        else ({"superseded_revision_state"} if revision == "r2" else set())
    )
    history = payload.get("history")
    if (
        set(payload) != expected_keys
        or payload.get("schema") != "realism_revision_state_v1"
        or any(payload.get(key) != value for key, value in EVIDENCE_METADATA.items())
        or payload.get("disclosure") != DISCLOSURE
        or payload.get("state") != "validation_screened"
        or payload.get("source_hashes") != source_hashes
        or not _is_hash_mapping(source_hashes)
        or not isinstance(history, list)
        or len(history) != 2
        or not all(isinstance(entry, Mapping) for entry in history)
    ):
        return False
    created, screened = history
    entry_keys = {"sequence", "from_state", "state", "source_hashes", "details"}
    if (
        set(created) != entry_keys
        or created.get("sequence") != 1
        or created.get("from_state") is not None
        or created.get("state") != "created"
        or created.get("source_hashes") != source_hashes
        or set(screened) != entry_keys
        or screened
        != {
            "sequence": 2,
            "from_state": "created",
            "state": "validation_screened",
            "source_hashes": {
                **source_hashes,
                "screening_report_sha256": screening_sha256,
            },
            "details": {
                "screening_report_sha256": screening_sha256,
                "screening_status": "failed",
            },
        }
    ):
        return False
    if modern_rollover:
        details = created.get("details")
        rollover = details.get("catalog_rollover") if isinstance(details, Mapping) else None
        try:
            predecessor_revision, catalog_sha256 = _registered_revision(revision)
        except ValueError:
            return False
        return (
            _valid_catalog_rollover_details(details)
            and isinstance(rollover, Mapping)
            and payload.get("catalog_revision") == revision
            and payload.get("catalog_sha256") == catalog_sha256
            and rollover.get("from_revision") == predecessor_revision
            and rollover.get("to_revision") == revision
            and rollover.get("new_catalog_sha256") == catalog_sha256
            and payload.get("superseded_revision_state")
            == rollover.get("predecessor")
        )
    if created.get("details") != {}:
        return False
    if revision == "r1":
        return True
    legacy = payload.get("superseded_revision_state")
    if not isinstance(legacy, Mapping) or set(legacy) != {
        "catalog_revision",
        "catalog_sha256",
        "rejected_catalog_audit_index_sha256",
        "revision_state_sha256",
        "payload",
    }:
        return False
    legacy_payload = legacy["payload"]
    return (
        legacy["catalog_revision"] == "r1"
        and legacy["catalog_sha256"] == _REJECTED_R1_CATALOG_SHA256
        and legacy["rejected_catalog_audit_index_sha256"]
        == _REJECTED_R1_INDEX_SHA256
        and legacy["revision_state_sha256"] == _REJECTED_R1_STATE_SHA256
        and isinstance(legacy_payload, Mapping)
        and hashlib.sha256(canonical_bytes(legacy_payload) + b"\n").hexdigest()
        == _REJECTED_R1_STATE_SHA256
        and _valid_rejected_revision_state_payload(
            legacy_payload,
            revision="r1",
            source_hashes=source_hashes,
            screening_sha256=_REJECTED_R1_ARTIFACTS[0]["sha256"],
        )
    )


def _valid_catalog_rollover_details(details: Any) -> bool:
    if not isinstance(details, Mapping) or set(details) != {"catalog_rollover"}:
        return False
    rollover = details["catalog_rollover"]
    if not isinstance(rollover, Mapping) or set(rollover) != {
        "from_revision",
        "to_revision",
        "predecessor",
        "new_catalog_sha256",
    }:
        return False
    predecessor = rollover["predecessor"]
    if not isinstance(predecessor, Mapping) or set(predecessor) != {
        "catalog_revision",
        "catalog_sha256",
        "rejected_catalog_audit_index_sha256",
        "revision_state_seal_index_sha256",
        "screening_sha256",
        "precheck_sha256",
        "revision_state_sha256",
        "source_hashes",
        "cohort_hashes",
        "payload",
    }:
        return False
    try:
        expected_predecessor, expected_catalog_sha256 = _registered_revision(
            str(rollover["to_revision"])
        )
        _earlier, predecessor_catalog_sha256 = _registered_revision(
            str(rollover["from_revision"])
        )
    except ValueError:
        return False
    predecessor_revision = str(rollover["from_revision"])
    try:
        sealed_catalog_sha256, sealed_index_sha256, artifacts, sealed_index = (
            _REJECTED_CATALOG_SEALS[predecessor_revision]
        )
    except KeyError:
        return False
    by_original = {str(row["original_path"]): row for row in artifacts}
    prefix = "scenarios/q2-realism-v2/"
    screening_sha256 = by_original.get(
        prefix + "00-manifest/realism-candidate-screening.json", {}
    ).get("sha256")
    precheck_sha256 = by_original.get(
        prefix + "09-logs/realism-v2-precheck.json", {}
    ).get("sha256")
    state_row = by_original.get(prefix + "09-logs/revision-state.json")
    sealed_state_sha256 = (
        state_row["sha256"] if state_row is not None else _REJECTED_R1_STATE_SHA256
    )
    state_seal_index_sha256 = (
        _REJECTED_R1_STATE_INDEX_SHA256
        if predecessor_revision == "r1"
        else sealed_index_sha256
    )
    source_hashes = predecessor["source_hashes"]
    cohort_hashes = predecessor["cohort_hashes"]
    payload = predecessor["payload"]
    return (
        expected_predecessor == rollover["from_revision"]
        and rollover["new_catalog_sha256"] == expected_catalog_sha256
        and predecessor["catalog_revision"] == rollover["from_revision"]
        and predecessor["catalog_sha256"] == predecessor_catalog_sha256
        == sealed_catalog_sha256
        and predecessor["rejected_catalog_audit_index_sha256"]
        == sealed_index_sha256
        and predecessor["revision_state_seal_index_sha256"]
        == state_seal_index_sha256
        and predecessor["screening_sha256"] == screening_sha256
        and predecessor["precheck_sha256"] == precheck_sha256
        and predecessor["revision_state_sha256"] == sealed_state_sha256
        and source_hashes == payload.get("source_hashes")
        and cohort_hashes == sealed_index.get("cohort_hashes")
        and _is_hash_mapping(source_hashes)
        and _is_hash_mapping(cohort_hashes)
        and hashlib.sha256(canonical_bytes(payload) + b"\n").hexdigest()
        == predecessor["revision_state_sha256"]
        and _valid_rejected_revision_state_payload(
            payload,
            revision=predecessor_revision,
            source_hashes=source_hashes,
            screening_sha256=str(screening_sha256),
        )
    )


class RevisionStateMachine:
    """Persist and validate every realism revision transition atomically."""

    def __init__(
        self,
        root: Path,
        source_hashes: Mapping[str, str],
        *,
        superseded_revision_state: Mapping[str, Any] | None = None,
        catalog_revision: str | None = None,
        catalog_sha256: str | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.path = self.root / "09-logs" / "revision-state.json"
        normalized = {str(key): str(value) for key, value in sorted(source_hashes.items())}
        if superseded_revision_state is not None:
            if catalog_revision is None or catalog_sha256 is None:
                raise ValueError("catalog rollover is missing the new catalog binding")
            self.payload = _rollover_state_payload(
                normalized,
                superseded_revision_state,
                catalog_revision=catalog_revision,
                catalog_sha256=catalog_sha256,
            )
            if not _valid_catalog_rollover_details(self.payload["history"][0]["details"]):
                raise ValueError("catalog rollover binding is malformed")
            write_json(self.path, self.payload)
            return
        if self.path.is_file():
            self.payload = json.loads(self.path.read_text(encoding="utf-8"))
            if self.payload.get("state") not in REVISION_STATES:
                raise ValueError("revision state file contains an unknown state")
            persisted = dict(self.payload.get("source_hashes", {}))
            changed = [
                key
                for key, value in normalized.items()
                if key in persisted and persisted[key] != value
            ]
            if changed:
                raise ValueError(
                    "source snapshot mismatch: revision source hash changed: "
                    + ", ".join(sorted(changed))
                )
            persisted.update(normalized)
            self.payload["source_hashes"] = persisted
            return
        self.payload = {
            **dict(EVIDENCE_METADATA),
            "disclosure": DISCLOSURE,
            "schema": "realism_revision_state_v1",
            "state": "created",
            "source_hashes": normalized,
            "history": [
                {
                    "sequence": 1,
                    "from_state": None,
                    "state": "created",
                    "source_hashes": normalized,
                    "details": {},
                }
            ],
        }
        write_json(self.path, self.payload)

    @property
    def state(self) -> str:
        return str(self.payload["state"])

    def transition(self, target: str, **details: Any) -> None:
        if target not in REVISION_STATES or target not in _TRANSITIONS[self.state]:
            raise ValueError(
                f"incompatible revision transition: {self.state} -> {target}"
            )
        bound_sources = details.get("materialization_source_hashes")
        transition_hashes = (
            dict(bound_sources)
            if isinstance(bound_sources, Mapping)
            else dict(self.payload.get("source_hashes", {}))
        )
        transition_hashes.update(
            {
                key: str(value)
                for key, value in sorted(details.items())
                if key.endswith("_sha256") and value is not None
            }
        )
        entry = {
            "sequence": len(self.payload["history"]) + 1,
            "from_state": self.state,
            "state": target,
            "source_hashes": transition_hashes,
            "details": details,
        }
        self.payload["state"] = target
        self.payload["history"].append(entry)
        write_json(self.path, self.payload)


def _assert_isolated(work_root: Path, source_root: Path) -> tuple[Path, Path]:
    work = Path(work_root).expanduser().resolve()
    source = Path(source_root).expanduser().resolve()
    if work == source or work.is_relative_to(source) or source.is_relative_to(work):
        raise ValueError("source and work roots must be isolated")
    return work, source


def _sealed_rejected_payload(
    raw: bytes,
    payload: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    seal: Mapping[str, Any],
    *,
    revision: str,
) -> dict[str, Any]:
    """Validate and bind one canonical predecessor state payload."""

    try:
        _catalog_sha256_value, _index_sha256, artifacts, _index = (
            _REJECTED_CATALOG_SEALS[revision]
        )
    except KeyError as error:
        raise ValueError("rejected revision state is unsupported") from error

    expected_sources = {
        str(key): str(value) for key, value in sorted(source_hashes.items())
    }
    valid = (
        _valid_rejected_revision_state_payload(
            payload,
            revision=revision,
            source_hashes=expected_sources,
            screening_sha256=artifacts[0]["sha256"],
        )
        and raw == canonical_bytes(payload) + b"\n"
    )
    if not valid:
        raise ValueError("rejected revision state does not match the sealed failure")
    by_original = {str(row["original_path"]): row for row in artifacts}
    prefix = "scenarios/q2-realism-v2/"
    state_row = by_original.get(prefix + "09-logs/revision-state.json")
    state_sha256 = hashlib.sha256(raw).hexdigest()
    if state_row is not None and state_sha256 != state_row["sha256"]:
        raise ValueError("rejected revision state hash does not match the content seal")
    return {
        "catalog_revision": revision,
        "catalog_sha256": str(seal["catalog_sha256"]),
        "rejected_catalog_audit_index_sha256": str(seal["audit_index_sha256"]),
        "revision_state_seal_index_sha256": (
            _REJECTED_R1_STATE_INDEX_SHA256
            if revision == "r1"
            else str(seal["audit_index_sha256"])
        ),
        "screening_sha256": str(
            by_original[
                prefix + "00-manifest/realism-candidate-screening.json"
            ]["sha256"]
        ),
        "precheck_sha256": str(
            by_original[prefix + "09-logs/realism-v2-precheck.json"]["sha256"]
        ),
        "revision_state_sha256": state_sha256,
        "source_hashes": expected_sources,
        "cohort_hashes": dict(_index["cohort_hashes"]),
        "payload": dict(payload),
    }


def _sealed_rejected_state(
    root: Path,
    source_hashes: Mapping[str, str],
    seal: Mapping[str, Any],
    *,
    revision: str,
) -> dict[str, Any] | None:
    """Return an exact rejected state snapshot when its successor starts fresh."""

    path = root / "09-logs" / "revision-state.json"
    if not path.is_file():
        return None
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("rejected revision state is malformed") from error
    if not isinstance(payload, dict):
        raise ValueError("rejected revision state is malformed")
    if payload.get("state") == "created":
        return None
    return _sealed_rejected_payload(
        raw, payload, source_hashes, seal, revision=revision
    )


def _resume_pending_rollover(
    root: Path,
    source_hashes: Mapping[str, str],
    *,
    revision: str,
    predecessor_revision: str,
    catalog_sha256: str,
) -> bool:
    """Validate the exact state left after an atomic pre-screen rollover."""

    path = root / "09-logs" / "revision-state.json"
    if not path.is_file():
        return False
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("pending catalog rollover state is malformed") from error
    if not isinstance(payload, dict) or payload.get("state") != "created":
        return False
    if payload.get("catalog_revision") is None:
        return False
    if payload.get("catalog_revision") != revision:
        raise ValueError("pending catalog rollover revision mismatch")

    state_original = "scenarios/q2-realism-v2/09-logs/revision-state.json"
    seal = _verify_rejected_catalog_archive(
        root,
        revision=predecessor_revision,
        verify_originals=True,
        replaced_originals=frozenset({state_original}),
    )
    embedded = payload.get("superseded_revision_state")
    if not isinstance(embedded, dict) or not isinstance(embedded.get("payload"), dict):
        raise ValueError("pending catalog rollover predecessor binding is malformed")
    predecessor_raw = canonical_bytes(embedded["payload"]) + b"\n"
    expected_predecessor = _sealed_rejected_payload(
        predecessor_raw,
        embedded["payload"],
        source_hashes,
        seal,
        revision=predecessor_revision,
    )
    if embedded != expected_predecessor:
        raise ValueError("pending catalog rollover predecessor binding mismatch")
    expected = _rollover_state_payload(
        source_hashes,
        expected_predecessor,
        catalog_revision=revision,
        catalog_sha256=catalog_sha256,
    )
    if raw != canonical_bytes(expected) + b"\n" or payload != expected:
        raise ValueError("pending catalog rollover state binding mismatch")
    return True


def _cohorts_from_manifests(manifests: Mapping[str, Any]) -> dict[str, Any]:
    supplied = manifests.get("validation_cohorts")
    if isinstance(supplied, Mapping):
        return {str(protocol): cohort for protocol, cohort in supplied.items()}

    from .cohort import select_cohort
    from .loader import load_normalized_sites

    required = ("dkasc_manifest", "nrel_manifest", "nrel_selection_audit")
    missing = [key for key in required if key not in manifests]
    if missing:
        raise ValueError(
            "validation screening manifests are missing: " + ", ".join(missing)
        )
    dkasc = load_normalized_sites(Path(manifests["dkasc_manifest"]), split="validation")
    audit_path = Path(manifests["nrel_selection_audit"]).expanduser().resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    selected_ids = set(map(str, audit.get("selected_site_ids_by_state", {}).get("OK", ())))
    if not selected_ids:
        raise ValueError("NREL selection audit contains no Oklahoma validation sites")
    nrel = load_normalized_sites(
        Path(manifests["nrel_manifest"]),
        split="validation",
        site_ids=selected_ids,
    )
    if {site.site_id for site in nrel} != selected_ids:
        raise ValueError("NREL validation sites do not match the selection audit")
    return {
        "dkasc-yulara": select_cohort(dkasc, split="validation"),
        "nrel-transfer": select_cohort(nrel, split="validation"),
    }


def _cohort_sizes(cohorts: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    sizes = {}
    for protocol, cohort in cohorts.items():
        site_ids = sorted(set(cohort.site_ids[:, 0].astype("U").tolist()))
        sizes[protocol] = {
            "forecast_sample_count": int(cohort.targets.size),
            "horizon": int(cohort.targets.shape[1]),
            "origin_count": int(cohort.targets.shape[0]),
            "selected_block_count": len(cohort.blocks),
            "site_count": len(site_ids),
            "site_ids": site_ids,
        }
    return sizes


def _source_hashes(
    manifests: Mapping[str, Any], source_root: Path | None = None
) -> dict[str, str]:
    hashes: dict[str, str] = {}

    def add_path(prefix: str, value: Any, *, required: bool = False) -> None:
        if isinstance(value, Mapping):
            for nested_key, nested_value in sorted(value.items()):
                add_path(f"{prefix}:{nested_key}", nested_value, required=required)
        elif isinstance(value, (str, Path)):
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                if required:
                    raise ValueError(
                        f"declared source input does not exist or is not a file: {prefix}"
                    )
                return
            hashes[f"manifest:{prefix}"] = sha256_file(path)
        elif required:
            raise ValueError(f"declared source input path is malformed: {prefix}")

    for key, value in sorted(manifests.items()):
        if key == "validation_cohorts" and isinstance(value, Mapping):
            for protocol, cohort in sorted(value.items()):
                hashes[f"validation_cohort:{protocol}"] = cohort_sha256(cohort)
            continue
        add_path(
            key,
            value,
            required=key in _DECLARED_SOURCE_PATH_KEYS or key == "prepared_manifests",
        )
    settings = {
        key: manifests.get(key)
        for key in ("smoke", "smoke_storage_override", "estimated_candidate_bytes")
        if key in manifests
    }
    hashes["manifest:orchestration_settings"] = hashlib.sha256(
        canonical_bytes(settings)
    ).hexdigest()
    if source_root is not None:
        hashes.update(_source_root_hashes(source_root))
    return hashes


def _source_root_hashes(source_root: Path) -> dict[str, str]:
    """Hash the complete small v1 audit surface and require its core artifacts."""

    source = Path(source_root).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("source scenario root does not exist")
    missing = [relative for relative in SOURCE_ROOT_MANDATORY_ARTIFACTS if not (source / relative).is_file()]
    if missing:
        raise ValueError(
            "source snapshot mismatch: mandatory source artifacts are missing: "
            + ", ".join(missing)
        )
    paths = set()
    for relative in ("00-manifest", "04-aggregates", "09-logs"):
        paths.update((source / relative).glob("*.json"))
    for protocol in ("dkasc-yulara", "nrel-transfer"):
        paths.update((source / "02-predictions" / protocol).glob("*.json"))
    return {
        f"source_root:{path.relative_to(source).as_posix()}": sha256_file(path)
        for path in sorted(paths)
    }


def _single_candidate_screen(
    candidate: RealismCandidate, cohorts: dict[str, Any]
) -> CandidateScreeningReport:
    result = evaluate_validation_candidate(candidate, cohorts)
    return CandidateScreeningReport(
        (result,),
        {protocol: cohort_sha256(cohort) for protocol, cohort in cohorts.items()},
        _cohort_sizes(cohorts),
    )


def run_realism_precheck(
    work_root: Path,
    manifests: Mapping[str, Any],
    *,
    candidates: Sequence[RealismCandidate] | None = None,
) -> RevisionRunReport:
    """Run validation-only screening and persist JSON audits, never predictions."""

    root = Path(work_root).expanduser().resolve()
    if (root / "02-predictions").exists():
        raise ValueError("realism precheck refuses an existing prediction layout")
    rows = tuple(candidate_catalog() if candidates is None else candidates)
    if not rows:
        raise ValueError("candidate screening requires at least one candidate")
    revisions = {row.config.revision for row in rows}
    if len(revisions) != 1:
        raise ValueError("candidate screening requires one catalog revision")
    revision = next(iter(revisions))
    rejected_revision, catalog_sha256 = _registered_revision(revision)
    source_hashes = _source_hashes(manifests)
    pending_rollover = False
    superseded_revision_state = None
    if rejected_revision is not None:
        if rejected_revision == "r1":
            verify_rejected_revision_state_archive(root, revision="r1")
        pending_rollover = _resume_pending_rollover(
            root,
            source_hashes,
            revision=revision,
            predecessor_revision=rejected_revision,
            catalog_sha256=catalog_sha256,
        )
        if not pending_rollover:
            seal = verify_rejected_catalog_archive(root, revision=rejected_revision)
            superseded_revision_state = _sealed_rejected_state(
                root, source_hashes, seal, revision=rejected_revision
            )
            if superseded_revision_state is None:
                raise ValueError("sealed predecessor revision state is missing")
    machine = (
        RevisionStateMachine(root, source_hashes)
        if pending_rollover
        else RevisionStateMachine(
            root,
            source_hashes,
            superseded_revision_state=superseded_revision_state,
            catalog_revision=(revision if superseded_revision_state else None),
            catalog_sha256=(catalog_sha256 if superseded_revision_state else None),
        )
    )
    if machine.state != "created":
        raise ValueError(
            f"incompatible revision transition: {machine.state} -> validation_screened"
        )
    cohorts = _cohorts_from_manifests(manifests)
    screening = (
        _single_candidate_screen(rows[0], cohorts)
        if len(rows) == 1
        else screen_candidates(list(rows), cohorts)
    )
    screening_path = write_candidate_screening(root, screening)
    screening_hash = sha256_file(screening_path)
    machine.transition(
        "validation_screened",
        screening_report_sha256=screening_hash,
        screening_status=screening.to_payload()["status"],
    )
    report = RevisionRunReport(
        status=("validation_screened" if screening.accepted else "failed_validation_screen"),
        state=machine.state,
        screening_sha256=screening_hash,
        evaluated_candidate_ids=tuple(
            result.candidate_id for result in screening.candidates
        ),
        rejected_candidate_ids=tuple(
            result.candidate_id for result in screening.rejected
        ),
    )
    write_json(root / "09-logs" / "realism-v2-precheck.json", report.to_payload())
    if list(root.rglob("*.npz")) or list(root.rglob("*.csv.gz")):
        raise RuntimeError("realism precheck created forbidden prediction artifacts")
    return report


def _screened_candidates(payload: Mapping[str, Any]) -> tuple[RealismCandidate, ...]:
    if payload.get("status") != "passed" or not payload.get("selected_candidate_id"):
        return ()
    candidates = []
    for row in payload.get("candidates", ()):
        if row.get("status") != "accepted":
            continue
        config = ScenarioGenerationConfig(**row["config"])
        candidate = RealismCandidate(
            candidate_id=str(row["candidate_id"]),
            config=config,
            config_sha256=str(row["config_sha256"]),
            scenario_seed=int(row["scenario_seed"]),
        )
        if canonical_config_sha256(config) != candidate.config_sha256:
            raise ValueError("screened candidate configuration hash mismatch")
        candidates.append(candidate)
    return tuple(sorted(candidates, key=lambda item: item.candidate_id))


def _available_bytes(root: Path) -> int:
    anchor = root if root.exists() else root.parent
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    return int(shutil.disk_usage(anchor).free)


def _estimated_candidate_bytes(root: Path, manifests: Mapping[str, Any]) -> int:
    if manifests.get("smoke") is True:
        if manifests.get("smoke_storage_override") is not True:
            raise ValueError("smoke materialization requires an explicit storage override")
        explicit = manifests.get("estimated_candidate_bytes", 0)
        if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit < 0:
            raise ValueError("smoke storage override must be a non-negative integer")
        return explicit
    preflight = root / "00-manifest" / "preflight-4a.json"
    if not preflight.is_file():
        raise ValueError("4A storage estimate is missing")
    try:
        payload = json.loads(preflight.read_text(encoding="utf-8"))
        storage = payload["storage"]
        total = storage["total_bytes"]
        protocols = storage["per_protocol"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("4A storage estimate is malformed") from error
    valid_integer = isinstance(total, int) and not isinstance(total, bool) and total > 0
    valid_protocols = set(protocols) == {"dkasc-yulara", "nrel-transfer"} and all(
        isinstance(protocols[protocol].get("total_bytes"), int)
        and protocols[protocol]["total_bytes"] > 0
        and protocols[protocol].get("prediction_archives") == 210
        and protocols[protocol].get("sensitivity_aliases") == 45
        for protocol in ("dkasc-yulara", "nrel-transfer")
    )
    limit = storage.get("preflight_limit_bytes")
    valid = (
        payload.get("status") == "passed"
        and payload.get("generation_ready") is True
        and storage.get("schema") == "compact_shared_v2"
        and valid_integer
        and isinstance(limit, int)
        and total <= limit
        and storage.get("prediction_archives") == 420
        and storage.get("sensitivity_aliases") == 90
        and valid_protocols
        and sum(protocols[protocol]["total_bytes"] for protocol in protocols) <= total
    )
    if not valid:
        raise ValueError("4A storage estimate is malformed or did not pass")
    return total


def _prepared_manifest(
    manifests: Mapping[str, Any], protocol: str
) -> Path | None:
    prepared = manifests.get("prepared_manifests")
    value = prepared.get(protocol) if isinstance(prepared, Mapping) else None
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _materialize_candidate(
    candidate_root: Path,
    candidate: RealismCandidate,
    manifests: Mapping[str, Any],
    *,
    resume: bool,
    screening_report_sha256: str,
    materialization_source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    journal_path = candidate_root / "09-logs" / "materialization-journal.json"
    identity = {
        "candidate_id": candidate.candidate_id,
        "candidate_config_sha256": candidate.config_sha256,
        "screening_report_sha256": screening_report_sha256,
        "materialization_source_hashes": dict(materialization_source_hashes),
    }
    if journal_path.is_file():
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        if any(journal.get(key) != value for key, value in identity.items()):
            raise ValueError("materialization journal identity mismatch")
    else:
        journal = {
            **dict(EVIDENCE_METADATA),
            "disclosure": DISCLOSURE,
            "schema": "realism_materialization_journal_v1",
            **identity,
            "completed_stages": {},
        }
        write_json(journal_path, journal)

    def markers(protocol_root: Path, stage: str) -> tuple[Path, ...]:
        names = {
            "prepare": ("manifest.json",),
            "generate": ("candidate-audit.json",),
            "evaluate": ("evaluations.json",),
            "aggregate": ("aggregate.json",),
        }[stage]
        return tuple(protocol_root / name for name in names)

    def registered_completion(stage_root: Path, stage: str) -> bool:
        index_path = stage_root / "artifact-index.json"
        if not index_path.is_file():
            return False
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        digest = index.get("stages", {}).get(stage, {}).get("sha256")
        return isinstance(digest, str) and len(digest) == 64

    def run_stage(
        key: str,
        paths: tuple[Path, ...],
        operation: Any,
        completion_proof: Any,
    ) -> None:
        completed = journal["completed_stages"].get(key)
        if completed is not None and completed.get("status", "completed") == "completed":
            actual = {
                str(path.relative_to(candidate_root)): sha256_file(path)
                for path in paths
                if path.is_file()
            }
            if actual != completed["artifact_hashes"]:
                raise ValueError(f"materialization journal hash mismatch: {key}")
            return
        if completed is not None and completion_proof() and all(path.is_file() for path in paths):
            completed.update(
                status="completed",
                artifact_hashes={
                    str(path.relative_to(candidate_root)): sha256_file(path)
                    for path in paths
                },
                recovered_after_interruption=True,
            )
            write_json(journal_path, journal)
            return
        journal["completed_stages"][key] = {"status": "running"}
        write_json(journal_path, journal)
        operation()
        if any(not path.is_file() for path in paths):
            raise ValueError(f"materialization stage did not write its marker: {key}")
        journal["completed_stages"][key] = {
            "status": "completed",
            "artifact_hashes": {
                str(path.relative_to(candidate_root)): sha256_file(path) for path in paths
            }
        }
        write_json(journal_path, journal)

    smoke = bool(manifests.get("smoke", False))
    for protocol in ("dkasc-yulara", "nrel-transfer"):
        protocol_root = candidate_root / "02-predictions" / protocol
        manifest_path = _prepared_manifest(manifests, protocol)
        if not smoke and manifest_path is None:
            raise ValueError(f"realism run requires a prepared manifest for {protocol}")
        run_stage(
            f"{protocol}:prepare",
            markers(protocol_root, "prepare"),
            partial(
                prepare,
                protocol_root,
                smoke=smoke,
                manifest_path=manifest_path,
                candidate_seeds=[candidate.scenario_seed],
                protocol=protocol,
                generation_config=asdict(candidate.config),
                generation_config_sha256=candidate.config_sha256,
            ),
            partial(registered_completion, protocol_root, "prepare"),
        )
        run_stage(
            f"{protocol}:generate",
            markers(protocol_root, "generate"),
            partial(generate, protocol_root, resume=resume),
            partial(registered_completion, protocol_root, "generate"),
        )
        run_stage(
            f"{protocol}:evaluate",
            markers(protocol_root, "evaluate"),
            partial(evaluate, protocol_root),
            partial(registered_completion, protocol_root, "evaluate"),
        )
        run_stage(
            f"{protocol}:aggregate",
            markers(protocol_root, "aggregate"),
            partial(aggregate, protocol_root),
            partial(registered_completion, protocol_root, "aggregate"),
        )
    unified_paths = (candidate_root / "04-aggregates" / "acceptance.json",)
    result: dict[str, Any] = {}

    def unified() -> None:
        result.update(run_unified_4d(candidate_root))

    run_stage(
        "unified-4d",
        unified_paths,
        unified,
        lambda: (candidate_root / "04-aggregates/unified-artifact-index.json").is_file(),
    )
    acceptance = evaluate_candidate_acceptance(candidate_root)
    return {
        "passed": acceptance.passed,
        "unmet_checks": list(acceptance.reasons),
        "provenance": acceptance.provenance,
    }


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _verify_candidate_gate_hashes(
    candidate_root: Path, work_root: Path, expected: Mapping[str, str]
) -> None:
    for relative, digest in expected.items():
        paths = [candidate_root / relative, work_root / relative]
        existing = [path for path in paths if path.is_file()]
        if len(existing) != 1 or sha256_file(existing[0]) != digest:
            raise ValueError(f"accepted candidate derived hash mismatch: {relative}")


def _tree_size_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _bundle_value(bundle: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(bundle, Mapping):
        return bundle.get(name)
    return getattr(bundle, name, None)


def _strict_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def reject_candidate(
    work_root: Path,
    bundle: Mapping[str, Any] | Any,
    *,
    before_cleanup: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Audit a failed candidate completely, then remove only its prediction tree."""

    root = Path(work_root).expanduser().resolve()
    raw_candidate_root = _bundle_value(bundle, "candidate_root")
    if raw_candidate_root is None:
        raise ValueError("rejected candidate bundle is missing candidate_root")
    candidate_root = Path(raw_candidate_root).expanduser().resolve()
    candidates_root = (root / "candidates").resolve()
    if candidate_root.parent != candidates_root or candidate_root == candidates_root:
        raise ValueError("rejected path is not an isolated candidate bundle")
    candidate_id = _bundle_value(bundle, "candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id or candidate_root.name != candidate_id:
        raise ValueError("rejected candidate bundle identity mismatch")

    audit_path = root / "09-logs" / "candidate-decisions.json"
    if audit_path.is_file():
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("candidate decision audit is malformed") from error
        existing = [
            row for row in audit.get("decisions", ()) if row.get("candidate_id") == candidate_id
        ]
        if existing:
            if len(existing) != 1 or existing[0].get("status") != "rejected":
                raise ValueError("candidate decision audit contains a conflicting identity")
            prediction_root = candidate_root / "02-predictions"
            if prediction_root.exists():
                current = {
                    protocol: _tree_sha256(prediction_root / protocol)
                    for protocol in ("dkasc-yulara", "nrel-transfer")
                    if (prediction_root / protocol).is_dir()
                }
                if current != existing[0].get("per_protocol_bundle_sha256"):
                    raise ValueError("rejected candidate retry bundle hash mismatch")
                if evaluate_candidate_acceptance(candidate_root).passed:
                    raise ValueError("refusing to delete a passing candidate on retry")
                if before_cleanup is not None:
                    before_cleanup(dict(existing[0]))
                shutil.rmtree(prediction_root)
            elif before_cleanup is not None:
                before_cleanup(dict(existing[0]))
            return dict(existing[0])
    else:
        audit = {
            **dict(EVIDENCE_METADATA),
            "disclosure": DISCLOSURE,
            "schema": "realism_candidate_decisions_v1",
            "decisions": [],
        }

    if not candidate_root.is_dir():
        raise ValueError("rejected candidate bundle does not exist")
    acceptance = evaluate_candidate_acceptance(candidate_root)
    if acceptance.passed:
        raise ValueError("refusing to reject a passing candidate")
    acceptance_path = candidate_root / "04-aggregates" / "acceptance.json"
    aggregate_path = candidate_root / "04-aggregates" / "unified-aggregate.json"
    if not acceptance_path.is_file() or not aggregate_path.is_file():
        raise ValueError("rejected candidate lacks derived acceptance provenance")
    raw_acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))

    config = _bundle_value(bundle, "config")
    config_sha256 = _bundle_value(bundle, "config_sha256")
    if not isinstance(config, Mapping) or not isinstance(config_sha256, str):
        raise ValueError("rejected candidate configuration provenance is missing")
    try:
        actual_config_sha256 = canonical_config_sha256(
            ScenarioGenerationConfig(**dict(config))
        )
    except (TypeError, ValueError):
        actual_config_sha256 = ""
    if actual_config_sha256 != config_sha256:
        raise ValueError("rejected candidate configuration hash mismatch")
    source_manifests = _bundle_value(bundle, "source_manifests")
    if not isinstance(source_manifests, Mapping):
        raise ValueError("rejected candidate source manifests are missing")
    normalized_sources = dict(
        sorted((str(key), str(value)) for key, value in source_manifests.items())
    )
    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in normalized_sources.values()
    ):
        raise ValueError("rejected candidate source manifest hash is malformed")

    prediction_root = candidate_root / "02-predictions"
    if not prediction_root.is_dir():
        raise ValueError("rejected candidate prediction tree is missing before audit")
    protocols = ("dkasc-yulara", "nrel-transfer")
    protocol_roots = {protocol: prediction_root / protocol for protocol in protocols}
    if any(not path.is_dir() for path in protocol_roots.values()):
        raise ValueError("rejected candidate lacks both protocol bundles")
    artifact_index_roots: dict[str, str] = {}
    artifact_index_sha256: dict[str, str] = {}
    candidate_manifest_sha256: dict[str, str] = {}
    for protocol, protocol_root in protocol_roots.items():
        index_path = protocol_root / "artifact-index.json"
        manifest_path = protocol_root / "manifest.json"
        if not index_path.is_file():
            raise ValueError(f"rejected candidate artifact index is missing: {protocol}")
        if not manifest_path.is_file():
            raise ValueError(f"rejected candidate manifest is missing: {protocol}")
        artifact_index_roots[protocol] = artifact_root_digest(
            load_artifact_index(protocol_root)
        )
        artifact_index_sha256[protocol] = sha256_file(index_path)
        candidate_manifest_sha256[protocol] = sha256_file(manifest_path)

    decision = _strict_json_value({
        **dict(EVIDENCE_METADATA),
        "disclosure": DISCLOSURE,
        "candidate_id": candidate_id,
        "status": "rejected",
        "unmet_checks": list(acceptance.reasons),
        "config": dict(config),
        "config_sha256": config_sha256,
        "acceptance": raw_acceptance,
        "acceptance_sha256": sha256_file(acceptance_path),
        "metrics": dict(aggregate.get("values", {})),
        "source_manifests": normalized_sources,
        "source_manifests_sha256": hashlib.sha256(
            canonical_bytes(normalized_sources)
        ).hexdigest(),
        "derived_acceptance_provenance": acceptance.provenance,
        "artifact_index_roots": artifact_index_roots,
        "artifact_index_sha256": artifact_index_sha256,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "per_protocol_bundle_sha256": {
            protocol: _tree_sha256(protocol_root)
            for protocol, protocol_root in protocol_roots.items()
        },
        "artifact_root_sha256": _tree_sha256(candidate_root),
        "storage_size_bytes": _tree_size_bytes(candidate_root),
        "prediction_storage_size_bytes": _tree_size_bytes(prediction_root),
        "removed_path": str(prediction_root.relative_to(root)),
    })
    audit["decisions"].append(decision)
    write_json(audit_path, audit)
    persisted = json.loads(audit_path.read_text(encoding="utf-8"))
    if persisted.get("decisions", [])[-1] != decision:
        raise RuntimeError("candidate decision audit did not persist before cleanup")
    if before_cleanup is not None:
        before_cleanup(dict(decision))
    shutil.rmtree(prediction_root)
    return decision


def _atomic_move(source: Path, destination: Path) -> None:
    source.replace(destination)


def _promote_candidate_layout(
    candidate_root: Path,
    work_root: Path,
    *,
    candidate: RealismCandidate,
    screening_report_sha256: str,
    candidate_acceptance_sha256: str,
    materialization_source_hashes: Mapping[str, str],
) -> None:
    journal_path = work_root / "09-logs" / "canonical-promotion.json"
    identity = {
        "candidate_id": candidate.candidate_id,
        "candidate_config_sha256": candidate.config_sha256,
        "screening_report_sha256": screening_report_sha256,
        "candidate_acceptance_sha256": candidate_acceptance_sha256,
        "materialization_source_hashes": dict(materialization_source_hashes),
    }
    if journal_path.is_file():
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        if any(journal.get(key) != value for key, value in identity.items()):
            raise ValueError("canonical promotion journal identity mismatch")
    else:
        sources = [
            candidate_root / name
            for name in _PROMOTED_LAYOUT
            if (candidate_root / name).exists()
        ]
        if not sources:
            raise ValueError("accepted candidate has no canonical numbered layout")
        conflicts = [
            work_root / source.name
            for source in sources
            if (work_root / source.name).exists()
        ]
        if conflicts:
            raise ValueError(
                "canonical v2 layout already exists: "
                + ", ".join(path.name for path in conflicts)
            )
        journal = {
            **dict(EVIDENCE_METADATA),
            "disclosure": DISCLOSURE,
            "schema": "realism_canonical_promotion_journal_v1",
            **identity,
            "directories": [
                {"name": source.name, "status": "pending", "tree_sha256": _tree_sha256(source)}
                for source in sources
            ],
        }
        write_json(journal_path, journal)
    for row in journal["directories"]:
        source = candidate_root / row["name"]
        destination = work_root / row["name"]
        if source.exists() and destination.exists():
            raise ValueError(f"canonical promotion has duplicate directory: {row['name']}")
        if not source.exists() and not destination.exists():
            raise ValueError(f"canonical promotion lost directory: {row['name']}")
        current = destination if destination.exists() else source
        if _tree_sha256(current) != row["tree_sha256"]:
            raise ValueError(f"canonical promotion tree hash mismatch: {row['name']}")
        if source.exists():
            _atomic_move(source, destination)
            if _tree_sha256(destination) != row["tree_sha256"]:
                raise ValueError(f"canonical promotion moved tree hash mismatch: {row['name']}")
        row["status"] = "moved"
        write_json(journal_path, journal)


def _materialization_source_snapshot(
    root: Path, manifests: Mapping[str, Any], source: Path
) -> dict[str, str]:
    hashes = _source_hashes(manifests, source)
    if manifests.get("smoke") is not True:
        prepared = manifests.get("prepared_manifests")
        if not isinstance(prepared, Mapping) or set(prepared) != {
            "dkasc-yulara",
            "nrel-transfer",
        }:
            raise ValueError(
                "source snapshot mismatch: both prepared protocol manifests are required"
            )
        missing_prepared = [
            protocol
            for protocol, path in prepared.items()
            if not Path(path).expanduser().resolve().is_file()
        ]
        if missing_prepared:
            raise ValueError(
                "source snapshot mismatch: prepared manifests are missing: "
                + ", ".join(sorted(missing_prepared))
            )
        preflight = root / "00-manifest" / "preflight-4a.json"
        if not preflight.is_file():
            raise ValueError("source snapshot mismatch: 4A preflight is missing")
        hashes["work_root:00-manifest/preflight-4a.json"] = sha256_file(preflight)
    return dict(sorted(hashes.items()))


def _bound_materialization_sources(machine: RevisionStateMachine) -> dict[str, str] | None:
    for entry in machine.payload["history"]:
        if entry.get("state") == "candidate_materializing":
            bound = entry.get("details", {}).get("materialization_source_hashes")
            return dict(bound) if isinstance(bound, dict) else None
    return None


def _require_exact_materialization_sources(
    machine: RevisionStateMachine, current: Mapping[str, str]
) -> None:
    bound = _bound_materialization_sources(machine)
    if bound is None:
        return
    if dict(current) != bound:
        missing = sorted(set(bound).difference(current))
        added = sorted(set(current).difference(bound))
        changed = sorted(
            key for key in set(bound).intersection(current) if bound[key] != current[key]
        )
        detail = "; ".join(
            part
            for part in (
                f"missing={','.join(missing)}" if missing else "",
                f"added={','.join(added)}" if added else "",
                f"changed={','.join(changed)}" if changed else "",
            )
            if part
        )
        raise ValueError(f"source snapshot mismatch: {detail}")


def _require_exact_screened_sources(
    machine: RevisionStateMachine, current: Mapping[str, str]
) -> None:
    screening_entry = next(
        entry
        for entry in machine.payload["history"]
        if entry.get("state") == "validation_screened"
    )
    screened = {
        key: value
        for key, value in screening_entry.get("source_hashes", {}).items()
        if key != "screening_report_sha256"
    }
    if dict(current) != screened:
        missing = sorted(set(screened).difference(current))
        added = sorted(set(current).difference(screened))
        changed = sorted(
            key
            for key in set(screened).intersection(current)
            if screened[key] != current[key]
        )
        detail = "; ".join(
            part
            for part in (
                f"missing={','.join(missing)}" if missing else "",
                f"added={','.join(added)}" if added else "",
                f"changed={','.join(changed)}" if changed else "",
            )
            if part
        )
        raise ValueError(f"validation-screen source snapshot mismatch: {detail}")


def _candidate_for_state(
    machine: RevisionStateMachine,
    screened: Sequence[RealismCandidate],
) -> RealismCandidate:
    details = machine.payload["history"][-1].get("details", {})
    candidate_id = details.get("candidate_id")
    candidate = next(
        (item for item in screened if item.candidate_id == candidate_id), None
    )
    if candidate is None:
        raise ValueError("recorded revision candidate is absent from the screening report")
    if details.get("candidate_config_sha256") != candidate.config_sha256:
        raise ValueError("recorded revision candidate config hash mismatch")
    return candidate


def _load_acceptance(
    candidate_root: Path,
    work_root: Path | None = None,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    paths = [candidate_root / "04-aggregates" / "acceptance.json"]
    if work_root is not None:
        paths.append(work_root / "04-aggregates" / "acceptance.json")
    existing = [path for path in paths if path.is_file()]
    if len(existing) != 1:
        raise ValueError("candidate acceptance artifact is missing or duplicated")
    digest = sha256_file(existing[0])
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("recorded candidate acceptance hash mismatch")
    try:
        payload = json.loads(existing[0].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError("candidate acceptance artifact is malformed") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("passed"), bool):
        raise ValueError("candidate acceptance artifact is malformed")
    return payload, digest


def run_realism_revision(
    work_root: Path,
    source_root: Path,
    manifests: Mapping[str, Any],
    *,
    candidates: Sequence[RealismCandidate] | None = None,
    screening_sha256: str | None = None,
    resume: bool = False,
) -> RevisionRunReport:
    """Run screened candidates in isolation through 4B--4D, stopping before 4E."""

    root, source = _assert_isolated(work_root, source_root)
    screening_path = root / "00-manifest" / "realism-candidate-screening.json"
    if not screening_path.is_file():
        if candidates is None:
            raise ValueError("realism run requires a successful screening report")
        precheck = run_realism_precheck(root, manifests, candidates=candidates)
        if precheck.status == "failed_validation_screen":
            return precheck
        screening_sha256 = precheck.screening_sha256
    elif screening_sha256 is None:
        raise ValueError("screening hash is required for realism run")

    actual_screening_hash = sha256_file(screening_path)
    if screening_sha256 != actual_screening_hash:
        raise ValueError("screening hash mismatch")
    screening = json.loads(screening_path.read_text(encoding="utf-8"))
    if any(screening.get(key) != value for key, value in EVIDENCE_METADATA.items()):
        raise ValueError("screening report evidence metadata mismatch")
    screening_candidates = screening.get("candidates", ())
    screening_revisions = {
        str(row["config"].get("revision"))
        for row in screening_candidates
        if isinstance(row, Mapping) and isinstance(row.get("config"), Mapping)
    }
    if len(screening_revisions) > 1:
        raise ValueError("screening report mixes catalog revisions")
    screened_revision = next(iter(screening_revisions), "r1")
    rejected_revision, _screened_catalog_sha256 = _registered_revision(
        screened_revision
    )
    if rejected_revision is not None:
        verify_rejected_catalog_archive(
            root, revision=rejected_revision, verify_originals=False
        )
        if rejected_revision == "r1":
            verify_rejected_revision_state_archive(root, revision="r1")
    screened = _screened_candidates(screening)
    if not screened:
        return RevisionRunReport(
            status="failed_validation_screen",
            state="validation_screened",
            screening_sha256=actual_screening_hash,
        )

    screened_sources = _source_hashes(manifests)
    machine = RevisionStateMachine(root, screened_sources)
    allowed = {"validation_screened", "candidate_rejected"}
    if resume:
        allowed.update(
            ("candidate_materializing", "candidate_evaluated", "candidate_accepted")
        )
    if machine.state not in allowed:
        raise ValueError(f"revision state cannot run with resume={resume}: {machine.state}")
    screening_entry = next(
        entry
        for entry in machine.payload["history"]
        if entry.get("state") == "validation_screened"
    )
    recorded_hash = screening_entry["source_hashes"].get("screening_report_sha256")
    if recorded_hash != actual_screening_hash:
        raise ValueError("revision state screening hash mismatch")
    _require_exact_screened_sources(machine, screened_sources)

    estimated = _estimated_candidate_bytes(root, manifests)
    current_sources = _materialization_source_snapshot(root, manifests, source)
    _require_exact_materialization_sources(machine, current_sources)
    evaluated = [
        str(entry["details"]["candidate_id"])
        for entry in machine.payload["history"]
        if entry.get("state") == "candidate_evaluated"
    ]
    rejected = [
        str(entry["details"]["candidate_id"])
        for entry in machine.payload["history"]
        if entry.get("state") == "candidate_rejected"
    ]

    if machine.state == "candidate_evaluated":
        candidate = _candidate_for_state(machine, screened)
        candidate_root = root / "candidates" / candidate.candidate_id
        recorded_acceptance = machine.payload["history"][-1]["details"].get(
            "candidate_acceptance_sha256"
        )
        if not isinstance(recorded_acceptance, str):
            raise ValueError("recorded candidate acceptance hash is missing")
        _acceptance_payload, actual_acceptance = _load_acceptance(
            candidate_root, expected_sha256=recorded_acceptance
        )
        complete_acceptance = evaluate_candidate_acceptance(candidate_root)
        if complete_acceptance.passed:
            machine.transition(
                "candidate_accepted",
                candidate_id=candidate.candidate_id,
                candidate_config_sha256=candidate.config_sha256,
                candidate_acceptance_sha256=actual_acceptance,
                candidate_gate_derived_sha256=complete_acceptance.provenance[
                    "derived_artifact_sha256"
                ],
                materialization_source_hashes=current_sources,
            )
        else:
            def record_resumed_rejection(decision: dict[str, Any]) -> None:
                machine.transition(
                    "candidate_rejected",
                    candidate_id=candidate.candidate_id,
                    candidate_config_sha256=candidate.config_sha256,
                    candidate_acceptance_sha256=actual_acceptance,
                    candidate_decision_sha256=hashlib.sha256(
                        canonical_bytes(decision)
                    ).hexdigest(),
                    rejection_reasons=list(complete_acceptance.reasons),
                    materialization_source_hashes=current_sources,
                )

            reject_candidate(
                root,
                {
                    "candidate_root": candidate_root,
                    "candidate_id": candidate.candidate_id,
                    "config": asdict(candidate.config),
                    "config_sha256": candidate.config_sha256,
                    "source_manifests": current_sources,
                },
                before_cleanup=record_resumed_rejection,
            )
            rejected.append(candidate.candidate_id)

    if machine.state == "candidate_accepted":
        candidate = _candidate_for_state(machine, screened)
        candidate_root = root / "candidates" / candidate.candidate_id
        recorded_acceptance = machine.payload["history"][-1]["details"].get(
            "candidate_acceptance_sha256"
        )
        if not isinstance(recorded_acceptance, str):
            raise ValueError("recorded candidate acceptance hash is missing")
        acceptance, actual_acceptance = _load_acceptance(
            candidate_root, root, expected_sha256=recorded_acceptance
        )
        if not acceptance["passed"]:
            raise ValueError("accepted candidate has a failing acceptance artifact")
        recorded_gate_hashes = machine.payload["history"][-1]["details"].get(
            "candidate_gate_derived_sha256"
        )
        if not isinstance(recorded_gate_hashes, Mapping):
            raise ValueError("accepted candidate derived hashes are missing")
        _verify_candidate_gate_hashes(candidate_root, root, recorded_gate_hashes)
        _promote_candidate_layout(
            candidate_root,
            root,
            candidate=candidate,
            screening_report_sha256=actual_screening_hash,
            candidate_acceptance_sha256=actual_acceptance,
            materialization_source_hashes=current_sources,
        )
        report = RevisionRunReport(
            status="candidate_accepted",
            state=machine.state,
            screening_sha256=actual_screening_hash,
            accepted_candidate_id=candidate.candidate_id,
            evaluated_candidate_ids=tuple(evaluated),
            rejected_candidate_ids=tuple(rejected),
        )
        write_json(root / "09-logs" / "revision-run.json", report.to_payload())
        return report

    candidates_to_run = [item for item in screened if item.candidate_id not in rejected]
    if machine.state == "candidate_materializing":
        candidate = _candidate_for_state(machine, screened)
        candidates_to_run = [candidate, *(
            item
            for item in candidates_to_run
            if item.candidate_id != candidate.candidate_id
        )]

    for candidate in candidates_to_run:
        current_sources = _materialization_source_snapshot(root, manifests, source)
        _require_exact_materialization_sources(machine, current_sources)
        required = estimated + MINIMUM_FREE_RESERVE_BYTES
        available = _available_bytes(root)
        if available < required:
            raise OSError(
                f"insufficient free space before candidate {candidate.candidate_id}: "
                f"requires {required}, found {available}"
            )
        candidate_root = root / "candidates" / candidate.candidate_id
        resuming_candidate = machine.state == "candidate_materializing"
        if candidate_root.exists() and not (resume and resuming_candidate):
            raise ValueError(f"candidate bundle already exists: {candidate.candidate_id}")
        if not resuming_candidate:
            machine.transition(
                "candidate_materializing",
                candidate_id=candidate.candidate_id,
                candidate_config_sha256=candidate.config_sha256,
                screening_report_sha256=actual_screening_hash,
                materialization_source_hashes=current_sources,
            )
            candidate_root.mkdir(parents=True, exist_ok=False)
        elif not candidate_root.exists():
            candidate_root.mkdir(parents=True, exist_ok=False)
        _materialize_candidate(
            candidate_root,
            candidate,
            manifests,
            resume=resume,
            screening_report_sha256=actual_screening_hash,
            materialization_source_hashes=current_sources,
        )
        acceptance, acceptance_hash = _load_acceptance(candidate_root)
        complete_acceptance = evaluate_candidate_acceptance(candidate_root)
        materialized_passed = complete_acceptance.passed
        materialized_reasons = list(complete_acceptance.reasons)
        evaluated.append(candidate.candidate_id)
        machine.transition(
            "candidate_evaluated",
            candidate_id=candidate.candidate_id,
            candidate_config_sha256=candidate.config_sha256,
            candidate_acceptance_sha256=acceptance_hash,
            candidate_gate_passed=materialized_passed,
            candidate_gate_reasons=materialized_reasons,
            candidate_gate_derived_sha256=complete_acceptance.provenance[
                "derived_artifact_sha256"
            ],
            materialization_source_hashes=current_sources,
        )
        if not materialized_passed:
            def record_rejection(
                decision: dict[str, Any],
                *,
                candidate_id: str = candidate.candidate_id,
                config_sha256: str = candidate.config_sha256,
                accepted_sha256: str = acceptance_hash,
                reasons: tuple[str, ...] = tuple(materialized_reasons),
                source_items: tuple[tuple[str, str], ...] = tuple(
                    current_sources.items()
                ),
            ) -> None:
                machine.transition(
                    "candidate_rejected",
                    candidate_id=candidate_id,
                    candidate_config_sha256=config_sha256,
                    candidate_acceptance_sha256=accepted_sha256,
                    candidate_decision_sha256=hashlib.sha256(
                        canonical_bytes(decision)
                    ).hexdigest(),
                    rejection_reasons=list(reasons),
                    materialization_source_hashes=dict(source_items),
                )

            reject_candidate(
                root,
                {
                    "candidate_root": candidate_root,
                    "candidate_id": candidate.candidate_id,
                    "config": asdict(candidate.config),
                    "config_sha256": candidate.config_sha256,
                    "source_manifests": current_sources,
                },
                before_cleanup=record_rejection,
            )
            rejected.append(candidate.candidate_id)
            continue
        machine.transition(
            "candidate_accepted",
            candidate_id=candidate.candidate_id,
            candidate_config_sha256=candidate.config_sha256,
            candidate_acceptance_sha256=acceptance_hash,
            candidate_gate_derived_sha256=complete_acceptance.provenance[
                "derived_artifact_sha256"
            ],
            materialization_source_hashes=current_sources,
        )
        _promote_candidate_layout(
            candidate_root,
            root,
            candidate=candidate,
            screening_report_sha256=actual_screening_hash,
            candidate_acceptance_sha256=acceptance_hash,
            materialization_source_hashes=current_sources,
        )
        report = RevisionRunReport(
            status="candidate_accepted",
            state=machine.state,
            screening_sha256=actual_screening_hash,
            accepted_candidate_id=candidate.candidate_id,
            evaluated_candidate_ids=tuple(evaluated),
            rejected_candidate_ids=tuple(rejected),
        )
        write_json(root / "09-logs" / "revision-run.json", report.to_payload())
        return report

    report = RevisionRunReport(
        status="failed_candidate_evaluation",
        state=machine.state,
        screening_sha256=actual_screening_hash,
        evaluated_candidate_ids=tuple(evaluated),
        rejected_candidate_ids=tuple(rejected),
    )
    write_json(root / "09-logs" / "revision-run.json", report.to_payload())
    return report


__all__ = [
    "REVISION_STATES",
    "RevisionRunReport",
    "RevisionStateMachine",
    "reject_candidate",
    "run_realism_precheck",
    "run_realism_revision",
    "r5_catalog_sha256",
    "r5_precheck_ranges_payload",
    "r5_precheck_ranges_sha256",
    "r5_validation_precheck_reasons",
    "r6_catalog_sha256",
    "r6_design_precheck_reasons",
    "r6_preregistration_payload",
    "r6_preregistration_sha256",
    "verify_rejected_catalog_archive",
    "verify_rejected_revision_state_archive",
]
