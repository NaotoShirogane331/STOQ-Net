"""Atomic two-stage design freeze and one-shot r5 confirmation orchestration."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .constants import RACE_PV
from .loader import role_allows_split
from .nested_validation import (
    ConfirmationCohorts,
    ConfirmationContract,
    confirmation_block_csv_bytes,
    confirmation_protocol_payload,
)
from .realism import RealismCandidate, candidate_catalog, canonical_config_sha256
from .scenario_artifacts import canonical_bytes
from .scenario_manifest import cohort_sha256
from .scenario_revision import (
    VALIDATION_PRECHECK_RANGES,
    CandidateScreeningReport,
    r5_catalog_sha256,
    r5_precheck_ranges_payload,
    r5_precheck_ranges_sha256,
    r5_validation_precheck_reasons,
    screen_candidates,
)

R5_TRANSITIONS = {
    "r5_created": frozenset({"confirmation_cohort_prepared"}),
    "confirmation_cohort_prepared": frozenset({"design_screened"}),
    "design_screened": frozenset({"candidate_frozen", "failed_design_screen"}),
    "candidate_frozen": frozenset({"confirmation_screened"}),
    "confirmation_screened": frozenset(
        {"formal_materializing", "failed_confirmation"}
    ),
}
R5_SELECTION_ORDER = "distance_to_range_midpoints,candidate_id"
_EXPECTED_REPLICATE_SEEDS = (13, 29, 47, 71, 101)
_DESIGN_BUNDLE = Path("00-manifest/r5-design-freeze")
_CONFIRMATION_BUNDLE = Path("09-logs/r5-confirmation")
_MAPPING_COHORT_KEYS = frozenset({"split", "blocks"})
_MAPPING_BLOCK_KEYS = frozenset(
    {
        "block_id",
        "dataset",
        "end",
        "n_steps",
        "quarter",
        "raw_hashes",
        "selection_hash",
        "site_id",
        "spatial_role",
        "split",
        "start",
        "state",
    }
)
_AUTHORITY_SCHEMA = "r5_confirmation_authority_v1"
_AUTHORITY_FIELDS = (
    "design_bundle_sha256",
    "design_screening_sha256",
    "candidate_freeze_sha256",
    "design_state_sha256",
    "cross_binding_contract_sha256",
    "artifact_roots_sha256",
    "confirmation_contract_sha256",
)


@dataclass(frozen=True)
class R5ArtifactPaths:
    """Independent files whose exact bytes are committed by both r5 decisions."""

    code: Path
    design_cohort: Path
    confirmation_protocol: Path
    confirmation_block_csv: Path
    extracted_source_log: Path
    prior_r1_r4_seal_chain: Path
    predecessor_state: Path


@dataclass(frozen=True)
class R5CrossBindingContract:
    """Caller-held hashes; none are derived from a decision being verified."""

    code_sha256: str
    r5_catalog_sha256: str
    range_definition_sha256: str
    design_cohort_sha256: str
    confirmation_protocol_sha256: str
    confirmation_block_csv_sha256: str
    extracted_source_log_sha256: str
    prior_r1_r4_seal_chain_sha256: str
    predecessor_state_sha256: str


@dataclass(frozen=True)
class R5ConfirmationAuthority:
    """Caller-held seal for the exact design bundle and its external authority."""

    design_bundle_sha256: str
    design_screening_sha256: str
    candidate_freeze_sha256: str
    design_state_sha256: str
    cross_binding_contract_sha256: str
    artifact_roots_sha256: str
    confirmation_contract_sha256: str


@dataclass(frozen=True)
class R5DesignInputs:
    """Pure-injection boundary for the validation-only r5 design screen."""

    work_root: Path
    candidates: Sequence[RealismCandidate]
    design_cohorts: Mapping[str, Any]
    confirmation_cohorts: ConfirmationCohorts
    confirmation_contract: ConfirmationContract
    artifacts: R5ArtifactPaths
    cross_binding_contract: R5CrossBindingContract


@dataclass(frozen=True)
class FrozenCandidateEvaluation:
    """Deeply immutable scalar result retained in an orchestration report."""

    candidate_id: str
    config_sha256: str
    model_profile_sha256: str
    scenario_seed: int
    passed: bool
    metric_values: tuple[tuple[str, float], ...]
    reasons: tuple[str, ...]
    distance_to_range_midpoints: float

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self.metric_values)

    def to_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "config_sha256": self.config_sha256,
            "model_profile_sha256": self.model_profile_sha256,
            "scenario_seed": self.scenario_seed,
            "status": "accepted" if self.passed else "rejected",
            "metrics": dict(self.metric_values),
            "reasons": list(self.reasons),
            "distance_to_range_midpoints": self.distance_to_range_midpoints,
        }


@dataclass(frozen=True)
class DesignFreezeReport:
    """Immutable view of the atomically published design screen/freeze bundle."""

    status: str
    evaluated_count: int
    accepted: tuple[FrozenCandidateEvaluation, ...]
    rejected: tuple[FrozenCandidateEvaluation, ...]
    frozen_candidate_id: str | None
    frozen_config_sha256: str | None
    frozen_model_profile_sha256: str | None
    frozen_scenario_seed: int | None
    selection_order: str
    screening_sha256: str
    freeze_sha256: str
    work_root: Path
    artifacts: R5ArtifactPaths
    cross_binding_contract: R5CrossBindingContract
    confirmation_contract: ConfirmationContract

    @property
    def screening_path(self) -> Path:
        return self.work_root / _DESIGN_BUNDLE / "screening.json"

    @property
    def freeze_path(self) -> Path:
        return self.work_root / _DESIGN_BUNDLE / "freeze.json"

    @property
    def state_path(self) -> Path:
        return self.work_root / _DESIGN_BUNDLE / "state.json"

    @property
    def decision_path(self) -> Path:
        return self.work_root / _CONFIRMATION_BUNDLE / "decision.json"


@dataclass(frozen=True)
class ConfirmationDecision:
    """Immutable terminal outcome of the single permitted confirmation attempt."""

    status: str
    candidate_id: str
    config_sha256: str
    scenario_seed: int
    metric_values: tuple[tuple[str, float], ...]
    reasons: tuple[str, ...]
    freeze_sha256: str
    decision_sha256: str
    decision_path: Path
    state_path: Path

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self.metric_values)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return canonical_bytes(dict(payload)) + b"\n"


def _reject_symlink_components(path: Path) -> None:
    absolute = path.expanduser().absolute()
    for component in (absolute, *absolute.parents):
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink path component is forbidden: {component}")


def _secure_root(value: Path) -> Path:
    raw = Path(value).expanduser()
    _reject_symlink_components(raw)
    root = raw.resolve()
    root.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(root)
    if not root.is_dir():
        raise ValueError("r5 work root is not a directory")
    return root


def _secure_file(path: Path, *, label: str) -> tuple[bytes, str]:
    raw = Path(path).expanduser()
    _reject_symlink_components(raw)
    try:
        mode = raw.lstat().st_mode
        content = raw.read_bytes()
    except (FileNotFoundError, OSError) as error:
        raise ValueError(f"{label} binding file is missing or unreadable") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} binding path must be a regular file")
    return content, _sha256_bytes(content)


def _artifact_rows(paths: R5ArtifactPaths, *, anchor: Path) -> dict[str, str]:
    rows = {}
    for label, value in asdict(paths).items():
        path = Path(value).expanduser().resolve()
        rows[label] = (
            path.relative_to(anchor).as_posix()
            if path.is_relative_to(anchor)
            else path.as_posix()
        )
    return rows


def _contract_payload(contract: R5CrossBindingContract) -> dict[str, str]:
    payload = {str(key): str(value) for key, value in asdict(contract).items()}
    if not all(_is_sha256(value) for value in payload.values()):
        raise ValueError("r5 cross-binding contract contains an invalid SHA-256")
    return payload


def _confirmation_contract_sha256(contract: ConfirmationContract) -> str:
    return _sha256_bytes(canonical_bytes(asdict(contract)))


def _cross_binding_contract_sha256(contract: R5CrossBindingContract) -> str:
    return _sha256_bytes(canonical_bytes(_contract_payload(contract)))


def _artifact_roots_sha256(report: DesignFreezeReport) -> str:
    return _sha256_bytes(
        canonical_bytes(
            {
                "work_root": report.work_root.as_posix(),
                "artifacts": {
                    key: Path(value).as_posix()
                    for key, value in sorted(asdict(report.artifacts).items())
                },
            }
        )
    )


def _design_bundle_sha256(
    *, screening_sha256: str, freeze_sha256: str, state_sha256: str
) -> str:
    return _sha256_bytes(
        canonical_bytes(
            {
                "candidate_freeze_sha256": freeze_sha256,
                "design_screening_sha256": screening_sha256,
                "design_state_sha256": state_sha256,
            }
        )
    )


def _actual_cross_bindings(
    *,
    artifacts: R5ArtifactPaths,
    confirmation_cohorts: ConfirmationCohorts,
    confirmation_contract: ConfirmationContract,
    design_cohorts: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    file_hashes = {}
    file_bytes = {}
    for label, value in asdict(artifacts).items():
        content, digest = _secure_file(Path(value), label=label)
        file_bytes[label] = content
        file_hashes[f"{label}_sha256"] = digest

    protocol = confirmation_protocol_payload(
        confirmation_cohorts,
        confirmation_contract=confirmation_contract,
    )
    expected_protocol = _json_bytes(protocol)
    expected_blocks = confirmation_block_csv_bytes(
        confirmation_cohorts,
        confirmation_contract=confirmation_contract,
    )
    if file_bytes["confirmation_protocol"] != expected_protocol:
        raise ValueError("confirmation protocol file differs from sealed cohort bytes")
    if file_bytes["confirmation_block_csv"] != expected_blocks:
        raise ValueError("confirmation block CSV differs from sealed cohort bytes")
    if design_cohorts is not None:
        expected_design_cohort = _json_bytes(r5_design_cohort_binding_payload(design_cohorts))
        if file_bytes["design_cohort"] != expected_design_cohort:
            raise ValueError("design cohort differs from its cross-binding file")

    return {
        "code_sha256": file_hashes["code_sha256"],
        "r5_catalog_sha256": r5_catalog_sha256(),
        "range_definition_sha256": r5_precheck_ranges_sha256(),
        "design_cohort_sha256": file_hashes["design_cohort_sha256"],
        "confirmation_protocol_sha256": file_hashes[
            "confirmation_protocol_sha256"
        ],
        "confirmation_block_csv_sha256": file_hashes[
            "confirmation_block_csv_sha256"
        ],
        "extracted_source_log_sha256": file_hashes[
            "extracted_source_log_sha256"
        ],
        "prior_r1_r4_seal_chain_sha256": file_hashes[
            "prior_r1_r4_seal_chain_sha256"
        ],
        "predecessor_state_sha256": file_hashes["predecessor_state_sha256"],
    }


def _verify_cross_bindings(
    expected: R5CrossBindingContract,
    *,
    artifacts: R5ArtifactPaths,
    confirmation_cohorts: ConfirmationCohorts,
    confirmation_contract: ConfirmationContract,
    design_cohorts: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    contract = _contract_payload(expected)
    actual = _actual_cross_bindings(
        artifacts=artifacts,
        confirmation_cohorts=confirmation_cohorts,
        confirmation_contract=confirmation_contract,
        design_cohorts=design_cohorts,
    )
    changed = [key for key in contract if actual.get(key) != contract[key]]
    if changed:
        raise ValueError("r5 cross-binding mismatch: " + ", ".join(changed))
    return actual


def _model_profile_sha256(candidate: RealismCandidate) -> str:
    return _sha256_bytes(canonical_bytes(asdict(candidate.model_profile)))


def _candidate_signature(candidate: RealismCandidate) -> tuple[str, str, int, str]:
    seed = candidate.scenario_seed
    if type(candidate.candidate_id) is not str or not _is_sha256(
        candidate.config_sha256
    ):
        raise ValueError("candidate identity/config hash has an invalid primitive type")
    if type(seed) is not int:
        raise ValueError("candidate scenario seed is missing or invalid")
    if canonical_config_sha256(candidate.config) != candidate.config_sha256:
        raise ValueError("candidate configuration hash mismatch")
    if candidate.model_profile != RACE_PV:
        raise ValueError("r5 candidate model profile must be the exact race_pv profile")
    return (
        candidate.candidate_id,
        candidate.config_sha256,
        seed,
        _model_profile_sha256(candidate),
    )


def _design_block_payload(block: Any) -> dict[str, Any]:
    def field(name: str) -> Any:
        return block.get(name) if isinstance(block, Mapping) else getattr(block, name)

    return {
        "block_id": str(field("block_id")),
        "dataset": str(field("dataset")),
        "end": (
            field("end").isoformat()
            if hasattr(field("end"), "isoformat")
            else str(field("end"))
        ),
        "n_steps": int(field("n_steps")),
        "quarter": str(field("quarter")),
        "raw_hashes": list(field("raw_hashes")),
        "selection_hash": str(field("selection_hash")),
        "site_id": str(field("site_id")),
        "spatial_role": field("spatial_role"),
        "split": str(field("split")),
        "start": (
            field("start").isoformat()
            if hasattr(field("start"), "isoformat")
            else str(field("start"))
        ),
        "state": field("state"),
    }


def _require_design_cohort_isolation(cohorts: Mapping[str, Any]) -> None:
    if not cohorts:
        raise ValueError("design cohorts are missing")
    protocol_names = tuple(cohorts)
    if any(not isinstance(protocol, str) or not protocol for protocol in protocol_names):
        raise ValueError("design cohort protocol keys must be nonempty strings")
    block_identities = set()
    for protocol, cohort in cohorts.items():
        if isinstance(cohort, Mapping):
            if set(cohort) != _MAPPING_COHORT_KEYS:
                raise ValueError(
                    f"design cohort {protocol!r} mapping schema key set mismatch"
                )
            split = cohort["split"]
            blocks = cohort["blocks"]
        else:
            split = cohort.split
            blocks = cohort.blocks
        if split != "validation":
            raise ValueError(f"design cohort {protocol!r} split is not validation")
        if not isinstance(blocks, (list, tuple)) or not blocks:
            raise ValueError(f"design cohort {protocol!r} blocks are missing")
        for block in blocks:
            if isinstance(block, Mapping) and set(block) != _MAPPING_BLOCK_KEYS:
                raise ValueError("design cohort block mapping schema key set mismatch")
            row = _design_block_payload(block)
            identity = (
                row["dataset"],
                row["site_id"],
                row["block_id"],
                row["start"],
                row["end"],
                row["selection_hash"],
                tuple(row["raw_hashes"]),
            )
            if identity in block_identities:
                raise ValueError("design cohorts contain a duplicate block identity")
            block_identities.add(identity)
            role = row["spatial_role"] or ""
            if row["split"] != "validation" or role in {
                "zero_parameter_test",
                "test",
            }:
                raise ValueError("design cohort contains a held-out block split/role")
            if not role_allows_split(
                row["dataset"],
                role,
                row["state"],
                "validation",
                site_id=row["site_id"],
            ):
                raise ValueError("design cohort block role/state is invalid")
            if (
                not row["raw_hashes"]
                or not all(_is_sha256(value) for value in row["raw_hashes"])
                or not _is_sha256(row["selection_hash"])
                or not row["block_id"]
            ):
                raise ValueError("design cohort block source provenance is invalid")


def r5_design_cohort_binding_payload(
    cohorts: Mapping[str, Any],
) -> dict[str, Any]:
    """Return deterministic bytes/hashed identities for the exact evaluator cohorts."""

    _require_design_cohort_isolation(cohorts)
    rows = {}
    for protocol, cohort in sorted(cohorts.items()):
        if isinstance(cohort, Mapping):
            rows[str(protocol)] = {
                "split": cohort["split"],
                "blocks": [
                    _design_block_payload(block) for block in cohort["blocks"]
                ],
            }
        else:
            rows[str(protocol)] = {
                "cohort_sha256": cohort_sha256(cohort),
                "split": cohort.split,
                "blocks": [_design_block_payload(block) for block in cohort.blocks],
            }
    return {"schema": "r5_design_cohort_binding_v1", "cohorts": rows}


def _exact_r5_candidates(
    candidates: Sequence[RealismCandidate],
) -> tuple[RealismCandidate, ...]:
    rows = tuple(candidates)
    expected = tuple(candidate_catalog("r5"))
    if len(rows) != 12:
        raise ValueError("r5 design requires the exact twelve-candidate catalog")
    actual_signatures = [_candidate_signature(candidate) for candidate in rows]
    expected_signatures = [_candidate_signature(candidate) for candidate in expected]
    if (
        len(set(actual_signatures)) != 12
        or sorted(actual_signatures) != sorted(expected_signatures)
    ):
        raise ValueError("r5 design candidate catalog identity/config/seed mismatch")
    expected_by_id = {candidate.candidate_id: candidate for candidate in expected}
    for candidate in rows:
        expected_candidate = expected_by_id[candidate.candidate_id]
        if candidate.config != expected_candidate.config:
            raise ValueError("r5 design candidate catalog configuration mismatch")
        if candidate.model_profile != expected_candidate.model_profile:
            raise ValueError("r5 design candidate catalog model profile mismatch")
    return tuple(sorted(rows, key=lambda item: item.candidate_id))


def _midpoint_distance(metrics: Mapping[str, float]) -> float:
    offsets = []
    for name, (lower, upper) in VALIDATION_PRECHECK_RANGES.items():
        midpoint = (lower + upper) / 2.0
        offsets.append((float(metrics[name]) - midpoint) / (upper - lower))
    return math.sqrt(sum(value * value for value in offsets) / len(offsets))


def _validated_design_results(
    report: CandidateScreeningReport,
    candidates: tuple[RealismCandidate, ...],
) -> tuple[tuple[FrozenCandidateEvaluation, ...], tuple[FrozenCandidateEvaluation, ...]]:
    if not isinstance(report, CandidateScreeningReport):
        raise ValueError("design evaluator returned an invalid screening report")
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    results = tuple(report.candidates)
    result_ids = [result.candidate_id for result in results]
    if len(results) != 12 or len(set(result_ids)) != 12 or set(result_ids) != set(by_id):
        raise ValueError("design evaluator did not evaluate all twelve candidates exactly once")

    normalized = []
    required_metrics = set(VALIDATION_PRECHECK_RANGES)
    for result in results:
        candidate = by_id[result.candidate_id]
        if (
            result.config_sha256 != candidate.config_sha256
            or result.scenario_seed != candidate.scenario_seed
            or result.config != asdict(candidate.config)
            or tuple(result.seeds) != _EXPECTED_REPLICATE_SEEDS
        ):
            raise ValueError("design evaluator candidate identity/config/seed mismatch")
        if set(result.metrics) != required_metrics:
            raise ValueError("design evaluator metric set is incomplete or contains extras")
        metrics = {name: float(result.metrics[name]) for name in sorted(required_metrics)}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("design evaluator returned a non-finite metric")
        reasons = r5_validation_precheck_reasons(metrics)
        distance = _midpoint_distance(metrics)
        if (
            bool(result.passed) != (not reasons)
            or tuple(result.reasons) != reasons
            or not math.isfinite(result.distance_to_range_midpoints)
            or not math.isclose(
                float(result.distance_to_range_midpoints),
                distance,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("design evaluator result disagrees with the exact r5 ranges")
        normalized.append(
            FrozenCandidateEvaluation(
                candidate_id=result.candidate_id,
                config_sha256=result.config_sha256,
                model_profile_sha256=_model_profile_sha256(candidate),
                scenario_seed=result.scenario_seed,
                passed=not reasons,
                metric_values=tuple(sorted(metrics.items())),
                reasons=reasons,
                distance_to_range_midpoints=distance,
            )
        )
    accepted = tuple(
        sorted(
            (result for result in normalized if result.passed),
            key=lambda item: (item.distance_to_range_midpoints, item.candidate_id),
        )
    )
    rejected = tuple(
        sorted(
            (result for result in normalized if not result.passed),
            key=lambda item: item.candidate_id,
        )
    )
    return accepted, rejected


def _state_entry(
    history: list[dict[str, Any]],
    target: str,
    *,
    details: Mapping[str, Any],
) -> None:
    previous = history[-1]["state"] if history else None
    if previous is not None and target not in R5_TRANSITIONS.get(previous, frozenset()):
        raise ValueError(f"illegal r5 transition: {previous} -> {target}")
    history.append(
        {
            "sequence": len(history) + 1,
            "from_state": previous,
            "state": target,
            "details": dict(details),
        }
    )


def _design_state_payload(
    cross_bindings: Mapping[str, str],
    *,
    screening_sha256: str,
    freeze_sha256: str,
    status: str,
) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    _state_entry(
        history,
        "r5_created",
        details={"predecessor_state_sha256": cross_bindings["predecessor_state_sha256"]},
    )
    _state_entry(
        history,
        "confirmation_cohort_prepared",
        details={
            "confirmation_protocol_sha256": cross_bindings[
                "confirmation_protocol_sha256"
            ],
            "confirmation_block_csv_sha256": cross_bindings[
                "confirmation_block_csv_sha256"
            ],
        },
    )
    _state_entry(
        history,
        "design_screened",
        details={"design_screening_sha256": screening_sha256},
    )
    _state_entry(
        history,
        status,
        details={"candidate_freeze_sha256": freeze_sha256},
    )
    return {
        "schema": "r5_confirmation_state_v1",
        "state": status,
        "history": history,
        "cross_bindings": dict(cross_bindings),
    }


def _assert_no_partial_bundle(final: Path) -> None:
    partials = tuple(final.parent.glob(f".{final.name}-*.tmp"))
    if partials:
        raise ValueError(f"partial r5 state exists for {final.name}")


def _preflight_bundle_destination(final: Path, *, one_shot: bool = False) -> None:
    _reject_symlink_components(final.parent)
    final.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(final.parent)
    _assert_no_partial_bundle(final)
    if final.exists() or final.is_symlink():
        if one_shot:
            raise ValueError("r5 confirmation is one-shot; a decision already exists")
        raise ValueError(f"r5 bundle already exists: {final.name}")


def _atomic_bundle(final: Path, payloads: Mapping[str, Mapping[str, Any]]) -> None:
    _preflight_bundle_destination(final)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{final.name}-", suffix=".tmp", dir=final.parent)
    )
    try:
        for name, payload in sorted(payloads.items()):
            if Path(name).name != name or not name.endswith(".json"):
                raise ValueError("r5 bundle entry name is invalid")
            path = temporary / name
            with path.open("xb") as handle:
                handle.write(_json_bytes(payload))
                handle.flush()
                os.fsync(handle.fileno())
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, final)
        descriptor = os.open(final.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _read_canonical_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    content, digest = _secure_file(path, label=label)
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} JSON is malformed") from error
    if not isinstance(payload, dict) or content != _json_bytes(payload):
        raise ValueError(f"{label} JSON is not canonical")
    return payload, digest


def run_r5_design_screen(
    inputs: R5DesignInputs,
    *,
    evaluator: Callable[
        [Sequence[RealismCandidate], Mapping[str, Any]], CandidateScreeningReport
    ] = screen_candidates,
) -> DesignFreezeReport:
    """Evaluate all exact r5 rows, rank range-passers, and atomically freeze one."""

    if not isinstance(inputs, R5DesignInputs):
        raise TypeError("inputs must be R5DesignInputs")
    root = _secure_root(inputs.work_root)
    final = root / _DESIGN_BUNDLE
    _preflight_bundle_destination(final)
    candidates = _exact_r5_candidates(inputs.candidates)
    _require_design_cohort_isolation(inputs.design_cohorts)
    cross_bindings = _verify_cross_bindings(
        inputs.cross_binding_contract,
        artifacts=inputs.artifacts,
        confirmation_cohorts=inputs.confirmation_cohorts,
        confirmation_contract=inputs.confirmation_contract,
        design_cohorts=inputs.design_cohorts,
    )
    result = evaluator(candidates, inputs.design_cohorts)
    accepted, rejected = _validated_design_results(result, candidates)
    _verify_cross_bindings(
        inputs.cross_binding_contract,
        artifacts=inputs.artifacts,
        confirmation_cohorts=inputs.confirmation_cohorts,
        confirmation_contract=inputs.confirmation_contract,
        design_cohorts=inputs.design_cohorts,
    )
    frozen = accepted[0] if accepted else None
    status = "candidate_frozen" if frozen is not None else "failed_design_screen"
    anchor = root.parent
    artifact_rows = _artifact_rows(inputs.artifacts, anchor=anchor)
    screening = {
        "schema": "r5_design_screening_v1",
        "status": "passed" if frozen is not None else "failed",
        "candidate_count": 12,
        "evaluated_count": 12,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "selection_order": R5_SELECTION_ORDER,
        "selected_candidate_id": frozen.candidate_id if frozen else None,
        "candidates": [
            result.to_payload() for result in (*accepted, *rejected)
        ],
        "precheck_ranges": r5_precheck_ranges_payload(),
        "cross_bindings": cross_bindings,
        "artifact_paths": artifact_rows,
        "held_out_used": False,
    }
    screening_sha256 = _sha256_bytes(_json_bytes(screening))
    freeze = {
        "schema": "r5_candidate_freeze_v1",
        "status": status,
        "candidate_id": frozen.candidate_id if frozen else None,
        "config_sha256": frozen.config_sha256 if frozen else None,
        "model_profile": asdict(RACE_PV) if frozen else None,
        "model_profile_sha256": frozen.model_profile_sha256 if frozen else None,
        "scenario_seed": frozen.scenario_seed if frozen else None,
        "design_screening_sha256": screening_sha256,
        "selection_order": R5_SELECTION_ORDER,
        "cross_bindings": cross_bindings,
        "artifact_paths": artifact_rows,
        "confirmation_contract_sha256": _confirmation_contract_sha256(
            inputs.confirmation_contract
        ),
    }
    freeze_sha256 = _sha256_bytes(_json_bytes(freeze))
    state = _design_state_payload(
        cross_bindings,
        screening_sha256=screening_sha256,
        freeze_sha256=freeze_sha256,
        status=status,
    )
    _atomic_bundle(
        final,
        {"freeze.json": freeze, "screening.json": screening, "state.json": state},
    )
    return DesignFreezeReport(
        status=status,
        evaluated_count=12,
        accepted=accepted,
        rejected=rejected,
        frozen_candidate_id=frozen.candidate_id if frozen else None,
        frozen_config_sha256=frozen.config_sha256 if frozen else None,
        frozen_model_profile_sha256=(
            frozen.model_profile_sha256 if frozen else None
        ),
        frozen_scenario_seed=frozen.scenario_seed if frozen else None,
        selection_order=R5_SELECTION_ORDER,
        screening_sha256=screening_sha256,
        freeze_sha256=freeze_sha256,
        work_root=root,
        artifacts=inputs.artifacts,
        cross_binding_contract=inputs.cross_binding_contract,
        confirmation_contract=inputs.confirmation_contract,
    )


def _read_verified_freeze_report(
    report: DesignFreezeReport,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    root = _secure_root(report.work_root)
    final = root / _DESIGN_BUNDLE
    _assert_no_partial_bundle(final)
    if final.is_symlink() or not final.is_dir():
        raise ValueError("candidate freeze bundle is missing or invalid")
    expected_names = {"freeze.json", "screening.json", "state.json"}
    if {path.name for path in final.iterdir()} != expected_names:
        raise ValueError("candidate freeze bundle entry set mismatch")
    freeze, freeze_sha256 = _read_canonical_json(
        final / "freeze.json", label="candidate freeze"
    )
    screening, screening_sha256 = _read_canonical_json(
        final / "screening.json", label="design screening"
    )
    state, state_sha256 = _read_canonical_json(final / "state.json", label="r5 state")
    report_candidates = [
        result.to_payload() for result in (*report.accepted, *report.rejected)
    ]
    expected_state = _design_state_payload(
        _contract_payload(report.cross_binding_contract),
        screening_sha256=screening_sha256,
        freeze_sha256=freeze_sha256,
        status="candidate_frozen",
    )
    if (
        report.status != "candidate_frozen"
        or state != expected_state
        or report.freeze_sha256 != freeze_sha256
        or report.screening_sha256 != screening_sha256
        or freeze.get("design_screening_sha256") != screening_sha256
        or freeze.get("candidate_id") != report.frozen_candidate_id
        or freeze.get("config_sha256") != report.frozen_config_sha256
        or freeze.get("model_profile_sha256")
        != report.frozen_model_profile_sha256
        or freeze.get("model_profile")
        != json.loads(canonical_bytes(asdict(RACE_PV)))
        or freeze.get("scenario_seed") != report.frozen_scenario_seed
        or freeze.get("cross_bindings") != _contract_payload(
            report.cross_binding_contract
        )
        or freeze.get("confirmation_contract_sha256")
        != _confirmation_contract_sha256(report.confirmation_contract)
        or screening.get("candidates") != report_candidates
        or screening.get("evaluated_count") != report.evaluated_count
        or screening.get("selection_order") != report.selection_order
    ):
        raise ValueError("candidate freeze report does not match its persisted binding")
    return freeze, state, state_sha256


def _verify_freeze_report(
    report: DesignFreezeReport,
    authority: R5ConfirmationAuthority,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    authority_values = _validated_authority_values(authority)
    freeze, state, state_sha256 = _read_verified_freeze_report(report)
    expected = {
        "design_bundle_sha256": _design_bundle_sha256(
            screening_sha256=report.screening_sha256,
            freeze_sha256=report.freeze_sha256,
            state_sha256=state_sha256,
        ),
        "design_screening_sha256": report.screening_sha256,
        "candidate_freeze_sha256": report.freeze_sha256,
        "design_state_sha256": state_sha256,
        "cross_binding_contract_sha256": _cross_binding_contract_sha256(
            report.cross_binding_contract
        ),
        "artifact_roots_sha256": _artifact_roots_sha256(report),
        "confirmation_contract_sha256": _confirmation_contract_sha256(
            report.confirmation_contract
        ),
    }
    if any(
        not hmac.compare_digest(authority_values[name], expected[name])
        for name in _AUTHORITY_FIELDS
    ):
        raise ValueError("r5 confirmation authority does not match the design bundle")
    return freeze, state, state_sha256


def _validated_authority_values(
    authority: R5ConfirmationAuthority,
) -> dict[str, str]:
    if type(authority) is not R5ConfirmationAuthority:
        raise ValueError("r5 confirmation authority must have the exact authority type")
    values = {name: getattr(authority, name) for name in _AUTHORITY_FIELDS}
    if not all(_is_sha256(values[name]) for name in _AUTHORITY_FIELDS):
        raise ValueError("r5 confirmation authority contains an invalid SHA-256")
    return values


def _authority_payload(authority: R5ConfirmationAuthority) -> dict[str, str]:
    values = _validated_authority_values(authority)
    return {"schema": _AUTHORITY_SCHEMA, **values}


def r5_confirmation_authority_bytes(authority: R5ConfirmationAuthority) -> bytes:
    """Return the only accepted canonical persistence bytes for an authority."""

    return _json_bytes(_authority_payload(authority))


def _exact_authority_path(value: Path | str, *, label: str) -> Path:
    if type(value) is str:
        return Path(value)
    if type(value) is type(Path()):
        return value
    raise ValueError(f"r5 authority {label} must have an exact path/string type")


def _contained_authority_path(
    path: Path | str, *, root: Path | str
) -> tuple[Path, Path]:
    raw_root = _exact_authority_path(root, label="root").expanduser()
    _reject_symlink_components(raw_root)
    resolved_root = raw_root.resolve()
    if not resolved_root.is_dir():
        raise ValueError("r5 authority root is missing or is not a directory")
    raw_path = _exact_authority_path(path, label="path").expanduser()
    candidate = raw_path if raw_path.is_absolute() else resolved_root / raw_path
    _reject_symlink_components(candidate)
    resolved_path = candidate.resolve(strict=False)
    if not resolved_path.is_relative_to(resolved_root) or resolved_path == resolved_root:
        raise ValueError("r5 authority path must be contained by its declared root")
    return resolved_path, resolved_root


def write_r5_confirmation_authority(
    authority: R5ConfirmationAuthority,
    path: Path | str,
    *,
    root: Path | str,
) -> str:
    """Atomically persist a new caller-held canonical authority below ``root``."""

    content = r5_confirmation_authority_bytes(authority)
    destination, _resolved_root = _contained_authority_path(path, root=root)
    _reject_symlink_components(destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination.parent)
    if destination.exists() or destination.is_symlink():
        raise ValueError("r5 confirmation authority already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        parent_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256_bytes(content)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key in r5 authority: {key}")
        payload[key] = value
    return payload


def load_r5_confirmation_authority(
    path: Path | str,
    *,
    root: Path | str,
) -> R5ConfirmationAuthority:
    """Load an exact canonical authority from a root-contained regular file."""

    source, _resolved_root = _contained_authority_path(path, root=root)
    content, _digest = _secure_file(source, label="r5 confirmation authority")
    try:
        payload = json.loads(content, object_pairs_hook=_reject_duplicate_json_keys)
    except json.JSONDecodeError as error:
        raise ValueError("r5 confirmation authority JSON is malformed") from error
    expected_keys = {"schema", *_AUTHORITY_FIELDS}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("r5 confirmation authority key set mismatch")
    if payload["schema"] != _AUTHORITY_SCHEMA:
        raise ValueError("r5 confirmation authority schema is invalid")
    authority = R5ConfirmationAuthority(
        **{name: payload[name] for name in _AUTHORITY_FIELDS}
    )
    if content != r5_confirmation_authority_bytes(authority):
        raise ValueError("r5 confirmation authority JSON is not canonical")
    return authority


def r5_confirmation_authority(
    report: DesignFreezeReport,
) -> R5ConfirmationAuthority:
    """Create a persistable caller-held seal immediately after a valid design freeze."""

    _freeze, _state, state_sha256 = _read_verified_freeze_report(report)
    return R5ConfirmationAuthority(
        design_bundle_sha256=_design_bundle_sha256(
            screening_sha256=report.screening_sha256,
            freeze_sha256=report.freeze_sha256,
            state_sha256=state_sha256,
        ),
        design_screening_sha256=report.screening_sha256,
        candidate_freeze_sha256=report.freeze_sha256,
        design_state_sha256=state_sha256,
        cross_binding_contract_sha256=_cross_binding_contract_sha256(
            report.cross_binding_contract
        ),
        artifact_roots_sha256=_artifact_roots_sha256(report),
        confirmation_contract_sha256=_confirmation_contract_sha256(
            report.confirmation_contract
        ),
    )


def _confirmation_metrics(value: Any) -> tuple[dict[str, float], tuple[str, ...]]:
    if not isinstance(value, Mapping) or set(value) != set(VALIDATION_PRECHECK_RANGES):
        raise ValueError("confirmation evaluator metric set is incomplete or contains extras")
    metrics = {name: float(value[name]) for name in sorted(value)}
    if not all(math.isfinite(metric) for metric in metrics.values()):
        raise ValueError("confirmation evaluator returned a non-finite metric")
    return metrics, r5_validation_precheck_reasons(metrics)


def run_r5_confirmation(
    freeze: DesignFreezeReport,
    cohorts: ConfirmationCohorts,
    *,
    authority: R5ConfirmationAuthority,
    evaluator: Callable[[RealismCandidate, ConfirmationCohorts], Mapping[str, float]],
) -> ConfirmationDecision:
    """Evaluate the exact frozen row once and atomically publish a terminal decision."""

    _validated_authority_values(authority)
    if not isinstance(freeze, DesignFreezeReport):
        raise TypeError("freeze must be a DesignFreezeReport")
    root = _secure_root(freeze.work_root)
    terminal = root / _CONFIRMATION_BUNDLE
    _preflight_bundle_destination(terminal, one_shot=True)
    persisted_freeze, design_state, design_state_sha256 = _verify_freeze_report(
        freeze, authority
    )
    _verify_cross_bindings(
        freeze.cross_binding_contract,
        artifacts=freeze.artifacts,
        confirmation_cohorts=cohorts,
        confirmation_contract=freeze.confirmation_contract,
    )
    expected_by_id = {
        candidate.candidate_id: candidate for candidate in candidate_catalog("r5")
    }
    candidate = expected_by_id.get(str(freeze.frozen_candidate_id))
    if (
        candidate is None
        or candidate.config_sha256 != freeze.frozen_config_sha256
        or candidate.scenario_seed != freeze.frozen_scenario_seed
        or _model_profile_sha256(candidate) != freeze.frozen_model_profile_sha256
        or persisted_freeze.get("candidate_id") != candidate.candidate_id
    ):
        raise ValueError("candidate freeze identity/config/hash changed")
    evaluation = evaluator(candidate, cohorts)
    metrics, reasons = _confirmation_metrics(evaluation)
    _verify_cross_bindings(
        freeze.cross_binding_contract,
        artifacts=freeze.artifacts,
        confirmation_cohorts=cohorts,
        confirmation_contract=freeze.confirmation_contract,
    )
    persisted_freeze, design_state, design_state_sha256 = _verify_freeze_report(
        freeze, authority
    )
    status = "failed_confirmation" if reasons else "formal_materializing"
    decision = {
        "schema": "r5_confirmation_decision_v1",
        "status": status,
        "candidate_id": candidate.candidate_id,
        "config": asdict(candidate.config),
        "config_sha256": candidate.config_sha256,
        "model_profile": asdict(candidate.model_profile),
        "model_profile_sha256": _model_profile_sha256(candidate),
        "scenario_seed": candidate.scenario_seed,
        "metrics": metrics,
        "reasons": list(reasons),
        "range_definition_sha256": freeze.cross_binding_contract.range_definition_sha256,
        "candidate_freeze_sha256": freeze.freeze_sha256,
        "design_state_sha256": design_state_sha256,
        "cross_bindings": _contract_payload(freeze.cross_binding_contract),
        "held_out_used": False,
    }
    decision_sha256 = _sha256_bytes(_json_bytes(decision))
    history = [dict(entry) for entry in design_state["history"]]
    _state_entry(
        history,
        "confirmation_screened",
        details={
            "confirmation_decision_sha256": decision_sha256,
            "design_state_sha256": design_state_sha256,
        },
    )
    _state_entry(
        history,
        status,
        details={
            "confirmation_decision_sha256": decision_sha256,
            "design_state_sha256": design_state_sha256,
        },
    )
    state = {
        "schema": "r5_confirmation_state_v1",
        "state": status,
        "history": history,
        "cross_bindings": _contract_payload(freeze.cross_binding_contract),
        "design_state_sha256": design_state_sha256,
    }
    _atomic_bundle(terminal, {"decision.json": decision, "state.json": state})
    return ConfirmationDecision(
        status=status,
        candidate_id=candidate.candidate_id,
        config_sha256=candidate.config_sha256,
        scenario_seed=candidate.scenario_seed,
        metric_values=tuple(metrics.items()),
        reasons=reasons,
        freeze_sha256=freeze.freeze_sha256,
        decision_sha256=decision_sha256,
        decision_path=terminal / "decision.json",
        state_path=terminal / "state.json",
    )


__all__ = [
    "ConfirmationDecision",
    "DesignFreezeReport",
    "FrozenCandidateEvaluation",
    "R5ArtifactPaths",
    "R5ConfirmationAuthority",
    "R5CrossBindingContract",
    "R5DesignInputs",
    "R5_TRANSITIONS",
    "load_r5_confirmation_authority",
    "r5_confirmation_authority",
    "r5_confirmation_authority_bytes",
    "r5_design_cohort_binding_payload",
    "run_r5_confirmation",
    "run_r5_design_screen",
    "write_r5_confirmation_authority",
]
