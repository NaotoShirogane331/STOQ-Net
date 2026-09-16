"""Strict verification and deterministic replay for realism-v2 revisions."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import shutil
import stat
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .compact_storage import ACTUAL_LIMIT_BYTES, MINIMUM_FREE_RESERVE_BYTES
from .constants import EVIDENCE_METADATA
from .nested_validation import (
    CONFIRMATION_EMBARGO_DAYS,
    EXPECTED_AUXILIARY_OK_SITE_IDS,
    ConfirmationContract,
)
from .realism import ScenarioGenerationConfig, candidate_catalog, canonical_config_sha256
from .scenario_artifacts import (
    DISCLOSURE,
    canonical_bytes,
    sha256_file,
    write_json,
)
from .scenario_confirmation import (
    R5ArtifactPaths,
    R5ConfirmationAuthority,
    R5CrossBindingContract,
)
from .scenario_manifest import ACCEPTANCE_RANGES
from .scenario_revision import (
    _TRANSITIONS,
    REVISION_STATES,
    _source_root_hashes,
    _valid_catalog_rollover_details,
    r5_catalog_sha256,
    r5_precheck_ranges_payload,
    r5_precheck_ranges_sha256,
    r5_validation_precheck_reasons,
)
from .scenario_unified import (
    evaluate_candidate_acceptance,
    evaluate_candidate_acceptance_payload,
)
from .scenario_verify import verify_protocol_core

PROTOCOLS = ("dkasc-yulara", "nrel-transfer")
REPORT_PATH = "09-logs/realism-v2-verification.json"
_LATER_CATALOG_CONFIG_SHA256S = frozenset(
    candidate.config_sha256
    for revision in ("r2", "r3")
    for candidate in candidate_catalog(revision)
)
STRUCTURAL_ACCEPTANCE_KEYS = (
    "all_ablations_degrade_a_primary_metric",
    "held_out_not_used_for_reference_selection",
    "state_lead_width_not_abnormal",
)
_CORE_REPLAY_FILES = (
    "manifest.json",
    "cohort-index.json",
    "candidate-audit.json",
    "evaluations.json",
    "selection.json",
    "aggregate.json",
    "artifact-index.json",
    "predictions/audit-sample.csv.gz",
    "predictions/audit-sample.json",
)
_UNIFIED_REPLAY_FILES = (
    "03-evaluations/unified-evaluation.json",
    "04-aggregates/unified-aggregate.json",
    "04-aggregates/acceptance.json",
    "04-aggregates/unified-artifact-index.json",
)
_DERIVED_GATE_FILES = (
    *(
        f"02-predictions/{protocol}/{name}"
        for protocol in PROTOCOLS
        for name in (
            "manifest.json",
            "evaluations.json",
            "selection.json",
            "aggregate.json",
            "artifact-index.json",
            "cohort-index.json",
        )
    ),
    "03-evaluations/unified-evaluation.json",
    "04-aggregates/unified-aggregate.json",
    "04-aggregates/acceptance.json",
    "04-aggregates/unified-artifact-index.json",
    "09-logs/materialization-journal.json",
)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_R5_EXECUTION_CODE_MODULES = (
    "cohort",
    "constants",
    "loader",
    "nested_validation",
    "pipeline",
    "realism",
    "scenario_artifacts",
    "scenario_candidate",
    "scenario_confirmation",
    "scenario_manifest",
    "scenario_revision",
    "scenario_revision_verify",
)


@dataclass(frozen=True)
class RevisionVerificationReport:
    """Persisted result of a complete fail-closed revision verification."""

    ok: bool
    status: str
    formal_mode: bool
    non_smoke: bool
    errors: tuple[str, ...]
    counts: dict[str, int]
    hashes: dict[str, Any]
    artifact_bytes: int
    free_bytes: int

    def to_payload(self) -> dict[str, Any]:
        return {
            **dict(EVIDENCE_METADATA),
            "disclosure": DISCLOSURE,
            **asdict(self),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class R5VerificationReport:
    """Deterministic persisted verdict for the nested-confirmation contract."""

    ok: bool
    status: str
    errors: tuple[str, ...]
    counts: dict[str, int]
    hashes: dict[str, str]

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "r5_verification_report_v1",
            "ok": self.ok,
            "status": self.status,
            "errors": list(self.errors),
            "counts": self.counts,
            "hashes": self.hashes,
        }


@dataclass(frozen=True)
class R5TerminalIntegrityResult:
    """Immutable non-eligibility result for an exact legal r5 terminal."""

    terminal_status: str
    counts: Mapping[str, int]
    hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.terminal_status not in {"formal_materializing", "failed_confirmation"}:
            raise ValueError("r5 terminal integrity status is invalid")
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))
        object.__setattr__(self, "hashes", MappingProxyType(dict(self.hashes)))


_R5_REPORT_PATH = "09-logs/r5-verification.json"
_R5_DESIGN_FILES = ("freeze.json", "screening.json", "state.json")
_R5_CONFIRMATION_FILES = ("decision.json", "state.json")
_R5_BLOCK_COLUMNS = (
    "cohort",
    "site_id",
    "dataset",
    "block_id",
    "start",
    "end",
    "quarter",
    "n_steps",
    "selection_hash",
    "raw_hashes",
    "spatial_role",
    "split",
    "state",
)


def _r5_sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _r5_sha256_payload(payload: Any) -> str:
    return _r5_sha256_bytes(canonical_bytes(payload))


def _r5_reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _r5_reject_symlinks(path: Path) -> None:
    cursor = path
    while True:
        if cursor.is_symlink():
            raise ValueError(f"symlink is forbidden: {path}")
        if cursor == cursor.parent:
            return
        cursor = cursor.parent


def _r5_regular_bytes(path: Path, *, label: str) -> bytes:
    _r5_reject_symlinks(path)
    try:
        mode = path.stat().st_mode
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{label} is missing or unreadable") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} is not a regular file")
    return content


def _r5_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _r5_regular_bytes(path, label=label)
    try:
        payload = json.loads(raw, object_pairs_hook=_r5_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} JSON is malformed") from error
    if not isinstance(payload, dict) or raw != canonical_bytes(payload) + b"\n":
        raise ValueError(f"{label} JSON is not canonical")
    return payload, raw


def _r5_exact_bundle(path: Path, names: tuple[str, ...], *, label: str) -> None:
    _r5_reject_symlinks(path)
    if not path.is_dir():
        raise ValueError(f"{label} bundle is missing")
    actual = {child.name for child in path.iterdir()}
    if actual != set(names) or any(child.is_symlink() for child in path.iterdir()):
        raise ValueError(f"{label} bundle entry set mismatch")
    if tuple(path.parent.glob(f".{path.name}-*.tmp")):
        raise ValueError(f"{label} partial bundle exists")


def _r5_contract_payload(contract: R5CrossBindingContract) -> dict[str, str]:
    if type(contract) is not R5CrossBindingContract:
        raise ValueError("r5 cross-binding contract type is invalid")
    payload = asdict(contract)
    if not all(type(value) is str and _is_sha256(value) for value in payload.values()):
        raise ValueError("r5 cross-binding contract hash is invalid")
    return payload


def _r5_authority_payload(authority: R5ConfirmationAuthority) -> dict[str, str]:
    if type(authority) is not R5ConfirmationAuthority:
        raise ValueError("r5 confirmation authority type is invalid")
    payload = asdict(authority)
    if not all(type(value) is str and _is_sha256(value) for value in payload.values()):
        raise ValueError("r5 confirmation authority hash is invalid")
    return payload


def _r5_artifact_map(artifacts: R5ArtifactPaths) -> dict[str, Path]:
    if type(artifacts) is not R5ArtifactPaths:
        raise ValueError("r5 artifact path contract type is invalid")
    return {name: Path(value).expanduser() for name, value in asdict(artifacts).items()}


def _r5_expected_execution_code_seal() -> dict[str, Any]:
    module_root = Path(__file__).resolve().parent
    return {
        "modules": [
            {
                "module": name,
                "path": (module_root / f"{name}.py")
                .resolve(strict=True)
                .relative_to(_PROJECT_ROOT)
                .as_posix(),
                "sha256": sha256_file(module_root / f"{name}.py"),
            }
            for name in _R5_EXECUTION_CODE_MODULES
        ],
        "schema": "r5_execution_code_seal_v1",
    }


def _verify_r5_execution_code_seal(raw: bytes) -> None:
    try:
        persisted = json.loads(raw, object_pairs_hook=_r5_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ValueError("r5 execution code seal is malformed") from error
    expected = _r5_expected_execution_code_seal()
    if (
        not isinstance(persisted, dict)
        or raw != canonical_bytes(persisted) + b"\n"
        or canonical_bytes(persisted) != canonical_bytes(expected)
    ):
        raise ValueError("r5 execution code seal changed or current code drifted")


def _r5_iso(value: Any, *, label: str) -> datetime:
    if type(value) is not str:
        raise ValueError(f"{label} timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} timestamp is invalid") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{label} timestamp lacks timezone")
    return parsed


def _verify_r5_artifacts(
    root: Path,
    artifacts: R5ArtifactPaths,
    contract: R5CrossBindingContract,
) -> tuple[dict[str, bytes], dict[str, str], dict[str, Any]]:
    paths = _r5_artifact_map(artifacts)
    contents = {
        name: _r5_regular_bytes(path, label=f"r5 {name}")
        for name, path in paths.items()
    }
    _verify_r5_execution_code_seal(contents["code"])
    hashes = {f"{name}_sha256": _r5_sha256_bytes(raw) for name, raw in contents.items()}
    expected = {
        **hashes,
        "r5_catalog_sha256": r5_catalog_sha256(),
        "range_definition_sha256": r5_precheck_ranges_sha256(),
    }
    if _r5_contract_payload(contract) != expected:
        raise ValueError("r5 independent cross-binding hash mismatch")

    anchor = root.parent
    artifact_rows: dict[str, str] = {}
    for name, path in paths.items():
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(anchor):
            raise ValueError("r5 artifact path escapes the independent anchor")
        relative = resolved.relative_to(anchor).as_posix()
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("r5 artifact path is not canonical")
        artifact_rows[name] = relative

    extraction = json.loads(
        contents["extracted_source_log"], object_pairs_hook=_r5_reject_duplicate_keys
    )
    if contents["extracted_source_log"] != canonical_bytes(extraction) + b"\n":
        raise ValueError("r5 extracted source log is not canonical")
    sources = extraction.get("sources") if isinstance(extraction, dict) else None
    if (
        extraction.get("schema") != "r5_extracted_source_log_v1"
        or not isinstance(sources, list)
        or tuple(sorted(row.get("site_id") for row in sources if isinstance(row, dict)))
        != EXPECTED_AUXILIARY_OK_SITE_IDS
        or len(sources) != 5
    ):
        raise ValueError("r5 extracted source identities are invalid")
    for row in sources:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "archive_sha256",
                "member",
                "member_sha256",
                "output_sha256",
                "site_id",
            }
            or not all(
                _is_sha256(row[key])
                for key in ("archive_sha256", "member_sha256", "output_sha256")
            )
            or type(row["member"]) is not str
            or Path(row["member"]).is_absolute()
            or ".." in Path(row["member"]).parts
        ):
            raise ValueError("r5 extracted ZIP/member hash contract is invalid")

    seal_chain = json.loads(contents["prior_r1_r4_seal_chain"])
    if (
        not isinstance(seal_chain, dict)
        or seal_chain.get("revisions") != ["r1", "r2", "r3", "r4"]
        or contents["prior_r1_r4_seal_chain"] != canonical_bytes(seal_chain) + b"\n"
    ):
        raise ValueError("r1-r4 seal chain preservation failed")
    predecessor = json.loads(contents["predecessor_state"])
    if (
        not isinstance(predecessor, dict)
        or predecessor.get("revision") != "r4"
        or predecessor.get("state") != "validation_screened"
        or contents["predecessor_state"] != canonical_bytes(predecessor) + b"\n"
    ):
        raise ValueError("r5 predecessor state is invalid")
    return contents, artifact_rows, extraction


def _verify_r5_design(
    root: Path,
    *,
    artifacts: R5ArtifactPaths,
    contract: R5CrossBindingContract,
    confirmation_contract: ConfirmationContract,
    authority: R5ConfirmationAuthority,
    artifact_rows: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    bundle = root / "00-manifest/r5-design-freeze"
    _r5_exact_bundle(bundle, _R5_DESIGN_FILES, label="r5 design")
    freeze, freeze_raw = _r5_json(bundle / "freeze.json", label="r5 freeze")
    screening, screening_raw = _r5_json(bundle / "screening.json", label="r5 screening")
    state, state_raw = _r5_json(bundle / "state.json", label="r5 design state")
    freeze_sha = _r5_sha256_bytes(freeze_raw)
    screening_sha = _r5_sha256_bytes(screening_raw)
    state_sha = _r5_sha256_bytes(state_raw)
    cross = _r5_contract_payload(contract)
    confirmation_contract_sha = _r5_sha256_payload(asdict(confirmation_contract))

    expected_authority = {
        "design_bundle_sha256": _r5_sha256_payload(
            {
                "candidate_freeze_sha256": freeze_sha,
                "design_screening_sha256": screening_sha,
                "design_state_sha256": state_sha,
            }
        ),
        "design_screening_sha256": screening_sha,
        "candidate_freeze_sha256": freeze_sha,
        "design_state_sha256": state_sha,
        "cross_binding_contract_sha256": _r5_sha256_payload(cross),
        "artifact_roots_sha256": _r5_sha256_payload(
            {
                "work_root": root.as_posix(),
                "artifacts": {
                    key: Path(value).as_posix()
                    for key, value in sorted(asdict(artifacts).items())
                },
            }
        ),
        "confirmation_contract_sha256": confirmation_contract_sha,
    }
    if _r5_authority_payload(authority) != expected_authority:
        raise ValueError("r5 confirmation authority mismatch")

    candidates = screening.get("candidates")
    expected_catalog = {row.candidate_id: row for row in candidate_catalog("r5")}
    if (
        screening.get("schema") != "r5_design_screening_v1"
        or screening.get("status") != "passed"
        or screening.get("candidate_count") != 12
        or screening.get("evaluated_count") != 12
        or screening.get("held_out_used") is not False
        or screening.get("precheck_ranges") != r5_precheck_ranges_payload()
        or screening.get("cross_bindings") != cross
        or screening.get("artifact_paths") != artifact_rows
        or not isinstance(candidates, list)
        or len(candidates) != 12
        or len({row.get("candidate_id") for row in candidates if isinstance(row, dict)}) != 12
    ):
        raise ValueError("r5 design cardinality/contract is invalid")
    accepted: list[tuple[float, str]] = []
    rejected = 0
    ranges = r5_precheck_ranges_payload()
    for row in candidates:
        candidate = expected_catalog.get(row.get("candidate_id")) if isinstance(row, dict) else None
        metrics = row.get("metrics") if isinstance(row, dict) else None
        if (
            candidate is None
            or row.get("config_sha256") != candidate.config_sha256
            or row.get("scenario_seed") != candidate.scenario_seed
            or not isinstance(metrics, dict)
            or set(metrics) != set(ranges)
            or not all(type(value) in (int, float) and math.isfinite(value) for value in metrics.values())
        ):
            raise ValueError("r5 design candidate identity/metric contract is invalid")
        reasons = r5_validation_precheck_reasons(metrics)
        status = "accepted" if not reasons else "rejected"
        if row.get("status") != status or row.get("reasons") != list(reasons):
            raise ValueError("r5 design acceptance-range recomputation mismatch")
        distance = math.sqrt(
            sum(
                (
                    (float(metrics[name]) - (bounds["minimum"] + bounds["maximum"]) / 2)
                    / (bounds["maximum"] - bounds["minimum"])
                )
                ** 2
                for name, bounds in ranges.items()
            )
            / len(ranges)
        )
        if not math.isclose(row.get("distance_to_range_midpoints", math.inf), distance, rel_tol=0, abs_tol=1e-12):
            raise ValueError("r5 design ranking distance mismatch")
        if reasons:
            rejected += 1
        else:
            accepted.append((distance, candidate.candidate_id))
    accepted.sort()
    selected = accepted[0][1] if accepted else None
    if (
        not accepted
        or screening.get("accepted_count") != len(accepted)
        or screening.get("rejected_count") != rejected
        or screening.get("selected_candidate_id") != selected
    ):
        raise ValueError("r5 design freeze selection mismatch")
    if (
        freeze.get("schema") != "r5_candidate_freeze_v1"
        or freeze.get("status") != "candidate_frozen"
        or freeze.get("candidate_id") != selected
        or freeze.get("config_sha256") != expected_catalog[selected].config_sha256
        or freeze.get("scenario_seed") != expected_catalog[selected].scenario_seed
        or freeze.get("design_screening_sha256") != screening_sha
        or freeze.get("cross_bindings") != cross
        or freeze.get("artifact_paths") != artifact_rows
        or freeze.get("confirmation_contract_sha256") != confirmation_contract_sha
    ):
        raise ValueError("r5 freeze record is invalid")
    expected_history = (
        "r5_created",
        "confirmation_cohort_prepared",
        "design_screened",
        "candidate_frozen",
    )
    history = state.get("history")
    if (
        state.get("schema") != "r5_confirmation_state_v1"
        or state.get("state") != "candidate_frozen"
        or state.get("cross_bindings") != cross
        or not isinstance(history, list)
        or tuple(row.get("state") for row in history if isinstance(row, dict)) != expected_history
        or [row.get("sequence") for row in history] != [1, 2, 3, 4]
        or [row.get("from_state") for row in history]
        != [None, "r5_created", "confirmation_cohort_prepared", "design_screened"]
        or history[0].get("details")
        != {"predecessor_state_sha256": cross["predecessor_state_sha256"]}
        or history[1].get("details")
        != {
            "confirmation_protocol_sha256": cross["confirmation_protocol_sha256"],
            "confirmation_block_csv_sha256": cross["confirmation_block_csv_sha256"],
        }
        or history[2].get("details") != {"design_screening_sha256": screening_sha}
        or history[3].get("details") != {"candidate_freeze_sha256": freeze_sha}
    ):
        raise ValueError("r5 design state history is illegal")
    return freeze, screening, state, {
        "design_freeze": freeze_sha,
        "design_screening": screening_sha,
        "design_state": state_sha,
    }


def _verify_r5_protocol(
    contents: dict[str, bytes],
    confirmation_contract: ConfirmationContract,
) -> tuple[dict[str, Any], dict[str, str]]:
    protocol_raw = contents["confirmation_protocol"]
    try:
        protocol = json.loads(protocol_raw, object_pairs_hook=_r5_reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ValueError("r5 confirmation protocol is malformed") from error
    if protocol_raw != canonical_bytes(protocol) + b"\n":
        raise ValueError("r5 confirmation protocol is not canonical")
    core = dict(protocol)
    protocol_sha = core.pop("protocol_sha256", None)
    if protocol_sha != _r5_sha256_payload(core):
        raise ValueError("r5 confirmation protocol digest mismatch")
    if set(protocol) != {
        "auxiliary_confirmation_only",
        "blocks",
        "dkasc_cohort_sha256",
        "dkasc_contract_site_ids",
        "dkasc_site_ids",
        "extraction_sha256",
        "oklahoma_cohort_sha256",
        "oklahoma_site_ids",
        "old_cohort_sha256",
        "prior_screen_exclusion_sha256",
        "protocol_sha256",
        "schema",
        "selection_seed",
        "source_series",
    }:
        raise ValueError("r5 confirmation protocol key set is invalid")
    blocks = protocol.get("blocks")
    dkasc_ids = tuple(protocol.get("dkasc_site_ids", ()))
    ok_ids = tuple(protocol.get("oklahoma_site_ids", ()))
    if (
        protocol.get("schema") != "nested_confirmation_protocol_v1"
        or protocol.get("auxiliary_confirmation_only") is not True
        or protocol.get("selection_seed") != 20260816
        or not isinstance(blocks, list)
        or len(blocks) != 27
        or len({row.get("block_id") for row in blocks if isinstance(row, dict)}) != 27
        or len(dkasc_ids) != 12
        or len(set(dkasc_ids)) != 12
        or tuple(sorted(dkasc_ids)) != dkasc_ids
        or tuple(protocol.get("dkasc_contract_site_ids", ())) != dkasc_ids
        or ok_ids != EXPECTED_AUXILIARY_OK_SITE_IDS
        or set(dkasc_ids) & set(ok_ids)
        or confirmation_contract.dkasc_site_ids != dkasc_ids
        or protocol.get("old_cohort_sha256") != confirmation_contract.old_cohort_sha256
        or protocol.get("extraction_sha256") != confirmation_contract.extraction_sha256
        or protocol.get("prior_screen_exclusion_sha256")
        != confirmation_contract.prior_screen_coordinates_sha256
    ):
        raise ValueError("r5 confirmation site/block identity contract is invalid")
    source_series = protocol.get("source_series")
    source_rows = {
        row["site_id"]: row
        for row in source_series or ()
        if isinstance(row, dict) and type(row.get("site_id")) is str
    }
    expected_source_series = json.loads(canonical_bytes(asdict(confirmation_contract)))[
        "source_series"
    ]
    if (
        source_series != expected_source_series
        or set(source_rows) != set(dkasc_ids) | set(ok_ids)
    ):
        raise ValueError("r5 confirmation source identity contract is invalid")
    counts = {"dkasc": 0, "oklahoma": 0}
    per_site: dict[str, int] = {}
    block_keys = {
        "block_id",
        "cohort",
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
    for row in blocks:
        if (
            not isinstance(row, dict)
            or set(row) != block_keys
            or row.get("cohort") not in counts
        ):
            raise ValueError("r5 confirmation block schema is invalid")
        cohort = row["cohort"]
        site_id = row.get("site_id")
        expected_ids = dkasc_ids if cohort == "dkasc" else ok_ids
        if (
            site_id not in expected_ids
            or row.get("split") != "validation"
            or "test" in str(row.get("spatial_role", "")).lower()
            or "held" in str(row.get("spatial_role", "")).lower()
            or row.get("n_steps") != 1344
            or not isinstance(row.get("raw_hashes"), list)
            or not row["raw_hashes"]
            or not all(_is_sha256(value) for value in row["raw_hashes"])
        ):
            raise ValueError("r5 confirmation block held-out/identity contract is invalid")
        start = _r5_iso(row.get("start"), label="r5 confirmation block start")
        end = _r5_iso(row.get("end"), label="r5 confirmation block end")
        if end - start != timedelta(minutes=15 * (1344 - 1)):
            raise ValueError("r5 confirmation block duration is invalid")
        if cohort == "dkasc":
            origin = _r5_iso(
                source_rows[site_id].get("source_origin_timestamp"),
                label="r5 confirmation source origin",
            )
            if start - origin < timedelta(days=14 + CONFIRMATION_EMBARGO_DAYS):
                raise ValueError("r5 confirmation 14-day embargo is invalid")
        counts[cohort] += 1
        per_site[site_id] = per_site.get(site_id, 0) + 1
    if (
        counts != {"dkasc": 12, "oklahoma": 15}
        or any(per_site.get(site_id) != 1 for site_id in dkasc_ids)
        or any(per_site.get(site_id) != 3 for site_id in ok_ids)
    ):
        raise ValueError("r5 confirmation block distribution is invalid")

    try:
        text = contents["confirmation_block_csv"].decode("utf-8")
        reader = csv.DictReader(text.splitlines())
        csv_rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError("r5 confirmation CSV is malformed") from error
    if tuple(reader.fieldnames or ()) != _R5_BLOCK_COLUMNS or len(csv_rows) != 27:
        raise ValueError("r5 confirmation CSV cardinality/schema is invalid")
    expected_rows = []
    for row in blocks:
        expected_rows.append(
            {
                key: (
                    ",".join(row[key])
                    if key == "raw_hashes"
                    else "" if row[key] is None else str(row[key])
                )
                for key in _R5_BLOCK_COLUMNS
            }
        )
    if csv_rows != expected_rows:
        raise ValueError("r5 confirmation CSV differs from protocol blocks")
    return protocol, {"confirmation_protocol": _r5_sha256_bytes(protocol_raw)}


def _verify_r5_terminal(
    root: Path,
    *,
    freeze: dict[str, Any],
    design_state: dict[str, Any],
    contract: R5CrossBindingContract,
    design_hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    bundle = root / "09-logs/r5-confirmation"
    _r5_exact_bundle(bundle, _R5_CONFIRMATION_FILES, label="r5 confirmation")
    decision, decision_raw = _r5_json(bundle / "decision.json", label="r5 decision")
    state, state_raw = _r5_json(bundle / "state.json", label="r5 confirmation state")
    decision_sha = _r5_sha256_bytes(decision_raw)
    state_sha = _r5_sha256_bytes(state_raw)
    cross = _r5_contract_payload(contract)
    expected = {row.candidate_id: row for row in candidate_catalog("r5")}
    candidate = expected.get(decision.get("candidate_id"))
    metrics = decision.get("metrics")
    config = decision.get("config")
    model_profile = decision.get("model_profile")
    decision_keys = {
        "candidate_freeze_sha256",
        "candidate_id",
        "config",
        "config_sha256",
        "cross_bindings",
        "design_state_sha256",
        "held_out_used",
        "metrics",
        "model_profile",
        "model_profile_sha256",
        "range_definition_sha256",
        "reasons",
        "scenario_seed",
        "schema",
        "status",
    }
    metrics_are_strict = (
        isinstance(metrics, dict)
        and set(metrics) == set(r5_precheck_ranges_payload())
        and all(
            type(value) in (int, float) and math.isfinite(value)
            for value in metrics.values()
        )
    )
    reasons = (
        r5_validation_precheck_reasons(metrics)
        if metrics_are_strict
        else ("invalid_metrics",)
    )
    expected_status = "failed_confirmation" if reasons else "formal_materializing"
    expected_config = (
        json.loads(canonical_bytes(asdict(candidate.config)))
        if candidate is not None
        else None
    )
    expected_profile = (
        json.loads(canonical_bytes(asdict(candidate.model_profile)))
        if candidate is not None
        else None
    )
    if (
        set(decision) != decision_keys
        or decision.get("schema") != "r5_confirmation_decision_v1"
        or decision.get("status") != expected_status
        or candidate is None
        or decision.get("candidate_id") != freeze.get("candidate_id")
        or not isinstance(config, dict)
        or canonical_bytes(config) != canonical_bytes(expected_config)
        or _r5_sha256_payload(config) != decision.get("config_sha256")
        or decision.get("config_sha256") != candidate.config_sha256
        or not isinstance(model_profile, dict)
        or canonical_bytes(model_profile) != canonical_bytes(expected_profile)
        or _r5_sha256_payload(model_profile)
        != decision.get("model_profile_sha256")
        or decision.get("model_profile_sha256")
        != _r5_sha256_payload(asdict(candidate.model_profile))
        or decision.get("scenario_seed") != candidate.scenario_seed
        or decision.get("range_definition_sha256") != r5_precheck_ranges_sha256()
        or decision.get("candidate_freeze_sha256") != design_hashes["design_freeze"]
        or decision.get("design_state_sha256") != design_hashes["design_state"]
        or decision.get("cross_bindings") != cross
        or decision.get("held_out_used") is not False
        or not metrics_are_strict
        or decision.get("reasons") != list(reasons)
    ):
        raise ValueError("r5 confirmation result/acceptance recomputation is invalid")
    history = state.get("history")
    expected_states = (
        "r5_created",
        "confirmation_cohort_prepared",
        "design_screened",
        "candidate_frozen",
        "confirmation_screened",
        expected_status,
    )
    expected_terminal_details = {
        "confirmation_decision_sha256": decision_sha,
        "design_state_sha256": design_hashes["design_state"],
    }
    if (
        state.get("schema") != "r5_confirmation_state_v1"
        or state.get("state") != expected_status
        or state.get("cross_bindings") != cross
        or state.get("design_state_sha256") != design_hashes["design_state"]
        or not isinstance(history, list)
        or history[:4] != design_state.get("history")
        or tuple(row.get("state") for row in history if isinstance(row, dict)) != expected_states
        or [row.get("sequence") for row in history] != [1, 2, 3, 4, 5, 6]
        or history[4].get("from_state") != "candidate_frozen"
        or history[5].get("from_state") != "confirmation_screened"
        or history[4].get("details") != expected_terminal_details
        or history[5].get("details") != expected_terminal_details
    ):
        raise ValueError("r5 one-shot confirmation state history is illegal")
    return decision, {
        "confirmation_decision": decision_sha,
        "confirmation_state": state_sha,
    }


def _r5_verification_destination(root: Path) -> Path:
    supplied = Path(root).expanduser()
    _r5_reject_symlinks(supplied.absolute())
    return supplied.resolve()


def _verify_r5_confirmation_core(
    destination: Path,
    *,
    artifacts: R5ArtifactPaths,
    cross_binding_contract: R5CrossBindingContract,
    confirmation_contract: ConfirmationContract,
    authority: R5ConfirmationAuthority,
) -> tuple[dict[str, int], dict[str, str], str]:
    """Validate either exact legal r5 terminal without publishing a verdict."""

    _r5_reject_symlinks(destination)
    if not destination.is_dir():
        raise ValueError("r5 root is missing")
    contents, artifact_rows, _extraction = _verify_r5_artifacts(
        destination, artifacts, cross_binding_contract
    )
    design_binding = json.loads(
        contents["design_cohort"], object_pairs_hook=_r5_reject_duplicate_keys
    )
    cohorts = design_binding.get("cohorts") if isinstance(design_binding, dict) else None
    design_blocks = [
        block
        for cohort in cohorts.values()
        for block in cohort.get("blocks", ())
        if isinstance(cohort, dict) and isinstance(block, dict)
    ] if isinstance(cohorts, dict) else []
    design_block_ids = [block.get("block_id") for block in design_blocks]
    if (
        contents["design_cohort"] != canonical_bytes(design_binding) + b"\n"
        or design_binding.get("schema") != "r5_design_cohort_binding_v1"
        or not cohorts
        or not design_blocks
        or len(design_block_ids) != len(set(design_block_ids))
        or any(
            block.get("split") != "validation"
            or "test" in str(block.get("spatial_role", "")).lower()
            or "held" in str(block.get("spatial_role", "")).lower()
            for block in design_blocks
        )
    ):
        raise ValueError("r5 design cohort held-out/identity contract is invalid")
    freeze, screening, design_state, design_hashes = _verify_r5_design(
        destination,
        artifacts=artifacts,
        contract=cross_binding_contract,
        confirmation_contract=confirmation_contract,
        authority=authority,
        artifact_rows=artifact_rows,
    )
    protocol, protocol_hashes = _verify_r5_protocol(contents, confirmation_contract)
    decision, terminal_hashes = _verify_r5_terminal(
        destination,
        freeze=freeze,
        design_state=design_state,
        contract=cross_binding_contract,
        design_hashes=design_hashes,
    )
    design_sites = {block.get("site_id") for block in design_blocks}
    if design_sites & set(protocol["oklahoma_site_ids"]):
        raise ValueError("r5 auxiliary and final/design sites are not disjoint")
    hashes = {
        **{name: _r5_sha256_bytes(raw) for name, raw in contents.items()},
        **design_hashes,
        **protocol_hashes,
        **terminal_hashes,
    }
    counts = {
        "confirmation_blocks": len(protocol["blocks"]),
        "confirmation_sites": len(protocol["dkasc_site_ids"])
        + len(protocol["oklahoma_site_ids"]),
        "design_candidates": len(screening["candidates"]),
        "freeze_records": 1,
        "confirmation_records": 1,
    }
    return counts, hashes, decision["status"]


def verify_r5_confirmation(
    root: Path,
    *,
    artifacts: R5ArtifactPaths,
    cross_binding_contract: R5CrossBindingContract,
    confirmation_contract: ConfirmationContract,
    authority: R5ConfirmationAuthority,
    publish: bool = True,
) -> R5VerificationReport:
    """Fail closed on every persisted r5 formal-success contract."""

    try:
        destination = _r5_verification_destination(root)
    except ValueError as error:
        raise ValueError(f"r5 verification failed: {error}") from error
    report_path = destination / _R5_REPORT_PATH
    if publish:
        _r5_reject_symlinks(report_path.parent)
        if report_path.is_symlink() or report_path.is_file():
            report_path.unlink()
        elif report_path.exists():
            raise ValueError("r5 verification failed: report path is not a regular file")
        report_temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        if report_temporary.exists() or report_temporary.is_symlink():
            raise ValueError("r5 verification failed: partial verification report exists")
    errors: list[str] = []
    hashes: dict[str, str] = {}
    counts = {
        "confirmation_blocks": 0,
        "confirmation_sites": 0,
        "design_candidates": 0,
        "freeze_records": 0,
        "confirmation_records": 0,
    }
    try:
        counts, hashes, terminal_status = _verify_r5_confirmation_core(
            destination,
            artifacts=artifacts,
            cross_binding_contract=cross_binding_contract,
            confirmation_contract=confirmation_contract,
            authority=authority,
        )
        if terminal_status != "formal_materializing":
            raise ValueError("r5 confirmation is intact but not formally successful")
    except Exception as error:  # noqa: BLE001 - verifier must persist fail-closed evidence
        errors.append(f"{type(error).__name__}: {error}")
    report = R5VerificationReport(
        ok=not errors,
        status="passed" if not errors else "failed",
        errors=tuple(errors),
        counts=counts,
        hashes=hashes,
    )
    if publish:
        write_json(report_path, report.to_payload())
    if errors:
        raise ValueError("r5 verification failed: " + "; ".join(errors))
    return report


def validate_r5_terminal_integrity(
    root: Path,
    *,
    artifacts: R5ArtifactPaths,
    cross_binding_contract: R5CrossBindingContract,
    confirmation_contract: ConfirmationContract,
    authority: R5ConfirmationAuthority,
) -> R5TerminalIntegrityResult:
    """Read-only validation of either exact legal terminal branch."""

    try:
        destination = _r5_verification_destination(root)
        counts, hashes, terminal_status = _verify_r5_confirmation_core(
            destination,
            artifacts=artifacts,
            cross_binding_contract=cross_binding_contract,
            confirmation_contract=confirmation_contract,
            authority=authority,
        )
    except Exception as error:
        raise ValueError(
            f"r5 verification failed: {type(error).__name__}: {error}"
        ) from error
    return R5TerminalIntegrityResult(
        terminal_status=terminal_status,
        counts=counts,
        hashes=hashes,
    )


def _r5_replay_paths(root: Path, artifacts: R5ArtifactPaths) -> dict[str, Path]:
    paths = {
        f"00-manifest/r5-design-freeze/{name}": root
        / "00-manifest/r5-design-freeze"
        / name
        for name in _R5_DESIGN_FILES
    }
    paths.update(
        {
            f"09-logs/r5-confirmation/{name}": root
            / "09-logs/r5-confirmation"
            / name
            for name in _R5_CONFIRMATION_FILES
        }
    )
    paths[_R5_REPORT_PATH] = root / _R5_REPORT_PATH
    paths.update(
        {
            f"independent/{name}": path
            for name, path in _r5_artifact_map(artifacts).items()
            if path.suffix in {".json", ".csv"}
        }
    )
    return paths


def verify_r5_deterministic_replay(
    first_root: Path,
    second_root: Path,
    *,
    first_artifacts: R5ArtifactPaths,
    second_artifacts: R5ArtifactPaths,
) -> dict[str, Any]:
    """Compare every r5 contracted JSON/CSV/report byte across isolated roots."""

    first_supplied = Path(first_root).expanduser()
    second_supplied = Path(second_root).expanduser()
    for supplied in (first_supplied, second_supplied):
        try:
            _r5_reject_symlinks(supplied.absolute())
        except ValueError as error:
            raise ValueError(
                f"r5 deterministic replay symlink root is forbidden: {supplied}"
            ) from error
    first = first_supplied.resolve()
    second = second_supplied.resolve()
    if first == second or first.is_relative_to(second) or second.is_relative_to(first):
        raise ValueError("r5 deterministic replay roots must be distinct and isolated")
    for replay_root in (first, second):
        try:
            _r5_exact_bundle(
                replay_root / "00-manifest/r5-design-freeze",
                _R5_DESIGN_FILES,
                label="r5 design",
            )
            _r5_exact_bundle(
                replay_root / "09-logs/r5-confirmation",
                _R5_CONFIRMATION_FILES,
                label="r5 confirmation",
            )
        except ValueError as error:
            raise ValueError(
                f"r5 deterministic replay bundle entry mismatch: {error}"
            ) from error
    first_paths = _r5_replay_paths(first, first_artifacts)
    second_paths = _r5_replay_paths(second, second_artifacts)
    if set(first_paths) != set(second_paths):
        raise ValueError("r5 deterministic replay file-set mismatch")
    hashes: dict[str, str] = {}
    first_inodes: set[tuple[int, int]] = set()
    second_inodes: set[tuple[int, int]] = set()
    for relative in sorted(first_paths):
        left = first_paths[relative]
        right = second_paths[relative]
        left_raw = _r5_regular_bytes(left, label=f"first replay {relative}")
        right_raw = _r5_regular_bytes(right, label=f"second replay {relative}")
        first_stat, second_stat = left.stat(), right.stat()
        first_inodes.add((first_stat.st_dev, first_stat.st_ino))
        second_inodes.add((second_stat.st_dev, second_stat.st_ino))
        if left_raw != right_raw:
            raise ValueError(f"r5 deterministic replay byte mismatch: {relative}")
        hashes[relative] = _r5_sha256_bytes(left_raw)
    if first_inodes & second_inodes:
        raise ValueError("r5 deterministic replay roots share a linked artifact inode")
    return {"matched_files": len(hashes), "hashes": hashes, "excluded_fields": []}


def _read_json(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"{label} hash/invariant unreadable: {error}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"{label} invariant is not an object")
        return {}
    return payload


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _has_evidence_markers(payload: dict[str, Any]) -> bool:
    return all(payload.get(key) == value for key, value in EVIDENCE_METADATA.items())


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_hash_mapping(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(key, str) and _is_sha256(digest)
        for key, digest in value.items()
    )


def _safe_hash(path: Path, label: str, errors: list[str]) -> str:
    if not path.is_file():
        errors.append(f"{label} hash invariant missing")
        return ""
    try:
        return sha256_file(path)
    except OSError as error:
        errors.append(f"{label} hash invariant unreadable: {error}")
        return ""


def _verify_preflight_hashes(root: Path, errors: list[str]) -> dict[str, Any]:
    report_path = root / "00-manifest/preflight-4a.json"
    index_path = root / "09-logs/preflight-4a-artifacts.json"
    report = _read_json(report_path, "4A preflight", errors)
    index = _read_json(index_path, "4A artifact index", errors)
    entries = index.get("artifacts", [])
    if not isinstance(entries, list) or not entries:
        errors.append("4A artifact hash index invariant failure")
        return report
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            errors.append("4A artifact hash entry invariant failure")
            continue
        relative = entry["path"]
        path = root / relative
        if (
            relative in seen
            or not path.is_file()
            or entry.get("sha256") != _safe_hash(path, relative, errors)
        ):
            errors.append(f"4A artifact hash mismatch: {relative}")
        seen.add(relative)
    declared = set(report.get("artifacts", {}).values())
    if declared and not declared.issubset(seen):
        errors.append("4A declared artifact hash coverage invariant failure")
    contract_relative = report.get("artifacts", {}).get("data_source_contract")
    if isinstance(contract_relative, str):
        contract = _read_json(root / contract_relative, "4A source contract", errors)
        for name, source in contract.get("sources", {}).items():
            path = Path(str(source.get("path", ""))).expanduser().resolve()
            if source.get("sha256") != _safe_hash(
                path, f"4A source {name}", errors
            ):
                errors.append(f"4A source hash mismatch: {name}")
    storage = report.get("storage", {})
    if (
        report.get("status") != "passed"
        or report.get("generation_ready") is not True
        or storage.get("schema") != "compact_shared_v2"
        or storage.get("prediction_archives") != 420
        or storage.get("sensitivity_aliases") != 90
        or set(storage.get("per_protocol", {})) != set(PROTOCOLS)
        or any(
            storage["per_protocol"][protocol].get("prediction_archives") != 210
            or storage["per_protocol"][protocol].get("sensitivity_aliases") != 45
            for protocol in PROTOCOLS
        )
    ):
        errors.append("4A storage/count invariant failure")
    return report


def _csv_rows(path: Path, errors: list[str]) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as error:
        errors.append(f"4A identity hash/invariant unreadable: {path}: {error}")
        return []


def _verify_4a_identities(
    root: Path,
    preflight: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    errors: list[str],
) -> None:
    artifacts = preflight.get("artifacts", {})
    site_path = root / str(artifacts.get("site_inventory", ""))
    block_path = root / str(artifacts.get("selected_blocks", ""))
    sites = _csv_rows(site_path, errors) if site_path.is_file() else []
    blocks = _csv_rows(block_path, errors) if block_path.is_file() else []
    if not sites or not blocks:
        errors.append("4A exact site/block identity invariant failure")
        return
    site_identities = [
        (row.get("protocol", ""), row.get("site_id", "")) for row in sites
    ]
    if (
        len(site_identities) != len(set(site_identities))
        or any(protocol not in PROTOCOLS or not site for protocol, site in site_identities)
        or preflight.get("site_count") != len(site_identities)
    ):
        errors.append("4A site identity duplicate/cardinality invariant failure")

    def normalized_site_identity(row: dict[str, Any]) -> tuple[Any, ...]:
        raw_splits = row.get("required_splits", "")
        splits = (
            raw_splits
            if isinstance(raw_splits, list)
            else [item for item in str(raw_splits).split(",") if item]
        )
        return (
            str(row.get("protocol", "")),
            str(row.get("site_id", "")),
            str(row.get("group", "")),
            str(row.get("source_kind", "")),
            tuple(splits),
        )

    contract_path = root / str(artifacts.get("data_source_contract", ""))
    contract = (
        _read_json(contract_path, "4A data source contract", errors)
        if contract_path.is_file()
        else {}
    )
    contract_sites = contract.get("sites", [])
    inventory_contract_identities = [normalized_site_identity(row) for row in sites]
    contract_identities = (
        [normalized_site_identity(row) for row in contract_sites]
        if isinstance(contract_sites, list)
        and all(isinstance(row, dict) for row in contract_sites)
        else []
    )
    if (
        len(contract_identities) != len(inventory_contract_identities)
        or len(contract_identities) != len(set(contract_identities))
        or set(contract_identities) != set(inventory_contract_identities)
        or contract.get("site_count") != len(contract_identities)
    ):
        errors.append("4A source-contract site identity invariant failure")
    expected_sites: dict[tuple[str, str], set[str]] = {}
    for row in sites:
        protocol = row.get("protocol", "")
        splits = [split for split in row.get("required_splits", "").split(",") if split]
        if not splits or len(splits) != len(set(splits)) or set(splits) - {
            "validation",
            "calibration",
            "test",
        }:
            errors.append("4A site split identity invariant failure")
        for split in splits:
            if split:
                expected_sites.setdefault((protocol, split), set()).add(
                    row.get("site_id", "")
                )
    block_identities = [
        (
            row.get("protocol", ""),
            row.get("split", ""),
            row.get("block_id", ""),
            row.get("site_id", ""),
            row.get("start", ""),
            row.get("end", ""),
        )
        for row in blocks
    ]
    expected_blocks = set(block_identities)
    if (
        len(block_identities) != len(expected_blocks)
        or any(
            protocol not in PROTOCOLS
            or split not in {"validation", "calibration", "test"}
            or not block_id
            or not site_id
            for protocol, split, block_id, site_id, _start, _end in block_identities
        )
        or preflight.get("selected_block_count") != len(block_identities)
    ):
        errors.append("4A block identity duplicate/cardinality invariant failure")
    actual_blocks: set[tuple[str, str, str, str, str, str]] = set()
    for protocol, manifest in manifests.items():
        cohort_path = root / "02-predictions" / protocol / str(
            manifest.get("cohort_index_path", "")
        )
        cohort_index = _read_json(cohort_path, f"{protocol} cohort index", errors)
        for split, cohort in cohort_index.get("cohorts", {}).items():
            raw_actual_sites = cohort.get("site_ids", [])
            if (
                not isinstance(raw_actual_sites, list)
                or not raw_actual_sites
                or any(not isinstance(site, str) or not site for site in raw_actual_sites)
                or len(raw_actual_sites) != len(set(raw_actual_sites))
            ):
                errors.append(
                    f"4A cohort site cardinality invariant: {protocol}/{split}"
                )
            actual_sites = (
                set(map(str, raw_actual_sites))
                if isinstance(raw_actual_sites, list)
                else set()
            )
            if actual_sites != expected_sites.get((protocol, split), set()):
                errors.append(f"4A site identity invariant mismatch: {protocol}/{split}")
            raw_actual_blocks = cohort.get("blocks", [])
            cohort_block_identities = (
                [
                    (
                        protocol,
                        split,
                        str(block.get("block_id", "")),
                        str(block.get("site_id", "")),
                        str(block.get("start", "")),
                        str(block.get("end", "")),
                    )
                    for block in raw_actual_blocks
                    if isinstance(block, dict)
                ]
                if isinstance(raw_actual_blocks, list)
                else []
            )
            if (
                not isinstance(raw_actual_blocks, list)
                or len(cohort_block_identities) != len(raw_actual_blocks)
                or len(cohort_block_identities) != len(set(cohort_block_identities))
            ):
                errors.append(
                    f"4A cohort block cardinality invariant: {protocol}/{split}"
                )
            actual_blocks.update(cohort_block_identities)
    if actual_blocks != expected_blocks:
        errors.append("4A block identity invariant mismatch")
    smoke = all(manifest.get("smoke") is True for manifest in manifests.values())
    if not smoke:
        recomputed_groups = {
            group: sum(row.get("group") == group for row in sites)
            for group in ("Arizona", "DKASC", "Oklahoma", "Yulara")
        }
        if preflight.get("group_counts") != recomputed_groups:
            errors.append("4A group count recomputation invariant failure")
    if not smoke and (
        len(site_identities) != 57
        or len({site for _protocol, site in site_identities}) != 57
        or len(block_identities) != 252
        or sum(protocol == "dkasc-yulara" for protocol, _site in site_identities) != 17
        or sum(protocol == "nrel-transfer" for protocol, _site in site_identities) != 40
        or len(expected_sites.get(("dkasc-yulara", "validation"), set())) != 12
        or len(expected_sites.get(("dkasc-yulara", "calibration"), set())) != 12
        or len(expected_sites.get(("dkasc-yulara", "test"), set())) != 5
        or len(expected_sites.get(("nrel-transfer", "validation"), set())) != 20
        or len(expected_sites.get(("nrel-transfer", "calibration"), set())) != 20
        or len(expected_sites.get(("nrel-transfer", "test"), set())) != 20
        or preflight.get("group_counts")
        != {"Arizona": 20, "DKASC": 12, "Oklahoma": 20, "Yulara": 5}
    ):
        errors.append("formal 57-site/252-block invariant failure")


def _final_state_details(state: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    history = state.get("history", [])
    if (
        state.get("state") not in {"candidate_accepted", "verified", "promotion_eligible"}
        or not isinstance(history, list)
        or not history
    ):
        errors.append("acceptance revision-state invariant failure")
        return {}
    if (
        state.get("schema") != "realism_revision_state_v1"
        or not _has_evidence_markers(state)
        or not _is_hash_mapping(state.get("source_hashes"))
    ):
        errors.append("revision-state schema/evidence/source invariant failure")
    initial_details = history[0].get("details") if isinstance(history[0], dict) else None
    rollover_in_history = (
        isinstance(initial_details, dict) and "catalog_rollover" in initial_details
    )
    rollover_top_level_keys = {
        "catalog_revision",
        "catalog_sha256",
        "superseded_revision_state",
    }
    later_catalog_provenance = any(
        isinstance(entry, dict)
        and entry.get("state") == "candidate_accepted"
        and isinstance(entry.get("details"), dict)
        and entry["details"].get("candidate_config_sha256")
        in _LATER_CATALOG_CONFIG_SHA256S
        for entry in history
    )
    if (
        bool(set(state) & rollover_top_level_keys) or later_catalog_provenance
    ) and not rollover_in_history:
        errors.append("revision-state catalog rollover presence invariant failure")
    if rollover_in_history:
        rollover = initial_details.get("catalog_rollover")
        expected_state_keys = {
            *EVIDENCE_METADATA,
            "disclosure",
            "schema",
            "state",
            "source_hashes",
            "history",
            "catalog_revision",
            "catalog_sha256",
            "superseded_revision_state",
        }
        if (
            not _valid_catalog_rollover_details(initial_details)
            or not isinstance(rollover, dict)
            or set(state) != expected_state_keys
            or state.get("catalog_revision") != rollover.get("to_revision")
            or state.get("catalog_sha256") != rollover.get("new_catalog_sha256")
            or state.get("superseded_revision_state") != rollover.get("predecessor")
        ):
            errors.append(
                "revision-state catalog rollover top-level invariant failure"
            )
    previous = None
    active_candidate: str | None = None
    active_config_sha256: str | None = None
    active_source_hashes: dict[str, str] = {}
    evaluated_details: dict[str, Any] = {}
    for sequence, entry in enumerate(history, start=1):
        if not isinstance(entry, dict):
            errors.append("revision-state history structure invariant failure")
            continue
        current = entry.get("state")
        if (
            entry.get("sequence") != sequence
            or entry.get("from_state") != previous
            or current not in REVISION_STATES
        ):
            errors.append("revision-state history invariant failure")
        if sequence == 1:
            if current != "created" or (
                entry.get("details") != {}
                and not _valid_catalog_rollover_details(entry.get("details"))
            ):
                errors.append("revision-state initial transition invariant failure")
        elif current not in _TRANSITIONS.get(str(previous), ()):
            errors.append(
                f"revision-state illegal transition invariant: {previous} -> {current}"
            )
        details = entry.get("details")
        if not isinstance(details, dict):
            errors.append("revision-state transition detail invariant failure")
            details = {}
        expected_detail_keys = {
            "created": (
                set(details)
                if _valid_catalog_rollover_details(details)
                else set()
            ),
            "validation_screened": {
                "screening_report_sha256",
                "screening_status",
            },
            "candidate_materializing": {
                "candidate_id",
                "candidate_config_sha256",
                "screening_report_sha256",
                "materialization_source_hashes",
            },
            "candidate_evaluated": {
                "candidate_id",
                "candidate_config_sha256",
                "candidate_acceptance_sha256",
                "candidate_gate_passed",
                "candidate_gate_reasons",
                "candidate_gate_derived_sha256",
                "materialization_source_hashes",
            },
            "candidate_rejected": {
                "candidate_id",
                "candidate_config_sha256",
                "candidate_acceptance_sha256",
                "candidate_decision_sha256",
                "rejection_reasons",
                "materialization_source_hashes",
            },
            "candidate_accepted": {
                "candidate_id",
                "candidate_config_sha256",
                "candidate_acceptance_sha256",
                "candidate_gate_derived_sha256",
                "materialization_source_hashes",
            },
        }.get(str(current))
        if expected_detail_keys is not None and set(details) != expected_detail_keys:
            errors.append(f"revision-state transition detail schema invariant: {current}")
        if current == "validation_screened" and (
            not _is_sha256(details.get("screening_report_sha256"))
            or details.get("screening_status") != "passed"
        ):
            errors.append("revision-state screening transition detail invariant")
        elif current == "candidate_materializing":
            active_candidate = details.get("candidate_id")
            active_config_sha256 = details.get("candidate_config_sha256")
            active_source_hashes = dict(
                details.get("materialization_source_hashes", {})
            )
            evaluated_details = {}
            if (
                not isinstance(active_candidate, str)
                or not active_candidate
                or not _is_sha256(details.get("candidate_config_sha256"))
                or not _is_sha256(details.get("screening_report_sha256"))
                or not _is_hash_mapping(details.get("materialization_source_hashes"))
            ):
                errors.append("revision-state materializing detail invariant")
        elif current == "candidate_evaluated":
            evaluated_details = details
            if (
                details.get("candidate_id") != active_candidate
                or not _is_sha256(details.get("candidate_config_sha256"))
                or details.get("candidate_config_sha256") != active_config_sha256
                or not _is_sha256(details.get("candidate_acceptance_sha256"))
                or not isinstance(details.get("candidate_gate_passed"), bool)
                or not isinstance(details.get("candidate_gate_reasons"), list)
                or not all(
                    isinstance(reason, str)
                    for reason in details.get("candidate_gate_reasons", [])
                )
                or not _is_hash_mapping(details.get("candidate_gate_derived_sha256"))
                or not _is_hash_mapping(details.get("materialization_source_hashes"))
                or details.get("materialization_source_hashes")
                != active_source_hashes
            ):
                errors.append("revision-state evaluated immutable chain invariant")
        elif current in {"candidate_rejected", "candidate_accepted"}:
            is_accepted = current == "candidate_accepted"
            reasons = details.get("rejection_reasons", [])
            if (
                details.get("candidate_id") != active_candidate
                or not _is_sha256(details.get("candidate_config_sha256"))
                or details.get("candidate_config_sha256") != active_config_sha256
                or not _is_sha256(details.get("candidate_acceptance_sha256"))
                or not _is_hash_mapping(details.get("materialization_source_hashes"))
                or details.get("materialization_source_hashes")
                != active_source_hashes
                or details.get("candidate_config_sha256")
                != evaluated_details.get("candidate_config_sha256")
                or details.get("candidate_acceptance_sha256")
                != evaluated_details.get("candidate_acceptance_sha256")
                or (is_accepted and evaluated_details.get("candidate_gate_passed") is not True)
                or (
                    is_accepted
                    and details.get("candidate_gate_derived_sha256")
                    != evaluated_details.get("candidate_gate_derived_sha256")
                )
                or (
                    not is_accepted
                    and (
                        not _is_sha256(details.get("candidate_decision_sha256"))
                        or evaluated_details.get("candidate_gate_passed") is not False
                        or not isinstance(reasons, list)
                        or reasons != evaluated_details.get("candidate_gate_reasons")
                    )
                )
            ):
                errors.append(f"revision-state {current} detail invariant")
            if not is_accepted:
                active_candidate = None
                active_config_sha256 = None
                active_source_hashes = {}
                evaluated_details = {}
        bound = details.get("materialization_source_hashes")
        source_base = bound if isinstance(bound, dict) else state.get("source_hashes", {})
        expected_sources = dict(source_base) if isinstance(source_base, dict) else {}
        detail_sources = {
            key: value
            for key, value in sorted(details.items())
            if key.endswith("_sha256") and value is not None
        }
        actual_sources = entry.get("source_hashes")
        source_keys = set(expected_sources) | set(detail_sources)
        valid_sources = isinstance(actual_sources, dict) and set(actual_sources) == source_keys
        if valid_sources:
            valid_sources = all(
                actual_sources.get(key) == value for key, value in expected_sources.items()
            )
        if valid_sources:
            for key, value in detail_sources.items():
                actual = actual_sources.get(key)
                if isinstance(value, dict) and isinstance(actual, str):
                    try:
                        actual = ast.literal_eval(actual)
                    except (SyntaxError, ValueError):
                        pass
                if actual != (value if isinstance(value, dict) else str(value)):
                    valid_sources = False
                    break
        if not valid_sources:
            errors.append(f"revision-state transition source invariant: {current}")
        previous = current
    if previous != state.get("state"):
        errors.append("revision-state terminal invariant failure")
    accepted = [entry for entry in history if entry.get("state") == "candidate_accepted"]
    if len(accepted) != 1:
        errors.append("acceptance revision-state cardinality invariant failure")
        return {}
    return dict(accepted[0].get("details", {}))


def _verify_state_sources_and_decision(
    root: Path,
    source: Path,
    acceptance_sha256: str,
    errors: list[str],
) -> dict[str, Any]:
    state = _read_json(root / "09-logs/revision-state.json", "revision state", errors)
    details = _final_state_details(state, errors)
    try:
        current_source_hashes = _source_root_hashes(source)
    except ValueError as error:
        errors.append(f"source hash invariant mismatch: {error}")
        current_source_hashes = {}
    recorded = details.get("materialization_source_hashes", {})
    recorded_source = {
        key: value
        for key, value in recorded.items()
        if str(key).startswith("source_root:")
    }
    if recorded_source != current_source_hashes:
        errors.append("source hash invariant mismatch")
    if details.get("candidate_acceptance_sha256") != acceptance_sha256:
        errors.append("acceptance hash mismatch in revision state")

    screening_path = root / "00-manifest/realism-candidate-screening.json"
    screening = _read_json(screening_path, "candidate screening", errors)
    screening_hash = _safe_hash(screening_path, "candidate screening", errors)
    if (
        screening.get("schema") != "realism_candidate_screening_v1"
        or not _has_evidence_markers(screening)
    ):
        errors.append("candidate screening schema/evidence invariant failure")
    screened_entries = [
        entry
        for entry in state.get("history", [])
        if isinstance(entry, dict) and entry.get("state") == "validation_screened"
    ]
    top_sources = state.get("source_hashes", {})
    screened_sources = (
        {
            key: value
            for key, value in screened_entries[0].get("source_hashes", {}).items()
            if key != "screening_report_sha256"
        }
        if screened_entries
        else {}
    )
    if (
        top_sources != screened_sources
        or not isinstance(recorded, dict)
        or any(recorded.get(key) != value for key, value in top_sources.items())
    ):
        errors.append("revision-state top-level source invariant failure")
    if (
        len(screened_entries) != 1
        or screened_entries[0].get("details", {}).get("screening_report_sha256")
        != screening_hash
        or screened_entries[0].get("details", {}).get("screening_status") != "passed"
        or screened_entries[0].get("source_hashes", {}).get(
            "screening_report_sha256"
        )
        != screening_hash
    ):
        errors.append("candidate screening/state hash invariant failure")
    candidate_id = details.get("candidate_id")
    candidates = [
        row
        for row in screening.get("candidates", [])
        if row.get("candidate_id") == candidate_id
    ]
    if (
        screening.get("status") != "passed"
        or screening.get("selected_candidate_id") != candidate_id
        or len(candidates) != 1
        or candidates[0].get("status") != "accepted"
    ):
        errors.append("candidate screening acceptance invariant failure")
        candidate = {}
    else:
        candidate = candidates[0]
    try:
        config = ScenarioGenerationConfig(**candidate.get("config", {}))
        actual_config_hash = canonical_config_sha256(config)
    except (TypeError, ValueError):
        actual_config_hash = ""
    if (
        not actual_config_hash
        or candidate.get("config_sha256") != actual_config_hash
        or details.get("candidate_config_sha256") != actual_config_hash
    ):
        errors.append("candidate configuration hash invariant mismatch")
    materializing_entries = [
        entry
        for entry in state.get("history", [])
        if isinstance(entry, dict) and entry.get("state") == "candidate_materializing"
    ]
    if any(
        entry.get("details", {}).get("screening_report_sha256") != screening_hash
        for entry in materializing_entries
    ):
        errors.append("candidate materializing screening hash invariant failure")
    journal_path = (
        root
        / "candidates"
        / str(candidate_id)
        / "09-logs"
        / "materialization-journal.json"
    )
    journal = _read_json(journal_path, "materialization journal", errors)
    current_gate_hashes = {
        relative: _safe_hash(
            journal_path if relative == "09-logs/materialization-journal.json" else root / relative,
            f"accepted gate {relative}",
            errors,
        )
        for relative in _DERIVED_GATE_FILES
    }
    if details.get("candidate_gate_derived_sha256") != current_gate_hashes:
        errors.append("accepted candidate derived hash invariant failure")
    expected_stage_markers = {
        **{
            f"{protocol}:{stage}": f"02-predictions/{protocol}/{filename}"
            for protocol in PROTOCOLS
            for stage, filename in (
                ("prepare", "manifest.json"),
                ("generate", "candidate-audit.json"),
                ("evaluate", "evaluations.json"),
                ("aggregate", "aggregate.json"),
            )
        },
        "unified-4d": "04-aggregates/acceptance.json",
    }
    completed_stages = journal.get("completed_stages")
    if (
        journal.get("schema") != "realism_materialization_journal_v1"
        or not _has_evidence_markers(journal)
        or journal.get("candidate_id") != candidate_id
        or journal.get("candidate_config_sha256") != actual_config_hash
        or journal.get("screening_report_sha256") != screening_hash
        or journal.get("materialization_source_hashes") != recorded
        or not isinstance(completed_stages, dict)
        or set(completed_stages) != set(expected_stage_markers)
    ):
        errors.append("materialization journal hash/invariant failure")
    if isinstance(completed_stages, dict):
        for stage, relative in expected_stage_markers.items():
            record = completed_stages.get(stage)
            marker = root / relative
            expected_hashes = (
                {relative: _safe_hash(marker, f"journal {stage}", errors)}
                if marker.is_file()
                else {}
            )
            if (
                not isinstance(record, dict)
                or set(record) - {"status", "artifact_hashes", "recovered_after_interruption"}
                or record.get("status") != "completed"
                or record.get("artifact_hashes") != expected_hashes
                or (
                    "recovered_after_interruption" in record
                    and record.get("recovered_after_interruption") is not True
                )
            ):
                errors.append(f"materialization journal stage invariant failure: {stage}")

    decisions_path = root / "09-logs/candidate-decisions.json"
    decisions = (
        _read_json(decisions_path, "candidate decisions", errors)
        if decisions_path.is_file()
        else {"decisions": []}
    )
    if decisions_path.is_file() and (
        decisions.get("schema") != "realism_candidate_decisions_v1"
        or not _has_evidence_markers(decisions)
    ):
        errors.append("candidate decision schema/evidence invariant failure")
    decision_rows = decisions.get("decisions", [])
    if not isinstance(decision_rows, list):
        errors.append("candidate decision invariant failure")
        decision_rows = []
    elif any(not isinstance(row, dict) for row in decision_rows):
        errors.append("candidate decision row structure invariant failure")
        decision_rows = [row for row in decision_rows if isinstance(row, dict)]
    rejected_entries = [
        entry for entry in state.get("history", []) if entry.get("state") == "candidate_rejected"
    ]
    expected_rejected = {
        entry.get("details", {}).get("candidate_id"): entry.get("details", {})
        for entry in rejected_entries
    }
    rejected_rows = [row for row in decision_rows if row.get("status") == "rejected"]
    if (
        len(expected_rejected) != len(rejected_entries)
        or None in expected_rejected
        or len(rejected_rows) != len(expected_rejected)
        or {row.get("candidate_id") for row in rejected_rows} != set(expected_rejected)
    ):
        errors.append("rejected candidate decision cardinality invariant failure")
    for decision in rejected_rows:
        rejected_id = decision.get("candidate_id")
        rejected_details = expected_rejected.get(rejected_id, {})
        try:
            rejected_config_hash = canonical_config_sha256(
                ScenarioGenerationConfig(**decision.get("config", {}))
            )
        except (TypeError, ValueError):
            rejected_config_hash = ""
        rejected_acceptance = (
            root
            / "candidates"
            / str(rejected_id)
            / "04-aggregates"
            / "acceptance.json"
        )
        rejected_acceptance_payload = _read_json(
            rejected_acceptance,
            f"rejected candidate {rejected_id} acceptance",
            errors,
        )
        rejected_aggregate = _read_json(
            root
            / "candidates"
            / str(rejected_id)
            / "04-aggregates"
            / "unified-aggregate.json",
            f"rejected candidate {rejected_id} aggregate",
            errors,
        )
        exact_rejected_keys = {
            *EVIDENCE_METADATA,
            "disclosure",
            "candidate_id",
            "status",
            "unmet_checks",
            "config",
            "config_sha256",
            "acceptance",
            "acceptance_sha256",
            "metrics",
            "source_manifests",
            "source_manifests_sha256",
            "derived_acceptance_provenance",
            "artifact_index_roots",
            "artifact_index_sha256",
            "candidate_manifest_sha256",
            "per_protocol_bundle_sha256",
            "artifact_root_sha256",
            "storage_size_bytes",
            "prediction_storage_size_bytes",
            "removed_path",
        }
        protocol_hash_fields = (
            "artifact_index_roots",
            "artifact_index_sha256",
            "candidate_manifest_sha256",
            "per_protocol_bundle_sha256",
        )
        derived_provenance = decision.get("derived_acceptance_provenance", {})
        derived_hashes = (
            derived_provenance.get("derived_artifact_sha256", {})
            if isinstance(derived_provenance, dict)
            else {}
        )
        if (
            set(decision) != exact_rejected_keys
            or not _has_evidence_markers(decision)
            or decision.get("disclosure") != DISCLOSURE
            or not rejected_config_hash
            or decision.get("config_sha256") != rejected_config_hash
            or rejected_details.get("candidate_config_sha256") != rejected_config_hash
            or decision.get("acceptance") != rejected_acceptance_payload
            or decision.get("acceptance_sha256")
            != rejected_details.get("candidate_acceptance_sha256")
            or rejected_details.get("candidate_decision_sha256")
            != hashlib.sha256(canonical_bytes(decision)).hexdigest()
            or decision.get("acceptance_sha256")
            != _safe_hash(
                rejected_acceptance,
                f"rejected candidate {rejected_id} acceptance",
                errors,
            )
            or decision.get("unmet_checks") != rejected_details.get("rejection_reasons")
            or decision.get("metrics") != rejected_aggregate.get("values")
            or decision.get("source_manifests")
            != rejected_details.get("materialization_source_hashes")
            or decision.get("source_manifests_sha256")
            != hashlib.sha256(canonical_bytes(decision.get("source_manifests", {}))).hexdigest()
            or decision.get("removed_path")
            != f"candidates/{rejected_id}/02-predictions"
            or not isinstance(derived_provenance, dict)
            or not _has_evidence_markers(derived_provenance)
            or not _is_hash_mapping(derived_hashes)
            or derived_hashes.get("04-aggregates/acceptance.json")
            != decision.get("acceptance_sha256")
            or any(
                not _is_hash_mapping(decision.get(field))
                or set(decision.get(field, {})) != set(PROTOCOLS)
                for field in protocol_hash_fields
            )
            or not _is_sha256(decision.get("artifact_root_sha256"))
            or isinstance(decision.get("storage_size_bytes"), bool)
            or not isinstance(decision.get("storage_size_bytes"), int)
            or isinstance(decision.get("prediction_storage_size_bytes"), bool)
            or not isinstance(decision.get("prediction_storage_size_bytes"), int)
            or decision.get("storage_size_bytes", 0)
            <= decision.get("prediction_storage_size_bytes", 0)
            or decision.get("prediction_storage_size_bytes", 0) <= 0
        ):
            errors.append(f"rejected candidate decision hash/invariant mismatch: {rejected_id}")
    accepted = [
        row
        for row in decision_rows
        if row.get("status") == "accepted"
    ]
    if len(decision_rows) != len(rejected_rows) + len(accepted) or len(accepted) > 1 or any(
        row.get("status") not in {"accepted", "rejected"} for row in decision_rows
    ):
        errors.append("candidate decision acceptance invariant failure")
    if accepted:
        decision = accepted[0]
        exact_accepted_keys = {
            *EVIDENCE_METADATA,
            "disclosure",
            "candidate_id",
            "status",
            "config",
            "config_sha256",
            "screening_report_sha256",
            "acceptance_sha256",
            "materialization_source_hashes",
        }
        if (
            set(decision) != exact_accepted_keys
            or not _has_evidence_markers(decision)
            or not isinstance(decision.get("disclosure"), str)
            or decision.get("candidate_id") != candidate_id
            or decision.get("config") != candidate.get("config")
            or decision.get("config_sha256") != actual_config_hash
            or decision.get("screening_report_sha256") != screening_hash
            or decision.get("acceptance_sha256") != acceptance_sha256
            or decision.get("materialization_source_hashes") != recorded
        ):
            errors.append("candidate decision hash/invariant mismatch")
    return {
        "candidate_id": candidate_id,
        "candidate_config_sha256": actual_config_hash,
        "screening_report_sha256": screening_hash,
        "source": current_source_hashes,
    }


def _verify_unified(
    root: Path,
    protocol_reports: dict[str, dict[str, Any]],
    *,
    smoke: bool,
    errors: list[str],
) -> tuple[dict[str, Any], dict[str, int]]:
    paths = {
        "evaluation": root / "03-evaluations/unified-evaluation.json",
        "aggregate": root / "04-aggregates/unified-aggregate.json",
        "acceptance": root / "04-aggregates/acceptance.json",
        "index": root / "04-aggregates/unified-artifact-index.json",
    }
    payloads = {
        name: _read_json(path, f"unified {name}", errors)
        for name, path in paths.items()
    }
    current_hashes = {
        name: _safe_hash(path, f"unified {name}", errors)
        for name, path in paths.items()
    }
    if any(not _all_finite(payload) for payload in payloads.values()):
        errors.append("unified finite-metric invariant failure")
    acceptance = payloads["acceptance"]
    if (
        acceptance.get("status") != "passed"
        or acceptance.get("passed") is not True
        or acceptance.get("unmet_checks") != []
    ):
        errors.append("acceptance top-level success invariant failure")
    acceptance_result = evaluate_candidate_acceptance_payload(payloads["acceptance"])
    if not acceptance_result.passed:
        errors.append("acceptance invariant failure: " + ",".join(acceptance_result.reasons))
    aggregate = payloads["aggregate"]
    checks = acceptance.get("checks", {})
    additional_checks = acceptance.get("additional_checks", {})
    if not isinstance(checks, dict) or set(checks) != set(ACCEPTANCE_RANGES):
        errors.append("acceptance exact 16-key invariant failure")
    if (
        not isinstance(additional_checks, dict)
        or set(additional_checks) != set(STRUCTURAL_ACCEPTANCE_KEYS)
    ):
        errors.append("acceptance exact 3-key structural invariant failure")
    check_values = (
        {name: check.get("value") for name, check in checks.items() if isinstance(check, dict)}
        if isinstance(checks, dict)
        else {}
    )
    if aggregate.get("values") != check_values:
        errors.append("aggregate/acceptance invariant mismatch")
    if (
        aggregate.get("evaluation_path") != str(paths["evaluation"].relative_to(root))
        or aggregate.get("evaluation_sha256") != current_hashes["evaluation"]
        or acceptance.get("unified_aggregate_path")
        != str(paths["aggregate"].relative_to(root))
        or acceptance.get("unified_aggregate_sha256") != current_hashes["aggregate"]
    ):
        errors.append("unified aggregate/acceptance hash mismatch")
    index = payloads["index"]
    expected_outputs = {
        str(paths[name].relative_to(root)): current_hashes[name]
        for name in ("evaluation", "aggregate", "acceptance")
    }
    if index.get("outputs") != expected_outputs:
        errors.append("unified artifact output hash mismatch")
    expected_roots = {
        protocol: report["artifact_root_sha256"]
        for protocol, report in protocol_reports.items()
    }
    if index.get("artifact_index_roots") != expected_roots:
        errors.append("unified protocol artifact-root hash mismatch")
    source_hashes = {
        protocol: {
            name: _safe_hash(
                root / "02-predictions" / protocol / name,
                f"{protocol} {name}",
                errors,
            )
            for name in (
                "manifest.json",
                "evaluations.json",
                "selection.json",
                "aggregate.json",
                "artifact-index.json",
            )
        }
        for protocol in PROTOCOLS
    }
    if any(payload.get("source_hashes") != source_hashes for payload in payloads.values() if payload is not index):
        errors.append("unified source hash invariant mismatch")
    if index.get("sources") != source_hashes:
        errors.append("unified index source hash invariant mismatch")

    evaluation = payloads["evaluation"]
    ablations = evaluation.get("ablations", [])
    ratio = evaluation.get("calibration", {}).get(
        "state_lead_to_lead_only_width_ratio"
    )
    if (
        len(ablations) != 9
        or any(
            not isinstance(row, dict) or row.get("passed") is not True
            for row in ablations
        )
        or not isinstance(ratio, (int, float))
        or not math.isfinite(float(ratio))
        or float(ratio) > 1.10
    ):
        errors.append("unified structural acceptance invariant failure")
    bootstrap_rows = sum(
        len(
            _read_json(
                root / "02-predictions" / protocol / "aggregate.json",
                f"{protocol} aggregate",
                errors,
            ).get("rows", [])
        )
        for protocol in PROTOCOLS
    )
    dm_rows = evaluation.get("site_lead_dm_bh", [])
    skill_rows = evaluation.get("forecast_skill", [])
    calibration_rows = evaluation.get("calibration", {}).get("rows", [])
    if bootstrap_rows != 102 or not all(
        _all_finite(row) for row in (*dm_rows, *skill_rows, *calibration_rows)
    ):
        errors.append("bootstrap/DM/BH/Skill finite-count invariant failure")
    if not smoke:
        root_acceptance = evaluate_candidate_acceptance(root)
        errors.extend(
            f"formal acceptance root gate:{reason}"
            for reason in root_acceptance.reasons
        )
        if (
            len(dm_rows) != 150
            or len(skill_rows) != 25
            or len(calibration_rows) != 10
            or not root_acceptance.passed
        ):
            errors.append("formal acceptance identity/count invariant failure")
    if smoke and (not dm_rows or not skill_rows or not calibration_rows):
        errors.append("smoke acceptance identity/count invariant failure")
    return {
        "evaluation_sha256": current_hashes["evaluation"],
        "aggregate_sha256": current_hashes["aggregate"],
        "acceptance_sha256": current_hashes["acceptance"],
        "artifact_index_sha256": current_hashes["index"],
        "numeric_check_keys": sorted(checks) if isinstance(checks, dict) else [],
        "structural_check_keys": (
            sorted(additional_checks) if isinstance(additional_checks, dict) else []
        ),
    }, {
        "bootstrap_rows": bootstrap_rows,
        "dm_bh_rows": len(dm_rows),
        "forecast_skill_rows": len(skill_rows),
        "calibration_rows": len(calibration_rows),
        "acceptance_check_count": len(checks) if isinstance(checks, dict) else 0,
        "structural_check_count": (
            len(additional_checks) if isinstance(additional_checks, dict) else 0
        ),
    }


def _verify_storage_and_placement(root: Path, errors: list[str]) -> tuple[int, int, int]:
    temporary = [path for path in root.rglob("*") if path.is_file() and ".tmp" in path.name]
    if temporary:
        errors.append("temporary-file invariant failure")
    for directory in ("05-tables", "06-figures", "07-paper"):
        for path in root.rglob(directory):
            if path.is_dir() and any(
                child.is_file() and child.name != "README.md" for child in path.rglob("*")
            ):
                errors.append(f"forbidden {directory} output invariant failure")
    artifact_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    if artifact_bytes > ACTUAL_LIMIT_BYTES:
        errors.append("storage size invariant exceeds 45 GiB")
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < MINIMUM_FREE_RESERVE_BYTES:
        errors.append("free-space invariant below 15 GiB")
    return artifact_bytes, free_bytes, len(temporary)


def _all_output_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
        if path.relative_to(root).as_posix() != REPORT_PATH
    }


def verify_realism_revision(
    root: Path, source_root: Path
) -> RevisionVerificationReport:
    """Verify an accepted two-protocol realism-v2 bundle and persist every hash."""

    destination = Path(root).expanduser().resolve()
    source = Path(source_root).expanduser().resolve()
    errors: list[str] = []
    counts: dict[str, int] = {}
    hashes: dict[str, Any] = {}
    artifact_bytes = 0
    free_bytes = 0
    protocol_reports: dict[str, dict[str, Any]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    unified_hashes: dict[str, Any] = {}
    provenance: dict[str, Any] = {}

    def phase_error(label: str, error: Exception) -> None:
        errors.append(f"{label} invariant error: {type(error).__name__}: {error}")

    if (
        destination == source
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        errors.append("source/root isolation invariant failure")
    if not destination.is_dir() or not source.is_dir():
        errors.append("source/root existence invariant failure")

    try:
        preflight = _verify_preflight_hashes(destination, errors)
    except Exception as error:  # noqa: BLE001 - verifier must persist every failure
        phase_error("4A preflight", error)
        preflight = {}
    for protocol in PROTOCOLS:
        protocol_root = destination / "02-predictions" / protocol
        try:
            manifests[protocol] = _read_json(
                protocol_root / "manifest.json", f"{protocol} manifest", errors
            )
            protocol_reports[protocol] = verify_protocol_core(protocol_root)
        except Exception as error:  # noqa: BLE001 - verifier must persist every failure
            phase_error(f"{protocol} protocol", error)
    if set(protocol_reports) == set(PROTOCOLS):
        counts.update(
            prediction_archives=sum(
                report["prediction_archives"] for report in protocol_reports.values()
            ),
            prediction_sidecars=sum(
                report["prediction_sidecars"] for report in protocol_reports.values()
            ),
            sensitivity_aliases=sum(
                report["sensitivity_aliases"] for report in protocol_reports.values()
            ),
            logical_runs=sum(report["logical_runs"] for report in protocol_reports.values()),
        )
        if (
            counts["prediction_archives"] != 420
            or counts["prediction_sidecars"] != 420
            or counts["sensitivity_aliases"] != 90
            or counts["logical_runs"] != 510
        ):
            errors.append("revision physical/logical count invariant failure")
    try:
        _verify_4a_identities(destination, preflight, manifests, errors)
    except Exception as error:  # noqa: BLE001 - verifier must persist every failure
        phase_error("4A identity", error)
    smoke = bool(manifests) and all(
        manifest.get("smoke") is True for manifest in manifests.values()
    )
    formal_mode = set(manifests) == set(PROTOCOLS) and all(
        manifest.get("smoke") is False for manifest in manifests.values()
    )
    counts["site_count"] = (
        preflight.get("site_count", 0)
        if isinstance(preflight.get("site_count", 0), int)
        else 0
    )
    counts["selected_block_count"] = (
        preflight.get("selected_block_count", 0)
        if isinstance(preflight.get("selected_block_count", 0), int)
        else 0
    )
    try:
        unified_hashes, metric_counts = _verify_unified(
            destination, protocol_reports, smoke=smoke, errors=errors
        )
        counts.update(metric_counts)
    except Exception as error:  # noqa: BLE001 - verifier must persist every failure
        phase_error("unified acceptance", error)
    acceptance_hash = unified_hashes.get("acceptance_sha256", "")
    try:
        provenance = _verify_state_sources_and_decision(
            destination, source, acceptance_hash, errors
        )
    except Exception as error:  # noqa: BLE001 - verifier must persist every failure
        phase_error("revision provenance", error)
    try:
        artifact_bytes, free_bytes, temporary_count = _verify_storage_and_placement(
            destination, errors
        )
        counts["temporary_files"] = temporary_count
    except Exception as error:  # noqa: BLE001 - verifier must persist every failure
        phase_error("storage/placement", error)
    try:
        output_hashes = _all_output_hashes(destination)
    except Exception as error:  # noqa: BLE001 - verifier must persist every failure
        phase_error("output hash", error)
        output_hashes = {}
    counts["hashed_output_files"] = len(output_hashes)
    hashes = {
        "candidate": {
            key: value for key, value in provenance.items() if key != "source"
        },
        "source": provenance.get("source", {}),
        "protocols": protocol_reports,
        "unified": unified_hashes,
        "outputs": output_hashes,
    }
    report = RevisionVerificationReport(
        ok=not errors,
        status="passed" if not errors else "failed",
        formal_mode=formal_mode,
        non_smoke=formal_mode,
        errors=tuple(errors),
        counts=counts,
        hashes=hashes,
        artifact_bytes=artifact_bytes,
        free_bytes=free_bytes,
    )
    write_json(destination / REPORT_PATH, report.to_payload())
    if errors:
        raise ValueError("revision verification failed: " + "; ".join(errors))
    return report


def _replay_paths(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for protocol in PROTOCOLS:
        protocol_root = root / "02-predictions" / protocol
        for relative in _CORE_REPLAY_FILES:
            path = protocol_root / relative
            paths[f"02-predictions/{protocol}/{relative}"] = path
        for path in sorted((protocol_root / "predictions/runs").rglob("*")):
            if path.is_file() and (
                path.suffix == ".npz" or path.suffix == ".json"
            ):
                paths[path.relative_to(root).as_posix()] = path
        for directory in (
            protocol_root / "predictions/shared",
            protocol_root / "evaluations",
        ):
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    paths[path.relative_to(root).as_posix()] = path
    for relative in _UNIFIED_REPLAY_FILES:
        paths[relative] = root / relative
    return paths


def _replay_bytes(path: Path) -> bytes:
    if path.name == "artifact-index.json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["verification_runs"] = []
        return canonical_bytes(payload)
    return path.read_bytes()


def _strict_replay_files(root: Path, paths: dict[str, Path]) -> set[tuple[int, int]]:
    """Reject linked/escaping contracted files and return physical identities."""

    root_resolved = root.resolve(strict=True)
    contracted_trees = [
        root / "02-predictions" / protocol / relative
        for protocol in PROTOCOLS
        for relative in ("predictions/runs", "predictions/shared", "evaluations")
    ]
    for tree in contracted_trees:
        if tree.is_symlink():
            raise ValueError(f"deterministic replay symlink artifact: {tree}")
        if tree.exists():
            for child in tree.rglob("*"):
                if child.is_symlink():
                    raise ValueError(
                        f"deterministic replay symlink artifact: "
                        f"{child.relative_to(root)}"
                    )
    identities: set[tuple[int, int]] = set()
    for relative, path in paths.items():
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"deterministic replay artifact outside root: {relative}"
            ) from error
        cursor = path
        while cursor != root:
            if cursor.is_symlink():
                raise ValueError(f"deterministic replay symlink artifact: {relative}")
            try:
                resolved = cursor.resolve(strict=True)
            except OSError as error:
                raise ValueError(
                    f"deterministic replay missing artifact: {relative}"
                ) from error
            if not resolved.is_relative_to(root_resolved):
                raise ValueError(
                    f"deterministic replay artifact outside root: {relative}"
                )
            cursor = cursor.parent
        if not path.is_file():
            raise ValueError(f"deterministic replay missing artifact: {relative}")
        stat = path.stat()
        identities.add((stat.st_dev, stat.st_ino))
    return identities


def verify_deterministic_replay(first_root: Path, second_root: Path) -> dict[str, Any]:
    """Require byte-identical contracted smoke artifacts across two executions.

    Only ``verification_runs`` counters inside protocol artifact indexes are
    normalized away; every other contracted byte is compared directly.
    """

    first = Path(first_root).expanduser().resolve()
    second = Path(second_root).expanduser().resolve()
    if (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    ):
        raise ValueError("deterministic replay roots must be distinct and isolated")
    if not first.is_dir() or not second.is_dir():
        raise ValueError("deterministic replay roots must be distinct existing directories")
    first_paths, second_paths = _replay_paths(first), _replay_paths(second)
    first_identities = _strict_replay_files(first, first_paths)
    second_identities = _strict_replay_files(second, second_paths)
    if first_identities & second_identities:
        raise ValueError("deterministic replay roots share a linked artifact inode")
    if set(first_paths) != set(second_paths):
        raise ValueError("deterministic replay file-set mismatch")
    hashes: dict[str, str] = {}
    for relative in sorted(first_paths):
        left, right = first_paths[relative], second_paths[relative]
        if not left.is_file() or not right.is_file():
            raise ValueError(f"deterministic replay missing artifact: {relative}")
        left_bytes, right_bytes = _replay_bytes(left), _replay_bytes(right)
        if left_bytes != right_bytes:
            raise ValueError(f"deterministic replay byte mismatch: {relative}")
        hashes[relative] = hashlib.sha256(left_bytes).hexdigest()
    return {
        "matched_files": len(hashes),
        "hashes": hashes,
        "excluded_fields": ["artifact-index.json:verification_runs"],
    }


__all__ = [
    "R5TerminalIntegrityResult",
    "R5VerificationReport",
    "RevisionVerificationReport",
    "verify_deterministic_replay",
    "verify_r5_confirmation",
    "verify_r5_deterministic_replay",
    "verify_realism_revision",
]
