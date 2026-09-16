"""Fail-closed, R6b-only validation Design and candidate-freeze contract.

The module owns no source loading.  A caller must supply already reconstructed
adaptive-design cohorts and a metadata-only confirmation-coordinate list.  The
only default filesystem authentication is the published R6b Prepare verifier;
the evaluator is deliberately given neither ``prepared_root`` nor any
confirmation source handle.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .cohort import Cohort
from .constants import RACE_PV
from .realism import RealismCandidate, candidate_catalog, canonical_config_sha256
from .scenario_artifacts import canonical_bytes, sha256_file
from .scenario_candidate import (
    EXPECTED_VALIDATION_SEEDS,
    VALIDATION_PRECHECK_RANGES,
    VALIDATION_PROTOCOLS,
    CandidateGateResult,
    CandidateScreeningReport,
)
from .scenario_r6b import r6b_confirmation_plan_sha256, r6b_oklahoma_candidate_sha256
from .scenario_revision import (
    r5_precheck_ranges_payload,
    r5_precheck_ranges_sha256,
    r6_catalog_sha256,
    r6_design_precheck_reasons,
    r6b_preregistration_sha256,
)

_BUNDLE = Path("00-manifest/r6b-design-freeze")
_AUTHORITY = Path("00-manifest/r6b-confirmation-authority.json")
_DESIGN_SEAL = Path("00-manifest/r6b-design-execution-code-seal.json")
_DESIGN_STAGE_RECEIPT = Path("09-logs/r6b-cli-stages/design.json")
_TASK21_SHA256 = "e70096b871f6a4d78c0b84628dd7a60c640b1148715baa6d254a026ae895134a"
_TASK32_PREPARE_RECEIPT_SHA256 = "5c2be80e49eb4cf6a3e8df6bd9e97960431e97b2645b3cb2ff7de15ec722cda8"
_TASK32_AUTHORITY_PREPARE_RECEIPT_SHA256 = (
    "5c2be80e49eb4cf6a3e8df6bd9e97960431e97b2645b3cb2ff7de15ec722cda8"
)
_R5_ADAPTIVE_DESIGN_MANIFEST_SHA256 = (
    "5aeeecc7f1a6af247b2053f0af5b547a60ce6eeca395ba65f46ec1e1bab72a9d"
)
_TASK32_CONFIRMATION_COORDINATE_SHA256 = (
    "fb08143873615cd45f74d2272caa834b19471f4a285a0a807555a8d70803e832"
)
_TASK32_AUTHORITY_CONFIRMATION_COORDINATE_SHA256 = (
    "fb08143873615cd45f74d2272caa834b19471f4a285a0a807555a8d70803e832"
)
_R6B_DESIGN_ADAPTIVE_AUTHORITY_SHA256 = (
    "59cf38b2780d33ec0ed36d2f6d2bdb9388f611aa46ef9adc271c4c0ea7d5eaa6"
)
_R6B_DESIGN_ADAPTIVE_COORDINATE_SHA256 = (
    "1a1e54d8e958b045dc7617de0e12fd7eef3b2ae8c0661a7b563131f1a42fe4a4"
)
_R6B_DESIGN_ADAPTIVE_IDENTITY_SHA256 = (
    "16607adf2ce9e7ee6df3c9ddc2d446a1299054b4b36579cf8877e9f1b162aa47"
)
_R6B_DESIGN_EXCLUDED_COORDINATE_SHA256 = (
    "3d6098479452bcf929045159be64eea7d393d95796acdb23348f2d2fc204c884"
)
_R6B_DESIGN_EXCLUDED_IDENTITY_SHA256 = (
    "fc3ae4e4f996c01c89a41ac5e0bea04533e5fa2f5cff10dfa0fded51f57fbfab"
)
_SELECTION_ORDER = "accepted:(distance_to_range_midpoints,candidate_id);rejected:candidate_id"
_REPLACE = os.replace
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_R6B_DESIGN_ADAPTIVE_AUTHORITY = Path(__file__).with_name(
    "r6b_design_adaptive_authority.json"
)


@dataclass(frozen=True)
class R6bDesignAdaptiveAuthority:
    """Authenticated metadata for the exact R5-minus-Task32 validation complement."""

    blocks: tuple[dict[str, Any], ...]
    coordinate_sha256: str
    identity_sha256: str
    counts: dict[str, int]
    coordinates: frozenset[tuple[str, str, str]]
    upstream_r5_adaptive_manifest_sha256: str
    task32_prepare_receipt_sha256: str
    task32_confirmation_coordinate_sha256: str
    sha256: str


@dataclass(frozen=True)
class R6bDesignInputs:
    """The narrow R6b Design boundary; confirmation data is metadata-only."""

    work_root: Path
    prepared_root: Path
    candidates: Sequence[RealismCandidate]
    design_cohorts: Mapping[str, Cohort]
    adaptive_design_authority: Path


@dataclass(frozen=True)
class R6bFrozenCandidateEvaluation:
    candidate_id: str
    config_sha256: str
    model_profile_sha256: str
    scenario_seed: int
    passed: bool
    metric_values: tuple[tuple[str, float], ...]
    reasons: tuple[str, ...]
    distance_to_range_midpoints: float

    def payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "config_sha256": self.config_sha256,
            "distance_to_range_midpoints": self.distance_to_range_midpoints,
            "metrics": dict(self.metric_values),
            "model_profile_sha256": self.model_profile_sha256,
            "reasons": list(self.reasons),
            "scenario_seed": self.scenario_seed,
            "status": "accepted" if self.passed else "rejected",
        }


@dataclass(frozen=True)
class R6bDesignReport:
    status: str
    evaluated_count: int
    accepted: tuple[R6bFrozenCandidateEvaluation, ...]
    rejected: tuple[R6bFrozenCandidateEvaluation, ...]
    frozen_candidate_id: str | None
    screening_sha256: str
    freeze_sha256: str
    authority_sha256: str | None
    adaptive_design_authority_sha256: str
    work_root: Path


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return canonical_bytes(dict(payload)) + b"\n"


def _is_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(item in "0123456789abcdef" for item in value)
    )


def _module_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _no_symlink(path: Path) -> None:
    absolute = path.absolute()
    for item in (absolute, *absolute.parents):
        try:
            if stat.S_ISLNK(item.lstat().st_mode):
                raise ValueError(f"R6b Design symlink is forbidden: {item}")
        except FileNotFoundError:
            continue


def _existing_directory(path: Path, *, label: str) -> Path:
    _no_symlink(path)
    try:
        resolved = path.resolve(strict=True)
        mode = path.lstat().st_mode
    except OSError as error:
        raise ValueError(f"R6b Design {label} is missing or unreadable") from error
    if resolved != path.absolute() or not stat.S_ISDIR(mode):
        raise ValueError(f"R6b Design {label} must be a canonical directory")
    return resolved


def _canonical_file(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    _no_symlink(path)
    try:
        mode = path.lstat().st_mode
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"R6b Design {label} is missing or unreadable") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"R6b Design {label} must be a regular file")
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"R6b Design {label} is malformed") from error
    if not isinstance(value, dict) or content != _json_bytes(value):
        raise ValueError(f"R6b Design {label} is not canonical")
    return value, _sha(content)


def _adaptive_authority_identity(rows: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    return [
        [
            row["cohort"],
            row["block_id"],
            row["site_id"],
            row["start"],
            row["end"],
            row["raw_hashes"],
            row["selection_hash"],
        ]
        for row in rows
    ]


def _adaptive_authority_coordinate_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    coordinates = sorted([[row["site_id"], row["start"], row["end"]] for row in rows])
    return _sha(canonical_bytes(coordinates))


def _adaptive_authority_identity_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return _sha(canonical_bytes(_adaptive_authority_identity(rows)))


def load_r6b_design_adaptive_authority(
    path: Path | None = None,
) -> R6bDesignAdaptiveAuthority:
    """Load the one frozen 26-block adaptive authority and reject every drift."""
    source = _R6B_DESIGN_ADAPTIVE_AUTHORITY if path is None else Path(path)
    payload, authority_sha256 = _canonical_file(source, label="adaptive authority")
    if authority_sha256 != _R6B_DESIGN_ADAPTIVE_AUTHORITY_SHA256:
        raise ValueError("R6b Design adaptive authority byte binding is invalid")
    expected_keys = {
        "adaptive_design_data",
        "blocks",
        "coordinate_sha256",
        "counts",
        "excluded_coordinate_sha256",
        "excluded_count",
        "excluded_identity_sha256",
        "identity_sha256",
        "schema",
        "task32_confirmation_coordinate_sha256",
        "task32_prepare_receipt_sha256",
        "upstream_r5_adaptive_manifest_sha256",
    }
    blocks = payload.get("blocks")
    if (
        set(payload) != expected_keys
        or payload.get("schema") != "r6b_design_adaptive_authority_v1"
        or payload.get("adaptive_design_data") is not True
        or payload.get("upstream_r5_adaptive_manifest_sha256")
        != _R5_ADAPTIVE_DESIGN_MANIFEST_SHA256
        or payload.get("task32_prepare_receipt_sha256")
        != _TASK32_AUTHORITY_PREPARE_RECEIPT_SHA256
        or payload.get("task32_confirmation_coordinate_sha256")
        != _TASK32_AUTHORITY_CONFIRMATION_COORDINATE_SHA256
        or payload.get("excluded_count") != 6
        or payload.get("excluded_coordinate_sha256")
        != _R6B_DESIGN_EXCLUDED_COORDINATE_SHA256
        or payload.get("excluded_identity_sha256")
        != _R6B_DESIGN_EXCLUDED_IDENTITY_SHA256
        or payload.get("counts")
        != {"dkasc-yulara": 12, "nrel-transfer": 14, "total": 26}
        or not isinstance(blocks, list)
        or len(blocks) != 26
    ):
        raise ValueError("R6b Design adaptive authority schema or provenance is invalid")
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
    counts = {"dkasc-yulara": 0, "nrel-transfer": 0}
    coordinates: list[tuple[str, str, str]] = []
    block_ids: list[str] = []
    sort_keys: list[tuple[str, str, str, str]] = []
    for row in blocks:
        if not isinstance(row, dict) or set(row) != block_keys:
            raise ValueError("R6b Design adaptive authority block schema is invalid")
        cohort = row.get("cohort")
        expected_source = {
            "dkasc-yulara": ("dkasc", "source_train", "NT"),
            "nrel-transfer": ("nrel", "validation_calibration", "OK"),
        }.get(cohort)
        raw_hashes = row.get("raw_hashes")
        try:
            start = datetime.fromisoformat(row["start"])
            end = datetime.fromisoformat(row["end"])
        except (TypeError, ValueError) as error:
            raise ValueError("R6b Design adaptive authority block boundary is invalid") from error
        if (
            expected_source is None
            or (row.get("dataset"), row.get("spatial_role"), row.get("state"))
            != expected_source
            or row.get("split") != "validation"
            or row.get("n_steps") != 1344
            or end - start != timedelta(days=13, hours=23, minutes=45)
            or not isinstance(row.get("site_id"), str)
            or not row["site_id"]
            or not isinstance(row.get("quarter"), str)
            or not _is_digest(row.get("selection_hash"))
            or not isinstance(row.get("block_id"), str)
            or row["block_id"] != row["selection_hash"][:20]
            or not isinstance(raw_hashes, list)
            or not raw_hashes
            or any(not _is_digest(value) for value in raw_hashes)
        ):
            raise ValueError("R6b Design adaptive authority block identity is invalid")
        counts[cohort] += 1
        coordinates.append((row["site_id"], row["start"], row["end"]))
        block_ids.append(row["block_id"])
        sort_keys.append((cohort, row["site_id"], row["start"], row["end"]))
    if (
        counts != {"dkasc-yulara": 12, "nrel-transfer": 14}
        or len(set(coordinates)) != 26
        or len(set(block_ids)) != 26
        or sort_keys != sorted(sort_keys)
        or _adaptive_authority_coordinate_sha256(blocks)
        != _R6B_DESIGN_ADAPTIVE_COORDINATE_SHA256
        or payload.get("coordinate_sha256")
        != _R6B_DESIGN_ADAPTIVE_COORDINATE_SHA256
        or _adaptive_authority_identity_sha256(blocks)
        != _R6B_DESIGN_ADAPTIVE_IDENTITY_SHA256
        or payload.get("identity_sha256") != _R6B_DESIGN_ADAPTIVE_IDENTITY_SHA256
    ):
        raise ValueError("R6b Design adaptive authority complement identity is invalid")
    return R6bDesignAdaptiveAuthority(
        blocks=tuple(dict(row) for row in blocks),
        coordinate_sha256=payload["coordinate_sha256"],
        identity_sha256=payload["identity_sha256"],
        counts=dict(payload["counts"]),
        coordinates=frozenset(coordinates),
        upstream_r5_adaptive_manifest_sha256=payload[
            "upstream_r5_adaptive_manifest_sha256"
        ],
        task32_prepare_receipt_sha256=payload["task32_prepare_receipt_sha256"],
        task32_confirmation_coordinate_sha256=payload[
            "task32_confirmation_coordinate_sha256"
        ],
        sha256=authority_sha256,
    )


def _default_prepare_verifier(root: Path) -> Mapping[str, Any]:
    # Local import prevents the domain contract from importing a CLI route.
    from .pipeline import _r6b_verify_published_prepare

    return _r6b_verify_published_prepare(root)


def _prepare_receipt(inputs: R6bDesignInputs) -> dict[str, Any]:
    root = _existing_directory(Path(inputs.prepared_root), label="Prepare root")
    # Task32 is immutable historical provenance in both fresh and recovery
    # paths.  The old Prepare CLI verifier re-hashes its *current* pipeline,
    # which is deliberately allowed to evolve when Design is added.
    receipt = _terminal_prepare_receipt(root)
    if _sha(_json_bytes(receipt)) != _TASK32_PREPARE_RECEIPT_SHA256:
        raise ValueError("R6b Design immutable Task32 Prepare receipt binding is invalid")
    expected = {
        "counts": {"dkasc": 12, "oklahoma": 15, "total": 27},
        "lifecycle": {
            "confirmation_executed": False,
            "design_executed": False,
            "prediction_generated": False,
        },
        "revision": "r6b",
        "schema": "r6b_cli_stage_v1",
        "source_revision": "r5",
        "stage": "prepare",
        "status": "completed",
        "task21_report_sha256": _TASK21_SHA256,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("R6b Design Task32 Prepare receipt binding is invalid")
    return receipt


def _terminal_prepare_receipt(root: Path) -> dict[str, Any]:
    """Authenticate Task32's sealed files after a Design terminal is present."""
    receipt, _receipt_sha256 = _canonical_file(
        root / "09-logs/r6b-cli-stages/prepare.json", label="Prepare receipt"
    )
    relative_hashes = receipt.get("relative_sha256")
    if (
        not isinstance(relative_hashes, dict)
        or len(relative_hashes) != 33
        or any(
            type(relative) is not str
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not _is_digest(digest)
            for relative, digest in relative_hashes.items()
        )
    ):
        raise ValueError("R6b Design Task32 Prepare receipt hash map is invalid")
    for relative, digest in relative_hashes.items():
        path = root / relative
        _no_symlink(path)
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise ValueError("R6b Design Task32 Prepare receipt member is missing") from error
        if not stat.S_ISREG(mode) or sha256_file(path) != digest:
            raise ValueError("R6b Design Task32 Prepare receipt member drift")
    # Reuse the source-manifest parser, which has no stage or CLI side effect.
    from .pipeline import _r6b_prepare_source_rows

    rows = _r6b_prepare_source_rows(root)
    if (
        len(rows) != 27
        or sum(row["dataset"] == "dkasc" for row in rows) != 12
        or sum(row["dataset"] == "nrel" for row in rows) != 15
        or any(row["output_path"] not in relative_hashes for row in rows)
    ):
        raise ValueError("R6b Design Task32 source manifest binding is invalid")
    return receipt


def _verify_task32_execution_code_seal(root: Path) -> str:
    """Validate persisted Task32 provenance without re-hashing later Design code."""
    seal, seal_sha256 = _canonical_file(
        root / "00-manifest/r6b-execution-code-seal.json", label="Task32 execution code seal"
    )
    expected_paths = {
        "cohort": "src/racepv/scenario/cohort.py",
        "loader": "src/racepv/scenario/loader.py",
        "pipeline": "src/racepv/scenario/pipeline.py",
        "realism": "src/racepv/scenario/realism.py",
        "scenario_r6b": "src/racepv/scenario/scenario_r6b.py",
        "scenario_revision": "src/racepv/scenario/scenario_revision.py",
    }
    modules = seal.get("modules")
    if seal.get("schema") != "r6b_execution_code_seal_v1" or not isinstance(modules, list):
        raise ValueError("R6b Design Task32 execution code seal is invalid")
    by_name = {row.get("module"): row for row in modules if isinstance(row, dict)}
    if (
        len(modules) != 6
        or set(by_name) != set(expected_paths)
        or any(set(row) != {"module", "path", "sha256"} for row in modules)
    ):
        raise ValueError("R6b Design Task32 execution code seal is invalid")
    for name, relative in expected_paths.items():
        row = by_name[name]
        if row.get("path") != relative or not _is_digest(row.get("sha256")):
            raise ValueError("R6b Design Task32 execution code seal is invalid")
    return seal_sha256


def _design_execution_code_seal_payload(
    receipt: Mapping[str, Any],
    task32_execution_code_seal_sha256: str,
    adaptive_design_authority_sha256: str,
) -> dict[str, Any]:
    """Seal the current Design-era dependencies separately from Task32 history."""
    from racepv import metrics

    from . import (
        cohort,
        constants,
        loader,
        pipeline,
        realism,
        scenario_artifacts,
        scenario_candidate,
        scenario_manifest,
        scenario_r6b,
        scenario_revision,
        synthesis,
    )

    modules = {
        "cohort": Path(cohort.__file__),
        "constants": Path(constants.__file__),
        "loader": Path(loader.__file__),
        "metrics": Path(metrics.__file__),
        "pipeline": Path(pipeline.__file__),
        "realism": Path(realism.__file__),
        "scenario_artifacts": Path(scenario_artifacts.__file__),
        "scenario_candidate": Path(scenario_candidate.__file__),
        "scenario_manifest": Path(scenario_manifest.__file__),
        "scenario_r6b": Path(scenario_r6b.__file__),
        "scenario_r6b_design": Path(__file__),
        "scenario_revision": Path(scenario_revision.__file__),
        "synthesis": Path(synthesis.__file__),
    }
    rows = []
    for name, path in sorted(modules.items()):
        _no_symlink(path)
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise ValueError("R6b Design execution dependency is missing") from error
        if not stat.S_ISREG(mode) or not path.is_relative_to(_PROJECT_ROOT):
            raise ValueError("R6b Design execution dependency is invalid")
        rows.append(
            {
                "module": name,
                "path": path.relative_to(_PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema": "r6b_design_execution_code_seal_v1",
        "adaptive_design_authority_sha256": adaptive_design_authority_sha256,
        "modules": rows,
        "semantic_pins": _pins(),
        "task32_prepare_receipt_sha256": _sha(_json_bytes(receipt)),
        "task32_execution_code_seal_sha256": task32_execution_code_seal_sha256,
    }


def _task32_confirmation_coordinates(root: Path) -> tuple[tuple[str, str, str], ...]:
    """Derive the only confirmation coordinates from Task32's sealed blocks CSV."""
    path = root / "01-cohorts/r6b-confirmation-blocks.csv"
    _no_symlink(path)
    try:
        mode = path.lstat().st_mode
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError("R6b Design Task32 confirmation blocks are malformed") from error
    columns = [
        "block_id",
        "dataset",
        "site_id",
        "spatial_role",
        "state",
        "split",
        "start",
        "end",
        "quarter",
        "n_steps",
        "selection_hash",
    ]
    if not stat.S_ISREG(mode) or len(rows) != 27 or any(list(row) != columns for row in rows):
        raise ValueError("R6b Design Task32 confirmation blocks are invalid")
    coordinates = tuple((row["site_id"], row["start"], row["end"]) for row in rows)
    if (
        len(set(coordinates)) != 27
        or len({row["block_id"] for row in rows}) != 27
        or any(row["split"] != "validation" or row["n_steps"] != "1344" for row in rows)
    ):
        raise ValueError("R6b Design Task32 confirmation coordinate binding is invalid")
    return coordinates


def _exact_terminal_root_paths(root: Path, receipt: Mapping[str, Any], *, frozen: bool) -> None:
    """Reject every root entry other than sealed Prepare plus legal Design files."""
    hashes = receipt.get("relative_sha256")
    # Temporary test receipts intentionally omit the Task32 map; production
    # receipts cannot pass _terminal_prepare_receipt without it.
    if not isinstance(hashes, dict):
        return
    expected_files = {
        *hashes,
        "09-logs/r6b-cli-stages/prepare.json",
        _DESIGN_SEAL.as_posix(),
        *((_BUNDLE / name).as_posix() for name in ("freeze.json", "screening.json", "state.json")),
    }
    if frozen:
        expected_files.add(_AUTHORITY.as_posix())
    if (root / _DESIGN_STAGE_RECEIPT).exists():
        expected_files.add(_DESIGN_STAGE_RECEIPT.as_posix())
    actual_files: set[str] = set()
    actual_dirs: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError("R6b Design terminal root contains a symlink")
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            actual_files.add(relative)
        elif stat.S_ISDIR(mode):
            actual_dirs.add(relative)
        else:
            raise ValueError("R6b Design terminal root contains a nonregular entry")
    expected_dirs = {
        parent.as_posix()
        for relative in expected_files
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    if actual_files != expected_files or actual_dirs != expected_dirs:
        raise ValueError("R6b Design terminal root path set is invalid")


def _verify_design_stage_receipt(
    root: Path,
    receipt: Mapping[str, Any],
    *,
    screening_sha256: str,
    freeze_sha256: str,
    design_seal_sha256: str,
    adaptive_design_authority_sha256: str,
    accepted_count: int,
    rejected_count: int,
    status: str,
) -> None:
    """Permit the Task2 receipt only after exact domain-level binding checks."""
    path = root / _DESIGN_STAGE_RECEIPT
    if not path.exists() and not path.is_symlink():
        return
    value, _value_sha256 = _canonical_file(path, label="Design stage receipt")
    expected = {
        "schema": "r6b_cli_stage_v1",
        "revision": "r6b",
        "source_revision": "r5",
        "stage": "design",
        "status": "completed",
        "prepare_receipt_sha256": _sha(_json_bytes(receipt)),
        "design_screening_sha256": screening_sha256,
        "candidate_freeze_sha256": freeze_sha256,
        "design_execution_code_seal_sha256": design_seal_sha256,
        "adaptive_design_authority_sha256": adaptive_design_authority_sha256,
        "counts": {
            "candidate_count": 6,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
        },
        "terminal_state": status,
        "lifecycle": {
            "confirmation_executed": False,
            "design_executed": True,
            "prediction_generated": False,
        },
    }
    if value != expected:
        raise ValueError("R6b Design stage receipt binding is invalid")


def _pins() -> dict[str, str]:
    pins = {
        "catalog_sha256": r6_catalog_sha256(),
        "candidate_universe_sha256": r6b_oklahoma_candidate_sha256(),
        "confirmation_plan_sha256": r6b_confirmation_plan_sha256(),
        "preregistration_sha256": r6b_preregistration_sha256(),
        "range_definition_sha256": r5_precheck_ranges_sha256(),
    }
    if not all(_is_digest(value) for value in pins.values()):
        raise ValueError("R6b Design provenance pins are invalid")
    return pins


def _exact_candidates(values: Sequence[RealismCandidate]) -> tuple[RealismCandidate, ...]:
    rows = tuple(values)
    expected = tuple(candidate_catalog("r6"))
    if len(rows) != 6:
        raise ValueError("R6b Design requires the exact six-candidate R6 catalog")

    def signature(row: RealismCandidate) -> tuple[str, str, int, str]:
        if (
            type(row.candidate_id) is not str
            or not _is_digest(row.config_sha256)
            or type(row.scenario_seed) is not int
            or canonical_config_sha256(row.config) != row.config_sha256
            or row.model_profile != RACE_PV
        ):
            raise ValueError("R6b Design candidate identity/config/profile is invalid")
        return (
            row.candidate_id,
            row.config_sha256,
            row.scenario_seed,
            _sha(canonical_bytes(asdict(row.model_profile))),
        )

    if [signature(row) for row in rows] != [signature(row) for row in expected]:
        raise ValueError("R6b Design candidate catalog identity/config/profile/seed drift")
    return rows


def _coordinate(value: object) -> tuple[str, str, str]:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or not all(type(item) is str and item for item in value)
    ):
        raise ValueError("R6b Design cohort coordinates are invalid")
    return value


def _cohort_hashes(cohorts: Mapping[str, Cohort]) -> dict[str, str]:
    from .scenario_manifest import cohort_sha256

    return {name: cohort_sha256(cohorts[name]) for name in VALIDATION_PROTOCOLS}


def _block_payload(block: Any) -> dict[str, Any]:
    return {
        "block_id": block.block_id,
        "dataset": block.dataset,
        "end": block.end.isoformat(),
        "n_steps": block.n_steps,
        "quarter": block.quarter,
        "raw_hashes": list(block.raw_hashes),
        "selection_hash": block.selection_hash,
        "site_id": block.site_id,
        "spatial_role": block.spatial_role,
        "split": block.split,
        "start": block.start.isoformat(),
        "state": block.state,
    }


def _authenticated_adaptive_authority(
    path: Path,
    cohorts: Mapping[str, Cohort],
    confirmation_coordinates: set[tuple[str, str, str]],
) -> R6bDesignAdaptiveAuthority:
    """Bind reconstructed cohort blocks to every row of the frozen complement."""
    authority = load_r6b_design_adaptive_authority(path)
    expected_by_cohort = {
        name: [
            {key: value for key, value in row.items() if key != "cohort"}
            for row in authority.blocks
            if row["cohort"] == name
        ]
        for name in VALIDATION_PROTOCOLS
    }
    if (
        authority.coordinates & confirmation_coordinates
        or authority.task32_confirmation_coordinate_sha256
        != _TASK32_AUTHORITY_CONFIRMATION_COORDINATE_SHA256
        or set(expected_by_cohort) != set(VALIDATION_PROTOCOLS)
    ):
        raise ValueError("R6b Design adaptive authority coordinate overlap is forbidden")
    for name in VALIDATION_PROTOCOLS:
        if [_block_payload(block) for block in cohorts[name].blocks] != expected_by_cohort[name]:
            raise ValueError("R6b Design adaptive authority cohort binding is invalid")
    return authority


def _validate_design_cohorts(
    cohorts: Mapping[str, Cohort],
    confirmation: Sequence[tuple[str, str, str]],
    adaptive_design_authority: Path,
) -> tuple[Mapping[str, Cohort], R6bDesignAdaptiveAuthority]:
    if not isinstance(cohorts, Mapping) or set(cohorts) != set(VALIDATION_PROTOCOLS):
        raise ValueError("R6b Design requires exactly the adaptive validation cohorts")
    confirmation_coordinates = {_coordinate(value) for value in confirmation}
    if (
        len(confirmation_coordinates) != 27
        or _sha(canonical_bytes([list(row) for row in sorted(confirmation_coordinates)]))
        != _TASK32_CONFIRMATION_COORDINATE_SHA256
    ):
        raise ValueError("R6b Design confirmation coordinate binding is invalid")
    validated: dict[str, Cohort] = {}
    for name in VALIDATION_PROTOCOLS:
        cohort = cohorts[name]
        if not isinstance(cohort, Cohort) or cohort.split != "validation" or not cohort.blocks:
            raise ValueError("R6b Design cohort role/split is invalid")
        rows = tuple(
            _coordinate((block.site_id, block.start.isoformat(), block.end.isoformat()))
            for block in cohort.blocks
        )
        if len(rows) != len(set(rows)) or set(rows) & confirmation_coordinates:
            raise ValueError("R6b Design adaptive cohort coordinate overlap is forbidden")
        if any(block.split != "validation" or block.n_steps != 1344 for block in cohort.blocks):
            raise ValueError("R6b Design cohort block role/split is invalid")
        validated[name] = cohort
    return validated, _authenticated_adaptive_authority(
        adaptive_design_authority, validated, confirmation_coordinates
    )


def _distance(metrics: Mapping[str, float]) -> float:
    offsets = []
    for name, (lower, upper) in VALIDATION_PRECHECK_RANGES.items():
        offsets.append(((metrics[name] - (lower + upper) / 2) / (upper - lower)) ** 2)
    return math.sqrt(sum(offsets) / len(offsets))


def _normalize_results(
    report: CandidateScreeningReport, candidates: tuple[RealismCandidate, ...]
) -> tuple[tuple[R6bFrozenCandidateEvaluation, ...], tuple[R6bFrozenCandidateEvaluation, ...]]:
    if not isinstance(report, CandidateScreeningReport):
        raise ValueError("R6b Design evaluator returned an invalid screening report")
    by_id = {row.candidate_id: row for row in candidates}
    rows = tuple(report.candidates)
    if len(rows) != 6 or {row.candidate_id for row in rows} != set(by_id):
        raise ValueError("R6b Design evaluator did not evaluate all six candidates exactly once")
    normalized: list[R6bFrozenCandidateEvaluation] = []
    required = set(VALIDATION_PRECHECK_RANGES)
    for row in rows:
        if not isinstance(row, CandidateGateResult):
            raise ValueError("R6b Design evaluator result is invalid")
        candidate = by_id[row.candidate_id]
        if (
            row.config_sha256 != candidate.config_sha256
            or row.config != asdict(candidate.config)
            or row.scenario_seed != candidate.scenario_seed
            or tuple(row.seeds) != EXPECTED_VALIDATION_SEEDS
            or set(row.metrics) != required
        ):
            raise ValueError("R6b Design evaluator identity/config/metric binding is invalid")
        metrics = {name: float(row.metrics[name]) for name in sorted(required)}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("R6b Design evaluator returned a non-finite metric")
        reasons = r6_design_precheck_reasons(metrics)
        if bool(row.passed) != (not reasons) or tuple(row.reasons) != reasons:
            raise ValueError("R6b Design evaluator result disagrees with frozen ranges")
        normalized.append(
            R6bFrozenCandidateEvaluation(
                candidate_id=candidate.candidate_id,
                config_sha256=candidate.config_sha256,
                model_profile_sha256=_sha(canonical_bytes(asdict(candidate.model_profile))),
                scenario_seed=candidate.scenario_seed,
                passed=not reasons,
                metric_values=tuple(sorted(metrics.items())),
                reasons=reasons,
                distance_to_range_midpoints=_distance(metrics),
            )
        )
    accepted = tuple(
        sorted(
            (row for row in normalized if row.passed),
            key=lambda row: (row.distance_to_range_midpoints, row.candidate_id),
        )
    )
    rejected = tuple(
        sorted((row for row in normalized if not row.passed), key=lambda row: row.candidate_id)
    )
    return accepted, rejected


def _state(
    screening_sha256: str,
    freeze_sha256: str,
    status: str,
    prepare_sha256: str,
    adaptive_design_authority_sha256: str,
) -> dict[str, Any]:
    history = [
        {
            "sequence": 1,
            "from_state": None,
            "state": "confirmation_cohort_prepared",
            "details": {"prepare_receipt_sha256": prepare_sha256},
        },
        {
            "sequence": 2,
            "from_state": "confirmation_cohort_prepared",
            "state": "design_screened",
            "details": {"design_screening_sha256": screening_sha256},
        },
        {
            "sequence": 3,
            "from_state": "design_screened",
            "state": status,
            "details": {"candidate_freeze_sha256": freeze_sha256},
        },
    ]
    return {
        "schema": "r6b_design_state_v1",
        "state": status,
        "history": history,
        "adaptive_design_authority_sha256": adaptive_design_authority_sha256,
    }


def _write_temp_bundle(parent: Path, payloads: Mapping[str, Mapping[str, Any]]) -> Path:
    temporary = Path(tempfile.mkdtemp(prefix=".r6b-design-freeze-", suffix=".tmp", dir=parent))
    try:
        for name, payload in sorted(payloads.items()):
            path = temporary / name
            with path.open("xb") as handle:
                handle.write(_json_bytes(payload))
                handle.flush()
                os.fsync(handle.fileno())
        return temporary
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _replace(source: Path, target: Path) -> None:
    _REPLACE(source, target)


def _atomic_publish(
    root: Path,
    payloads: Mapping[str, Mapping[str, Any]],
    authority: Mapping[str, Any] | None,
    design_seal: Mapping[str, Any],
) -> None:
    manifest = root / "00-manifest"
    _no_symlink(manifest)
    manifest.mkdir(parents=True, exist_ok=True)
    _no_symlink(manifest)
    final = root / _BUNDLE
    authority_path = root / _AUTHORITY
    design_seal_path = root / _DESIGN_SEAL
    partials = tuple(manifest.glob(".r6b-design-freeze-*.tmp"))
    if (
        partials
        or final.exists()
        or final.is_symlink()
        or authority_path.exists()
        or authority_path.is_symlink()
        or design_seal_path.exists()
        or design_seal_path.is_symlink()
    ):
        raise ValueError("R6b Design terminal bundle already exists or is partial")
    temporary = _write_temp_bundle(manifest, payloads)
    authority_tmp: Path | None = None
    design_seal_tmp = Path(
        tempfile.mkstemp(prefix=".r6b-design-execution-code-seal-", suffix=".tmp", dir=manifest)[1]
    )
    try:
        design_seal_tmp.write_bytes(_json_bytes(design_seal))
        if authority is not None:
            authority_tmp = Path(
                tempfile.mkstemp(
                    prefix=".r6b-confirmation-authority-", suffix=".tmp", dir=manifest
                )[1]
            )
            authority_tmp.write_bytes(_json_bytes(authority))
        _replace(temporary, final)
        _replace(design_seal_tmp, design_seal_path)
        if authority_tmp is not None:
            _replace(authority_tmp, authority_path)
    except BaseException:
        if authority_path.exists() and authority is not None:
            authority_path.unlink()
        if final.exists():
            shutil.rmtree(final)
        if design_seal_path.exists():
            design_seal_path.unlink()
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        if authority_tmp is not None and authority_tmp.exists():
            authority_tmp.unlink()
        if design_seal_tmp.exists():
            design_seal_tmp.unlink()


def _report_from_payloads(
    root: Path,
    screening: dict[str, Any],
    freeze: dict[str, Any],
    screening_sha256: str,
    freeze_sha256: str,
    authority_sha256: str | None,
    adaptive_design_authority_sha256: str,
) -> R6bDesignReport:
    rows = screening.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("R6b Design screening candidates are invalid")
    normalized = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "candidate_id",
            "config_sha256",
            "distance_to_range_midpoints",
            "metrics",
            "model_profile_sha256",
            "reasons",
            "scenario_seed",
            "status",
        }:
            raise ValueError("R6b Design screening candidate schema is invalid")
        normalized.append(
            R6bFrozenCandidateEvaluation(
                candidate_id=row["candidate_id"],
                config_sha256=row["config_sha256"],
                model_profile_sha256=row["model_profile_sha256"],
                scenario_seed=row["scenario_seed"],
                passed=row["status"] == "accepted",
                metric_values=tuple(sorted(row["metrics"].items())),
                reasons=tuple(row["reasons"]),
                distance_to_range_midpoints=row["distance_to_range_midpoints"],
            )
        )
    accepted = tuple(row for row in normalized if row.passed)
    rejected = tuple(row for row in normalized if not row.passed)
    return R6bDesignReport(
        status=freeze["status"],
        evaluated_count=screening["evaluated_count"],
        accepted=accepted,
        rejected=rejected,
        frozen_candidate_id=freeze["candidate_id"],
        screening_sha256=screening_sha256,
        freeze_sha256=freeze_sha256,
        authority_sha256=authority_sha256,
        adaptive_design_authority_sha256=adaptive_design_authority_sha256,
        work_root=root,
    )


def run_r6b_design_screen(
    inputs: R6bDesignInputs,
    *,
    evaluator: Callable[[Sequence[RealismCandidate], Mapping[str, Any]], CandidateScreeningReport],
) -> R6bDesignReport:
    """Evaluate the exact R6 catalog once and atomically freeze one accepted row."""
    if not isinstance(inputs, R6bDesignInputs):
        raise TypeError("inputs must be R6bDesignInputs")
    root = _existing_directory(Path(inputs.work_root), label="work root")
    if root != _existing_directory(Path(inputs.prepared_root), label="Prepare root"):
        raise ValueError("R6b Design work and Prepare roots must be identical")
    if any(
        path.exists() or path.is_symlink()
        for path in (root / _BUNDLE, root / _AUTHORITY, root / _DESIGN_SEAL, root / _DESIGN_STAGE_RECEIPT)
    ):
        return verify_r6b_design_bundle(root, inputs)
    receipt = _prepare_receipt(inputs)
    task32_execution_code_seal_sha256 = _verify_task32_execution_code_seal(root)
    pins = _pins()
    candidates = _exact_candidates(inputs.candidates)
    cohorts, adaptive_authority = _validate_design_cohorts(
        inputs.design_cohorts,
        _task32_confirmation_coordinates(root),
        Path(inputs.adaptive_design_authority),
    )
    cohort_hashes = _cohort_hashes(cohorts)
    design_seal = _design_execution_code_seal_payload(
        receipt, task32_execution_code_seal_sha256, adaptive_authority.sha256
    )
    result = evaluator(candidates, cohorts)
    # Detect lazy evaluator side effects or mutation of the authority inputs.
    if (
        _pins() != pins
        or _prepare_receipt(inputs) != receipt
        or _cohort_hashes(cohorts) != cohort_hashes
        or _authenticated_adaptive_authority(
            Path(inputs.adaptive_design_authority),
            cohorts,
            set(_task32_confirmation_coordinates(root)),
        ).sha256
        != adaptive_authority.sha256
        or _design_execution_code_seal_payload(
            receipt, task32_execution_code_seal_sha256, adaptive_authority.sha256
        )
        != design_seal
    ):
        raise ValueError("R6b Design input binding changed during evaluator execution")
    accepted, rejected = _normalize_results(result, candidates)
    frozen = accepted[0] if accepted else None
    status = "candidate_frozen" if frozen else "failed_design"
    screening = {
        "schema": "r6b_design_screening_v1",
        "status": "passed" if frozen else "failed",
        "candidate_count": 6,
        "evaluated_count": 6,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "selection_order": _SELECTION_ORDER,
        "selected_candidate_id": frozen.candidate_id if frozen else None,
        "candidates": [row.payload() for row in (*accepted, *rejected)],
        "precheck_ranges": r5_precheck_ranges_payload(),
        "pins": pins,
        "prepare_receipt_sha256": _sha(_json_bytes(receipt)),
        "adaptive_design_cohort_sha256": cohort_hashes,
        "adaptive_design_manifest_sha256": (
            adaptive_authority.upstream_r5_adaptive_manifest_sha256
        ),
        "adaptive_design_authority_sha256": adaptive_authority.sha256,
        "adaptive_design_data": True,
        "held_out_used": False,
    }
    screening_sha256 = _sha(_json_bytes(screening))
    freeze = {
        "schema": "r6b_candidate_freeze_v1",
        "status": status,
        "candidate_id": frozen.candidate_id if frozen else None,
        "config_sha256": frozen.config_sha256 if frozen else None,
        "model_profile_sha256": frozen.model_profile_sha256 if frozen else None,
        "scenario_seed": frozen.scenario_seed if frozen else None,
        "design_screening_sha256": screening_sha256,
        "selection_order": _SELECTION_ORDER,
        "pins": pins,
        "prepare_receipt_sha256": _sha(_json_bytes(receipt)),
        "adaptive_design_authority_sha256": adaptive_authority.sha256,
    }
    freeze_sha256 = _sha(_json_bytes(freeze))
    state = _state(
        screening_sha256,
        freeze_sha256,
        status,
        _sha(_json_bytes(receipt)),
        adaptive_authority.sha256,
    )
    authority = (
        None
        if frozen is None
        else {
            "schema": "r6b_confirmation_authority_v1",
            "status": "candidate_frozen",
            "candidate_id": frozen.candidate_id,
            "config_sha256": frozen.config_sha256,
            "model_profile_sha256": frozen.model_profile_sha256,
            "scenario_seed": frozen.scenario_seed,
            "candidate_freeze_sha256": freeze_sha256,
            "design_screening_sha256": screening_sha256,
            "pins": pins,
            "adaptive_design_authority_sha256": adaptive_authority.sha256,
        }
    )
    _atomic_publish(
        root,
        {"freeze.json": freeze, "screening.json": screening, "state.json": state},
        authority,
        design_seal,
    )
    return verify_r6b_design_bundle(root, inputs)


def verify_r6b_design_bundle(root: Path, inputs: R6bDesignInputs) -> R6bDesignReport:
    """Re-read the complete terminal R6b Design contract without evaluation."""
    if not isinstance(inputs, R6bDesignInputs):
        raise TypeError("inputs must be R6bDesignInputs")
    secure_root = _existing_directory(Path(root), label="work root")
    if secure_root != _existing_directory(Path(inputs.prepared_root), label="Prepare root"):
        raise ValueError("R6b Design verifier root mismatch")
    receipt = _prepare_receipt(inputs)
    task32_execution_code_seal_sha256 = _verify_task32_execution_code_seal(secure_root)
    pins = _pins()
    candidates = _exact_candidates(inputs.candidates)
    cohorts, adaptive_authority = _validate_design_cohorts(
        inputs.design_cohorts,
        _task32_confirmation_coordinates(secure_root),
        Path(inputs.adaptive_design_authority),
    )
    cohort_hashes = _cohort_hashes(cohorts)
    design_seal, _design_seal_sha256 = _canonical_file(
        secure_root / _DESIGN_SEAL, label="Design execution code seal"
    )
    if design_seal != _design_execution_code_seal_payload(
        receipt, task32_execution_code_seal_sha256, adaptive_authority.sha256
    ):
        raise ValueError("R6b Design execution code seal drift")
    bundle = secure_root / _BUNDLE
    _existing_directory(bundle, label="freeze bundle")
    expected = {"freeze.json", "screening.json", "state.json"}
    actual = {path.name for path in bundle.iterdir()}
    if actual != expected or any(
        path.is_symlink() or not path.is_file() for path in bundle.iterdir()
    ):
        raise ValueError("R6b Design bundle path set is invalid")
    screening, screening_sha256 = _canonical_file(bundle / "screening.json", label="screening")
    freeze, freeze_sha256 = _canonical_file(bundle / "freeze.json", label="freeze")
    state, _state_sha256 = _canonical_file(bundle / "state.json", label="state")
    if (
        screening.get("schema") != "r6b_design_screening_v1"
        or freeze.get("schema") != "r6b_candidate_freeze_v1"
        or state.get("schema") != "r6b_design_state_v1"
        or screening.get("pins") != pins
        or freeze.get("pins") != pins
        or screening.get("prepare_receipt_sha256") != _sha(_json_bytes(receipt))
        or freeze.get("prepare_receipt_sha256") != _sha(_json_bytes(receipt))
        or screening.get("adaptive_design_authority_sha256")
        != adaptive_authority.sha256
        or freeze.get("adaptive_design_authority_sha256")
        != adaptive_authority.sha256
        or freeze.get("design_screening_sha256") != screening_sha256
    ):
        raise ValueError("R6b Design external/freeze binding is invalid")
    status = freeze.get("status")
    if (
        status not in {"candidate_frozen", "failed_design"}
        or screening.get("evaluated_count") != 6
        or screening.get("candidate_count") != 6
    ):
        raise ValueError("R6b Design terminal state is invalid")
    expected_state = _state(
        screening_sha256,
        freeze_sha256,
        status,
        _sha(_json_bytes(receipt)),
        adaptive_authority.sha256,
    )
    if state != expected_state:
        raise ValueError("R6b Design state history binding is invalid")
    # Reconstruct normalized rows from persistent values, catching coordinated edits.
    rows = screening.get("candidates")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("R6b Design screening rows are invalid")
    by_id = {row.candidate_id: row for row in candidates}
    rebuilt = []
    for value in rows:
        if not isinstance(value, dict) or value.get("candidate_id") not in by_id:
            raise ValueError("R6b Design screening candidate binding is invalid")
        candidate = by_id[value["candidate_id"]]
        metrics = value.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != set(VALIDATION_PRECHECK_RANGES):
            raise ValueError("R6b Design screening metric binding is invalid")
        numeric = {name: float(metrics[name]) for name in sorted(metrics)}
        reasons = r6_design_precheck_reasons(numeric)
        expected_row = R6bFrozenCandidateEvaluation(
            candidate.candidate_id,
            candidate.config_sha256,
            _sha(canonical_bytes(asdict(candidate.model_profile))),
            candidate.scenario_seed,
            not reasons,
            tuple(sorted(numeric.items())),
            reasons,
            _distance(numeric),
        ).payload()
        if value != expected_row:
            raise ValueError("R6b Design screening candidate result binding is invalid")
        rebuilt.append(expected_row)
    accepted = sorted(
        (row for row in rebuilt if row["status"] == "accepted"),
        key=lambda row: (row["distance_to_range_midpoints"], row["candidate_id"]),
    )
    rejected = sorted(
        (row for row in rebuilt if row["status"] == "rejected"), key=lambda row: row["candidate_id"]
    )
    frozen = accepted[0] if accepted else None
    if (
        screening.get("candidates") != [*accepted, *rejected]
        or screening.get("accepted_count") != len(accepted)
        or screening.get("rejected_count") != len(rejected)
    ):
        raise ValueError("R6b Design screening ordering/count binding is invalid")
    expected_screening = {
        "schema": "r6b_design_screening_v1",
        "status": "passed" if frozen else "failed",
        "candidate_count": 6,
        "evaluated_count": 6,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "selection_order": _SELECTION_ORDER,
        "selected_candidate_id": frozen["candidate_id"] if frozen else None,
        "candidates": [*accepted, *rejected],
        "precheck_ranges": r5_precheck_ranges_payload(),
        "pins": pins,
        "prepare_receipt_sha256": _sha(_json_bytes(receipt)),
        "adaptive_design_cohort_sha256": cohort_hashes,
        "adaptive_design_manifest_sha256": (
            adaptive_authority.upstream_r5_adaptive_manifest_sha256
        ),
        "adaptive_design_authority_sha256": adaptive_authority.sha256,
        "adaptive_design_data": True,
        "held_out_used": False,
    }
    if screening != expected_screening:
        raise ValueError("R6b Design screening top-level binding is invalid")
    expected_freeze = {
        "schema": "r6b_candidate_freeze_v1",
        "status": "candidate_frozen" if frozen else "failed_design",
        "candidate_id": frozen["candidate_id"] if frozen else None,
        "config_sha256": frozen["config_sha256"] if frozen else None,
        "model_profile_sha256": frozen["model_profile_sha256"] if frozen else None,
        "scenario_seed": frozen["scenario_seed"] if frozen else None,
        "design_screening_sha256": screening_sha256,
        "selection_order": _SELECTION_ORDER,
        "pins": pins,
        "prepare_receipt_sha256": _sha(_json_bytes(receipt)),
        "adaptive_design_authority_sha256": adaptive_authority.sha256,
    }
    if freeze != expected_freeze:
        raise ValueError("R6b Design freeze binding is invalid")
    authority_path = secure_root / _AUTHORITY
    authority_sha256 = None
    if frozen:
        authority, authority_sha256 = _canonical_file(
            authority_path, label="confirmation authority"
        )
        expected_authority = {
            "schema": "r6b_confirmation_authority_v1",
            "status": "candidate_frozen",
            "candidate_id": frozen["candidate_id"],
            "config_sha256": frozen["config_sha256"],
            "model_profile_sha256": frozen["model_profile_sha256"],
            "scenario_seed": frozen["scenario_seed"],
            "candidate_freeze_sha256": freeze_sha256,
            "design_screening_sha256": screening_sha256,
            "pins": pins,
            "adaptive_design_authority_sha256": adaptive_authority.sha256,
        }
        if authority != expected_authority:
            raise ValueError("R6b Design confirmation authority binding is invalid")
    elif authority_path.exists() or authority_path.is_symlink():
        raise ValueError("R6b Design failed terminal must not have confirmation authority")
    _verify_design_stage_receipt(
        secure_root,
        receipt,
        screening_sha256=screening_sha256,
        freeze_sha256=freeze_sha256,
        design_seal_sha256=_design_seal_sha256,
        adaptive_design_authority_sha256=adaptive_authority.sha256,
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        status=status,
    )
    _exact_terminal_root_paths(secure_root, receipt, frozen=frozen is not None)
    return _report_from_payloads(
        secure_root,
        screening,
        freeze,
        screening_sha256,
        freeze_sha256,
        authority_sha256,
        adaptive_authority.sha256,
    )


__all__ = [
    "R6bDesignAdaptiveAuthority",
    "R6bDesignInputs",
    "R6bDesignReport",
    "R6bFrozenCandidateEvaluation",
    "load_r6b_design_adaptive_authority",
    "run_r6b_design_screen",
    "verify_r6b_design_bundle",
]
