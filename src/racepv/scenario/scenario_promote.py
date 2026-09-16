"""Authorized, journaled promotion of a formally verified realism-v2 bundle."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .compact_storage import MINIMUM_FREE_RESERVE_BYTES
from .constants import EVIDENCE_METADATA
from .scenario_artifacts import DISCLOSURE, canonical_bytes, sha256_file, write_json
from .scenario_manifest import ACCEPTANCE_RANGES
from .scenario_revision import SOURCE_ROOT_MANDATORY_ARTIFACTS, _source_root_hashes
from .scenario_revision_verify import (
    REPORT_PATH,
    STRUCTURAL_ACCEPTANCE_KEYS,
    verify_realism_revision,
)

PROMOTION_CONFIRMATION = "DELETE-RETIRED-Q2-IDEAL-AND-PROMOTE-REALISM-V2"
_V1_NAME = "q2-ideal"
_V2_NAME = "q2-realism-v2"
_RETIRING_NAME = ".q2-ideal-retiring"
_JOURNAL_NAME = ".q2-realism-v2-promotion.json"
_AUDIT_PATH = "09-logs/replaced-v1-audit.json"
_COMPLETE_PATH = "09-logs/promotion-complete.json"
_FINAL_REPORT_PREFIX = "09-logs/promotion-final-verification-"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_AUTHORITY = object()
_FORMAL_COUNTS = {
    "site_count": 57,
    "selected_block_count": 252,
    "prediction_archives": 420,
    "prediction_sidecars": 420,
    "sensitivity_aliases": 90,
    "logical_runs": 510,
    "acceptance_check_count": 16,
    "structural_check_count": 3,
    "temporary_files": 0,
}

Verifier = Callable[[Path, Path], Any]
FailureInjector = Callable[[str], None]


@dataclass(frozen=True)
class PromotionAuthorization:
    """Opaque capability created only after exact destructive confirmation."""

    _authority: object


@dataclass(frozen=True)
class _PromotionFilesystemPolicy:
    project_root: Path
    verifier: Verifier
    test_only: bool

    @property
    def scenarios(self) -> Path:
        return self.project_root / "scenarios"


_PRODUCTION_POLICY = _PromotionFilesystemPolicy(
    _PROJECT_ROOT, verify_realism_revision, False
)


def authorize_promotion(confirmation: str) -> PromotionAuthorization:
    if confirmation != PROMOTION_CONFIRMATION:
        raise ValueError("exact destructive promotion confirmation is required")
    return PromotionAuthorization(_AUTHORITY)


def _test_filesystem_policy(
    project_root: Path, *, verifier: Verifier = verify_realism_revision
) -> _PromotionFilesystemPolicy:
    """Explicit test-only seam for temporary repository fixtures."""

    root = Path(project_root).expanduser().resolve()
    if root == _PROJECT_ROOT:
        raise ValueError("test filesystem policy must not target the production project")
    return _PromotionFilesystemPolicy(root, verifier, True)


def _policy(
    supplied: _PromotionFilesystemPolicy | None,
) -> _PromotionFilesystemPolicy:
    if supplied is None:
        return _PRODUCTION_POLICY
    if not isinstance(supplied, _PromotionFilesystemPolicy) or not supplied.test_only:
        raise ValueError("invalid promotion filesystem policy")
    return supplied


def _raw_absolute(path: Path) -> Path:
    return Path(path).expanduser().absolute()


def _path_has_symlink(path: Path, stop: Path) -> bool:
    current = _raw_absolute(path)
    boundary = _raw_absolute(stop)
    while True:
        if current.is_symlink():
            return True
        if current == boundary:
            return False
        if current.parent == current:
            return True
        current = current.parent


def _expected_paths(policy: _PromotionFilesystemPolicy) -> dict[str, Path]:
    scenarios = policy.scenarios.resolve()
    return {
        "scenarios": scenarios,
        "v1": scenarios / _V1_NAME,
        "v2": scenarios / _V2_NAME,
        "retiring": scenarios / _RETIRING_NAME,
        "journal": scenarios / _JOURNAL_NAME,
    }


def _require_requested_paths(
    v1: Path, v2: Path, policy: _PromotionFilesystemPolicy
) -> dict[str, Path]:
    paths = _expected_paths(policy)
    raw_v1, raw_v2 = _raw_absolute(v1), _raw_absolute(v2)
    if raw_v1 != paths["v1"] or raw_v2 != paths["v2"]:
        label = "test policy" if policy.test_only else "exact production"
        raise ValueError(
            f"promotion requires {label} paths {paths['v1']} and {paths['v2']}"
        )
    if _path_has_symlink(raw_v1, paths["scenarios"]) or _path_has_symlink(
        raw_v2, paths["scenarios"]
    ):
        raise ValueError("promotion roots and scenarios ancestry must not be symlinks")
    return paths


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is missing or unreadable: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _audit_file(v1: Path, relative: str) -> tuple[Path, dict[str, Any]]:
    path = v1 / relative
    if (
        not path.is_file()
        or _path_has_symlink(path, v1)
        or not path.resolve().is_relative_to(v1)
    ):
        raise ValueError(f"v1 audit input is missing, linked, or escaping: {relative}")
    return path, _read_json(path, f"v1 audit input {relative}")


def build_replaced_v1_audit(
    v1: Path,
    v2: Path,
    *,
    filesystem_policy: _PromotionFilesystemPolicy | None = None,
) -> dict[str, Any]:
    """Build the complete small-artifact audit for the retiring v1 bundle."""

    policy = _policy(filesystem_policy)
    paths = _require_requested_paths(v1, v2, policy)
    if not paths["v1"].is_dir() or not paths["v2"].is_dir():
        raise ValueError("both exact promotion roots must exist")
    source_hashes = _source_root_hashes(paths["v1"])
    artifacts: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_ROOT_MANDATORY_ARTIFACTS:
        path, payload = _audit_file(paths["v1"], relative)
        artifacts[relative] = {"sha256": sha256_file(path), "payload": payload}
    acceptance = artifacts["04-aggregates/acceptance.json"]["payload"]
    unmet = acceptance.get("unmet_checks", [])
    if not isinstance(unmet, list) or not all(isinstance(item, str) for item in unmet):
        raise ValueError("v1 acceptance unmet-check audit is malformed")
    return {
        **dict(EVIDENCE_METADATA),
        "disclosure": DISCLOSURE,
        "schema": "replaced_v1_audit_v2",
        "v1_root_basename": _V1_NAME,
        "v2_root_basename": _V2_NAME,
        "v1_acceptance_sha256": artifacts["04-aggregates/acceptance.json"]["sha256"],
        "v1_acceptance_status": acceptance.get("status"),
        "v1_failure_reasons": unmet,
        "v1_source_hashes": source_hashes,
        "v1_source_hashes_sha256": hashlib.sha256(canonical_bytes(source_hashes)).hexdigest(),
        "v1_manifest": artifacts["00-manifest/compact-shared-v2-validation.json"]["payload"],
        "v1_phase_4d_report": artifacts["09-logs/phase-4d-report.json"]["payload"],
        "artifacts": artifacts,
    }


def _acceptance_passed(root: Path) -> bool:
    acceptance = _read_json(root / "04-aggregates/acceptance.json", "v2 acceptance")
    checks = acceptance.get("checks")
    additional = acceptance.get("additional_checks")
    return (
        acceptance.get("status") == "passed"
        and acceptance.get("passed") is True
        and acceptance.get("unmet_checks") == []
        and isinstance(checks, dict)
        and set(checks) == set(ACCEPTANCE_RANGES)
        and all(isinstance(row, dict) and row.get("passed") is True for row in checks.values())
        and isinstance(additional, dict)
        and set(additional) == set(STRUCTURAL_ACCEPTANCE_KEYS)
        and all(isinstance(row, dict) and row.get("passed") is True for row in additional.values())
    )


def _manifests_are_formal(root: Path) -> bool:
    return all(
        _read_json(
            root / "02-predictions" / protocol / "manifest.json", f"{protocol} manifest"
        ).get("smoke")
        is False
        for protocol in ("dkasc-yulara", "nrel-transfer")
    )


def _formal_report_errors(report: Any, persisted: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if (
        getattr(report, "ok", None) is not True
        or getattr(report, "formal_mode", None) is not True
        or getattr(report, "non_smoke", None) is not True
        or persisted.get("ok") is not True
        or persisted.get("status") != "passed"
        or persisted.get("formal_mode") is not True
        or persisted.get("non_smoke") is not True
        or persisted.get("errors") != []
    ):
        errors.append("verification is not a passed formal non-smoke report")
    counts = persisted.get("counts", {})
    if not isinstance(counts, dict) or any(
        counts.get(key) != value for key, value in _FORMAL_COUNTS.items()
    ):
        errors.append("formal 57/252/420/90/510 and 16+3 counts are not exact")
    unified = persisted.get("hashes", {}).get("unified", {})
    if set(unified.get("numeric_check_keys", ())) != set(ACCEPTANCE_RANGES):
        errors.append("formal numeric acceptance key identity is not exact")
    if set(unified.get("structural_check_keys", ())) != set(STRUCTURAL_ACCEPTANCE_KEYS):
        errors.append("formal structural acceptance key identity is not exact")
    return errors


def promotion_eligibility(
    v1: Path,
    v2: Path,
    *,
    filesystem_policy: _PromotionFilesystemPolicy | None = None,
) -> dict[str, Any]:
    """Freshly prove formal promotion gates without authorizing mutation."""

    reasons: list[str] = []
    result: dict[str, Any] = {"eligible": False, "reasons": reasons}
    try:
        policy = _policy(filesystem_policy)
        paths = _require_requested_paths(v1, v2, policy)
        if paths["journal"].exists() or paths["retiring"].exists():
            raise ValueError("promotion transaction already exists and must be resumed")
        if not paths["v1"].is_dir() or not paths["v2"].is_dir():
            raise ValueError("both exact promotion roots must exist")
        audit = build_replaced_v1_audit(paths["v1"], paths["v2"], filesystem_policy=policy)
        result["audit"] = audit
        if not _acceptance_passed(paths["v2"]):
            reasons.append("v2 acceptance is not exactly 16+3 passed")
        if not _manifests_are_formal(paths["v2"]):
            reasons.append("promotion requires a formal non-smoke bundle")
        state = _read_json(paths["v2"] / "09-logs/revision-state.json", "v2 revision state")
        if state.get("state") not in {"candidate_accepted", "verified", "promotion_eligible"}:
            reasons.append("v2 revision state is not successful")
        report = policy.verifier(paths["v2"], paths["v1"])
        report_path = paths["v2"] / REPORT_PATH
        persisted = _read_json(report_path, "v2 verification report")
        reasons.extend(_formal_report_errors(report, persisted))
        acceptance_sha256 = sha256_file(paths["v2"] / "04-aggregates/acceptance.json")
        if (
            persisted.get("hashes", {}).get("outputs", {}).get("04-aggregates/acceptance.json")
            != acceptance_sha256
        ):
            reasons.append("v2 verification acceptance hash is stale")
        if getattr(report, "free_bytes", 0) < MINIMUM_FREE_RESERVE_BYTES:
            reasons.append("v2 storage gate has insufficient free space")
        result.update(
            v1=str(paths["v1"]),
            v2=str(paths["v2"]),
            scenarios=str(paths["scenarios"]),
            verification_sha256=sha256_file(report_path),
            acceptance_sha256=acceptance_sha256,
            source_hashes=_source_root_hashes(paths["v1"]),
        )
    except (OSError, ValueError) as error:
        reasons.append(str(error))
    result["eligible"] = not reasons
    return result


def _tree_sha256(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"promotion tree is missing or linked: {root}")
    linked = [path for path in root.rglob("*") if path.is_symlink()]
    if linked:
        raise ValueError(f"promotion tree contains symlinks: {linked[0]}")
    entries = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    return hashlib.sha256(canonical_bytes(entries)).hexdigest()


def _journal_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "journal_sha256"}
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def _read_journal(path: Path) -> dict[str, Any]:
    payload = _read_json(path, "promotion journal")
    if (
        payload.get("schema") != "realism_v2_promotion_journal_v2"
        or payload.get("journal_sha256") != _journal_hash(payload)
    ):
        raise ValueError("promotion journal hash/schema mismatch")
    return payload


def _write_journal(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    persisted = dict(payload)
    persisted["journal_sha256"] = _journal_hash(persisted)
    write_json(path, persisted)
    return _read_journal(path)


def _phase(path: Path, journal: dict[str, Any], phase: str, **updates: Any) -> dict[str, Any]:
    return _write_journal(path, {**journal, **updates, "phase": phase})


def _inject(injector: FailureInjector | None, point: str) -> None:
    if injector is not None:
        injector(point)


def _atomic_rename(source: Path, destination: Path) -> None:
    source.replace(destination)


def _complete_payload(journal: dict[str, Any], canonical: Path) -> dict[str, Any]:
    return {
        **dict(EVIDENCE_METADATA),
        "disclosure": DISCLOSURE,
        "schema": "realism_v2_promotion_complete_v2",
        "status": "complete",
        "transaction_id": journal["transaction_id"],
        "promoted_path": str(canonical),
        "removed_v2_path": journal["v2_path"],
        "replaced_v1_audit_sha256": journal["audit_sha256"],
        "final_verification_path": journal["final_verification_path"],
        "final_verification_sha256": journal["final_verification_sha256"],
        "retired_v1_tree_sha256": journal["v1_tree_sha256"],
        "promoted_v2_tree_sha256": journal["audited_v2_tree_sha256"],
    }


def _canonical_verification(
    paths: dict[str, Path],
    policy: _PromotionFilesystemPolicy,
    journal: dict[str, Any],
) -> tuple[str, str]:
    report = policy.verifier(paths["v1"], paths["retiring"])
    report_path = paths["v1"] / REPORT_PATH
    persisted = _read_json(report_path, "final verification report")
    errors = _formal_report_errors(report, persisted)
    if errors:
        raise ValueError("final verification failed: " + "; ".join(errors))
    if sha256_file(paths["v1"] / _AUDIT_PATH) != journal["audit_sha256"]:
        raise ValueError("final verification audit hash mismatch")
    return sha256_file(report_path), _tree_sha256(paths["v1"])


def _final_report_relative(transaction_id: str) -> str:
    return f"{_FINAL_REPORT_PREFIX}{transaction_id}.json"


def _retained_final_report(
    paths: dict[str, Path], journal: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    relative = journal.get("final_verification_path")
    expected = _final_report_relative(journal["transaction_id"])
    if relative != expected:
        raise ValueError("retained final verification path identity mismatch")
    path = paths["v1"] / relative
    if (
        not path.is_file()
        or path.is_symlink()
        or not path.resolve().is_relative_to(paths["v1"])
        or sha256_file(path) != journal.get("final_verification_sha256")
    ):
        raise ValueError("retained final verification hash/path mismatch")
    persisted = _read_json(path, "retained final verification")
    errors = _formal_report_errors(SimpleNamespace(**persisted), persisted)
    if errors:
        raise ValueError("retained final verification is invalid: " + "; ".join(errors))
    return path, persisted


def _read_valid_completion(canonical: Path) -> dict[str, Any]:
    completion = _read_json(canonical / _COMPLETE_PATH, "promotion completion")
    transaction_id = completion.get("transaction_id")
    expected_relative = (
        _final_report_relative(transaction_id) if isinstance(transaction_id, str) else None
    )
    if (
        completion.get("schema") != "realism_v2_promotion_complete_v2"
        or completion.get("status") != "complete"
        or completion.get("promoted_path") != str(canonical)
        or completion.get("final_verification_path") != expected_relative
    ):
        raise ValueError("promotion completion identity is invalid")
    retained = canonical / expected_relative
    if (
        not retained.is_file()
        or retained.is_symlink()
        or not retained.resolve().is_relative_to(canonical)
        or sha256_file(retained) != completion.get("final_verification_sha256")
    ):
        raise ValueError("promotion completion final verification is invalid")
    persisted = _read_json(retained, "retained final verification")
    errors = _formal_report_errors(SimpleNamespace(**persisted), persisted)
    if errors:
        raise ValueError("promotion completion final verification failed validation")
    audit = canonical / _AUDIT_PATH
    if (
        not audit.is_file()
        or audit.is_symlink()
        or sha256_file(audit) != completion.get("replaced_v1_audit_sha256")
    ):
        raise ValueError("promotion completion audit is invalid")
    return completion


def _transaction_completion(
    paths: dict[str, Path], journal: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    path = paths["v1"] / _COMPLETE_PATH
    return path, _complete_payload(journal, paths["v1"])


def _completion_is_exact(paths: dict[str, Path], journal: dict[str, Any]) -> bool:
    path, expected = _transaction_completion(paths, journal)
    return path.is_file() and _read_json(path, "promotion completion") == expected


def _remove_transaction_completion(paths: dict[str, Path], journal: dict[str, Any]) -> None:
    path = paths["v1"] / _COMPLETE_PATH
    if not path.is_file():
        return
    payload = _read_json(path, "promotion completion")
    if payload.get("transaction_id") == journal.get("transaction_id"):
        path.unlink()


def _remove_transaction_final_report(
    paths: dict[str, Path], journal: dict[str, Any]
) -> None:
    relative = journal.get("final_verification_path")
    if relative != _final_report_relative(journal.get("transaction_id", "")):
        return
    for root in (paths["v1"], paths["v2"]):
        path = root / relative
        if path.is_file() and not path.is_symlink():
            if sha256_file(path) == journal.get("final_verification_sha256"):
                path.unlink()


def _rollback(paths: dict[str, Path], journal: dict[str, Any]) -> None:
    if journal.get("phase") in {
        "old_delete_in_progress",
        "retiring_contents_deleted",
        "old_deleted",
        "record_cleanup_pending",
    }:
        return
    _remove_transaction_completion(paths, journal)
    _remove_transaction_final_report(paths, journal)
    if paths["v1"].is_dir() and paths["retiring"].is_dir() and not paths["v2"].exists():
        journal = _phase(paths["journal"], journal, "rollback_v2_pending")
        _atomic_rename(paths["v1"], paths["v2"])
    if paths["retiring"].is_dir() and not paths["v1"].exists():
        journal = _phase(paths["journal"], journal, "rollback_v1_pending")
        _atomic_rename(paths["retiring"], paths["v1"])
    if paths["v1"].is_dir() and paths["v2"].is_dir() and not paths["retiring"].exists():
        paths["journal"].unlink(missing_ok=True)


def _directory_identity(path: Path) -> dict[str, int]:
    try:
        status = path.lstat()
    except FileNotFoundError as error:
        raise ValueError("retiring deletion root identity mismatch: missing") from error
    if not stat.S_ISDIR(status.st_mode) or stat.S_ISLNK(status.st_mode):
        raise ValueError("retiring deletion root identity mismatch: not a directory")
    return {"st_dev": status.st_dev, "st_ino": status.st_ino}


def _delete_fd_contents(
    directory_fd: int,
    injector: FailureInjector | None,
    injected: list[bool],
) -> None:
    with os.scandir(directory_fd) as entries:
        names = sorted(entry.name for entry in entries)
    for name in names:
        status = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(status.st_mode):
            flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
            child_fd = os.open(name, flags, dir_fd=directory_fd)
            try:
                _delete_fd_contents(child_fd, injector, injected)
            finally:
                os.close(child_fd)
            os.rmdir(name, dir_fd=directory_fd)
        else:
            os.unlink(name, dir_fd=directory_fd)
        if not injected[0]:
            injected[0] = True
            _inject(injector, "during_old_delete")


def _require_retiring_transaction_path(
    paths: dict[str, Path], journal: dict[str, Any]
) -> Path:
    retiring = paths["retiring"]
    if (
        journal.get("retiring_path") != str(retiring)
        or retiring.name != _RETIRING_NAME
        or retiring.parent != paths["scenarios"]
    ):
        raise ValueError("retiring deletion transaction identity mismatch")
    return retiring


def _delete_retiring_contents(
    paths: dict[str, Path],
    journal: dict[str, Any],
    injector: FailureInjector | None,
) -> None:
    retiring = _require_retiring_transaction_path(paths, journal)
    expected_identity = journal.get("retiring_identity")
    if _directory_identity(retiring) != expected_identity:
        raise ValueError("retiring deletion root identity mismatch")
    parent_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(paths["scenarios"], parent_flags)
    root_fd: int | None = None
    try:
        root_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        root_fd = os.open(retiring.name, root_flags, dir_fd=parent_fd)
        opened = os.fstat(root_fd)
        if {"st_dev": opened.st_dev, "st_ino": opened.st_ino} != expected_identity:
            raise ValueError("retiring deletion root identity mismatch")
        _delete_fd_contents(root_fd, injector, [False])
        current = os.stat(retiring.name, dir_fd=parent_fd, follow_symlinks=False)
        if {"st_dev": current.st_dev, "st_ino": current.st_ino} != expected_identity:
            raise ValueError("retiring deletion root identity mismatch")
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)


def _remove_empty_retiring_root(
    paths: dict[str, Path], journal: dict[str, Any]
) -> bool:
    retiring = _require_retiring_transaction_path(paths, journal)
    expected_identity = journal.get("retiring_identity")
    try:
        actual_identity = _directory_identity(retiring)
    except ValueError as error:
        if not retiring.exists() and not retiring.is_symlink():
            return False
        raise error
    if actual_identity != expected_identity:
        raise ValueError("retiring deletion root identity mismatch")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(paths["scenarios"], flags)
    root_fd: int | None = None
    try:
        root_fd = os.open(retiring.name, flags, dir_fd=parent_fd)
        opened = os.fstat(root_fd)
        if {"st_dev": opened.st_dev, "st_ino": opened.st_ino} != expected_identity:
            raise ValueError("retiring deletion root identity mismatch")
        with os.scandir(root_fd) as entries:
            if next(entries, None) is not None:
                raise ValueError("retiring deletion root has unexpected contents")
        current = os.stat(retiring.name, dir_fd=parent_fd, follow_symlinks=False)
        if {"st_dev": current.st_dev, "st_ino": current.st_ino} != expected_identity:
            raise ValueError("retiring deletion root identity mismatch")
        os.rmdir(retiring.name, dir_fd=parent_fd)
        return True
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)


def _run_transaction(
    paths: dict[str, Path],
    policy: _PromotionFilesystemPolicy,
    journal: dict[str, Any],
    injector: FailureInjector | None,
) -> dict[str, Any]:
    while True:
        phase = journal["phase"]
        if phase == "record_written":
            journal = _phase(paths["journal"], journal, "audit_write_pending")
        elif phase == "audit_write_pending":
            audit_path = paths["v2"] / _AUDIT_PATH
            expected = journal["audit_payload"]
            if audit_path.is_file() and _read_json(audit_path, "replaced-v1 audit") != expected:
                raise ValueError("existing replaced-v1 audit differs from journal")
            if not audit_path.is_file():
                write_json(audit_path, expected)
            if sha256_file(audit_path) != journal["audit_sha256"]:
                raise ValueError("replaced-v1 audit hash mismatch")
            _inject(injector, "after_audit_write")
            journal = _phase(
                paths["journal"], journal, "audit_written", audited_v2_tree_sha256=_tree_sha256(paths["v2"])
            )
        elif phase == "audit_written":
            journal = _phase(paths["journal"], journal, "v1_retire_pending")
        elif phase == "v1_retire_pending":
            if paths["v1"].is_dir() and not paths["retiring"].exists():
                if _tree_sha256(paths["v1"]) != journal["v1_tree_sha256"]:
                    raise ValueError("retiring v1 tree hash changed")
                _atomic_rename(paths["v1"], paths["retiring"])
            elif not paths["v1"].exists() and paths["retiring"].is_dir():
                if _tree_sha256(paths["retiring"]) != journal["v1_tree_sha256"]:
                    raise ValueError("retired v1 tree hash mismatch")
            else:
                raise ValueError("v1 retirement phase has ambiguous roots")
            _inject(injector, "after_v1_retire_rename")
            journal = _phase(paths["journal"], journal, "v1_retired")
        elif phase == "v1_retired":
            journal = _phase(paths["journal"], journal, "v2_promote_pending")
        elif phase == "v2_promote_pending":
            if paths["v2"].is_dir() and not paths["v1"].exists():
                if _tree_sha256(paths["v2"]) != journal["audited_v2_tree_sha256"]:
                    raise ValueError("promoting v2 tree hash changed")
                _atomic_rename(paths["v2"], paths["v1"])
            elif not paths["v2"].exists() and paths["v1"].is_dir():
                if _tree_sha256(paths["v1"]) != journal["audited_v2_tree_sha256"]:
                    raise ValueError("canonical promoted v2 tree hash mismatch")
            else:
                raise ValueError("v2 canonical promotion phase has ambiguous roots")
            _inject(injector, "after_v2_canonical_rename")
            journal = _phase(paths["journal"], journal, "v2_promoted")
        elif phase == "v2_promoted":
            journal = _phase(paths["journal"], journal, "final_verify_pending")
        elif phase == "final_verify_pending":
            report_sha256, tree_sha256 = _canonical_verification(paths, policy, journal)
            _inject(injector, "after_final_verification")
            journal = _phase(
                paths["journal"],
                journal,
                "final_verified",
                final_verification_sha256=report_sha256,
                final_verified_tree_sha256=tree_sha256,
            )
            _inject(injector, "after_final_verified_commit")
        elif phase == "final_verified":
            if _tree_sha256(paths["v1"]) != journal["final_verified_tree_sha256"]:
                raise ValueError("canonical tree changed after final verification")
            report_sha256, tree_sha256 = _canonical_verification(paths, policy, journal)
            journal = _phase(
                paths["journal"],
                journal,
                "final_report_retain_pending",
                final_verification_path=_final_report_relative(
                    journal["transaction_id"]
                ),
                final_verification_sha256=report_sha256,
                final_verified_tree_sha256=tree_sha256,
            )
        elif phase == "final_report_retain_pending":
            report_path = paths["v1"] / REPORT_PATH
            if (
                _tree_sha256(paths["v1"]) != journal["final_verified_tree_sha256"]
                or sha256_file(report_path) != journal["final_verification_sha256"]
            ):
                raise ValueError("final verification changed before retention")
            retained_path = paths["v1"] / journal["final_verification_path"]
            report_payload = _read_json(report_path, "final verification report")
            if retained_path.is_file():
                if _read_json(retained_path, "retained final verification") != report_payload:
                    raise ValueError("retained final verification differs from final report")
            else:
                write_json(retained_path, report_payload)
            _retained_final_report(paths, journal)
            journal = _phase(
                paths["journal"],
                journal,
                "completion_write_pending",
                final_verified_tree_sha256=_tree_sha256(paths["v1"]),
            )
        elif phase == "completion_write_pending":
            complete_path = paths["v1"] / _COMPLETE_PATH
            expected = _complete_payload(journal, paths["v1"])
            if complete_path.is_file() and _read_json(complete_path, "promotion completion") != expected:
                raise ValueError("promotion completion differs from journal")
            if not complete_path.is_file():
                write_json(complete_path, expected)
            _inject(injector, "after_completion_write")
            journal = _phase(
                paths["journal"],
                journal,
                "complete_written",
                completion_sha256=sha256_file(complete_path),
                completed_tree_sha256=_tree_sha256(paths["v1"]),
            )
            _inject(injector, "after_complete_written_commit")
        elif phase == "complete_written":
            if not _completion_is_exact(paths, journal):
                raise ValueError("transaction-bound promotion completion is invalid")
            if _tree_sha256(paths["v1"]) != journal["completed_tree_sha256"]:
                raise ValueError("canonical tree changed before old deletion")
            _retained_final_report(paths, journal)
            if sha256_file(paths["v1"] / _AUDIT_PATH) != journal["audit_sha256"]:
                raise ValueError("retained audit changed before old deletion")
            journal = _phase(
                paths["journal"],
                journal,
                "old_delete_pending",
                pre_delete_verification_sha256=journal["final_verification_sha256"],
                pre_delete_tree_sha256=journal["completed_tree_sha256"],
            )
        elif phase == "old_delete_pending":
            complete_path = paths["v1"] / _COMPLETE_PATH
            if sha256_file(complete_path) != journal["completion_sha256"]:
                raise ValueError("durable completion hash mismatch before old deletion")
            if _tree_sha256(paths["v1"]) != journal["pre_delete_tree_sha256"]:
                raise ValueError("canonical tree changed after pre-delete verification")
            if sha256_file(paths["v1"] / _AUDIT_PATH) != journal["audit_sha256"]:
                raise ValueError("retained audit changed before old deletion")
            _retained_final_report(paths, journal)
            retiring_identity = _directory_identity(paths["retiring"])
            if _tree_sha256(paths["retiring"]) != journal["v1_tree_sha256"]:
                raise ValueError("retired v1 changed before deletion authorization")
            if _directory_identity(paths["retiring"]) != retiring_identity:
                raise ValueError("retiring deletion root identity changed")
            journal = _phase(
                paths["journal"],
                journal,
                "old_delete_in_progress",
                retiring_identity=retiring_identity,
            )
        elif phase == "old_delete_in_progress":
            _delete_retiring_contents(paths, journal, injector)
            if _directory_identity(paths["retiring"]) != journal["retiring_identity"]:
                raise ValueError("retiring deletion root identity mismatch")
            if any(paths["retiring"].iterdir()):
                raise ValueError("retiring deletion root has unexpected contents")
            journal = _phase(
                paths["journal"], journal, "retiring_contents_deleted"
            )
            _inject(injector, "after_retiring_contents_deleted_commit")
        elif phase == "retiring_contents_deleted":
            removed = _remove_empty_retiring_root(paths, journal)
            if removed:
                _inject(injector, "after_retiring_root_rmdir")
            journal = _phase(paths["journal"], journal, "old_deleted")
            _inject(injector, "after_old_delete")
        elif phase == "old_deleted":
            journal = _phase(paths["journal"], journal, "record_cleanup_pending")
        elif phase == "record_cleanup_pending":
            result = _read_valid_completion(paths["v1"])
            paths["journal"].unlink()
            _inject(injector, "after_record_cleanup")
            return result
        else:
            raise ValueError(f"unknown promotion journal phase: {phase}")


def promote_realism_v2(
    v1: Path,
    v2: Path,
    *,
    authorization: PromotionAuthorization,
    filesystem_policy: _PromotionFilesystemPolicy | None = None,
    failure_injector: FailureInjector | None = None,
) -> dict[str, Any]:
    """Promote only with explicit authority and a crash-resumable transaction."""

    if not isinstance(authorization, PromotionAuthorization) or authorization._authority is not _AUTHORITY:
        raise TypeError("valid typed promotion authorization is required")
    policy = _policy(filesystem_policy)
    paths = _require_requested_paths(v1, v2, policy)
    if paths["journal"].is_file():
        journal = _read_journal(paths["journal"])
        if (
            journal.get("v1_path") != str(paths["v1"])
            or journal.get("v2_path") != str(paths["v2"])
            or journal.get("retiring_path") != str(paths["retiring"])
        ):
            raise ValueError("promotion journal path identity mismatch")
    elif (
        paths["v1"].is_dir()
        and not paths["v2"].exists()
        and not paths["retiring"].exists()
        and (paths["v1"] / _COMPLETE_PATH).is_file()
    ):
        return _read_valid_completion(paths["v1"])
    else:
        eligibility = promotion_eligibility(paths["v1"], paths["v2"], filesystem_policy=policy)
        if eligibility.get("eligible") is not True:
            reasons = "; ".join(eligibility.get("reasons", ()))
            raise ValueError(f"not promotion eligible: {reasons}")
        audit = eligibility["audit"]
        audit_sha256 = hashlib.sha256(canonical_bytes(audit) + b"\n").hexdigest()
        v1_tree, v2_tree = _tree_sha256(paths["v1"]), _tree_sha256(paths["v2"])
        transaction_id = hashlib.sha256(
            canonical_bytes({"v1": v1_tree, "v2": v2_tree, "audit": audit_sha256})
        ).hexdigest()
        journal = {
            **dict(EVIDENCE_METADATA),
            "disclosure": DISCLOSURE,
            "schema": "realism_v2_promotion_journal_v2",
            "phase": "record_written",
            "transaction_id": transaction_id,
            "v1_path": str(paths["v1"]),
            "v2_path": str(paths["v2"]),
            "retiring_path": str(paths["retiring"]),
            "v1_tree_sha256": v1_tree,
            "v2_tree_sha256": v2_tree,
            "v1_source_hashes": eligibility["source_hashes"],
            "eligibility_verification_sha256": eligibility["verification_sha256"],
            "acceptance_sha256": eligibility["acceptance_sha256"],
            "audit_sha256": audit_sha256,
            "audit_payload": audit,
        }
        journal = _write_journal(paths["journal"], journal)
        _inject(failure_injector, "after_record_write")
    try:
        return _run_transaction(paths, policy, journal, failure_injector)
    except Exception:
        current = _read_journal(paths["journal"]) if paths["journal"].is_file() else journal
        if current.get("phase") == "completion_write_pending":
            try:
                exact_completion = _completion_is_exact(paths, current)
            except ValueError:
                exact_completion = False
            if exact_completion:
                _phase(
                    paths["journal"],
                    current,
                    "complete_written",
                    completion_sha256=sha256_file(paths["v1"] / _COMPLETE_PATH),
                    completed_tree_sha256=_tree_sha256(paths["v1"]),
                )
                raise
            _remove_transaction_completion(paths, current)
        _rollback(paths, current)
        raise


__all__ = [
    "PROMOTION_CONFIRMATION",
    "PromotionAuthorization",
    "authorize_promotion",
    "build_replaced_v1_audit",
    "promote_realism_v2",
    "promotion_eligibility",
]
