"""Thin CLI orchestrator for the Q2 oracle-conditioned scenario bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .compact_storage import export_prediction_csv
from .preflight import run_preflight_4a
from .scenario_artifacts import (
    atomic_bytes,
    canonical_bytes,
    generate,
    sha256_file,
    write_json,
)
from .scenario_evaluate import aggregate, evaluate
from .scenario_export import export
from .scenario_manifest import _cohort, prepare
from .scenario_verify import verify

_REALISM_ROOT = Path("scenarios/q2-realism-v2")
_R6_REALISM_ROOT = Path("scenarios/q2-realism-v2-r6")
_R6B_REALISM_ROOT = Path("scenarios/q2-realism-v2-r6b")
_LEGACY_ROOT = Path("scenario-q2-ideal")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_REALISM_ROOT_RESOLVED = (_PROJECT_ROOT / _REALISM_ROOT).resolve()
_R5_STAGES = ("prepare", "design", "confirm")
_R5_STAGE_ROOT = Path("09-logs/r5-cli-stages")
_R5_EXECUTION_CODE_SEAL = Path("00-manifest/r5-execution-code-seal.json")
_R5_SELECTED_BLOCKS_SHA256 = "d1496e86d93f3073b05b053234e4078f5c85f9903cc7e7fd3325fbfa1f380dd6"
_R5_SELECTED_BLOCKS_SIZE_BYTES = 53090
_R6_R5_EXTRACTED_MANIFEST_SHA256 = "9e184c537c4043143ba617c610f5baa72befa65a4a1b4bffe0e82e55466ac53f"
_R6_R5_EXTRACTED_MANIFEST_SIZE = 4630
_R6_R5_EXTRACTION_LOG_SHA256 = "b99d1737bbad12cd5b0e84f138436957b2d41ff1abd359e7e8cfbc0f289bc6ce"
_R6_R5_EXTRACTION_LOG_SIZE = 1845
_R6B_NREL_MANIFEST_SHA256 = "a0d6934419b10c08f648646d3a7acf86435e53af97dc18ad0b1d23d8724fc450"
_R6B_NREL_AUDIT_SHA256 = "35120739dd1000e148040836c04bb310b1315499eba736413a800ec694a0ab82"
_R6B_OK_ARCHIVE_SHA256 = "b83cfcf2deaa12f5b85f4bf5578b3b8bb72e8ee83154e2d615a953d7ea32df89"
_R6B_DKASC_MANIFEST_SHA256 = "99f4a2641b39d9c6f92f2e4120af56f80ac27d41a859a2cc52006b4e0cf75071"
_R6B_TASK21_REPORT = (
    _PROJECT_ROOT.parent
    / ".superpowers/sdd/2026-08-16-r5-nested-confirmation/"
    "task-21-r6b-live-preflight-retry-2-report.md"
)
_R6B_TASK21_REPORT_SHA256 = "e70096b871f6a4d78c0b84628dd7a60c640b1148715baa6d254a026ae895134a"
_R6B_PREPARE_PINNED = {
    "candidate_universe_sha256": "0330e0ffc2e7a0445fb72d75d045034668cff7113170cfd6e5493275c1a69fa4",
    "catalog_sha256": "975777378945a7cd8b0dbaad32d3642fe79785e09f091b5a1af7bbde0946435e",
    "confirmation_plan_sha256": "80e02db14d1ef9815fcf756cd0d4fe51e6ef8dd60abb8bab7f392d79f333d9cb",
    "preregistration_sha256": "d06310a7f6a788da487ec8a44fbcb8f378ddaacb0538d50bc93216fefb53a28e",
    "prior_coordinate_sha256": "58073a22e4d92c77be0008c3de707d9fe06c7c1cadfafdde9bbeb46121361e98",
    "range_definition_sha256": "8aa7adfe3b001f6d7dbc34fc09f9e4fd423591d91e867605373d9d13f6503d4c",
    "task21_report_sha256": _R6B_TASK21_REPORT_SHA256,
}


@dataclass(frozen=True)
class _R5PreparedContext:
    root: Path
    design_cohorts: dict[str, Any]
    confirmation_cohorts: Any
    confirmation_contract: Any
    artifacts: Any
    cross_binding_contract: Any


@dataclass(frozen=True)
class _R6bPrepareBundle:
    """Verified, non-published contents of one R6b Prepare bundle."""

    binding: dict[str, Any]
    counts: dict[str, int]
    expected_members: frozenset[str]
    expected_paths: frozenset[str]
    relative_hashes: dict[str, str]
    source_rows: tuple[dict[str, Any], ...]


def _r5_execution_code_paths() -> dict[str, Path]:
    from . import (
        cohort,
        constants,
        loader,
        nested_validation,
        realism,
        scenario_artifacts,
        scenario_candidate,
        scenario_confirmation,
        scenario_manifest,
        scenario_revision,
        scenario_revision_verify,
    )

    return {
        "cohort": Path(cohort.__file__),
        "constants": Path(constants.__file__),
        "loader": Path(loader.__file__),
        "nested_validation": Path(nested_validation.__file__),
        "pipeline": Path(__file__),
        "realism": Path(realism.__file__),
        "scenario_artifacts": Path(scenario_artifacts.__file__),
        "scenario_candidate": Path(scenario_candidate.__file__),
        "scenario_confirmation": Path(scenario_confirmation.__file__),
        "scenario_manifest": Path(scenario_manifest.__file__),
        "scenario_revision": Path(scenario_revision.__file__),
        "scenario_revision_verify": Path(scenario_revision_verify.__file__),
    }


def _r5_execution_code_seal_payload() -> dict[str, Any]:
    modules = []
    for name, source in sorted(_r5_execution_code_paths().items()):
        _reject_r5_cli_symlinks(source.absolute())
        resolved = source.resolve(strict=True)
        try:
            relative = resolved.relative_to(_PROJECT_ROOT).as_posix()
        except ValueError as error:
            raise ValueError(f"r5 execution code drift: {name} left the project root") from error
        modules.append(
            {"module": name, "path": relative, "sha256": sha256_file(resolved)}
        )
    return {"modules": modules, "schema": "r5_execution_code_seal_v1"}


def _r5_assert_execution_code_seal(root: Path) -> None:
    seal_path = root / _R5_EXECUTION_CODE_SEAL
    persisted = _r5_json(seal_path, label="execution code seal")
    expected = _r5_execution_code_seal_payload()
    if canonical_bytes(persisted) != canonical_bytes(expected):
        raise ValueError("r5 execution code seal changed or current code drifted")


class _ScenarioArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        if parsed.scenario_root is None:
            if parsed.stage in {
                "realism-r6b-preflight",
                "realism-r6b-prepare",
                "realism-r6b-design",
            }:
                parsed.scenario_root = _R6B_REALISM_ROOT
            elif parsed.stage == "realism-r6-preflight":
                parsed.scenario_root = _R6_REALISM_ROOT
            else:
                parsed.scenario_root = (
                    _REALISM_ROOT
                    if parsed.stage
                    in {
                        "realism-precheck",
                        "realism-run",
                        "realism-promote",
                        "realism-r5-prepare",
                        "realism-r5-design",
                        "realism-r5-confirm",
                    }
                    else _LEGACY_ROOT
                )
        return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = _ScenarioArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "preflight",
            "prepare",
            "generate",
            "evaluate",
            "aggregate",
            "export",
            "verify",
            "export-prediction",
            "all",
            "realism-precheck",
            "realism-run",
            "realism-promote",
            "realism-r5-prepare",
            "realism-r5-design",
            "realism-r5-confirm",
            "realism-r6-preflight",
            "realism-r6b-preflight",
            "realism-r6b-prepare",
            "realism-r6b-design",
        ),
    )
    parser.add_argument("--scenario-root", type=Path)
    parser.add_argument(
        "--source-scenario-root", type=Path, default=Path("scenarios/q2-ideal")
    )
    parser.add_argument("--screening-sha256")
    parser.add_argument("--confirm-promotion")
    parser.add_argument("--dkasc-prepared-manifest", type=Path)
    parser.add_argument("--nrel-prepared-manifest", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--candidate-seeds", nargs="+", type=int, default=[20260814])
    parser.add_argument("--protocol", default="dkasc-yulara")
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dkasc-manifest",
        type=Path,
        default=Path("data/manifests/dkasc-training-candidates.json"),
    )
    parser.add_argument(
        "--yulara-manifest",
        type=Path,
        default=Path("data/manifests/yulara-zero-parameter.json"),
    )
    parser.add_argument(
        "--nrel-manifest", type=Path, default=Path("data/manifests/nrel-candidates.json")
    )
    parser.add_argument(
        "--nrel-selection-audit",
        type=Path,
        default=Path("data/manifests/nrel-selected-cohort.audit.json"),
    )
    return parser


def _execute_r6_preflight(args: argparse.Namespace) -> dict[str, object]:
    """Authenticate and select the live R6 cohort without writing artifacts."""
    import pandas as pd

    from .loader import load_normalized_sites
    from .nested_validation import EXPECTED_AUXILIARY_OK_SITE_IDS
    from .scenario_r6 import (
        load_r6_prior_coordinates,
        r6_confirmation_plan_sha256,
        select_r6_confirmation_blocks,
    )
    from .scenario_revision import r6_catalog_sha256, r6_preregistration_sha256

    catalog_sha256 = r6_catalog_sha256()
    preregistration_sha256 = r6_preregistration_sha256()
    plan_sha256 = r6_confirmation_plan_sha256()
    if catalog_sha256 != "975777378945a7cd8b0dbaad32d3642fe79785e09f091b5a1af7bbde0946435e":
        raise ValueError("R6 catalog does not match the live preflight pin")
    if preregistration_sha256 != "2c17c2cc527528c024e7944559bc811c9757277c587d24d744312b0ebe5350d1":
        raise ValueError("R6 preregistration does not match the live preflight pin")
    if plan_sha256 != "483b452aa74d15a073137c596dde1102bc25c201a66958bf45edc88887f69f81":
        raise ValueError("R6 confirmation plan does not match the live preflight pin")

    source_root = _r6_existing_path(
        args.source_scenario_root, label="source scenario root", kind="directory"
    )
    target_root = _r6_absent_target(args.scenario_root)
    if (
        target_root == source_root
        or target_root.is_relative_to(source_root)
        or source_root.is_relative_to(target_root)
    ):
        raise ValueError("R6 target and retained source roots must be isolated")
    dkasc_manifest = _r6_existing_path(
        args.dkasc_manifest, label="DKASC manifest", kind="file"
    )
    _r6_existing_path(args.nrel_manifest, label="NREL manifest", kind="file")
    _r6_existing_path(
        args.nrel_selection_audit, label="NREL selection audit", kind="file"
    )
    extracted_manifest = _r6_existing_path(
        source_root / "01-cohorts/r5-confirmation-sources/manifest.json",
        label="R5 extracted-source manifest",
        kind="file",
    )
    selected_blocks = _r6_existing_path(
        source_root / "01-cohorts/selected-blocks.csv",
        label="R5 selected-block authority",
        kind="file",
    )
    if not extracted_manifest.is_relative_to(source_root) or not selected_blocks.is_relative_to(source_root):
        raise ValueError("R6 retained evidence escapes the source scenario root")
    free_bytes = _r6_free_bytes(target_root.parent)
    if free_bytes < 15 * 1024**3:
        raise ValueError("R6 preflight requires at least 15 GiB free")

    prior = tuple(load_r6_prior_coordinates(source_root, selected_blocks))
    dkasc_sites = tuple(load_normalized_sites(dkasc_manifest, split="validation"))
    expected_ok_ids = tuple(EXPECTED_AUXILIARY_OK_SITE_IDS)
    retained_oklahoma_hashes = _r6_authenticate_oklahoma_sources(
        source_root, extracted_manifest, expected_ok_ids
    )
    oklahoma_sites = tuple(
        load_normalized_sites(
            extracted_manifest,
            split="validation",
            site_ids=set(expected_ok_ids),
        )
    )

    _r6_validate_sources(
        dkasc_sites,
        oklahoma_sites,
        expected_ok_ids,
        retained_oklahoma_hashes=retained_oklahoma_hashes,
    )
    _r6_validate_prior(prior, dkasc_sites, expected_ok_ids, pd=pd)
    selection = select_r6_confirmation_blocks(dkasc_sites, oklahoma_sites, prior)
    counts = _r6_validate_selection(
        selection,
        prior=prior,
        dkasc_sites=dkasc_sites,
        oklahoma_sites=oklahoma_sites,
        plan_sha256=plan_sha256,
        pd=pd,
    )
    final_target_root = _r6_absent_target(args.scenario_root)
    if final_target_root != target_root:
        raise ValueError("R6 target scenario root changed during preflight")
    return {
        "schema": "r6_live_fresh_precheck_v1",
        "status": "passed",
        "catalog_sha256": catalog_sha256,
        "preregistration_sha256": preregistration_sha256,
        "confirmation_plan_sha256": plan_sha256,
        "source_revision": "r5",
        "revision": "r6",
        "prior_coordinate_count": 75,
        "selected_block_count": 27,
        "selected_site_count": 17,
        "dkasc_block_count": 12,
        "oklahoma_block_count": 15,
        "selection_seed": 20260818,
        "embargo_days": 14,
        "prior_coordinate_sha256": selection.prior_coordinate_sha256,
        "selected_coordinate_sha256": selection.selected_coordinate_sha256,
        "exact_overlap_count": counts["exact_overlap"],
        "prior_embargo_violation_count": counts["prior_embargo"],
        "oklahoma_pairwise_violation_count": counts["oklahoma_pairwise"],
        "held_out_used": False,
        "cohort_materialized": False,
        "artifacts_written": False,
        "r6_stage_executed": False,
        "prediction_generated": False,
        "free_bytes": free_bytes,
    }


def _execute_r6b_preflight(args: argparse.Namespace) -> dict[str, object]:
    """Authenticate and select the live R6b cohort without writing artifacts."""
    from .scenario_r6 import load_r6_prior_coordinates
    from .scenario_r6b import (
        r6b_confirmation_plan_sha256,
        r6b_oklahoma_candidate_sha256,
    )
    from .scenario_revision import r6_catalog_sha256, r6b_preregistration_sha256

    catalog_sha256 = r6_catalog_sha256()
    preregistration_sha256 = r6b_preregistration_sha256()
    plan_sha256 = r6b_confirmation_plan_sha256()
    candidate_sha256 = r6b_oklahoma_candidate_sha256()
    if catalog_sha256 != "975777378945a7cd8b0dbaad32d3642fe79785e09f091b5a1af7bbde0946435e":
        raise ValueError("R6b catalog does not match the live preflight pin")
    if preregistration_sha256 != "d06310a7f6a788da487ec8a44fbcb8f378ddaacb0538d50bc93216fefb53a28e":
        raise ValueError("R6b preregistration does not match the live preflight pin")
    if plan_sha256 != "80e02db14d1ef9815fcf756cd0d4fe51e6ef8dd60abb8bab7f392d79f333d9cb":
        raise ValueError("R6b confirmation plan does not match the live preflight pin")
    if candidate_sha256 != "0330e0ffc2e7a0445fb72d75d045034668cff7113170cfd6e5493275c1a69fa4":
        raise ValueError("R6b candidate universe does not match the live preflight pin")

    source_root = _r6b_existing_path(
        args.source_scenario_root, label="source scenario root", kind="directory"
    )
    target_root = _r6b_absent_target(args.scenario_root)
    if (
        target_root == source_root
        or target_root.is_relative_to(source_root)
        or source_root.is_relative_to(target_root)
    ):
        raise ValueError("R6b target and retained source roots must be isolated")
    selected_blocks = _r6b_existing_path(
        source_root / "01-cohorts/selected-blocks.csv",
        label="R5 selected-block authority",
        kind="file",
    )
    if not selected_blocks.is_relative_to(source_root):
        raise ValueError("R6b retained evidence escapes the source scenario root")
    free_bytes = _r6b_free_bytes(target_root.parent)
    if free_bytes < 15 * 1024**3:
        raise ValueError("R6b preflight requires at least 15 GiB free")

    prior = tuple(load_r6_prior_coordinates(source_root, selected_blocks))
    selection = _r6b_preflight_selection(args, prior)
    report = {
        "schema": "r6b_live_fresh_precheck_v1",
        "status": "passed",
        "catalog_sha256": catalog_sha256,
        "preregistration_sha256": preregistration_sha256,
        "confirmation_plan_sha256": plan_sha256,
        "candidate_universe_sha256": candidate_sha256,
        "source_revision": "r5",
        "revision": "r6b",
        "prior_coordinate_count": 75,
        "selected_block_count": 27,
        "selected_site_count": 27,
        "dkasc_block_count": 12,
        "oklahoma_block_count": 15,
        "selection_seed": 20260818,
        "embargo_days": 14,
        "prior_coordinate_sha256": str(selection.prior_coordinate_sha256),
        "selected_coordinate_sha256": str(selection.selected_coordinate_sha256),
        "selected_oklahoma_site_sha256": str(
            selection.selected_oklahoma_site_sha256
        ),
        "exact_overlap_count": 0,
        "prior_embargo_violation_count": 0,
        "oklahoma_pairwise_violation_count": 0,
        "held_out_used": False,
        "cohort_materialized": False,
        "artifacts_written": False,
        "r6b_stage_executed": False,
        "prediction_generated": False,
        "free_bytes": free_bytes,
    }
    final_target_root = _r6b_absent_target(args.scenario_root)
    if final_target_root != target_root:
        raise ValueError("R6b target scenario root changed during preflight")
    return report


def _r6b_preflight_selection(args: argparse.Namespace, prior):
    dkasc_sites, oklahoma_sites = _r6b_load_preflight_sources(args)
    selection = _r6b_select_confirmation_blocks(
        dkasc_sites, oklahoma_sites, prior
    )
    _r6b_validate_preflight_selection(
        selection,
        prior=prior,
        dkasc_sites=dkasc_sites,
        oklahoma_sites=oklahoma_sites,
    )
    return selection


def _r6b_raw_ok_archive_member(extraction_member: object) -> str:
    """Map one sealed Oklahoma extraction path to its raw flat ZIP member."""

    if not isinstance(extraction_member, str) or not extraction_member:
        raise ValueError("R6b Oklahoma extraction member is invalid")
    member_path = Path(extraction_member)
    if (
        member_path.is_absolute()
        or ".." in member_path.parts
        or member_path.as_posix() != extraction_member
        or len(member_path.parts) != 3
        or member_path.parts[:2] != ("sites", "OK")
        or member_path.name != member_path.parts[2]
    ):
        raise ValueError("R6b Oklahoma extraction member namespace is invalid")
    return member_path.name


def _r6b_load_preflight_sources(args: argparse.Namespace):
    import zipfile

    import numpy as np
    import pandas as pd

    from racepv.data import chronological_split_plan

    from .cohort import NormalizedSiteSeries
    from .loader import load_normalized_sites

    contract = _r6b_authenticated_ok20_contract(args)
    manifest = _r6b_authenticated_nrel_manifest(args.nrel_manifest)
    root = manifest.get("root")
    root_path = Path(root) if isinstance(root, str) else Path()
    if not isinstance(root, str) or not root or root_path.is_absolute():
        raise ValueError("R6b NREL manifest root is invalid")
    manifest_path = _r6b_existing_path(
        args.nrel_manifest, label="NREL manifest", kind="file"
    )
    archive_raw = manifest_path.parent / root_path / "archives/ok-pv-2006.zip"
    _reject_r5_cli_symlinks(archive_raw.absolute())
    try:
        archive_path = archive_raw.resolve(strict=True)
        archive_mode = archive_path.lstat().st_mode
    except OSError as error:
        raise ValueError("R6b Oklahoma archive is missing or unreadable") from error
    if not stat.S_ISREG(archive_mode):
        raise ValueError("R6b Oklahoma archive must be a regular file")
    declared_archive_sha256 = contract.get("archive_sha256")
    if (
        declared_archive_sha256 != _R6B_OK_ARCHIVE_SHA256
        or sha256_file(archive_path) != declared_archive_sha256
    ):
        raise ValueError("R6b Oklahoma archive hash binding is invalid")

    site_rows = contract.get("sites")
    if not isinstance(site_rows, tuple) or len(site_rows) != 20:
        raise ValueError("R6b Oklahoma source contract is invalid")
    raw_members = tuple(
        _r6b_raw_ok_archive_member(row["member"]) for row in site_rows
    )
    if len(set(raw_members)) != len(raw_members):
        raise ValueError("R6b Oklahoma source contract has duplicate raw members")
    oklahoma: list[NormalizedSiteSeries] = []
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            names = tuple(info.filename for info in infos)
            if len(names) != len(set(names)):
                raise ValueError("R6b Oklahoma archive contains duplicate members")
            for info in infos:
                member_path = Path(info.filename)
                if (
                    member_path.is_absolute()
                    or ".." in member_path.parts
                    or member_path.as_posix() != info.filename
                ):
                        raise ValueError("R6b Oklahoma archive contains an unsafe member")
            info_by_name = {info.filename: info for info in infos}
            infos_by_basename: dict[str, list[zipfile.ZipInfo]] = {}
            for info in infos:
                infos_by_basename.setdefault(Path(info.filename).name, []).append(info)
            if any(raw_member not in info_by_name for raw_member in raw_members):
                raise ValueError("R6b Oklahoma archive is missing an audited member")
            if any(
                len(infos_by_basename.get(raw_member, ())) != 1
                or infos_by_basename[raw_member][0].filename != raw_member
                for raw_member in raw_members
            ):
                raise ValueError("R6b Oklahoma archive has ambiguous raw member mapping")
            for row, raw_member in zip(site_rows, raw_members, strict=True):
                info = info_by_name[raw_member]
                unix_mode = info.external_attr >> 16
                kind = stat.S_IFMT(unix_mode)
                if info.is_dir() or kind not in {0, stat.S_IFREG}:
                    raise ValueError("R6b Oklahoma archive member is not regular")
                with archive.open(info, "r") as source:
                    content = source.read()
                if (
                    len(content) != row["member_size"]
                    or info.file_size != row["member_size"]
                    or hashlib.sha256(content).hexdigest() != row["member_sha256"]
                ):
                    raise ValueError("R6b Oklahoma archive member hash or size changed")
                lines = content.splitlines()
                try:
                    header = lines[0].split(b",")
                    timestamp_column = str(row["timestamp_column"]).encode("ascii")
                    timestamp_index = header.index(timestamp_column)
                    timestamp_values = [
                        line.split(b",")[timestamp_index].decode("ascii")
                        for line in lines[1:]
                        if line
                    ]
                    timestamps = pd.DatetimeIndex(
                        pd.to_datetime(
                            timestamp_values,
                            format=row["timestamp_format"],
                            errors="raise",
                        )
                    ).tz_localize(str(row["timezone"]))
                except (IndexError, UnicodeDecodeError, ValueError) as error:
                    raise ValueError("R6b Oklahoma timestamp column is malformed") from error
                if (
                    len(timestamps) < 4
                    or timestamps.has_duplicates
                    or not timestamps.is_monotonic_increasing
                    or not np.all(
                        timestamps[1:] - timestamps[:-1] == pd.Timedelta(minutes=5)
                    )
                ):
                    raise ValueError("R6b Oklahoma timestamp continuity is invalid")
                start = pd.Timestamp(row["analysis_start"]).tz_convert(str(row["timezone"]))
                end = pd.Timestamp(row["analysis_end"]).tz_convert(str(row["timezone"]))
                timestamps = timestamps[(timestamps >= start) & (timestamps <= end)]
                timestamps = timestamps[
                    (timestamps.minute % 15 == 0) & (timestamps.second == 0)
                ]
                if (
                    len(timestamps) < 4
                    or not np.all(
                        timestamps[1:] - timestamps[:-1] == pd.Timedelta(minutes=15)
                    )
                ):
                    raise ValueError("R6b Oklahoma 15-minute timestamp axis is invalid")
                timestamps = timestamps[chronological_split_plan(len(timestamps))["validation"]]
                if timestamps.empty:
                    raise ValueError("R6b Oklahoma validation timestamps are empty")
                zeros = np.zeros(len(timestamps), dtype=float)
                oklahoma.append(
                    NormalizedSiteSeries(
                        site_id=str(row["site_id"]),
                        dataset="nrel",
                        timestamps=timestamps,
                        power=zeros,
                        daylight=np.zeros(len(timestamps), dtype=bool),
                        split="validation",
                        raw_hashes=(str(row["member_sha256"]),),
                        spatial_role="validation_calibration",
                        state="OK",
                    )
                )
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("R6b Oklahoma archive is malformed") from error

    dkasc = tuple(load_normalized_sites(args.dkasc_manifest, split="validation"))
    if (
        len(dkasc) != 12
        or len({site.site_id for site in dkasc}) != 12
        or any(
            site.dataset != "dkasc"
            or site.split != "validation"
            or site.spatial_role != "source_train"
            or site.spatial_role in {"test", "zero_parameter_test"}
            for site in dkasc
        )
    ):
        raise ValueError("R6b DKASC source contract is invalid")
    if len(oklahoma) != 20 or len({site.site_id for site in oklahoma}) != 20:
        raise ValueError("R6b Oklahoma loaded source contract is invalid")
    return dkasc, tuple(oklahoma)


def _r6b_select_confirmation_blocks(dkasc_sites, oklahoma_sites, prior):
    from .scenario_r6b import select_r6b_confirmation_blocks

    return select_r6b_confirmation_blocks(dkasc_sites, oklahoma_sites, prior)


def _r6b_validate_preflight_selection(
    selection, *, prior, dkasc_sites, oklahoma_sites
) -> None:
    from dataclasses import replace

    import numpy as np
    import pandas as pd

    from .cohort import BLOCK_STEPS, _valid_blocks, stable_hash
    from .constants import STEP_MINUTES

    expected = {
        "candidate_universe_sha256": "0330e0ffc2e7a0445fb72d75d045034668cff7113170cfd6e5493275c1a69fa4",
        "embargo_days": 14,
        "plan_sha256": "80e02db14d1ef9815fcf756cd0d4fe51e6ef8dd60abb8bab7f392d79f333d9cb",
        "preregistration_sha256": "d06310a7f6a788da487ec8a44fbcb8f378ddaacb0538d50bc93216fefb53a28e",
        "prior_coordinate_sha256": "58073a22e4d92c77be0008c3de707d9fe06c7c1cadfafdde9bbeb46121361e98",
        "selection_seed": 20260818,
    }
    if any(
        type(getattr(selection, name, None)) is not type(value)
        or getattr(selection, name, None) != value
        for name, value in expected.items()
    ):
        raise ValueError("R6b selector selection pin binding is invalid")
    for name in ("selected_coordinate_sha256", "selected_oklahoma_site_sha256"):
        digest = getattr(selection, name, None)
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("R6b selector selection digest is invalid")

    blocks = getattr(selection, "blocks", None)
    dkasc_sources = tuple(dkasc_sites)
    oklahoma_sources = tuple(oklahoma_sites)
    source_by_id = {
        site.site_id: site for site in (*dkasc_sources, *oklahoma_sources)
    }
    dkasc_ids = {site.site_id for site in dkasc_sources}
    oklahoma_ids = {site.site_id for site in oklahoma_sources}
    if (
        not isinstance(blocks, tuple)
        or len(blocks) != 27
        or len(dkasc_sources) != 12
        or len(oklahoma_sources) != 20
        or len(source_by_id) != 32
        or len(dkasc_ids) != 12
        or len(oklahoma_ids) != 20
    ):
        raise ValueError("R6b selector selection cardinality is invalid")

    coordinates = tuple(
        (block.site_id, block.start.isoformat(), block.end.isoformat())
        for block in blocks
    )
    selected_dkasc = tuple(block for block in blocks if block.site_id in dkasc_ids)
    selected_oklahoma = tuple(
        block for block in blocks if block.site_id in oklahoma_ids
    )
    if (
        len(selected_dkasc) != 12
        or len(selected_oklahoma) != 15
        or len({block.site_id for block in blocks}) != 27
        or len(set(coordinates)) != 27
        or len({block.block_id for block in blocks}) != 27
        or {block.site_id for block in selected_dkasc} != dkasc_ids
        or len({block.site_id for block in selected_oklahoma}) != 15
    ):
        raise ValueError("R6b selector selection identities are invalid")

    interval = pd.Timedelta(minutes=STEP_MINUTES)
    for block in blocks:
        source = source_by_id.get(block.site_id)
        digest = stable_hash(
            block.dataset,
            block.site_id,
            block.split,
            block.start.isoformat(),
            block.end.isoformat(),
        )
        if source is None:
            raise ValueError("R6b selector selection source identity is invalid")
        location = source.timestamps.get_indexer([block.start])[0]
        source_interval = source.timestamps[location : location + BLOCK_STEPS]
        if (
            block.dataset != source.dataset
            or block.split != "validation"
            or block.spatial_role != source.spatial_role
            or block.state != source.state
            or block.raw_hashes != source.raw_hashes
            or block.spatial_role in {"test", "zero_parameter_test"}
            or block.block_id != digest[:20]
            or block.selection_hash != digest
            or block.n_steps != BLOCK_STEPS
            or block.end
            != block.start + pd.Timedelta(minutes=STEP_MINUTES * (BLOCK_STEPS - 1))
            or block.quarter != f"{block.start.year}Q{block.start.quarter}"
            or location < 0
            or len(source_interval) != BLOCK_STEPS
            or source_interval[-1] != block.end
            or not all(source_interval[1:] - source_interval[:-1] == interval)
        ):
            raise ValueError("R6b selector selection block binding is invalid")

    prior_rows = tuple(prior)
    prior_set = set(prior_rows)
    if (
        len(prior_rows) != 75
        or len(prior_set) != 75
        or _r6b_coordinate_sha256(prior_rows) != expected["prior_coordinate_sha256"]
        or set(coordinates) & prior_set
    ):
        raise ValueError("R6b selector selection prior overlap is invalid")
    embargo = pd.Timedelta(days=14)
    if any(
        block.site_id == site_id
        and _r6b_interval_gap(block.start, block.end, start, end, pd=pd) < embargo
        for block in blocks
        for site_id, start, end in prior_rows
    ):
        raise ValueError("R6b selector selection prior embargo is invalid")

    def eligible_blocks(site):
        metadata_site = replace(
            site, power=np.zeros(len(site.timestamps), dtype=float)
        )
        return tuple(
            block
            for _index, block in _valid_blocks(metadata_site)
            if (block.site_id, block.start.isoformat(), block.end.isoformat())
            not in prior_set
            and not any(
                block.site_id == site_id
                and _r6b_interval_gap(
                    block.start, block.end, start, end, pd=pd
                )
                < embargo
                for site_id, start, end in prior_rows
            )
        )

    def block_rank(block):
        return stable_hash(
            "nested-confirmation-r6b",
            20260818,
            block.site_id,
            block.start.isoformat(),
            block.end.isoformat(),
        )

    expected_blocks = []
    for site in sorted(dkasc_sources, key=lambda item: item.site_id):
        eligible = eligible_blocks(site)
        if not eligible:
            raise ValueError("R6b DKASC source lacks an authoritative block")
        expected_blocks.append(min(eligible, key=block_rank))
    feasible_oklahoma = {}
    for site in sorted(oklahoma_sources, key=lambda item: item.site_id):
        eligible = eligible_blocks(site)
        if eligible:
            feasible_oklahoma[site.site_id] = min(eligible, key=block_rank)
    if len(feasible_oklahoma) < 15:
        raise ValueError("R6b fewer than fifteen Oklahoma sources are eligible")
    authoritative_oklahoma = sorted(
        feasible_oklahoma,
        key=lambda site_id: stable_hash(
            "nested-confirmation-r6b-oklahoma-site", 20260818, site_id
        ),
    )[:15]
    expected_blocks.extend(
        feasible_oklahoma[site_id] for site_id in authoritative_oklahoma
    )
    expected_blocks = tuple(
        sorted(
            expected_blocks,
            key=lambda block: (
                block.dataset,
                block.site_id,
                block.start,
                block.end,
                block.block_id,
            ),
        )
    )
    if blocks != expected_blocks:
        raise ValueError("R6b selector selection is not the authoritative tuple")
    if (
        selection.selected_coordinate_sha256
        != _r6b_coordinate_sha256(coordinates)
        or selection.selected_oklahoma_site_sha256
        != _r6b_identity_sha256(block.site_id for block in selected_oklahoma)
    ):
        raise ValueError("R6b selector selection digest binding is invalid")


def _r6b_coordinate_sha256(coordinates) -> str:
    payload = [list(row) for row in sorted(coordinates)]
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _r6b_identity_sha256(site_ids) -> str:
    return hashlib.sha256(canonical_bytes(sorted(site_ids))).hexdigest()


def _r6b_interval_gap(start, end, other_start, other_end, *, pd):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    other_start, other_end = pd.Timestamp(other_start), pd.Timestamp(other_end)
    if end < other_start:
        return other_start - end
    if other_end < start:
        return start - other_end
    return pd.Timedelta(0)


def _r6b_authenticated_ok20_contract(args: argparse.Namespace) -> dict[str, object]:
    from .nested_validation import EXPECTED_AUXILIARY_OK_SITE_IDS
    from .scenario_r6b import validate_r6b_oklahoma_candidate_ids

    manifest = _r6b_authenticated_nrel_manifest(args.nrel_manifest)
    audit = _r6b_authenticated_nrel_selection_audit(args.nrel_selection_audit)
    expected_ids = validate_r6b_oklahoma_candidate_ids(
        audit.get("selected_site_ids_by_state", {}).get("OK", ())
        if isinstance(audit.get("selected_site_ids_by_state"), dict)
        else ()
    )
    if set(expected_ids) & set(EXPECTED_AUXILIARY_OK_SITE_IDS):
        raise ValueError("R6b OK20 candidate identities overlap the retained R5 five")
    if (
        manifest.get("schema_version") != 1
        or audit.get("schema_version") != 1
        or audit.get("status") != "selected_and_extracted"
        or audit.get("manifest") != Path(args.nrel_manifest).name
    ):
        raise ValueError("R6b NREL manifest/audit schema is invalid")

    provenance = manifest.get("provenance")
    archive_hashes = provenance.get("archive_sha256") if isinstance(provenance, dict) else None
    archive_sha256 = (
        archive_hashes.get("ok-pv-2006.zip")
        if isinstance(archive_hashes, dict)
        else None
    )
    if not _r6b_sha256(archive_sha256):
        raise ValueError("R6b Oklahoma archive hash is invalid")

    raw_entries = manifest.get("entries")
    raw_rows = audit.get("selected_raw_files")
    if not isinstance(raw_entries, list) or not isinstance(raw_rows, list):
        raise ValueError("R6b NREL manifest/audit entries are invalid")
    entries = [
        entry
        for entry in raw_entries
        if isinstance(entry, dict) and entry.get("site_id") in expected_ids
    ]
    rows = [
        row
        for row in raw_rows
        if isinstance(row, dict) and row.get("state") == "OK"
    ]
    if len(entries) != 20 or len(rows) != 20:
        raise ValueError("R6b manifest/audit must bind exactly twenty Oklahoma sites")
    entries_by_id = {entry.get("site_id"): entry for entry in entries}
    rows_by_id = {row.get("site_id"): row for row in rows}
    if (
        len(entries_by_id) != 20
        or len(rows_by_id) != 20
        or tuple(sorted(entries_by_id)) != expected_ids
        or tuple(sorted(rows_by_id)) != expected_ids
    ):
        raise ValueError("R6b manifest/audit Oklahoma identities are invalid")

    sites: list[dict[str, object]] = []
    for site_id in expected_ids:
        entry = entries_by_id[site_id]
        row = rows_by_id[site_id]
        files = entry.get("files")
        member = files[0] if isinstance(files, list) and len(files) == 1 else None
        member_path = Path(member) if isinstance(member, str) else Path()
        split = entry.get("split", "validation")
        if (
            "quality_scan" in entry
            or "quality_scan" in row
            or entry.get("dataset") != "nrel"
            or entry.get("state") != "OK"
            or entry.get("spatial_role") != "validation_calibration"
            or split != "validation"
            or entry.get("timestamp_column") != "LocalTime"
            or entry.get("timestamp_format") != "%m/%d/%y %H:%M"
            or entry.get("timezone") != "Etc/GMT+6"
            or entry.get("analysis_start") != "2006-01-01T08:00:00Z"
            or entry.get("analysis_end") != "2007-01-01T04:55:00Z"
            or not isinstance(member, str)
            or not member
            or member_path.is_absolute()
            or ".." in member_path.parts
            or member_path.as_posix() != member
            or not member.startswith("sites/OK/")
            or row.get("file") != member
            or row.get("state") != "OK"
            or type(row.get("size_bytes")) is not int
            or row["size_bytes"] <= 0
            or not _r6b_sha256(row.get("sha256"))
        ):
            raise ValueError("R6b Oklahoma identity/member metadata is invalid")
        try:
            _r6b_raw_ok_archive_member(member)
        except ValueError as error:
            raise ValueError("R6b Oklahoma identity/member metadata is invalid") from error
        sites.append(
            {
                "analysis_end": entry["analysis_end"],
                "analysis_start": entry["analysis_start"],
                "dataset": "nrel",
                "member": member,
                "member_sha256": row["sha256"],
                "member_size": row["size_bytes"],
                "site_id": site_id,
                "spatial_role": "validation_calibration",
                "split": "validation",
                "state": "OK",
                "timestamp_column": "LocalTime",
                "timestamp_format": "%m/%d/%y %H:%M",
                "timezone": "Etc/GMT+6",
            }
        )
    if len({_r6b_raw_ok_archive_member(row["member"]) for row in sites}) != 20:
        raise ValueError("R6b Oklahoma identity/member metadata has duplicate basenames")
    return {"archive_sha256": archive_sha256, "sites": tuple(sites)}


def _r6b_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _r6b_frozen_nrel_manifest_sha256() -> str:
    from .scenario_r6b import r6b_nrel_manifest_sha256

    return r6b_nrel_manifest_sha256()


def _r6b_frozen_nrel_audit_sha256() -> str:
    from .scenario_r6b import r6b_nrel_selection_audit_sha256

    return r6b_nrel_selection_audit_sha256()


def _r6b_authenticated_nrel_manifest(path: Path) -> dict[str, Any]:
    expected_sha256 = _r6b_frozen_nrel_manifest_sha256()
    if _R6B_NREL_MANIFEST_SHA256 != expected_sha256:
        raise ValueError("R6b NREL manifest runtime pin is invalid")
    return _r6b_pinned_canonical_json(
        path,
        label="NREL manifest",
        expected_sha256=expected_sha256,
    )


def _r6b_authenticated_nrel_selection_audit(path: Path) -> dict[str, Any]:
    expected_sha256 = _r6b_frozen_nrel_audit_sha256()
    if _R6B_NREL_AUDIT_SHA256 != expected_sha256:
        raise ValueError("R6b NREL selection audit runtime pin is invalid")
    return _r6b_pinned_canonical_json(
        path,
        label="NREL selection audit",
        expected_sha256=expected_sha256,
    )


def _r6b_pinned_canonical_json(
    path: Path, *, label: str, expected_sha256: str
) -> dict[str, Any]:
    secure = _r6b_existing_path(path, label=label, kind="file")
    try:
        content = secure.read_bytes()
        payload = json.loads(content)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"R6b {label} is malformed") from error
    if (
        hashlib.sha256(content).hexdigest() != expected_sha256
        or not isinstance(payload, dict)
        or content != canonical_bytes(payload) + b"\n"
        or _r6b_existing_path(secure, label=label, kind="file") != secure
    ):
        raise ValueError(f"R6b {label} canonical hash binding is invalid")
    return payload


def _r6b_absolute(path: Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if ".." in raw.parts:
        raise ValueError(f"R6b {label} path is noncanonical")
    absolute = raw if raw.is_absolute() else _PROJECT_ROOT / raw
    _reject_r5_cli_symlinks(absolute)
    return absolute


def _r6b_existing_path(path: Path, *, label: str, kind: str) -> Path:
    absolute = _r6b_absolute(path, label=label)
    try:
        resolved = absolute.resolve(strict=True)
        mode = absolute.lstat().st_mode
    except OSError as error:
        raise ValueError(f"R6b {label} is missing or unreadable") from error
    if absolute != resolved:
        raise ValueError(f"R6b {label} path is noncanonical")
    if kind == "file" and not stat.S_ISREG(mode):
        raise ValueError(f"R6b {label} must be a regular file")
    if kind == "directory" and not stat.S_ISDIR(mode):
        raise ValueError(f"R6b {label} must be a directory")
    return resolved


def _r6b_absent_target(path: Path) -> Path:
    absolute = _r6b_absolute(path, label="target scenario root")
    if absolute.exists() or absolute.is_symlink():
        raise ValueError("R6b target scenario root must be absent")
    parent = _r6b_existing_path(
        absolute.parent, label="target scenario parent", kind="directory"
    )
    if absolute.parent != parent:
        raise ValueError("R6b target scenario root path is noncanonical")
    return absolute


def _r6b_free_bytes(path: Path) -> int:
    free = shutil.disk_usage(path).free
    if type(free) is not int or free < 0:
        raise ValueError("R6b free-space result is invalid")
    return free


def _r6b_prepare_code_seal_payload() -> dict[str, Any]:
    """Seal the code that authenticates and writes the isolated R6b bundle."""
    from . import cohort, loader, realism, scenario_r6b, scenario_revision

    modules = {
        "cohort": Path(cohort.__file__),
        "loader": Path(loader.__file__),
        "pipeline": Path(__file__),
        "realism": Path(realism.__file__),
        "scenario_r6b": Path(scenario_r6b.__file__),
        "scenario_revision": Path(scenario_revision.__file__),
    }
    rows = []
    for name, path in sorted(modules.items()):
        secure = _r6b_existing_path(path, label=f"R6b code module {name}", kind="file")
        if not secure.is_relative_to(_PROJECT_ROOT):
            raise ValueError("R6b execution code escaped the project root")
        rows.append(
            {"module": name, "path": secure.relative_to(_PROJECT_ROOT).as_posix(), "sha256": sha256_file(secure)}
        )
    return {"modules": rows, "schema": "r6b_execution_code_seal_v1"}


def _r6b_prepare_pins() -> dict[str, str]:
    """Authenticate the frozen R6b/Task21 provenance before source reads."""
    from .scenario_r6b import r6b_confirmation_plan_sha256, r6b_oklahoma_candidate_sha256
    from .scenario_revision import (
        r5_precheck_ranges_sha256,
        r6_catalog_sha256,
        r6b_preregistration_sha256,
    )

    pins = {
        "catalog_sha256": r6_catalog_sha256(),
        "range_definition_sha256": r5_precheck_ranges_sha256(),
        "preregistration_sha256": r6b_preregistration_sha256(),
        "confirmation_plan_sha256": r6b_confirmation_plan_sha256(),
        "candidate_universe_sha256": r6b_oklahoma_candidate_sha256(),
        "prior_coordinate_sha256": _R6B_PREPARE_PINNED["prior_coordinate_sha256"],
    }
    report = _r6b_existing_path(
        _R6B_TASK21_REPORT, label="Task 21 live-preflight report", kind="file"
    )
    pins["task21_report_sha256"] = sha256_file(report)
    if pins != _R6B_PREPARE_PINNED:
        raise ValueError("R6b Prepare frozen Task21/catalog/range/preregistration pin drift")
    return pins


def _r6b_prepare_dkasc_sources(args: argparse.Namespace, dkasc_sites) -> tuple[dict[str, Any], ...]:
    """Bind the twelve selected DKASC series to their sealed raw source bytes."""
    manifest_path = _r6b_existing_path(
        args.dkasc_manifest, label="DKASC manifest", kind="file"
    )
    if sha256_file(manifest_path) != _R6B_DKASC_MANIFEST_SHA256:
        raise ValueError("R6b DKASC manifest pin drift")
    try:
        payload = json.loads(manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("R6b DKASC manifest is malformed") from error
    if not isinstance(payload, dict):
        raise ValueError("R6b DKASC manifest is malformed")
    root = payload.get("root")
    provenance = payload.get("provenance")
    raw_files = provenance.get("raw_files") if isinstance(provenance, dict) else None
    entries = payload.get("entries")
    if not isinstance(root, str) or not root or not isinstance(raw_files, list) or not isinstance(entries, list):
        raise ValueError("R6b DKASC manifest source contract is invalid")
    raw_by_name = {
        row.get("file"): row for row in raw_files if isinstance(row, dict) and isinstance(row.get("file"), str)
    }
    entries_by_id = {
        row.get("site_id"): row for row in entries if isinstance(row, dict) and isinstance(row.get("site_id"), str)
    }
    sites_by_id = {site.site_id: site for site in dkasc_sites}
    if len(sites_by_id) != 12 or set(entries_by_id) != set(sites_by_id):
        raise ValueError("R6b DKASC source identities are invalid")
    output: list[dict[str, Any]] = []
    for site_id in sorted(sites_by_id):
        site = sites_by_id[site_id]
        entry = entries_by_id[site_id]
        files = entry.get("files")
        filename = files[0] if isinstance(files, list) and len(files) == 1 else None
        raw = raw_by_name.get(filename)
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or raw is None
            or entry.get("dataset") != "dkasc"
            or entry.get("spatial_role") != "source_train"
            or site.dataset != "dkasc"
            or site.split != "validation"
            or site.spatial_role != "source_train"
            or site.spatial_role in {"test", "zero_parameter_test"}
            or not _r6b_sha256(raw.get("sha256"))
            or type(raw.get("size_bytes")) is not int
            or raw["size_bytes"] <= 0
        ):
            raise ValueError("R6b DKASC source role/hash contract is invalid")
        source = _r6b_resolve_prepare_dkasc_source(
            manifest_path,
            payload,
            filename=filename,
            expected_sha256=raw["sha256"],
            expected_size=raw["size_bytes"],
        )
        digest = raw["sha256"]
        if site.raw_hashes != (digest,):
            raise ValueError("R6b DKASC source hash binding is invalid")
        output.append(
            {
                "dataset": "dkasc",
                "input_path": source,
                "input_sha256": digest,
                "logical_member": f"sites/DKASC/{filename}",
                "raw_member": filename,
                "site_id": site_id,
                "spatial_role": "source_train",
                "split": "validation",
                "state": site.state,
            }
        )
    return tuple(output)


def _r6b_resolve_prepare_dkasc_source(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    filename: str,
    expected_sha256: str,
    expected_size: int,
) -> Path:
    """Resolve one sealed DKASC source from the sole frozen manifest root."""
    root = manifest.get("root")
    root_path = Path(root) if isinstance(root, str) else Path()
    filename_path = Path(filename) if isinstance(filename, str) else Path()
    if (
        not isinstance(root, str)
        or root_path.parts != ("..", "raw", "dkasc")
        or root_path.is_absolute()
        or not isinstance(filename, str)
        or not filename
        or filename_path.is_absolute()
        or filename_path.name != filename
        or len(filename_path.parts) != 1
        or not _r6b_sha256(expected_sha256)
        or type(expected_size) is not int
        or expected_size <= 0
    ):
        if not isinstance(filename, str) or not filename or filename_path.is_absolute() or filename_path.name != filename or len(filename_path.parts) != 1:
            raise ValueError("R6b DKASC source filename is invalid")
        raise ValueError("R6b DKASC manifest root is invalid")
    secure_manifest = _r6b_existing_path(
        manifest_path, label="DKASC manifest", kind="file"
    )
    project_root = _PROJECT_ROOT.resolve(strict=True)
    if not secure_manifest.is_relative_to(project_root):
        raise ValueError("R6b DKASC manifest escaped the project root")
    source_raw = secure_manifest.parent / root_path / filename_path
    _reject_r5_cli_symlinks(source_raw.absolute())
    try:
        source = source_raw.resolve(strict=True)
        source_mode = source_raw.lstat().st_mode
    except OSError as error:
        raise ValueError("R6b DKASC source is missing or unreadable") from error
    if not source.is_relative_to(project_root):
        raise ValueError("R6b DKASC source escaped the project root")
    if not stat.S_ISREG(source_mode):
        raise ValueError("R6b DKASC source must be a regular file")
    if source.stat().st_size != expected_size or sha256_file(source) != expected_sha256:
        raise ValueError("R6b DKASC source hash binding is invalid")
    return source


def _r6b_resolve_prepare_ok_archive(
    manifest_path: Path, manifest: Mapping[str, Any], *, expected_sha256: str
) -> Path:
    """Resolve only the sealed manifest-relative Oklahoma archive for Prepare."""
    root = manifest.get("root")
    root_path = Path(root) if isinstance(root, str) else Path()
    if (
        not isinstance(root, str)
        or root_path.parts != ("..", "raw", "nrel")
        or root_path.is_absolute()
        or not _r6b_sha256(expected_sha256)
    ):
        raise ValueError("R6b NREL manifest root is invalid")
    secure_manifest = _r6b_existing_path(
        manifest_path, label="NREL manifest", kind="file"
    )
    project_root = _PROJECT_ROOT.resolve(strict=True)
    if not secure_manifest.is_relative_to(project_root):
        raise ValueError("R6b NREL manifest escaped the project root")
    archive_raw = secure_manifest.parent / root_path / "archives/ok-pv-2006.zip"
    _reject_r5_cli_symlinks(archive_raw.absolute())
    try:
        archive = archive_raw.resolve(strict=True)
        archive_mode = archive.lstat().st_mode
    except OSError as error:
        raise ValueError("R6b Oklahoma archive is missing or unreadable") from error
    if not archive.is_relative_to(project_root):
        raise ValueError("R6b Oklahoma archive escaped the project root")
    if not stat.S_ISREG(archive_mode):
        raise ValueError("R6b Oklahoma archive must be a regular file")
    if sha256_file(archive) != expected_sha256:
        raise ValueError("R6b Oklahoma archive hash binding is invalid")
    return archive


def _r6b_prepare_context(args: argparse.Namespace) -> Any:
    """Rebuild the authenticated selection once and attach only sealed sources."""
    from .scenario_r6 import load_r6_prior_coordinates

    pins = _r6b_prepare_pins()
    source_root = _r6b_existing_path(
        args.source_scenario_root, label="source scenario root", kind="directory"
    )
    selected_blocks = _r6b_existing_path(
        source_root / "01-cohorts/selected-blocks.csv",
        label="R5 selected-block authority",
        kind="file",
    )
    prior = tuple(load_r6_prior_coordinates(source_root, selected_blocks))
    dkasc_sites, oklahoma_sites = _r6b_load_preflight_sources(args)
    selection = _r6b_select_confirmation_blocks(dkasc_sites, oklahoma_sites, prior)
    _r6b_validate_preflight_selection(
        selection,
        prior=prior,
        dkasc_sites=dkasc_sites,
        oklahoma_sites=oklahoma_sites,
    )
    contract = _r6b_authenticated_ok20_contract(args)
    manifest_path = _r6b_existing_path(args.nrel_manifest, label="NREL manifest", kind="file")
    manifest = _r6b_authenticated_nrel_manifest(manifest_path)
    archive = _r6b_resolve_prepare_ok_archive(
        manifest_path, manifest, expected_sha256=str(contract["archive_sha256"])
    )
    return type("_R6bPrepareContext", (), {
        "archive_path": archive,
        "dkasc_sources": _r6b_prepare_dkasc_sources(args, dkasc_sites),
        "oklahoma_sources": tuple(contract["sites"]),
        "pins": pins,
        "selection": selection,
    })()


def _r6b_prepare_blocks(selection: Any, dkasc_sources, oklahoma_sources) -> tuple[Any, ...]:
    """Fail closed on the exact 12+15 unique selected tuple before any write."""
    blocks = tuple(getattr(selection, "blocks", ()))
    dkasc_by_id = {row["site_id"]: row for row in dkasc_sources}
    oklahoma_by_id = {row["site_id"]: row for row in oklahoma_sources}
    selected_dkasc = tuple(block for block in blocks if getattr(block, "dataset", None) == "dkasc")
    selected_oklahoma = tuple(block for block in blocks if getattr(block, "dataset", None) == "nrel")
    coordinates = tuple(
        (getattr(block, "site_id", None), getattr(block, "start", None), getattr(block, "end", None))
        for block in blocks
    )
    if (
        len(blocks) != 27
        or len(selected_dkasc) != 12
        or len(selected_oklahoma) != 15
        or len({getattr(block, "site_id", None) for block in blocks}) != 27
        or len(set(coordinates)) != 27
        or len({getattr(block, "block_id", None) for block in blocks}) != 27
        or {block.site_id for block in selected_dkasc} != set(dkasc_by_id)
        or len({block.site_id for block in selected_oklahoma}) != 15
        or any(block.site_id not in oklahoma_by_id for block in selected_oklahoma)
    ):
        raise ValueError("R6b Prepare selected block topology is invalid")
    for block in blocks:
        row = dkasc_by_id.get(block.site_id) or oklahoma_by_id.get(block.site_id)
        if (
            row is None
            or block.dataset != row["dataset"]
            or block.split != "validation"
            or block.spatial_role != row["spatial_role"]
            or block.state != row["state"]
            or not isinstance(block.block_id, str)
            or not isinstance(block.selection_hash, str)
            or block.n_steps != 1344
        ):
            raise ValueError("R6b Prepare selected source/block binding is invalid")
    return tuple(sorted(blocks, key=lambda block: (block.dataset, block.site_id, block.start, block.end, block.block_id)))


def _r6b_prepare_ok_source_bytes(archive_path: Path, rows: tuple[dict[str, Any], ...]) -> dict[str, bytes]:
    """Read only selected, sealed flat ZIP members after namespace validation."""
    selected = {row["site_id"]: row for row in rows}
    if len(selected) != len(rows):
        raise ValueError("R6b Prepare Oklahoma selected sites are duplicated")
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("R6b Prepare Oklahoma archive has duplicate members")
            by_name = {info.filename: info for info in infos}
            content: dict[str, bytes] = {}
            for site_id, row in selected.items():
                logical = row.get("logical_member", row.get("member"))
                raw = row.get("raw_member") or _r6b_raw_ok_archive_member(logical)
                if (
                    not isinstance(logical, str)
                    or _r6b_raw_ok_archive_member(logical) != raw
                    or not isinstance(raw, str)
                    or raw not in by_name
                    or by_name[raw].filename != raw
                ):
                    raise ValueError("R6b Prepare Oklahoma ZIP namespace binding is invalid")
                info = by_name[raw]
                kind = stat.S_IFMT(info.external_attr >> 16)
                if info.is_dir() or kind not in {0, stat.S_IFREG}:
                    raise ValueError("R6b Prepare Oklahoma ZIP member is not regular")
                with archive.open(info, "r") as handle:
                    raw_bytes = handle.read()
                if (
                    len(raw_bytes) != row.get("member_size")
                    or info.file_size != row.get("member_size")
                    or hashlib.sha256(raw_bytes).hexdigest() != row.get("member_sha256")
                ):
                    raise ValueError("R6b Prepare Oklahoma ZIP member hash binding is invalid")
                content[site_id] = raw_bytes
            return content
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("R6b Prepare Oklahoma archive is malformed") from error


def _r6b_prepare_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _r6b_prepare_block_csv(blocks: tuple[Any, ...]) -> bytes:
    columns = (
        "block_id", "dataset", "site_id", "spatial_role", "state", "split",
        "start", "end", "quarter", "n_steps", "selection_hash",
    )
    lines = [",".join(columns)]
    for block in blocks:
        lines.append(
            ",".join(
                str(value)
                for value in (
                    block.block_id, block.dataset, block.site_id, block.spatial_role,
                    block.state or "", block.split, block.start.isoformat(), block.end.isoformat(),
                    block.quarter, block.n_steps, block.selection_hash,
                )
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def build_r6b_prepare_bundle(args: argparse.Namespace, *, temporary_root: Path) -> _R6bPrepareBundle:
    """Materialize and verify an R6b source bundle without publishing a stage/root."""
    context = _r6b_prepare_context(args)
    if dict(context.pins) != _R6B_PREPARE_PINNED:
        raise ValueError("R6b Prepare frozen input pin drift")
    selection = context.selection
    if (
        getattr(selection, "prior_coordinate_sha256", None) != _R6B_PREPARE_PINNED["prior_coordinate_sha256"]
        or getattr(selection, "candidate_universe_sha256", None) != _R6B_PREPARE_PINNED["candidate_universe_sha256"]
        or getattr(selection, "plan_sha256", None) != _R6B_PREPARE_PINNED["confirmation_plan_sha256"]
        or getattr(selection, "preregistration_sha256", None) != _R6B_PREPARE_PINNED["preregistration_sha256"]
        or getattr(selection, "selection_seed", None) != 20260818
        or getattr(selection, "embargo_days", None) != 14
    ):
        raise ValueError("R6b Prepare selection pin drift")
    dkasc_sources = tuple(context.dkasc_sources)
    oklahoma_sources = tuple(context.oklahoma_sources)
    blocks = _r6b_prepare_blocks(selection, dkasc_sources, oklahoma_sources)
    selected_ids = {block.site_id for block in blocks}
    selected_dkasc = tuple(row for row in dkasc_sources if row["site_id"] in selected_ids)
    selected_oklahoma = tuple(row for row in oklahoma_sources if row["site_id"] in selected_ids)
    if len(dkasc_sources) != 12 or len(oklahoma_sources) != 20 or len(selected_dkasc) != 12 or len(selected_oklahoma) != 15:
        raise ValueError("R6b Prepare source selection cardinality is invalid")
    if any(
        row.get("dataset") != "dkasc" or row.get("split") != "validation" or row.get("spatial_role") != "source_train"
        for row in selected_dkasc
    ):
        raise ValueError("R6b Prepare DKASC source role is invalid")
    if any(
        row.get("dataset") != "nrel" or row.get("split") != "validation" or row.get("spatial_role") != "validation_calibration" or row.get("state") != "OK"
        for row in selected_oklahoma
    ):
        raise ValueError("R6b Prepare Oklahoma source role is invalid")
    if temporary_root.exists() or temporary_root.is_symlink():
        raise ValueError("R6b Prepare temporary root must be absent")
    if getattr(args, "scenario_root", None) is not None and Path(args.scenario_root).exists():
        raise ValueError("R6b Prepare final scenario root must remain absent during bundle build")

    dkasc_bytes: dict[str, bytes] = {}
    for row in selected_dkasc:
        source = _r6b_existing_path(row["input_path"], label=f"R6b DKASC source {row['site_id']}", kind="file")
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != row.get("input_sha256"):
            raise ValueError("R6b Prepare DKASC source hash changed")
        dkasc_bytes[row["site_id"]] = content
    oklahoma_bytes = _r6b_prepare_ok_source_bytes(Path(context.archive_path), selected_oklahoma)

    source_rows: list[dict[str, Any]] = []
    source_files: list[tuple[Path, bytes]] = []
    for row in sorted(selected_dkasc, key=lambda item: item["site_id"]):
        output_relative = f"01-cohorts/r6b-confirmation-sources/dkasc/{row['raw_member']}"
        output = temporary_root / output_relative
        content = dkasc_bytes[row["site_id"]]
        source_files.append((output, content))
        source_rows.append({
            "dataset": "dkasc", "input_sha256": row["input_sha256"], "logical_member": row["logical_member"],
            "output_path": output_relative, "output_sha256": hashlib.sha256(content).hexdigest(), "raw_member": row["raw_member"],
            "site_id": row["site_id"], "spatial_role": row["spatial_role"], "split": row["split"], "state": row["state"],
        })
    for row in sorted(selected_oklahoma, key=lambda item: item["site_id"]):
        logical_member = row.get("logical_member") or row.get("member")
        raw_member = _r6b_raw_ok_archive_member(logical_member)
        output_relative = f"01-cohorts/r6b-confirmation-sources/oklahoma/{raw_member}"
        output = temporary_root / output_relative
        content = oklahoma_bytes[row["site_id"]]
        source_files.append((output, content))
        source_rows.append({
            "dataset": "nrel", "input_sha256": row["member_sha256"], "logical_member": logical_member,
            "output_path": output_relative, "output_sha256": hashlib.sha256(content).hexdigest(), "raw_member": raw_member,
            "site_id": row["site_id"], "spatial_role": row["spatial_role"], "split": row["split"], "state": row["state"],
        })
    source_rows.sort(key=lambda row: (row["dataset"], row["site_id"]))
    expected_members = frozenset(row["logical_member"] for row in source_rows)
    if len(source_rows) != 27 or len(expected_members) != 27:
        raise ValueError("R6b Prepare source member identities are invalid")
    counts = {"dkasc": 12, "oklahoma": 15, "total": 27}
    block_csv = _r6b_prepare_block_csv(blocks)
    protocol = {
        "schema": "r6b_confirmation_protocol_v1", "counts": counts,
        "embargo_days": 14, "held_out_allowed": False, "selection_seed": 20260818,
        "source_revision": "r5", "revision": "r6b", **dict(context.pins),
    }
    source_manifest = {"schema": "r6b_confirmation_source_manifest_v1", "sources": source_rows}
    extraction_log = {"schema": "r6b_source_extraction_log_v1", "sources": source_rows}
    binding = {
        "schema": "r6b_prepare_binding_v1", "counts": counts, "selected_block_count": 27,
        "selected_site_count": 27, "selected_coordinate_sha256": selection.selected_coordinate_sha256,
        "selected_oklahoma_site_sha256": selection.selected_oklahoma_site_sha256,
        "sources": source_rows, **dict(context.pins),
    }
    payloads = {
        "00-manifest/r6b-execution-code-seal.json": _r6b_prepare_code_seal_payload(),
        "00-manifest/r6b-prepare-binding.json": binding,
        "01-cohorts/r6b-confirmation-blocks.csv": block_csv,
        "01-cohorts/r6b-confirmation-protocol.json": protocol,
        "01-cohorts/r6b-confirmation-sources/manifest.json": source_manifest,
        "09-logs/r6b-source-extraction.json": extraction_log,
    }
    temporary_root.mkdir(parents=True, exist_ok=False)
    for path, content in source_files:
        _r6b_prepare_write(path, content)
    for relative, payload in payloads.items():
        content = payload if isinstance(payload, bytes) else canonical_bytes(payload) + b"\n"
        _r6b_prepare_write(temporary_root / relative, content)
    expected_paths = frozenset([*payloads, *(path.relative_to(temporary_root).as_posix() for path, _content in source_files)])
    relative_hashes = _r6b_verify_prepare_bundle(temporary_root, expected_paths, source_rows, payloads)
    return _R6bPrepareBundle(
        binding=binding, counts=counts, expected_members=expected_members, expected_paths=expected_paths,
        relative_hashes=relative_hashes, source_rows=tuple(source_rows),
    )


def _r6b_verify_prepare_bundle(root: Path, expected_paths: frozenset[str], source_rows, payloads) -> dict[str, str]:
    """Verify exact path, canonical payload, regular-file and hash bindings."""
    secure_root = _r6b_existing_path(root, label="R6b Prepare temporary bundle", kind="directory")
    actual_paths = frozenset(
        path.relative_to(secure_root).as_posix()
        for path in secure_root.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    if actual_paths != expected_paths:
        raise ValueError("R6b Prepare bundle path set is invalid")
    hashes: dict[str, str] = {}
    for relative in sorted(expected_paths):
        path = secure_root / relative
        secure = _r6b_existing_path(path, label=f"R6b Prepare bundle {relative}", kind="file")
        expected = payloads.get(relative)
        if expected is not None:
            expected_bytes = expected if isinstance(expected, bytes) else canonical_bytes(expected) + b"\n"
            if secure.read_bytes() != expected_bytes:
                raise ValueError("R6b Prepare canonical payload changed")
        hashes[relative] = sha256_file(secure)
    for row in source_rows:
        output = secure_root / row["output_path"]
        if sha256_file(output) != row["output_sha256"] or row["input_sha256"] != row["output_sha256"]:
            raise ValueError("R6b Prepare source output hash binding is invalid")
    return hashes


_R6B_PREPARE_STAGE = Path("09-logs/r6b-cli-stages/prepare.json")
_R6B_DESIGN_STAGE = Path("09-logs/r6b-cli-stages/design.json")
_R6B_DESIGN_BUNDLE = Path("00-manifest/r6b-design-freeze")
_R6B_DESIGN_SEAL = Path("00-manifest/r6b-design-execution-code-seal.json")
_R6B_DESIGN_AUTHORITY = Path("00-manifest/r6b-confirmation-authority.json")
_R6B_PREPARE_STATIC_PATHS = frozenset(
    {
        "00-manifest/r6b-execution-code-seal.json",
        "00-manifest/r6b-prepare-binding.json",
        "01-cohorts/r6b-confirmation-blocks.csv",
        "01-cohorts/r6b-confirmation-protocol.json",
        "01-cohorts/r6b-confirmation-sources/manifest.json",
        "09-logs/r6b-source-extraction.json",
    }
)


def _r6b_prepare_json(root: Path, relative: str, *, schema: str) -> dict[str, Any]:
    """Read one canonical, regular persisted Prepare payload."""
    path = _r6b_existing_path(
        root / relative, label=f"R6b Prepare persisted {relative}", kind="file"
    )
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"R6b Prepare persisted {relative} is malformed") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != schema
        or raw != canonical_bytes(payload) + b"\n"
    ):
        raise ValueError(f"R6b Prepare persisted {relative} is not canonical")
    return payload


def _r6b_prepare_stage_payload(bundle: _R6bPrepareBundle) -> dict[str, Any]:
    """Return the single non-self-referential completed Prepare receipt."""
    return {
        "counts": dict(bundle.counts),
        "lifecycle": {
            "confirmation_executed": False,
            "design_executed": False,
            "prediction_generated": False,
        },
        "relative_sha256": dict(sorted(bundle.relative_hashes.items())),
        "revision": "r6b",
        "schema": "r6b_cli_stage_v1",
        "source_revision": "r5",
        "stage": "prepare",
        "status": "completed",
        "task21_report_sha256": bundle.binding["task21_report_sha256"],
    }


def _r6b_prepare_source_rows(root: Path) -> tuple[dict[str, Any], ...]:
    manifest = _r6b_prepare_json(
        root,
        "01-cohorts/r6b-confirmation-sources/manifest.json",
        schema="r6b_confirmation_source_manifest_v1",
    )
    rows = manifest.get("sources")
    expected_keys = {
        "dataset",
        "input_sha256",
        "logical_member",
        "output_path",
        "output_sha256",
        "raw_member",
        "site_id",
        "spatial_role",
        "split",
        "state",
    }
    if (
        not isinstance(rows, list)
        or len(rows) != 27
        or any(not isinstance(row, dict) or set(row) != expected_keys for row in rows)
    ):
        raise ValueError("R6b Prepare source manifest contract is invalid")
    source_rows = tuple(rows)
    identifiers = {(row["dataset"], row["site_id"]) for row in source_rows}
    output_paths = {row["output_path"] for row in source_rows}
    if len(identifiers) != 27 or len(output_paths) != 27:
        raise ValueError("R6b Prepare source manifest identities are invalid")
    for row in source_rows:
        dataset = row["dataset"]
        logical = row["logical_member"]
        raw_member = row["raw_member"]
        output_path = row["output_path"]
        if (
            dataset not in {"dkasc", "nrel"}
            or not all(isinstance(row[key], str) and row[key] for key in expected_keys - {"state"})
            or row["state"] is not None and not isinstance(row["state"], str)
            or not isinstance(output_path, str)
            or Path(output_path).is_absolute()
            or ".." in Path(output_path).parts
            or not isinstance(raw_member, str)
            or Path(raw_member).name != raw_member
            or len(row["input_sha256"]) != 64
            or len(row["output_sha256"]) != 64
        ):
            raise ValueError("R6b Prepare source manifest row is invalid")
        if dataset == "dkasc":
            valid = (
                logical == f"sites/DKASC/{raw_member}"
                and output_path == f"01-cohorts/r6b-confirmation-sources/dkasc/{raw_member}"
                and row["spatial_role"] == "source_train"
                and row["split"] == "validation"
            )
        else:
            valid = (
                _r6b_raw_ok_archive_member(logical) == raw_member
                and output_path == f"01-cohorts/r6b-confirmation-sources/oklahoma/{raw_member}"
                and row["state"] == "OK"
                and row["spatial_role"] == "validation_calibration"
                and row["split"] == "validation"
            )
        if not valid:
            raise ValueError("R6b Prepare source namespace/role contract is invalid")
    if sum(row["dataset"] == "dkasc" for row in source_rows) != 12:
        raise ValueError("R6b Prepare DKASC source count is invalid")
    return source_rows


def _r6b_verify_published_prepare(root: Path) -> dict[str, Any]:
    """Fail closed on an exact, previously completed R6b Prepare root."""
    secure_root = _r6b_existing_path(root, label="R6b Prepare target scenario root", kind="directory")
    source_rows = _r6b_prepare_source_rows(secure_root)
    expected_bundle_paths = frozenset(
        {*_R6B_PREPARE_STATIC_PATHS, *(row["output_path"] for row in source_rows)}
    )
    expected_all_paths = frozenset({*expected_bundle_paths, _R6B_PREPARE_STAGE.as_posix()})
    actual_files: set[str] = set()
    actual_dirs: set[str] = set()
    for path in secure_root.rglob("*"):
        relative = path.relative_to(secure_root).as_posix()
        if path.is_symlink():
            raise ValueError("R6b Prepare persisted bundle contains a symlink")
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            actual_files.add(relative)
        elif stat.S_ISDIR(mode):
            actual_dirs.add(relative)
        else:
            raise ValueError("R6b Prepare persisted bundle contains a nonregular entry")
    expected_dirs = {
        parent.as_posix()
        for relative in expected_all_paths
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    if actual_files != expected_all_paths or actual_dirs != expected_dirs:
        raise ValueError("R6b Prepare persisted bundle path set is invalid")

    binding = _r6b_prepare_json(
        secure_root, "00-manifest/r6b-prepare-binding.json", schema="r6b_prepare_binding_v1"
    )
    protocol = _r6b_prepare_json(
        secure_root, "01-cohorts/r6b-confirmation-protocol.json", schema="r6b_confirmation_protocol_v1"
    )
    extraction = _r6b_prepare_json(
        secure_root, "09-logs/r6b-source-extraction.json", schema="r6b_source_extraction_log_v1"
    )
    seal = _r6b_prepare_json(
        secure_root, "00-manifest/r6b-execution-code-seal.json", schema="r6b_execution_code_seal_v1"
    )
    counts = {"dkasc": 12, "oklahoma": 15, "total": 27}
    if (
        binding.get("counts") != counts
        or binding.get("selected_block_count") != 27
        or binding.get("selected_site_count") != 27
        or binding.get("sources") != list(source_rows)
        or extraction != {"schema": "r6b_source_extraction_log_v1", "sources": list(source_rows)}
        or protocol.get("counts") != counts
        or protocol.get("source_revision") != "r5"
        or protocol.get("revision") != "r6b"
        or protocol.get("selection_seed") != 20260818
        or protocol.get("embargo_days") != 14
        or protocol.get("held_out_allowed") is not False
        or seal != _r6b_prepare_code_seal_payload()
    ):
        raise ValueError("R6b Prepare persisted payload binding is invalid")
    for key, digest in _R6B_PREPARE_PINNED.items():
        if binding.get(key) != digest or protocol.get(key) != digest:
            raise ValueError("R6b Prepare persisted provenance pin is invalid")

    block_path = _r6b_existing_path(
        secure_root / "01-cohorts/r6b-confirmation-blocks.csv",
        label="R6b Prepare persisted block CSV",
        kind="file",
    )
    try:
        rows = list(csv.DictReader(block_path.read_text(encoding="utf-8").splitlines()))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError("R6b Prepare persisted block CSV is malformed") from error
    expected_columns = [
        "block_id", "dataset", "site_id", "spatial_role", "state", "split",
        "start", "end", "quarter", "n_steps", "selection_hash",
    ]
    if (
        len(rows) != 27
        or set((row.get("dataset"), row.get("site_id")) for row in rows)
        != {(row["dataset"], row["site_id"]) for row in source_rows}
        or len({row.get("block_id") for row in rows}) != 27
        or any(list(row) != expected_columns for row in rows)
    ):
        raise ValueError("R6b Prepare persisted block CSV contract is invalid")

    relative_hashes: dict[str, str] = {}
    for relative in sorted(expected_bundle_paths):
        path = _r6b_existing_path(
            secure_root / relative, label=f"R6b Prepare persisted {relative}", kind="file"
        )
        relative_hashes[relative] = sha256_file(path)
    for row in source_rows:
        if (
            sha256_file(secure_root / row["output_path"]) != row["output_sha256"]
            or row["input_sha256"] != row["output_sha256"]
        ):
            raise ValueError("R6b Prepare persisted source hash binding is invalid")
    expected_report = {
        "counts": counts,
        "lifecycle": {
            "confirmation_executed": False,
            "design_executed": False,
            "prediction_generated": False,
        },
        "relative_sha256": relative_hashes,
        "revision": "r6b",
        "schema": "r6b_cli_stage_v1",
        "source_revision": "r5",
        "stage": "prepare",
        "status": "completed",
        "task21_report_sha256": _R6B_PREPARE_PINNED["task21_report_sha256"],
    }
    report = _r6b_prepare_json(
        secure_root, _R6B_PREPARE_STAGE.as_posix(), schema="r6b_cli_stage_v1"
    )
    if report != expected_report:
        raise ValueError("R6b Prepare completed stage report is invalid")
    return report


def _r6b_publish_prepare_bundle(temporary_root: Path, target_root: Path) -> None:
    """Atomically publish an already verified private bundle to an absent target."""
    if target_root.exists() or target_root.is_symlink():
        raise ValueError("R6b Prepare final target appeared during publication")
    temporary_root.replace(target_root)


def _r6b_write_prepare_stage_report(root: Path, report: dict[str, Any]) -> None:
    """Write the sole completed receipt only after the bundle is published."""
    atomic_bytes(root / _R6B_PREPARE_STAGE, canonical_bytes(report) + b"\n")


def _r6b_prepare_target(args: argparse.Namespace) -> Path:
    target = _r6b_absolute(Path(args.scenario_root), label="target scenario root")
    parent = _r6b_existing_path(target.parent, label="target scenario parent", kind="directory")
    if parent != target.parent:
        raise ValueError("R6b Prepare target scenario root path is noncanonical")
    supplied_source = getattr(args, "source_scenario_root", None)
    if supplied_source is not None and (Path(supplied_source).exists() or Path(supplied_source).is_symlink()):
        source = _r6b_existing_path(
            Path(supplied_source), label="source scenario root", kind="directory"
        )
        if target == source or target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError("R6b Prepare target and retained source roots must be isolated")
    return target


def _execute_r6b_prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Publish or read-only verify exactly one isolated R6b Prepare bundle."""
    target = _r6b_prepare_target(args)
    if target.exists() or target.is_symlink():
        return _r6b_verify_published_prepare(target)
    if _r6b_free_bytes(target.parent) < 15 * 1024**3:
        raise ValueError("R6b Prepare requires at least 15 GiB free")
    temporary_parent = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.r6b-prepare-", dir=target.parent)
    )
    temporary_root = temporary_parent / "bundle"
    published = False
    try:
        bundle = build_r6b_prepare_bundle(args, temporary_root=temporary_root)
        if _r6b_absent_target(target) != target:
            raise ValueError("R6b Prepare final target changed before publication")
        _r6b_publish_prepare_bundle(temporary_root, target)
        published = True
        _r6b_write_prepare_stage_report(target, _r6b_prepare_stage_payload(bundle))
        return _r6b_verify_published_prepare(target)
    except BaseException:
        if published and (target.exists() or target.is_symlink()):
            shutil.rmtree(target)
        raise
    finally:
        if temporary_parent.exists() or temporary_parent.is_symlink():
            shutil.rmtree(temporary_parent)


def _r6b_design_root(args: argparse.Namespace) -> Path:
    """Require an existing Task32 root and an isolated retained R5 source root."""
    root = _r6b_existing_path(
        Path(args.scenario_root), label="R6b Design target scenario root", kind="directory"
    )
    source = _r6b_existing_path(
        Path(args.source_scenario_root),
        label="R6b Design retained adaptive source root",
        kind="directory",
    )
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("R6b Design target and retained source roots must be isolated")
    return root


def _r6b_design_inputs(args: argparse.Namespace, root: Path) -> Any:
    """Reconstruct exactly the frozen 12+14 R5-minus-Task32 authority."""
    from .cohort import select_cohort
    from .loader import load_normalized_sites
    from .realism import candidate_catalog
    from .scenario_r6b_design import (
        R6bDesignInputs,
        load_r6b_design_adaptive_authority,
    )

    source = _r6b_existing_path(
        Path(args.source_scenario_root),
        label="R6b Design retained adaptive source root",
        kind="directory",
    )
    upstream_adaptive_manifest = _r6b_existing_path(
        source / "00-manifest/r5-design-cohort.json",
        label="R6b Design adaptive cohort manifest",
        kind="file",
    )
    adaptive_authority_path = _r6b_existing_path(
        Path(__file__).with_name("r6b_design_adaptive_authority.json"),
        label="R6b Design adaptive authority",
        kind="file",
    )
    adaptive_authority = load_r6b_design_adaptive_authority(adaptive_authority_path)
    if (
        sha256_file(upstream_adaptive_manifest)
        != adaptive_authority.upstream_r5_adaptive_manifest_sha256
    ):
        raise ValueError("R6b Design upstream adaptive manifest pin drift")
    dkasc_manifest = _r6b_existing_path(
        Path(args.dkasc_manifest), label="R6b Design DKASC manifest", kind="file"
    )
    if sha256_file(dkasc_manifest) != _R6B_DKASC_MANIFEST_SHA256:
        raise ValueError("R6b Design DKASC manifest pin drift")
    contract = _r6b_authenticated_ok20_contract(args)
    contract_by_id = {row["site_id"]: row for row in contract["sites"]}
    if len(contract_by_id) != 20:
        raise ValueError("R6b Design Oklahoma adaptive source contract is invalid")
    authority_by_cohort = {
        name: tuple(row for row in adaptive_authority.blocks if row["cohort"] == name)
        for name in ("dkasc-yulara", "nrel-transfer")
    }
    dkasc_ids = {row["site_id"] for row in authority_by_cohort["dkasc-yulara"]}
    oklahoma_ids = {row["site_id"] for row in authority_by_cohort["nrel-transfer"]}
    if (
        len(dkasc_ids) != 12
        or len(oklahoma_ids) != 14
        or not oklahoma_ids.issubset(contract_by_id)
        or any(
            row["raw_hashes"] != [contract_by_id[row["site_id"]]["member_sha256"]]
            for row in authority_by_cohort["nrel-transfer"]
        )
    ):
        raise ValueError("R6b Design adaptive authority source identities are invalid")
    dkasc_sites = load_normalized_sites(
        dkasc_manifest, split="validation", site_ids=dkasc_ids
    )
    oklahoma_sites = load_normalized_sites(
        Path(args.nrel_manifest), split="validation", site_ids=oklahoma_ids
    )
    cohorts = {
        "dkasc-yulara": select_cohort(
            dkasc_sites, split="validation", max_blocks_per_site=1
        ),
        "nrel-transfer": select_cohort(
            oklahoma_sites, split="validation", max_blocks_per_site=1
        ),
    }
    return R6bDesignInputs(
        work_root=root,
        prepared_root=root,
        candidates=candidate_catalog("r6"),
        design_cohorts=cohorts,
        adaptive_design_authority=adaptive_authority_path,
    )


def _r6b_design_evaluator(candidates, cohorts):
    """Use the existing validation-only evaluator without R5/later-stage orchestration."""
    from .scenario_candidate import CandidateScreeningReport, screen_candidates
    from .scenario_revision import r6_design_precheck_reasons

    report = screen_candidates(tuple(candidates), dict(cohorts))
    classified = tuple(
        replace(
            row,
            passed=not (reasons := r6_design_precheck_reasons(row.metrics)),
            reasons=reasons,
        )
        for row in report.candidates
    )
    return CandidateScreeningReport(
        candidates=classified,
        cohort_hashes=report.cohort_hashes,
        protocol_cohort_sizes=report.protocol_cohort_sizes,
    )


def _r6b_design_prepare_path_set(root: Path) -> None:
    """Reject an extra, partial, or symlinked root before the one-shot evaluator runs."""
    receipt = _r6b_prepare_json(root, _R6B_PREPARE_STAGE.as_posix(), schema="r6b_cli_stage_v1")
    hashes = receipt.get("relative_sha256")
    if (
        not isinstance(hashes, dict)
        or len(hashes) != 33
        or any(
            type(relative) is not str
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not _r6b_sha256(digest)
            for relative, digest in hashes.items()
        )
    ):
        raise ValueError("R6b Design Task32 Prepare receipt hash map is invalid")
    expected_files = set(hashes) | {_R6B_PREPARE_STAGE.as_posix()}
    actual_files: set[str] = set()
    actual_dirs: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError("R6b Design Prepare root contains a symlink")
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            actual_files.add(relative)
        elif stat.S_ISDIR(mode):
            actual_dirs.add(relative)
        else:
            raise ValueError("R6b Design Prepare root contains a nonregular entry")
    expected_dirs = {
        parent.as_posix()
        for relative in expected_files
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    if actual_files != expected_files or actual_dirs != expected_dirs:
        raise ValueError("R6b Design Prepare root path set is invalid")


def _r6b_design_stage_payload(root: Path, report: Any) -> dict[str, Any]:
    """Build the non-self-referential receipt that Task1 verifies byte-for-byte."""
    _r6b_prepare_json(root, _R6B_PREPARE_STAGE.as_posix(), schema="r6b_cli_stage_v1")
    return {
        "schema": "r6b_cli_stage_v1",
        "revision": "r6b",
        "source_revision": "r5",
        "stage": "design",
        "status": "completed",
        "prepare_receipt_sha256": sha256_file(root / _R6B_PREPARE_STAGE),
        "design_screening_sha256": report.screening_sha256,
        "candidate_freeze_sha256": report.freeze_sha256,
        "design_execution_code_seal_sha256": sha256_file(root / _R6B_DESIGN_SEAL),
        "adaptive_design_authority_sha256": report.adaptive_design_authority_sha256,
        "counts": {
            "candidate_count": 6,
            "accepted_count": len(report.accepted),
            "rejected_count": len(report.rejected),
        },
        "terminal_state": report.status,
        "lifecycle": {
            "confirmation_executed": False,
            "design_executed": True,
            "prediction_generated": False,
        },
    }


def _r6b_design_terminal_present(root: Path) -> bool:
    return any(
        path.exists() or path.is_symlink()
        for path in (
            root / _R6B_DESIGN_BUNDLE,
            root / _R6B_DESIGN_SEAL,
            root / _R6B_DESIGN_AUTHORITY,
            root / _R6B_DESIGN_STAGE,
        )
    )


def _r6b_write_design_stage_receipt(root: Path, report: Any) -> None:
    """Atomically add the sole CLI receipt without leaving a retry-blocking temp file."""
    stage_path = root / _R6B_DESIGN_STAGE
    temporary = stage_path.with_suffix(stage_path.suffix + ".tmp")
    if stage_path.exists() or stage_path.is_symlink():
        raise ValueError("R6b Design stage receipt already exists")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("R6b Design stage receipt is partial")
    try:
        atomic_bytes(stage_path, canonical_bytes(_r6b_design_stage_payload(root, report)) + b"\n")
    except BaseException:
        if temporary.exists() and not temporary.is_symlink() and temporary.is_file():
            temporary.unlink()
        raise


def _r6b_design_relative_hashes(root: Path, report: Any) -> dict[str, str]:
    paths = [
        root / _R6B_DESIGN_SEAL,
        root / _R6B_DESIGN_BUNDLE / "freeze.json",
        root / _R6B_DESIGN_BUNDLE / "screening.json",
        root / _R6B_DESIGN_BUNDLE / "state.json",
        root / _R6B_DESIGN_STAGE,
    ]
    if report.status == "candidate_frozen":
        paths.append(root / _R6B_DESIGN_AUTHORITY)
    return {
        _r6b_existing_path(path, label="R6b Design output", kind="file")
        .relative_to(root)
        .as_posix(): sha256_file(path)
        for path in sorted(paths)
    }


def _execute_r6b_design(args: argparse.Namespace) -> dict[str, str]:
    """Run exactly one isolated R6b Design screen or verify its terminal receipt."""
    from .scenario_r6b_design import run_r6b_design_screen, verify_r6b_design_bundle

    root = _r6b_design_root(args)
    inputs = _r6b_design_inputs(args, root)
    if _r6b_design_terminal_present(root):
        report = verify_r6b_design_bundle(root, inputs)
    else:
        _r6b_design_prepare_path_set(root)
        report = run_r6b_design_screen(inputs, evaluator=_r6b_design_evaluator)
    stage_path = root / _R6B_DESIGN_STAGE
    if not stage_path.exists() and not stage_path.is_symlink():
        _r6b_write_design_stage_receipt(root, report)
    report = verify_r6b_design_bundle(root, inputs)
    return _r6b_design_relative_hashes(root, report)


def _r6_absolute(path: Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if ".." in raw.parts:
        raise ValueError(f"R6 {label} path is noncanonical")
    absolute = raw if raw.is_absolute() else _PROJECT_ROOT / raw
    _reject_r5_cli_symlinks(absolute)
    return absolute


def _r6_existing_path(path: Path, *, label: str, kind: str) -> Path:
    absolute = _r6_absolute(path, label=label)
    try:
        resolved = absolute.resolve(strict=True)
        mode = absolute.lstat().st_mode
    except OSError as error:
        raise ValueError(f"R6 {label} is missing or unreadable") from error
    if absolute != resolved:
        raise ValueError(f"R6 {label} path is noncanonical")
    if kind == "file" and not stat.S_ISREG(mode):
        raise ValueError(f"R6 {label} must be a regular file")
    if kind == "directory" and not stat.S_ISDIR(mode):
        raise ValueError(f"R6 {label} must be a directory")
    return resolved


def _r6_absent_target(path: Path) -> Path:
    absolute = _r6_absolute(path, label="target scenario root")
    if absolute.exists() or absolute.is_symlink():
        raise ValueError("R6 target scenario root must be absent")
    parent = _r6_existing_path(
        absolute.parent, label="target scenario parent", kind="directory"
    )
    if absolute.parent != parent:
        raise ValueError("R6 target scenario root path is noncanonical")
    return absolute


def _r6_free_bytes(path: Path) -> int:
    free = shutil.disk_usage(path).free
    if type(free) is not int or free < 0:
        raise ValueError("R6 free-space result is invalid")
    return free


def _r6_pinned_bytes(
    path: Path, *, label: str, expected_sha256: str, expected_size: int
) -> bytes:
    secure = _r6_existing_path(path, label=label, kind="file")
    try:
        content = secure.read_bytes()
    except OSError as error:
        raise ValueError(f"R6 {label} is unreadable") from error
    if len(content) != expected_size or hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"R6 {label} hash or size changed")
    if _r6_existing_path(secure, label=label, kind="file") != secure:
        raise ValueError(f"R6 {label} changed during authentication")
    return content


def _r6_canonical_json(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"R6 {label} is malformed") from error
    if not isinstance(payload, dict) or content != canonical_bytes(payload) + b"\n":
        raise ValueError(f"R6 {label} is not canonical")
    return payload


def _r6_authenticate_oklahoma_sources(
    source_root: Path,
    extracted_manifest: Path,
    expected_ok_ids: tuple[str, ...],
) -> dict[str, str]:
    manifest_content = _r6_pinned_bytes(
        extracted_manifest,
        label="R5 extracted-source manifest",
        expected_sha256=_R6_R5_EXTRACTED_MANIFEST_SHA256,
        expected_size=_R6_R5_EXTRACTED_MANIFEST_SIZE,
    )
    log_content = _r6_pinned_bytes(
        source_root / "09-logs/r5-source-extraction.json",
        label="R5 source extraction log",
        expected_sha256=_R6_R5_EXTRACTION_LOG_SHA256,
        expected_size=_R6_R5_EXTRACTION_LOG_SIZE,
    )
    manifest = _r6_canonical_json(manifest_content, label="R5 extracted-source manifest")
    log = _r6_canonical_json(log_content, label="R5 source extraction log")
    entries = manifest.get("entries")
    rows = log.get("sources")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("root") != "."
        or not isinstance(entries, list)
        or len(entries) != 5
        or log.get("schema") != "r5_extracted_source_log_v1"
        or not isinstance(rows, list)
        or len(rows) != 5
    ):
        raise ValueError("R6 retained Oklahoma manifest/log contract changed")
    entries_by_id = {
        entry.get("site_id"): entry for entry in entries if isinstance(entry, dict)
    }
    rows_by_id = {row.get("site_id"): row for row in rows if isinstance(row, dict)}
    if (
        len(entries_by_id) != 5
        or len(rows_by_id) != 5
        or tuple(sorted(entries_by_id)) != expected_ok_ids
        or tuple(sorted(rows_by_id)) != expected_ok_ids
    ):
        raise ValueError("R6 retained Oklahoma source identities changed")
    extraction_root = _r6_existing_path(
        source_root / "01-cohorts/r5-confirmation-sources",
        label="R5 extracted-source root",
        kind="directory",
    )
    output_hashes: dict[str, str] = {}
    for site_id in expected_ok_ids:
        entry = entries_by_id[site_id]
        row = rows_by_id[site_id]
        if (
            entry.get("dataset") != "nrel"
            or entry.get("spatial_role") != "validation_calibration"
            or entry.get("state") != "OK"
            or set(row) != {
                "archive_sha256",
                "member",
                "member_sha256",
                "output_sha256",
                "site_id",
            }
        ):
            raise ValueError("R6 retained Oklahoma source role or log schema changed")
        member = row.get("member")
        output_sha256 = row.get("output_sha256")
        member_sha256 = row.get("member_sha256")
        member_path = Path(member) if isinstance(member, str) else Path()
        if (
            not isinstance(member, str)
            or not member
            or member_path.is_absolute()
            or ".." in member_path.parts
            or member_path.as_posix() != member
            or entry.get("files") != [member]
            or not isinstance(output_sha256, str)
            or len(output_sha256) != 64
            or any(character not in "0123456789abcdef" for character in output_sha256)
            or member_sha256 != output_sha256
        ):
            raise ValueError("R6 retained Oklahoma source member or hash changed")
        output = _r6_existing_path(
            extraction_root / member_path,
            label=f"R5 extracted Oklahoma source {site_id}",
            kind="file",
        )
        if not output.is_relative_to(extraction_root):
            raise ValueError("R6 retained Oklahoma source escapes its extraction root")
        if sha256_file(output) != output_sha256:
            raise ValueError("R6 retained Oklahoma source output hash changed")
        output_hashes[site_id] = output_sha256
    return output_hashes


def _r6_validate_sources(
    dkasc_sites,
    oklahoma_sites,
    expected_ok_ids,
    *,
    retained_oklahoma_hashes: dict[str, str],
) -> None:
    dkasc_ids = tuple(sorted(site.site_id for site in dkasc_sites))
    ok_ids = tuple(sorted(site.site_id for site in oklahoma_sites))
    if len(dkasc_sites) != 12 or len(set(dkasc_ids)) != 12:
        raise ValueError("R6 preflight requires exactly twelve DKASC sites")
    if len(oklahoma_sites) != 5 or ok_ids != expected_ok_ids:
        raise ValueError("R6 preflight requires the exact five Oklahoma sites")
    if any(
        site.dataset != "dkasc"
        or site.split != "validation"
        or site.spatial_role != "source_train"
        or site.spatial_role in {"test", "zero_parameter_test"}
        for site in dkasc_sites
    ):
        raise ValueError("R6 DKASC source role/split is invalid or held-out")
    if any(
        site.dataset != "nrel"
        or site.split != "validation"
        or site.spatial_role != "validation_calibration"
        or site.state != "OK"
        or site.spatial_role in {"test", "zero_parameter_test"}
        for site in oklahoma_sites
    ):
        raise ValueError("R6 Oklahoma source role/split is invalid or held-out")
    if any(
        site.raw_hashes != (retained_oklahoma_hashes.get(site.site_id),)
        for site in oklahoma_sites
    ):
        raise ValueError("R6 Oklahoma loaded raw hash differs from retained output")


def _r6_validate_prior(prior, dkasc_sites, expected_ok_ids, *, pd) -> None:
    if (
        len(prior) != 75
        or any(
            not isinstance(row, tuple)
            or len(row) != 3
            or any(not isinstance(value, str) for value in row)
            for row in prior
        )
    ):
        raise ValueError("R6 prior coordinate contract is malformed")
    if len(set(prior)) != 75:
        raise ValueError("R6 prior coordinate contract is malformed")
    for _site_id, start, end in prior:
        try:
            parsed_start, parsed_end = pd.Timestamp(start), pd.Timestamp(end)
        except (TypeError, ValueError) as error:
            raise ValueError("R6 prior coordinate interval is malformed") from error
        if (
            parsed_start.tz is None
            or parsed_end.tz is None
            or parsed_end < parsed_start
            or parsed_start.isoformat() != start
            or parsed_end.isoformat() != end
        ):
            raise ValueError("R6 prior coordinate interval is malformed")
    counts: dict[str, int] = {}
    for site_id, _start, _end in prior:
        counts[site_id] = counts.get(site_id, 0) + 1
    dkasc_ids = {site.site_id for site in dkasc_sites}
    if set(counts) != dkasc_ids | set(expected_ok_ids):
        raise ValueError("R6 prior coordinate identities changed")
    if any(counts[site_id] != 5 for site_id in dkasc_ids) or any(
        counts[site_id] != 3 for site_id in expected_ok_ids
    ):
        raise ValueError("R6 prior coordinate distribution changed")


def _r6_coordinate_sha256(coordinates) -> str:
    payload = [list(row) for row in sorted(coordinates)]
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _r6_interval_gap(start, end, other_start, other_end, *, pd):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    other_start, other_end = pd.Timestamp(other_start), pd.Timestamp(other_end)
    if end < other_start:
        return other_start - end
    if other_end < start:
        return start - other_end
    return pd.Timedelta(0)


def _r6_validate_selection(
    selection, *, prior, dkasc_sites, oklahoma_sites, plan_sha256, pd
) -> dict[str, int]:
    blocks = tuple(selection.blocks)
    source_by_id = {site.site_id: site for site in (*dkasc_sites, *oklahoma_sites)}
    coordinates = tuple(
        (block.site_id, block.start.isoformat(), block.end.isoformat()) for block in blocks
    )
    dkasc_blocks = tuple(block for block in blocks if block.dataset == "dkasc")
    oklahoma_blocks = tuple(block for block in blocks if block.dataset == "nrel")
    if (
        len(blocks) != 27
        or len(set(coordinates)) != 27
        or len({block.block_id for block in blocks}) != 27
        or len({block.site_id for block in blocks}) != 17
        or len(dkasc_blocks) != 12
        or len(oklahoma_blocks) != 15
    ):
        raise ValueError("R6 selected block cardinality is invalid")
    if any(
        block.site_id not in source_by_id
        or block.split != "validation"
        or block.spatial_role != source_by_id[block.site_id].spatial_role
        or block.dataset != source_by_id[block.site_id].dataset
        or block.state != source_by_id[block.site_id].state
        or block.spatial_role in {"test", "zero_parameter_test"}
        for block in blocks
    ):
        raise ValueError("R6 selected block source role/split is invalid or held-out")
    prior_set = set(prior)
    exact_overlap = sum(coordinate in prior_set for coordinate in coordinates)
    prior_embargo = sum(
        block.site_id == site_id
        and _r6_interval_gap(
            block.start, block.end, start, end, pd=pd
        )
        < pd.Timedelta(days=14)
        for block in blocks
        for site_id, start, end in prior
    )
    oklahoma_pairwise = 0
    for site in oklahoma_sites:
        site_blocks = tuple(block for block in oklahoma_blocks if block.site_id == site.site_id)
        if len(site_blocks) != 3:
            raise ValueError("R6 Oklahoma selected block distribution is invalid")
        oklahoma_pairwise += sum(
            _r6_interval_gap(
                left.start, left.end, right.start, right.end, pd=pd
            )
            < pd.Timedelta(days=14)
            for index, left in enumerate(site_blocks)
            for right in site_blocks[index + 1 :]
        )
    if exact_overlap or prior_embargo or oklahoma_pairwise:
        raise ValueError("R6 selection violates overlap or embargo constraints")
    if (
        selection.plan_sha256 != plan_sha256
        or selection.selection_seed != 20260818
        or selection.embargo_days != 14
        or selection.prior_coordinate_sha256 != _r6_coordinate_sha256(prior)
        or selection.selected_coordinate_sha256 != _r6_coordinate_sha256(coordinates)
    ):
        raise ValueError("R6 selection hash or plan binding is invalid")
    return {
        "exact_overlap": exact_overlap,
        "prior_embargo": prior_embargo,
        "oklahoma_pairwise": oklahoma_pairwise,
    }


def _reject_r5_cli_symlinks(path: Path) -> None:
    cursor = path
    while True:
        if cursor.is_symlink():
            raise ValueError("r5 CLI root/stage path must not contain symlinks")
        if cursor == cursor.parent:
            return
        cursor = cursor.parent


def _r5_resolve_auxiliary_ok_archive(
    nrel_manifest: Path, nrel_payload: dict[str, Any]
) -> Path:
    """Resolve the audited Oklahoma archive without accepting path indirection."""
    try:
        archive = (
            nrel_manifest.parent
            / nrel_payload["root"]
            / "archives"
            / "ok-pv-2006.zip"
        )
    except (KeyError, TypeError) as error:
        raise ValueError("r5 NREL manifest archive root is malformed") from error
    _reject_r5_cli_symlinks(archive)
    try:
        resolved = archive.resolve(strict=True)
    except OSError as error:
        raise ValueError("auxiliary Oklahoma archive is unavailable") from error
    if not resolved.is_file():
        raise ValueError("auxiliary Oklahoma archive must be a regular file")
    return resolved


def _read_r5_stage_report(
    path: Path, *, root: Path, stage: str
) -> dict[str, Any]:
    _reject_r5_cli_symlinks(path)
    if not path.is_file():
        raise ValueError(f"r5 {stage} stage report is missing")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"r5 {stage} stage report is malformed") from error
    artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or set(payload) != {"artifacts", "schema", "stage", "status"}
        or payload.get("schema") != "r5_cli_stage_report_v1"
        or payload.get("stage") != stage
        or payload.get("status") != "completed"
        or not isinstance(artifacts, dict)
        or not artifacts
        or not all(
            type(relative) is str
            and type(digest) is str
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts
            for relative, digest in artifacts.items()
        )
        or raw != canonical_bytes(payload) + b"\n"
    ):
        raise ValueError(f"r5 {stage} stage report is not canonical")
    for relative, digest in artifacts.items():
        artifact = root / relative
        _reject_r5_cli_symlinks(artifact)
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise ValueError(f"r5 {stage} stage artifact contract changed: {relative}")
    return payload


def _run_r5_cli_stage(
    root_value: Path,
    stage: str,
    *,
    operation: Callable[[], dict[str, str]],
) -> dict[str, Any]:
    if stage not in _R5_STAGES:
        raise ValueError("r5 CLI stage is invalid")
    supplied = root_value.expanduser()
    _reject_r5_cli_symlinks(supplied.absolute())
    root = supplied.resolve()
    report_root = root / _R5_STAGE_ROOT
    _reject_r5_cli_symlinks(report_root)
    reports = {name: report_root / f"{name}.json" for name in _R5_STAGES}
    for name, path in reports.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        if temporary.exists() or temporary.is_symlink():
            raise ValueError(f"r5 {name} stage has partial state")
    position = _R5_STAGES.index(stage)
    for predecessor in _R5_STAGES[:position]:
        _read_r5_stage_report(reports[predecessor], root=root, stage=predecessor)
    for forbidden in _R5_STAGES[position:]:
        path = reports[forbidden]
        if path.exists() or path.is_symlink():
            raise ValueError(f"r5 {stage} stage is out of order or already completed")
    artifacts = operation()
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"r5 {stage} stage produced no contracted artifacts")
    for relative, digest in artifacts.items():
        if (
            type(relative) is not str
            or type(digest) is not str
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError(f"r5 {stage} stage returned an invalid artifact contract")
        artifact = root / relative
        _reject_r5_cli_symlinks(artifact)
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise ValueError(f"r5 {stage} stage artifact was not produced: {relative}")
    report = {
        "artifacts": dict(sorted(artifacts.items())),
        "schema": "r5_cli_stage_report_v1",
        "stage": stage,
        "status": "completed",
    }
    write_json(reports[stage], report)
    return report


def _r5_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"r5 {label} is missing or malformed") from error
    if not isinstance(payload, dict) or raw != canonical_bytes(payload) + b"\n":
        raise ValueError(f"r5 {label} is not canonical")
    return payload


def _r5_input_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"r5 {label} is missing or malformed") from error
    if not isinstance(payload, dict):
        raise ValueError(f"r5 {label} must contain a JSON object")
    return payload


def _r5_load_pinned_dkasc_history(root: Path, dkasc_sites: tuple[Any, ...]) -> tuple[Any, ...]:
    """Load the complete 4A-selected DKASC validation history from its pinned artifact."""
    from .cohort import BLOCK_STEPS, _valid_blocks

    resolved_root = root.resolve(strict=True)

    def regular_contained(path: Path, *, label: str) -> Path:
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"r5 {label} is missing") from error
        try:
            resolved.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"r5 {label} escapes the scenario root") from error
        relative = path.absolute().relative_to(resolved_root)
        cursor = resolved_root
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ValueError(f"r5 {label} must not contain symlinks")
        try:
            mode = path.stat().st_mode
        except OSError as error:
            raise ValueError(f"r5 {label} is unavailable") from error
        if not stat.S_ISREG(mode):
            raise ValueError(f"r5 {label} must be a regular file")
        return resolved

    index_path = regular_contained(
        resolved_root / "09-logs/preflight-4a-artifacts.json", label="4A artifact index"
    )
    selected_path = regular_contained(
        resolved_root / "01-cohorts/selected-blocks.csv", label="selected-block artifact"
    )
    try:
        index = json.loads(index_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("r5 4A artifact index is malformed") from error
    entries = index.get("artifacts") if isinstance(index, dict) else None
    if not isinstance(entries, list):
        raise ValueError("r5 4A artifact index has no artifact list")
    selected_entries = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("path") == "01-cohorts/selected-blocks.csv"
    ]
    if len(selected_entries) != 1:
        raise ValueError("r5 4A selected-block artifact index entry is invalid")
    entry = selected_entries[0]
    if (
        set(entry) != {"path", "sha256", "size_bytes"}
        or type(entry["sha256"]) is not str
        or len(entry["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in entry["sha256"])
        or type(entry["size_bytes"]) is not int
        or entry["size_bytes"] < 0
    ):
        raise ValueError("r5 4A selected-block artifact index schema is invalid")
    raw = selected_path.read_bytes()
    if (
        len(raw) != _R5_SELECTED_BLOCKS_SIZE_BYTES
        or hashlib.sha256(raw).hexdigest() != _R5_SELECTED_BLOCKS_SHA256
    ):
        raise ValueError("r5 selected-block artifact differs from the authoritative 4A pin")
    if len(raw) != entry["size_bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("r5 4A selected-block artifact hash or size changed")
    try:
        reader = csv.DictReader(raw.decode("utf-8").splitlines())
    except UnicodeDecodeError as error:
        raise ValueError("r5 selected-block artifact is not UTF-8") from error
    columns = (
        "block_id", "site_id", "group", "protocol", "split", "start", "end",
        "quarter", "n_steps", "selection_hash",
    )
    if tuple(reader.fieldnames or ()) != columns:
        raise ValueError("r5 selected-block artifact schema changed")
    source_blocks = {
        (
            block.block_id,
            block.site_id,
            block.start.isoformat(),
            block.end.isoformat(),
            block.quarter,
            str(block.n_steps),
            block.selection_hash,
        ): block
        for site in dkasc_sites
        for _start, block in _valid_blocks(site)
    }
    expected_site_ids = tuple(sorted(site.site_id for site in dkasc_sites))
    if len(expected_site_ids) != 12 or len(set(expected_site_ids)) != 12:
        raise ValueError("r5 selected-block source series must contain twelve DKASC sites")
    matched = []
    for row in reader:
        if set(row) != set(columns) or any(type(value) is not str for value in row.values()):
            raise ValueError("r5 selected-block row schema changed")
        if (row["group"], row["protocol"], row["split"]) != (
            "DKASC", "dkasc-yulara", "validation"
        ):
            continue
        key = (
            row["block_id"], row["site_id"], row["start"], row["end"], row["quarter"],
            row["n_steps"], row["selection_hash"],
        )
        block = source_blocks.get(key)
        if block is None or block.n_steps != BLOCK_STEPS:
            raise ValueError("r5 selected-block row does not match a loaded source block")
        matched.append(block)
    if len(matched) != 48:
        raise ValueError("r5 selected-block history requires exactly forty-eight rows")
    coordinates = {(block.site_id, block.start, block.end) for block in matched}
    if len(coordinates) != 48 or len({block.block_id for block in matched}) != 48:
        raise ValueError("r5 selected-block history rows must be unique")
    if tuple(sorted(block.site_id for block in matched)) != tuple(
        site_id for site_id in expected_site_ids for _ in range(4)
    ):
        raise ValueError("r5 selected-block history requires four rows per DKASC site")
    return tuple(sorted(matched, key=lambda block: (block.site_id, block.start, block.block_id)))


def _r5_relative_hashes(root: Path, paths: tuple[Path, ...]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or not path.is_file():
            raise ValueError("r5 stage artifact is outside its scenario root or missing")
        hashes[resolved.relative_to(root).as_posix()] = sha256_file(path)
    return hashes


def _r5_source_rows(sources: tuple[Any, ...]) -> list[dict[str, str]]:
    return [
        {
            "archive_sha256": source.archive_sha256,
            "member": source.member,
            "member_sha256": source.member_sha256,
            "output_sha256": source.output_sha256,
            "site_id": source.site_id,
        }
        for source in sorted(sources, key=lambda item: item.site_id)
    ]


def _r5_load_extracted_sources(root: Path) -> tuple[Any, ...]:
    from .nested_validation import EXPECTED_AUXILIARY_OK_SITE_IDS, ExtractedSource

    extraction = _r5_json(
        root / "09-logs/r5-source-extraction.json", label="source extraction log"
    )
    rows = extraction.get("sources")
    if (
        extraction.get("schema") != "r5_extracted_source_log_v1"
        or not isinstance(rows, list)
        or tuple(row.get("site_id") for row in rows if isinstance(row, dict))
        != EXPECTED_AUXILIARY_OK_SITE_IDS
    ):
        raise ValueError("r5 source extraction contract changed")
    extraction_root = root / "01-cohorts/r5-confirmation-sources"
    sources = []
    for row in rows:
        if set(row) != {
            "archive_sha256",
            "member",
            "member_sha256",
            "output_sha256",
            "site_id",
        }:
            raise ValueError("r5 source extraction row schema changed")
        output = extraction_root / row["member"]
        _reject_r5_cli_symlinks(output)
        if not output.is_file() or sha256_file(output) != row["output_sha256"]:
            raise ValueError("r5 extracted confirmation source changed")
        sources.append(ExtractedSource(output_path=str(output), **row))
    return tuple(sources)


def _r5_prepare_context(args: argparse.Namespace, *, publish: bool) -> _R5PreparedContext:
    """Build the Task 1--3 contracts; publish them only for the prepare stage."""

    from .cohort import _valid_blocks, select_cohort
    from .loader import load_normalized_sites
    from .nested_validation import (
        ConfirmationContract,
        ConfirmationInputs,
        ConfirmationSourceBinding,
        _old_block_rows,
        _sha256_json,
        auxiliary_ok_entries,
        build_confirmation_cohorts,
        confirmation_block_csv_bytes,
        confirmation_protocol_payload,
        extract_auxiliary_ok_sources,
    )
    from .realism import candidate_catalog
    from .scenario_confirmation import (
        R5ArtifactPaths,
        R5CrossBindingContract,
        r5_design_cohort_binding_payload,
    )
    from .scenario_revision import r5_catalog_sha256, r5_precheck_ranges_sha256

    root = args.scenario_root.expanduser().resolve()
    if publish:
        root.mkdir(parents=True, exist_ok=True)
    elif not root.is_dir():
        raise ValueError("r5 prepared scenario root is missing")
    _reject_r5_cli_symlinks(root)
    if not publish:
        _r5_assert_execution_code_seal(root)
    dkasc_manifest = args.dkasc_manifest.expanduser().resolve()
    nrel_manifest = args.nrel_manifest.expanduser().resolve()
    audit_path = args.nrel_selection_audit.expanduser().resolve()
    nrel_payload = _r5_input_json(nrel_manifest, label="NREL manifest")
    audit = _r5_input_json(audit_path, label="NREL selection audit")

    dkasc_sites = load_normalized_sites(dkasc_manifest, split="validation")
    selected_ok = set(audit["selected_site_ids_by_state"]["OK"])
    design_ok_sites = load_normalized_sites(
        nrel_manifest, split="validation", site_ids=selected_ok
    )
    design_cohorts = {
        "dkasc-yulara": select_cohort(
            dkasc_sites, split="validation", max_blocks_per_site=1
        ),
        "nrel-transfer": select_cohort(
            design_ok_sites, split="validation", max_blocks_per_site=1
        ),
    }

    extraction_root = root / "01-cohorts/r5-confirmation-sources"
    extracted_manifest = extraction_root / "manifest.json"
    extraction_log = root / "09-logs/r5-source-extraction.json"
    if publish:
        entries = auxiliary_ok_entries(nrel_payload, audit)
        archive = _r5_resolve_auxiliary_ok_archive(nrel_manifest, nrel_payload)
        extracted = extract_auxiliary_ok_sources(entries, archive, extraction_root)
        manifest_payload = {
            **nrel_payload,
            "root": ".",
            "entries": [
                {key: value for key, value in row.items() if key != "archive_sha256"}
                for row in entries
            ],
        }
        write_json(extracted_manifest, manifest_payload)
        write_json(
            extraction_log,
            {"schema": "r5_extracted_source_log_v1", "sources": _r5_source_rows(extracted)},
        )
    else:
        extracted = _r5_load_extracted_sources(root)
        _r5_json(extracted_manifest, label="extracted source manifest")

    auxiliary_sites = load_normalized_sites(
        extracted_manifest,
        split="validation",
        site_ids={source.site_id for source in extracted},
    )
    old_blocks = _r5_load_pinned_dkasc_history(root, tuple(dkasc_sites))
    prior_coordinates = frozenset(
        {
            *(
                (block.site_id, block.start.isoformat(), block.end.isoformat())
                for block in old_blocks
            ),
            *(
                (block.site_id, block.start.isoformat(), block.end.isoformat())
                for cohort in design_cohorts.values()
                for block in cohort.blocks
            ),
            *(
                (block.site_id, block.start.isoformat(), block.end.isoformat())
                for site in dkasc_sites
                for _index, block in _valid_blocks(site)[:2]
            ),
        }
    )
    old_sha = _sha256_json(_old_block_rows(old_blocks))
    extraction_sha = _sha256_json(_r5_source_rows(extracted))
    bindings = tuple(
        ConfirmationSourceBinding(
            site_id=site.site_id,
            raw_hashes=site.raw_hashes,
            source_origin_timestamp=site.timestamps[0].isoformat(),
            block_start_indices=tuple(index for index, _block in _valid_blocks(site)),
        )
        for site in sorted((*dkasc_sites, *auxiliary_sites), key=lambda item: item.site_id)
    )
    confirmation_contract = ConfirmationContract(
        dkasc_site_ids=tuple(sorted(site.site_id for site in dkasc_sites)),
        old_cohort_sha256=old_sha,
        extraction_sha256=extraction_sha,
        prior_screen_coordinates_sha256=_sha256_json(
            [list(item) for item in sorted(prior_coordinates)]
        ),
        source_series=bindings,
    )
    confirmation_cohorts = build_confirmation_cohorts(
        ConfirmationInputs(
            dkasc_sites=tuple(dkasc_sites),
            oklahoma_sites=tuple(auxiliary_sites),
            old_selected_blocks=old_blocks,
            old_cohort_sha256=old_sha,
            extraction_sources=tuple(extracted),
            extraction_sha256=extraction_sha,
            prior_screen_coordinates=prior_coordinates,
            confirmation_contract=confirmation_contract,
        )
    )

    artifact_paths = R5ArtifactPaths(
        code=root / _R5_EXECUTION_CODE_SEAL,
        design_cohort=root / "00-manifest/r5-design-cohort.json",
        confirmation_protocol=root / "00-manifest/r5-confirmation-protocol.json",
        confirmation_block_csv=root / "01-cohorts/r5-confirmation-blocks.csv",
        extracted_source_log=extraction_log,
        prior_r1_r4_seal_chain=root / "00-manifest/r5-prior-r1-r4-seal-chain.json",
        predecessor_state=root / "00-manifest/r5-predecessor-state.json",
    )
    design_payload = r5_design_cohort_binding_payload(design_cohorts)
    protocol_payload = confirmation_protocol_payload(
        confirmation_cohorts, confirmation_contract=confirmation_contract
    )
    block_bytes = confirmation_block_csv_bytes(
        confirmation_cohorts, confirmation_contract=confirmation_contract
    )
    if publish:
        write_json(artifact_paths.code, _r5_execution_code_seal_payload())
        write_json(artifact_paths.design_cohort, design_payload)
        write_json(artifact_paths.confirmation_protocol, protocol_payload)
        atomic_bytes(artifact_paths.confirmation_block_csv, block_bytes)
        write_json(
            artifact_paths.prior_r1_r4_seal_chain,
            {"revisions": ["r1", "r2", "r3", "r4"], "schema": "r5_prior_seal_chain_v1"},
        )
        predecessor = _r5_json(root / "09-logs/revision-state.json", label="r4 state")
        if predecessor.get("catalog_revision") != "r4" or predecessor.get("state") != "validation_screened":
            raise ValueError("r5 predecessor is not the terminal r4 validation state")
        write_json(
            artifact_paths.predecessor_state,
            {
                "revision": "r4",
                "source_sha256": sha256_file(root / "09-logs/revision-state.json"),
                "state": "validation_screened",
            },
        )
    else:
        expected = {
            artifact_paths.design_cohort: canonical_bytes(design_payload) + b"\n",
            artifact_paths.confirmation_protocol: canonical_bytes(protocol_payload) + b"\n",
            artifact_paths.confirmation_block_csv: block_bytes,
        }
        if any(not path.is_file() or path.read_bytes() != content for path, content in expected.items()):
            raise ValueError("r5 prepared cohort contract changed")

    cross = R5CrossBindingContract(
        code_sha256=sha256_file(artifact_paths.code),
        r5_catalog_sha256=r5_catalog_sha256(),
        range_definition_sha256=r5_precheck_ranges_sha256(),
        design_cohort_sha256=sha256_file(artifact_paths.design_cohort),
        confirmation_protocol_sha256=sha256_file(artifact_paths.confirmation_protocol),
        confirmation_block_csv_sha256=sha256_file(artifact_paths.confirmation_block_csv),
        extracted_source_log_sha256=sha256_file(artifact_paths.extracted_source_log),
        prior_r1_r4_seal_chain_sha256=sha256_file(artifact_paths.prior_r1_r4_seal_chain),
        predecessor_state_sha256=sha256_file(artifact_paths.predecessor_state),
    )
    if tuple(candidate_catalog("r5")) == ():
        raise ValueError("r5 catalog is empty")
    return _R5PreparedContext(
        root, design_cohorts, confirmation_cohorts, confirmation_contract, artifact_paths, cross
    )


def _execute_r5_prepare(args: argparse.Namespace) -> dict[str, str]:
    context = _r5_prepare_context(args, publish=True)
    extracted = tuple(
        context.root / "01-cohorts/r5-confirmation-sources" / source.member
        for source in _r5_load_extracted_sources(context.root)
    )
    return _r5_relative_hashes(
        context.root,
        (*tuple(context.artifacts.__dict__.values())[1:], context.artifacts.code,
         context.root / "01-cohorts/r5-confirmation-sources/manifest.json", *extracted),
    )


def _execute_r5_design(args: argparse.Namespace) -> dict[str, str]:
    from .realism import candidate_catalog
    from .scenario_confirmation import (
        R5DesignInputs,
        r5_confirmation_authority,
        run_r5_design_screen,
        write_r5_confirmation_authority,
    )

    context = _r5_prepare_context(args, publish=False)
    bundle = context.root / "00-manifest/r5-design-freeze"
    inputs = R5DesignInputs(
        work_root=context.root,
        candidates=candidate_catalog("r5"),
        design_cohorts=context.design_cohorts,
        confirmation_cohorts=context.confirmation_cohorts,
        confirmation_contract=context.confirmation_contract,
        artifacts=context.artifacts,
        cross_binding_contract=context.cross_binding_contract,
    )
    report = (
        _r5_load_design_report(context)
        if bundle.exists()
        else run_r5_design_screen(inputs)
    )
    paths = tuple(bundle / name for name in ("freeze.json", "screening.json", "state.json"))
    if report.status == "candidate_frozen":
        authority_path = context.root / "00-manifest/r5-confirmation-authority.json"
        expected_authority = r5_confirmation_authority(report)
        if authority_path.exists() or authority_path.is_symlink():
            from .scenario_confirmation import load_r5_confirmation_authority

            if (
                load_r5_confirmation_authority(authority_path, root=context.root)
                != expected_authority
            ):
                raise ValueError("r5 persisted confirmation authority changed")
        else:
            write_r5_confirmation_authority(
                expected_authority, authority_path, root=context.root
            )
        paths += (authority_path,)
    return _r5_relative_hashes(context.root, paths)


def _r5_load_design_report(context: _R5PreparedContext) -> Any:
    from .scenario_confirmation import DesignFreezeReport, FrozenCandidateEvaluation

    bundle = context.root / "00-manifest/r5-design-freeze"
    screening_path = bundle / "screening.json"
    freeze_path = bundle / "freeze.json"
    state_path = bundle / "state.json"
    screening = _r5_json(screening_path, label="design screening")
    freeze = _r5_json(freeze_path, label="candidate freeze")
    _r5_json(state_path, label="design state")
    if freeze.get("design_screening_sha256") != sha256_file(screening_path):
        raise ValueError("r5 design screening/freeze binding changed")
    rows = screening.get("candidates")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("r5 persisted design cardinality changed")

    def evaluation(row: dict[str, Any]) -> FrozenCandidateEvaluation:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("r5 persisted design metrics are malformed")
        return FrozenCandidateEvaluation(
            candidate_id=row["candidate_id"],
            config_sha256=row["config_sha256"],
            model_profile_sha256=row["model_profile_sha256"],
            scenario_seed=row["scenario_seed"],
            passed=row["status"] == "accepted",
            metric_values=tuple(sorted((key, float(value)) for key, value in metrics.items())),
            reasons=tuple(row["reasons"]),
            distance_to_range_midpoints=float(row["distance_to_range_midpoints"]),
        )

    evaluated = tuple(evaluation(row) for row in rows if isinstance(row, dict))
    if len(evaluated) != 12:
        raise ValueError("r5 persisted design rows are malformed")
    return DesignFreezeReport(
        status=freeze["status"],
        evaluated_count=screening["evaluated_count"],
        accepted=tuple(row for row in evaluated if row.passed),
        rejected=tuple(row for row in evaluated if not row.passed),
        frozen_candidate_id=freeze["candidate_id"],
        frozen_config_sha256=freeze["config_sha256"],
        frozen_model_profile_sha256=freeze["model_profile_sha256"],
        frozen_scenario_seed=freeze["scenario_seed"],
        selection_order=freeze["selection_order"],
        screening_sha256=sha256_file(screening_path),
        freeze_sha256=sha256_file(freeze_path),
        work_root=context.root,
        artifacts=context.artifacts,
        cross_binding_contract=context.cross_binding_contract,
        confirmation_contract=context.confirmation_contract,
    )


def _execute_r5_confirm(args: argparse.Namespace) -> dict[str, str]:
    """Load the design contract and invoke Task 4's one-shot confirmation API."""

    from .scenario_candidate import evaluate_validation_candidate
    from .scenario_confirmation import (
        load_r5_confirmation_authority,
        run_r5_confirmation,
    )
    from .scenario_revision_verify import validate_r5_terminal_integrity

    context = _r5_prepare_context(args, publish=False)
    freeze = _r5_load_design_report(context)
    if freeze.status != "candidate_frozen":
        raise ValueError("r5 design did not freeze a confirmation candidate")
    authority_path = context.root / "00-manifest/r5-confirmation-authority.json"
    authority = load_r5_confirmation_authority(authority_path, root=context.root)
    bundle = context.root / "09-logs/r5-confirmation"
    paths = tuple(bundle / name for name in ("decision.json", "state.json"))
    if not bundle.exists():
        run_r5_confirmation(
            freeze,
            context.confirmation_cohorts,
            authority=authority,
            evaluator=lambda candidate, cohorts: evaluate_validation_candidate(
                candidate,
                {"dkasc-yulara": cohorts.dkasc, "nrel-transfer": cohorts.oklahoma},
            ).metrics,
        )
    integrity = validate_r5_terminal_integrity(
        context.root,
        artifacts=context.artifacts,
        cross_binding_contract=context.cross_binding_contract,
        confirmation_contract=context.confirmation_contract,
        authority=authority,
    )
    if integrity.terminal_status not in {"formal_materializing", "failed_confirmation"}:
        raise ValueError("r5 confirmation terminal identity is invalid")
    return {
        paths[0].relative_to(context.root).as_posix(): integrity.hashes[
            "confirmation_decision"
        ],
        paths[1].relative_to(context.root).as_posix(): integrity.hashes[
            "confirmation_state"
        ],
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if any(seed < 0 for seed in args.candidate_seeds):
            raise ValueError("candidate seeds must be non-negative integers")
        if args.stage == "realism-r6b-prepare":
            report = _execute_r6b_prepare(args)
            print(canonical_bytes(report).decode("utf-8"))
            return
        if args.stage == "realism-r6b-design":
            report = _execute_r6b_design(args)
            print(canonical_bytes(report).decode("utf-8"))
            return
        if args.stage == "realism-r6b-preflight":
            report = _execute_r6b_preflight(args)
            print(canonical_bytes(report).decode("utf-8"))
            return
        if args.stage == "realism-r6-preflight":
            report = _execute_r6_preflight(args)
            print(canonical_bytes(report).decode("utf-8"))
            return
        if args.stage.startswith("realism-r5-"):
            stage = args.stage.removeprefix("realism-r5-")
            operations = {
                "prepare": lambda: _execute_r5_prepare(args),
                "design": lambda: _execute_r5_design(args),
                "confirm": lambda: _execute_r5_confirm(args),
            }
            report = _run_r5_cli_stage(
                args.scenario_root, stage, operation=operations[stage]
            )
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return
        if args.stage == "realism-promote":
            from .scenario_promote import (
                PROMOTION_CONFIRMATION,
                authorize_promotion,
                promote_realism_v2,
            )

            if args.confirm_promotion != PROMOTION_CONFIRMATION:
                raise ValueError(
                    "exact destructive promotion confirmation flag/token is required"
                )
            supplied_root = args.scenario_root.expanduser()
            supplied_source = args.source_scenario_root.expanduser()
            resolved_root = (
                _REALISM_ROOT_RESOLVED
                if not supplied_root.is_absolute() and supplied_root == _REALISM_ROOT
                else supplied_root.resolve()
            )
            expected_source = (_PROJECT_ROOT / "scenarios/q2-ideal").resolve()
            resolved_source = (
                expected_source
                if not supplied_source.is_absolute()
                and supplied_source == Path("scenarios/q2-ideal")
                else supplied_source.resolve()
            )
            if resolved_root != _REALISM_ROOT_RESOLVED or resolved_source != expected_source:
                raise ValueError(
                    "realism promotion requires exact scenarios/q2-ideal and "
                    "scenarios/q2-realism-v2 roots"
                )
            authorization = authorize_promotion(args.confirm_promotion)
            report = promote_realism_v2(
                resolved_source, resolved_root, authorization=authorization
            )
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return
        if args.stage in {"realism-precheck", "realism-run"}:
            supplied_root = args.scenario_root.expanduser()
            if supplied_root.is_absolute():
                resolved_root = supplied_root.resolve()
            elif supplied_root == _REALISM_ROOT:
                resolved_root = _REALISM_ROOT_RESOLVED
            else:
                resolved_root = supplied_root.resolve()
            if resolved_root != _REALISM_ROOT_RESOLVED:
                raise ValueError(
                    "realism CLI requires the exact scenarios/q2-realism-v2 root"
                )
            args.scenario_root = _REALISM_ROOT_RESOLVED
            from .realism import candidate_catalog
            from .scenario_revision import run_realism_precheck, run_realism_revision

            manifests = {
                "dkasc_manifest": args.dkasc_manifest,
                "yulara_manifest": args.yulara_manifest,
                "nrel_manifest": args.nrel_manifest,
                "nrel_selection_audit": args.nrel_selection_audit,
            }
            prepared = {
                protocol: path
                for protocol, path in (
                    ("dkasc-yulara", args.dkasc_prepared_manifest),
                    ("nrel-transfer", args.nrel_prepared_manifest),
                )
                if path is not None
            }
            if prepared:
                manifests["prepared_manifests"] = prepared
            if args.stage == "realism-precheck":
                report = run_realism_precheck(
                    args.scenario_root, manifests, candidates=candidate_catalog()
                )
            else:
                screening_hash = args.screening_sha256
                if screening_hash is None:
                    freeze = (
                        args.scenario_root
                        / "00-manifest"
                        / "realism-v2-candidate-freeze.json"
                    )
                    if freeze.is_file():
                        screening_hash = json.loads(
                            freeze.read_text(encoding="utf-8")
                        ).get("screening_sha256")
                report = run_realism_revision(
                    args.scenario_root,
                    args.source_scenario_root,
                    manifests,
                    screening_sha256=screening_hash,
                    resume=args.resume,
                )
            print(json.dumps(report.to_payload(), ensure_ascii=False, sort_keys=True))
            return
        if args.stage == "preflight":
            run_preflight_4a(
                args.scenario_root,
                dkasc_manifest=args.dkasc_manifest,
                yulara_manifest=args.yulara_manifest,
                nrel_manifest=args.nrel_manifest,
                nrel_selection_audit=args.nrel_selection_audit,
                progress=print,
            )
            return
        if args.stage == "export-prediction":
            if not args.run_id or args.output is None:
                raise ValueError("export-prediction requires --run-id and --output")
            result = export_prediction_csv(args.scenario_root, args.run_id, args.output)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return
        stages = (
            ("prepare", "generate", "evaluate", "aggregate", "export", "verify")
            if args.stage == "all"
            else (args.stage,)
        )
        for stage in stages:
            if stage == "prepare":
                prepare(
                    args.scenario_root,
                    smoke=args.smoke,
                    manifest_path=args.manifest,
                    candidate_seeds=args.candidate_seeds,
                    protocol=args.protocol,
                )
            elif stage == "generate":
                generate(args.scenario_root, resume=args.resume)
            elif stage == "evaluate":
                evaluate(args.scenario_root)
            elif stage == "aggregate":
                aggregate(args.scenario_root)
            elif stage == "export":
                export(args.scenario_root)
            else:
                verify(args.scenario_root)
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error)) from error


__all__ = ["_cohort", "build_parser", "main"]
