"""Read-only phase 4A audit for the 57-site Q2 scenario cohort."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from racepv.data import ManifestEntry, adapter_for, chronological_split_plan, load_manifest

from .cohort import NormalizedSiteSeries, _valid_blocks, select_cohort
from .compact_storage import compact_storage_estimate
from .scenario_artifacts import atomic_bytes, sha256_file, write_json

PREFLIGHT_EVIDENCE = {
    "evidence_status": "preflight_only_no_predictions",
    "oracle_conditioned": False,
    "registered_evidence_eligible": False,
    "phase": "4A",
}


@dataclass(frozen=True)
class SitePlan:
    """One immutable site assignment and the temporal splits it must support."""

    group: str
    protocol: str
    source_kind: str
    required_splits: tuple[str, ...]
    entry: ManifestEntry
    data_root: Path


def _require_count(label: str, values: tuple[Any, ...] | list[Any], expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"{label} requires exactly {expected} sites; found {len(values)}")


def _validate_entries(
    label: str,
    entries: tuple[ManifestEntry, ...],
    *,
    dataset: str,
    role: str,
    state: str | None,
) -> None:
    for entry in entries:
        if entry.dataset != dataset or entry.spatial_role != role:
            raise ValueError(
                f"{label} site {entry.site_id!r} has invalid dataset/role "
                f"{entry.dataset}/{entry.spatial_role}"
            )
        if state is not None and (entry.state or "").upper() != state:
            raise ValueError(
                f"{label} site {entry.site_id!r} has state {entry.state!r}, expected {state}"
            )


def build_site_plan(
    dkasc_manifest: str | Path,
    yulara_manifest: str | Path,
    nrel_manifest: str | Path,
    nrel_selection_audit: str | Path,
) -> tuple[SitePlan, ...]:
    """Resolve the registered 12+5+20+20 cohort without opening raw labels."""

    dkasc = load_manifest(dkasc_manifest)
    yulara = load_manifest(yulara_manifest)
    nrel = load_manifest(nrel_manifest)
    dkasc_entries = tuple(dkasc.entries)
    yulara_entries = tuple(yulara.entries)
    _require_count("DKASC", dkasc_entries, 12)
    _require_count("Yulara", yulara_entries, 5)
    _validate_entries(
        "DKASC", dkasc_entries, dataset="dkasc", role="source_train", state="NT"
    )
    _validate_entries(
        "Yulara",
        yulara_entries,
        dataset="dkasc",
        role="zero_parameter_test",
        state=None,
    )

    audit_path = Path(nrel_selection_audit).expanduser().resolve()
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_site_ids_by_state", {})
    ok_ids = tuple(map(str, selected.get("OK", ())))
    az_ids = tuple(map(str, selected.get("AZ", ())))
    _require_count("Oklahoma", ok_ids, 20)
    _require_count("Arizona", az_ids, 20)
    if len(set(ok_ids)) != 20 or len(set(az_ids)) != 20:
        raise ValueError("Oklahoma and Arizona selections must contain unique site IDs")

    by_id = {entry.site_id: entry for entry in nrel.entries}
    if len(by_id) != len(nrel.entries):
        raise ValueError("NREL manifest contains duplicate site IDs")
    missing = sorted(set((*ok_ids, *az_ids)).difference(by_id))
    if missing:
        raise ValueError(f"selected NREL sites are absent from the manifest: {', '.join(missing)}")
    ok_entries = tuple(by_id[site_id] for site_id in ok_ids)
    az_entries = tuple(by_id[site_id] for site_id in az_ids)
    _validate_entries(
        "Oklahoma",
        ok_entries,
        dataset="nrel",
        role="validation_calibration",
        state="OK",
    )
    _validate_entries(
        "Arizona",
        az_entries,
        dataset="nrel",
        role="zero_parameter_test",
        state="AZ",
    )

    plans = (
        *(
            SitePlan(
                "DKASC",
                "dkasc-yulara",
                "observed_pv",
                ("validation", "calibration"),
                entry,
                dkasc.root,
            )
            for entry in dkasc_entries
        ),
        *(
            SitePlan(
                "Yulara",
                "dkasc-yulara",
                "observed_pv",
                ("test",),
                entry,
                yulara.root,
            )
            for entry in yulara_entries
        ),
        *(
            SitePlan(
                "Oklahoma",
                "nrel-transfer",
                "simulated_pv",
                ("validation", "calibration"),
                entry,
                nrel.root,
            )
            for entry in ok_entries
        ),
        *(
            SitePlan(
                "Arizona",
                "nrel-transfer",
                "simulated_pv",
                ("test",),
                entry,
                nrel.root,
            )
            for entry in az_entries
        ),
    )
    if len({plan.entry.site_id for plan in plans}) != 57:
        raise ValueError("the 57-site plan contains duplicate site IDs")
    return plans


def _stream_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _longest_finite_run(index: Any, finite: np.ndarray) -> int:
    if not len(index):
        return 0
    contiguous = np.r_[True, np.asarray(index[1:] - index[:-1]) == np.timedelta64(15, "m")]
    best = current = 0
    for valid, follows in zip(finite, contiguous, strict=True):
        current = current + 1 if valid and follows else (1 if valid else 0)
        best = max(best, current)
    return int(best)


def inspect_site(
    plan: SitePlan,
    *,
    file_hash_cache: dict[Path, str] | None = None,
) -> dict[str, Any]:
    """Read one site once and audit every required split without making forecasts."""

    started = time.perf_counter()
    entry = plan.entry
    cache = {} if file_hash_cache is None else file_hash_cache
    site_row: dict[str, Any] = {
        "site_id": entry.site_id,
        "group": plan.group,
        "protocol": plan.protocol,
        "dataset": entry.dataset,
        "source_kind": plan.source_kind,
        "state": entry.state or "",
        "spatial_role": entry.spatial_role,
        "capacity_kw": entry.capacity_kw,
        "required_splits": ",".join(plan.required_splits),
        "source_file_count": 0,
        "source_bytes": 0,
        "resampled_rows": 0,
        "ready": False,
        "read_error": "",
        "elapsed_seconds": 0.0,
    }
    split_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    try:
        adapter = adapter_for(entry, plan.data_root)
        paths = adapter.resolve_files(entry)
        for path in paths:
            if path not in cache:
                cache[path] = _stream_hash(path)
            source_files.append(
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": cache[path],
                }
            )
        frame = adapter.load_resampled(entry)
        split_plan = chronological_split_plan(len(frame.values))
        site_row.update(
            source_file_count=len(paths),
            source_bytes=sum(item["size_bytes"] for item in source_files),
            resampled_rows=len(frame.values),
        )
        for split in plan.required_splits:
            boundary = split_plan[split]
            values = frame.values.iloc[boundary]
            observed = frame.observed_mask.iloc[boundary]
            imputed = frame.imputed_mask.iloc[boundary]
            power_kw = np.asarray(values["power"], dtype=float)
            normalized = np.clip(power_kw / float(entry.capacity_kw), 0.0, 1.05)
            normalized_site = NormalizedSiteSeries(
                entry.site_id,
                entry.dataset,
                values.index,
                normalized,
                np.ones(len(values), dtype=bool),
                split,
                tuple(item["sha256"] for item in source_files),
                entry.spatial_role,
                entry.state,
            )
            candidates = _valid_blocks(normalized_site)
            if candidates:
                cohort = select_cohort((normalized_site,), split=split)
                selected_blocks = cohort.blocks
                selected_origins = int(cohort.targets.shape[0])
            else:
                selected_blocks = ()
                selected_origins = 0
            residual_missing = ~np.isfinite(power_kw)
            split_row = {
                "site_id": entry.site_id,
                "group": plan.group,
                "protocol": plan.protocol,
                "split": split,
                "rows": len(values),
                "start": values.index[0].isoformat() if len(values) else "",
                "end": values.index[-1].isoformat() if len(values) else "",
                "observed_missing_fraction": float(1.0 - observed["power"].mean()),
                "imputed_fraction": float(imputed["power"].mean()),
                "residual_missing_fraction": float(residual_missing.mean()),
                "longest_finite_run_steps": _longest_finite_run(values.index, ~residual_missing),
                "longest_finite_run_days": _longest_finite_run(
                    values.index, ~residual_missing
                )
                / 96.0,
                "valid_block_count": len(candidates),
                "selected_block_count": len(selected_blocks),
                "selected_origin_count": selected_origins,
                "eligible": bool(selected_blocks),
            }
            split_rows.append(split_row)
            for block in selected_blocks:
                block_rows.append(
                    {
                        "block_id": block.block_id,
                        "site_id": block.site_id,
                        "group": plan.group,
                        "protocol": plan.protocol,
                        "split": split,
                        "start": block.start.isoformat(),
                        "end": block.end.isoformat(),
                        "quarter": block.quarter,
                        "n_steps": block.n_steps,
                        "selection_hash": block.selection_hash,
                    }
                )
        site_row["ready"] = all(row["eligible"] for row in split_rows)
    except Exception as error:  # noqa: BLE001 - audit must preserve one row per registered site
        site_row["read_error"] = f"{type(error).__name__}: {error}"
        for split in plan.required_splits:
            split_rows.append(
                {
                    "site_id": entry.site_id,
                    "group": plan.group,
                    "protocol": plan.protocol,
                    "split": split,
                    "rows": 0,
                    "start": "",
                    "end": "",
                    "observed_missing_fraction": 1.0,
                    "imputed_fraction": 0.0,
                    "residual_missing_fraction": 1.0,
                    "longest_finite_run_steps": 0,
                    "longest_finite_run_days": 0.0,
                    "valid_block_count": 0,
                    "selected_block_count": 0,
                    "selected_origin_count": 0,
                    "eligible": False,
                }
            )
    site_row["elapsed_seconds"] = time.perf_counter() - started
    return {
        "site": site_row,
        "splits": split_rows,
        "blocks": block_rows,
        "source_files": source_files,
    }


def estimate_storage(
    protocol_origin_counts: Mapping[str, Mapping[str, int]],
    *,
    calibration_origin_counts: Mapping[str, int] | None = None,
    protocol_block_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """Conservatively estimate the compact full two-protocol contract."""

    return compact_storage_estimate(
        {key: dict(value) for key, value in protocol_origin_counts.items()},
        calibration_origin_counts=(
            None
            if calibration_origin_counts is None
            else dict(calibration_origin_counts)
        ),
        protocol_block_counts=(
            None if protocol_block_counts is None else {
                key: dict(value) for key, value in protocol_block_counts.items()
            }
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, buffer.getvalue().encode("utf-8"))


def run_preflight_4a(
    root: str | Path,
    *,
    dkasc_manifest: str | Path,
    yulara_manifest: str | Path,
    nrel_manifest: str | Path,
    nrel_selection_audit: str | Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run phase 4A and persist only audit metadata, never predictions."""

    started_at = datetime.now(UTC)
    started = time.perf_counter()
    destination = Path(root).expanduser().resolve()
    before_predictions = {
        path.resolve()
        for pattern in ("*.npz", "*.csv.gz")
        for path in destination.glob(f"02-predictions/**/{pattern}")
    }
    plans = build_site_plan(
        dkasc_manifest, yulara_manifest, nrel_manifest, nrel_selection_audit
    )
    results = []
    hash_cache: dict[Path, str] = {}
    for index, plan in enumerate(plans, start=1):
        if progress is not None:
            progress(f"[4A {index:02d}/57] {plan.group}: {plan.entry.site_id}")
        results.append(inspect_site(plan, file_hash_cache=hash_cache))

    site_rows = [result["site"] for result in results]
    split_rows = [row for result in results for row in result["splits"]]
    block_rows = [row for result in results for row in result["blocks"]]
    protocol_counts = {
        protocol: {
            split: sum(
                int(row["selected_origin_count"])
                for row in split_rows
                if row["protocol"] == protocol and row["split"] == split
            )
            for split in ("validation", "test")
        }
        for protocol in ("dkasc-yulara", "nrel-transfer")
    }
    calibration_counts = {
        protocol: sum(
            int(row["selected_origin_count"])
            for row in split_rows
            if row["protocol"] == protocol and row["split"] == "calibration"
        )
        for protocol in protocol_counts
    }
    block_counts = {
        protocol: {
            split: sum(
                int(row["selected_block_count"])
                for row in split_rows
                if row["protocol"] == protocol and row["split"] == split
            )
            for split in ("validation", "test")
        }
        for protocol in protocol_counts
    }
    storage = estimate_storage(
        protocol_counts,
        calibration_origin_counts=calibration_counts,
        protocol_block_counts=block_counts,
    )
    disk = shutil.disk_usage(destination.parent if destination.parent.exists() else Path.cwd())
    storage.update(
        cap_bytes=storage["preflight_limit_bytes"],
        available_bytes=disk.free,
        within_cap=storage["total_bytes"] <= storage.get("preflight_limit_bytes", 40 * 1024**3),
        within_available_space=(
            storage["total_bytes"] + storage["minimum_free_reserve_bytes"] <= disk.free
        ),
    )
    data_ready = all(bool(row["ready"]) for row in site_rows)
    anomalies = [
        {
            "kind": "site_not_ready",
            "site_id": row["site_id"],
            "detail": row["read_error"] or "one or more required splits lack a 14-day block",
        }
        for row in site_rows
        if not row["ready"]
    ]
    if not storage["within_cap"]:
        anomalies.append(
            {
                "kind": "storage_cap_exceeded",
                "detail": (
                    f"{storage['total_bytes']} estimated bytes exceeds "
                    f"{storage['preflight_limit_bytes']}"
                ),
            }
        )
    if not storage["within_available_space"]:
        anomalies.append(
            {
                "kind": "available_space_insufficient",
                "detail": (
                    f"{storage['total_bytes']} estimated bytes plus the 15 GiB reserve "
                    f"exceeds {disk.free} free bytes"
                ),
            }
        )

    manifest_dir = destination / "00-manifest"
    cohort_dir = destination / "01-cohorts"
    log_dir = destination / "09-logs"
    for directory in (manifest_dir, cohort_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)
    inventory_path = cohort_dir / "site-inventory.csv"
    quality_path = cohort_dir / "site-split-quality.csv"
    blocks_path = cohort_dir / "selected-blocks.csv"
    _write_csv(inventory_path, site_rows)
    _write_csv(quality_path, split_rows)
    _write_csv(blocks_path, block_rows)

    source_paths = {
        "dkasc_manifest": Path(dkasc_manifest).expanduser().resolve(),
        "yulara_manifest": Path(yulara_manifest).expanduser().resolve(),
        "nrel_manifest": Path(nrel_manifest).expanduser().resolve(),
        "nrel_selection_audit": Path(nrel_selection_audit).expanduser().resolve(),
    }
    source_contract = {
        **PREFLIGHT_EVIDENCE,
        "site_count": len(plans),
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
        "sites": [
            {
                "site_id": plan.entry.site_id,
                "group": plan.group,
                "protocol": plan.protocol,
                "source_kind": plan.source_kind,
                "required_splits": list(plan.required_splits),
            }
            for plan in plans
        ],
        "raw_files": [item for result in results for item in result["source_files"]],
    }
    source_contract_path = manifest_dir / "data-source-contract.json"
    write_json(source_contract_path, source_contract)

    finished_at = datetime.now(UTC)
    report = {
        **PREFLIGHT_EVIDENCE,
        "status": (
            "passed"
            if data_ready and storage["within_cap"] and storage["within_available_space"]
            else "blocked"
        ),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "site_count": len(site_rows),
        "site_split_count": len(split_rows),
        "selected_block_count": len(block_rows),
        "group_counts": {
            group: sum(row["group"] == group for row in site_rows)
            for group in ("Arizona", "DKASC", "Oklahoma", "Yulara")
        },
        "data_ready": data_ready,
        "generation_ready": data_ready
        and storage["within_cap"]
        and storage["within_available_space"],
        "protocol_origin_counts": protocol_counts,
        "calibration_origin_counts": calibration_counts,
        "storage": storage,
        "anomalies": anomalies,
        "prediction_generation_performed": False,
        "prediction_artifacts_created": 0,
        "artifacts": {
            "site_inventory": str(inventory_path.relative_to(destination)),
            "site_split_quality": str(quality_path.relative_to(destination)),
            "selected_blocks": str(blocks_path.relative_to(destination)),
            "data_source_contract": str(source_contract_path.relative_to(destination)),
        },
    }
    report_path = manifest_dir / "preflight-4a.json"
    write_json(report_path, report)
    artifact_paths = (report_path, source_contract_path, inventory_path, quality_path, blocks_path)
    artifact_index = {
        **PREFLIGHT_EVIDENCE,
        "artifacts": [
            {
                "path": str(path.relative_to(destination)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
    }
    write_json(log_dir / "preflight-4a-artifacts.json", artifact_index)
    after_predictions = {
        path.resolve()
        for pattern in ("*.npz", "*.csv.gz")
        for path in destination.glob(f"02-predictions/**/{pattern}")
    }
    if after_predictions != before_predictions:
        raise RuntimeError("4A preflight changed prediction artifacts")
    return report


__all__ = [
    "SitePlan",
    "build_site_plan",
    "estimate_storage",
    "inspect_site",
    "run_preflight_4a",
]
