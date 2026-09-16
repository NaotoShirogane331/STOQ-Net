"""Shared-cohort and compact prediction storage for scenario schema v2."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import ABLATIONS, BASELINES, HORIZON, QUANTILES, SEEDS
from .scenario_artifacts import sha256_file, write_json, write_npz

COMPACT_SCHEMA = "compact_shared_v2"
PREFLIGHT_LIMIT_BYTES = 40 * 1024**3
ACTUAL_LIMIT_BYTES = 45 * 1024**3
MINIMUM_FREE_RESERVE_BYTES = 15 * 1024**3


def prediction_storage_contract() -> dict[str, Any]:
    return {
        "schema": COMPACT_SCHEMA,
        "persistent_dtype": "float32",
        "csv_mode": "deterministic_block_audit",
        "preflight_limit_bytes": PREFLIGHT_LIMIT_BYTES,
        "actual_limit_bytes": ACTUAL_LIMIT_BYTES,
        "minimum_free_reserve_bytes": MINIMUM_FREE_RESERVE_BYTES,
    }


def ensure_bundle_space(root: Path, estimated_bytes: int) -> None:
    """Enforce the actual package cap and the 15 GiB free-space reserve."""

    root.mkdir(parents=True, exist_ok=True)
    current = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    if current + int(estimated_bytes) > ACTUAL_LIMIT_BYTES:
        raise ValueError("next bundle would exceed the 45 GiB scenario hard limit")
    free = shutil.disk_usage(root).free
    if free - int(estimated_bytes) < MINIMUM_FREE_RESERVE_BYTES:
        raise ValueError("next bundle would violate the 15 GiB temporary-space reserve")


def compact_storage_estimate(
    protocol_origin_counts: dict[str, dict[str, int]],
    *,
    calibration_origin_counts: dict[str, int] | None = None,
    protocol_block_counts: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Return an uncompressed upper bound for the complete compact matrix."""

    model_count = len(BASELINES) + 1 + len(ABLATIONS)
    runs_per_split = model_count * len(SEEDS)
    archives_per_protocol = runs_per_split * 2
    aliases_per_protocol = 9 * len(SEEDS)
    per_protocol: dict[str, dict[str, int]] = {}
    for protocol, counts in sorted(protocol_origin_counts.items()):
        unique_cells = sum(int(counts[split]) * HORIZON for split in ("validation", "test"))
        run_cells = unique_cells * runs_per_split
        prediction_bytes = run_cells * (len(QUANTILES) + 3) * np.dtype(np.float32).itemsize
        race_cells = unique_cells * len(SEEDS)
        race_extra_bytes = race_cells * (3 * len(QUANTILES) + 3) * 4
        shared_cohort_bytes = unique_cells * 32
        block_count = sum(
            int((protocol_block_counts or {}).get(protocol, {}).get(split, 0))
            for split in ("validation", "test")
        )
        audit_rows = block_count * runs_per_split * HORIZON * len(QUANTILES)
        audit_bytes = audit_rows * 160
        sidecar_bytes = archives_per_protocol * 4096 + aliases_per_protocol * 2048
        total = (
            prediction_bytes
            + race_extra_bytes
            + shared_cohort_bytes
            + audit_bytes
            + sidecar_bytes
        )
        per_protocol[protocol] = {
            "prediction_archives": archives_per_protocol,
            "sensitivity_aliases": aliases_per_protocol,
            "prediction_array_bytes": prediction_bytes,
            "race_extra_array_bytes": race_extra_bytes,
            "shared_cohort_bytes": shared_cohort_bytes,
            "audit_sample_rows": audit_rows,
            "audit_sample_estimated_bytes": audit_bytes,
            "sidecar_and_alias_estimated_bytes": sidecar_bytes,
            "total_bytes": total,
        }
    calibration_parameter_bytes = len(per_protocol) * runs_per_split * 4096
    evaluation_reserve = 2 * 1024**3
    combined_fields = (
        "prediction_archives",
        "sensitivity_aliases",
        "prediction_array_bytes",
        "race_extra_array_bytes",
        "shared_cohort_bytes",
        "audit_sample_rows",
        "audit_sample_estimated_bytes",
        "sidecar_and_alias_estimated_bytes",
        "total_bytes",
    )
    result: dict[str, Any] = {
        field: sum(value[field] for value in per_protocol.values())
        for field in combined_fields
    }
    result.update(
        schema=COMPACT_SCHEMA,
        persistent_dtype="float32",
        preflight_limit_bytes=PREFLIGHT_LIMIT_BYTES,
        actual_limit_bytes=ACTUAL_LIMIT_BYTES,
        minimum_free_reserve_bytes=MINIMUM_FREE_RESERVE_BYTES,
        calibration_parameter_estimated_bytes=calibration_parameter_bytes,
        evaluation_and_export_reserve_bytes=evaluation_reserve,
        calibration_origin_count=sum((calibration_origin_counts or {}).values()),
        per_protocol=per_protocol,
        method=(
            "float32 quantiles/state probabilities, shared cohort arrays, RaCE-only expert/router "
            "extras, one deterministic origin per selected block/run audit, compact CQR parameters, "
            "and a 2 GiB evaluation/export reserve"
        ),
    )
    result["total_bytes"] += calibration_parameter_bytes + evaluation_reserve
    return result


def _dictionary_encode(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dictionary, inverse = np.unique(np.asarray(values).astype("U"), return_inverse=True)
    if len(dictionary) > np.iinfo(np.uint16).max:
        raise ValueError("shared cohort dictionary exceeds uint16 capacity")
    return dictionary, inverse.reshape(np.asarray(values).shape).astype(np.uint16)


def shared_cohort_paths(root: Path, protocol: str, split: str) -> tuple[Path, Path]:
    path = root / "predictions" / "shared" / protocol / f"{split}.npz"
    return path, path.with_suffix(".json")


def write_shared_cohort(root: Path, protocol: str, cohort: Any) -> dict[str, Any]:
    path, sidecar_path = shared_cohort_paths(root, protocol, cohort.split)
    site_dictionary, site_codes = _dictionary_encode(cohort.site_ids)
    block_dictionary, block_codes = _dictionary_encode(cohort.block_ids)
    timestamps = np.asarray(
        [[pd.Timestamp(value).value for value in row] for row in cohort.timestamps],
        dtype=np.int64,
    )
    arrays = {
        "targets": np.asarray(cohort.targets, dtype=np.float32),
        "previous_targets": np.asarray(cohort.previous_targets, dtype=np.float32),
        "daylight": np.asarray(cohort.daylight, dtype=bool),
        "site_codes": site_codes,
        "site_dictionary": site_dictionary,
        "timestamps_ns": timestamps,
        "day_ids": np.asarray(cohort.day_ids).astype("datetime64[D]").astype(np.int32),
        "source_indices": np.asarray(cohort.source_indices, dtype=np.int64),
        "block_codes": block_codes[:, 0] if block_codes.ndim == 2 else block_codes,
        "block_dictionary": block_dictionary,
    }
    ensure_bundle_space(root, sum(value.nbytes for value in arrays.values()) + 65_536)
    write_npz(path, arrays)
    write_json(
        sidecar_path,
        {
            "storage_schema": COMPACT_SCHEMA,
            "protocol": protocol,
            "split": cohort.split,
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "arrays": sorted(arrays),
            "shape": list(cohort.targets.shape),
        },
    )
    return json.loads(sidecar_path.read_text(encoding="utf-8"))


def load_shared_cohort(root: Path, sidecar: dict[str, Any]) -> dict[str, np.ndarray]:
    path = root / sidecar["cohort_path"]
    if sha256_file(path) != sidecar["cohort_sha256"]:
        raise ValueError("shared cohort hash mismatch")
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    sites = arrays["site_dictionary"][arrays["site_codes"]]
    blocks = arrays["block_dictionary"][arrays["block_codes"]]
    return {**arrays, "site_ids": sites, "block_ids": blocks}


def compact_prediction_paths(
    root: Path, protocol: str, model: str, seed: int, split: str
) -> tuple[Path, Path]:
    path = root / "predictions" / "runs" / protocol / model / str(seed) / f"{split}.npz"
    return path, path.with_suffix(".json")


def state_labels_from_shared(shared: dict[str, np.ndarray]) -> np.ndarray:
    targets = np.asarray(shared["targets"], dtype=np.float64)
    previous = np.asarray(shared["previous_targets"], dtype=np.float64)
    daylight = np.asarray(shared["daylight"], dtype=bool)
    delta = targets - previous
    labels = np.zeros(targets.shape, dtype=np.int8)
    labels[delta >= 0.045] = 1
    labels[delta <= -0.045] = 2
    labels[~daylight | ~np.isfinite(previous)] = -100
    return labels


def load_prediction_arrays(root: Path, sidecar: dict[str, Any]) -> dict[str, np.ndarray]:
    """Load v2 compact or legacy v1 arrays through one compatibility boundary."""

    path_value = sidecar.get("path") or sidecar.get("prediction_path")
    if path_value is None:
        raise ValueError("prediction sidecar lacks an archive path")
    path = root / path_value
    if sidecar.get("content_sha256") != sha256_file(path):
        raise ValueError("prediction hash mismatch")
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    if sidecar.get("storage_schema", "legacy_full_v1") != COMPACT_SCHEMA:
        return arrays
    shared = load_shared_cohort(root, sidecar)
    probabilities = np.asarray(arrays["state_probabilities"], dtype=np.float64)
    result = {
        **arrays,
        "targets": np.asarray(shared["targets"], dtype=np.float64),
        "previous_targets": np.asarray(shared["previous_targets"], dtype=np.float64),
        "daylight": shared["daylight"],
        "site_ids": shared["site_ids"],
        "block_ids": shared["block_ids"],
        "timestamps_ns": shared["timestamps_ns"],
        "day_ids": shared["day_ids"],
        "state_labels": state_labels_from_shared(shared),
        "ramp_scores": probabilities[..., 1:].sum(axis=-1),
    }
    if "expert_quantiles" in arrays:
        result["gates"] = probabilities
    return result


def audit_origin_indices(cohort: Any, identity: str) -> list[tuple[str, int]]:
    selected = []
    for block_id in sorted(set(map(str, cohort.block_ids))):
        candidates = np.flatnonzero(cohort.block_ids == block_id)
        digest = hashlib.sha256(f"{identity}|{block_id}".encode()).digest()
        selected.append((block_id, int(candidates[int.from_bytes(digest[:8], "big") % len(candidates)])))
    return selected


def write_audit_sample(
    root: Path,
    records: list[tuple[dict[str, Any], Any]],
    aliases: dict[str, str],
) -> tuple[Path, Path, str]:
    path = root / "predictions" / "audit-sample.csv.gz"
    sidecar_path = path.with_suffix("").with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    estimated_rows = sum(
        len(set(map(str, cohort.block_ids))) * HORIZON * len(QUANTILES)
        for _sidecar, cohort in records
    )
    ensure_bundle_space(root, estimated_rows * 160 + 65_536)
    temporary = path.with_suffix(path.suffix + ".tmp")
    row_count = 0
    try:
        with temporary.open("wb") as binary:
            with gzip.GzipFile(filename="", mode="wb", fileobj=binary, mtime=0) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
                    fields = (
                        "run_id",
                        "protocol",
                        "split",
                        "model",
                        "seed",
                        "block_id",
                        "site_id",
                        "origin",
                        "lead",
                        "quantile",
                        "prediction",
                        "target",
                    )
                    writer = csv.DictWriter(text, fieldnames=fields)
                    writer.writeheader()
                    for sidecar, cohort in sorted(records, key=lambda item: item[0]["run_id"]):
                        archive_path = root / sidecar["path"]
                        with np.load(archive_path, allow_pickle=False) as source:
                            quantiles = source["quantiles"]
                        for block_id, origin in audit_origin_indices(cohort, sidecar["run_id"]):
                            for lead in range(HORIZON):
                                for q_index, level in enumerate(QUANTILES):
                                    writer.writerow(
                                        {
                                            "run_id": sidecar["run_id"],
                                            "protocol": sidecar["protocol"],
                                            "split": sidecar["split"],
                                            "model": sidecar["model"],
                                            "seed": sidecar["seed"],
                                            "block_id": block_id,
                                            "site_id": str(cohort.site_ids[origin, lead]),
                                            "origin": origin,
                                            "lead": lead + 1,
                                            "quantile": format(level, ".12g"),
                                            "prediction": format(
                                                float(quantiles[origin, lead, q_index]), ".12g"
                                            ),
                                            "target": format(
                                                float(np.float32(cohort.targets[origin, lead])),
                                                ".12g",
                                            ),
                                        }
                                    )
                                    row_count += 1
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    digest = sha256_file(path)
    write_json(
        sidecar_path,
        {
            "storage_schema": COMPACT_SCHEMA,
            "path": str(path.relative_to(root)),
            "sha256": digest,
            "sampling": "one_stable_hash_origin_per_run_and_selected_block",
            "row_count": row_count,
            "prediction_run_ids": sorted(sidecar["run_id"] for sidecar, _ in records),
            "aliases": dict(sorted(aliases.items())),
        },
    )
    return path, sidecar_path, digest


def _find_run_sidecar(root: Path, identity: str) -> dict[str, Any]:
    for path in sorted((root / "predictions" / "runs").rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("run_id") == identity:
            payload["_sidecar_path"] = str(path.relative_to(root))
            return payload
    raise ValueError(f"unknown prediction run ID: {identity}")


def prediction_entries(
    root: Path,
) -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    """Discover logical runs and resolve aliases across v2 and legacy v1 layouts."""

    runs_root = root / "predictions" / "runs"
    entries: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    if runs_root.is_dir():
        sidecars: dict[str, tuple[Path, dict[str, Any]]] = {}
        aliases: list[tuple[Path, dict[str, Any]]] = []
        for path in sorted(runs_root.rglob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "run_id" not in payload:
                continue
            if "alias_of_run_id" in payload:
                aliases.append((path, payload))
            elif "path" in payload:
                sidecars[payload["run_id"]] = (path, payload)
        entries.extend((path, payload, payload) for path, payload in sidecars.values())
        for path, payload in aliases:
            base = sidecars.get(payload["alias_of_run_id"])
            if base is None:
                raise ValueError(f"alias source sidecar missing: {payload['run_id']}")
            entries.append((path, payload, base[1]))
        return sorted(entries, key=lambda item: item[1]["run_id"])

    predictions = root / "predictions"
    for archive in sorted(predictions.glob("*.npz")):
        sidecar = json.loads(archive.with_suffix(".json").read_text(encoding="utf-8"))
        entries.append((archive, sidecar, sidecar))
    legacy = {item[1]["run_id"]: item[2] for item in entries}
    for path in sorted(predictions.glob("*.alias.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        base = legacy.get(payload["alias_of_run_id"])
        if base is None:
            source_path = root / payload["alias_of_path"]
            base = json.loads(source_path.with_suffix(".json").read_text(encoding="utf-8"))
        entries.append((path, payload, base))
    return sorted(entries, key=lambda item: item[1]["run_id"])


def export_prediction_csv(root: Path, identity: str, output: Path) -> dict[str, Any]:
    """Materialize one complete long CSV outside the capped scenario package."""

    scenario_root = root.expanduser().resolve()
    destination = output.expanduser().resolve()
    if destination.is_relative_to(scenario_root):
        raise ValueError("on-demand full CSV must be written outside the scenario root")
    requested = _find_run_sidecar(scenario_root, identity)
    source = requested
    if "alias_of_run_id" in requested:
        source = _find_run_sidecar(scenario_root, requested["alias_of_run_id"])
    arrays = load_prediction_arrays(scenario_root, source)
    targets = np.asarray(arrays["targets"], dtype=np.float64)
    quantiles = np.asarray(arrays["quantiles"], dtype=np.float64)
    sites = np.asarray(arrays["site_ids"])
    row_count = int(targets.size * len(QUANTILES))
    estimated_bytes = row_count * 160
    destination.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination.parent).free
    if free - estimated_bytes < MINIMUM_FREE_RESERVE_BYTES:
        raise ValueError("insufficient free space for on-demand full CSV plus 15 GiB reserve")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as binary:
            with gzip.GzipFile(filename="", mode="wb", fileobj=binary, mtime=0) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8", newline="") as text:
                    fields = (
                        "run_id",
                        "protocol",
                        "split",
                        "model",
                        "seed",
                        "site_id",
                        "origin",
                        "lead",
                        "quantile",
                        "prediction",
                        "target",
                    )
                    writer = csv.DictWriter(text, fieldnames=fields)
                    writer.writeheader()
                    for origin in range(targets.shape[0]):
                        for lead in range(targets.shape[1]):
                            for q_index, level in enumerate(QUANTILES):
                                writer.writerow(
                                    {
                                        "run_id": identity,
                                        "protocol": requested["protocol"],
                                        "split": requested["split"],
                                        "model": requested["model"],
                                        "seed": requested["seed"],
                                        "site_id": str(sites[origin, lead]),
                                        "origin": origin,
                                        "lead": lead + 1,
                                        "quantile": format(level, ".12g"),
                                        "prediction": format(
                                            float(quantiles[origin, lead, q_index]), ".12g"
                                        ),
                                        "target": format(float(targets[origin, lead]), ".12g"),
                                    }
                                )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "run_id": identity,
        "path": str(destination),
        "row_count": row_count,
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "source_prediction_sha256": source["content_sha256"],
        "source_cohort_sha256": source["cohort_sha256"],
    }


__all__ = [
    "ACTUAL_LIMIT_BYTES",
    "COMPACT_SCHEMA",
    "MINIMUM_FREE_RESERVE_BYTES",
    "PREFLIGHT_LIMIT_BYTES",
    "audit_origin_indices",
    "compact_prediction_paths",
    "compact_storage_estimate",
    "export_prediction_csv",
    "ensure_bundle_space",
    "load_prediction_arrays",
    "load_shared_cohort",
    "prediction_storage_contract",
    "prediction_entries",
    "shared_cohort_paths",
    "state_labels_from_shared",
    "write_audit_sample",
    "write_shared_cohort",
]
