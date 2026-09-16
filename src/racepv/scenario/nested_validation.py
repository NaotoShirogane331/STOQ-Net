"""Inputs that are isolated from the registered r1--r4 evidence pipeline."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import stat
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from .cohort import (
    BLOCK_STEPS,
    HORIZON,
    BlockRecord,
    Cohort,
    NormalizedSiteSeries,
    _valid_blocks,
    stable_hash,
)
from .constants import STEP_MINUTES
from .loader import role_allows_split

EXPECTED_AUXILIARY_OK_SITE_IDS = (
    "nrel_ok_p34d75_m98d45_29mw_dpv",
    "nrel_ok_p35d25_m97d25_31mw_dpv",
    "nrel_ok_p35d55_m97d45_37mw_dpv",
    "nrel_ok_p35d55_m97d55_37mw_dpv",
    "nrel_ok_p36d35_m95d65_8mw_dpv",
)
EXPECTED_AUXILIARY_OK_ARCHIVE_SHA256 = (
    "b83cfcf2deaa12f5b85f4bf5578b3b8bb72e8ee83154e2d615a953d7ea32df89"
)
CONFIRMATION_SELECTION_SEED = 20260816
CONFIRMATION_EMBARGO_DAYS = 14


@dataclass(frozen=True)
class ExtractedSource:
    site_id: str
    archive_sha256: str
    member: str
    member_sha256: str
    output_path: str
    output_sha256: str


@dataclass(frozen=True)
class ConfirmationSourceBinding:
    """Independent anchor for one confirmation source series."""

    site_id: str
    raw_hashes: tuple[str, ...]
    source_origin_timestamp: str
    block_start_indices: tuple[int, ...]


@dataclass(frozen=True)
class ConfirmationContract:
    """Independent, predeclared bindings that cannot be derived from mutable inputs."""

    dkasc_site_ids: tuple[str, ...]
    old_cohort_sha256: str
    extraction_sha256: str
    prior_screen_coordinates_sha256: str
    source_series: tuple[ConfirmationSourceBinding, ...] = ()


@dataclass(frozen=True)
class ConfirmationInputs:
    """The source-bound inputs required before confirmation screening may start."""

    dkasc_sites: tuple[NormalizedSiteSeries, ...]
    oklahoma_sites: tuple[NormalizedSiteSeries, ...]
    old_selected_blocks: tuple[BlockRecord, ...]
    old_cohort_sha256: str
    extraction_sources: tuple[ExtractedSource, ...]
    extraction_sha256: str
    prior_screen_coordinates: frozenset[tuple[str, str, str]] = frozenset()
    confirmation_contract: ConfirmationContract | None = None


@dataclass(frozen=True)
class ConfirmationCohorts:
    """The two isolated confirmation cohorts and their sealed provenance."""

    dkasc: Cohort
    oklahoma: Cohort
    old_cohort_sha256: str
    extraction_sha256: str
    prior_screen_exclusion_sha256: str
    selection_seed: int
    auxiliary_confirmation_only: bool
    dkasc_contract_site_ids: tuple[str, ...]
    dkasc_cohort_sha256: str
    oklahoma_cohort_sha256: str
    protocol_sha256: str
    confirmation_contract: ConfirmationContract


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _old_block_rows(blocks: Sequence[BlockRecord]) -> list[dict[str, Any]]:
    return [
        {
            "block_id": block.block_id,
            "dataset": block.dataset,
            "end": block.end.isoformat(),
            "raw_hashes": list(block.raw_hashes),
            "site_id": block.site_id,
            "spatial_role": block.spatial_role,
            "split": block.split,
            "state": block.state,
            "start": block.start.isoformat(),
        }
        for block in sorted(blocks, key=lambda item: (item.site_id, item.start, item.block_id))
    ]


def _source_rows(sources: Sequence[ExtractedSource]) -> list[dict[str, str]]:
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


def _source_binding_rows(
    bindings: Sequence[ConfirmationSourceBinding],
) -> list[dict[str, Any]]:
    return [
        {
            "block_start_indices": list(binding.block_start_indices),
            "raw_hashes": list(binding.raw_hashes),
            "site_id": binding.site_id,
            "source_origin_timestamp": binding.source_origin_timestamp,
        }
        for binding in sorted(bindings, key=lambda item: item.site_id)
    ]


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validated_source_bindings(
    contract: ConfirmationContract,
) -> dict[str, ConfirmationSourceBinding]:
    expected_site_ids = tuple(
        sorted((*contract.dkasc_site_ids, *EXPECTED_AUXILIARY_OK_SITE_IDS))
    )
    bindings = tuple(sorted(contract.source_series, key=lambda item: item.site_id))
    if tuple(binding.site_id for binding in bindings) != expected_site_ids:
        raise ValueError("confirmation contract source binding identities are invalid")
    for binding in bindings:
        if (
            not binding.raw_hashes
            or not all(_valid_sha256(value) for value in binding.raw_hashes)
            or not binding.source_origin_timestamp
        ):
            raise ValueError("confirmation contract source binding hash is invalid")
        origin = pd.Timestamp(binding.source_origin_timestamp)
        if origin.tz is None or origin.isoformat() != binding.source_origin_timestamp:
            raise ValueError("confirmation contract source origin is invalid")
        starts = binding.block_start_indices
        if (
            not starts
            or tuple(sorted(starts)) != starts
            or len(set(starts)) != len(starts)
            or any(isinstance(start, bool) or not isinstance(start, int) or start < 0 for start in starts)
        ):
            raise ValueError("confirmation contract source block indices are invalid")
    return {binding.site_id: binding for binding in bindings}


def _require_source_binding_matches_site(
    binding: ConfirmationSourceBinding, site: NormalizedSiteSeries
) -> None:
    if binding.raw_hashes != site.raw_hashes:
        raise ValueError("confirmation source binding raw hashes do not match source series")
    if binding.source_origin_timestamp != site.timestamps[0].isoformat():
        raise ValueError("confirmation source binding origin does not match source series")
    source_starts = tuple(start for start, _ in _valid_blocks(site))
    if binding.block_start_indices != source_starts:
        raise ValueError("confirmation source binding block indices do not match source series")


def _require_bound_source_hashes(inputs: ConfirmationInputs) -> None:
    contract = inputs.confirmation_contract
    if contract is None:
        raise ValueError("confirmation requires an independent source contract")
    if (
        len(contract.dkasc_site_ids) != 12
        or tuple(sorted(contract.dkasc_site_ids)) != contract.dkasc_site_ids
        or len(set(contract.dkasc_site_ids)) != 12
        or any(not isinstance(site_id, str) or not site_id for site_id in contract.dkasc_site_ids)
    ):
        raise ValueError("confirmation contract DKASC identity is invalid")
    if not all(
        _valid_sha256(value)
        for value in (
            contract.old_cohort_sha256,
            contract.extraction_sha256,
            contract.prior_screen_coordinates_sha256,
        )
    ):
        raise ValueError("confirmation contract hash is invalid")
    source_bindings = _validated_source_bindings(contract)
    if (
        inputs.old_cohort_sha256 != contract.old_cohort_sha256
        or _sha256_json(_old_block_rows(inputs.old_selected_blocks)) != contract.old_cohort_sha256
    ):
        raise ValueError("old-cohort hash integrity failure")
    if (
        inputs.extraction_sha256 != contract.extraction_sha256
        or _sha256_json(_source_rows(inputs.extraction_sources)) != contract.extraction_sha256
    ):
        raise ValueError("extraction hash integrity failure")
    coordinates = tuple(sorted(inputs.prior_screen_coordinates))
    if any(
        not isinstance(item, tuple)
        or len(item) != 3
        or any(not isinstance(value, str) or not value for value in item)
        for item in coordinates
    ):
        raise ValueError("prior-screen coordinates are invalid")
    if _sha256_json([list(item) for item in coordinates]) != contract.prior_screen_coordinates_sha256:
        raise ValueError("prior-screen coordinate contract integrity failure")
    _require_complete_old_history(inputs, source_bindings)
    for source in inputs.extraction_sources:
        if not all(
            _valid_sha256(value)
            for value in (
                source.archive_sha256,
                source.member_sha256,
                source.output_sha256,
            )
        ):
            raise ValueError("auxiliary extraction source hash is invalid")
    for site in (*inputs.dkasc_sites, *inputs.oklahoma_sites):
        binding = source_bindings.get(site.site_id)
        if binding is None:
            raise ValueError("confirmation source binding is missing a source series")
        _require_source_binding_matches_site(binding, site)


def _coordinate(block: BlockRecord) -> tuple[str, str, str]:
    return block.site_id, block.start.isoformat(), block.end.isoformat()


def _require_complete_old_history(
    inputs: ConfirmationInputs,
    source_bindings: Mapping[str, ConfirmationSourceBinding],
) -> None:
    """Require the immutable r1--r4 DKASC validation history before screening."""
    old_blocks = inputs.old_selected_blocks
    if len(old_blocks) != 48:
        raise ValueError("confirmation old history requires exactly forty-eight blocks")
    coordinates = tuple(_coordinate(block) for block in old_blocks)
    if len(set(coordinates)) != 48 or len({block.block_id for block in old_blocks}) != 48:
        raise ValueError("confirmation old history blocks and coordinates must be unique")
    expected_sites = inputs.confirmation_contract.dkasc_site_ids
    if any(block.dataset != "dkasc" for block in old_blocks):
        raise ValueError("confirmation old history dataset is invalid")
    if tuple(sorted(block.site_id for block in old_blocks)) != tuple(
        site_id for site_id in expected_sites for _ in range(4)
    ):
        raise ValueError("confirmation old history requires four blocks per DKASC site")
    prior_coordinates = set(inputs.prior_screen_coordinates)
    if not set(coordinates).issubset(prior_coordinates):
        raise ValueError("confirmation prior-screen coordinates omit old history")
    for block in old_blocks:
        if block.split != "validation":
            raise ValueError("held-out old block split is forbidden in confirmation")
        role = block.spatial_role or ""
        if role in {"zero_parameter_test", "test"}:
            raise ValueError("held-out old block role is forbidden in confirmation")
        if not role_allows_split(
            block.dataset, role, block.state, "validation", site_id=block.site_id
        ):
            raise ValueError("old block role/state is invalid for confirmation")
        binding = source_bindings.get(block.site_id)
        if binding is None or block.raw_hashes != binding.raw_hashes:
            raise ValueError("confirmation old history source identity is invalid")
        if not block.raw_hashes or not all(_valid_sha256(value) for value in block.raw_hashes):
            raise ValueError("old block source hash is invalid")
        expected_end = block.start + pd.Timedelta(minutes=STEP_MINUTES * (BLOCK_STEPS - 1))
        if block.n_steps != BLOCK_STEPS or block.end != expected_end:
            raise ValueError("confirmation old history block boundary is invalid")
        if block.quarter != f"{block.start.year}Q{block.start.quarter}":
            raise ValueError("confirmation old history quarter is invalid")
        source_digest = _source_block_digest(block)
        if block.block_id != source_digest[:20]:
            raise ValueError("confirmation old history block id is invalid")
        if block.selection_hash != source_digest:
            raise ValueError("confirmation old history selection hash is invalid")
        origin = pd.Timestamp(binding.source_origin_timestamp)
        source_starts = {
            origin + pd.Timedelta(minutes=STEP_MINUTES * start)
            for start in binding.block_start_indices
        }
        if block.start not in source_starts:
            raise ValueError("confirmation old history source boundary is invalid")


def _same_source(left: BlockRecord, right: BlockRecord) -> bool:
    return (left.dataset, left.site_id) == (right.dataset, right.site_id) or bool(
        set(left.raw_hashes).intersection(right.raw_hashes)
    )


def _interval_gap(left: BlockRecord, right: BlockRecord) -> pd.Timedelta:
    if left.end < right.start:
        return right.start - left.end
    if right.end < left.start:
        return left.start - right.end
    return pd.Timedelta(0)


def _require_dkasc_selection_isolation(
    selected: Sequence[tuple[NormalizedSiteSeries, int, BlockRecord]],
    *,
    old_blocks: Sequence[BlockRecord],
    prior_coordinates: frozenset[tuple[str, str, str]],
) -> None:
    """Recheck exact reuse and embargo after deterministic ranking."""
    embargo = pd.Timedelta(days=CONFIRMATION_EMBARGO_DAYS)
    for _site, _start, block in selected:
        if _coordinate(block) in prior_coordinates:
            raise ValueError("DKASC confirmation selection reuses a prior coordinate")
        if any(
            block.site_id == old.site_id and _interval_gap(block, old) < embargo
            for old in old_blocks
        ):
            raise ValueError("DKASC confirmation selection violates the old-history embargo")


def _confirmation_rank(site_id: str, block: BlockRecord) -> str:
    return stable_hash(
        "nested-confirmation-v1",
        CONFIRMATION_SELECTION_SEED,
        site_id,
        block.start.isoformat(),
        block.end.isoformat(),
    )


def _require_validation_role(site: NormalizedSiteSeries, *, auxiliary: bool) -> None:
    role = site.spatial_role or ""
    if role in {"zero_parameter_test", "test"}:
        raise ValueError("held-out role is forbidden in confirmation")
    if not role_allows_split(
        site.dataset, role, site.state, "validation", site_id=site.site_id
    ):
        label = "auxiliary" if auxiliary else "DKASC"
        raise ValueError(f"{label} site role/state is not valid for confirmation")


def _cohort_from_selected(
    selected: Sequence[tuple[NormalizedSiteSeries, int, BlockRecord]],
) -> Cohort:
    """Materialize selected complete blocks using the established cohort cell contract."""
    targets: list[np.ndarray] = []
    daylight: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    day_ids: list[np.ndarray] = []
    site_ids: list[np.ndarray] = []
    source_indices: list[np.ndarray] = []
    previous_targets: list[np.ndarray] = []
    block_ids: list[np.ndarray] = []
    records: list[BlockRecord] = []
    for site, start, record in sorted(selected, key=lambda item: (item[2].site_id, item[2].start)):
        offsets = np.arange(BLOCK_STEPS - HORIZON)[:, None] + np.arange(HORIZON)[None, :]
        target_index = start + offsets
        prior_index = target_index - 2
        keep = np.all(
            (prior_index >= start)
            & np.isfinite(site.power[target_index])
            & np.isfinite(site.power[np.maximum(prior_index, start)]),
            axis=1,
        )
        target_index = target_index[keep]
        prior_index = prior_index[keep]
        times = site.timestamps.to_numpy()[target_index]
        targets.append(site.power[target_index])
        daylight.append(site.daylight[target_index])
        timestamps.append(times)
        day_ids.append(pd.DatetimeIndex(times).tz_localize(None).to_numpy(dtype="datetime64[D]"))
        site_ids.append(np.full(times.shape, site.site_id, dtype=object))
        source_indices.append(target_index)
        previous_targets.append(site.power[prior_index])
        block_ids.append(np.full(target_index.shape[0], record.block_id, dtype=object))
        records.append(record)
    return Cohort(
        "validation",
        tuple(records),
        np.concatenate(targets),
        np.concatenate(daylight),
        np.concatenate(timestamps),
        np.concatenate(day_ids),
        np.concatenate(site_ids),
        np.concatenate(source_indices),
        np.concatenate(previous_targets),
        np.concatenate(block_ids),
    )


def _cohort_integrity_sha256(cohort: Cohort) -> str:
    """Bind every mutable cohort array before it crosses the screening boundary."""
    digest = hashlib.sha256()
    digest.update(cohort.split.encode("utf-8"))
    values = (
        cohort.targets,
        cohort.daylight,
        np.asarray(
            [[pd.Timestamp(value).value for value in row] for row in cohort.timestamps],
            dtype=np.int64,
        ),
        cohort.day_ids,
        cohort.site_ids.astype("U"),
        cohort.source_indices,
    )
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    for name, value in (
        ("previous_targets", cohort.previous_targets),
        ("block_ids", cohort.block_ids),
    ):
        if value is None:
            digest.update(f"{name}:none".encode())
            continue
        array = np.ascontiguousarray(value.astype("U") if name == "block_ids" else value)
        digest.update(f"{name}:present".encode())
        digest.update(array.dtype.str.encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _require_selected_block_isolation(cohort: Cohort) -> None:
    for block in cohort.blocks:
        if block.split != "validation":
            raise ValueError("selected confirmation block split is invalid")
        role = block.spatial_role or ""
        if role in {"zero_parameter_test", "test"}:
            raise ValueError("selected confirmation block held-out role is invalid")
        if not role_allows_split(
            block.dataset, role, block.state, "validation", site_id=block.site_id
        ):
            raise ValueError("selected confirmation block role/state is invalid")
        if not block.raw_hashes or not all(_valid_sha256(value) for value in block.raw_hashes):
            raise ValueError("selected confirmation block source hash is invalid")


def _source_block_digest(block: BlockRecord) -> str:
    """Return the immutable source-block identity used by ``_valid_blocks``."""
    return stable_hash(
        block.dataset,
        block.site_id,
        block.split,
        block.start.isoformat(),
        block.end.isoformat(),
    )


def _require_canonical_block_record(
    block: BlockRecord,
    *,
    cohort: str,
    source_binding: ConfirmationSourceBinding,
) -> None:
    """Validate the fields that identify one complete source block."""
    expected_dataset = "dkasc" if cohort == "dkasc" else "nrel"
    if block.dataset != expected_dataset:
        raise ValueError(f"{cohort} block dataset is invalid")
    if block.raw_hashes != source_binding.raw_hashes:
        raise ValueError(f"{cohort} block source hashes do not match source binding")
    if block.n_steps != BLOCK_STEPS:
        raise ValueError(f"{cohort} block n_steps is invalid")
    expected_end = block.start + pd.Timedelta(minutes=STEP_MINUTES * (BLOCK_STEPS - 1))
    if block.end != expected_end:
        raise ValueError(f"{cohort} block boundary is invalid")
    if block.quarter != f"{block.start.year}Q{block.start.quarter}":
        raise ValueError(f"{cohort} block quarter is invalid")
    source_digest = _source_block_digest(block)
    if block.block_id != source_digest[:20]:
        raise ValueError(f"{cohort} block_id is not canonical")
    expected_selection_hash = (
        _confirmation_rank(block.site_id, block) if cohort == "dkasc" else source_digest
    )
    if block.selection_hash != expected_selection_hash:
        raise ValueError(f"{cohort} block selection hash is not canonical")


def _require_block_row_alignment(
    cohort: Cohort,
    block: BlockRecord,
    *,
    label: str,
    source_binding: ConfirmationSourceBinding,
) -> None:
    """Reconstruct a block's public sliding-window rows from its record boundary."""
    assert cohort.block_ids is not None
    block_ids = np.asarray(cohort.block_ids)
    row_indices = np.flatnonzero(block_ids == block.block_id)
    expected_rows = BLOCK_STEPS - HORIZON - 2
    if len(row_indices) != expected_rows:
        raise ValueError(f"{label} block row cardinality is invalid")
    source_indices = np.asarray(cohort.source_indices)[row_indices]
    if not np.issubdtype(source_indices.dtype, np.integer):
        raise ValueError(f"{label} block source indices are invalid")
    source_start = int(source_indices[0, 0]) - 2
    if source_start not in source_binding.block_start_indices:
        raise ValueError(f"{label} block source index is not anchored to its source binding")
    origin = pd.Timestamp(source_binding.source_origin_timestamp)
    expected_start = origin + pd.Timedelta(minutes=STEP_MINUTES * source_start)
    if block.start != expected_start:
        raise ValueError(f"{label} block source boundary is not anchored")
    expected_indices = (
        source_start
        + 2
        + np.arange(expected_rows)[:, None]
        + np.arange(HORIZON)[None, :]
    )
    if not np.array_equal(source_indices, expected_indices):
        raise ValueError(f"{label} block source index rows are invalid")
    timestamps = np.asarray(cohort.timestamps, dtype=object)[row_indices]
    for row_timestamps, row_sources in zip(timestamps, source_indices, strict=True):
        for timestamp, source_index in zip(row_timestamps, row_sources, strict=True):
            expected_timestamp = block.start + pd.Timedelta(
                minutes=STEP_MINUTES * (int(source_index) - source_start)
            )
            if pd.Timestamp(timestamp) != expected_timestamp:
                raise ValueError(f"{label} block timestamp rows are invalid")


def _require_cohort_structure(
    cohort: Cohort,
    *,
    label: str,
    expected_block_count: int,
    expected_site_ids: tuple[str, ...],
    source_bindings: Mapping[str, ConfirmationSourceBinding],
) -> None:
    """Reject a sealed result whose block records and public arrays disagree."""
    if cohort.split != "validation":
        raise ValueError(f"{label} cohort split is invalid")
    if len(cohort.blocks) != expected_block_count:
        raise ValueError(f"{label} cohort block cardinality is invalid")
    block_ids = tuple(block.block_id for block in cohort.blocks)
    block_site_ids = tuple(sorted(block.site_id for block in cohort.blocks))
    if (
        len(set(block_ids)) != expected_block_count
        or tuple(sorted(set(block_site_ids))) != expected_site_ids
    ):
        raise ValueError(f"{label} cohort block/site identity is invalid")
    expected_blocks_per_site = expected_block_count // len(expected_site_ids)
    if any(block_site_ids.count(site_id) != expected_blocks_per_site for site_id in expected_site_ids):
        raise ValueError(f"{label} cohort site block distribution is invalid")
    if cohort.previous_targets is None or cohort.block_ids is None:
        raise ValueError(f"{label} cohort optional arrays are required")
    origin_count = int(cohort.targets.shape[0]) if cohort.targets.ndim == 2 else -1
    expected_matrix_shape = (origin_count, HORIZON)
    if origin_count <= 0 or cohort.targets.shape != expected_matrix_shape:
        raise ValueError(f"{label} cohort target array shape is invalid")
    for name, value in (
        ("daylight", cohort.daylight),
        ("timestamps", cohort.timestamps),
        ("day_ids", cohort.day_ids),
        ("site_ids", cohort.site_ids),
        ("source_indices", cohort.source_indices),
        ("previous_targets", cohort.previous_targets),
    ):
        if np.asarray(value).shape != expected_matrix_shape:
            raise ValueError(f"{label} cohort {name} array shape is invalid")
    for name, value in (
        ("targets", cohort.targets),
        ("previous_targets", cohort.previous_targets),
    ):
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
            raise ValueError(f"{label} cohort {name} must be finite")
    if not np.issubdtype(np.asarray(cohort.daylight).dtype, np.bool_):
        raise ValueError(f"{label} cohort daylight must be boolean")
    if np.asarray(cohort.block_ids).shape != (origin_count,):
        raise ValueError(f"{label} cohort block_ids array shape is invalid")
    ownership = {block.block_id: block.site_id for block in cohort.blocks}
    observed_block_ids = tuple(str(value) for value in cohort.block_ids)
    if set(observed_block_ids) != set(ownership):
        raise ValueError(f"{label} cohort block_ids do not match selected blocks")
    for block_id, row_site_ids in zip(observed_block_ids, cohort.site_ids, strict=True):
        if not np.all(np.asarray(row_site_ids, dtype=object) == ownership[block_id]):
            raise ValueError(f"{label} cohort site_ids do not match block ownership")
    cohort_name = "dkasc" if label == "DKASC" else "oklahoma"
    for block in cohort.blocks:
        source_binding = source_bindings.get(block.site_id)
        if source_binding is None:
            raise ValueError(f"{label} block source binding is missing")
        _require_canonical_block_record(
            block,
            cohort=cohort_name,
            source_binding=source_binding,
        )
        _require_block_row_alignment(
            cohort,
            block,
            label=label,
            source_binding=source_binding,
        )


def _require_confirmation_result_structure(result: ConfirmationCohorts) -> None:
    contract = result.confirmation_contract
    source_bindings = _validated_source_bindings(contract)
    if (
        result.dkasc_contract_site_ids != contract.dkasc_site_ids
        or result.old_cohort_sha256 != contract.old_cohort_sha256
        or result.extraction_sha256 != contract.extraction_sha256
        or result.prior_screen_exclusion_sha256 != contract.prior_screen_coordinates_sha256
    ):
        raise ValueError("confirmation result contract anchor integrity failure")
    _require_cohort_structure(
        result.dkasc,
        label="DKASC",
        expected_block_count=12,
        expected_site_ids=result.dkasc_contract_site_ids,
        source_bindings=source_bindings,
    )
    _require_cohort_structure(
        result.oklahoma,
        label="Oklahoma",
        expected_block_count=15,
        expected_site_ids=EXPECTED_AUXILIARY_OK_SITE_IDS,
        source_bindings=source_bindings,
    )


def _require_selected_source_values(
    cohort: Cohort,
    selected: Sequence[tuple[NormalizedSiteSeries, int, BlockRecord]],
) -> None:
    """Verify materialized arrays while their original source series are still available."""
    if cohort.previous_targets is None or cohort.block_ids is None:
        raise ValueError("selected confirmation cohort is missing source-aligned arrays")
    selected_by_id = {record.block_id: site for site, _, record in selected}
    for block_id, site in selected_by_id.items():
        rows = np.flatnonzero(np.asarray(cohort.block_ids) == block_id)
        source_indices = np.asarray(cohort.source_indices)[rows]
        if np.any(source_indices < 2) or np.any(source_indices >= len(site.power)):
            raise ValueError("selected confirmation source indices escape their source series")
        if not np.array_equal(np.asarray(cohort.targets)[rows], site.power[source_indices]):
            raise ValueError("selected confirmation targets do not match their source series")
        if not np.array_equal(
            np.asarray(cohort.previous_targets)[rows], site.power[source_indices - 2]
        ):
            raise ValueError("selected confirmation prior targets do not match their source series")
        if not np.array_equal(np.asarray(cohort.daylight)[rows], site.daylight[source_indices]):
            raise ValueError("selected confirmation daylight does not match its source series")
        expected_timestamps = site.timestamps.to_numpy()[source_indices]
        observed_timestamps = np.asarray(cohort.timestamps, dtype=object)[rows]
        if any(
            pd.Timestamp(observed) != pd.Timestamp(expected)
            for observed, expected in zip(
                observed_timestamps.ravel(), expected_timestamps.ravel(), strict=True
            )
        ):
            raise ValueError("selected confirmation timestamps do not match their source series")


def _block_payload(block: BlockRecord, *, cohort: str) -> dict[str, Any]:
    return {
        "block_id": block.block_id,
        "cohort": cohort,
        "dataset": block.dataset,
        "end": block.end.isoformat(),
        "n_steps": block.n_steps,
        "quarter": block.quarter,
        "raw_hashes": sorted(block.raw_hashes),
        "selection_hash": block.selection_hash,
        "site_id": block.site_id,
        "spatial_role": block.spatial_role,
        "split": block.split,
        "start": block.start.isoformat(),
        "state": block.state,
    }


def _protocol_core(result: ConfirmationCohorts) -> dict[str, Any]:
    blocks = [
        *(_block_payload(block, cohort="dkasc") for block in result.dkasc.blocks),
        *(_block_payload(block, cohort="oklahoma") for block in result.oklahoma.blocks),
    ]
    return {
        "auxiliary_confirmation_only": result.auxiliary_confirmation_only,
        "blocks": sorted(blocks, key=lambda item: (item["cohort"], item["site_id"], item["start"])),
        "dkasc_cohort_sha256": result.dkasc_cohort_sha256,
        "dkasc_contract_site_ids": list(result.dkasc_contract_site_ids),
        "dkasc_site_ids": sorted({block.site_id for block in result.dkasc.blocks}),
        "extraction_sha256": result.extraction_sha256,
        "oklahoma_cohort_sha256": result.oklahoma_cohort_sha256,
        "oklahoma_site_ids": sorted({block.site_id for block in result.oklahoma.blocks}),
        "old_cohort_sha256": result.old_cohort_sha256,
        "prior_screen_exclusion_sha256": result.prior_screen_exclusion_sha256,
        "schema": "nested_confirmation_protocol_v1",
        "selection_seed": result.selection_seed,
        "source_series": _source_binding_rows(result.confirmation_contract.source_series),
    }


def build_confirmation_cohorts(
    inputs: ConfirmationInputs,
    *,
    selection_seed: int = CONFIRMATION_SELECTION_SEED,
    screening_callback: Callable[[dict[str, Any]], Any] | None = None,
) -> ConfirmationCohorts:
    """Build the 12+15 isolated confirmation blocks and seal their protocol."""
    if selection_seed != CONFIRMATION_SELECTION_SEED:
        raise ValueError("confirmation selection seed is fixed")
    if len(inputs.dkasc_sites) != 12 or len({site.site_id for site in inputs.dkasc_sites}) != 12:
        raise ValueError("confirmation requires exactly twelve distinct DKASC sites")
    if (
        inputs.confirmation_contract is not None
        and tuple(sorted(site.site_id for site in inputs.dkasc_sites))
        != inputs.confirmation_contract.dkasc_site_ids
    ):
        raise ValueError("confirmation DKASC identity does not match old-cohort contract")
    _require_bound_source_hashes(inputs)
    if tuple(sorted(site.site_id for site in inputs.oklahoma_sites)) != EXPECTED_AUXILIARY_OK_SITE_IDS:
        raise ValueError("confirmation requires the exact five auxiliary Oklahoma sites")
    if tuple(source.site_id for source in sorted(inputs.extraction_sources, key=lambda item: item.site_id)) != (
        EXPECTED_AUXILIARY_OK_SITE_IDS
    ):
        raise ValueError("auxiliary extraction sources do not match confirmation sites")
    embargo = pd.Timedelta(days=CONFIRMATION_EMBARGO_DAYS)
    dkasc_selected: list[tuple[NormalizedSiteSeries, int, BlockRecord]] = []
    for site in sorted(inputs.dkasc_sites, key=lambda item: item.site_id):
        _require_validation_role(site, auxiliary=False)
        if site.split != "validation":
            raise ValueError("DKASC confirmation series must be validation split")
        candidates: list[tuple[int, BlockRecord]] = []
        for start, block in _valid_blocks(site):
            if _coordinate(block) in inputs.prior_screen_coordinates:
                continue
            if any(
                block.site_id == old.site_id and _interval_gap(block, old) < embargo
                for old in inputs.old_selected_blocks
            ):
                continue
            candidates.append((start, block))
        if not candidates:
            raise ValueError(f"DKASC site {site.site_id!r} has no embargo-eligible block")
        start, block = min(candidates, key=lambda item: _confirmation_rank(site.site_id, item[1]))
        dkasc_selected.append((site, start, replace(block, selection_hash=_confirmation_rank(site.site_id, block))))
    _require_dkasc_selection_isolation(
        dkasc_selected,
        old_blocks=inputs.old_selected_blocks,
        prior_coordinates=inputs.prior_screen_coordinates,
    )
    oklahoma_selected: list[tuple[NormalizedSiteSeries, int, BlockRecord]] = []
    for site in sorted(inputs.oklahoma_sites, key=lambda item: item.site_id):
        _require_validation_role(site, auxiliary=True)
        if site.split != "validation":
            raise ValueError("auxiliary confirmation series must be validation split")
        candidates = _valid_blocks(site)
        if len(candidates) != 3:
            raise ValueError(f"auxiliary site {site.site_id!r} requires exactly three complete blocks")
        for start, block in candidates:
            if _coordinate(block) in inputs.prior_screen_coordinates:
                raise ValueError("confirmation coordinate overlap with prior screening")
            oklahoma_selected.append((site, start, block))
    result = ConfirmationCohorts(
        dkasc=_cohort_from_selected(dkasc_selected),
        oklahoma=_cohort_from_selected(oklahoma_selected),
        old_cohort_sha256=inputs.old_cohort_sha256,
        extraction_sha256=inputs.extraction_sha256,
        prior_screen_exclusion_sha256=_sha256_json(
            [list(item) for item in sorted(inputs.prior_screen_coordinates)]
        ),
        selection_seed=selection_seed,
        auxiliary_confirmation_only=True,
        dkasc_contract_site_ids=inputs.confirmation_contract.dkasc_site_ids,
        dkasc_cohort_sha256="",
        oklahoma_cohort_sha256="",
        protocol_sha256="",
        confirmation_contract=inputs.confirmation_contract,
    )
    _require_selected_source_values(result.dkasc, dkasc_selected)
    _require_selected_source_values(result.oklahoma, oklahoma_selected)
    result = replace(
        result,
        dkasc_cohort_sha256=_cohort_integrity_sha256(result.dkasc),
        oklahoma_cohort_sha256=_cohort_integrity_sha256(result.oklahoma),
    )
    result = replace(result, protocol_sha256=_sha256_json(_protocol_core(result)))
    # This is the sole callback boundary. The result is not returned until all
    # provenance, mutable-array, and selected-block checks have completed.
    payload = confirmation_protocol_payload(
        result,
        confirmation_contract=inputs.confirmation_contract,
    )
    if screening_callback is not None:
        screening_callback(payload)
    return result


def confirmation_protocol_payload(
    result: ConfirmationCohorts,
    *,
    confirmation_contract: ConfirmationContract,
) -> dict[str, Any]:
    """Return a canonical, integrity-checked protocol payload without side effects."""
    if result.confirmation_contract != confirmation_contract:
        raise ValueError("confirmation external contract anchor integrity failure")
    _require_selected_block_isolation(result.dkasc)
    _require_selected_block_isolation(result.oklahoma)
    _require_confirmation_result_structure(result)
    if (
        _cohort_integrity_sha256(result.dkasc) != result.dkasc_cohort_sha256
        or _cohort_integrity_sha256(result.oklahoma) != result.oklahoma_cohort_sha256
    ):
        raise ValueError("confirmation cohort integrity failure")
    payload = _protocol_core(result)
    if not _valid_sha256(result.protocol_sha256) or _sha256_json(payload) != result.protocol_sha256:
        raise ValueError("confirmation protocol integrity failure")
    return {**payload, "protocol_sha256": result.protocol_sha256}


def confirmation_block_csv_bytes(
    result: ConfirmationCohorts,
    *,
    confirmation_contract: ConfirmationContract,
    root: str | Path | None = None,
) -> bytes:
    """Serialize selected blocks without embedding a filesystem-dependent root."""
    del root
    payload = confirmation_protocol_payload(
        result,
        confirmation_contract=confirmation_contract,
    )
    columns = (
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
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in payload["blocks"]:
        writer.writerow({**row, "raw_hashes": ",".join(row["raw_hashes"])})
    return output.getvalue().encode("utf-8")


def auxiliary_ok_entries(
    manifest: Mapping[str, Any], audit: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    """Return the immutable five-site Oklahoma confirmation-source contract."""
    try:
        selected = set(audit["selected_site_ids_by_state"]["OK"])
        archive_sha256 = manifest["provenance"]["archive_sha256"]["ok-pv-2006.zip"]
        entries = manifest["entries"]
    except (KeyError, TypeError) as exc:
        raise ValueError("auxiliary Oklahoma source inputs are incomplete") from exc
    if archive_sha256 != EXPECTED_AUXILIARY_OK_ARCHIVE_SHA256:
        raise ValueError("auxiliary Oklahoma archive contract changed")

    rows = tuple(
        sorted(
            (
                {**row, "archive_sha256": archive_sha256}
                for row in entries
                if row["state"] == "OK"
                and row["spatial_role"] == "validation_calibration"
                and row["site_id"] not in selected
            ),
            key=lambda row: row["site_id"],
        )
    )
    if tuple(row["site_id"] for row in rows) != EXPECTED_AUXILIARY_OK_SITE_IDS:
        raise ValueError("auxiliary Oklahoma site contract changed")
    return rows


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK


def _is_zip_regular(info: zipfile.ZipInfo) -> bool:
    mode_type = stat.S_IFMT(info.external_attr >> 16)
    dos_directory = info.create_system == 0 and bool(info.external_attr & 0x10)
    return (
        not info.is_dir()
        and not dos_directory
        and not _is_zip_symlink(info)
        and mode_type in {0, stat.S_IFREG}
    )


def _entry_member(entry: Mapping[str, Any]) -> str:
    files = entry.get("files")
    if isinstance(files, (str, bytes)) or not isinstance(files, Sequence) or len(files) != 1:
        raise ValueError("auxiliary source entry must declare exactly one archive member")
    member = files[0]
    if not isinstance(member, str):
        raise ValueError("auxiliary source member must be text")
    member_path = PurePosixPath(member)
    if (
        not member
        or member_path.is_absolute()
        or member_path.name in {"", ".", ".."}
        or ".." in member_path.parts
    ):
        raise ValueError("auxiliary source member escapes destination root")
    return member


def _validated_entries(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, str, str, str], ...]:
    if isinstance(entries, (str, bytes)) or len(entries) != len(EXPECTED_AUXILIARY_OK_SITE_IDS):
        raise ValueError("auxiliary Oklahoma extraction requires the exact five-site contract")
    prepared: list[tuple[str, str, str, str]] = []
    archive_hashes: set[str] = set()
    basenames: set[str] = set()
    for entry in entries:
        site_id = entry.get("site_id")
        archive_sha256 = entry.get("archive_sha256")
        if not isinstance(site_id, str) or not isinstance(archive_sha256, str):
            raise ValueError("auxiliary source entry is missing its identity or archive hash")
        if len(archive_sha256) != 64 or any(char not in "0123456789abcdef" for char in archive_sha256):
            raise ValueError("auxiliary source archive hash is invalid")
        member = _entry_member(entry)
        basename = PurePosixPath(member).name
        if basename in basenames:
            raise ValueError("auxiliary source members have duplicate basenames")
        basenames.add(basename)
        archive_hashes.add(archive_sha256)
        prepared.append((site_id, member, basename, archive_sha256))
    if tuple(site_id for site_id, _, _, _ in prepared) != EXPECTED_AUXILIARY_OK_SITE_IDS:
        raise ValueError("auxiliary Oklahoma extraction site contract changed")
    if len(archive_hashes) != 1:
        raise ValueError("auxiliary Oklahoma sources disagree on archive hash")
    return tuple(prepared)


def _destination_for(destination_root: Path, member: str) -> Path:
    root = destination_root.resolve()
    destination = root.joinpath(*PurePosixPath(member).parts)
    if not destination.is_relative_to(root):
        raise ValueError("auxiliary source member escapes destination root")
    if os.path.lexists(destination):
        raise ValueError("auxiliary source destination already exists")
    parent = root
    for component in destination.relative_to(root).parts[:-1]:
        parent /= component
        if not os.path.lexists(parent):
            continue
        mode = parent.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ValueError("auxiliary source destination parent is unsafe")
    return destination


def extract_auxiliary_ok_sources(
    entries: Sequence[Mapping[str, Any]], archive_path: str | Path, destination_root: str | Path
) -> tuple[ExtractedSource, ...]:
    """Extract the contract's CSVs without publishing any until every member validates."""
    prepared = _validated_entries(entries)
    archive = Path(archive_path)
    if not archive.is_file():
        raise ValueError("auxiliary Oklahoma archive is unavailable")
    archive_sha256 = _sha256_path(archive)
    expected_archive_sha256 = prepared[0][3]
    if archive_sha256 != expected_archive_sha256:
        raise ValueError("auxiliary Oklahoma archive hash changed")

    root = Path(destination_root)
    destinations = tuple(_destination_for(root, member) for _, member, _, _ in prepared)
    if len(set(destinations)) != len(destinations):
        raise ValueError("auxiliary source destinations are not unique")

    temporary_paths: list[Path] = []
    staged: list[tuple[str, str, str, Path, Path]] = []
    published_paths: list[Path] = []
    try:
        with zipfile.ZipFile(archive) as source_archive:
            infos_by_name: dict[str, list[zipfile.ZipInfo]] = {}
            for info in source_archive.infolist():
                infos_by_name.setdefault(info.filename, []).append(info)
            for (site_id, member, archive_member, _), destination in zip(
                prepared, destinations, strict=True
            ):
                infos = infos_by_name.get(archive_member, [])
                if len(infos) != 1:
                    raise ValueError("auxiliary source member is missing or duplicated")
                info = infos[0]
                if not _is_zip_regular(info):
                    raise ValueError("auxiliary source member is not a regular file")
                destination.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
                )
                temporary = Path(temporary_name)
                temporary_paths.append(temporary)
                digest = hashlib.sha256()
                with os.fdopen(descriptor, "wb") as output, source_archive.open(info) as member_file:
                    for chunk in iter(lambda: member_file.read(1024 * 1024), b""):
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                member_sha256 = digest.hexdigest()
                if _sha256_path(temporary) != member_sha256:
                    raise ValueError("auxiliary source member hash did not persist")
                staged.append((site_id, member, member_sha256, temporary, destination))

        for _, _, _, temporary, destination in staged:
            published_paths.append(destination)
            os.replace(temporary, destination)
            temporary_paths.remove(temporary)
        extracted: list[ExtractedSource] = []
        for site_id, member, member_sha256, _, destination in staged:
            output_sha256 = _sha256_path(destination)
            if output_sha256 != member_sha256:
                raise ValueError("auxiliary source output hash did not persist")
            extracted.append(
                ExtractedSource(
                    site_id=site_id,
                    archive_sha256=archive_sha256,
                    member=member,
                    member_sha256=member_sha256,
                    output_path=str(destination),
                    output_sha256=output_sha256,
                )
            )
        return tuple(extracted)
    except Exception:
        for destination in published_paths:
            destination.unlink(missing_ok=True)
        raise
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
