"""Strict in-memory source selection and resampling for one complete split."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .adapters import BasePVAdapter, ManifestEntry
from .origin_clock import GRID_STEP_NS, validate_source_time_contract
from .preprocessing import (
    ResampledPVFrame,
    interpolate_resampled_partitions,
    resample_to_15min,
)
from .scoped_prepared import row_scope_from_plan

_INT64_INFO = np.iinfo(np.int64)


def _validated_source_and_scope(
    entry: ManifestEntry,
    *,
    frozen_split_contract: dict,
    selected_split: str,
    source_time_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(entry, ManifestEntry):
        raise TypeError("entry must be a ManifestEntry")
    source = validate_source_time_contract(source_time_contract)
    entry_source = validate_source_time_contract(entry.source_time_contract)
    if source != entry_source:
        raise ValueError("source time contract differs from manifest entry")
    if source["timezone"] != entry.timezone:
        raise ValueError("source timestamp timezone differs from manifest timezone")
    scope = row_scope_from_plan(
        frozen_split_contract,
        selected_split=selected_split,
    )
    grid_start_ns = frozen_split_contract["grid_start_ns"]
    if grid_start_ns % GRID_STEP_NS:
        raise ValueError("split plan grid_start_ns must align to the 15-minute UTC grid")
    return source, scope


def scoped_source_positions(
    entry: ManifestEntry,
    timestamp_values,
    *,
    frozen_split_contract: dict,
    selected_split: str,
    source_time_contract: Mapping[str, Any],
) -> np.ndarray:
    """Return original positions whose represented starts lie in one split."""
    source, scope = _validated_source_and_scope(
        entry,
        frozen_split_contract=frozen_split_contract,
        selected_split=selected_split,
        source_time_contract=source_time_contract,
    )
    index = BasePVAdapter._strict_source_index(timestamp_values, entry, source)
    starts = index.asi8
    selected = (starts >= scope["stored_start_ns"]) & (starts < scope["stored_end_exclusive_ns"])
    positions = np.flatnonzero(selected).astype(np.int64, copy=True)
    positions.setflags(write=False)
    return positions


def _selected_labels(scope: dict[str, Any]) -> np.ndarray:
    row_count = int(scope["global_stop_row"]) - int(scope["global_start_row"])
    start_ns = int(scope["stored_start_ns"])
    end_exclusive_ns = int(scope["stored_end_exclusive_ns"])
    last_ns = start_ns + (row_count - 1) * GRID_STEP_NS
    if last_ns < _INT64_INFO.min or last_ns > _INT64_INFO.max:
        raise OverflowError("selected split label exceeds int64")
    if last_ns + GRID_STEP_NS != end_exclusive_ns:
        raise ValueError("selected split timestamps disagree with row scope")
    return np.fromiter(
        (start_ns + offset * GRID_STEP_NS for offset in range(row_count)),
        dtype=np.int64,
        count=row_count,
    )


def resample_scoped_source_frame(
    adapter: BasePVAdapter,
    entry: ManifestEntry,
    source_frame: pd.DataFrame,
    *,
    frozen_split_contract: dict,
    selected_split: str,
    source_time_contract: Mapping[str, Any],
) -> ResampledPVFrame:
    """Canonicalize and resample an already-selected provider-format frame."""
    if not isinstance(adapter, BasePVAdapter):
        raise TypeError("adapter must be a BasePVAdapter")
    if not isinstance(entry, ManifestEntry):
        raise TypeError("entry must be a ManifestEntry")
    if not isinstance(source_frame, pd.DataFrame):
        raise TypeError("source_frame must be a pandas DataFrame")
    if adapter.dataset_key != entry.dataset:
        raise ValueError("adapter dataset does not match source")

    source, scope = _validated_source_and_scope(
        entry,
        frozen_split_contract=frozen_split_contract,
        selected_split=selected_split,
        source_time_contract=source_time_contract,
    )
    if not source_frame.columns.is_unique:
        raise ValueError("source_frame column names must be unique")
    if entry.timestamp_column not in source_frame.columns:
        raise KeyError(f"Timestamp column {entry.timestamp_column!r} is missing")

    positions = scoped_source_positions(
        entry,
        source_frame[entry.timestamp_column],
        frozen_split_contract=frozen_split_contract,
        selected_split=selected_split,
        source_time_contract=source,
    )
    if positions.size != len(source_frame):
        raise ValueError("source_frame contains records outside the selected split")
    if source_frame.empty:
        raise ValueError("source_frame contains no records for the selected split")

    canonical = adapter.canonicalize_frame(
        entry,
        source_frame,
        require_origin_safe=True,
        source=source,
    )
    canonical_starts = canonical.index.as_unit("ns").asi8
    if np.any(canonical_starts < scope["stored_start_ns"]) or np.any(
        canonical_starts >= scope["stored_end_exclusive_ns"]
    ):
        raise ValueError("canonical source records escaped the selected split")

    regular = resample_to_15min(canonical, max_interpolation_gap="0min")
    regular.values.attrs.update(canonical.attrs)
    selected_labels_ns = _selected_labels(scope)
    grid = pd.to_datetime(selected_labels_ns, utc=True).rename("timestamp")
    regular = ResampledPVFrame(
        regular.values.reindex(grid),
        regular.observed_mask.reindex(grid, fill_value=False),
        regular.imputed_mask.reindex(grid, fill_value=False),
    )
    regular.values.attrs.update(canonical.attrs)
    result = interpolate_resampled_partitions(
        regular,
        {selected_split: slice(0, len(selected_labels_ns))},
    )
    return result
