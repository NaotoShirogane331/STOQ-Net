"""Deterministic, auditable common cohorts made from complete 14-day blocks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import pandas as pd

from .constants import HORIZON, STEP_MINUTES

BLOCK_STEPS = 14 * 24 * 60 // STEP_MINUTES


def stable_hash(*parts: object) -> str:
    return sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NormalizedSiteSeries:
    site_id: str
    dataset: str
    timestamps: pd.DatetimeIndex
    power: np.ndarray
    daylight: np.ndarray
    split: str = "test"
    raw_hashes: tuple[str, ...] = ()
    spatial_role: str | None = None
    state: str | None = None

    def __post_init__(self) -> None:
        index = pd.DatetimeIndex(self.timestamps)
        values = np.asarray(self.power, dtype=float)
        day = np.asarray(self.daylight, dtype=bool)
        if index.tz is None:
            raise ValueError("site timestamps must be timezone-aware")
        if len(index) != values.size or values.size != day.size:
            raise ValueError("timestamps, power, and daylight must have equal length")
        if not index.is_monotonic_increasing or index.has_duplicates:
            raise ValueError("site timestamps must be strictly increasing")
        object.__setattr__(self, "timestamps", index)
        object.__setattr__(self, "power", values)
        object.__setattr__(self, "daylight", day)


@dataclass(frozen=True)
class BlockRecord:
    block_id: str
    site_id: str
    dataset: str
    split: str
    start: pd.Timestamp
    end: pd.Timestamp
    quarter: str
    n_steps: int
    selection_hash: str
    raw_hashes: tuple[str, ...]
    spatial_role: str | None = None
    state: str | None = None


@dataclass(frozen=True)
class Cohort:
    split: str
    blocks: tuple[BlockRecord, ...]
    targets: np.ndarray
    daylight: np.ndarray
    timestamps: np.ndarray
    day_ids: np.ndarray
    site_ids: np.ndarray
    source_indices: np.ndarray
    previous_targets: np.ndarray | None = None
    block_ids: np.ndarray | None = None


def _valid_blocks(site: NormalizedSiteSeries) -> list[tuple[int, BlockRecord]]:
    expected = pd.Timedelta(minutes=STEP_MINUTES)
    output: list[tuple[int, BlockRecord]] = []
    # Blocks are non-overlapping by construction, which avoids duplicate target
    # days in the common cohort and makes each selected record easy to audit.
    boundaries = np.flatnonzero(site.timestamps[1:] - site.timestamps[:-1] != expected) + 1
    run_starts = np.r_[0, boundaries]
    run_stops = np.r_[boundaries, len(site.timestamps)]
    for run_start, run_stop in zip(run_starts, run_stops, strict=True):
        for start_index in range(int(run_start), int(run_stop) - BLOCK_STEPS + 1, BLOCK_STEPS):
            end_index = start_index + BLOCK_STEPS
            index = site.timestamps[start_index:end_index]
            power = site.power[start_index:end_index]
            if len(index) != BLOCK_STEPS or not np.all(index[1:] - index[:-1] == expected):
                continue
            if not np.all(np.isfinite(power)):
                continue
            start, end = index[0], index[-1]
            quarter = f"{start.year}Q{start.quarter}"
            digest = stable_hash(
                site.dataset, site.site_id, site.split, start.isoformat(), end.isoformat()
            )
            output.append(
                (
                    start_index,
                    BlockRecord(
                        block_id=digest[:20],
                        site_id=site.site_id,
                        dataset=site.dataset,
                        split=site.split,
                        start=start,
                        end=end,
                        quarter=quarter,
                        n_steps=BLOCK_STEPS,
                        selection_hash=digest,
                        raw_hashes=site.raw_hashes,
                        spatial_role=site.spatial_role,
                        state=site.state,
                    ),
                )
            )
    return output


def select_cohort(
    sites: Iterable[NormalizedSiteSeries],
    *,
    split: str,
    max_blocks_per_site: int | None = None,
) -> Cohort:
    """Select identical model-independent complete blocks, quarter-stratified when possible."""
    if split not in {"train", "validation", "calibration", "test"}:
        raise ValueError("split must be train, validation, calibration, or test")
    limit = (
        (2 if split == "calibration" else 4)
        if max_blocks_per_site is None
        else int(max_blocks_per_site)
    )
    hard_limit = 2 if split == "calibration" else 4
    if limit < 1 or limit > hard_limit:
        raise ValueError(
            f"{split} cohorts permit at most {hard_limit} complete 14-day blocks per site"
        )
    chosen: list[tuple[NormalizedSiteSeries, int, BlockRecord]] = []
    for site in sorted(tuple(sites), key=lambda item: (item.dataset, item.site_id)):
        if site.split != split:
            raise ValueError(
                f"site {site.site_id!r} belongs to split {site.split!r}, not {split!r}"
            )
        candidates = _valid_blocks(site)
        if not candidates:
            raise ValueError(f"site {site.site_id!r} has no complete 14-day block")
        by_quarter: dict[str, list[tuple[int, BlockRecord]]] = {}
        for candidate in candidates:
            by_quarter.setdefault(candidate[1].quarter, []).append(candidate)
        # First pass guarantees represented quarters where the limit permits;
        # remaining positions use the same stable hash ordering.
        ordered: list[tuple[int, BlockRecord]] = []
        for quarter in sorted(by_quarter):
            ordered.append(min(by_quarter[quarter], key=lambda item: item[1].selection_hash))
        remainder = [item for items in by_quarter.values() for item in items if item not in ordered]
        ordered.extend(sorted(remainder, key=lambda item: item[1].selection_hash))
        chosen.extend((site, index, record) for index, record in ordered[:limit])
    chosen.sort(key=lambda item: (item[2].site_id, item[2].start))
    targets: list[np.ndarray] = []
    daylight: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    days: list[np.ndarray] = []
    site_ids: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    previous: list[np.ndarray] = []
    block_ids: list[np.ndarray] = []
    for site, start, record in chosen:
        rows = BLOCK_STEPS - HORIZON
        offsets = np.arange(rows)[:, None] + np.arange(HORIZON)[None, :]
        target_index = start + offsets
        prior_index = target_index - 2
        # Every public cell must have a finite 30-minute, in-block prior. This
        # removes only origins at the beginning of a selected block and avoids
        # both NaN labels and accidental use of data outside that block.
        keep_rows = np.all(
            (prior_index >= start)
            & np.isfinite(site.power[target_index])
            & np.isfinite(site.power[np.maximum(prior_index, start)]),
            axis=1,
        )
        target_index = target_index[keep_rows]
        prior_index = prior_index[keep_rows]
        targets.append(site.power[target_index])
        daylight.append(site.daylight[target_index])
        times = site.timestamps.to_numpy()[target_index]
        timestamps.append(times)
        days.append(pd.DatetimeIndex(times).tz_localize(None).to_numpy(dtype="datetime64[D]"))
        site_ids.append(np.full(times.shape, site.site_id, dtype=object))
        indices.append(target_index)
        previous.append(site.power[prior_index])
        block_ids.append(np.full(target_index.shape[0], record.block_id, dtype=object))
    return Cohort(
        split,
        tuple(item[2] for item in chosen),
        np.concatenate(targets),
        np.concatenate(daylight),
        np.concatenate(timestamps),
        np.concatenate(days),
        np.concatenate(site_ids),
        np.concatenate(indices),
        np.concatenate(previous),
        np.concatenate(block_ids),
    )


def validate_split_isolation(*cohorts: Cohort) -> None:
    """Reject any held-out/source role inversion or raw interval reuse."""
    from .loader import role_allows_split

    seen: list[tuple[Cohort, BlockRecord]] = []
    for cohort in cohorts:
        if cohort.split not in {"validation", "calibration", "test"}:
            raise ValueError("split isolation accepts validation, calibration, and test cohorts")
        for record in cohort.blocks:
            if record.split != cohort.split:
                raise ValueError("block split does not match cohort split")
            if not role_allows_split(
                record.dataset,
                record.spatial_role or "",
                record.state,
                cohort.split,
                site_id=record.site_id,
            ):
                raise ValueError("block role/state is invalid for its cohort split")
            for other_cohort, other in seen:
                same_source = (record.dataset, record.site_id) == (
                    other.dataset,
                    other.site_id,
                ) or bool(set(record.raw_hashes).intersection(other.raw_hashes))
                overlaps = record.start <= other.end and other.start <= record.end
                if same_source and overlaps and cohort.split != other_cohort.split:
                    raise ValueError("source raw/site interval overlap across held-out cohorts")
            seen.append((cohort, record))
