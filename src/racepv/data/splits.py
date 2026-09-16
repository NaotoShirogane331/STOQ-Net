"""Leakage-safe chronological data splitting utilities.

The split itself is deliberately expressed in integer positions.  This keeps the
boundary independent of daylight filtering and prevents a later preprocessing
step from silently moving observations between partitions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar, overload

import numpy as np
import pandas as pd

from .origin_clock import GRID_STEP_NS, freeze_split_plan

SPLIT_NAMES = ("train", "validation", "calibration", "test")
DEFAULT_SPLIT_RATIOS = (0.60, 0.15, 0.10, 0.15)


def frozen_split_view(contract: Mapping[str, object]) -> tuple[np.ndarray, dict[str, slice]]:
    """Validate an absolute plan and return its grid and slices without refitting boundaries."""
    import json

    if not isinstance(contract, Mapping) or set(contract) != {
        "grid_start_ns",
        "grid_step_ns",
        "row_count",
        "split_order",
        "splits",
    }:
        raise ValueError("frozen split contract requires exact fields")
    for key in ("grid_start_ns", "grid_step_ns", "row_count"):
        if type(contract[key]) is not int:
            raise TypeError(f"frozen split {key} must be an integer")
    start, rows = contract["grid_start_ns"], contract["row_count"]
    if rows < 1 or contract["grid_step_ns"] != GRID_STEP_NS:
        raise ValueError("frozen split grid must have positive rows and 15-minute steps")
    if start <= np.iinfo(np.int64).min or start + rows * GRID_STEP_NS > np.iinfo(np.int64).max:
        raise ValueError("frozen split timestamps exceed int64")
    if contract["split_order"] != list(SPLIT_NAMES):
        raise ValueError("frozen split order must be train/validation/calibration/test")
    if not isinstance(contract["splits"], Mapping) or set(contract["splits"]) != set(SPLIT_NAMES):
        raise ValueError("frozen split names disagree")
    slices = {}
    for name in SPLIT_NAMES:
        bounds = contract["splits"][name]
        if not isinstance(bounds, Mapping) or set(bounds) != {
            "start_row",
            "stop_row",
            "start_ns",
            "end_exclusive_ns",
        }:
            raise ValueError("frozen split bounds require exact fields")
        if any(type(value) is not int for value in bounds.values()):
            raise TypeError("frozen split bounds must be integers")
        slices[name] = slice(bounds["start_row"], bounds["stop_row"])
    labels = start + np.arange(rows, dtype=np.int64) * GRID_STEP_NS
    expected = freeze_split_plan(labels, slices)
    if json.dumps(expected, sort_keys=True) != json.dumps(dict(contract), sort_keys=True):
        raise ValueError("frozen split row/time identities disagree")
    return labels, slices


FrameT = TypeVar("FrameT", pd.DataFrame, pd.Series)


@dataclass(frozen=True)
class TemporalSplitPlan:
    """Half-open positional intervals for chronological partitions."""

    n_observations: int
    slices: Mapping[str, slice]
    ratios: tuple[float, float, float, float]

    def __getitem__(self, name: str) -> slice:
        return self.slices[name]

    def window_starts(
        self,
        name: str,
        lookback: int,
        horizon: int,
        stride: int = 1,
    ) -> np.ndarray:
        """Return starts whose complete history and target stay in one split."""

        if lookback <= 0 or horizon <= 0 or stride <= 0:
            raise ValueError("lookback, horizon and stride must be positive")
        boundary = self.slices[name]
        start = int(boundary.start or 0)
        stop = int(boundary.stop or self.n_observations)
        final_start = stop - lookback - horizon
        if final_start < start:
            return np.empty(0, dtype=np.int64)
        return np.arange(start, final_start + 1, stride, dtype=np.int64)

    def assert_window_contained(self, name: str, start: int, lookback: int, horizon: int) -> None:
        boundary = self.slices[name]
        split_start = int(boundary.start or 0)
        split_stop = int(boundary.stop or self.n_observations)
        if start < split_start or start + lookback + horizon > split_stop:
            raise ValueError(
                f"Window [{start}, {start + lookback + horizon}) crosses the "
                f"{name} split [{split_start}, {split_stop})."
            )


def _validated_ratios(ratios: Sequence[float]) -> tuple[float, float, float, float]:
    if len(ratios) != 4:
        raise ValueError("Exactly four ratios are required: train/validation/calibration/test")
    values = tuple(float(value) for value in ratios)
    if any(value <= 0 for value in values):
        raise ValueError("All split ratios must be positive")
    if not np.isclose(sum(values), 1.0, atol=1e-10):
        raise ValueError(f"Split ratios must sum to 1.0; received {sum(values):.8f}")
    return values  # type: ignore[return-value]


def chronological_split_plan(
    n_observations: int,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
) -> TemporalSplitPlan:
    """Create the fixed 60/15/10/15 (by default) chronological split plan."""

    if n_observations < 4:
        raise ValueError("At least four observations are needed for four non-empty splits")
    ratio_tuple = _validated_ratios(ratios)
    cumulative = np.cumsum((0.0, *ratio_tuple))
    boundaries = np.floor(cumulative * n_observations).astype(int)
    boundaries[0] = 0
    boundaries[-1] = n_observations
    if np.any(np.diff(boundaries) <= 0):
        raise ValueError("The series is too short for all requested split ratios to be non-empty")
    slices = {
        name: slice(int(boundaries[i]), int(boundaries[i + 1]))
        for i, name in enumerate(SPLIT_NAMES)
    }
    return TemporalSplitPlan(n_observations, slices, ratio_tuple)


@overload
def chronological_split(
    data: FrameT,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    *,
    copy: bool = True,
) -> dict[str, FrameT]: ...


@overload
def chronological_split(
    data: pd.Index,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    *,
    copy: bool = True,
) -> dict[str, pd.Index]: ...


def chronological_split(
    data: FrameT | pd.Index,
    ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    *,
    copy: bool = True,
) -> dict[str, FrameT | pd.Index]:
    """Split a pandas object without shuffling.

    Window creation must happen *after* this function (or use
    :class:`TemporalSplitPlan.window_starts` on the full frame).  Doing so ensures
    neither history nor targets can cross a partition boundary.
    """

    plan = chronological_split_plan(len(data), ratios)
    result: dict[str, FrameT | pd.Index] = {}
    for name, boundary in plan.slices.items():
        if isinstance(data, pd.Index):
            value = data[boundary]
            result[name] = value.copy() if copy else value
        else:
            value = data.iloc[boundary]
            result[name] = value.copy() if copy else value
    return result
