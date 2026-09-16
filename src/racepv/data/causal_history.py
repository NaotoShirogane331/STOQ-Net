"""Immutable, origin-relative resolution of causal historical features."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from racepv.data.origin_clock import (
    _INT64_INFO,
    _positive_int,
    _validated_labels,
    _validated_starts,
)

_FLOAT32_MAX = np.float64(np.finfo(np.float32).max)
_KNOWN_FEATURES = {
    "power_norm",
    "clear_sky_power_norm",
    "clear_sky_residual",
    "power_diff_1",
    "power_diff_2",
    "rolling_volatility_1h",
    "rolling_volatility_3h",
    "clear_sky_index",
    "power_observed",
    "power_imputed",
    "ghi_z",
    "temperature_z",
    "humidity_z",
    "wind_speed_z",
}
_POWER_DEPENDENCIES = {
    "power_norm": (0,),
    "clear_sky_residual": (0,),
    "power_diff_1": (0, -1),
    "power_diff_2": (0, -1, -2),
    "rolling_volatility_1h": tuple(range(0, -5, -1)),
    "rolling_volatility_3h": tuple(range(0, -13, -1)),
    "clear_sky_index": (0,),
}
_WEATHER_FEATURES = {
    "ghi_z": 0,
    "temperature_z": 1,
    "humidity_z": 2,
    "wind_speed_z": 3,
}


def _typed_array(value: object, *, name: str, dtype: np.dtype, ndim: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    return value


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    copied = np.array(value, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class HistoryBatch:
    values: np.ndarray
    mask: np.ndarray

    def __post_init__(self) -> None:
        values = _typed_array(self.values, name="values", dtype=np.dtype(np.float32), ndim=3)
        mask = _typed_array(self.mask, name="mask", dtype=np.dtype(np.float32), ndim=3)
        if values.shape != mask.shape:
            raise ValueError("values and mask must have identical shapes")
        if not np.isfinite(values).all():
            raise ValueError("values must be finite")
        if not np.all((mask == 0.0) | (mask == 1.0)):
            raise ValueError("mask must contain only zero and one")
        if np.any(values[mask == 0.0] != 0.0):
            raise ValueError("masked values must be zero")
        object.__setattr__(self, "values", np.array(values, copy=True))
        object.__setattr__(self, "mask", np.array(mask, copy=True))


@dataclass(frozen=True)
class AdmissionBatch:
    accepted: np.ndarray
    reasons: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        accepted = _typed_array(self.accepted, name="accepted", dtype=np.dtype(np.bool_), ndim=1)
        if not isinstance(self.reasons, tuple):
            raise TypeError("reasons must be a tuple of tuples")
        if len(self.reasons) != accepted.shape[0]:
            raise ValueError("reasons length must match accepted")
        for row in self.reasons:
            if not isinstance(row, tuple) or not all(isinstance(item, str) for item in row):
                raise TypeError("each reasons row must be a tuple of strings")
        object.__setattr__(self, "accepted", np.array(accepted, copy=True))
        object.__setattr__(self, "reasons", tuple(tuple(row) for row in self.reasons))


def _floating_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.dtype.kind != "f":
        raise TypeError(f"{name} must have a floating dtype")
    if value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    return value


def _feature_names(value: object, width: int) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("feature_names must be a sequence of strings")
    names = tuple(value)
    if len(names) != width:
        raise ValueError("feature_names length must equal base_values width")
    if any(not isinstance(name, str) for name in names):
        raise TypeError("feature_names must contain only strings")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must not contain duplicates")
    unknown = set(names) - _KNOWN_FEATURES
    if unknown:
        raise ValueError(f"unsupported history features: {sorted(unknown)!r}")
    return names


@dataclass(frozen=True, init=False)
class CausalHistoryState:
    labels_ns: np.ndarray
    split_ids: np.ndarray
    power_observed: np.ndarray
    weather_observed: np.ndarray
    clear_sky_norm: np.ndarray
    base_values: np.ndarray
    feature_names: tuple[str, ...]
    gap_left: np.ndarray
    gap_right: np.ndarray
    _effective_power_observed: np.ndarray

    def __init__(
        self,
        *,
        labels_ns: np.ndarray,
        split_ids: np.ndarray,
        power_observed: np.ndarray,
        weather_observed: np.ndarray,
        clear_sky_norm: np.ndarray,
        base_values: np.ndarray,
        feature_names: Sequence[str],
    ) -> None:
        labels = _validated_labels(labels_ns)
        n_rows = labels.shape[0]
        splits = _typed_array(split_ids, name="split_ids", dtype=np.dtype(np.int64), ndim=1)
        declared_power = _typed_array(
            power_observed,
            name="power_observed",
            dtype=np.dtype(np.bool_),
            ndim=1,
        )
        weather = _typed_array(
            weather_observed,
            name="weather_observed",
            dtype=np.dtype(np.bool_),
            ndim=2,
        )
        clear = _floating_array(clear_sky_norm, name="clear_sky_norm", ndim=1)
        base = _floating_array(base_values, name="base_values", ndim=2)
        names = _feature_names(feature_names, base.shape[1])
        if splits.shape != (n_rows,):
            raise ValueError("split_ids shape must equal (N,)")
        if declared_power.shape != (n_rows,):
            raise ValueError("power_observed shape must equal (N,)")
        if weather.shape != (n_rows, 4):
            raise ValueError("weather_observed shape must equal (N, 4)")
        if clear.shape != (n_rows,):
            raise ValueError("clear_sky_norm shape must equal (N,)")
        if base.shape[0] != n_rows:
            raise ValueError("base_values first dimension must equal N")
        if not np.isfinite(clear).all():
            raise ValueError("clear_sky_norm must be finite")

        representable = np.isfinite(base) & (base >= -_FLOAT32_MAX) & (base <= _FLOAT32_MAX)
        retained_base = np.full(base.shape, np.nan, dtype=np.float32)
        retained_base[representable] = base[representable].astype(np.float32, copy=False)
        effective_power = np.array(declared_power, copy=True)
        if "power_norm" in names:
            power_index = names.index("power_norm")
            effective_power &= representable[:, power_index]

        labels_copy = _readonly_copy(labels)
        splits_copy = _readonly_copy(splits)
        declared_power_copy = _readonly_copy(declared_power)
        effective_power_copy = _readonly_copy(effective_power)
        weather_copy = _readonly_copy(weather)
        clear_copy = _readonly_copy(clear)
        base_copy = _readonly_copy(retained_base)
        left, right = self._scan_endpoints(splits_copy, effective_power_copy)
        object.__setattr__(self, "labels_ns", labels_copy)
        object.__setattr__(self, "split_ids", splits_copy)
        object.__setattr__(self, "power_observed", declared_power_copy)
        object.__setattr__(self, "weather_observed", weather_copy)
        object.__setattr__(self, "clear_sky_norm", clear_copy)
        object.__setattr__(self, "base_values", base_copy)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "gap_left", left)
        object.__setattr__(self, "gap_right", right)
        object.__setattr__(self, "_effective_power_observed", effective_power_copy)

    @staticmethod
    def _scan_endpoints(
        split_ids: np.ndarray, observed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n_rows = observed.shape[0]
        left = np.full(n_rows, -1, dtype=np.int64)
        right = np.full(n_rows, -1, dtype=np.int64)
        last_observed = -1
        for row in range(n_rows):
            if row == 0 or split_ids[row] != split_ids[row - 1]:
                last_observed = -1
            if observed[row]:
                last_observed = row
            else:
                left[row] = last_observed
        next_observed = -1
        for row in range(n_rows - 1, -1, -1):
            if row == n_rows - 1 or split_ids[row] != split_ids[row + 1]:
                next_observed = -1
            if observed[row]:
                next_observed = row
            else:
                right[row] = next_observed
        left.setflags(write=False)
        right.setflags(write=False)
        return left, right

    @classmethod
    def from_arrays(
        cls,
        *,
        labels_ns: np.ndarray,
        split_ids: np.ndarray,
        power_observed: np.ndarray,
        weather_observed: np.ndarray,
        clear_sky_norm: np.ndarray,
        base_values: np.ndarray,
        feature_names: Sequence[str],
    ) -> CausalHistoryState:
        return cls(
            labels_ns=labels_ns,
            split_ids=split_ids,
            power_observed=power_observed,
            weather_observed=weather_observed,
            clear_sky_norm=clear_sky_norm,
            base_values=base_values,
            feature_names=feature_names,
        )

    def _history_rows(self, starts: object, lookback: object) -> tuple[np.ndarray, np.ndarray]:
        lookback_value = _positive_int(lookback, "lookback")
        starts_array = _validated_starts(starts)
        for start in starts_array.tolist():
            if start > _INT64_INFO.max - lookback_value:
                raise OverflowError("history row arithmetic exceeds int64")
            if start < 0 or start + lookback_value > self.labels_ns.shape[0]:
                raise IndexError("history window start is out of bounds")
        rows = starts_array[:, None] + np.arange(lookback_value, dtype=np.int64)
        origins = starts_array + np.int64(lookback_value)
        return rows, origins

    def _closed_short(self, rows: np.ndarray, origins: np.ndarray) -> np.ndarray:
        left = self.gap_left[rows]
        right = self.gap_right[rows]
        endpoints_present = (left >= 0) & (right >= 0)
        safe_left = np.where(endpoints_present, left, 0)
        safe_right = np.where(endpoints_present, right, 0)
        current_split = self.split_ids[rows]
        same_split = (
            (self.split_ids[safe_left] == current_split)
            & (self.split_ids[safe_right] == current_split)
            & self._effective_power_observed[safe_left]
            & self._effective_power_observed[safe_right]
        )
        return (
            (~self._effective_power_observed[rows])
            & endpoints_present
            & same_split
            & (right < origins[:, None])
            & ((right - left - 1) <= 2)
        )

    def _power_dependency_mask(self, rows: np.ndarray, offsets: tuple[int, ...]) -> np.ndarray:
        valid = np.ones(rows.shape, dtype=bool)
        current_split = self.split_ids[rows]
        for offset in offsets:
            dependency = rows + np.int64(offset)
            present = dependency >= 0
            safe_dependency = np.where(present, dependency, 0)
            valid &= (
                present
                & (self.split_ids[safe_dependency] == current_split)
                & self._effective_power_observed[safe_dependency]
            )
        return valid

    def resolve_history(self, starts: object, *, lookback: object) -> HistoryBatch:
        rows, origins = self._history_rows(starts, lookback)
        shape = (*rows.shape, len(self.feature_names))
        values = np.zeros(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.float32)
        closed_short = self._closed_short(rows, origins)

        for feature_index, name in enumerate(self.feature_names):
            if name == "power_observed":
                values[..., feature_index] = self._effective_power_observed[rows]
                mask[..., feature_index] = 1.0
                continue
            if name == "power_imputed":
                values[..., feature_index] = closed_short
                mask[..., feature_index] = 1.0
                continue

            numeric = self.base_values[rows, feature_index]
            available = np.isfinite(numeric)
            if name in _POWER_DEPENDENCIES:
                available &= self._power_dependency_mask(rows, _POWER_DEPENDENCIES[name])
            elif name == "clear_sky_power_norm":
                available &= np.isfinite(self.clear_sky_norm[rows])
            else:
                available &= self.weather_observed[rows, _WEATHER_FEATURES[name]]
            values[..., feature_index] = np.where(available, numeric, 0.0)
            mask[..., feature_index] = available

        return HistoryBatch(values=values, mask=mask)

    def history_admissible(self, starts: object, *, lookback: object) -> AdmissionBatch:
        rows, origins = self._history_rows(starts, lookback)
        closed_short = self._closed_short(rows, origins)
        night = self.clear_sky_norm[rows] < 0.01
        unresolved = (~self._effective_power_observed[rows]) & (~night) & (~closed_short)
        accepted = ~np.any(unresolved, axis=1)
        reasons = tuple(
            () if item else ("unresolved_daylight_power",) for item in accepted.tolist()
        )
        return AdmissionBatch(accepted=accepted, reasons=reasons)
