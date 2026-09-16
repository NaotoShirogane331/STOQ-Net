"""Preprocessing primitives with explicit missingness and train-only statistics."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TARGET_FREQUENCY = "15min"
MAX_INTERPOLATION_GAP = pd.Timedelta("30min")


@dataclass(frozen=True)
class ResampledPVFrame:
    """A regular frame and masks describing its provenance.

    ``observed_mask`` is one only when the corresponding 15-minute bin contains
    a real observation. ``imputed_mask`` identifies the short gaps filled by
    interpolation.  Both are retained so an imputed value is never mistaken for
    a measured one.
    """

    values: pd.DataFrame
    observed_mask: pd.DataFrame
    imputed_mask: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        # Convenient and backwards-friendly: ``values, mask = result``.
        yield self.values
        yield self.observed_mask


def _ordered_partition_bounds(
    partitions: Mapping[str, slice], n_rows: int
) -> list[tuple[str, int, int]]:
    ordered = [
        (
            str(name),
            int(boundary.start or 0),
            int(boundary.stop if boundary.stop is not None else n_rows),
        )
        for name, boundary in partitions.items()
    ]
    ordered.sort(key=lambda item: (item[1], item[2], item[0]))
    cursor = 0
    for name, start, stop in ordered:
        if start != cursor or not start < stop <= n_rows:
            raise ValueError(
                f"Temporal partitions must cover [0,{n_rows}) exactly; "
                f"{name!r} starts/stops at [{start},{stop}) after cursor {cursor}"
            )
        cursor = stop
    if cursor != n_rows:
        raise ValueError(
            f"Temporal partitions stop at {cursor}, leaving {n_rows - cursor} rows uncovered"
        )
    return ordered


def _as_datetime_index(frame: pd.DataFrame, timestamp_column: str | None) -> pd.DataFrame:
    data = frame.copy()
    if timestamp_column is not None:
        if timestamp_column not in data:
            raise KeyError(f"Timestamp column {timestamp_column!r} is missing")
        index = pd.to_datetime(data.pop(timestamp_column), errors="raise")
        data.index = pd.DatetimeIndex(index, name="timestamp")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Input must have a DatetimeIndex or provide timestamp_column")
    if data.index.hasnans:
        raise ValueError("Timestamps may not contain NaT")
    return data.sort_index()


def _fill_only_complete_short_gaps(
    series: pd.Series,
    max_steps: int,
) -> tuple[pd.Series, pd.Series]:
    """Interpolate only runs whose *entire* length is within ``max_steps``.

    ``Series.interpolate(limit=max_steps)`` partly fills the beginning of a long
    gap, which would violate the stated 30-minute rule.  The run-length check
    below restores every long run in full.
    """

    missing = series.isna()
    if not missing.any():
        return series.copy(), pd.Series(False, index=series.index)
    groups = missing.ne(missing.shift(fill_value=False)).cumsum()
    run_lengths = missing.groupby(groups).transform("sum")
    fillable = missing & run_lengths.le(max_steps)
    candidate = series.interpolate(method="time", limit_area="inside")
    result = series.copy()
    result.loc[fillable] = candidate.loc[fillable]
    actually_filled = fillable & result.notna()
    return result, actually_filled


def resample_to_15min(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None = None,
    value_columns: Sequence[str] | None = None,
    max_interpolation_gap: str | pd.Timedelta = MAX_INTERPOLATION_GAP,
) -> ResampledPVFrame:
    """Aggregate numeric observations to 15 minutes and fill gaps up to 30 minutes.

    A 15-minute bin is marked observed if it contains at least one real value.
    Missing runs longer than ``max_interpolation_gap`` remain NaN and can be used
    by :class:`WindowedPVDataset` to reject affected target windows.
    """

    data = _as_datetime_index(frame, timestamp_column)
    if value_columns is None:
        columns = data.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    else:
        columns = list(value_columns)
        missing_columns = sorted(set(columns).difference(data.columns))
        if missing_columns:
            raise KeyError(f"Missing value columns: {missing_columns}")
    if not columns:
        raise ValueError("No numeric value columns were selected for resampling")

    numeric = data.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.where(np.isfinite(numeric), np.nan)
    values = numeric.resample(TARGET_FREQUENCY, label="left", closed="left").mean()
    # An entirely empty bin produces NaN from max(); casting it directly to
    # bool would falsely mark an interpolated value as a real observation.
    observed = (
        numeric.notna()
        .resample(TARGET_FREQUENCY, label="left", closed="left")
        .max()
        .fillna(False)
        .astype(bool)
    )

    gap = pd.Timedelta(max_interpolation_gap)
    step = pd.Timedelta(TARGET_FREQUENCY)
    if gap < pd.Timedelta(0) or gap % step:
        raise ValueError("max_interpolation_gap must be a non-negative multiple of 15 minutes")
    max_steps = int(gap / step)
    imputed = pd.DataFrame(False, index=values.index, columns=values.columns)
    if max_steps:
        for column in values.columns:
            values[column], imputed[column] = _fill_only_complete_short_gaps(
                values[column], max_steps
            )

    return ResampledPVFrame(
        values=values,
        observed_mask=observed.reindex_like(values).fillna(False).astype(bool),
        imputed_mask=imputed.astype(bool),
    )


def interpolate_resampled_partitions(
    resampled: ResampledPVFrame,
    partitions: Mapping[str, slice],
    *,
    max_interpolation_gap: str | pd.Timedelta = MAX_INTERPOLATION_GAP,
) -> ResampledPVFrame:
    """Fill short gaps independently inside pre-declared temporal partitions.

    Interpolating the full series before chronological splitting can leak a
    validation endpoint into a missing training value at the boundary.  This
    helper validates an exact, non-overlapping partition of the frame and never
    supplies an interpolation endpoint from another split.
    """

    values = resampled.values.copy()
    if not values.index.equals(resampled.observed_mask.index) or not values.index.equals(
        resampled.imputed_mask.index
    ):
        raise ValueError("Resampled values and masks must have identical indexes")
    value_columns = tuple(values.columns)
    if value_columns != tuple(resampled.observed_mask.columns) or value_columns != tuple(
        resampled.imputed_mask.columns
    ):
        raise ValueError("Resampled values and masks must have identical columns")
    observed = resampled.observed_mask.copy().astype(bool)
    imputed = resampled.imputed_mask.copy().astype(bool)

    ordered = _ordered_partition_bounds(partitions, len(values))

    gap = pd.Timedelta(max_interpolation_gap)
    step = pd.Timedelta(TARGET_FREQUENCY)
    if gap < pd.Timedelta(0) or gap % step:
        raise ValueError("max_interpolation_gap must be a non-negative multiple of 15 minutes")
    max_steps = int(gap / step)
    if not max_steps:
        return ResampledPVFrame(values, observed, imputed)

    for _, start, stop in ordered:
        for column in values.columns:
            filled, new_imputed = _fill_only_complete_short_gaps(
                values[column].iloc[start:stop], max_steps
            )
            segment_index = values.index[start:stop]
            values.loc[segment_index, column] = filled.to_numpy()
            imputed.loc[segment_index, column] = (
                imputed.loc[segment_index, column].to_numpy() | new_imputed.to_numpy()
            )
    values.attrs.update(resampled.values.attrs)
    return ResampledPVFrame(values, observed, imputed)


@dataclass
class TrainOnlyNormalizer:
    """Capacity normalization and optional z-scaling fitted on training data only."""

    capacity_quantile: float = 0.995
    clip_power: tuple[float, float] = (0.0, 1.05)
    eps: float = 1e-6
    capacity: float | None = field(default=None, init=False)
    capacity_source: str | None = field(default=None, init=False)
    feature_mean: dict[str, float] = field(default_factory=dict, init=False)
    feature_scale: dict[str, float] = field(default_factory=dict, init=False)
    power_column: str | None = field(default=None, init=False)
    feature_columns: tuple[str, ...] = field(default_factory=tuple, init=False)
    training_start: str | None = field(default=None, init=False)
    training_end: str | None = field(default=None, init=False)
    fitted: bool = field(default=False, init=False)

    def fit(
        self,
        training_frame: pd.DataFrame,
        *,
        power_column: str = "power",
        feature_columns: Sequence[str] = (),
        rated_capacity: float | None = None,
        partition: str = "train",
    ) -> TrainOnlyNormalizer:
        if partition != "train":
            raise ValueError("Normalizer statistics may only be fitted with partition='train'")
        if not 0 < self.capacity_quantile <= 1:
            raise ValueError("capacity_quantile must lie in (0, 1]")
        if power_column not in training_frame:
            raise KeyError(f"Power column {power_column!r} is missing")
        features = tuple(feature_columns)
        missing = sorted(set(features).difference(training_frame.columns))
        if missing:
            raise KeyError(f"Feature columns missing from training frame: {missing}")

        power = pd.to_numeric(training_frame[power_column], errors="coerce")
        finite_power = power[np.isfinite(power)]
        if finite_power.empty:
            raise ValueError("Training power has no finite observations")
        if rated_capacity is not None and np.isfinite(rated_capacity) and rated_capacity > 0:
            capacity = float(rated_capacity)
            source = "rated_capacity"
        else:
            capacity = float(finite_power.quantile(self.capacity_quantile))
            source = f"training_q{100 * self.capacity_quantile:g}"
        if not np.isfinite(capacity) or capacity <= self.eps:
            raise ValueError("Rated capacity or its training-only fallback must be positive")

        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for column in features:
            values = pd.to_numeric(training_frame[column], errors="coerce")
            mean = float(values.mean())
            scale = float(values.std(ddof=0))
            means[column] = mean if np.isfinite(mean) else 0.0
            scales[column] = scale if np.isfinite(scale) and scale > self.eps else 1.0

        self.capacity = capacity
        self.capacity_source = source
        self.feature_mean = means
        self.feature_scale = scales
        self.power_column = power_column
        self.feature_columns = features
        if isinstance(training_frame.index, pd.DatetimeIndex) and len(training_frame):
            self.training_start = training_frame.index.min().isoformat()
            self.training_end = training_frame.index.max().isoformat()
        self.fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self.fitted or self.capacity is None or self.power_column is None:
            raise RuntimeError("TrainOnlyNormalizer must be fitted before transformation")

    def normalize_power(self, values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
        self._check_fitted()
        normalized = values / float(self.capacity)
        low, high = self.clip_power
        if isinstance(normalized, pd.Series):
            return normalized.clip(lower=low, upper=high)
        return np.clip(normalized, low, high)

    def inverse_power(self, values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
        self._check_fitted()
        return values * float(self.capacity)

    def transform(
        self,
        frame: pd.DataFrame,
        *,
        output_power_column: str = "power_norm",
        feature_suffix: str = "_z",
        copy: bool = True,
    ) -> pd.DataFrame:
        self._check_fitted()
        if self.power_column not in frame:
            raise KeyError(f"Power column {self.power_column!r} is missing")
        result = frame.copy() if copy else frame
        result[output_power_column] = self.normalize_power(
            pd.to_numeric(result[self.power_column], errors="coerce")
        )
        for column in self.feature_columns:
            if column not in result:
                raise KeyError(f"Feature column {column!r} is missing")
            result[f"{column}{feature_suffix}"] = (
                pd.to_numeric(result[column], errors="coerce") - self.feature_mean[column]
            ) / self.feature_scale[column]
        return result

    def state_dict(self) -> dict[str, object]:
        self._check_fitted()
        return {
            "capacity_quantile": self.capacity_quantile,
            "clip_power": list(self.clip_power),
            "capacity": self.capacity,
            "capacity_source": self.capacity_source,
            "feature_mean": self.feature_mean.copy(),
            "feature_scale": self.feature_scale.copy(),
            "power_column": self.power_column,
            "feature_columns": list(self.feature_columns),
            "training_start": self.training_start,
            "training_end": self.training_end,
        }


def add_causal_history_features(
    frame: pd.DataFrame,
    *,
    power_column: str = "power_norm",
    clear_sky_column: str = "clear_sky_power_norm",
    partitions: Mapping[str, slice] | None = None,
) -> pd.DataFrame:
    """Add past-only residual, differences, and rolling volatility features.

    When ``partitions`` is supplied, differencing and rolling windows restart at
    every boundary. This is required by the strict no-cross-split window protocol.
    """

    if power_column not in frame or clear_sky_column not in frame:
        raise KeyError(f"Both {power_column!r} and {clear_sky_column!r} are required")
    result = frame.copy()
    derived_columns = (
        "clear_sky_residual",
        "power_diff_1",
        "power_diff_2",
        "rolling_volatility_1h",
        "rolling_volatility_3h",
        "clear_sky_index",
    )
    for column in derived_columns:
        result[column] = np.nan
    segments = (
        [("all", 0, len(result))]
        if partitions is None
        else _ordered_partition_bounds(partitions, len(result))
    )
    for _, start, stop in segments:
        index = result.index[start:stop]
        power = pd.to_numeric(result.loc[index, power_column], errors="coerce")
        clear_sky = pd.to_numeric(result.loc[index, clear_sky_column], errors="coerce")
        difference = power.diff(1)
        result.loc[index, "clear_sky_residual"] = (power - clear_sky).to_numpy()
        result.loc[index, "power_diff_1"] = difference.to_numpy()
        result.loc[index, "power_diff_2"] = difference.diff(1).to_numpy()
        result.loc[index, "rolling_volatility_1h"] = (
            difference.rolling(4, min_periods=4).std().to_numpy()
        )
        result.loc[index, "rolling_volatility_3h"] = (
            difference.rolling(12, min_periods=12).std().to_numpy()
        )
        result.loc[index, "clear_sky_index"] = (power / clear_sky.clip(lower=0.01)).to_numpy()
    return result
