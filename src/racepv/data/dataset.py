"""PyTorch window dataset implementing the RaCE-PV tensor contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from racepv.io import MODEL_ARRAYS

from .causal_history import AdmissionBatch, CausalHistoryState
from .origin_clock import _positive_int, _validated_starts, origin_times
from .ramp import IGNORE_STATE_LABEL

LOOKBACK_STEPS = 192
HORIZON_STEPS = 24
SAMPLING_INTERVAL = pd.Timedelta("15min")

# This is deliberately an allowlist, not a heuristic blacklist.  Newly added
# future variables must be reviewed as information that is genuinely available
# at forecast issuance time.
FUTURE_KNOWN_ALLOWLIST = frozenset(
    {
        "solar_elevation",
        "solar_zenith",
        "solar_azimuth",
        "solar_azimuth_sin",
        "solar_azimuth_cos",
        "clear_sky_ghi",
        "clear_sky_dni",
        "clear_sky_dhi",
        "clear_sky_power",
        "clear_sky_power_norm",
        "is_daylight",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "lead_time_hours",
        "lead_time_sin",
        "lead_time_cos",
    }
)

DERIVED_LEAD_COLUMNS = frozenset({"lead_time_hours", "lead_time_sin", "lead_time_cos"})
WEATHER_HISTORY_COLUMNS = (
    "ghi_z",
    "temperature_z",
    "humidity_z",
    "wind_speed_z",
)


class FutureInformationLeakageError(ValueError):
    """Raised when a future feature has not passed the known-future audit."""


def validate_future_known_columns(columns: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(columns)
    disallowed = sorted(set(requested).difference(FUTURE_KNOWN_ALLOWLIST))
    if disallowed:
        raise FutureInformationLeakageError(
            "Future features are restricted to solar geometry, clear-sky values, "
            f"calendar encodings and lead time. Disallowed columns: {disallowed}"
        )
    if len(set(requested)) != len(requested):
        raise ValueError("future_known_columns contains duplicates")
    return requested


def _split_bounds(split: slice | tuple[int, int] | None, n_rows: int) -> tuple[int, int]:
    if split is None:
        return 0, n_rows
    if isinstance(split, slice):
        if split.step not in (None, 1):
            raise ValueError("Split slices may not have a step")
        start = 0 if split.start is None else int(split.start)
        stop = n_rows if split.stop is None else int(split.stop)
    else:
        start, stop = (int(value) for value in split)
    if not 0 <= start < stop <= n_rows:
        raise ValueError(f"Invalid split bounds [{start}, {stop}) for {n_rows} rows")
    return start, stop


def _causal_split_bounds(split: slice | tuple[int, int] | None, n_rows: int) -> tuple[int, int]:
    if split is None:
        return 0, n_rows
    if isinstance(split, slice):
        if split.step is not None:
            step = int(_validated_starts((split.step,))[0])
            if step != 1:
                raise ValueError("causal split slice step must equal one")
        endpoints = (
            0 if split.start is None else split.start,
            n_rows if split.stop is None else split.stop,
        )
    elif isinstance(split, tuple):
        if len(split) != 2:
            raise ValueError("causal split tuples must contain two endpoints")
        endpoints = split
    else:
        raise TypeError("causal split must be a slice or two-integer tuple")
    start, stop = _validated_starts(endpoints).tolist()
    if not 0 <= start < stop <= n_rows:
        raise ValueError(f"Invalid split bounds [{start}, {stop}) for {n_rows} rows")
    return start, stop


def _validated_window_indices(indices: object, length: int) -> np.ndarray:
    if isinstance(indices, np.ndarray):
        if indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if indices.dtype.kind != "i":
            raise TypeError("indices must have a signed integer dtype")
        result = np.array(indices, dtype=np.int64, copy=True)
    else:
        if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
            raise TypeError("indices must be a sequence of signed integers")
        values = tuple(indices)
        if any(
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, np.integer))
            or (isinstance(item, np.integer) and item.dtype.kind != "i")
            for item in values
        ):
            raise TypeError("indices must contain only signed integers")
        result = np.asarray(values, dtype=np.int64)
    normalized = result.copy()
    normalized[normalized < 0] += np.int64(length)
    if np.any((normalized < 0) | (normalized >= length)):
        raise IndexError("window index is out of bounds")
    return normalized


def _validated_tensor_names(names: object) -> tuple[str, ...]:
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        raise TypeError("names must be a sequence of tensor names")
    requested = tuple(names)
    if any(not isinstance(name, str) for name in requested):
        raise TypeError("names must contain only strings")
    if len(set(requested)) != len(requested):
        raise ValueError("names must not contain duplicates")
    unknown = [name for name in requested if name not in MODEL_ARRAYS]
    if unknown:
        raise KeyError(f"names are not registered model arrays: {unknown}")
    return requested


class WindowedPVDataset(Dataset[dict[str, Tensor]]):
    """Create leakage-safe 48-hour to 6-hour windows.

    The returned mapping contains exactly six tensors:
    ``history_values``, ``history_mask``, ``future_known``, ``site_metadata``,
    ``targets``, and ``state_labels``.  A split boundary, when supplied, encloses
    the complete history and target horizon; no window can cross it.
    """

    tensor_names = MODEL_ARRAYS

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        history_columns: Sequence[str],
        future_known_columns: Sequence[str],
        site_metadata: Mapping[str, float] | Sequence[float],
        target_column: str = "power_norm",
        state_column: str = "state_label",
        history_mask: pd.DataFrame | None = None,
        target_observed_mask: pd.Series | np.ndarray | None = None,
        night_column: str | None = "is_night",
        lookback: int = LOOKBACK_STEPS,
        horizon: int = HORIZON_STEPS,
        stride: int = 1,
        split: slice | tuple[int, int] | None = None,
        require_observed_targets: bool = False,
        causal_state: CausalHistoryState | None = None,
        prepared_contract: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError("WindowedPVDataset requires a DatetimeIndex")
        if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            raise ValueError("Frame index must be sorted and unique")
        if causal_state is None:
            if lookback <= 0 or horizon <= 0 or stride <= 0:
                raise ValueError("lookback, horizon and stride must be positive")
            normalized_lookback = int(lookback)
            normalized_horizon = int(horizon)
            normalized_stride = int(stride)
        else:
            normalized_lookback = _positive_int(lookback, "lookback")
            normalized_horizon = _positive_int(horizon, "horizon")
            normalized_stride = _positive_int(stride, "stride")

        self.frame = frame
        self.history_columns = tuple(history_columns)
        self.future_known_columns = validate_future_known_columns(future_known_columns)
        self.target_column = target_column
        self.state_column = state_column
        self.night_column = night_column
        self.lookback = normalized_lookback
        self.horizon = normalized_horizon
        self.stride = normalized_stride
        self.require_observed_targets = bool(require_observed_targets)
        self.causal_state = causal_state
        self.prepared_contract = None

        if causal_state is not None:
            if not isinstance(causal_state, CausalHistoryState):
                raise TypeError("causal_state must be a CausalHistoryState")
            if require_observed_targets is not True:
                raise ValueError("causal_state requires require_observed_targets=True")
            if history_mask is None:
                raise ValueError("causal_state requires an explicit history_mask")
            if target_observed_mask is None:
                raise ValueError("causal_state requires an explicit target_observed_mask")
            if frame.index.tz is None or str(frame.index.tz) != "UTC":
                raise ValueError("causal_state requires a UTC frame index")
            if len(frame) != causal_state.labels_ns.shape[0]:
                raise ValueError("causal_state row count must equal the frame row count")
            if not np.array_equal(frame.index.asi8, causal_state.labels_ns):
                raise ValueError("causal_state row labels must exactly equal the frame index")
            if self.history_columns != causal_state.feature_names:
                raise ValueError("causal_state feature order must equal history_columns")

        required = set(self.history_columns) | {target_column, state_column}
        required |= set(self.future_known_columns).difference(DERIVED_LEAD_COLUMNS)
        if night_column is not None:
            required.add(night_column)
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise KeyError(f"Dataset columns are missing: {missing}")
        if not self.history_columns:
            raise ValueError("At least one history column is required")

        if isinstance(site_metadata, Mapping):
            self.site_metadata_names = tuple(site_metadata.keys())
            metadata_values = tuple(site_metadata.values())
        else:
            self.site_metadata_names = tuple(f"metadata_{i}" for i in range(len(site_metadata)))
            metadata_values = tuple(site_metadata)
        self._site_metadata = np.asarray(metadata_values, dtype=np.float32)
        if self._site_metadata.ndim != 1 or not np.isfinite(self._site_metadata).all():
            raise ValueError("site_metadata must be a finite one-dimensional vector")

        if causal_state is not None:
            if not isinstance(history_mask, pd.DataFrame):
                raise TypeError("history_mask must be a pandas DataFrame")
            if not history_mask.index.equals(frame.index):
                raise ValueError("history_mask index must exactly equal the frame index")
            if tuple(history_mask.columns) != self.history_columns:
                raise ValueError("history_mask columns must exactly equal history_columns")
            mask_array = history_mask.to_numpy(copy=True)
            if mask_array.dtype != np.dtype(np.bool_):
                raise TypeError("history_mask must contain strict boolean provenance")
            finite_history = np.isfinite(
                frame.loc[:, self.history_columns]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=np.float64)
            )
            if np.any(mask_array & ~finite_history):
                raise ValueError("history_mask cannot mark a non-finite value observed")
            power_index = self.history_columns.index("power_norm")
            if not np.array_equal(mask_array[:, power_index], causal_state.power_observed):
                raise ValueError("history power provenance must align with causal_state")
            for source_index, column in enumerate(WEATHER_HISTORY_COLUMNS):
                if column in self.history_columns and not np.array_equal(
                    mask_array[:, self.history_columns.index(column)],
                    causal_state.weather_observed[:, source_index],
                ):
                    raise ValueError(
                        f"history weather provenance for {column} must align with causal_state"
                    )
            self._history_mask = mask_array.astype(np.float32)
        elif history_mask is None:
            mask = frame.loc[:, self.history_columns].apply(
                lambda values: np.isfinite(pd.to_numeric(values, errors="coerce"))
            )
        else:
            if not history_mask.index.equals(frame.index):
                raise ValueError("history_mask index must exactly equal the frame index")
            mask = pd.DataFrame(index=frame.index)
            for column in self.history_columns:
                supplied = (
                    history_mask[column].astype(bool)
                    if column in history_mask
                    else frame[column].notna()
                )
                finite = np.isfinite(pd.to_numeric(frame[column], errors="coerce"))
                mask[column] = supplied & finite
        if causal_state is None:
            self._history_mask = mask.loc[:, self.history_columns].to_numpy(dtype=np.float32)

        if causal_state is not None:
            if isinstance(target_observed_mask, pd.Series):
                if not target_observed_mask.index.equals(frame.index):
                    raise ValueError(
                        "target_observed_mask index must exactly equal the frame index"
                    )
                target_mask_array = target_observed_mask.to_numpy(copy=True)
            elif isinstance(target_observed_mask, np.ndarray):
                target_mask_array = np.array(target_observed_mask, copy=True)
            else:
                raise TypeError("target_observed_mask must be a Series or numpy array")
            if target_mask_array.dtype != np.dtype(np.bool_):
                raise TypeError("target_observed_mask must contain strict boolean provenance")
            if target_mask_array.shape != (len(frame),):
                raise ValueError("target_observed_mask must have one value per frame row")
            if not np.array_equal(target_mask_array, causal_state.power_observed):
                raise ValueError("target observation provenance must align with causal_state")
            self._target_observed = target_mask_array
        elif target_observed_mask is None:
            self._target_observed = frame[target_column].notna().to_numpy(dtype=bool)
        else:
            target_mask_array = np.asarray(target_observed_mask, dtype=bool)
            if target_mask_array.shape != (len(frame),):
                raise ValueError("target_observed_mask must have one value per frame row")
            self._target_observed = target_mask_array

        self._history_values = frame.loc[:, self.history_columns].to_numpy(dtype=np.float32)
        raw_targets = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(
            dtype=np.float32, copy=True
        )
        raw_states = (
            pd.to_numeric(frame[state_column], errors="coerce")
            .fillna(IGNORE_STATE_LABEL)
            .to_numpy(dtype=np.int64, copy=True)
        )
        invalid_states = set(np.unique(raw_states)).difference({IGNORE_STATE_LABEL, 0, 1, 2})
        if invalid_states:
            raise ValueError(
                f"state labels must be stable/up/down (0/1/2) or {IGNORE_STATE_LABEL}; "
                f"received {sorted(invalid_states)}"
            )
        if causal_state is not None:
            self._night = causal_state.clear_sky_norm < 0.01
        elif night_column is None:
            self._night = np.zeros(len(frame), dtype=bool)
        else:
            self._night = frame[night_column].fillna(False).to_numpy(dtype=bool)
        # Physics rule: night output is exactly zero and carries no ramp class loss.
        raw_targets[self._night] = 0.0
        raw_states[self._night] = IGNORE_STATE_LABEL
        self._targets = raw_targets
        self._states = raw_states

        if causal_state is not None and np.any(self._target_observed & ~np.isfinite(self._targets)):
            raise ValueError("observed target provenance requires finite target values")

        base_future_columns = [
            column for column in self.future_known_columns if column not in DERIVED_LEAD_COLUMNS
        ]
        self._base_future_names = tuple(base_future_columns)
        self._base_future = frame.loc[:, base_future_columns].to_numpy(dtype=np.float32)

        split_start, split_stop = (
            _split_bounds(split, len(frame))
            if causal_state is None
            else _causal_split_bounds(split, len(frame))
        )
        self.split_bounds = (split_start, split_stop)
        if causal_state is not None:
            selected_split_ids = causal_state.split_ids[split_start:split_stop]
            if not np.all(selected_split_ids == selected_split_ids[0]):
                raise ValueError("dataset split bounds must contain one frozen split role")
            selected_id = selected_split_ids[0]
            if split_start > 0 and causal_state.split_ids[split_start - 1] == selected_id:
                raise ValueError("dataset split bounds must start at a frozen split role boundary")
            if split_stop < len(frame) and causal_state.split_ids[split_stop] == selected_id:
                raise ValueError("dataset split bounds must stop at a frozen split role boundary")
        final_start = split_stop - self.lookback - self.horizon
        candidates = (
            np.arange(split_start, final_start + 1, self.stride, dtype=np.int64)
            if final_start >= split_start
            else np.empty(0, dtype=np.int64)
        )
        if self.causal_state is None:
            self.window_starts = np.asarray(
                [start for start in candidates if self._window_is_valid(int(start))],
                dtype=np.int64,
            )
        else:
            history = self.causal_state.history_admissible(candidates, lookback=self.lookback)
            target = self.target_eligible(candidates)
            self.candidate_starts = candidates.copy()
            self.history_candidate_starts = candidates[history.accepted].copy()
            self.window_starts = candidates[history.accepted & target.accepted].copy()

        if prepared_contract is not None:
            from racepv.data.prepared_contract import require_origin_safe_history_contract

            self.prepared_contract = prepared_contract
            require_origin_safe_history_contract(self)

    def target_eligible(self, starts: object) -> AdmissionBatch:
        if self.causal_state is None:
            raise ValueError("target_eligible requires causal_state")
        times = origin_times(
            self.causal_state.labels_ns,
            starts,
            lookback=self.lookback,
            horizon=self.horizon,
        )
        starts_array = (
            np.searchsorted(self.causal_state.labels_ns, times.issue_ns) - np.int64(self.lookback)
        ).astype(np.int64, copy=False)
        target_rows = (
            starts_array[:, None] + self.lookback + np.arange(self.horizon, dtype=np.int64)
        )
        history_rows = starts_array[:, None] + np.arange(self.lookback, dtype=np.int64)
        inside_bounds = (starts_array >= self.split_bounds[0]) & (
            target_rows[:, -1] < self.split_bounds[1]
        )
        full_rows = np.concatenate((history_rows, target_rows), axis=1)
        same_split = np.all(
            self.causal_state.split_ids[full_rows] == self.causal_state.split_ids[full_rows[:, :1]],
            axis=1,
        )
        night = self._night[target_rows]
        target_ok = np.all(
            night | (self._target_observed[target_rows] & np.isfinite(self._targets[target_rows])),
            axis=1,
        )
        future_ok = (
            np.all(np.isfinite(self._base_future[target_rows]), axis=(1, 2))
            if self._base_future.shape[1]
            else np.ones(len(starts_array), dtype=bool)
        )
        accepted = inside_bounds & same_split & target_ok & future_ok
        reasons_list: list[tuple[str, ...]] = []
        for index in range(len(starts_array)):
            row_reasons: list[str] = []
            if not inside_bounds[index]:
                row_reasons.append("outside_dataset_split")
            if not same_split[index]:
                row_reasons.append("crosses_split")
            if not target_ok[index]:
                row_reasons.append("unobserved_or_nonfinite_daylight_target")
            if not future_ok[index]:
                row_reasons.append("nonfinite_future_known")
            reasons_list.append(tuple(row_reasons))
        reasons = tuple(reasons_list)
        return AdmissionBatch(accepted=accepted, reasons=reasons)

    def _window_is_valid(self, start: int) -> bool:
        target_start = start + self.lookback
        stop = target_start + self.horizon
        index = self.frame.index[start:stop]
        if len(index) != self.lookback + self.horizon:
            return False
        if len(index) > 1:
            elapsed = index[1:] - index[:-1]
            if not np.all(elapsed == SAMPLING_INTERVAL):
                return False

        future_targets = self._targets[target_start:stop]
        # Long power gaps remain NaN after preprocessing and invalidate every
        # sample whose history or target intersects them. Short imputed gaps are
        # finite but retain mask=0, so their provenance is still visible.
        if not np.isfinite(self._targets[start:target_start]).all():
            return False
        if not np.isfinite(future_targets).all():
            return False
        if self.require_observed_targets:
            required = ~self._night[target_start:stop]
            if not self._target_observed[target_start:stop][required].all():
                return False
        if (
            self._base_future.shape[1]
            and not np.isfinite(self._base_future[target_start:stop]).all()
        ):
            return False
        return True

    def __len__(self) -> int:
        return int(len(self.window_starts))

    def _future_matrix(self, start: int, stop: int) -> np.ndarray:
        base_lookup = {
            name: self._base_future[start:stop, i] for i, name in enumerate(self._base_future_names)
        }
        lead_hours = np.arange(1, self.horizon + 1, dtype=np.float32) * 0.25
        columns: list[np.ndarray] = []
        for name in self.future_known_columns:
            if name == "lead_time_hours":
                values = lead_hours
            elif name == "lead_time_sin":
                values = np.sin(2 * np.pi * lead_hours / 6.0).astype(np.float32)
            elif name == "lead_time_cos":
                values = np.cos(2 * np.pi * lead_hours / 6.0).astype(np.float32)
            else:
                values = base_lookup[name]
            columns.append(np.asarray(values, dtype=np.float32))
        if not columns:
            return np.empty((self.horizon, 0), dtype=np.float32)
        return np.stack(columns, axis=-1)

    def read_model_window_batch(
        self,
        indices: object,
        names: Sequence[str] = MODEL_ARRAYS,
    ) -> dict[str, np.ndarray]:
        positions = _validated_window_indices(indices, len(self))
        requested = _validated_tensor_names(names)
        starts = self.window_starts[positions]
        target_rows = starts[:, None] + self.lookback + np.arange(self.horizon, dtype=np.int64)

        result: dict[str, np.ndarray] = {}
        need_history = "history_values" in requested or "history_mask" in requested
        if need_history:
            if self.causal_state is not None:
                history = self.causal_state.resolve_history(starts, lookback=self.lookback)
                history_values = history.values
                history_mask = history.mask
            else:
                history_rows = starts[:, None] + np.arange(self.lookback, dtype=np.int64)
                history_values = np.nan_to_num(
                    self._history_values[history_rows],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                history_mask = self._history_mask[history_rows]
                history_values = np.where(history_mask.astype(bool), history_values, 0.0)
            result["history_values"] = np.array(history_values, dtype=np.float32, copy=True)
            result["history_mask"] = np.array(history_mask, dtype=np.float32, copy=True)

        if "future_known" in requested:
            result["future_known"] = (
                np.stack(
                    [
                        self._future_matrix(
                            int(start) + self.lookback,
                            int(start) + self.lookback + self.horizon,
                        )
                        for start in starts
                    ],
                    axis=0,
                )
                if len(starts)
                else np.empty((0, self.horizon, len(self.future_known_columns)), dtype=np.float32)
            )
        if "site_metadata" in requested:
            result["site_metadata"] = np.repeat(
                self._site_metadata[None, :], len(starts), axis=0
            ).astype(np.float32, copy=False)
        if "targets" in requested:
            result["targets"] = np.array(self._targets[target_rows], dtype=np.float32, copy=True)
        if "state_labels" in requested:
            result["state_labels"] = np.array(self._states[target_rows], dtype=np.int64, copy=True)
        return {name: result[name] for name in requested}

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        batch = self.read_model_window_batch([item])
        return {
            name: torch.from_numpy(np.array(values[0], copy=True)) for name, values in batch.items()
        }

    def sample_timestamps(self, item: int) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        """Return history/target timestamps for audits without adding a seventh tensor."""

        start = int(self.window_starts[item])
        target_start = start + self.lookback
        stop = target_start + self.horizon
        return self.frame.index[start:target_start], self.frame.index[target_start:stop]

    def sample_daylight_mask(self, item: int) -> np.ndarray:
        """Return the exact target daylight mask used for night zeroing/labels."""

        start = int(self.window_starts[item]) + self.lookback
        stop = start + self.horizon
        return (~self._night[start:stop]).copy()

    def sample_target_observed_mask(self, item: int) -> np.ndarray:
        """Return target observation provenance without changing the six-tensor API."""

        start = int(self.window_starts[item]) + self.lookback
        stop = start + self.horizon
        return self._target_observed[start:stop].copy()
