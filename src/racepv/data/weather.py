"""Leakage-safe issued-weather sidecars for forecast-only model inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from racepv.data.origin_clock import origin_times

FORECAST_WEATHER_FEATURES = (
    "gfs_downward_shortwave_wm2",
    "gfs_clear_sky_index",
    "gfs_total_cloud_cover_pct",
    "gfs_temperature_2m_c",
    "gfs_relative_humidity_2m_pct",
    "gfs_wind_u_10m_ms",
    "gfs_wind_v_10m_ms",
)
ISSUED_WEATHER_FORMAT = "racepv-issued-weather-v1"
ISSUED_WEATHER_CONTRACT_VERSION = 1
ISSUED_WEATHER_HORIZON = 24
ISSUED_WEATHER_SAMPLING_MINUTES = 15
_STEP_NS = ISSUED_WEATHER_SAMPLING_MINUTES * 60 * 1_000_000_000


def expected_origin_contract(version=None, profile=None) -> dict[str, object] | None:
    """Validate independently supplied, paired prepared protocol expectations."""
    if version is None and profile is None:
        return None
    if type(version) is not int or version != 2 or profile != "as-of-origin-v1":
        raise ValueError("expected prepared version 2 and profile as-of-origin-v1 must be paired")
    return {
        "schema_kind": "racepv.prepared-window-contract",
        "schema_version": 2,
        "causality_profile": profile,
    }


def forecast_origin_timestamps(timestamps: np.ndarray, *, contract=None) -> np.ndarray:
    """Use the shared interval clock for v2, retaining legacy right-label behavior."""
    timestamps = np.asarray(timestamps)
    if timestamps.ndim != 2 or timestamps.shape[1] == 0 or timestamps.dtype != np.dtype("int64"):
        raise ValueError("target timestamps must have int64 shape [N,H]")
    if contract is None:
        return timestamps[:, 0] - _STEP_NS
    expected = expected_origin_contract(
        contract.get("schema_version"), contract.get("causality_profile")
    )
    if expected is None or contract.get("schema_kind") != expected["schema_kind"]:
        raise ValueError("invalid origin contract")
    origins = np.empty(len(timestamps), dtype=np.int64)
    for row, targets in enumerate(timestamps):
        if int(targets[0]) <= np.iinfo(np.int64).min + _STEP_NS:
            raise ValueError("origin timestamp underflow")
        labels = np.concatenate((np.asarray([int(targets[0]) - _STEP_NS], dtype=np.int64), targets))
        origins[row] = origin_times(labels, [0], lookback=1, horizon=len(targets)).issue_ns[0]
    return origins


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class IssuedWeatherStore:
    """Validated point-weather tensors indexed by forecast-origin timestamp."""

    _required_arrays = {
        "origin_times_ns": np.dtype("int64"),
        "future_weather": np.dtype("float32"),
        "future_weather_mask": np.dtype("uint8"),
        "selected_issue_times_ns": np.dtype("int64"),
        "selected_available_times_ns": np.dtype("int64"),
        "weather_point_ids": np.dtype("int32"),
    }

    def __init__(self, path: str | Path, *, expected_version=None, expected_profile=None) -> None:
        self.origin_contract = expected_origin_contract(expected_version, expected_profile)
        self.path = Path(path)
        schema_path = self.path.with_suffix(".schema.json")
        with schema_path.open(encoding="utf-8") as handle:
            schema = json.load(handle)
        if not isinstance(schema, dict):
            raise TypeError("issued-weather schema must be a JSON object")
        self.schema: dict[str, Any] = schema
        self._validate_schema()
        self.content_sha256 = _sha256_file(self.path)
        if self.schema.get("npz_sha256") != self.content_sha256:
            raise ValueError("issued-weather NPZ SHA-256 does not match its schema")

        with np.load(self.path, allow_pickle=False) as archive:
            missing = sorted(set(self._required_arrays).difference(archive.files))
            if missing:
                raise KeyError(f"issued-weather store is missing arrays: {missing}")
            arrays = {name: np.asarray(archive[name]) for name in self._required_arrays}
        self._validate_arrays(arrays)
        self.origin_times_ns = arrays["origin_times_ns"]
        self.future_weather = np.where(
            arrays["future_weather_mask"].astype(bool),
            arrays["future_weather"],
            0.0,
        ).astype(np.float32, copy=False)
        self.future_weather_mask = arrays["future_weather_mask"]
        self.selected_issue_times_ns = arrays["selected_issue_times_ns"]
        self.selected_available_times_ns = arrays["selected_available_times_ns"]
        self.weather_point_ids = arrays["weather_point_ids"]
        self._row_by_origin = {int(origin): row for row, origin in enumerate(self.origin_times_ns)}

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.schema["feature_names"])

    def _validate_schema(self) -> None:
        version = 2 if self.origin_contract is not None else 1
        if self.schema.get("format") != f"racepv-issued-weather-v{version}":
            raise ValueError("unsupported issued-weather schema format")
        if self.schema.get("forecast_weather_contract_version") != version:
            raise ValueError("unsupported issued-weather contract version")
        if version == 2:
            for field, expected in {
                "causality_profile": "as-of-origin-v1",
                "timestamp_semantics": "target_interval_start_ns",
                "issue_definition": "last_history_interval_end",
                "missing_value_policy": "zero_value_zero_mask",
            }.items():
                if self.schema.get(field) != expected:
                    raise ValueError(f"issued-weather {field} differs from v2 contract")
        elif self.schema.get("causality_profile") is not None:
            raise ValueError("legacy weather cannot claim a causality profile")
        if tuple(self.schema.get("feature_names", ())) != FORECAST_WEATHER_FEATURES:
            raise ValueError("issued-weather feature order does not match the registered contract")
        if self.schema.get("horizon") != ISSUED_WEATHER_HORIZON:
            raise ValueError("issued-weather horizon must be 24")
        if self.schema.get("sampling_minutes") != ISSUED_WEATHER_SAMPLING_MINUTES:
            raise ValueError("issued-weather sampling interval must be 15 minutes")
        if self.schema.get("selection_boundary") != "validation_only":
            raise ValueError("issued-weather store must be validation_only")

    def _validate_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        for name, dtype in self._required_arrays.items():
            if arrays[name].dtype != dtype:
                raise TypeError(f"{name} must use dtype {dtype}")
        origins = arrays["origin_times_ns"]
        if origins.ndim != 1 or len(origins) == 0:
            raise ValueError("origin_times_ns must be a non-empty vector")
        if np.any(np.diff(origins) <= 0):
            raise ValueError("origin_times_ns must be strictly increasing")
        expected_weather_shape = (
            len(origins),
            ISSUED_WEATHER_HORIZON,
            len(FORECAST_WEATHER_FEATURES),
        )
        if arrays["future_weather"].shape != expected_weather_shape:
            raise ValueError(f"future_weather must have shape {expected_weather_shape}")
        if arrays["future_weather_mask"].shape != expected_weather_shape:
            raise ValueError(f"future_weather_mask must have shape {expected_weather_shape}")
        mask = arrays["future_weather_mask"]
        if not np.isin(mask, (0, 1)).all():
            raise ValueError("future_weather_mask must be binary")
        values = arrays["future_weather"]
        if not np.isfinite(values[mask.astype(bool)]).all():
            raise ValueError("available future_weather values must be finite")
        for name in (
            "selected_issue_times_ns",
            "selected_available_times_ns",
            "weather_point_ids",
        ):
            if arrays[name].shape != origins.shape:
                raise ValueError(f"{name} must contain one value per origin")
        issues = arrays["selected_issue_times_ns"]
        available = arrays["selected_available_times_ns"]
        if np.any(issues > available):
            raise ValueError("issued-weather issue time cannot follow available time")
        if np.any(available > origins):
            raise ValueError("issued-weather available time cannot follow forecast origin")

    def lookup(self, origin_time_ns: int) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of the weather tensor and mask for one origin."""

        row = self._row_by_origin.get(int(origin_time_ns))
        if row is None:
            shape = (ISSUED_WEATHER_HORIZON, len(FORECAST_WEATHER_FEATURES))
            return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.uint8)
        return self.future_weather[row].copy(), self.future_weather_mask[row].copy()


class WeatherAugmentedDataset(Dataset[dict[str, Tensor]]):
    """Add issued-weather tensors to a window dataset without mutating it."""

    def __init__(
        self,
        base: Dataset[dict[str, Tensor]],
        store: IssuedWeatherStore,
        *,
        expected_version=None,
        expected_profile=None,
    ) -> None:
        self.origin_contract = expected_origin_contract(expected_version, expected_profile)
        if self.origin_contract != store.origin_contract:
            raise ValueError("weather store and window origin contracts differ")
        schema = getattr(base, "schema", None) or {}
        if schema.get("schema_version") == 2 and self.origin_contract is None:
            raise ValueError("v2 weather windows require explicit origin expectations")
        self.base = base
        self.weather_store = store

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        base_sample = self.base[index]
        if "timestamps" not in base_sample:
            raise KeyError("weather augmentation requires target timestamps")
        timestamps = base_sample["timestamps"]
        if timestamps.ndim != 1 or timestamps.shape[0] != ISSUED_WEATHER_HORIZON:
            raise ValueError("weather augmentation requires exactly 24 target timestamps")
        origin_time_ns = int(
            forecast_origin_timestamps(
                timestamps.detach().cpu().numpy()[None, :], contract=self.origin_contract
            )[0]
        )
        values, mask = self.weather_store.lookup(origin_time_ns)
        sample = dict(base_sample)
        sample["future_weather"] = torch.from_numpy(values)
        sample["future_weather_mask"] = torch.from_numpy(mask.astype(np.float32))
        return sample

    def __getattr__(self, name: str) -> Any:
        if name in {"base", "weather_store"}:
            raise AttributeError(name)
        return getattr(self.base, name)
