"""UTC origin-clock contracts for origin-safe prepared data."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

GRID_STEP_NS = 900_000_000_000
_INT64_INFO = np.iinfo(np.int64)
_SOURCE_FIELDS = {
    "source_timestamp_semantics",
    "timezone",
    "interval_duration_ns",
    "stored_label",
    "resample_closed",
    "resample_label",
    "sampling_interval_ns",
    "issue_definition",
    "measurement_delay_ns",
    "available_time_definition",
    "availability_assumption",
    "duplicate_timestamp_policy",
    "invalid_timestamp_policy",
    "evidence",
}
_EVIDENCE_FIELDS = {"kind", "reference", "sha256"}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class OriginTimes:
    issue_ns: np.ndarray
    target_start_ns: np.ndarray
    target_end_ns: np.ndarray
    lead_hours: np.ndarray

    def __post_init__(self) -> None:
        fields = (
            ("issue_ns", np.dtype(np.int64), 1),
            ("target_start_ns", np.dtype(np.int64), 2),
            ("target_end_ns", np.dtype(np.int64), 2),
            ("lead_hours", np.dtype(np.float32), 1),
        )
        for name, dtype, ndim in fields:
            value = getattr(self, name)
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{name} must be a numpy array")
            if value.dtype != dtype:
                raise TypeError(f"{name} must have dtype {dtype}")
            if value.ndim != ndim:
                raise ValueError(f"{name} must be {ndim}-dimensional")

        batch_size = self.issue_ns.shape[0]
        horizon = self.lead_hours.shape[0]
        if horizon == 0:
            raise ValueError("lead_hours must contain at least one horizon")
        expected_targets_shape = (batch_size, horizon)
        if self.target_start_ns.shape != expected_targets_shape:
            raise ValueError("target_start_ns shape must equal (B, H)")
        if self.target_end_ns.shape != expected_targets_shape:
            raise ValueError("target_end_ns shape must equal (B, H)")

        for name, _, _ in fields:
            copied = np.array(getattr(self, name), copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)


def _validated_labels(labels_ns: np.ndarray) -> np.ndarray:
    if not isinstance(labels_ns, np.ndarray):
        raise TypeError("labels_ns must be a numpy array")
    if labels_ns.ndim != 1:
        raise ValueError("labels_ns must be one-dimensional")
    if labels_ns.dtype != np.dtype(np.int64):
        raise TypeError("labels_ns must have dtype int64")
    if labels_ns.size == 0:
        raise ValueError("labels_ns must contain at least one label")
    if np.any(labels_ns == _INT64_INFO.min):
        raise ValueError("labels_ns must not contain the int64 NaT sentinel")

    if labels_ns.size > 1:
        previous = labels_ns[:-1]
        if np.any(previous > _INT64_INFO.max - GRID_STEP_NS):
            raise ValueError("labels_ns is not an exact 15-minute grid")
        if not np.array_equal(labels_ns[1:], previous + GRID_STEP_NS):
            raise ValueError("labels_ns must increase by exactly 15 minutes")
    return labels_ns


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    if normalized > _INT64_INFO.max:
        raise OverflowError(f"{name} exceeds int64")
    return normalized


def _validated_starts(starts: np.ndarray | Sequence[int]) -> np.ndarray:
    if isinstance(starts, np.ndarray):
        if starts.ndim != 1:
            raise ValueError("starts must be one-dimensional")
        if starts.dtype.kind != "i":
            raise TypeError("starts array must have a signed integer dtype")
        values = starts.tolist()
    elif isinstance(starts, Sequence) and not isinstance(starts, (str, bytes)):
        values = list(starts)
    else:
        raise TypeError("starts must be a signed integer array or integer sequence")

    normalized: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_, np.unsignedinteger)) or not isinstance(
            value, Integral
        ):
            raise TypeError("starts must contain only signed integers")
        item = int(value)
        if item < _INT64_INFO.min or item > _INT64_INFO.max:
            raise OverflowError("start exceeds int64")
        normalized.append(item)
    return np.asarray(normalized, dtype=np.int64)


def origin_times(
    labels_ns: np.ndarray,
    starts: np.ndarray | Sequence[int],
    *,
    lookback: int,
    horizon: int,
) -> OriginTimes:
    labels = _validated_labels(labels_ns)
    lookback_value = _positive_int(lookback, "lookback")
    horizon_value = _positive_int(horizon, "horizon")
    starts_array = _validated_starts(starts)

    max_offset = lookback_value + horizon_value - 1
    if max_offset > _INT64_INFO.max:
        raise OverflowError("window row offset exceeds int64")
    for start in starts_array.tolist():
        if start > _INT64_INFO.max - max_offset:
            raise OverflowError("window row arithmetic exceeds int64")
        if start < 0 or start + max_offset >= labels.size:
            raise IndexError("window start is out of bounds")

    rows = (
        starts_array[:, None] + np.int64(lookback_value) + np.arange(horizon_value, dtype=np.int64)
    )
    target_start_ns = labels[rows].copy()
    if np.any(target_start_ns > _INT64_INFO.max - GRID_STEP_NS):
        raise OverflowError("target interval end exceeds int64")
    target_end_ns = target_start_ns + GRID_STEP_NS
    issue_ns = target_start_ns[:, 0].copy()
    lead_hours = np.arange(1, horizon_value + 1, dtype=np.float32) * np.float32(0.25)
    return OriginTimes(
        issue_ns=issue_ns,
        target_start_ns=target_start_ns,
        target_end_ns=target_end_ns,
        lead_hours=lead_hours,
    )


def validate_source_time_contract(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("source time contract must be a mapping")
    if set(value) != _SOURCE_FIELDS:
        raise ValueError("source time contract fields must match the exact schema")

    semantics = value["source_timestamp_semantics"]
    if not isinstance(semantics, str) or semantics not in {
        "instantaneous",
        "interval_start",
        "interval_end",
    }:
        raise ValueError("source_timestamp_semantics is invalid")

    timezone = value["timezone"]
    if not isinstance(timezone, str) or not timezone:
        raise TypeError("timezone must be a nonempty IANA name")
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError("timezone must be a valid IANA name") from exc

    duration = value["interval_duration_ns"]
    if semantics == "instantaneous":
        if duration is not None:
            raise ValueError("instantaneous timestamps require null interval_duration_ns")
        normalized_duration = None
    else:
        normalized_duration = _positive_int(duration, "interval_duration_ns")

    fixed_values: tuple[tuple[str, object], ...] = (
        ("stored_label", "interval_start"),
        ("resample_closed", "left"),
        ("resample_label", "left"),
        ("sampling_interval_ns", GRID_STEP_NS),
        ("issue_definition", "last_history_interval_end"),
        ("measurement_delay_ns", 0),
        ("available_time_definition", "interval_end"),
        ("availability_assumption", "zero_additional_delay_event_time"),
        ("duplicate_timestamp_policy", "reject"),
        ("invalid_timestamp_policy", "reject"),
    )
    normalized_fixed: dict[str, object] = {}
    for key, expected in fixed_values:
        actual = value[key]
        if isinstance(expected, int):
            if isinstance(actual, (bool, np.bool_)) or not isinstance(actual, Integral):
                raise TypeError(f"{key} must be an integer")
            actual = int(actual)
        elif not isinstance(actual, str):
            raise TypeError(f"{key} must be a string")
        if actual != expected:
            raise ValueError(f"{key} must equal {expected!r}")
        normalized_fixed[key] = actual

    evidence = value["evidence"]
    if not isinstance(evidence, Mapping):
        raise TypeError("evidence must be a mapping")
    if set(evidence) != _EVIDENCE_FIELDS:
        raise ValueError("evidence fields must match the exact schema")
    kind = evidence["kind"]
    # Structural validation is not formal source qualification. The latter
    # independently rejects research assumptions in its evidence-domain guard.
    if not isinstance(kind, str) or kind not in {
        "provider_document", "synthetic_fixture", "research_assumption"
    }:
        raise ValueError("evidence kind is invalid")
    reference = evidence["reference"]
    if not isinstance(reference, str):
        raise TypeError("evidence reference must be a string")
    reference = reference.strip()
    if not reference:
        raise ValueError("evidence reference must be nonempty")
    sha256 = evidence["sha256"]
    if not isinstance(sha256, str):
        raise TypeError("evidence sha256 must be a string")
    if _SHA256_PATTERN.fullmatch(sha256) is None:
        raise ValueError("evidence sha256 must be 64 lowercase hexadecimal characters")

    return {
        "source_timestamp_semantics": semantics,
        "timezone": timezone,
        "interval_duration_ns": normalized_duration,
        **normalized_fixed,
        "evidence": {"kind": kind, "reference": reference, "sha256": sha256},
    }


def freeze_split_plan(labels_ns: np.ndarray, slices: Mapping[str, slice]) -> dict[str, object]:
    labels = _validated_labels(labels_ns)
    if not isinstance(slices, Mapping):
        raise TypeError("slices must be a mapping")
    if not slices:
        raise ValueError("slices must contain at least one split")

    split_order: list[str] = []
    frozen_splits: dict[str, dict[str, int]] = {}
    expected_start = 0
    for name, split_slice in slices.items():
        if not isinstance(name, str) or not name:
            raise TypeError("split names must be nonempty strings")
        if not isinstance(split_slice, slice):
            raise TypeError("split values must be slices")
        if split_slice.step is not None:
            raise ValueError("split slices must not specify a step")
        start = split_slice.start
        stop = split_slice.stop
        if (
            isinstance(start, (bool, np.bool_))
            or not isinstance(start, Integral)
            or isinstance(stop, (bool, np.bool_))
            or not isinstance(stop, Integral)
        ):
            raise TypeError("split start and stop must be explicit integers")
        start_row = int(start)
        stop_row = int(stop)
        if start_row < 0 or stop_row <= start_row or stop_row > labels.size:
            raise ValueError("split slice is empty or out of bounds")
        if start_row != expected_start:
            raise ValueError("split slices must form a contiguous ordered cover")

        last_label = int(labels[stop_row - 1])
        if last_label > _INT64_INFO.max - GRID_STEP_NS:
            raise OverflowError("split end-exclusive timestamp exceeds int64")
        split_order.append(name)
        frozen_splits[name] = {
            "start_row": start_row,
            "stop_row": stop_row,
            "start_ns": int(labels[start_row]),
            "end_exclusive_ns": last_label + GRID_STEP_NS,
        }
        expected_start = stop_row

    if expected_start != labels.size:
        raise ValueError("split slices must cover every label exactly once")
    return {
        "grid_start_ns": int(labels[0]),
        "grid_step_ns": GRID_STEP_NS,
        "row_count": int(labels.size),
        "split_order": split_order,
        "splits": frozen_splits,
    }
