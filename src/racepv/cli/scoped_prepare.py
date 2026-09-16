"""Scoped in-memory preparation and recomputation; no formal qualification or IO."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from racepv import io as window_io
from racepv.cli.prepare import FUTURE_COLUMNS, _build_origin_safe_frame, _json_bytes, _site_vector
from racepv.cli.scoped_fit import build_scoped_train_fit, restore_scoped_train_fit
from racepv.data import prepared_contract as contract
from racepv.data.adapters import adapter_for
from racepv.data.dataset import WindowedPVDataset
from racepv.data.origin_clock import origin_times
from racepv.data.scoped_input import (
    _validated_source_and_scope,
    resample_scoped_source_frame,
    scoped_source_positions,
)

_OPTION_FIELDS = (
    "frozen_split_contract",
    "selected_split",
    "source_time_contract",
    "frozen_fit_blob",
    "expected_fit_sha256",
    "train_raw_input_sha256",
    "site_index",
    "ramp_method",
    "stride",
    "expected_records_sha256",
    "allowed_splits",
    "allow_approximate_solar",
)


@dataclass
class ScopedPreparation:
    """Transient engineering object, deliberately not a qualified prepared contract."""

    dataset: WindowedPVDataset
    arrays: dict
    identity: dict


def _scalar(value):
    if value is pd.NaT:
        return ["NaT"]
    if value is pd.NA:
        return ["NA"]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ["none"]
    if isinstance(value, (pd.Timestamp, datetime)):
        return ["timestamp", value.isoformat()]
    if type(value) is float:
        return ["float", value.hex()]
    if type(value) in (str, int, bool):
        return [type(value).__name__, value]
    raise TypeError("provider records must contain scalar strings, numbers or timestamps")


def scoped_records_sha256(frame):
    """Exact typed, ordered records hash; not original file/extraction identity.

    Pandas row labels are excluded because they are not provider record content.
    Float hex encoding avoids lossy decimal rounding; no numeric coercion is used.
    """
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise TypeError("records must be a DataFrame with unique columns")
    if any(type(name) is not str for name in frame.columns):
        raise TypeError("record column names must be strings")
    digest = hashlib.sha256()
    header = [
        "racepv.ordered-provider-records-v1",
        list(frame.columns),
        [str(dtype) for dtype in frame.dtypes],
        len(frame),
    ]
    digest.update(_json_bytes(header) + b"\n")
    for row in frame.itertuples(index=False, name=None):
        digest.update(_json_bytes([_scalar(value) for value in row]) + b"\n")
    return digest.hexdigest()


def _admit(entry, supplied):
    options = dict(supplied)
    options.setdefault("allow_approximate_solar", False)
    contract._keys(options, _OPTION_FIELDS, "scoped preparation options")
    split = options["selected_split"]
    if type(split) is not str or split not in {"train", "validation"}:
        raise ValueError("this preparation entry accepts train/validation only")
    expected_scope = ("train",) if split == "train" else ("train", "validation")
    contract._exact(options["allowed_splits"], expected_scope, "allowed_splits")
    contract._integer(options["stride"], "stride", minimum=1)
    contract._sha(options["expected_records_sha256"], "expected_records_sha256")
    _, scope = _validated_source_and_scope(
        entry,
        frozen_split_contract=options["frozen_split_contract"],
        selected_split=split,
        source_time_contract=options["source_time_contract"],
    )
    fit = restore_scoped_train_fit(
        options["frozen_fit_blob"],
        entry,
        expected_sha256=options["expected_fit_sha256"],
        frozen_split_contract=options["frozen_split_contract"],
        source_time_contract=options["source_time_contract"],
        raw_input_sha256=options["train_raw_input_sha256"],
        site_index=options["site_index"],
        ramp_method=options["ramp_method"],
        allow_approximate_solar=options["allow_approximate_solar"],
    )
    return options, scope, fit


def _sample(entry, frame, options, split, expected_records_sha256):
    contract._sha(expected_records_sha256, "expected_records_sha256")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("provider input must be a DataFrame")
    if not frame.columns.is_unique or entry.timestamp_column not in frame:
        raise ValueError("provider columns must be unique and contain timestamp")
    positions = scoped_source_positions(
        entry,
        frame[entry.timestamp_column],
        frozen_split_contract=options["frozen_split_contract"],
        selected_split=split,
        source_time_contract=options["source_time_contract"],
    )
    if len(positions) != len(frame):
        raise ValueError("provider records escaped selected split")
    if scoped_records_sha256(frame) != expected_records_sha256:
        raise ValueError("provider records SHA-256 mismatch")
    return resample_scoped_source_frame(
        adapter_for(entry, "."),
        entry,
        frame,
        frozen_split_contract=options["frozen_split_contract"],
        selected_split=split,
        source_time_contract=options["source_time_contract"],
    )


def _array_identities(arrays):
    return {name: contract._bounded_array_identity(array) for name, array in arrays.items()}


def prepare_scoped_split(entry, source_frame, **supplied_options):
    """Build one selected split from pinned records and frozen fit, without refitting."""
    options, scope, fit = _admit(entry, supplied_options)
    split = options["selected_split"]
    sampled = _sample(entry, source_frame, options, split, options["expected_records_sha256"])
    labels = sampled.values.index.as_unit("ns").asi8.copy()
    slices = {split: slice(0, len(labels))}
    physical = fit.build_physical(sampled.values.index)
    frame, raw, observed, weather, state, history_mask, split_ids = _build_origin_safe_frame(
        sampled,
        labels,
        slices,
        fit.normalizer,
        physical,
        fit.labeler,
        available=fit.normalizer.feature_columns,
        split_numbers={split: contract.SPLITS.index(split)},
    )
    site = _site_vector(entry, float(fit.normalizer.capacity))
    dataset = WindowedPVDataset(
        frame,
        history_columns=window_io.REGISTERED_HISTORY_FEATURES,
        future_known_columns=FUTURE_COLUMNS,
        site_metadata=site,
        history_mask=pd.DataFrame(
            history_mask, index=frame.index, columns=window_io.REGISTERED_HISTORY_FEATURES
        ),
        target_observed_mask=observed,
        stride=options["stride"],
        split=slices[split],
        require_observed_targets=True,
        causal_state=state,
    )
    times = origin_times(labels, dataset.window_starts, lookback=192, horizon=24)
    rows = dataset.window_starts[:, None] + 192 + np.arange(24)
    arrays = {
        "labels_ns": labels,
        "split_ids": split_ids,
        "power_observed": observed,
        "weather_observed": weather,
        "clear_sky_norm": raw[:, 1].copy(),
        "base_values_raw": raw,
        "effective_power_observed": state._effective_power_observed.copy(),
        "gap_left": state.gap_left.copy(),
        "gap_right": state.gap_right.copy(),
        "row_targets": raw[:, 0].copy(),
        "row_state_labels": frame["state_label"].to_numpy(dtype=np.int64),
        "row_future_known": frame.loc[:, FUTURE_COLUMNS[:-1]].to_numpy(dtype=np.float64),
        "site_vector": np.asarray(list(site.values()), dtype=np.float32),
        "candidate_starts": dataset.candidate_starts.copy(),
        "history_candidate_starts": dataset.history_candidate_starts.copy(),
        "window_starts": dataset.window_starts.copy(),
        "issue_ns": times.issue_ns,
        "timestamps": times.target_start_ns,
        "day_ids": times.target_start_ns // 86_400_000_000_000,
        "site_ids": np.full(len(dataset), options["site_index"], dtype=np.int64),
        "daylight_mask": raw[rows, 1] >= 0.01,
        "target_observed_mask": observed[rows],
    }
    sample = dataset.read_model_window_batch(np.arange(min(1, len(dataset)), dtype=np.int64))
    arrays.update(
        {
            name: contract.ResolverModelArray(
                dataset,
                name,
                {"shape": (len(dataset), *value.shape[1:]), "dtype": value.dtype},
            )
            for name, value in sample.items()
        }
    )
    identity = {
        "kind": "racepv.scoped-preparation-memory",
        "version": 1,
        "formal_ready": False,
        "manifest_entry": contract._strict_json_value(_json_bytes(asdict(entry))),
        "split_plan": copy.deepcopy(options["frozen_split_contract"]),
        "row_scope": scope,
        "source_time_contract": copy.deepcopy(options["source_time_contract"]),
        "selected_split": split,
        "allowed_splits": list(options["allowed_splits"]),
        "site_index": options["site_index"],
        "stride": options["stride"],
        "train_raw_input_sha256": options["train_raw_input_sha256"],
        "records_sha256": options["expected_records_sha256"],
        "fit_sha256": options["expected_fit_sha256"],
        "arrays": _array_identities(arrays),
    }
    return ScopedPreparation(dataset, arrays, identity)


def verify_scoped_preparation(
    candidate,
    entry,
    source_frame,
    *,
    train_source_frame,
    expected_train_records_sha256,
    expected_preparation_sha256,
    **supplied_options,
):
    """Recompute fit and windows from separately supplied pinned in-memory inputs.

    This is not source registration or original-file extraction acceptance. The
    caller must provide those external trust roots before any formal publication.
    """
    contract._sha(expected_preparation_sha256, "expected_preparation_sha256")
    if type(candidate) is not ScopedPreparation or type(candidate.dataset) is not WindowedPVDataset:
        raise TypeError("candidate must be a scoped engineering preparation")
    if contract.canonical_json_sha256(candidate.identity) != expected_preparation_sha256:
        raise ValueError("preparation identity SHA-256 mismatch")
    options, _, _ = _admit(entry, supplied_options)
    train = _sample(entry, train_source_frame, options, "train", expected_train_records_sha256)
    recomputed_fit = build_scoped_train_fit(
        entry,
        train,
        frozen_split_contract=options["frozen_split_contract"],
        source_time_contract=options["source_time_contract"],
        raw_input_sha256=options["train_raw_input_sha256"],
        site_index=options["site_index"],
        ramp_method=options["ramp_method"],
        allow_approximate_solar=options["allow_approximate_solar"],
    )
    if recomputed_fit != options["frozen_fit_blob"]:
        raise ValueError("recomputed train fit differs from frozen evidence")
    expected = prepare_scoped_split(entry, source_frame, **options)
    contract._exact(candidate.identity, expected.identity, "recomputed preparation")
    contract._keys(candidate.arrays, expected.arrays, "candidate arrays")
    contract._exact(
        _array_identities(candidate.arrays), expected.identity["arrays"], "actual candidate arrays"
    )
    if len(candidate.dataset) != len(expected.dataset):
        raise ValueError("candidate Dataset population differs from recomputation")
    for start in range(0, len(expected.dataset), 64):
        positions = np.arange(start, min(start + 64, len(expected.dataset)), dtype=np.int64)
        actual_batch = candidate.dataset.read_model_window_batch(positions)
        expected_batch = expected.dataset.read_model_window_batch(positions)
        contract._keys(actual_batch, window_io.MODEL_ARRAYS, "candidate batch")
        for name in window_io.MODEL_ARRAYS:
            contract._same(
                actual_batch[name], expected_batch[name], "recomputed model tensor " + name
            )
    return {
        "preparation_sha256": expected_preparation_sha256,
        "fit_recomputed": True,
        "preparation_recomputed": True,
        "source_interpretation_verified": False,
        "extraction_completeness_verified": False,
        "formal_ready": False,
    }
