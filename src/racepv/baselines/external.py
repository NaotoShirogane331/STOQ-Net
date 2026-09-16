"""Audited adapters for official NeuralForecast baselines.

This module never substitutes a local approximation for N-HiTS, TFT, or xLSTM.  Model
construction imports the maintained classes from ``neuralforecast`` lazily.
The dataframe adapter accepts legacy NPZ archives and the registered mmap/shard
stores, but it reads a sharded collection one leaf at a time and materialises
only bounded prediction batches.

NeuralForecast consumes every ``futr_exog_list`` column over both the input and
forecast portions of a training window.  Registered sequence stores preserve
the historical values of deterministic solar/time features in ``base_future``;
those exact values are therefore used.  Legacy pre-windowed stores only expose
historical clear-sky power, so their audited fallback is clear-sky-only.  The
issue-relative ``lead_time_hours`` feature is deliberately excluded because it
has no unique value on a deduplicated continuous training panel.

Official capability references (checked 2026-08-04):

* https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/exogenous_variables.html
* https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/overview.html
* https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/patchtst.py
* https://nixtlaverse.nixtla.io/neuralforecast/models.xlstm.html
* https://nixtlaverse.nixtla.io/neuralforecast/models.timexer.html
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pandas as pd

from racepv.audit import audit_future_feature_names
from racepv.io import (
    REGISTERED_FUTURE_FEATURES,
    REGISTERED_HISTORY_FEATURES,
    REGISTERED_SITE_FEATURES,
    REGISTERED_STATE_LABELS,
    NpySequenceWindowDataset,
    ShardedWindowDataset,
    WindowDataset,
    read_model_window_batch,
)

OFFICIAL_RUNNER_MODELS = ("nhits", "tft", "xlstm")
OFFICIAL_MODEL_ALIASES = {
    "nhits": "NHITS_official",
    "tft": "TFT_official",
    "temporal_fusion_transformer": "TFT_official",
    "xlstm": "xLSTM_official",
}

# Every feature below is deterministic at issue time and is stored as a
# row-level sequence by the lazy preparation path.  lead_time_hours is omitted:
# the same timestamp has different lead times for different forecast origins.
FUTURE_SOURCE_FEATURES = tuple(
    name for name in REGISTERED_FUTURE_FEATURES if name != "lead_time_hours"
)
FUTURE_EXOGENOUS = tuple(f"futr__{name}" for name in FUTURE_SOURCE_FEATURES)

_HISTORY_VALUE_SOURCE_FEATURES = tuple(
    name
    for name in REGISTERED_HISTORY_FEATURES
    if name not in {"power_norm", "clear_sky_power_norm"}
)
HISTORICAL_VALUE_EXOGENOUS = tuple(f"hist__{name}" for name in _HISTORY_VALUE_SOURCE_FEATURES)
HISTORICAL_MASK_EXOGENOUS = tuple(f"hist_mask__{name}" for name in REGISTERED_HISTORY_FEATURES)
HISTORICAL_EXOGENOUS = HISTORICAL_VALUE_EXOGENOUS + HISTORICAL_MASK_EXOGENOUS
STATIC_EXOGENOUS = tuple(f"stat__{name}" for name in REGISTERED_SITE_FEATURES)

FREQUENCY_MINUTES = 15
FREQUENCY = "15min"
FREQUENCY_NS = FREQUENCY_MINUTES * 60 * 1_000_000_000
LOOKBACK = 192
HORIZON = 24
_AUDIT_BATCH_WINDOWS = 8_192


@dataclass(frozen=True)
class FutureExogenousPlan:
    """Exact deterministic columns available on both sides of every window."""

    source_features: tuple[str, ...]
    columns: tuple[str, ...]
    storage_mode: str
    omitted_features: Mapping[str, str]


@dataclass(frozen=True)
class TrainingPanel:
    """A deduplicated official-NeuralForecast training panel and its audit."""

    frame: pd.DataFrame
    static_frame: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class PredictionFrames:
    """One bounded batch of issue-time contexts for ``NeuralForecast.predict``."""

    history: pd.DataFrame
    future: pd.DataFrame
    static: pd.DataFrame
    unique_ids: tuple[str, ...]
    expected_timestamps: np.ndarray


@dataclass(frozen=True)
class _WindowGroup:
    dataset: WindowDataset
    site: str
    indices: np.ndarray
    origins: np.ndarray


def _normalise_model_name(name: str) -> str:
    value = name.strip().lower().replace("-", "_")
    return "tft" if value == "temporal_fusion_transformer" else value


def neuralforecast_version() -> str:
    """Return the installed package version without importing its heavy modules."""

    try:
        return version("neuralforecast")
    except PackageNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install race-pv[forecasting] to use official baselines") from exc


def load_neuralforecast_class():
    """Load the official high-level class and verify the dataframe API used here."""

    try:
        from neuralforecast import NeuralForecast
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install race-pv[forecasting] to use official baselines") from exc

    predict_parameters = inspect.signature(NeuralForecast.predict).parameters
    required_predict = {"df", "static_df", "futr_df"}
    missing_predict = sorted(required_predict.difference(predict_parameters))
    if missing_predict:
        raise RuntimeError(
            "the installed NeuralForecast predict API is too old for the audited runner; "
            f"missing parameters: {missing_predict}"
        )
    fit_parameters = inspect.signature(NeuralForecast.fit).parameters
    if not {"df", "static_df", "val_size"}.issubset(fit_parameters):
        raise RuntimeError(
            "the installed NeuralForecast fit API is incompatible with the audited runner"
        )
    return NeuralForecast


def build_neuralforecast_model(
    name: str,
    *,
    horizon: int = HORIZON,
    input_size: int = LOOKBACK,
    quantiles: Sequence[float] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
    historical_exogenous: Sequence[str] | None = None,
    future_exogenous: Sequence[str] | None = None,
    static_exogenous: Sequence[str] | None = None,
    max_steps: int = 1_000,
    val_check_steps: int = 100,
    early_stop_patience_steps: int = -1,
    random_seed: int = 13,
    alias: str | None = None,
    **kwargs: object,
):
    """Construct an official N-HiTS, TFT, or xLSTM with multi-quantile loss.

    ``max_steps`` is an optimizer-step budget, not an epoch budget.  Official
    PatchTST is rejected here because the maintained class currently declares
    historical, future, and static exogenous support as false.  Official TimeXer
    is deliberately excluded: it is a fixed-channel multivariate model, declares
    no future-exogenous support, and cannot use this runner's global/zero-shot
    per-site input contract without changing the forecasting task.
    """

    normalised = _normalise_model_name(name)
    if normalised in {"patchtst", "patchtst_qr", "patchtst_target_only"}:
        raise ValueError(
            "official NeuralForecast PatchTST does not support historical, future, or "
            "static exogenous variables; it is excluded from the same-input official runner"
        )
    if normalised in {"timexer", "time_xer"}:
        raise ValueError(
            "official NeuralForecast TimeXer is multivariate with a fixed n_series and "
            "declares EXOGENOUS_FUTR=False; it is excluded from the global/zero-shot "
            "same-input runner rather than being reported under a changed task"
        )
    if normalised not in OFFICIAL_RUNNER_MODELS:
        raise KeyError(
            f"unsupported official NeuralForecast baseline {name!r}; "
            f"choose from {OFFICIAL_RUNNER_MODELS}"
        )
    if horizon != HORIZON or input_size != LOOKBACK:
        raise ValueError("the registered official baseline interface requires L=192 and H=24")
    levels = tuple(float(value) for value in quantiles)
    if levels != tuple(sorted(set(levels))) or 0.5 not in levels:
        raise ValueError("quantiles must be unique, ordered, and include 0.5")
    if max_steps <= 0 or val_check_steps <= 0:
        raise ValueError("max_steps and val_check_steps must be positive")

    try:
        from neuralforecast.losses.pytorch import MQLoss
        from neuralforecast.models import NHITS, TFT, xLSTM
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install race-pv[forecasting] to use official baselines") from exc

    model_class = {"nhits": NHITS, "tft": TFT, "xlstm": xLSTM}[normalised]
    capabilities = {
        "future": bool(getattr(model_class, "EXOGENOUS_FUTR", False)),
        "historical": bool(getattr(model_class, "EXOGENOUS_HIST", False)),
        "static": bool(getattr(model_class, "EXOGENOUS_STAT", False)),
    }
    if not all(capabilities.values()):
        raise RuntimeError(
            f"installed official {model_class.__name__} lacks required exogenous support: "
            f"{capabilities}"
        )
    try:
        loss = MQLoss(quantiles=list(levels))
        valid_loss = MQLoss(quantiles=list(levels))
    except TypeError as exc:
        raise RuntimeError(
            "the installed NeuralForecast MQLoss does not support the quantiles= API"
        ) from exc

    common: dict[str, object] = {
        "h": horizon,
        "input_size": input_size,
        "loss": loss,
        "valid_loss": valid_loss,
        "max_steps": int(max_steps),
        "val_check_steps": int(val_check_steps),
        "early_stop_patience_steps": int(early_stop_patience_steps),
        "random_seed": int(random_seed),
        "alias": alias or OFFICIAL_MODEL_ALIASES[normalised],
        **kwargs,
    }
    if historical_exogenous:
        common["hist_exog_list"] = list(historical_exogenous)
    if future_exogenous:
        common["futr_exog_list"] = list(future_exogenous)
    if static_exogenous:
        common["stat_exog_list"] = list(static_exogenous)
    try:
        return model_class(**common)
    except ImportError as exc:
        if normalised == "xlstm":
            raise ImportError(
                "official NeuralForecast xLSTM requires the optional xlstm package; "
                "install race-pv[forecasting] in the training environment"
            ) from exc
        raise


def _leaf_datasets(dataset: WindowDataset) -> tuple[WindowDataset, ...]:
    schema = getattr(dataset, "schema", None) or {}
    if schema.get("schema_version") == 2 or schema.get("causality_profile") is not None:
        return (dataset,)
    if isinstance(dataset, ShardedWindowDataset):
        return tuple(dataset.shards)
    return (dataset,)


def reject_origin_dynamic_panel(dataset: WindowDataset) -> None:
    """Origin-dependent history cannot be deduplicated to a continuous panel."""
    for leaf in _leaf_datasets(dataset):
        schema = getattr(leaf, "schema", None) or {}
        if (
            schema.get("schema_version") == 2
            or schema.get("causality_profile") is not None
            or getattr(leaf, "prepared_contract", None) is not None
        ):
            raise ValueError("origin-dynamic prepared contracts cannot form a continuous panel")


def _validate_registered_schema_without_global_materialisation(
    dataset: WindowDataset,
) -> None:
    """Validate names/semantics while avoiding ``np.asarray(_ShardedArray)``."""

    reject_origin_dynamic_panel(dataset)
    if dataset.schema is None:
        raise ValueError("registered schema is required for the official baseline")
    schema = dataset.schema
    if schema.get("schema_version") != 1:
        raise ValueError("unsupported or missing prepared schema_version")
    if tuple(schema.get("history_columns", ())) != REGISTERED_HISTORY_FEATURES:
        raise ValueError("history schema does not match the registered feature order")
    if tuple(schema.get("future_known_columns", ())) != REGISTERED_FUTURE_FEATURES:
        raise ValueError("future-known schema does not match the registered feature order")
    if tuple(schema.get("site_metadata_names", ())) != REGISTERED_SITE_FEATURES:
        raise ValueError("site metadata schema does not match the registered feature order")
    if schema.get("target") != "power_norm":
        raise ValueError("registered target must be power_norm")
    if schema.get("state_labels") != REGISTERED_STATE_LABELS:
        raise ValueError("state-label mapping does not match the registered contract")
    for name in ("site_id_map", "spatial_roles", "ramp_thresholds"):
        if not isinstance(schema.get(name), dict) or not schema[name]:
            raise ValueError(f"registered schema is missing {name}")
    if not isinstance(dataset, ShardedWindowDataset):
        # Legacy/single stores have no global concatenation proxy, so the class's
        # full numeric audit is safe and remains useful.
        dataset.require_registered_schema()


def _schema_signature(dataset: WindowDataset) -> dict[str, Any]:
    _validate_registered_schema_without_global_materialisation(dataset)
    assert dataset.schema is not None
    keys = (
        "schema_version",
        "history_columns",
        "future_known_columns",
        "site_metadata_names",
        "target",
        "state_labels",
    )
    return {key: dataset.schema[key] for key in keys}


def _available_historical_future_features(dataset: WindowDataset) -> set[str]:
    available = set(FUTURE_SOURCE_FEATURES)
    for leaf in _leaf_datasets(dataset):
        if isinstance(leaf, NpySequenceWindowDataset):
            available.intersection_update(leaf.base_future_names)
        else:
            # Legacy window archives retain only historical clear-sky power.
            available.intersection_update({"clear_sky_power_norm"})
    return available


def resolve_future_exogenous_plan(
    datasets: Mapping[str, WindowDataset],
    *,
    allowed_future_features: Sequence[str],
) -> FutureExogenousPlan:
    """Choose the richest exact future-exogenous contract shared by all splits."""

    if not datasets:
        raise ValueError("at least one dataset is required to resolve future exogenous inputs")
    common = set(FUTURE_SOURCE_FEATURES)
    all_sequence = True
    for dataset in datasets.values():
        reject_origin_dynamic_panel(dataset)
        common.intersection_update(_available_historical_future_features(dataset))
        all_sequence &= all(
            isinstance(leaf, NpySequenceWindowDataset) for leaf in _leaf_datasets(dataset)
        )
    selected = tuple(name for name in FUTURE_SOURCE_FEATURES if name in common)
    if "clear_sky_power_norm" not in selected:
        raise ValueError("official baselines require exact historical and future clear-sky power")
    allowlist_audit = audit_future_feature_names(selected, allowed_future_features)
    allowlist_audit.require_pass()
    omitted: dict[str, str] = {}
    for name in REGISTERED_FUTURE_FEATURES:
        if name == "lead_time_hours":
            omitted[name] = (
                "issue-relative lead time has no unique value on a deduplicated continuous panel"
            )
        elif name not in selected:
            omitted[name] = (
                "an input split is a legacy/materialized window store without exact historical "
                "row-level values for this NeuralForecast future-exogenous column"
            )
    mode = "sequence_base_future" if all_sequence else "legacy_clear_sky_fallback"
    return FutureExogenousPlan(
        source_features=selected,
        columns=tuple(f"futr__{name}" for name in selected),
        storage_mode=mode,
        omitted_features=omitted,
    )


def _sequence_window_batch(
    dataset: NpySequenceWindowDataset,
    start: int,
    stop: int,
    names: Sequence[str],
) -> dict[str, np.ndarray]:
    return read_window_batch(dataset, start, stop, names)


def read_window_batch(
    dataset: WindowDataset,
    start: int,
    stop: int,
    names: Sequence[str],
) -> dict[str, np.ndarray]:
    """Read a bounded contiguous batch without expanding a sharded virtual array."""

    for value in (start, stop):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError("window batch bounds must be integers")
    if not 0 <= start <= stop <= len(dataset):
        raise ValueError("window batch bounds are invalid")
    return read_model_window_batch(dataset, range(start, stop), names)


def _read_leaf_indices(dataset: WindowDataset, name: str, indices: np.ndarray) -> np.ndarray:
    if isinstance(dataset, ShardedWindowDataset):
        raise TypeError("_read_leaf_indices must receive a non-sharded leaf")
    return read_model_window_batch(dataset, indices, [name])[name]


def _historical_future_batch(
    dataset: WindowDataset,
    start: int,
    stop: int,
    source_features: Sequence[str],
) -> np.ndarray:
    features = tuple(source_features)
    if isinstance(dataset, ShardedWindowDataset):
        pieces = []
        for shard_index, shard in enumerate(dataset.shards):
            shard_start = int(dataset._offsets[shard_index])
            shard_stop = int(dataset._offsets[shard_index + 1])
            overlap_start = max(start, shard_start)
            overlap_stop = min(stop, shard_stop)
            if overlap_start < overlap_stop:
                pieces.append(
                    _historical_future_batch(
                        shard,
                        overlap_start - shard_start,
                        overlap_stop - shard_start,
                        features,
                    )
                )
        if not pieces:
            raise RuntimeError("no shard covered historical future-exogenous batch")
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)
    if isinstance(dataset, NpySequenceWindowDataset):
        missing = sorted(set(features).difference(dataset.base_future_names))
        if missing:
            raise ValueError(f"sequence store lacks historical deterministic features: {missing}")
        starts = np.asarray(dataset.window_starts[start:stop], dtype=np.int64)
        positions = starts[:, None] + np.arange(LOOKBACK, dtype=np.int64)[None, :]
        source_indices = [dataset.base_future_names.index(name) for name in features]
        return np.asarray(
            dataset.sequence_arrays["base_future"][positions][..., source_indices],
            dtype=np.float32,
        )
    if features != ("clear_sky_power_norm",):
        raise ValueError("legacy windows only support exact historical clear-sky future exogenous")
    batch = read_window_batch(dataset, start, stop, ("history_values", "history_mask"))
    history = batch["history_values"]
    masks = batch["history_mask"]
    index = REGISTERED_HISTORY_FEATURES.index("clear_sky_power_norm")
    return np.where(
        np.asarray(masks[..., index : index + 1]) > 0,
        np.asarray(history[..., index : index + 1], dtype=np.float32),
        0.0,
    ).astype(np.float32, copy=False)


def _canonical_site_groups(dataset: WindowDataset) -> list[_WindowGroup]:
    _validate_registered_schema_without_global_materialisation(dataset)
    assert dataset.schema is not None
    site_map = {str(int(key)): str(value) for key, value in dataset.schema["site_id_map"].items()}
    if len(set(site_map.values())) != len(site_map):
        raise ValueError("schema.site_id_map values must be unique")
    groups: list[_WindowGroup] = []
    observed_ids: set[str] = set()
    for leaf in _leaf_datasets(dataset):
        if isinstance(leaf, NpySequenceWindowDataset):
            raw_ids = np.full(len(leaf), leaf.site_index, dtype=np.int64)
            starts = np.asarray(leaf.window_starts, dtype=np.int64)
            origins = np.asarray(
                leaf.sequence_arrays["timestamps"][starts + LOOKBACK], dtype=np.int64
            )
        else:
            raw_ids = np.asarray(leaf.arrays["site_ids"], dtype=np.int64).reshape(-1)
            origins = np.asarray(leaf.arrays["timestamps"][:, 0], dtype=np.int64)
        if raw_ids.shape != origins.shape or raw_ids.size != len(leaf):
            raise ValueError("site IDs and forecast origins must contain one value per window")
        for raw_id in np.unique(raw_ids):
            key = str(int(raw_id))
            if key not in site_map:
                raise ValueError(f"site id {key} is absent from schema.site_id_map")
            observed_ids.add(key)
            indices = np.flatnonzero(raw_ids == raw_id)
            order = np.argsort(origins[indices], kind="stable")
            ordered = indices[order]
            groups.append(
                _WindowGroup(
                    dataset=leaf,
                    site=site_map[key],
                    indices=ordered.astype(np.int64, copy=False),
                    origins=origins[ordered].astype(np.int64, copy=False),
                )
            )
    if observed_ids != set(site_map):
        raise ValueError("schema.site_id_map must exactly match observed lazy shards")
    for name in ("spatial_roles", "ramp_thresholds"):
        declared = {str(int(key)) for key in dataset.schema[name]}
        if declared != observed_ids:
            raise ValueError(f"schema.{name} must exactly match observed lazy shards")
    return groups


def _datetime_index(values: np.ndarray) -> pd.DatetimeIndex:
    timestamps = pd.to_datetime(np.asarray(values, dtype=np.int64), unit="ns", utc=True)
    return pd.DatetimeIndex(timestamps).tz_convert(None)


def _validate_window_times(dataset: WindowDataset, split: str) -> dict[str, Any]:
    site_bounds: dict[str, dict[str, Any]] = {}
    site_origins: dict[str, list[np.ndarray]] = {}
    site_ends: dict[str, list[int]] = {}
    for group in _canonical_site_groups(dataset):
        if group.origins.size > 1 and np.any(np.diff(group.origins) <= 0):
            raise ValueError(
                f"{split}/{group.site}: forecast origins must be unique and increasing"
            )
        maximum_end = np.iinfo(np.int64).min
        for start in range(0, group.indices.size, _AUDIT_BATCH_WINDOWS):
            indices = group.indices[start : start + _AUDIT_BATCH_WINDOWS]
            timestamps = _read_leaf_indices(group.dataset, "timestamps", indices).astype(
                np.int64, copy=False
            )
            if timestamps.shape != (indices.size, HORIZON):
                raise ValueError(f"{split}: timestamps must have shape [N,24]")
            expected = timestamps[:, :1] + FREQUENCY_NS * np.arange(HORIZON)[None, :]
            if not np.array_equal(timestamps, expected):
                raise ValueError(
                    f"{split}: every forecast horizon must be a regular 15-minute grid"
                )
            maximum_end = max(maximum_end, int(timestamps[:, -1].max()))
        site_origins.setdefault(group.site, []).append(group.origins)
        site_ends.setdefault(group.site, []).append(maximum_end)

    for site, origin_parts in site_origins.items():
        origins = np.sort(np.concatenate(origin_parts))
        if origins.size > 1 and np.any(np.diff(origins) <= 0):
            raise ValueError(f"{split}/{site}: duplicate origins occur across storage shards")
        target_end = max(site_ends[site])
        history_start = int(origins[0] - LOOKBACK * FREQUENCY_NS)
        site_bounds[site] = {
            "origins": int(origins.size),
            "history_start_ns": history_start,
            "target_start_ns": int(origins[0]),
            "target_end_ns": target_end,
            "history_start_utc": _datetime_index(np.asarray([history_start]))[0].isoformat(),
            "target_start_utc": _datetime_index(origins[:1])[0].isoformat(),
            "target_end_utc": _datetime_index(np.asarray([target_end]))[0].isoformat(),
        }
    return {"sites": site_bounds, "site_count": len(site_bounds), "windows": len(dataset)}


def validate_archive_contract(
    datasets: Mapping[str, WindowDataset],
    *,
    allowed_future_features: Sequence[str],
    future_plan: FutureExogenousPlan | None = None,
    validation_only: bool = False,
) -> dict[str, Any]:
    """Audit schemas, deterministic inputs, roles, and chronological boundaries."""

    for dataset in datasets.values():
        reject_origin_dynamic_panel(dataset)

    expected_splits = (
        ("train", "validation")
        if validation_only
        else ("train", "validation", "calibration", "test")
    )
    missing = [name for name in expected_splits if name not in datasets]
    if missing:
        raise ValueError(f"missing required archives: {missing}")
    plan = future_plan or resolve_future_exogenous_plan(
        datasets, allowed_future_features=allowed_future_features
    )
    train_signature: dict[str, Any] | None = None
    split_audits: dict[str, Any] = {}
    for split in expected_splits:
        dataset = datasets[split]
        declared = getattr(dataset, "collection_split", None)
        if declared is not None and declared != split:
            raise ValueError(f"{split} input points to a collection declared as {declared!r}")
        signature = _schema_signature(dataset)
        if train_signature is None:
            train_signature = signature
        elif signature != train_signature:
            raise ValueError("all official baseline archives must share the registered schema")
        split_audits[split] = _validate_window_times(dataset, split)

    future_audit = audit_future_feature_names(plan.source_features, allowed_future_features)
    future_audit.require_pass()

    chronology_checks: list[dict[str, Any]] = []
    for earlier, later in zip(expected_splits, expected_splits[1:], strict=False):
        earlier_sites = split_audits[earlier]["sites"]
        later_sites = split_audits[later]["sites"]
        for site in sorted(set(earlier_sites).intersection(later_sites)):
            prior_end = int(earlier_sites[site]["target_end_ns"])
            next_history_start = int(later_sites[site]["history_start_ns"])
            passed = prior_end < next_history_start
            chronology_checks.append(
                {
                    "site": site,
                    "earlier_split": earlier,
                    "later_split": later,
                    "prior_target_end_ns": prior_end,
                    "next_history_start_ns": next_history_start,
                    "passed": passed,
                }
            )
            if not passed:
                raise ValueError(
                    f"{site}: {later} history overlaps {earlier} targets; "
                    "the split archive crosses an information boundary"
                )

    train_sites = set(split_audits["train"]["sites"])
    transfer = {
        split: sorted(set(split_audits[split]["sites"]).difference(train_sites))
        for split in expected_splits
        if split != "train"
    }
    return {
        "scope": "validation_only" if validation_only else "formal_four_split",
        "registered_schema_signature": train_signature,
        "splits": split_audits,
        "same_site_chronology_checks": chronology_checks,
        "future_feature_allowlist": {
            "passed": future_audit.passed,
            "source_features_used": list(plan.source_features),
            "neuralforecast_columns": list(plan.columns),
            "registered_allowed_features": list(allowed_future_features),
            "storage_mode": plan.storage_mode,
            "omitted_features": dict(plan.omitted_features),
        },
        "unseen_sites_relative_to_train": transfer,
    }


def infer_origin_stride_steps(dataset: WindowDataset) -> int:
    """Infer the preparation stride as the GCD of within-shard origin differences."""

    differences: list[np.ndarray] = []
    for group in _canonical_site_groups(dataset):
        if group.origins.size < 2:
            continue
        delta_ns = np.diff(group.origins)
        if np.any(delta_ns % FREQUENCY_NS != 0):
            raise ValueError(f"{group.site}: origin differences are not multiples of 15 minutes")
        differences.append((delta_ns // FREQUENCY_NS).astype(np.int64))
    if not differences:
        return 1
    stride = int(np.gcd.reduce(np.concatenate(differences)))
    if stride <= 0 or stride > LOOKBACK:
        raise ValueError(f"could not infer a valid origin stride in [1,{LOOKBACK}]")
    return stride


def _consecutive_segments(group: _WindowGroup, stride_steps: int) -> list[np.ndarray]:
    if group.indices.size == 0:
        return []
    if group.indices.size == 1:
        return [group.indices]
    expected = stride_steps * FREQUENCY_NS
    split_points = np.flatnonzero(np.diff(group.origins) != expected) + 1
    return [part for part in np.split(group.indices, split_points) if part.size]


def estimate_training_panel_rows(
    dataset: WindowDataset, *, origin_stride_steps: int | None = None
) -> dict[str, Any]:
    """Estimate deduplicated rows before allocating a pandas training panel."""

    stride = origin_stride_steps or infer_origin_stride_steps(dataset)
    if stride <= 0 or stride > LOOKBACK:
        raise ValueError(f"origin_stride_steps must be in [1,{LOOKBACK}]")
    rows = 0
    segments = 0
    per_site: dict[str, dict[str, int]] = {}
    for group in _canonical_site_groups(dataset):
        site_state = per_site.setdefault(
            group.site, {"windows": 0, "segments": 0, "deduplicated_rows": 0}
        )
        group_segments = _consecutive_segments(group, stride)
        for segment in group_segments:
            segment_rows = LOOKBACK + (int(segment.size) - 1) * stride + HORIZON
            rows += segment_rows
            site_state["deduplicated_rows"] += segment_rows
        site_state["windows"] += int(group.indices.size)
        site_state["segments"] += len(group_segments)
        segments += len(group_segments)
    naive_rows = int(len(dataset) * (LOOKBACK + HORIZON))
    numeric_columns = 3 + len(HISTORICAL_EXOGENOUS) + len(FUTURE_EXOGENOUS)
    return {
        "origin_stride_steps": stride,
        "origin_stride_minutes": stride * FREQUENCY_MINUTES,
        "deduplicated_rows": rows,
        "naive_window_rows": naive_rows,
        "row_reduction_ratio": float(rows / naive_rows) if naive_rows else math.nan,
        "segments": segments,
        "sites": per_site,
        "numeric_payload_bytes_lower_bound": int(rows * numeric_columns * 4),
    }


def _assert_overlap_consistency(
    existing_values: np.ndarray,
    existing_masks: np.ndarray,
    existing_future: np.ndarray,
    current_values: np.ndarray,
    current_masks: np.ndarray,
    current_future: np.ndarray,
    overlap: int,
    *,
    site: str,
) -> None:
    if overlap <= 0:
        return
    if not np.allclose(
        existing_values[-overlap:],
        current_values[:overlap],
        rtol=1e-5,
        atol=1e-6,
        equal_nan=True,
    ):
        raise ValueError(f"{site}: overlapping history values disagree across windows")
    if not np.array_equal(existing_masks[-overlap:], current_masks[:overlap]):
        raise ValueError(f"{site}: overlapping history masks disagree across windows")
    if not np.allclose(existing_future[-overlap:], current_future[:overlap], rtol=1e-5, atol=1e-6):
        raise ValueError(f"{site}: overlapping deterministic future values disagree")


def _panel_frame(
    *,
    site: str,
    segment_number: int,
    values: np.ndarray,
    masks: np.ndarray,
    future_values: np.ndarray,
    times: np.ndarray,
    site_values: np.ndarray,
    training_available: np.ndarray,
    plan: FutureExogenousPlan,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_count = len(times)
    training_available = np.asarray(training_available, dtype=np.float32)
    if (
        training_available.shape != (row_count,)
        or not np.isin(training_available, (0.0, 1.0)).all()
    ):
        raise ValueError(f"{site}: training availability must be one binary value per row")
    if np.any(np.abs(values[training_available == 0, 0]) > 1e-7):
        raise ValueError(f"{site}: unavailable autoregressive y must be exactly zero")
    history_indices = [
        REGISTERED_HISTORY_FEATURES.index(name) for name in _HISTORY_VALUE_SOURCE_FEATURES
    ]
    historical_values = values[:, history_indices]
    historical_masks = masks[:, history_indices]
    if np.any(np.abs(historical_values[historical_masks == 0]) > 1e-7):
        raise ValueError(f"{site}: masked historical exogenous values must be exactly zero")
    unique_id = f"{site}::train_segment_{segment_number:04d}"
    data: dict[str, Any] = {
        "unique_id": np.repeat(unique_id, row_count),
        "ds": _datetime_index(times),
        "y": values[:, 0],
        "available_mask": training_available,
    }
    for column, source_index in zip(HISTORICAL_VALUE_EXOGENOUS, history_indices, strict=True):
        data[column] = values[:, source_index]
    for source_index, column in enumerate(HISTORICAL_MASK_EXOGENOUS):
        data[column] = masks[:, source_index]
    for source_index, column in enumerate(plan.columns):
        data[column] = future_values[:, source_index]
    frame = pd.DataFrame(data)
    numeric = frame.drop(columns=["unique_id", "ds"]).to_numpy(dtype=np.float32)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{site}: non-finite value in official training dataframe")
    static_data: dict[str, Any] = {"unique_id": [unique_id]}
    for source_index, column in enumerate(STATIC_EXOGENOUS):
        static_data[column] = [float(site_values[source_index])]
    return frame, pd.DataFrame(static_data)


def _sequence_segment_frames(
    dataset: NpySequenceWindowDataset,
    site: str,
    indices: np.ndarray,
    *,
    segment_number: int,
    stride_steps: int,
    plan: FutureExogenousPlan,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    starts = np.asarray(dataset.window_starts[indices], dtype=np.int64)
    if starts.size > 1 and not np.all(np.diff(starts) == stride_steps):
        raise ValueError(f"{site}: sequence window positions disagree with timestamp stride")
    row_start = int(starts[0])
    row_stop = int(starts[-1] + LOOKBACK + HORIZON)
    expected_rows = LOOKBACK + (len(starts) - 1) * stride_steps + HORIZON
    if row_stop - row_start != expected_rows:
        raise RuntimeError("sequence training-panel row accounting failed")
    arrays = dataset.sequence_arrays
    values = np.asarray(arrays["history_values"][row_start:row_stop], dtype=np.float32)
    masks = np.asarray(arrays["history_mask"][row_start:row_stop], dtype=np.float32)
    values = np.where(masks > 0, values, 0.0).astype(np.float32, copy=False)
    targets = np.asarray(arrays["targets"][row_start:row_stop], dtype=np.float32)
    observed = masks[:, 0] > 0
    if observed.any() and not np.allclose(
        values[observed, 0], targets[observed], rtol=1e-5, atol=1e-6
    ):
        raise ValueError(f"{site}: row-level targets disagree with observed power history")
    training_available = masks[:, 0].copy()
    for position in range(len(starts)):
        origin = LOOKBACK + position * stride_steps
        training_available[origin : origin + HORIZON] = 1.0
    # A masked history value must never re-enter through autoregressive y.  Exact
    # row targets are restored only where y is legitimately available: observed
    # history or a registered target horizon.
    values = values.copy()
    values[:, 0] = np.where(training_available > 0, targets, 0.0)
    source_indices = [dataset.base_future_names.index(name) for name in plan.source_features]
    future_values = np.asarray(
        arrays["base_future"][row_start:row_stop, source_indices], dtype=np.float32
    )
    times = np.asarray(arrays["timestamps"][row_start:row_stop], dtype=np.int64)
    if times.size > 1 and not np.all(np.diff(times) == FREQUENCY_NS):
        raise ValueError(f"{site}: sequence segment is not a regular 15-minute series")
    return _panel_frame(
        site=site,
        segment_number=segment_number,
        values=values,
        masks=masks,
        future_values=future_values,
        times=times,
        site_values=np.asarray(arrays["site_metadata"], dtype=np.float32),
        training_available=training_available,
        plan=plan,
    )


def _legacy_segment_frames(
    dataset: WindowDataset,
    site: str,
    indices: np.ndarray,
    *,
    segment_number: int,
    stride_steps: int,
    plan: FutureExogenousPlan,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(dataset, ShardedWindowDataset):
        raise TypeError("legacy segment builder must receive one storage leaf")
    count = int(indices.size)
    row_count = LOOKBACK + (count - 1) * stride_steps + HORIZON
    values = np.zeros((row_count, len(REGISTERED_HISTORY_FEATURES)), dtype=np.float32)
    masks = np.zeros_like(values)
    future_values = np.zeros((row_count, len(plan.source_features)), dtype=np.float32)
    times = np.empty(row_count, dtype=np.int64)

    first = int(indices[0])
    sample = read_window_batch(
        dataset,
        first,
        first + 1,
        ("history_values", "history_mask", "site_metadata"),
    )
    first_future = _historical_future_batch(dataset, first, first + 1, plan.source_features)[0]
    first_times = read_window_batch(dataset, first, first + 1, ("timestamps",))["timestamps"][0]
    values[:LOOKBACK] = sample["history_values"][0]
    masks[:LOOKBACK] = sample["history_mask"][0]
    values[:LOOKBACK] = np.where(masks[:LOOKBACK] > 0, values[:LOOKBACK], 0.0)
    future_values[:LOOKBACK] = first_future
    origins = np.empty(count, dtype=np.int64)
    origins[0] = int(first_times[0])
    times[:LOOKBACK] = origins[0] + FREQUENCY_NS * np.arange(-LOOKBACK, 0)
    write = LOOKBACK
    previous_targets = np.asarray(
        read_window_batch(dataset, first, first + 1, ("targets",))["targets"][0],
        dtype=np.float32,
    )
    registered_targets = np.zeros(row_count, dtype=np.float32)
    registered_target_mask = np.zeros(row_count, dtype=bool)
    registered_targets[LOOKBACK : LOOKBACK + HORIZON] = previous_targets
    registered_target_mask[LOOKBACK : LOOKBACK + HORIZON] = True
    site_values = np.asarray(sample["site_metadata"][0], dtype=np.float32)

    for position in range(1, count):
        index = int(indices[position])
        batch = read_window_batch(
            dataset,
            index,
            index + 1,
            ("history_values", "history_mask", "timestamps", "targets", "site_metadata"),
        )
        current_values = np.asarray(batch["history_values"][0], dtype=np.float32)
        current_masks = np.asarray(batch["history_mask"][0], dtype=np.float32)
        current_values = np.where(current_masks > 0, current_values, 0.0).astype(
            np.float32, copy=False
        )
        current_future = _historical_future_batch(dataset, index, index + 1, plan.source_features)[
            0
        ]
        if not np.allclose(
            np.asarray(batch["site_metadata"][0], dtype=np.float32),
            site_values,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError(f"{site}: site metadata changes across forecast origins")
        origins[position] = int(batch["timestamps"][0, 0])
        delta_steps = int((origins[position] - origins[position - 1]) // FREQUENCY_NS)
        if delta_steps != stride_steps:
            raise RuntimeError("non-consecutive windows reached a segment builder")
        overlap = LOOKBACK - delta_steps
        _assert_overlap_consistency(
            values[:write],
            masks[:write],
            future_values[:write],
            current_values,
            current_masks,
            current_future,
            overlap,
            site=site,
        )
        comparable = min(delta_steps, HORIZON)
        observed = (
            current_masks[LOOKBACK - delta_steps : LOOKBACK - delta_steps + comparable, 0] > 0
        )
        if observed.any() and not np.allclose(
            current_values[LOOKBACK - delta_steps : LOOKBACK - delta_steps + comparable, 0][
                observed
            ],
            previous_targets[:comparable][observed],
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError(f"{site}: training targets disagree with later observed history")
        values[write : write + delta_steps] = current_values[-delta_steps:]
        masks[write : write + delta_steps] = current_masks[-delta_steps:]
        future_values[write : write + delta_steps] = current_future[-delta_steps:]
        times[write : write + delta_steps] = origins[position] + FREQUENCY_NS * np.arange(
            -delta_steps, 0
        )
        write += delta_steps
        current_targets = np.asarray(batch["targets"][0], dtype=np.float32)
        target_start = LOOKBACK + position * stride_steps
        target_stop = target_start + HORIZON
        already_defined = registered_target_mask[target_start:target_stop]
        if already_defined.any() and not np.allclose(
            registered_targets[target_start:target_stop][already_defined],
            current_targets[already_defined],
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError(f"{site}: overlapping registered targets disagree")
        registered_targets[target_start:target_stop] = current_targets
        registered_target_mask[target_start:target_stop] = True
        previous_targets = current_targets

    last_index = int(indices[-1])
    last = read_window_batch(
        dataset,
        last_index,
        last_index + 1,
        ("targets", "future_known", "timestamps", "site_metadata"),
    )
    future_indices = [REGISTERED_FUTURE_FEATURES.index(name) for name in plan.source_features]
    last_target_start = LOOKBACK + (count - 1) * stride_steps
    last_targets = np.asarray(last["targets"][0], dtype=np.float32)
    if not np.allclose(
        registered_targets[last_target_start : last_target_start + HORIZON],
        last_targets,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError(f"{site}: final registered targets changed during panel assembly")
    future_values[write : write + HORIZON] = last["future_known"][0][:, future_indices]
    times[write : write + HORIZON] = last["timestamps"][0]
    write += HORIZON
    if write != row_count:
        raise RuntimeError("legacy training-panel row accounting failed")
    if times.size > 1 and not np.all(np.diff(times) == FREQUENCY_NS):
        raise ValueError(f"{site}: deduplicated segment is not a regular 15-minute series")
    if not np.allclose(
        np.asarray(last["site_metadata"][0], dtype=np.float32),
        site_values,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError(f"{site}: site metadata changes across forecast origins")
    training_available = masks[:, 0].copy()
    training_available[registered_target_mask] = 1.0
    values[:, 0] = np.where(training_available > 0, values[:, 0], 0.0)
    values[registered_target_mask, 0] = registered_targets[registered_target_mask]
    return _panel_frame(
        site=site,
        segment_number=segment_number,
        values=values,
        masks=masks,
        future_values=future_values,
        times=times,
        site_values=site_values,
        training_available=training_available,
        plan=plan,
    )


def build_training_panel(
    dataset: WindowDataset,
    *,
    future_plan: FutureExogenousPlan | None = None,
    origin_stride_steps: int | None = None,
    max_panel_rows: int = 3_100_000,
) -> TrainingPanel:
    """Deduplicate train windows into regular segments under a pre-allocation guard."""

    _validate_registered_schema_without_global_materialisation(dataset)
    plan = future_plan or resolve_future_exogenous_plan(
        {"train": dataset}, allowed_future_features=REGISTERED_FUTURE_FEATURES
    )
    estimate = estimate_training_panel_rows(dataset, origin_stride_steps=origin_stride_steps)
    if max_panel_rows <= 0:
        raise ValueError("max_panel_rows must be positive")
    if int(estimate["deduplicated_rows"]) > max_panel_rows:
        raise MemoryError(
            "deduplicated official-NeuralForecast panel would contain "
            f"{estimate['deduplicated_rows']:,} rows, exceeding --max-panel-rows="
            f"{max_panel_rows:,}; run per site/region or raise the limit explicitly"
        )
    stride = int(estimate["origin_stride_steps"])
    frames: list[pd.DataFrame] = []
    static_frames: list[pd.DataFrame] = []
    segment_counter = 0
    for group in _canonical_site_groups(dataset):
        for segment in _consecutive_segments(group, stride):
            if isinstance(group.dataset, NpySequenceWindowDataset):
                frame, static = _sequence_segment_frames(
                    group.dataset,
                    group.site,
                    segment,
                    segment_number=segment_counter,
                    stride_steps=stride,
                    plan=plan,
                )
            else:
                frame, static = _legacy_segment_frames(
                    group.dataset,
                    group.site,
                    segment,
                    segment_number=segment_counter,
                    stride_steps=stride,
                    plan=plan,
                )
            frames.append(frame)
            static_frames.append(static)
            segment_counter += 1
    if not frames:
        raise ValueError("train archive produced no official-NeuralForecast segments")
    panel = pd.concat(frames, ignore_index=True)
    static_panel = pd.concat(static_frames, ignore_index=True)
    if len(panel) != int(estimate["deduplicated_rows"]):
        raise RuntimeError("training panel size differs from its pre-allocation audit")
    if panel.duplicated(["unique_id", "ds"]).any():
        raise ValueError("training panel contains duplicate series timestamps")
    audit = {
        **estimate,
        "max_panel_rows": int(max_panel_rows),
        "guard_checked_before_dataframe_construction": True,
        "expected_registered_training_windows": int(len(dataset)),
        "neuralforecast_window_filter": {
            "start_padding_enabled": False,
            "training_data_availability_threshold": [0.0, 1.0],
            "purpose": (
                "retain registered origins with at least one available history point and "
                "exclude right-padded origins lacking a complete 24-step target"
            ),
        },
        "history_exogenous": list(HISTORICAL_EXOGENOUS),
        "history_source_features": list(_HISTORY_VALUE_SOURCE_FEATURES),
        "history_mask_source_features": list(REGISTERED_HISTORY_FEATURES),
        "future_exogenous": list(plan.columns),
        "future_source_features": list(plan.source_features),
        "future_storage_mode": plan.storage_mode,
        "omitted_future_features": dict(plan.omitted_features),
        "static_exogenous": list(STATIC_EXOGENOUS),
        "construction": (
            "overlapping windows are deduplicated per storage shard into regular segments; "
            "sequence stores are sliced directly without materialising virtual windows"
        ),
        "target_availability": (
            "available_mask is forced to one only on the union of registered target "
            "horizons; historical feature masks retain original missingness"
        ),
        "masked_history_values": (
            "every historical feature cell with history_mask=0 is replaced by zero before "
            "constructing NeuralForecast dataframes"
        ),
    }
    return TrainingPanel(frame=panel, static_frame=static_panel, audit=audit)


def build_prediction_frames(
    dataset: WindowDataset,
    *,
    split: str,
    start: int,
    stop: int,
    future_plan: FutureExogenousPlan | None = None,
) -> PredictionFrames:
    """Build one bounded inference dataframe without reading future targets."""

    if not 0 <= start < stop <= len(dataset):
        raise ValueError("prediction chunk bounds are invalid")
    plan = future_plan or resolve_future_exogenous_plan(
        {split: dataset}, allowed_future_features=REGISTERED_FUTURE_FEATURES
    )
    batch = read_window_batch(
        dataset,
        start,
        stop,
        ("history_values", "history_mask", "future_known", "timestamps", "site_metadata"),
    )
    values = np.asarray(batch["history_values"], dtype=np.float32)
    masks = np.asarray(batch["history_mask"], dtype=np.float32)
    values = np.where(masks > 0, values, 0.0).astype(np.float32, copy=False)
    if np.any(np.abs(values[masks == 0]) > 1e-7):
        raise RuntimeError("masked prediction-history cells were not zeroed")
    future_known = np.asarray(batch["future_known"], dtype=np.float32)
    target_times = np.asarray(batch["timestamps"], dtype=np.int64)
    site_values = np.asarray(batch["site_metadata"], dtype=np.float32)
    historical_future = _historical_future_batch(dataset, start, stop, plan.source_features)
    count = stop - start
    unique_ids = tuple(f"{split}::window_{index:012d}" for index in range(start, stop))
    history_times = (
        target_times[:, :1] + FREQUENCY_NS * np.arange(-LOOKBACK, 0, dtype=np.int64)[None, :]
    )
    history_data: dict[str, Any] = {
        "unique_id": np.repeat(np.asarray(unique_ids, dtype=object), LOOKBACK),
        "ds": _datetime_index(history_times.reshape(-1)),
        "y": values[..., 0].reshape(-1),
        "available_mask": masks[..., 0].reshape(-1),
    }
    history_indices = [
        REGISTERED_HISTORY_FEATURES.index(name) for name in _HISTORY_VALUE_SOURCE_FEATURES
    ]
    for column, source_index in zip(HISTORICAL_VALUE_EXOGENOUS, history_indices, strict=True):
        history_data[column] = values[..., source_index].reshape(-1)
    for source_index, column in enumerate(HISTORICAL_MASK_EXOGENOUS):
        history_data[column] = masks[..., source_index].reshape(-1)
    for source_index, column in enumerate(plan.columns):
        history_data[column] = historical_future[..., source_index].reshape(-1)
    history = pd.DataFrame(history_data)

    future_indices = [REGISTERED_FUTURE_FEATURES.index(name) for name in plan.source_features]
    future_data: dict[str, Any] = {
        "unique_id": np.repeat(np.asarray(unique_ids, dtype=object), HORIZON),
        "ds": _datetime_index(target_times.reshape(-1)),
    }
    for column, source_index in zip(plan.columns, future_indices, strict=True):
        future_data[column] = future_known[..., source_index].reshape(-1)
    future = pd.DataFrame(future_data)
    if "y" in future.columns:
        raise RuntimeError("future inference dataframe unexpectedly contains the target")

    static_data: dict[str, Any] = {"unique_id": list(unique_ids)}
    for source_index, column in enumerate(STATIC_EXOGENOUS):
        static_data[column] = site_values[:, source_index]
    static = pd.DataFrame(static_data)
    for name, frame in (("history", history), ("future", future), ("static", static)):
        numeric = frame.drop(columns=["unique_id", "ds"], errors="ignore").to_numpy(
            dtype=np.float32
        )
        if not np.isfinite(numeric).all():
            raise ValueError(f"{split}: non-finite value in {name} inference dataframe")
    if len(history) != count * LOOKBACK or len(future) != count * HORIZON:
        raise RuntimeError("bounded prediction dataframe row accounting failed")
    return PredictionFrames(
        history=history,
        future=future,
        static=static,
        unique_ids=unique_ids,
        expected_timestamps=target_times.copy(),
    )


def official_model_metadata(model: Any) -> dict[str, Any]:
    """Return JSON-safe provenance and capability metadata for a fitted model."""

    module = type(model).__module__
    if not module.startswith("neuralforecast."):
        raise RuntimeError(
            f"model class {module}.{type(model).__name__} is not official NeuralForecast"
        )
    loss = getattr(model, "loss", None)
    levels = getattr(loss, "quantiles", None)
    if levels is None:
        raise RuntimeError("official model is not configured with a multi-quantile loss")
    if hasattr(levels, "detach"):
        levels = levels.detach().cpu().numpy()
    return {
        "package": "neuralforecast",
        "package_version": neuralforecast_version(),
        "class": f"{module}.{type(model).__name__}",
        "alias": repr(model),
        "loss_class": f"{type(loss).__module__}.{type(loss).__name__}",
        "loss_api": "MQLoss(quantiles=...)",
        "quantiles": [float(value) for value in np.asarray(levels).reshape(-1)],
        "capabilities": {
            "future_exogenous": bool(getattr(model, "EXOGENOUS_FUTR", False)),
            "historical_exogenous": bool(getattr(model, "EXOGENOUS_HIST", False)),
            "static_exogenous": bool(getattr(model, "EXOGENOUS_STAT", False)),
        },
        "historical_exogenous": list(getattr(model, "hist_exog_list", ())),
        "future_exogenous": list(getattr(model, "futr_exog_list", ())),
        "static_exogenous": list(getattr(model, "stat_exog_list", ())),
        "max_steps": int(model.max_steps),
        "val_check_steps": int(model.val_check_steps),
        "early_stop_patience_steps": int(model.early_stop_patience_steps),
    }


def extract_official_quantiles(
    forecasts: Any,
    model: Any,
    frames: PredictionFrames,
    requested_quantiles: Sequence[float],
) -> np.ndarray:
    """Map official MQLoss output columns back to ``[N,H,Q]`` strictly."""

    if hasattr(forecasts, "to_pandas") and not isinstance(forecasts, pd.DataFrame):
        forecasts = forecasts.to_pandas()
    if not isinstance(forecasts, pd.DataFrame):
        raise TypeError("NeuralForecast.predict must return a pandas or polars dataframe")
    loss = getattr(model, "loss", None)
    model_levels = getattr(loss, "quantiles", None)
    output_names = getattr(loss, "output_names", None)
    if model_levels is None or output_names is None:
        raise RuntimeError("official model loss does not expose quantile output metadata")
    if hasattr(model_levels, "detach"):
        model_levels = model_levels.detach().cpu().numpy()
    model_levels = np.asarray(model_levels, dtype=float).reshape(-1)
    requested = np.asarray(tuple(requested_quantiles), dtype=float)
    if model_levels.size != requested.size or not np.allclose(
        np.sort(model_levels), np.sort(requested), atol=5e-7, rtol=0.0
    ):
        raise ValueError(
            f"official model quantiles {model_levels.tolist()} do not match requested "
            f"{requested.tolist()}"
        )
    base_columns = [f"{repr(model)}{suffix}" for suffix in output_names]
    missing = [column for column in base_columns if column not in forecasts.columns]
    if missing:
        raise ValueError(f"official prediction dataframe is missing MQLoss columns: {missing}")
    level_to_column = {
        float(level): column
        for level, column in zip(model_levels.tolist(), base_columns, strict=True)
    }
    ordered_columns = [
        level_to_column[min(level_to_column, key=lambda value: abs(value - float(level)))]
        for level in requested
    ]
    output = np.empty(
        (len(frames.unique_ids), frames.expected_timestamps.shape[1], requested.size),
        dtype=np.float32,
    )
    forecast_ids = forecasts["unique_id"].astype(str)
    for item, unique_id in enumerate(frames.unique_ids):
        group = forecasts.loc[forecast_ids == unique_id].sort_values("ds")
        if len(group) != frames.expected_timestamps.shape[1]:
            raise ValueError(
                f"official forecast for {unique_id} has {len(group)} rows, expected 24"
            )
        predicted_times = pd.to_datetime(group["ds"], utc=True).astype("int64").to_numpy()
        if not np.array_equal(predicted_times, frames.expected_timestamps[item]):
            raise ValueError(f"official forecast timestamps do not match {unique_id}'s horizon")
        output[item] = group[ordered_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(output).all():
        raise ValueError("official NeuralForecast emitted non-finite quantiles")
    return output
