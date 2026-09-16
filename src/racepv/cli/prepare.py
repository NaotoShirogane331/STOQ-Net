"""Prepare leakage-safe window archives from a local dataset manifest."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from racepv import io as window_io
from racepv.audit import audit_future_feature_names, sha256_file
from racepv.data import (
    FUTURE_KNOWN_ALLOWLIST,
    SPATIAL_ROLES,
    ManifestEntry,
    RampLabeler,
    TrainOnlyNormalizer,
    WindowedPVDataset,
    adapter_for,
    add_causal_history_features,
    chronological_split_plan,
    load_manifest,
    select_dkasc_manifest_entries,
    select_nrel_sites,
)
from racepv.data.causal_history import _POWER_DEPENDENCIES, _WEATHER_FEATURES, CausalHistoryState
from racepv.data.origin_clock import origin_times
from racepv.data.prepared_contract import (
    ResolverModelArray,
    _bounded_array_identity,
    deterministic_sampling_points,
    forecast_population_sha256,
    read_strict_json,
    validate_preparation_contract,
)
from racepv.data.splits import frozen_split_view
from racepv.io import save_window_sequence_store, save_window_shard_index
from racepv.physics import ClearSkyFeatureBuilder, SiteMetadata
from racepv.utils import atomic_json_dump

WEATHER_COLUMNS = ("ghi", "temperature", "humidity", "wind_speed")
BASE_HISTORY_COLUMNS = (
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
)
FUTURE_COLUMNS = (
    "clear_sky_power_norm",
    "solar_elevation",
    "solar_azimuth_sin",
    "solar_azimuth_cos",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "lead_time_hours",
)
SITE_METADATA_NAMES = (
    "latitude_scaled",
    "longitude_scaled",
    "log_capacity_kw_scaled",
    "surface_tilt_scaled",
    "surface_azimuth_scaled",
    "has_geometry",
)

NREL_SELECTION_SEED = 20260804
NREL_STATES = ("AL", "AZ", "CA", "NY", "OK")
NREL_STATE_ALIASES = {
    "AL": "AL",
    "ALABAMA": "AL",
    "AZ": "AZ",
    "ARIZONA": "AZ",
    "CA": "CA",
    "CALIFORNIA": "CA",
    "NY": "NY",
    "NEWYORK": "NY",
    "OK": "OK",
    "OKLAHOMA": "OK",
}
NREL_SPATIAL_ROLE = {
    "AL": "source_train",
    "CA": "source_train",
    "NY": "source_train",
    "OK": "validation_calibration",
    "AZ": "zero_parameter_test",
}


def _normalize_nrel_state(value: str | None) -> str:
    if value is None:
        raise ValueError("Every NREL manifest entry must declare its state")
    token = re.sub(r"[^A-Za-z]", "", str(value)).upper()
    try:
        return NREL_STATE_ALIASES[token]
    except KeyError as error:
        raise ValueError(
            f"Unsupported NREL state {value!r}; registered states are "
            "Alabama, Arizona, California, New York, and Oklahoma"
        ) from error


def _with_registered_nrel_role(entry: ManifestEntry) -> ManifestEntry:
    state = _normalize_nrel_state(entry.state)
    expected_role = NREL_SPATIAL_ROLE[state]
    if entry.spatial_role not in {"standard", expected_role}:
        raise ValueError(
            f"NREL site {entry.site_id!r} in {state} declares spatial_role="
            f"{entry.spatial_role!r}; the registered role is {expected_role!r}"
        )
    return replace(entry, state=state, spatial_role=expected_role)


def _select_registered_nrel_entries(
    entries: Sequence[ManifestEntry],
    *,
    enforce: bool,
) -> tuple[tuple[ManifestEntry, ...], dict[str, Any]]:
    """Select the registered 5x20 NREL cohort before any state/site filtering."""

    normalized = tuple(_with_registered_nrel_role(entry) for entry in entries)
    site_ids = [entry.site_id for entry in normalized]
    if len(site_ids) != len(set(site_ids)):
        raise ValueError("NREL candidate site_id values must be unique")
    candidate_counts = dict(sorted(Counter(entry.state for entry in normalized).items()))
    if not enforce:
        return normalized, {
            "applied": False,
            "development_bypass": True,
            "seed": NREL_SELECTION_SEED,
            "required_states": list(NREL_STATES),
            "sites_per_state": 20,
            "candidate_count": len(normalized),
            "candidate_state_counts": candidate_counts,
            "selected_count": len(normalized),
            "selected_state_counts": candidate_counts,
            "selected_site_ids": site_ids,
            "registered_evidence_eligible": False,
        }

    metadata = pd.DataFrame(
        {
            "site_id": site_ids,
            "state": [entry.state for entry in normalized],
            "capacity_kw": [entry.capacity_kw for entry in normalized],
            "fixed_tilt": [entry.fixed_tilt for entry in normalized],
        }
    )
    selected_frame = select_nrel_sites(
        metadata,
        states=NREL_STATES,
        sites_per_state=20,
        seed=NREL_SELECTION_SEED,
    )
    by_id = {entry.site_id: entry for entry in normalized}
    selected = tuple(by_id[str(site_id)] for site_id in selected_frame["site_id"])
    selected_counts = dict(sorted(Counter(entry.state for entry in selected).items()))
    expected_counts = {state: 20 for state in NREL_STATES}
    if len(selected) != 100 or selected_counts != expected_counts:
        raise RuntimeError(
            "registered NREL selection did not produce exactly 20 sites in each state"
        )
    selected_by_state = {
        state: sorted(entry.site_id for entry in selected if entry.state == state)
        for state in NREL_STATES
    }
    cohort_payload = json.dumps(selected_by_state, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return selected, {
        "applied": True,
        "development_bypass": False,
        "seed": NREL_SELECTION_SEED,
        "required_states": list(NREL_STATES),
        "sites_per_state": 20,
        "candidate_count": len(normalized),
        "candidate_state_counts": candidate_counts,
        "selected_count": len(selected),
        "selected_state_counts": selected_counts,
        "selected_site_ids": [entry.site_id for entry in selected],
        "selected_site_ids_by_state": selected_by_state,
        "selected_cohort_sha256": hashlib.sha256(cohort_payload).hexdigest(),
        "registered_evidence_eligible": True,
    }


def _validate_zero_parameter_transfer_entry(entry: ManifestEntry) -> None:
    """Forbid target-power-fitted physical preprocessing in strict holdouts."""

    if entry.spatial_role != "zero_parameter_test":
        return
    violations = []
    if entry.capacity_kw is None:
        violations.append("rated capacity is missing (would fit q99.5 from target power)")
    if entry.surface_tilt is None or entry.surface_azimuth is None:
        violations.append(
            "tilt/azimuth geometry is incomplete (would fit clear-sky mapping from target power)"
        )
    if violations:
        raise ValueError(
            f"zero-parameter test site {entry.site_id!r} violates the transfer boundary: "
            + "; ".join(violations)
        )


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "site"


def _raw_source_provenance(
    entries: Sequence[ManifestEntry], manifest_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Fingerprint every selected provider file without path-dependent hashes."""

    records: list[dict[str, Any]] = []
    identity_records: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda item: item.site_id):
        adapter = adapter_for(entry, manifest_root)
        resolved = adapter.resolve_files(entry)
        if len(resolved) != len(entry.files):
            raise RuntimeError("adapter file resolution changed the manifest file count")
        for ordinal, (manifest_reference, path) in enumerate(
            zip(entry.files, resolved, strict=True)
        ):
            size_bytes = int(path.stat().st_size)
            content_sha256 = sha256_file(path)
            record = {
                "dataset": entry.dataset,
                "site_id": entry.site_id,
                "file_ordinal": ordinal,
                "manifest_reference": manifest_reference,
                "size_bytes": size_bytes,
                "sha256": content_sha256,
            }
            records.append(record)
            # Absolute or manifest-relative path spellings are deliberately not
            # part of the identity: the same frozen bytes should retain their
            # identity after transfer from Windows to WSL or another machine.
            identity_records.append(
                {
                    "dataset": entry.dataset,
                    "site_id": entry.site_id,
                    "file_ordinal": ordinal,
                    "size_bytes": size_bytes,
                    "sha256": content_sha256,
                }
            )
    if not records:
        raise ValueError("selected manifest entries contain no raw source files")
    payload = json.dumps(identity_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return records, identity_records, hashlib.sha256(payload).hexdigest()


def _site_vector(entry: ManifestEntry, capacity_kw: float) -> dict[str, float]:
    has_geometry = entry.surface_tilt is not None and entry.surface_azimuth is not None
    return {
        "latitude_scaled": float(entry.latitude) / 90.0,
        "longitude_scaled": float(entry.longitude) / 180.0,
        "log_capacity_kw_scaled": float(np.log1p(capacity_kw) / 10.0),
        "surface_tilt_scaled": float(entry.surface_tilt or 0.0) / 90.0,
        "surface_azimuth_scaled": float(entry.surface_azimuth or 0.0) / 360.0,
        "has_geometry": float(has_geometry),
    }


def _prepare_site(
    entry: ManifestEntry,
    manifest_root: Path,
    *,
    stride: int,
    ramp_method: str,
    allow_approximate_solar: bool,
    require_observed_targets: bool,
    require_origin_safe: bool = False,
    frozen_split_contract=None,
    output_dir: Path | None = None,
    frozen_fit=None,
    site_index: int = 0,
) -> tuple[dict[str, WindowedPVDataset], dict[str, Any]]:
    if require_origin_safe:
        return _prepare_site_v2(
            entry,
            manifest_root,
            stride=stride,
            ramp_method=ramp_method,
            allow_approximate_solar=allow_approximate_solar,
            require_observed_targets=require_observed_targets,
            frozen_split_contract=frozen_split_contract,
            output_dir=output_dir,
            frozen_fit=frozen_fit,
            site_index=site_index,
        )
    _validate_zero_parameter_transfer_entry(entry)
    resampled = adapter_for(entry, manifest_root).load_resampled(entry)
    frame = resampled.values.copy()
    split_plan = chronological_split_plan(len(frame))
    train_frame = frame.iloc[split_plan["train"]]

    available_weather = tuple(
        column
        for column in WEATHER_COLUMNS
        if column in train_frame and train_frame[column].notna().any()
    )
    normalizer = TrainOnlyNormalizer().fit(
        train_frame,
        power_column="power",
        feature_columns=available_weather,
        rated_capacity=entry.capacity_kw,
        partition="train",
    )
    capacity_kw = float(normalizer.capacity)
    site = SiteMetadata(
        latitude=entry.latitude,
        longitude=entry.longitude,
        timezone=entry.timezone,
        capacity_kw=capacity_kw,
        surface_tilt=entry.surface_tilt,
        surface_azimuth=entry.surface_azimuth,
        name=entry.site_id,
    )
    physics = ClearSkyFeatureBuilder(
        site,
        allow_approximate_fallback=allow_approximate_solar,
    )
    fitted_clear_sky_mapping = entry.surface_tilt is None or entry.surface_azimuth is None
    if fitted_clear_sky_mapping:
        physics.fit_empirical_mapping(
            train_frame.index,
            train_frame["power"],
            partition="train",
        )
    physical_features = physics.build(frame.index)
    frame = frame.join(physical_features)
    frame = normalizer.transform(frame)
    frame["power_observed"] = resampled.observed_mask["power"].astype(float)
    frame["power_imputed"] = resampled.imputed_mask["power"].astype(float)
    frame.loc[frame["is_night"].astype(bool), "power_norm"] = 0.0
    frame = add_causal_history_features(frame, partitions=split_plan.slices)

    weather_features = []
    for column in WEATHER_COLUMNS:
        normalized_name = f"{column}_z"
        if column in available_weather:
            weather_features.append(normalized_name)
        else:
            frame[normalized_name] = 0.0
            weather_features.append(normalized_name)
    history_columns = (*BASE_HISTORY_COLUMNS, *weather_features)

    daylight = frame["is_daylight"].astype(bool)
    observed_power = resampled.observed_mask["power"].astype(bool)
    labeler = RampLabeler(method=ramp_method).fit(
        frame.iloc[split_plan["train"]]["power_norm"],
        daylight_mask=daylight.iloc[split_plan["train"]],
        observed_mask=observed_power.iloc[split_plan["train"]],
        partition="train",
    )
    frame["state_label"] = -100
    for boundary in split_plan.slices.values():
        index = frame.index[boundary]
        frame.loc[index, "state_label"] = labeler.transform(
            frame.loc[index, "power_norm"],
            daylight_mask=daylight.loc[index],
            observed_mask=observed_power.loc[index],
        ).to_numpy()
    frame["state_label"] = frame["state_label"].astype("int64")

    history_mask = frame.loc[:, history_columns].notna()
    history_mask["power_norm"] = observed_power
    history_mask["clear_sky_residual"] = observed_power
    history_mask["power_diff_1"] = observed_power & observed_power.shift(1, fill_value=False)
    history_mask["power_diff_2"] = (
        observed_power
        & observed_power.shift(1, fill_value=False)
        & observed_power.shift(2, fill_value=False)
    )
    history_mask["rolling_volatility_1h"] = observed_power.rolling(5, min_periods=5).sum().eq(5)
    history_mask["rolling_volatility_3h"] = observed_power.rolling(13, min_periods=13).sum().eq(13)
    history_mask["clear_sky_index"] = observed_power
    # Provenance flags and deterministic clear-sky power are always available.
    history_mask["clear_sky_power_norm"] = True
    history_mask["power_observed"] = True
    history_mask["power_imputed"] = True
    # Missing or interpolated weather remains explicit in the mask.
    for column in WEATHER_COLUMNS:
        normalized_name = f"{column}_z"
        history_mask[normalized_name] = (
            resampled.observed_mask[column].astype(bool) if column in available_weather else False
        )
    site_metadata = _site_vector(entry, capacity_kw)
    split_arrays = {}
    for split_name, boundary in split_plan.slices.items():
        dataset = WindowedPVDataset(
            frame,
            history_columns=history_columns,
            future_known_columns=FUTURE_COLUMNS,
            site_metadata=site_metadata,
            history_mask=history_mask,
            target_observed_mask=resampled.observed_mask["power"].to_numpy(),
            stride=stride,
            split=boundary,
            require_observed_targets=require_observed_targets,
        )
        split_arrays[split_name] = dataset

    split_timestamps = {}
    observation_counts = {}
    for split_name, boundary in split_plan.slices.items():
        split_index = frame.index[boundary]
        observed_split = resampled.observed_mask["power"].iloc[boundary]
        imputed_split = resampled.imputed_mask["power"].iloc[boundary]
        split_timestamps[split_name] = {
            "start": split_index.min().isoformat(),
            "end": split_index.max().isoformat(),
            "rows": int(len(split_index)),
        }
        observation_counts[split_name] = {
            "observed_power_rows": int(observed_split.sum()),
            "imputed_power_rows": int(imputed_split.sum()),
            "missing_power_rows": int((~(observed_split | imputed_split)).sum()),
        }

    target_history_fitting = {
        "model_parameter_update": False if entry.spatial_role == "zero_parameter_test" else None,
        "capacity_normalization": {
            "fitted_on_target_history": normalizer.capacity_source != "rated_capacity",
            "uses_target_power": normalizer.capacity_source != "rated_capacity",
            "source": normalizer.capacity_source,
            "partition": "train" if normalizer.capacity_source != "rated_capacity" else None,
        },
        "weather_standardization": {
            "fitted_on_target_history": bool(available_weather),
            "uses_target_power": False,
            "features": list(available_weather),
            "partition": "train" if available_weather else None,
        },
        "clear_sky_power_mapping": {
            "fitted_on_target_history": fitted_clear_sky_mapping,
            "uses_target_power": fitted_clear_sky_mapping,
            "partition": "train" if fitted_clear_sky_mapping else None,
        },
        "ramp_label_threshold": {
            "fitted_on_target_history": True,
            "uses_target_power": True,
            "partition": "train",
            "purpose": "target labeling and evaluation; not a model input parameter",
        },
        "conformal_calibration": {
            "performed_during_preparation": False,
            "allowed_source_for_zero_parameter_test": "external validation/calibration sites only",
        },
    }

    metadata = {
        "dataset": entry.dataset,
        "site_id": entry.site_id,
        "simulated": entry.dataset in {"nrel", "synthetic"},
        "timezone": entry.timezone,
        "latitude": float(entry.latitude),
        "longitude": float(entry.longitude),
        "state": entry.state,
        "fixed_tilt": entry.fixed_tilt,
        "spatial_role": entry.spatial_role,
        "geometry": {
            "surface_tilt": entry.surface_tilt,
            "surface_azimuth": entry.surface_azimuth,
            "complete": not fitted_clear_sky_mapping,
        },
        "zero_parameter_transfer_boundary": {
            "active": entry.spatial_role == "zero_parameter_test",
            "target_power_fitted_clear_sky_mapping_forbidden": (
                entry.spatial_role == "zero_parameter_test"
            ),
            "passed": not (
                entry.spatial_role == "zero_parameter_test"
                and (fitted_clear_sky_mapping or normalizer.capacity_source != "rated_capacity")
            ),
        },
        "target_history_fitting": target_history_fitting,
        "target_policy": {
            "require_observed_daylight_targets": require_observed_targets,
            "maximum_interpolated_gap_minutes": 30,
        },
        "input_sanitization": {
            "out_of_range_values_treated_as_missing": bool(
                resampled.values.attrs.get("column_valid_ranges")
            ),
            "column_valid_ranges": resampled.values.attrs.get("column_valid_ranges", {}),
            "out_of_range_counts": resampled.values.attrs.get("out_of_range_counts", {}),
        },
        "analysis_window": {
            "start": entry.analysis_start,
            "end": entry.analysis_end,
            "filtered_raw_rows": int(resampled.values.attrs.get("analysis_filtered_row_count", 0)),
            "purpose": (
                "common-calendar alignment before relative temporal splitting"
                if entry.analysis_start is not None or entry.analysis_end is not None
                else None
            ),
        },
        "temporal_splits": split_timestamps,
        "power_observation_counts": observation_counts,
        "history_columns": list(history_columns),
        "future_known_columns": list(FUTURE_COLUMNS),
        "site_metadata_names": list(SITE_METADATA_NAMES),
        "capacity": normalizer.state_dict(),
        "ramp": labeler.state_dict(),
        "clear_sky": {
            "solar_engine": physical_features.attrs.get("solar_engine"),
            "clear_sky_model": physical_features.attrs.get("clear_sky_model"),
            "power_model": physical_features.attrs.get("power_model"),
            "fitted_empirical_mapping": fitted_clear_sky_mapping,
        },
    }
    return split_arrays, metadata


def _json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _bytes_identity(path, data):
    return {"path": path, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _array_identity(array):
    return _bounded_array_identity(array)


def _finite_list(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def _fit_evidence(
    entry, sampled, plan, normalizer, physics, physical, labeler, *, site_index, evidence, raw_sha
):
    """Produce actual train-input receipts from the same data used by existing fit policies."""
    _, slices = frozen_split_view(plan)
    boundary = slices["train"]
    return _fit_evidence_from_train(
        entry,
        sampled.values.iloc[boundary],
        sampled.observed_mask.iloc[boundary],
        sampled.imputed_mask.iloc[boundary],
        plan,
        normalizer,
        physics,
        physical.iloc[boundary],
        labeler,
        site_index=site_index,
        evidence=evidence,
        raw_sha=raw_sha,
    )


def _fit_evidence_from_train(
    entry,
    train,
    observed,
    filled,
    plan,
    normalizer,
    physics,
    physical,
    labeler,
    *,
    site_index,
    evidence,
    raw_sha,
):
    """Same fit accounting on an already validated complete training slice."""
    normalized = normalizer.normalize_power(train["power"])
    normalized = normalized.mask(physical["is_night"], 0.0)
    daylight = physical["is_daylight"]
    ramp_used = (
        normalized.diff(2).notna()
        & np.isfinite(normalized.diff(2))
        & daylight
        & daylight.shift(2, fill_value=False)
        & observed["power"]
        & observed["power"].shift(2, fill_value=False)
    )
    clear_used = (
        np.isfinite(train["power"]) & (train["power"] >= 0) & (physical["clear_sky_ghi"] > 50)
    )
    inputs = {
        "labels_ns": train.index.as_unit("ns").asi8.tolist(),
        "power": _finite_list(train["power"]),
        "power_normalized": _finite_list(normalized),
        "clear_sky_ghi": _finite_list(physical["clear_sky_ghi"]),
        "observed": observed["power"].tolist(),
        "imputed": filled["power"].tolist(),
        "daylight": daylight.tolist(),
        "weather": {name: _finite_list(train[name]) for name in normalizer.feature_columns},
        "weather_observed": {name: observed[name].tolist() for name in normalizer.feature_columns},
        "weather_imputed": {name: filled[name].tolist() for name in normalizer.feature_columns},
    }

    def receipt(role, selection, used):
        counts = {}
        for name, selected in used.items():
            obs, imp = observed[name].to_numpy(), filled[name].to_numpy()
            selected = np.asarray(selected, dtype=bool)
            counts[name] = {
                "observed_rows": int(obs.sum()),
                "filled_rows": int(imp.sum()),
                "missing_rows": int((~(obs | imp)).sum()),
                "used_observed_rows": int((selected & obs).sum()),
                "used_filled_rows": int((selected & imp).sum()),
                "used_rows": int(selected.sum()),
            }
        path = f"fit-input/{role}.json"
        evidence[path] = _json_bytes(
            {
                **inputs,
                "selection": selection,
                "used": {
                    name: np.asarray(mask, dtype=bool).tolist() for name, mask in used.items()
                },
            }
        )
        return {
            "partition": "train",
            **plan["splits"]["train"],
            "input_file": _bytes_identity(path, evidence[path]),
            "raw_input_sha256": raw_sha,
            "selection": selection,
            "accounting": counts,
        }

    roles = {}

    def add(
        role, method, purpose, state, *, fit=None, audit=None, parameter_source="fixed_protocol"
    ):
        record = {
            "method": method,
            "purpose": purpose,
            "state": state,
            "fit": fit,
            "training_audit": audit,
            "provenance": {
                "site_id": site_index,
                "spatial_role": entry.spatial_role,
                "parameter_source": parameter_source,
                "training_role": (
                    "source_site" if entry.spatial_role == "source_train" else "target_site"
                )
                if fit is not None or audit is not None
                else None,
            },
        }
        path = f"fit/{role}.json"
        evidence[path] = _json_bytes({"role": role, **record})
        roles[role] = {**record, "file": _bytes_identity(path, evidence[path])}

    capacity_fit = (
        None
        if normalizer.capacity_source == "rated_capacity"
        else receipt(
            "capacity",
            "finite_train_power_including_closed_fills",
            {"power": np.isfinite(train["power"])},
        )
    )
    add(
        "capacity_normalization",
        normalizer.capacity_source,
        "power_capacity_normalization",
        {
            "capacity_kw": float(normalizer.capacity),
            "capacity_source": normalizer.capacity_source,
            "capacity_quantile": normalizer.capacity_quantile,
            "clip_power": list(normalizer.clip_power),
            "eps": normalizer.eps,
        },
        fit=capacity_fit,
        parameter_source="site_metadata" if capacity_fit is None else "site_training",
    )
    weather_fit = (
        receipt(
            "weather",
            "finite_train_weather_including_closed_fills",
            {name: np.isfinite(train[name]) for name in normalizer.feature_columns},
        )
        if normalizer.feature_columns
        else None
    )
    add(
        "weather_standardization",
        "train_mean_population_std" if weather_fit else "absent",
        "historical_weather_normalization",
        {
            "feature_columns": list(normalizer.feature_columns),
            "feature_mean": normalizer.feature_mean.copy(),
            "feature_scale": normalizer.feature_scale.copy(),
            "ddof": 0,
            "eps": normalizer.eps,
            "nonfinite_mean_fallback": 0.0,
            "nonfinite_or_small_scale_fallback": 1.0,
        },
        fit=weather_fit,
        parameter_source="site_training" if weather_fit else "fixed_protocol",
    )
    clear_state = {
        "capacity_kw": float(normalizer.capacity),
        "solar_engine": physical.attrs["solar_engine"],
        "clear_sky_model": physical.attrs["clear_sky_model"],
        "power_model": physical.attrs["power_model"],
        "clip_power": [0.0, 1.05],
    }
    empirical = entry.surface_tilt is None or entry.surface_azimuth is None
    if empirical:
        method = "train_empirical_q95"
        clear_state.update(
            mapping_quantile=physics.mapper.mapping_quantile,
            scale_kw_per_wm2=physics.mapper.scale_kw_per_wm2,
            n_training_points=physics.mapper.n_training_points,
        )
        clear_fit = receipt(
            "clear-sky",
            "finite_nonnegative_train_power_ghi_gt_50_daylight_including_closed_fills",
            {"power": clear_used},
        )
    elif physical.attrs["power_model"] == "pvwatts":
        method, clear_fit = "geometry_pvwatts", None
        clear_state.update(
            gamma_pdc=physics.site.gamma_pdc,
            losses_fraction=physics.site.losses_fraction,
            temp_cell_celsius=25.0,
            poa_model="isotropic",
        )
    else:
        method, clear_fit = "geometry_capacity_ghi_approximation", None
        clear_state["ghi_divisor"] = 1000.0
    add(
        "clear_sky_power_mapping",
        method,
        "deterministic_clear_sky_power",
        clear_state,
        fit=clear_fit,
        parameter_source="site_training" if empirical else "site_metadata",
    )
    ramp_receipt = receipt("ramp", "finite_observed_daylight_30min_endpoints", {"power": ramp_used})
    fixed = labeler.method == "fixed0.1"
    add(
        "ramp_label_threshold",
        labeler.method,
        "target_labeling_and_evaluation_not_model_input",
        {
            "method": labeler.method,
            "threshold": labeler.threshold_,
            "fixed_threshold": labeler.fixed_threshold,
            "n_training_ramps": labeler.n_training_ramps_,
            "quantile": None if fixed else {"q85": 0.85, "q90": 0.9, "q95": 0.95}[labeler.method],
        },
        fit=None if fixed else ramp_receipt,
        audit=ramp_receipt if fixed else None,
        parameter_source="fixed_protocol" if fixed else "site_training",
    )
    add(
        "site_vector",
        "fixed_site_transforms",
        "site_metadata_features",
        {
            "latitude_divisor": 90.0,
            "longitude_divisor": 180.0,
            "log_capacity_divisor": 10.0,
            "tilt_divisor": 90.0,
            "azimuth_divisor": 360.0,
            "missing_geometry_value": 0.0,
            "capacity_kw": float(normalizer.capacity),
            "values": list(_site_vector(entry, float(normalizer.capacity)).values()),
        },
    )
    return roles


def _restore_frozen_fit(
    frozen, entry, plan, source, sampled, *, site_index, ramp_method, evidence, current_raw_sha
):
    """Validate the original complete preparation and restore its fitted parameters verbatim."""
    if type(frozen) is not dict or set(frozen) != {"schema", "causal_arrays", "evidence"}:
        raise ValueError("frozen fit requires a complete original prepared contract")
    validate_preparation_contract(
        frozen["schema"], frozen["causal_arrays"], evidence=frozen["evidence"]
    )
    schema = frozen["schema"]
    if (
        schema["split_plan"] != plan
        or schema["source_time_contract"] != source
        or schema["site"]
        != {"id": site_index, "name": entry.site_id, "spatial_role": entry.spatial_role}
    ):
        raise ValueError("frozen fit site/time/split contract mismatch")
    fit_source = schema["sources"].get("frozen_fit_raw_input", schema["sources"]["raw_input"])
    fit_bytes = frozen["evidence"][fit_source["path"]]
    original_manifest = json.loads(fit_bytes)["manifest_entry"]
    current_manifest = json.loads(_json_bytes(asdict(entry)))
    for field in (
        "dataset",
        "site_id",
        "timezone",
        "latitude",
        "longitude",
        "capacity_kw",
        "surface_tilt",
        "surface_azimuth",
        "spatial_role",
        "columns",
        "timestamp_column",
        "power_column",
        "column_scales",
        "column_valid_ranges",
        "power_scale_to_kw",
        "timestamp_format",
        "dayfirst",
        "analysis_start",
        "analysis_end",
        "read_options",
    ):
        if original_manifest[field] != current_manifest[field]:
            raise ValueError(f"frozen fit manifest {field} mismatch")
    roles = copy.deepcopy(schema["fit_states"])
    if roles["ramp_label_threshold"]["method"] != ramp_method:
        raise ValueError("frozen fit ramp method mismatch")
    for record in roles.values():
        paths = [record["file"]["path"]]
        paths += [
            record[key]["input_file"]["path"]
            for key in ("fit", "training_audit")
            if record[key] is not None
        ]
        for path in paths:
            evidence[path] = frozen["evidence"][path]
    frozen_source = None
    if fit_source["sha256"] != current_raw_sha:
        path = "evidence/frozen-fit-raw-input.json"
        evidence[path] = fit_bytes
        frozen_source = _bytes_identity(path, fit_bytes)
    ramp = roles["ramp_label_threshold"]
    fit_receipt = ramp["fit"] if ramp["fit"] is not None else ramp["training_audit"]
    original_inputs = json.loads(evidence[fit_receipt["input_file"]["path"]])
    _, slices = frozen_split_view(plan)
    train = sampled.values.iloc[slices["train"]]
    columns = roles["weather_standardization"]["state"]["feature_columns"]
    expected = {
        "labels_ns": train.index.as_unit("ns").asi8.tolist(),
        "power": _finite_list(train["power"]),
        "observed": sampled.observed_mask["power"].iloc[slices["train"]].tolist(),
        "imputed": sampled.imputed_mask["power"].iloc[slices["train"]].tolist(),
        "weather": {name: _finite_list(train[name]) for name in columns},
        "weather_observed": {
            name: sampled.observed_mask[name].iloc[slices["train"]].tolist() for name in columns
        },
        "weather_imputed": {
            name: sampled.imputed_mask[name].iloc[slices["train"]].tolist() for name in columns
        },
    }
    if any(original_inputs.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen fit training inputs changed")
    normalizer = TrainOnlyNormalizer()
    capacity = roles["capacity_normalization"]["state"]
    normalizer.capacity = capacity["capacity_kw"]
    normalizer.capacity_source = capacity["capacity_source"]
    normalizer.power_column = "power"
    normalizer.feature_columns = tuple(columns)
    normalizer.feature_mean = roles["weather_standardization"]["state"]["feature_mean"].copy()
    normalizer.feature_scale = roles["weather_standardization"]["state"]["feature_scale"].copy()
    normalizer.fitted = True
    labeler = RampLabeler(method=ramp_method)
    labeler.threshold_ = ramp["state"]["threshold"]
    labeler.n_training_ramps_ = ramp["state"]["n_training_ramps"]
    return roles, normalizer, labeler, frozen_source


def _build_origin_safe_frame(
    sampled,
    labels,
    slices,
    normalizer,
    physical,
    labeler,
    *,
    available,
    split_numbers,
):
    # Fit values may contain closed fills. Source state uses only actual observations.
    frame = normalizer.transform(sampled.values.where(sampled.observed_mask))
    frame = frame.join(physical)
    frame = add_causal_history_features(frame, partitions=slices)
    raw = np.zeros((len(frame), 14), dtype=np.float64)
    raw[:, 0] = frame["power_norm"]
    raw[:, 1] = physical["clear_sky_power_norm"]
    for i, name in enumerate(window_io.REGISTERED_HISTORY_FEATURES[2:8], start=2):
        raw[:, i] = frame[name]
    observed = sampled.observed_mask["power"].to_numpy(dtype=bool)
    raw[:, 8] = observed
    weather_observed = np.zeros((len(frame), 4), dtype=bool)
    for i, name in enumerate(WEATHER_COLUMNS):
        if name in available:
            raw[:, 10 + i] = frame[f"{name}_z"]
            weather_observed[:, i] = sampled.observed_mask[name]
        else:
            raw[:, 10 + i] = np.nan
    split_ids = np.empty(len(frame), dtype=np.int64)
    for name, boundary in slices.items():
        split_ids[boundary] = split_numbers[name]
    state = CausalHistoryState.from_arrays(
        labels_ns=labels,
        split_ids=split_ids,
        power_observed=observed,
        weather_observed=weather_observed,
        clear_sky_norm=raw[:, 1],
        base_values=raw,
        feature_names=window_io.REGISTERED_HISTORY_FEATURES,
    )
    for i, name in enumerate(window_io.REGISTERED_HISTORY_FEATURES):
        frame[name] = raw[:, i]
    frame["state_label"] = -100
    for boundary in slices.values():
        part = frame.iloc[boundary]
        frame.loc[part.index, "state_label"] = labeler.transform(
            part["power_norm"].mask(part["is_night"], 0),
            daylight_mask=part["is_daylight"],
            observed_mask=observed[boundary],
        )
    frame["state_label"] = frame["state_label"].astype("int64")
    history_mask = np.isfinite(raw)
    history_mask[:, 0] = observed
    history_mask[:, 10:] = weather_observed
    return frame, raw, observed, weather_observed, state, history_mask, split_ids


def _prepare_site_v2(
    entry,
    manifest_root,
    *,
    stride,
    ramp_method,
    allow_approximate_solar,
    require_observed_targets,
    frozen_split_contract,
    output_dir,
    frozen_fit,
    site_index,
):
    """Explicit synthetic engineering producer; source-scientific qualification stays separate."""
    entry = copy.deepcopy(entry)
    frozen_split_contract = copy.deepcopy(frozen_split_contract)
    if output_dir is None:
        raise ValueError("v2 preparation requires an independent output_dir")
    if require_observed_targets is not True:
        raise ValueError("v2 preparation requires observed daylight targets")
    if allow_approximate_solar and entry.dataset != "synthetic":
        raise ValueError("approximate solar is restricted to explicitly synthetic inputs")
    if type(site_index) is not int or site_index < 0:
        raise TypeError("site_index must be a nonnegative integer")
    if ramp_method not in {"q85", "q90", "q95", "fixed0.1"}:
        raise ValueError("v2 requires a registered ramp method")
    _validate_zero_parameter_transfer_entry(entry)
    labels, slices = frozen_split_view(frozen_split_contract)
    if any(boundary.stop - boundary.start < 216 for boundary in slices.values()):
        raise ValueError("v2 splits require complete 192/24 candidates")
    output_dir = Path(output_dir)
    for component in (output_dir, *output_dir.parents):
        if component.is_symlink():
            raise ValueError("v2 output must not contain symlink components")
    output_dir.mkdir(parents=True, exist_ok=False)
    adapter = adapter_for(entry, manifest_root)
    source, paths = adapter.verify_source_files(entry)
    sampled = adapter.load_resampled(
        entry, frozen_split_contract=frozen_split_contract, require_origin_safe=True
    )
    raw_bundle = _json_bytes(
        {
            "source_files": [
                {**pin, "base64": base64.b64encode(path.read_bytes()).decode("ascii")}
                for pin, path in zip(entry.source_file_identities, paths, strict=True)
            ],
            "manifest_entry": asdict(entry),
        }
    )
    adapter.verify_source_files(entry)
    proof_path = adapter.data_root / source["evidence"]["reference"]
    source = copy.deepcopy(source)
    source["evidence"]["reference"] = "evidence/time.txt"
    evidence = {
        "evidence/time.txt": proof_path.read_bytes(),
        "evidence/raw-input.json": raw_bundle,
        "evidence/site.json": _json_bytes(
            {
                "latitude": float(entry.latitude),
                "longitude": float(entry.longitude),
                "timezone": entry.timezone,
                "rated_capacity_kw": entry.capacity_kw,
                "surface_tilt": entry.surface_tilt,
                "surface_azimuth": entry.surface_azimuth,
            }
        ),
        "evidence/producer.json": _json_bytes(
            {
                "modules": {
                    name: path.read_text(encoding="utf-8")
                    for name, path in {
                        "prepare": Path(__file__),
                        "adapters": Path(__file__).parents[1] / "data/adapters.py",
                        "preprocessing": Path(__file__).parents[1] / "data/preprocessing.py",
                        "splits": Path(__file__).parents[1] / "data/splits.py",
                    }.items()
                }
            }
        ),
    }
    train = sampled.values.iloc[slices["train"]]
    available = tuple(
        name for name in WEATHER_COLUMNS if name in train and train[name].notna().any()
    )
    frozen_source = None
    if frozen_fit is None:
        normalizer = TrainOnlyNormalizer().fit(
            train, feature_columns=available, rated_capacity=entry.capacity_kw
        )
    else:
        roles, normalizer, labeler, frozen_source = _restore_frozen_fit(
            frozen_fit,
            entry,
            frozen_split_contract,
            source,
            sampled,
            site_index=site_index,
            ramp_method=ramp_method,
            evidence=evidence,
            current_raw_sha=hashlib.sha256(raw_bundle).hexdigest(),
        )
        if tuple(normalizer.feature_columns) != available:
            raise ValueError("frozen fit weather feature availability changed")
    site = SiteMetadata(
        latitude=entry.latitude,
        longitude=entry.longitude,
        timezone=entry.timezone,
        capacity_kw=float(normalizer.capacity),
        surface_tilt=entry.surface_tilt,
        surface_azimuth=entry.surface_azimuth,
        name=entry.site_id,
    )
    physics = ClearSkyFeatureBuilder(site, allow_approximate_fallback=allow_approximate_solar)
    if frozen_fit is not None:
        clear = roles["clear_sky_power_mapping"]
        if clear["method"] == "train_empirical_q95":
            physics.mapper.scale_kw_per_wm2 = clear["state"]["scale_kw_per_wm2"]
            physics.mapper.capacity_kw = float(normalizer.capacity)
            physics.mapper.n_training_points = clear["state"]["n_training_points"]
    elif entry.surface_tilt is None or entry.surface_azimuth is None:
        physics.fit_empirical_mapping(train.index, train["power"], partition="train")
    physical = physics.build(sampled.values.index)
    if frozen_fit is None:
        fit_power = normalizer.normalize_power(train["power"]).mask(
            physical["is_night"].iloc[slices["train"]], 0.0
        )
        labeler = RampLabeler(method=ramp_method).fit(
            fit_power,
            daylight_mask=physical["is_daylight"].iloc[slices["train"]],
            observed_mask=sampled.observed_mask["power"].iloc[slices["train"]],
        )
        roles = _fit_evidence(
            entry,
            sampled,
            frozen_split_contract,
            normalizer,
            physics,
            physical,
            labeler,
            site_index=site_index,
            evidence=evidence,
            raw_sha=hashlib.sha256(raw_bundle).hexdigest(),
        )
    elif any(
        physical.attrs[key] != roles["clear_sky_power_mapping"]["state"][key]
        for key in ("solar_engine", "clear_sky_model", "power_model")
    ):
        raise ValueError("frozen fit solar engine changed")
    frame, raw, observed, weather_observed, state, history_mask, split_ids = (
        _build_origin_safe_frame(
            sampled,
            labels,
            slices,
            normalizer,
            physical,
            labeler,
            available=available,
            split_numbers={name: i for i, name in enumerate(slices)},
        )
    )
    datasets, populations, storage_paths = {}, {}, {}
    for split, boundary in slices.items():
        arguments = dict(
            history_columns=window_io.REGISTERED_HISTORY_FEATURES,
            future_known_columns=FUTURE_COLUMNS,
            site_metadata=_site_vector(entry, float(normalizer.capacity)),
            history_mask=pd.DataFrame(
                history_mask, index=frame.index, columns=window_io.REGISTERED_HISTORY_FEATURES
            ),
            target_observed_mask=observed,
            stride=stride,
            split=boundary,
            require_observed_targets=True,
            causal_state=state,
        )
        dataset = WindowedPVDataset(frame, **arguments)
        times = origin_times(labels, dataset.window_starts, lookback=192, horizon=24)
        rows = dataset.window_starts[:, None] + 192 + np.arange(24)
        arrays = {
            "labels_ns": labels,
            "split_ids": split_ids,
            "power_observed": observed,
            "weather_observed": weather_observed,
            "clear_sky_norm": raw[:, 1].copy(),
            "base_values_raw": raw,
            "effective_power_observed": state._effective_power_observed.copy(),
            "gap_left": state.gap_left.copy(),
            "gap_right": state.gap_right.copy(),
            "row_targets": raw[:, 0].copy(),
            "row_state_labels": frame["state_label"].to_numpy(dtype=np.int64),
            "row_future_known": frame.loc[:, FUTURE_COLUMNS[:-1]].to_numpy(dtype=np.float64),
            "site_vector": np.asarray(
                list(_site_vector(entry, float(normalizer.capacity)).values()), dtype=np.float32
            ),
            "candidate_starts": dataset.candidate_starts.copy(),
            "history_candidate_starts": dataset.history_candidate_starts.copy(),
            "window_starts": dataset.window_starts.copy(),
            "issue_ns": times.issue_ns,
            "timestamps": times.target_start_ns,
            "day_ids": times.target_start_ns // 86_400_000_000_000,
            "site_ids": np.full(len(dataset), site_index, dtype=np.int64),
            "daylight_mask": raw[rows, 1] >= 0.01,
            "target_observed_mask": observed[rows],
        }
        # Infer only tensor geometry from a bounded sample. Retain resolver-backed
        # roles, never all overlapping windows; identities and writers stream them.
        sample = dataset.read_model_window_batch(np.arange(min(1, len(dataset)), dtype=np.int64))
        arrays.update(
            {
                name: ResolverModelArray(
                    dataset, name, {"shape": (len(dataset), *value.shape[1:]), "dtype": value.dtype}
                )
                for name, value in sample.items()
            }
        )
        del sample
        schema = {
            "schema_kind": "racepv.prepared-window-contract",
            "schema_version": 2,
            "causality_profile": "as-of-origin-v1",
            "dataset_kind": entry.dataset,
            "simulated": entry.dataset in {"synthetic", "nrel"},
            "history_columns": list(window_io.REGISTERED_HISTORY_FEATURES),
            "future_known_columns": list(FUTURE_COLUMNS),
            "site_metadata_names": list(SITE_METADATA_NAMES),
            "target": "power_norm",
            "state_labels": dict(window_io.REGISTERED_STATE_LABELS),
            "site": {"id": site_index, "name": entry.site_id, "spatial_role": entry.spatial_role},
            "geometry": {"lookback": 192, "horizon": 24, "stride": stride},
            "clock": {
                "sampling_interval_ns": 900_000_000_000,
                "stored_timezone": "UTC",
                "timestamps": "target_interval_start_ns",
                "issue": "last_history_interval_end",
                "lead": "target_interval_end_minus_issue",
                "day_ids": "utc_days_since_epoch",
                "deterministic_sampling_points": deterministic_sampling_points(),
            },
            "source_time_contract": source,
            "split_plan": copy.deepcopy(frozen_split_contract),
            "split": split,
            "causal_rules": {
                "max_gap_steps": 2,
                "gap_right": "strictly_before_first_target",
                "endpoint_scope": "same_split_real_observations",
                "unclosed_daylight": "reject",
                "night": "clear_sky_norm_lt_0.01_source_precision",
                "missing_values": "zero_value_zero_mask",
                "imputed_values": "never_exposed",
                "provenance_flag_mask": 1,
                "power_dependencies": {
                    name: list(offsets) for name, offsets in _POWER_DEPENDENCIES.items()
                },
                "weather_sources": dict(_WEATHER_FEATURES),
                "deterministic_history": ["clear_sky_power_norm"],
                "provenance_flags": ["power_observed", "power_imputed"],
            },
            "sources": {
                role: _bytes_identity(path, evidence[path])
                for role, path in {
                    "time_evidence": "evidence/time.txt",
                    "raw_input": "evidence/raw-input.json",
                    "producer": "evidence/producer.json",
                    "site_metadata": "evidence/site.json",
                }.items()
            },
            "fit_states": roles,
            "arrays": {name: _array_identity(array) for name, array in arrays.items()},
            "forecast_population_sha256": "",
        }
        if frozen_source is not None:
            schema["sources"]["frozen_fit_raw_input"] = frozen_source
        schema["forecast_population_sha256"] = forecast_population_sha256(schema)
        prepared = {"schema": schema, "causal_arrays": arrays, "evidence": evidence}
        datasets[split] = WindowedPVDataset(frame, prepared_contract=prepared, **arguments)
        history = state.history_admissible(dataset.candidate_starts, lookback=192)
        target = dataset.target_eligible(dataset.candidate_starts)
        populations[split] = {
            "candidate_starts": dataset.candidate_starts.tolist(),
            "history_candidate_starts": dataset.history_candidate_starts.tolist(),
            "window_starts": dataset.window_starts.tolist(),
            "history_reasons": [list(reason) for reason in history.reasons],
            "target_reasons": [list(reason) for reason in target.reasons],
            "raw_observed_rows": int(observed[boundary].sum()),
            "physical_night_rows": int(frame["is_night"].iloc[boundary].sum()),
            "unobserved_physical_night_rows": int(
                (~observed[boundary] & frame["is_night"].iloc[boundary].to_numpy()).sum()
            ),
        }
    adapter.verify_source_files(entry)
    sequence_path = output_dir / "sequence"
    save_window_sequence_store(
        sequence_path,
        datasets,
        site_index=site_index,
        schema=datasets["train"].prepared_contract["schema"],
    )
    for split, dataset in datasets.items():
        prepared = dataset.prepared_contract
        paths = {
            "sequence": "sequence",
            "npz": f"{split}-npz/prepared.npz",
            "eager": f"{split}-eager",
            "shard": f"{split}.index.json",
        }
        adapter.verify_source_files(entry)
        window_io.save_windows_npz(
            output_dir / paths["npz"],
            {name: prepared["causal_arrays"][name] for name in window_io.MODEL_ARRAYS},
            schema=prepared["schema"],
            causal_arrays=prepared["causal_arrays"],
            prepared_contract=prepared,
        )
        window_io.save_window_dataset_store(
            output_dir / paths["eager"], dataset, site_index=site_index, schema=prepared["schema"]
        )
        save_window_shard_index(
            output_dir / paths["shard"],
            [{"path": "sequence", "split": split, "length": len(dataset)}],
            schema=prepared["schema"],
        )
        storage_paths[split] = paths
    receipt = {
        "schema_kind": "racepv.prepare-receipt",
        "schema_version": 2,
        "registered_evidence_eligible": False,
        "source_interpretation_verified": False,
        "fit_scope": "frozen_full_train_not_per_origin",
        "frozen_split_contract": frozen_split_contract,
        "current_raw_input": datasets["train"].prepared_contract["schema"]["sources"]["raw_input"],
        "frozen_fit_raw_input": frozen_source,
        "source_file_identities": entry.source_file_identities,
        "source_time_contract": entry.source_time_contract,
        "populations": populations,
        "storage_paths": storage_paths,
        "fit_states": roles,
        "source_fit_role": "source_site" if entry.spatial_role == "source_train" else "target_site",
    }
    with (output_dir / "prepare-receipt.json").open("xb") as stream:
        stream.write(_json_bytes(receipt))
    return datasets, receipt


def prepare_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    dataset_name: str | None,
    site_ids: set[str] | None,
    states: set[str] | None,
    stride: int,
    ramp_method: str,
    allow_approximate_solar: bool,
    require_observed_targets: bool,
    enforce_dkasc_quality: bool,
    enforce_nrel_selection: bool = True,
    spatial_roles: set[str] | None = None,
    require_origin_safe: bool = False,
    frozen_split_contract=None,
) -> None:
    if require_origin_safe:
        frozen_split_view(frozen_split_contract)
        if Path(manifest_path).suffix.lower() != ".json":
            raise ValueError("v2 manifest requires strict JSON")
        read_strict_json(manifest_path)
    elif frozen_split_contract is not None:
        raise ValueError("frozen split contract requires the explicit origin-safe entry")
    manifest = load_manifest(manifest_path)
    manifest_dataset_counts = dict(
        sorted(Counter(entry.dataset for entry in manifest.entries).items())
    )
    entries = manifest.entries
    if dataset_name:
        dataset_name = dataset_name.lower()
        entries = tuple(entry for entry in entries if entry.dataset == dataset_name)
    selected_dataset_kinds = sorted({entry.dataset for entry in entries})
    if len(selected_dataset_kinds) > 1:
        raise ValueError(
            "prepare refuses mixed datasets by default; pass --dataset so each "
            "evidence source is prepared and audited separately"
        )
    if not entries:
        raise ValueError("dataset filter selected no sites")

    if require_origin_safe:
        # Explicit engineering entry: no legacy quality/cohort eligibility is promoted.
        selected = tuple(
            entry
            for entry in entries
            if (not site_ids or entry.site_id in site_ids)
            and (not states or entry.state in states)
            and (not spatial_roles or entry.spatial_role in spatial_roles)
        )
        if not selected:
            raise ValueError("manifest filters selected no sites")
        names = [_safe_name(entry.site_id) for entry in selected]
        if len(set(names)) != len(names):
            raise ValueError("site_id values collide after filesystem-safe normalization")
        for entry in selected:
            adapter_for(entry, manifest.root).verify_source_files(entry)
        output_dir = Path(output_dir)
        for component in (output_dir, *output_dir.parents):
            if component.is_symlink():
                raise ValueError("v2 output must not contain symlink components")
        output_dir.mkdir(parents=True, exist_ok=False)
        sites = []
        for site_index, entry in enumerate(selected):
            relative = _safe_name(entry.site_id)
            _, receipt = _prepare_site(
                entry,
                manifest.root,
                stride=stride,
                ramp_method=ramp_method,
                allow_approximate_solar=allow_approximate_solar,
                require_observed_targets=require_observed_targets,
                require_origin_safe=True,
                frozen_split_contract=frozen_split_contract,
                output_dir=output_dir / relative,
                site_index=site_index,
            )
            sites.append(
                {
                    "site_id": entry.site_id,
                    "path": relative,
                    "receipt": _bytes_identity(
                        f"{relative}/prepare-receipt.json", _json_bytes(receipt)
                    ),
                }
            )
        with (output_dir / "prepare-manifest-receipt.json").open("xb") as stream:
            stream.write(
                _json_bytes(
                    {
                        "schema_version": 2,
                        "registered_evidence_eligible": False,
                        "source_interpretation_verified": False,
                        "sites": sites,
                        "frozen_split_contract": frozen_split_contract,
                        "legacy_cohort_quality_qualification_performed": False,
                    }
                )
            )
        return

    nrel_selection: dict[str, Any] = {
        "applicable": False,
        "applied": False,
        "development_bypass": False,
        "registered_evidence_eligible": True,
    }
    registered_nrel_cohort: dict[str, Any] | None = None
    if entries[0].dataset == "nrel":
        entries, nrel_selection = _select_registered_nrel_entries(
            entries,
            enforce=enforce_nrel_selection,
        )
        nrel_selection = {"applicable": True, **nrel_selection}

    # Registered NREL cohort selection deliberately precedes these user-facing
    # filters. A command requesting only Arizona must still prove that it came
    # from the fixed five-state, 100-site cohort.
    if site_ids:
        entries = tuple(entry for entry in entries if entry.site_id in site_ids)
    if states:
        normalized_states = (
            {_normalize_nrel_state(state) for state in states}
            if entries and entries[0].dataset == "nrel"
            else {state.upper() for state in states}
        )
        entries = tuple(
            entry
            for entry in entries
            if entry.state is not None and entry.state.upper() in normalized_states
        )
    if spatial_roles:
        invalid_roles = sorted(set(spatial_roles).difference(SPATIAL_ROLES))
        if invalid_roles:
            raise ValueError(f"unknown spatial role filters: {invalid_roles}")
        entries = tuple(entry for entry in entries if entry.spatial_role in spatial_roles)
    if not entries:
        raise ValueError("manifest filters selected no sites")
    selected_roles = {entry.spatial_role for entry in entries}
    if "zero_parameter_test" in selected_roles and len(selected_roles) > 1:
        raise ValueError(
            "zero-parameter test sites cannot share an archive with training or "
            "calibration roles; filter by --state, --site, or --spatial-role"
        )
    for entry in entries:
        _validate_zero_parameter_transfer_entry(entry)
    if allow_approximate_solar and any(entry.dataset != "synthetic" for entry in entries):
        raise ValueError(
            "--allow-approximate-solar is restricted to explicitly synthetic smoke data"
        )
    if entries[0].dataset == "nrel":
        invalid = [entry.site_id for entry in entries if entry.fixed_tilt is not True]
        if invalid:
            raise ValueError(
                "registered NREL evidence uses fixed-tilt sites only; invalid sites: "
                + ", ".join(invalid[:10])
            )
        if enforce_nrel_selection:
            observed_counts = dict(sorted(Counter(str(entry.state) for entry in entries).items()))
            profiles = {
                "source_train_60": {"AL": 20, "CA": 20, "NY": 20},
                "validation_calibration_20": {"OK": 20},
                "zero_parameter_test_20": {"AZ": 20},
            }
            matching = [name for name, expected in profiles.items() if observed_counts == expected]
            if len(matching) != 1:
                raise ValueError(
                    "registered NREL preparation must emit exactly one frozen profile: "
                    "AL+CA+NY=60 source sites, OK=20 validation/calibration sites, or "
                    f"AZ=20 zero-parameter test sites; received {observed_counts}. "
                    "Use --skip-nrel-selection only for explicitly ineligible development data."
                )
            full_by_state = nrel_selection.get("selected_site_ids_by_state")
            full_hash = nrel_selection.get("selected_cohort_sha256")
            if not isinstance(full_by_state, dict) or not isinstance(full_hash, str):
                raise RuntimeError("registered NREL selector omitted its cohort identity")
            output_ids = sorted(entry.site_id for entry in entries)
            expected_ids = sorted(
                site_id for state in observed_counts for site_id in full_by_state[state]
            )
            if output_ids != expected_ids:
                raise RuntimeError("filtered NREL profile differs from the frozen cohort")
            registered_nrel_cohort = {
                "seed": NREL_SELECTION_SEED,
                "full_cohort_sha256": full_hash,
                "full_site_ids_by_state": full_by_state,
                "output_profile": matching[0],
                "output_state_counts": observed_counts,
                "output_site_ids": output_ids,
            }

    quality_reports: list[dict[str, Any]] = []
    quality_selection_status: dict[str, Any] = {
        "applicable": entries[0].dataset == "dkasc",
        "applied": False,
        "development_bypass": entries[0].dataset == "dkasc" and not enforce_dkasc_quality,
        "registered_evidence_eligible": not (
            entries[0].dataset == "dkasc" and not enforce_dkasc_quality
        ),
    }
    if enforce_dkasc_quality and entries[0].dataset == "dkasc":
        selection = select_dkasc_manifest_entries(
            entries,
            manifest.root,
            minimum_continuous_months=24,
            minimum_daylight_completeness=0.90,
            maximum_sites=12,
            allow_approximate_solar=allow_approximate_solar,
        )
        quality_reports = [asdict(report) for report in selection.reports]
        entries = selection.entries
        quality_selection_status = {
            "applicable": True,
            "applied": True,
            "development_bypass": False,
            "minimum_continuous_months": 24,
            "minimum_daylight_completeness": 0.90,
            "maximum_sites": 12,
            "candidate_count": len(selection.reports),
            "selected_count": len(selection.entries),
            "selected_site_ids": list(selection.selected_site_ids),
            "registered_evidence_eligible": True,
        }
        if not entries:
            raise ValueError("no DKASC device passed the registered quality criteria")

    safe_names = [_safe_name(entry.site_id) for entry in entries]
    if len(set(safe_names)) != len(safe_names):
        raise ValueError("site_id values collide after filesystem-safe normalization")

    combined: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "calibration": [],
        "test": [],
    }
    normalized_state_filters = None
    if states:
        normalized_state_filters = sorted(
            {_normalize_nrel_state(state) for state in states}
            if entries[0].dataset == "nrel"
            else {state.upper() for state in states}
        )
    cli_configuration = {
        "dataset": dataset_name,
        "site_ids": sorted(site_ids) if site_ids else None,
        "states_requested": sorted(states) if states else None,
        "states_normalized": normalized_state_filters,
        "spatial_roles": sorted(spatial_roles) if spatial_roles else None,
        "stride": stride,
        "ramp_method": ramp_method,
        "allow_approximate_solar": allow_approximate_solar,
        "require_observed_targets": require_observed_targets,
        "enforce_dkasc_quality": enforce_dkasc_quality,
        "enforce_nrel_selection": enforce_nrel_selection,
    }
    selected_dataset_kind = entries[0].dataset
    development_bypasses = []
    if selected_dataset_kind == "dkasc" and not enforce_dkasc_quality:
        development_bypasses.append("dkasc_quality_selection")
    if selected_dataset_kind == "nrel" and not enforce_nrel_selection:
        development_bypasses.append("nrel_registered_cohort_selection")
    if not require_observed_targets:
        development_bypasses.append("observed_daylight_target_requirement")
    if allow_approximate_solar:
        development_bypasses.append("pvlib_solar_physics")
    if selected_dataset_kind == "synthetic":
        development_bypasses.append("real_or_registered_simulated_evidence_source")
    if selected_dataset_kind == "solar_energy_137":
        development_bypasses.append("appendix_benchmark_only_evidence_source")
    registered_evidence_eligible = bool(
        quality_selection_status.get("registered_evidence_eligible", True)
        and nrel_selection.get("registered_evidence_eligible", True)
        and require_observed_targets
        and not allow_approximate_solar
        and selected_dataset_kind not in {"synthetic", "solar_energy_137"}
    )
    raw_source_files, raw_source_identities, raw_input_sha256 = _raw_source_provenance(
        entries, manifest.root
    )
    preparation_provenance = {
        "source_manifest_sha256": sha256_file(manifest_path.resolve()),
        "raw_input_sha256": raw_input_sha256,
        "raw_file_count": len(raw_source_files),
        "selected_dataset": selected_dataset_kind,
        "ramp_method": ramp_method,
        "require_observed_targets": bool(require_observed_targets),
        "allow_approximate_solar": bool(allow_approximate_solar),
        "dkasc_quality_enforced": bool(selected_dataset_kind != "dkasc" or enforce_dkasc_quality),
        "nrel_selection_enforced": bool(selected_dataset_kind != "nrel" or enforce_nrel_selection),
        "development_bypasses": development_bypasses,
    }
    quality_by_site = {str(report["site_id"]): report for report in quality_reports}
    site_metadata = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for site_index, entry in enumerate(entries):
        datasets, metadata = _prepare_site(
            entry,
            manifest.root,
            stride=stride,
            ramp_method=ramp_method,
            allow_approximate_solar=allow_approximate_solar,
            require_observed_targets=require_observed_targets,
        )
        metadata["quality_selection"] = {
            **quality_selection_status,
            "site_report": quality_by_site.get(entry.site_id),
            "selected": entry.site_id in set(quality_selection_status.get("selected_site_ids", ())),
        }
        metadata["nrel_selection"] = {
            "applicable": nrel_selection.get("applicable", False),
            "applied": nrel_selection.get("applied", False),
            "development_bypass": nrel_selection.get("development_bypass", False),
            "seed": nrel_selection.get("seed"),
            "selected_in_registered_cohort": entry.site_id
            in set(nrel_selection.get("selected_site_ids", ())),
            "registered_evidence_eligible": nrel_selection.get(
                "registered_evidence_eligible", True
            ),
        }
        metadata["preparation_filters"] = cli_configuration
        site_dir = output_dir / "sites" / _safe_name(entry.site_id)
        site_dir.mkdir(parents=True, exist_ok=True)
        schema = {
            "schema_version": 1,
            "dataset_kind": selected_dataset_kind,
            "simulated": selected_dataset_kind in {"nrel", "synthetic"},
            "registered_evidence_eligible": registered_evidence_eligible,
            "preparation_provenance": preparation_provenance,
            "raw_source_identities": raw_source_identities,
            "ramp_method": ramp_method,
            "registered_nrel_cohort": registered_nrel_cohort,
            "history_columns": metadata["history_columns"],
            "future_known_columns": metadata["future_known_columns"],
            "site_metadata_names": metadata["site_metadata_names"],
            "target": "power_norm",
            "state_labels": {"ignore": -100, "stable": 0, "ramp_up": 1, "ramp_down": 2},
            "site_id_map": {str(site_index): entry.site_id},
            "site_states": {str(site_index): entry.state},
            "spatial_roles": {str(site_index): entry.spatial_role},
            # Baselines that infer a ramp state from their median power forecast
            # must use the exact same train-only, site-specific threshold as the
            # target labels.  Persist it in every split sidecar so a cross-site
            # test never falls back to a pooled threshold from the source sites.
            "ramp_thresholds": {str(site_index): metadata["ramp"]},
        }
        if site_metadata:
            canonical = {
                "history_columns": site_metadata[0]["history_columns"],
                "future_known_columns": site_metadata[0]["future_known_columns"],
                "site_metadata_names": site_metadata[0]["site_metadata_names"],
            }
            for key, names in canonical.items():
                if metadata[key] != names:
                    raise ValueError(
                        f"site {entry.site_id} has a different {key} schema and cannot be combined"
                    )
        store_path = site_dir / "sequence.store"
        counts = save_window_sequence_store(
            store_path,
            datasets,
            site_index=site_index,
            schema=schema,
        )
        relative_store = store_path.relative_to(output_dir).as_posix()
        for split_name, count in counts.items():
            combined[split_name].append(
                {
                    "path": relative_store,
                    "split": split_name,
                    "length": count,
                    "site_index": site_index,
                    "site_id": entry.site_id,
                }
            )
        metadata["site_index"] = site_index
        metadata["window_counts"] = counts
        metadata["window_sequence_store"] = relative_store
        atomic_json_dump(metadata, site_dir / "metadata.json")
        site_metadata.append(metadata)

    split_content_identities = {}
    for split_name, parts in combined.items():
        split_content_identities[split_name] = save_window_shard_index(
            output_dir / f"{split_name}.index.json",
            parts,
            schema={
                "schema_version": 1,
                "dataset_kind": selected_dataset_kind,
                "simulated": selected_dataset_kind in {"nrel", "synthetic"},
                "registered_evidence_eligible": registered_evidence_eligible,
                "preparation_provenance": preparation_provenance,
                "raw_source_identities": raw_source_identities,
                "ramp_method": ramp_method,
                "registered_nrel_cohort": registered_nrel_cohort,
                "history_columns": site_metadata[0]["history_columns"],
                "future_known_columns": site_metadata[0]["future_known_columns"],
                "site_metadata_names": site_metadata[0]["site_metadata_names"],
                "target": "power_norm",
                "state_labels": {
                    "ignore": -100,
                    "stable": 0,
                    "ramp_up": 1,
                    "ramp_down": 2,
                },
                "site_id_map": {str(site["site_index"]): site["site_id"] for site in site_metadata},
                "site_states": {str(site["site_index"]): site["state"] for site in site_metadata},
                "spatial_roles": {
                    str(site["site_index"]): site["spatial_role"] for site in site_metadata
                },
                "ramp_thresholds": {
                    str(site["site_index"]): site["ramp"] for site in site_metadata
                },
            },
        )
    atomic_json_dump(
        {
            "manifest": str(manifest_path.resolve()),
            "manifest_root": str(manifest.root),
            "manifest_dataset_counts": manifest_dataset_counts,
            "selected_dataset": entries[0].dataset,
            "preparation_filters": cli_configuration,
            "ramp_method": ramp_method,
            "stride": stride,
            "require_observed_targets": require_observed_targets,
            "window_storage": {
                "format": "racepv-window-shard-index-v1",
                "site_store_format": "racepv-window-sequence-store-v1",
                "materialization": "lazy windows over one memory-mapped sequence per site",
                "overlapping_windows_stored": False,
                "combined_windows_materialized_in_memory": False,
                "split_indexes": {split: f"{split}.index.json" for split in combined},
                "split_content_sha256": split_content_identities,
            },
            "selected_spatial_roles": sorted({site["spatial_role"] for site in site_metadata}),
            "sites": site_metadata,
            "dkasc_quality_selection": quality_selection_status,
            "dkasc_quality_reports": quality_reports,
            "nrel_selection": nrel_selection,
            "dataset_kind": selected_dataset_kind,
            "simulated": selected_dataset_kind in {"nrel", "synthetic"},
            "preparation_provenance": preparation_provenance,
            "raw_source_files": raw_source_files,
            "registered_nrel_cohort": registered_nrel_cohort,
            "registered_evidence_eligible": registered_evidence_eligible,
        },
        output_dir / "prepared_manifest.json",
    )
    future_audit = audit_future_feature_names(FUTURE_COLUMNS, FUTURE_KNOWN_ALLOWLIST)
    atomic_json_dump(
        {
            "passed": future_audit.passed,
            "errors": future_audit.errors,
            "warnings": future_audit.warnings,
            "checks": {
                **future_audit.checks,
                "chronological_split_60_15_10_15": True,
                "windows_contained_inside_split": True,
                "normalizer_fit_partition": "train",
                "ramp_threshold_fit_partition": "train",
                "clear_sky_mapping_fit_partition": "train",
                "future_measured_weather_used": False,
                "mixed_dataset_archive": False,
                "nrel_registered_selection_applied": nrel_selection.get("applied", False)
                if nrel_selection.get("applicable", False)
                else None,
                "dkasc_registered_quality_applied": quality_selection_status.get("applied", False)
                if quality_selection_status.get("applicable", False)
                else None,
                "zero_parameter_target_power_mapping_boundary_passed": all(
                    site["zero_parameter_transfer_boundary"]["passed"] for site in site_metadata
                ),
                "observed_daylight_targets_required": require_observed_targets,
            },
            "evidence": {
                "future_known_columns": list(FUTURE_COLUMNS),
                "site_capacity_sources": {
                    site["site_id"]: site["capacity"]["capacity_source"] for site in site_metadata
                },
                "site_ramp_thresholds": {site["site_id"]: site["ramp"] for site in site_metadata},
                "data_sources": {
                    site["site_id"]: {
                        "dataset": site["dataset"],
                        "simulated": site["simulated"],
                    }
                    for site in site_metadata
                },
                "spatial_roles": {site["site_id"]: site["spatial_role"] for site in site_metadata},
                "site_coordinates_and_geometry": {
                    site["site_id"]: {
                        "timezone": site["timezone"],
                        "latitude": site["latitude"],
                        "longitude": site["longitude"],
                        "state": site["state"],
                        "fixed_tilt": site["fixed_tilt"],
                        "geometry": site["geometry"],
                    }
                    for site in site_metadata
                },
                "target_history_fitting": {
                    site["site_id"]: site["target_history_fitting"] for site in site_metadata
                },
                "preparation_filters": cli_configuration,
                "dkasc_quality_selection": quality_selection_status,
                "nrel_selection": nrel_selection,
            },
        },
        output_dir / "leakage_audit.json",
    )
    future_audit.require_pass()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--origin-safe",
        action="store_true",
        help="explicit schema-v2 engineering preparation; requires frozen splits and source pins",
    )
    parser.add_argument(
        "--frozen-split-contract",
        type=Path,
        help="existing absolute split contract JSON; never inferred from row counts",
    )
    parser.add_argument("--dataset", choices=("dkasc", "nrel", "solar_energy_137", "synthetic"))
    parser.add_argument("--site", action="append", dest="sites")
    parser.add_argument("--state", action="append", dest="states")
    parser.add_argument(
        "--spatial-role",
        action="append",
        dest="spatial_roles",
        choices=SPATIAL_ROLES,
        help="filter source_train, validation_calibration, or zero_parameter_test sites",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--ramp-method", choices=("q85", "q90", "q95", "fixed0.1"), default="q90")
    parser.add_argument("--allow-approximate-solar", action="store_true")
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--require-observed-targets",
        dest="require_observed_targets",
        action="store_true",
        help="registered default: every daylight target must be measured",
    )
    target_group.add_argument(
        "--allow-imputed-targets",
        dest="require_observed_targets",
        action="store_false",
        help="development only: permit <=30-minute interpolated targets",
    )
    parser.set_defaults(require_observed_targets=True)
    parser.add_argument(
        "--skip-dkasc-quality",
        action="store_true",
        help="development only; registered DKASC runs enforce 24 months/90%%/capacity/top-12",
    )
    parser.add_argument(
        "--skip-nrel-selection",
        action="store_true",
        help=(
            "development only; bypass deterministic five-state x20 fixed-tilt selection "
            "and mark the archive ineligible for registered evidence"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_manifest(
        args.manifest,
        args.output,
        dataset_name=args.dataset,
        site_ids=set(args.sites) if args.sites else None,
        states=set(args.states) if args.states else None,
        stride=args.stride,
        ramp_method=args.ramp_method,
        allow_approximate_solar=args.allow_approximate_solar,
        require_observed_targets=args.require_observed_targets,
        enforce_dkasc_quality=not args.skip_dkasc_quality,
        enforce_nrel_selection=not args.skip_nrel_selection,
        spatial_roles=set(args.spatial_roles) if args.spatial_roles else None,
        require_origin_safe=args.origin_safe,
        frozen_split_contract=read_strict_json(args.frozen_split_contract)
        if args.frozen_split_contract
        else None,
    )


if __name__ == "__main__":
    main()
