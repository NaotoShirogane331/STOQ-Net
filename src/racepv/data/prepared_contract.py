"""Strict origin-safe prepared metadata and semantic verification."""

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np

from racepv.data.causal_history import _POWER_DEPENDENCIES, _WEATHER_FEATURES, CausalHistoryState
from racepv.data.origin_clock import freeze_split_plan, origin_times, validate_source_time_contract
from racepv.io import (
    MODEL_ARRAYS,
    REGISTERED_DATASET_KINDS,
    REGISTERED_FUTURE_FEATURES,
    REGISTERED_HISTORY_FEATURES,
    REGISTERED_SITE_FEATURES,
    REGISTERED_SPATIAL_ROLES,
    REGISTERED_STATE_LABELS,
    _aggregate_file_identities,
)

SCHEMA_KIND = "racepv.prepared-window-contract"
PROFILE = "as-of-origin-v1"
V2_WINDOW_STORE_FORMAT = "racepv-window-store-v2"
V2_WINDOW_SEQUENCE_STORE_FORMAT = "racepv-window-sequence-store-v2"
V2_WINDOW_SHARD_INDEX_FORMAT = "racepv-window-shard-index-v2"
SPLITS = ("train", "validation", "calibration", "test")
STEP_NS = 900_000_000_000
POPULATION_ARRAYS = (
    "labels_ns",
    "split_ids",
    "power_observed",
    "weather_observed",
    "clear_sky_norm",
    "base_values_raw",
    "effective_power_observed",
    "gap_left",
    "gap_right",
    "row_targets",
    "row_state_labels",
    "row_future_known",
    "site_vector",
    "candidate_starts",
    "history_candidate_starts",
    "window_starts",
    "site_ids",
    "issue_ns",
)
ARRAY_ROLES = (
    *POPULATION_ARRAYS,
    "timestamps",
    "day_ids",
    "daylight_mask",
    "target_observed_mask",
    *MODEL_ARRAYS,
)
SCHEMA_FIELDS = (
    "schema_kind",
    "schema_version",
    "causality_profile",
    "dataset_kind",
    "simulated",
    "history_columns",
    "future_known_columns",
    "site_metadata_names",
    "target",
    "state_labels",
    "site",
    "geometry",
    "clock",
    "source_time_contract",
    "split_plan",
    "split",
    "causal_rules",
    "sources",
    "fit_states",
    "arrays",
    "forecast_population_sha256",
)
FIT_PURPOSES = {
    "capacity_normalization": "power_capacity_normalization",
    "weather_standardization": "historical_weather_normalization",
    "clear_sky_power_mapping": "deterministic_clear_sky_power",
    "ramp_label_threshold": "target_labeling_and_evaluation_not_model_input",
    "site_vector": "site_metadata_features",
}


@dataclass(frozen=True)
class _MemoryQualification:
    schema: dict
    metadata: dict
    evidence: dict
    arrays: dict


# Values contain only copied metadata and digests, never a strong Dataset reference.
_MEMORY_QUALIFICATIONS = WeakKeyDictionary()


def _bounded_array_identity(array):
    """C-order identity with fixed buffers, including noncontiguous resident arrays."""
    if type(array) is ResolverModelArray:
        return _array_identity(array)  # Existing resolver hashing streams 64-window chunks.
    if not isinstance(array, np.ndarray) or array.dtype.hasobject:
        raise TypeError("qualified arrays must be non-object numpy arrays")
    digest = hashlib.sha256()
    with np.nditer(
        array,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="C",
        buffersize=8192,
    ) as iterator:
        for chunk in iterator:
            digest.update(chunk.tobytes(order="C"))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": digest.hexdigest(),
        "bytes": array.nbytes,
    }


def _memory_qualification_inputs(dataset):
    prepared = dataset.prepared_contract
    _keys(prepared, ("schema", "causal_arrays", "evidence"), "qualified prepared_contract")
    _keys(prepared["causal_arrays"], ARRAY_ROLES, "qualified causal arrays")
    if type(dataset.causal_state) is not CausalHistoryState:
        raise TypeError("qualified Dataset requires its causal state")
    metadata = {}
    for name in (
        "lookback",
        "horizon",
        "stride",
        "split_bounds",
        "history_columns",
        "future_known_columns",
        "site_metadata_names",
        "require_observed_targets",
        "_base_future_names",
    ):
        value = getattr(dataset, name)
        metadata[name] = list(value) if isinstance(value, tuple) else value
    metadata["state_feature_names"] = list(dataset.causal_state.feature_names)
    metadata["index_dtype"] = str(dataset.frame.index.dtype)
    arrays = {"prepared." + name: array for name, array in prepared["causal_arrays"].items()}
    for name in (
        "labels_ns",
        "split_ids",
        "power_observed",
        "weather_observed",
        "clear_sky_norm",
        "base_values",
        "gap_left",
        "gap_right",
        "_effective_power_observed",
    ):
        arrays["state." + name] = getattr(dataset.causal_state, name)
    for name in (
        "candidate_starts",
        "history_candidate_starts",
        "window_starts",
        "_targets",
        "_states",
        "_target_observed",
        "_base_future",
        "_night",
        "_site_metadata",
        "_history_values",
        "_history_mask",
    ):
        arrays["dataset." + name] = getattr(dataset, name)
    arrays["dataset.index"] = dataset.frame.index.asi8
    return prepared, metadata, arrays


def _register_memory_qualification(dataset):
    prepared, metadata, arrays = _memory_qualification_inputs(dataset)
    _MEMORY_QUALIFICATIONS[dataset] = _MemoryQualification(
        schema=copy.deepcopy(prepared["schema"]),
        metadata=copy.deepcopy(metadata),
        evidence={
            name: (len(data), hashlib.sha256(data).hexdigest())
            for name, data in prepared["evidence"].items()
        },
        arrays={name: _bounded_array_identity(array) for name, array in arrays.items()},
    )


def read_qualified_memory_audits(dataset, indices, names):
    """Revalidate prior full qualification with bounded memory, then copy audit rows.

    Validation scans all resident bytes (O(N) time), but neither reconstructs source
    state nor creates population-sized windows/audit expectations. Even empty reads
    recheck qualification. Call the full public guard explicitly before the first read.
    """
    from racepv.data.dataset import WindowedPVDataset
    from racepv.io import AUDIT_ARRAYS, _window_indices, _window_names

    if type(dataset) is not WindowedPVDataset:
        raise TypeError("memory audits require a WindowedPVDataset")
    positions = _window_indices(indices, len(dataset))
    requested = _window_names(names)
    if any(name not in AUDIT_ARRAYS for name in requested):
        raise KeyError("qualified memory seam accepts only audit outputs")
    record = _MEMORY_QUALIFICATIONS.get(dataset)
    if record is None:
        raise ValueError("memory audits require prior successful full qualification")
    prepared, metadata, arrays = _memory_qualification_inputs(dataset)
    _exact(prepared["schema"], record.schema, "qualified schema")
    _exact(metadata, record.metadata, "qualified Dataset metadata")
    _keys(prepared["evidence"], record.evidence, "qualified evidence")
    for name, (size, digest) in record.evidence.items():
        data = prepared["evidence"][name]
        if (
            type(data) is not bytes
            or len(data) != size
            or hashlib.sha256(data).hexdigest() != digest
        ):
            raise ValueError("qualified evidence changed: " + name)
    for name, array in arrays.items():
        if _bounded_array_identity(array) != record.arrays[name]:
            raise ValueError("qualified array changed: " + name)
    return {
        name: np.array(prepared["causal_arrays"][name][positions], copy=True) for name in requested
    }


def _keys(value, keys, name):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{name} requires exactly fields {sorted(keys)}")


def _integer(value, name, *, minimum=0):
    if type(value) is not int:
        raise TypeError(f"{name} must be a strict integer")
    if value < minimum or value > np.iinfo(np.int64).max:
        raise ValueError(f"{name} is outside its integer bounds")
    return value


def _exact(value, expected, name):
    # JSON equality alone equates True, 1 and 1.0, including inside lists/dicts.
    if type(value) is not type(expected):
        raise TypeError(f"{name} has an invalid type")
    if isinstance(expected, dict):
        _keys(value, expected, name)
        for key in expected:
            _exact(value[key], expected[key], f"{name}.{key}")
    elif isinstance(expected, list):
        if len(value) != len(expected):
            raise ValueError(f"{name} has an invalid length")
        for i, item in enumerate(expected):
            _exact(value[i], item, f"{name}[{i}]")
    elif value != expected:
        raise ValueError(f"{name} does not match the registered contract")


def _sha(value, name):
    if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _relative_name(relative):
    if type(relative) is not str or not relative or relative.startswith("/"):
        raise ValueError("artifact path must be a canonical POSIX relative file")
    if any(part in {"", ".", ".."} or part.endswith((" ", ".")) for part in relative.split("/")):
        raise ValueError("artifact path must be a canonical POSIX relative file")
    if re.search(r"[\\\\:\x00-\x1f]", relative):
        raise ValueError("artifact path must be a canonical POSIX relative file")
    return relative


def _file_identity(value):
    _keys(value, ("path", "sha256", "bytes"), "file identity")
    _relative_name(value["path"])
    _sha(value["sha256"], "file sha256")
    _integer(value["bytes"], "file bytes")


def canonical_json_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def forecast_population_sha256(schema):
    """Logical population identity, deliberately distinct from the file aggregate."""
    return canonical_json_sha256(
        {
            "causality_profile": schema["causality_profile"],
            "clock": schema["clock"],
            "source_time_contract": schema["source_time_contract"],
            "split_plan": schema["split_plan"],
            "split": schema["split"],
            "geometry": schema["geometry"],
            "site": schema["site"],
            "arrays": {name: schema["arrays"][name] for name in POPULATION_ARRAYS},
        }
    )


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number: {value}")
    return parsed


def read_strict_json(path):
    return _strict_json_value(Path(path).read_text(encoding="utf-8"))


def _strict_json_value(text):
    value = json.loads(
        text,
        object_pairs_hook=unique_object,
        parse_constant=_invalid_constant,
        parse_float=_finite_float,
    )
    if type(value) is not dict:
        raise ValueError("JSON root must be an object")
    return value


def contained_artifact_file(root, relative):
    _relative_name(relative)
    parts = relative.split("/")
    root = Path(root)
    for parent in (root, *root.parents):
        if parent.is_symlink():
            raise ValueError("artifact root must not contain symlink components")
    root = root.resolve(strict=True)
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("artifact file must not contain symlink components")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("artifact path escapes root or is not a file")
    return resolved


def _schema_evidence(schema):
    result = list(schema["sources"].values())
    for state in schema["fit_states"].values():
        result.append(state["file"])
        for name in ("fit", "training_audit"):
            if state[name] is not None:
                result.append(state[name]["input_file"])
    return result


def _number(value, name, *, minimum=None, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite numeric scalar")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _training_receipt(receipt, schema, *, selection, columns, all_finite=False):
    _keys(
        receipt,
        (
            "partition",
            "start_row",
            "stop_row",
            "start_ns",
            "end_exclusive_ns",
            "input_file",
            "raw_input_sha256",
            "selection",
            "accounting",
        ),
        "fit receipt",
    )
    train = schema["split_plan"]["splits"]["train"]
    _exact(receipt["partition"], "train", "fit.partition")
    for field in ("start_row", "stop_row", "start_ns", "end_exclusive_ns"):
        _exact(receipt[field], train[field], "fit." + field)
    _file_identity(receipt["input_file"])
    _exact(
        receipt["raw_input_sha256"],
        schema["sources"].get("frozen_fit_raw_input", schema["sources"]["raw_input"])["sha256"],
        "fit.raw_input_sha256",
    )
    _exact(receipt["selection"], selection, "fit.selection")
    _keys(receipt["accounting"], columns, "fit.accounting")
    rows = train["stop_row"] - train["start_row"]
    for column, counts in receipt["accounting"].items():
        _keys(
            counts,
            (
                "observed_rows",
                "filled_rows",
                "missing_rows",
                "used_observed_rows",
                "used_filled_rows",
                "used_rows",
            ),
            "fit accounting " + column,
        )
        for key, value in counts.items():
            _integer(value, "fit accounting " + key)
        if counts["observed_rows"] + counts["filled_rows"] + counts["missing_rows"] != rows:
            raise ValueError("fit accounting must cover the actual training interval")
        if (
            counts["used_observed_rows"] > counts["observed_rows"]
            or counts["used_filled_rows"] > counts["filled_rows"]
            or counts["used_rows"] != counts["used_observed_rows"] + counts["used_filled_rows"]
            or counts["used_rows"] < 1
        ):
            raise ValueError("fit accounting used/observed/filled counts disagree")
        if all_finite and (
            counts["used_observed_rows"] != counts["observed_rows"]
            or counts["used_filled_rows"] != counts["filled_rows"]
        ):
            raise ValueError("fit selection must preserve finite observed and closed-filled rows")


def _validate_fit_states(schema):
    states = schema["fit_states"]
    _keys(states, FIT_PURPOSES, "fit_states")
    for role, record in states.items():
        _keys(
            record,
            ("method", "purpose", "file", "state", "fit", "training_audit", "provenance"),
            "fit_states." + role,
        )
        _exact(record["purpose"], FIT_PURPOSES[role], role + ".purpose")
        _file_identity(record["file"])
        if type(record["method"]) is not str:
            raise TypeError("state method must be a string")
    capacity = states["capacity_normalization"]
    method, state = capacity["method"], capacity["state"]
    if method not in {"rated_capacity", "training_q99.5"}:
        raise ValueError("unsupported capacity method")
    _keys(
        state,
        ("capacity_kw", "capacity_source", "capacity_quantile", "clip_power", "eps"),
        "capacity state",
    )
    _number(state["capacity_kw"], "capacity_kw", minimum=1e-6)
    if state["capacity_kw"] <= 1e-6:
        raise ValueError("capacity_kw must exceed the normalizer eps")
    _exact(state["capacity_source"], method, "capacity_source")
    _exact(state["capacity_quantile"], 0.995, "capacity_quantile")
    _exact(state["clip_power"], [0.0, 1.05], "capacity.clip_power")
    _exact(state["eps"], 1e-6, "capacity.eps")
    learned = {"capacity_normalization": method == "training_q99.5"}
    parameter_sources = {
        "capacity_normalization": "site_training"
        if learned["capacity_normalization"]
        else "site_metadata"
    }
    if learned["capacity_normalization"]:
        _training_receipt(
            capacity["fit"],
            schema,
            selection="finite_train_power_including_closed_fills",
            columns=("power",),
            all_finite=True,
        )

    weather = states["weather_standardization"]
    method, state = weather["method"], weather["state"]
    if method not in {"train_mean_population_std", "absent"}:
        raise ValueError("unsupported weather method")
    _keys(
        state,
        (
            "feature_columns",
            "feature_mean",
            "feature_scale",
            "ddof",
            "eps",
            "nonfinite_mean_fallback",
            "nonfinite_or_small_scale_fallback",
        ),
        "weather state",
    )
    columns = state["feature_columns"]
    names = ("ghi", "temperature", "humidity", "wind_speed")
    if type(columns) is not list or any(
        type(name) is not str or name not in names for name in columns
    ):
        raise ValueError("weather feature selection must use registered columns")
    _exact(columns, [name for name in names if name in columns], "weather feature order")
    if bool(columns) != (method == "train_mean_population_std"):
        raise ValueError("weather absent/normalization method disagrees with available columns")
    for key in ("feature_mean", "feature_scale"):
        _keys(state[key], columns, "weather." + key)
        for name, value in state[key].items():
            _number(value, name)
            if key == "feature_scale" and value <= 1e-6:
                raise ValueError("weather scale must exceed eps, including unit fallback")
    for key, expected in (
        ("ddof", 0),
        ("eps", 1e-6),
        ("nonfinite_mean_fallback", 0.0),
        ("nonfinite_or_small_scale_fallback", 1.0),
    ):
        _exact(state[key], expected, "weather." + key)
    learned["weather_standardization"] = bool(columns)
    parameter_sources["weather_standardization"] = "site_training" if columns else "fixed_protocol"
    if columns:
        _training_receipt(
            weather["fit"],
            schema,
            selection="finite_train_weather_including_closed_fills",
            columns=columns,
            all_finite=True,
        )

    clear = states["clear_sky_power_mapping"]
    method, state = clear["method"], clear["state"]
    common = ("capacity_kw", "solar_engine", "clear_sky_model", "power_model", "clip_power")
    extras = {
        "geometry_pvwatts": ("gamma_pdc", "losses_fraction", "temp_cell_celsius", "poa_model"),
        "geometry_capacity_ghi_approximation": ("ghi_divisor",),
        "train_empirical_q95": ("mapping_quantile", "scale_kw_per_wm2", "n_training_points"),
    }
    if method not in extras:
        raise ValueError("unsupported clear-sky method")
    _keys(state, (*common, *extras[method]), "clear-sky state")
    _exact(state["capacity_kw"], capacity["state"]["capacity_kw"], "clear-sky capacity")
    _exact(state["clip_power"], [0.0, 1.05], "clear-sky clip_power")
    solar_pairs = {"pvlib": "ineichen", "noaa_approximation": "haurwitz_approximation"}
    if type(state["solar_engine"]) is not str or state["solar_engine"] not in solar_pairs:
        raise ValueError("unsupported solar engine")
    _exact(
        state["clear_sky_model"], solar_pairs[state["solar_engine"]], "clear-sky irradiance model"
    )
    learned["clear_sky_power_mapping"] = method == "train_empirical_q95"
    parameter_sources["clear_sky_power_mapping"] = (
        "site_training" if learned["clear_sky_power_mapping"] else "site_metadata"
    )
    if method == "geometry_pvwatts":
        for key, expected in (
            ("solar_engine", "pvlib"),
            ("power_model", "pvwatts"),
            ("gamma_pdc", -0.0035),
            ("losses_fraction", 0.14),
            ("temp_cell_celsius", 25.0),
            ("poa_model", "isotropic"),
        ):
            _exact(state[key], expected, "clear-sky." + key)
    elif method == "geometry_capacity_ghi_approximation":
        _exact(state["solar_engine"], "noaa_approximation", "clear-sky solar_engine")
        _exact(state["power_model"], "approximate_capacity_ghi_mapping", "clear-sky power_model")
        _exact(state["ghi_divisor"], 1000.0, "clear-sky ghi_divisor")
        if schema["dataset_kind"] != "synthetic":
            raise ValueError("approximate capacity/GHI mapping is synthetic-only")
    else:
        _exact(state["power_model"], "training_scale_mapping", "clear-sky power_model")
        _exact(state["mapping_quantile"], 0.95, "clear-sky mapping_quantile")
        _number(state["scale_kw_per_wm2"], "clear-sky scale", minimum=0)
        if state["scale_kw_per_wm2"] <= 0:
            raise ValueError("clear-sky scale must be positive")
        _integer(state["n_training_points"], "n_training_points", minimum=20)
        _training_receipt(
            clear["fit"],
            schema,
            selection="finite_nonnegative_train_power_ghi_gt_50_daylight_including_closed_fills",
            columns=("power",),
        )
        _exact(
            state["n_training_points"],
            clear["fit"]["accounting"]["power"]["used_rows"],
            "clear-sky training point accounting",
        )

    ramp = states["ramp_label_threshold"]
    method, state = ramp["method"], ramp["state"]
    quantiles = {"q85": 0.85, "q90": 0.9, "q95": 0.95, "fixed0.1": None}
    if method not in quantiles:
        raise ValueError("unsupported ramp method")
    _keys(
        state,
        ("method", "threshold", "fixed_threshold", "n_training_ramps", "quantile"),
        "ramp state",
    )
    _exact(state["method"], method, "ramp method/state")
    _exact(state["quantile"], quantiles[method], "ramp quantile")
    _exact(state["fixed_threshold"], 0.1, "ramp fixed_threshold")
    _number(state["threshold"], "ramp threshold", minimum=0)
    if state["threshold"] <= 0:
        raise ValueError("ramp threshold must be positive")
    learned["ramp_label_threshold"] = method != "fixed0.1"
    parameter_sources["ramp_label_threshold"] = (
        "site_training" if learned["ramp_label_threshold"] else "fixed_protocol"
    )
    audit = ramp["fit"] if learned["ramp_label_threshold"] else ramp["training_audit"]
    _training_receipt(
        audit, schema, selection="finite_observed_daylight_30min_endpoints", columns=("power",)
    )
    _integer(state["n_training_ramps"], "n_training_ramps", minimum=1)
    _exact(audit["accounting"]["power"]["used_filled_rows"], 0, "ramp fitted endpoints")
    _exact(state["n_training_ramps"], audit["accounting"]["power"]["used_rows"], "ramp count")
    if state["n_training_ramps"] > audit["stop_row"] - audit["start_row"] - 2:
        raise ValueError("ramp training count exceeds possible 30-minute endpoint pairs")
    if method == "fixed0.1":
        _exact(state["threshold"], 0.1, "ramp fixed threshold state")

    site = states["site_vector"]
    _exact(site["method"], "fixed_site_transforms", "site_vector method")
    _keys(
        site["state"],
        (
            "latitude_divisor",
            "longitude_divisor",
            "log_capacity_divisor",
            "tilt_divisor",
            "azimuth_divisor",
            "missing_geometry_value",
            "capacity_kw",
            "values",
        ),
        "site_vector state",
    )
    for key, expected in (
        ("latitude_divisor", 90.0),
        ("longitude_divisor", 180.0),
        ("log_capacity_divisor", 10.0),
        ("tilt_divisor", 90.0),
        ("azimuth_divisor", 360.0),
        ("missing_geometry_value", 0.0),
        ("capacity_kw", capacity["state"]["capacity_kw"]),
    ):
        _exact(site["state"][key], expected, "site_vector." + key)
    if type(site["state"]["values"]) is not list or len(site["state"]["values"]) != 6:
        raise ValueError("site_vector values require six fixed-transform values")
    for value in site["state"]["values"]:
        _number(value, "site_vector value")
    learned["site_vector"] = False
    parameter_sources["site_vector"] = "fixed_protocol"
    for role, record in states.items():
        if not learned[role] and record["fit"] is not None:
            raise ValueError(f"fixed state {role} must not claim a training fit")
        has_audit = role == "ramp_label_threshold" and record["method"] == "fixed0.1"
        if not has_audit and record["training_audit"] is not None:
            raise ValueError(f"unexpected training audit for {role}")
        _exact(
            record["provenance"],
            {
                "site_id": schema["site"]["id"],
                "spatial_role": schema["site"]["spatial_role"],
                "parameter_source": parameter_sources[role],
                "training_role": (
                    (
                        "source_site"
                        if schema["site"]["spatial_role"] == "source_train"
                        else "target_site"
                    )
                    if learned[role] or has_audit
                    else None
                ),
            },
            role + ".provenance",
        )
    if schema["site"]["spatial_role"] == "zero_parameter_test" and (
        learned["capacity_normalization"] or learned["clear_sky_power_mapping"]
    ):
        raise ValueError("zero-parameter sites require rated capacity and complete geometry")


def _validate_state_documents(schema, documents):
    """Check typed saved state receipts; never execute producers or refit statistics."""
    site = documents[schema["sources"]["site_metadata"]["path"]]
    _keys(
        site,
        (
            "latitude",
            "longitude",
            "timezone",
            "rated_capacity_kw",
            "surface_tilt",
            "surface_azimuth",
        ),
        "site source",
    )
    _number(site["latitude"], "latitude", minimum=-90, maximum=90)
    _number(site["longitude"], "longitude", minimum=-180, maximum=180)
    validate_source_time_contract({**schema["source_time_contract"], "timezone": site["timezone"]})
    for key in ("surface_tilt", "surface_azimuth"):
        if site[key] is not None:
            _number(site[key], key)
    if site["rated_capacity_kw"] is not None:
        _number(site["rated_capacity_kw"], "rated capacity", minimum=1e-6)
    states = schema["fit_states"]
    for role, record in states.items():
        expected = {"role": role, **{key: value for key, value in record.items() if key != "file"}}
        _exact(documents[record["file"]["path"]], expected, role + " state receipt identity")
    capacity = states["capacity_normalization"]
    if capacity["method"] == "rated_capacity":
        _exact(site["rated_capacity_kw"], capacity["state"]["capacity_kw"], "rated capacity source")
    elif site["rated_capacity_kw"] is not None:
        raise ValueError("train capacity fallback cannot replace supplied rated capacity")
    geometry = site["surface_tilt"] is not None and site["surface_azimuth"] is not None
    if geometry == (states["clear_sky_power_mapping"]["method"] == "train_empirical_q95"):
        raise ValueError("clear-sky method disagrees with actual site geometry")
    expected_vector = [
        site["latitude"] / 90.0,
        site["longitude"] / 180.0,
        float(np.log1p(capacity["state"]["capacity_kw"]) / 10.0),
        (site["surface_tilt"] or 0.0) / 90.0,
        (site["surface_azimuth"] or 0.0) / 360.0,
        float(geometry),
    ]
    _exact(states["site_vector"]["state"]["values"], expected_vector, "fixed site vector state")


def deterministic_sampling_points():
    """Frozen left-label sampling; lead_time_hours remains an interval-end distance."""
    return {
        "history": {"clear_sky_power_norm": "history_interval_start_point"},
        "night_threshold": {"clear_sky_power_norm": "history_interval_start_point"},
        "future_known": {
            name: "target_interval_start_point"
            for name in REGISTERED_FUTURE_FEATURES
            if name != "lead_time_hours"
        },
    }


def _validate_global_split_plan(plan, selected_split):
    """Validate the complete four-split metadata plan without allocating grid rows."""
    _keys(
        plan, ("grid_start_ns", "grid_step_ns", "row_count", "split_order", "splits"), "split_plan"
    )
    _integer(plan["grid_start_ns"], "grid_start_ns", minimum=np.iinfo(np.int64).min + 1)
    _exact(plan["grid_step_ns"], STEP_NS, "grid_step_ns")
    n = _integer(plan["row_count"], "row_count", minimum=1)
    _exact(plan["split_order"], list(SPLITS), "split_order")
    _keys(plan["splits"], SPLITS, "splits")
    next_row = 0
    for name in SPLITS:
        bounds = plan["splits"][name]
        _keys(bounds, ("start_row", "stop_row", "start_ns", "end_exclusive_ns"), f"split.{name}")
        for field in ("start_row", "stop_row"):
            _integer(bounds[field], field)
        if bounds["start_row"] != next_row or bounds["stop_row"] - next_row < 216:
            raise ValueError("split partitions must be contiguous with nonempty formal candidates")
        for field, row in (("start_ns", next_row), ("end_exclusive_ns", bounds["stop_row"])):
            expected = plan["grid_start_ns"] + row * STEP_NS
            _integer(expected, field, minimum=np.iinfo(np.int64).min + 1)
            _exact(bounds[field], expected, field)
        next_row = bounds["stop_row"]
    if next_row != n or type(selected_split) is not str or selected_split not in SPLITS:
        raise ValueError("split plan must cover all rows and select a registered split")
    return n


def _validate_scientific_schema_fields(schema):
    """Validate shared scientific metadata while leaving version and array scope explicit."""
    if (
        type(schema["dataset_kind"]) is not str
        or schema["dataset_kind"] not in REGISTERED_DATASET_KINDS
    ):
        raise ValueError("unsupported dataset_kind")
    _exact(schema["simulated"], schema["dataset_kind"] in {"synthetic", "nrel"}, "simulated")
    for field, expected in (
        ("history_columns", list(REGISTERED_HISTORY_FEATURES)),
        ("future_known_columns", list(REGISTERED_FUTURE_FEATURES)),
        ("site_metadata_names", list(REGISTERED_SITE_FEATURES)),
        ("state_labels", REGISTERED_STATE_LABELS),
        ("target", "power_norm"),
    ):
        _exact(schema[field], expected, field)
    _keys(schema["site"], ("id", "name", "spatial_role"), "site")
    _integer(schema["site"]["id"], "site.id")
    if type(schema["site"]["name"]) is not str or not schema["site"]["name"].strip():
        raise ValueError("site.name must be nonempty")
    if (
        type(schema["site"]["spatial_role"]) is not str
        or schema["site"]["spatial_role"] not in REGISTERED_SPATIAL_ROLES
    ):
        raise ValueError("unsupported spatial role")
    _keys(schema["geometry"], ("lookback", "horizon", "stride"), "geometry")
    _exact(schema["geometry"]["lookback"], 192, "lookback")
    _exact(schema["geometry"]["horizon"], 24, "horizon")
    _integer(schema["geometry"]["stride"], "stride", minimum=1)
    _exact(
        schema["clock"],
        {
            "sampling_interval_ns": STEP_NS,
            "stored_timezone": "UTC",
            "timestamps": "target_interval_start_ns",
            "issue": "last_history_interval_end",
            "lead": "target_interval_end_minus_issue",
            "day_ids": "utc_days_since_epoch",
            "deterministic_sampling_points": deterministic_sampling_points(),
        },
        "clock",
    )
    source = validate_source_time_contract(schema["source_time_contract"])
    if (source["evidence"]["kind"] == "synthetic_fixture") != (
        schema["dataset_kind"] == "synthetic"
    ):
        raise ValueError("synthetic source evidence and dataset_kind must agree")
    plan = schema["split_plan"]
    n = _validate_global_split_plan(plan, schema["split"])
    _exact(
        schema["causal_rules"],
        {
            "max_gap_steps": 2,
            "gap_right": "strictly_before_first_target",
            "endpoint_scope": "same_split_real_observations",
            "unclosed_daylight": "reject",
            "night": "clear_sky_norm_lt_0.01_source_precision",
            "missing_values": "zero_value_zero_mask",
            "imputed_values": "never_exposed",
            "provenance_flag_mask": 1,
            "power_dependencies": {key: list(value) for key, value in _POWER_DEPENDENCIES.items()},
            "weather_sources": dict(_WEATHER_FEATURES),
            "deterministic_history": ["clear_sky_power_norm"],
            "provenance_flags": ["power_observed", "power_imputed"],
        },
        "causal_rules",
    )
    source_roles = ("time_evidence", "raw_input", "producer", "site_metadata")
    if type(schema["sources"]) is dict and "frozen_fit_raw_input" in schema["sources"]:
        source_roles = (*source_roles, "frozen_fit_raw_input")
    _keys(schema["sources"], source_roles, "sources")
    for value in schema["sources"].values():
        _file_identity(value)
    if "frozen_fit_raw_input" in schema["sources"] and (
        schema["sources"]["frozen_fit_raw_input"]["sha256"]
        == schema["sources"]["raw_input"]["sha256"]
    ):
        raise ValueError("frozen fit raw input role requires a distinct original source")
    time_file = schema["sources"]["time_evidence"]
    if (
        source["evidence"]["reference"] != time_file["path"]
        or source["evidence"]["sha256"] != time_file["sha256"]
    ):
        raise ValueError("time evidence reference identity disagrees with sources")
    _validate_fit_states(schema)
    paths = [item["path"] for item in _schema_evidence(schema)]
    if len({path.casefold() for path in paths}) != len(paths):
        raise ValueError("evidence file roles must not alias")
    return n


def _validate_array_schema_fields(schema, *, row_count, population_hasher):
    """Validate logical role identities and shapes for an explicit stored row scope."""
    _keys(schema["arrays"], ARRAY_ROLES, "arrays")
    for name, item in schema["arrays"].items():
        _keys(item, ("shape", "dtype", "sha256", "bytes"), f"array.{name}")
        if type(item["shape"]) is not list:
            raise TypeError("array shape must be a list")
        for dim in item["shape"]:
            _integer(dim, "array dimension")
        if type(item["dtype"]) is not str or item["dtype"] not in {
            "float32",
            "float64",
            "int64",
            "bool",
        }:
            raise ValueError("unsupported array dtype")
        _sha(item["sha256"], f"{name}.sha256")
        _integer(item["bytes"], f"{name}.bytes")
        if item["bytes"] != math.prod(item["shape"]) * np.dtype(item["dtype"]).itemsize:
            raise ValueError(f"array {name} shape/dtype/bytes mismatch")
    specs = schema["arrays"]
    for role in ("candidate_starts", "history_candidate_starts", "window_starts"):
        if len(specs[role]["shape"]) != 1:
            raise ValueError("population indices must be one dimensional")
    c, h, f = [
        specs[role]["shape"][0]
        for role in ("candidate_starts", "history_candidate_starts", "window_starts")
    ]
    bounds = schema["split_plan"]["splits"][schema["split"]]
    expected_c = (bounds["stop_row"] - bounds["start_row"] - 216) // schema["geometry"][
        "stride"
    ] + 1
    if c != expected_c or not 0 < f <= h <= c:
        raise ValueError("candidate/history/final population counts are inconsistent or empty")
    shapes = {
        "labels_ns": [row_count],
        "split_ids": [row_count],
        "power_observed": [row_count],
        "weather_observed": [row_count, 4],
        "clear_sky_norm": [row_count],
        "base_values_raw": [row_count, 14],
        "effective_power_observed": [row_count],
        "gap_left": [row_count],
        "gap_right": [row_count],
        "row_targets": [row_count],
        "row_state_labels": [row_count],
        "row_future_known": [row_count, 8],
        "site_vector": [6],
        "candidate_starts": [c],
        "history_candidate_starts": [h],
        "window_starts": [f],
        "site_ids": [f],
        "issue_ns": [f],
        "timestamps": [f, 24],
        "day_ids": [f, 24],
        "daylight_mask": [f, 24],
        "target_observed_mask": [f, 24],
        "history_values": [f, 192, 14],
        "history_mask": [f, 192, 14],
        "future_known": [f, 24, 9],
        "site_metadata": [f, 6],
        "targets": [f, 24],
        "state_labels": [f, 24],
    }
    bools = {
        "power_observed",
        "weather_observed",
        "effective_power_observed",
        "daylight_mask",
        "target_observed_mask",
    }
    source_floats = {"clear_sky_norm", "base_values_raw", "row_targets", "row_future_known"}
    model_floats = set(MODEL_ARRAYS) - {"state_labels"}
    for name, shape in shapes.items():
        _exact(specs[name]["shape"], shape, f"{name}.shape")
        allowed = (
            {"bool"}
            if name in bools
            else {"float32", "float64"}
            if name in source_floats
            else {"float32"}
            if name in model_floats | {"site_vector"}
            else {"int64"}
        )
        if specs[name]["dtype"] not in allowed:
            raise ValueError(f"{name} has an invalid dtype for its role")
    _sha(schema["forecast_population_sha256"], "forecast_population_sha256")
    if population_hasher(schema) != schema["forecast_population_sha256"]:
        raise ValueError("forecast population identity mismatch")


def validate_v2_schema(schema):
    """Validate the complete storage-independent schema; no file or array opens."""
    _keys(schema, SCHEMA_FIELDS, "schema")
    _exact(schema["schema_kind"], SCHEMA_KIND, "schema_kind")
    _exact(schema["schema_version"], 2, "schema_version")
    _exact(schema["causality_profile"], PROFILE, "causality_profile")
    n = _validate_scientific_schema_fields(schema)
    _validate_array_schema_fields(
        schema,
        row_count=n,
        population_hasher=forecast_population_sha256,
    )
    return copy.deepcopy(schema)


def _verify_bytes(item, data):
    if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise ValueError(f"content identity mismatch: {item['path']}")


def _leaf_graph(descriptor_path, *, read_json=None):
    """Collect and validate the entire leaf reference graph before hashing files."""
    read_json = read_strict_json if read_json is None else read_json
    descriptor_path = Path(descriptor_path)
    if descriptor_path.suffix != ".json":
        raise ValueError("descriptor must use a JSON publication name")
    root = descriptor_path.parent
    descriptor_path = contained_artifact_file(root, descriptor_path.name)
    receipt_name = _receipt_name(descriptor_path)
    receipt_path = contained_artifact_file(root, receipt_name)
    descriptor = read_json(descriptor_path)
    _keys(
        descriptor,
        (
            "artifact_kind",
            "schema_version",
            "causality_profile",
            "format",
            "schema",
            "files",
            "arrays",
            "aggregate_content_sha256",
            "forecast_population_sha256",
        ),
        "descriptor",
    )
    _exact(descriptor["artifact_kind"], "racepv.prepared-descriptor", "artifact_kind")
    _exact(descriptor["schema_version"], 2, "descriptor.schema_version")
    _exact(descriptor["causality_profile"], PROFILE, "descriptor.causality_profile")
    if type(descriptor["format"]) is not str or descriptor["format"] not in {
        "npz",
        V2_WINDOW_STORE_FORMAT,
        V2_WINDOW_SEQUENCE_STORE_FORMAT,
    }:
        raise ValueError("unsupported artifact format")
    if descriptor["format"] == "npz" and not descriptor_path.name.endswith(".descriptor.json"):
        raise ValueError("NPZ requires its same-stem .descriptor.json")
    if type(descriptor["files"]) is not list or not descriptor["files"]:
        raise ValueError("descriptor.files must be nonempty")
    files = {}
    resolved = {}
    aliases = set()
    physical = set()
    for item in descriptor["files"]:
        _file_identity(item)
        path = item["path"]
        if path.casefold() in aliases:
            raise ValueError("duplicate or aliased file path")
        aliases.add(path.casefold())
        full = contained_artifact_file(root, path)
        inode = (full.stat().st_dev, full.stat().st_ino)
        if inode in physical:
            raise ValueError("same physical file has conflicting logical roles")
        physical.add(inode)
        files[path] = item
        resolved[path] = full
    schema_name = _relative_name(descriptor["schema"])
    if schema_name not in files or not schema_name.endswith(".schema.json"):
        raise ValueError("descriptor must pin its schema")
    if (
        descriptor["format"] == "npz"
        and schema_name != descriptor_path.name.removesuffix(".descriptor.json") + ".schema.json"
    ):
        raise ValueError("NPZ schema and descriptor must share a stem")
    schema = validate_v2_schema(read_json(resolved[schema_name]))
    _keys(descriptor["arrays"], ARRAY_ROLES, "descriptor arrays")
    used = {schema_name}
    role_pairs = set()
    array_paths = set()
    for name, ref in descriptor["arrays"].items():
        if descriptor["format"] == V2_WINDOW_SEQUENCE_STORE_FORMAT and name in MODEL_ARRAYS:
            _exact(ref, resolver_reference(name), "sequence virtual array reference " + name)
            continue
        _keys(ref, ("path", "member"), f"array reference {name}")
        _relative_name(ref["path"])
        if ref["path"] not in files:
            raise ValueError(f"array reference {name} has no file identity")
        if descriptor["format"] == "npz":
            _exact(ref["member"], name, f"{name}.member")
            if ref["path"] != descriptor_path.name.removesuffix(".descriptor.json") + ".npz":
                raise ValueError("NPZ member containers must share the descriptor stem")
        else:
            if ref["member"] is not None or not ref["path"].endswith(".npy"):
                raise ValueError("non-NPZ roles require distinct NPY files")
        pair = (ref["path"], ref["member"])
        if pair in role_pairs:
            raise ValueError("array roles alias the same file/member")
        role_pairs.add(pair)
        array_paths.add(ref["path"])
        used.add(ref["path"])
    evidence = _schema_evidence(schema)
    for item in evidence:
        if item["path"] in used or files.get(item["path"]) != item:
            raise ValueError("evidence role conflict or missing file identity")
        used.add(item["path"])
    if schema_name in array_paths or used != set(files):
        raise ValueError("file graph contains conflicting or unreferenced roles")
    if descriptor_path.name in files or receipt_name in files:
        raise ValueError("descriptor/schema/receipt graph must be acyclic")
    state_paths = [
        schema["sources"]["site_metadata"]["path"],
        *(record["file"]["path"] for record in schema["fit_states"].values()),
    ]
    _validate_state_documents(schema, {name: read_json(resolved[name]) for name in state_paths})
    _sha(descriptor["aggregate_content_sha256"], "aggregate_content_sha256")
    _exact(
        descriptor["forecast_population_sha256"],
        schema["forecast_population_sha256"],
        "descriptor.forecast_population_sha256",
    )
    receipt = read_json(receipt_path)
    _keys(receipt, ("artifact_kind", "schema_version", "descriptor"), "receipt")
    _exact(receipt["artifact_kind"], "racepv.prepared-receipt", "receipt.artifact_kind")
    _exact(receipt["schema_version"], 2, "receipt.schema_version")
    _file_identity(receipt["descriptor"])
    _exact(receipt["descriptor"]["path"], descriptor_path.name, "receipt.descriptor.path")
    return {
        "schema": schema,
        "descriptor": descriptor,
        "receipt": receipt,
        "artifact_root": str(root.resolve()),
        "descriptor_path": str(descriptor_path),
        "_resolved": resolved,
    }


def _verify_file(item, path, verified):
    key = (str(path), item["sha256"], item["bytes"])
    if key in verified:
        return
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    if size != item["bytes"] or digest.hexdigest() != item["sha256"]:
        raise ValueError(f"content identity mismatch: {item['path']}")
    verified.add(key)


def _verify_graph_content(metadata, verified):
    for item in metadata["descriptor"]["files"]:
        _verify_file(item, metadata["_resolved"][item["path"]], verified)
    _verify_file(metadata["receipt"]["descriptor"], Path(metadata["descriptor_path"]), verified)
    if (
        _aggregate_file_identities(metadata["descriptor"]["files"])
        != metadata["descriptor"]["aggregate_content_sha256"]
    ):
        raise ValueError("aggregate content identity mismatch")


def validate_v2_metadata(descriptor_path):
    """Metadata-first file identity validation. Never opens NumPy arrays."""
    metadata = _metadata_graph(descriptor_path, ancestors=())
    # The full recursively referenced graph is checked before the first file hash.
    verified = set()
    for node in _graph_nodes(metadata):
        _verify_graph_content(node, verified)
    for node in _graph_nodes(metadata):
        del node["_resolved"]
    return metadata


def _graph_nodes(metadata):
    yield metadata
    for child in metadata.get("children", ()):
        yield from _graph_nodes(child)


def _metadata_graph(descriptor_path, *, ancestors, read_json=None):
    read_json = read_strict_json if read_json is None else read_json
    path = Path(descriptor_path)
    path = contained_artifact_file(path.parent, path.name)
    if path in ancestors:
        raise ValueError("descriptor reference cycle")
    kind = read_json(path).get("artifact_kind")
    if kind == "racepv.prepared-descriptor":
        return _leaf_graph(path, read_json=read_json)
    if kind != "racepv.prepared-index":
        raise ValueError("unknown artifact kind; legacy fallback is forbidden")
    return _index_graph(path, ancestors=(*ancestors, path), read_json=read_json)


def _receipt_name(path):
    stem = (
        path.name.removesuffix(".descriptor.json")
        if path.name.endswith(".descriptor.json")
        else path.stem
    )
    return stem + ".receipt.json"


def _index_graph(path, *, ancestors, read_json=None):
    read_json = read_strict_json if read_json is None else read_json
    root = path.parent
    descriptor = read_json(path)
    _keys(
        descriptor,
        (
            "artifact_kind",
            "schema_version",
            "causality_profile",
            "format",
            "schema",
            "selected_split",
            "leaves",
            "files",
            "aggregate_content_sha256",
            "forecast_population_sha256",
        ),
        "index",
    )
    _exact(descriptor["schema_version"], 2, "index.schema_version")
    _exact(descriptor["causality_profile"], PROFILE, "index.causality_profile")
    fmt = descriptor["format"]
    if type(fmt) is not str or fmt not in {
        V2_WINDOW_SEQUENCE_STORE_FORMAT,
        V2_WINDOW_SHARD_INDEX_FORMAT,
    }:
        raise ValueError("unsupported index format")
    if fmt == V2_WINDOW_SEQUENCE_STORE_FORMAT:
        if descriptor["selected_split"] is not None:
            raise ValueError("sequence index must expose all four split views")
    elif (
        type(descriptor["selected_split"]) is not str or descriptor["selected_split"] not in SPLITS
    ):
        raise ValueError("shard index must select one registered split")
    if type(descriptor["files"]) is not list or not descriptor["files"]:
        raise ValueError("index files must be a nonempty list")
    files, resolved, physical, lexical = {}, {}, set(), set()
    for item in descriptor["files"]:
        _file_identity(item)
        name = item["path"]
        full = contained_artifact_file(root, name)
        inode = (full.stat().st_dev, full.stat().st_ino)
        if name.casefold() in lexical or inode in physical:
            raise ValueError("index file aliases or duplicate logical paths")
        lexical.add(name.casefold())
        physical.add(inode)
        files[name], resolved[name] = item, full
    if type(descriptor["leaves"]) is not list or not descriptor["leaves"]:
        raise ValueError("index leaves must be nonempty")
    children, views = [], []
    expected_files = {}
    child_paths = set()
    role_paths = {}
    for entry in descriptor["leaves"]:
        _keys(
            entry,
            (
                "descriptor",
                "site_id",
                "split",
                "aggregate_content_sha256",
                "forecast_population_sha256",
            ),
            "index leaf",
        )
        _file_identity(entry["descriptor"])
        _integer(entry["site_id"], "leaf.site_id")
        if type(entry["split"]) is not str or entry["split"] not in SPLITS:
            raise ValueError("leaf must select a registered split")
        if fmt == V2_WINDOW_SHARD_INDEX_FORMAT and entry["split"] != descriptor["selected_split"]:
            raise ValueError("shard leaf selected split mismatch")
        ref = entry["descriptor"]
        if files.get(ref["path"]) != ref or ref["path"] in child_paths:
            raise ValueError("leaf descriptor pin missing, conflicting, or duplicated")
        child_paths.add(ref["path"])
        child = _metadata_graph(resolved[ref["path"]], ancestors=ancestors, read_json=read_json)
        if fmt == V2_WINDOW_SEQUENCE_STORE_FORMAT and "children" in child:
            raise ValueError("sequence views must be leaf descriptors")
        if "children" in child and child["descriptor"]["format"] != V2_WINDOW_SEQUENCE_STORE_FORMAT:
            raise ValueError("shard children must be leaf or complete sequence descriptors")
        selected = [
            view
            for view in child.get("views", [child])
            if view["schema"]["split"] == entry["split"]
            and view["schema"]["site"]["id"] == entry["site_id"]
        ]
        if len(selected) != 1:
            raise ValueError("index leaf selection does not identify one site/split")
        view = selected[0]
        _exact(
            entry["aggregate_content_sha256"],
            child["descriptor"]["aggregate_content_sha256"],
            "leaf aggregate identity",
        )
        _exact(
            entry["forecast_population_sha256"],
            view["schema"]["forecast_population_sha256"],
            "leaf population identity",
        )
        for node in _graph_nodes(child):
            node_root = Path(node["artifact_root"])
            for item in node["descriptor"]["files"]:
                full = node_root / item["path"]
                relative = full.relative_to(root).as_posix()
                normalized = {**item, "path": relative}
                if relative in expected_files and expected_files[relative] != normalized:
                    raise ValueError("conflicting shared content identity")
                expected_files[relative] = normalized
            desc_full = Path(node["descriptor_path"])
            desc_ref = {
                **node["receipt"]["descriptor"],
                "path": desc_full.relative_to(root).as_posix(),
            }
            expected_files[desc_ref["path"]] = desc_ref
            receipt_full = desc_full.with_name(_receipt_name(desc_full))
            receipt_relative = receipt_full.relative_to(root).as_posix()
            # Outer index pins receipt bytes; its schema was validated recursively.
            if receipt_relative not in files:
                raise ValueError("index must pin every child publication receipt")
            expected_files[receipt_relative] = files[receipt_relative]
            if "children" not in node:
                roles = {node["descriptor"]["schema"]: "schema"}
                for name, arr_ref in node["descriptor"]["arrays"].items():
                    if "resolver" in arr_ref:
                        # Inputs name physical logical roles in this same leaf; they
                        # join the ordinary collection role map below, without aliases.
                        continue
                    role = "npz" if arr_ref["member"] is not None else "array:" + name
                    roles[arr_ref["path"]] = role
                for name, item in node["schema"]["sources"].items():
                    roles[item["path"]] = "source:" + name
                for name, fit in node["schema"]["fit_states"].items():
                    roles[fit["file"]["path"]] = "fit:" + name
                    for kind in ("fit", "training_audit"):
                        if fit[kind] is not None:
                            roles[fit[kind]["input_file"]["path"]] = f"{kind}_input:{name}"
                for name, role in roles.items():
                    full = str(node_root / name)
                    if full in role_paths and role_paths[full] != role:
                        raise ValueError("shared file has conflicting logical roles")
                    role_paths[full] = role
        children.append(child)
        views.append(view)
    if expected_files != files:
        raise ValueError("index must pin exactly the complete descendant file graph")
    if fmt == V2_WINDOW_SEQUENCE_STORE_FORMAT:
        if [view["schema"]["split"] for view in views] != list(SPLITS):
            raise ValueError("sequence requires ordered four nonempty split views")
        if len({view["schema"]["site"]["id"] for view in views}) != 1:
            raise ValueError("sequence split views must share one site")
        static_roles = POPULATION_ARRAYS[:13]
        for view in views[1:]:
            for role in static_roles:
                _exact(
                    view["schema"]["arrays"][role],
                    views[0]["schema"]["arrays"][role],
                    "sequence static source " + role,
                )
            for key in ("site", "sources", "fit_states"):
                _exact(view["schema"][key], views[0]["schema"][key], "sequence " + key)
    else:
        site_ids = [view["schema"]["site"]["id"] for view in views]
        if site_ids != sorted(set(site_ids)):
            raise ValueError("shard sites must be unique and canonically ordered")
    consistency = (
        "schema_kind",
        "schema_version",
        "causality_profile",
        "dataset_kind",
        "simulated",
        "history_columns",
        "future_known_columns",
        "site_metadata_names",
        "target",
        "state_labels",
        "geometry",
        "clock",
        "split_plan",
        "causal_rules",
    )
    all_views = [
        node for child in children for node in _graph_nodes(child) if "children" not in node
    ]
    for view in all_views:
        for field in consistency:
            _exact(view["schema"][field], views[0]["schema"][field], "collection " + field)
        left_source = {
            key: value
            for key, value in view["schema"]["source_time_contract"].items()
            if key != "evidence"
        }
        right_source = {
            key: value
            for key, value in views[0]["schema"]["source_time_contract"].items()
            if key != "evidence"
        }
        _exact(left_source, right_source, "collection source clock")
    schema_name = _relative_name(descriptor["schema"])
    if schema_name not in resolved:
        raise ValueError("index root schema must be pinned")
    schema = validate_v2_schema(read_json(resolved[schema_name]))
    _exact(schema, views[0]["schema"], "index representative schema")
    _sha(descriptor["aggregate_content_sha256"], "index aggregate_content_sha256")
    population = {
        "causality_profile": PROFILE,
        "format": fmt,
        "selected_split": descriptor["selected_split"],
        "leaves": [
            {key: entry[key] for key in ("site_id", "split", "forecast_population_sha256")}
            for entry in descriptor["leaves"]
        ],
    }
    _exact(
        descriptor["forecast_population_sha256"],
        canonical_json_sha256(population),
        "index forecast_population_sha256",
    )
    receipt_name = _receipt_name(path)
    receipt = read_json(contained_artifact_file(root, receipt_name))
    _keys(receipt, ("artifact_kind", "schema_version", "descriptor"), "index receipt")
    _exact(receipt["artifact_kind"], "racepv.prepared-receipt", "receipt.artifact_kind")
    _exact(receipt["schema_version"], 2, "receipt.schema_version")
    _file_identity(receipt["descriptor"])
    _exact(receipt["descriptor"]["path"], path.name, "receipt.descriptor.path")
    return {
        "schema": schema,
        "descriptor": descriptor,
        "receipt": receipt,
        "artifact_root": str(root),
        "descriptor_path": str(path),
        "_resolved": resolved,
        "children": children,
        "views": views,
    }


def _array_identity(array):
    if type(array) is ResolverModelArray:
        digest = hashlib.sha256()
        for start in range(0, len(array), 64):
            digest.update(array.read_batch(np.arange(start, min(start + 64, len(array)))).tobytes())
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": digest.hexdigest(),
            "bytes": array.nbytes,
        }
    return _bounded_array_identity(array)


def resolver_reference(name):
    """Exact virtual role schema, bound to this leaf's complete physical inputs."""
    if name not in MODEL_ARRAYS:
        raise ValueError("unsupported virtual model role")
    return {"resolver": PROFILE, "role": name, "inputs": list(POPULATION_ARRAYS[:16])}


class ResolverModelArray:
    """Bounded logical model tensor over the accepted causal Dataset resolver."""

    def __init__(self, dataset, name, spec):
        self.dataset, self.name = dataset, name
        self.shape = tuple(spec["shape"])
        self.dtype = np.dtype(spec["dtype"])
        self.ndim = len(self.shape)
        self.nbytes = math.prod(self.shape) * self.dtype.itemsize

    def __len__(self):
        return self.shape[0]

    def read_batch(self, indices):
        return self.dataset.read_model_window_batch(indices, names=(self.name,))[self.name]

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_)):
            return self.read_batch([index])[0]
        raise TypeError("use read_model_window_batch for bounded virtual selection")

    def __array__(self, dtype=None, copy=None):
        raise TypeError("full virtual model materialization is forbidden; use bounded batches")


def virtual_model_arrays(schema, causal_arrays):
    """Attach resolver roles without materializing any full model tensor."""
    schema = validate_v2_schema(schema)
    _keys(causal_arrays, set(ARRAY_ROLES) - set(MODEL_ARRAYS), "physical sequence arrays")
    dataset = _reconstruct(schema, causal_arrays)
    return {
        **causal_arrays,
        **{
            name: ResolverModelArray(dataset, name, schema["arrays"][name]) for name in MODEL_ARRAYS
        },
    }


def _same(actual, expected, name):
    if (
        actual.shape != expected.shape
        or actual.dtype != expected.dtype
        or not np.array_equal(actual, expected, equal_nan=True)
    ):
        raise ValueError(f"semantic mismatch: {name}")


def _validate_causal_array_identities(schema, causal_arrays, *, allow_resolver_model_arrays):
    _keys(causal_arrays, ARRAY_ROLES, "causal_arrays")
    for name, array in causal_arrays.items():
        if not isinstance(array, np.ndarray) and type(array) is not ResolverModelArray:
            raise TypeError(f"{name} must be a non-object numpy array")
        if array.dtype.hasobject or (
            type(array) is ResolverModelArray
            and (name not in MODEL_ARRAYS or not allow_resolver_model_arrays)
        ):
            raise TypeError(f"{name} cannot use this array storage form")
        if _array_identity(array) != schema["arrays"][name]:
            raise ValueError(f"array identity mismatch: {name}")


def _validate_site_and_weather_provenance(schema, causal_arrays):
    _same(
        causal_arrays["site_vector"],
        np.asarray(schema["fit_states"]["site_vector"]["state"]["values"], dtype=np.float32),
        "fixed site_vector state",
    )
    weather_columns = schema["fit_states"]["weather_standardization"]["state"]["feature_columns"]
    for i, column in enumerate(("ghi", "temperature", "humidity", "wind_speed")):
        if column not in weather_columns and np.any(causal_arrays["weather_observed"][:, i]):
            raise ValueError("semantic absent weather cannot carry observed provenance")


def _dataset_from_causal_state(schema, arrays, state, *, selected_bounds):
    """Build the accepted causal Dataset over already validated global or local bounds."""
    import pandas as pd

    from racepv.data.dataset import WindowedPVDataset

    _same(arrays["gap_left"], state.gap_left, "gap_left endpoints")
    _same(arrays["gap_right"], state.gap_right, "gap_right endpoints")
    _same(
        arrays["effective_power_observed"],
        state._effective_power_observed,
        "effective power provenance",
    )
    raw = arrays["base_values_raw"]
    if not np.array_equal(raw[:, 0], arrays["row_targets"], equal_nan=True):
        raise ValueError("semantic target source disagrees with raw power")
    if not np.array_equal(raw[:, 1], arrays["clear_sky_norm"], equal_nan=True):
        raise ValueError("semantic clear-sky source mismatch")
    if not np.array_equal(arrays["row_future_known"][:, 0], arrays["clear_sky_norm"]):
        raise ValueError("semantic future clear-sky source mismatch")
    frame = pd.DataFrame(
        raw.copy(),
        columns=REGISTERED_HISTORY_FEATURES,
        index=pd.to_datetime(state.labels_ns, utc=True),
    )
    for i, name in enumerate(REGISTERED_FUTURE_FEATURES[:-1]):
        frame[name] = arrays["row_future_known"][:, i]
    frame["state_label"] = arrays["row_state_labels"]
    frame["is_night"] = state.clear_sky_norm < 0.01
    history_mask = np.isfinite(raw)
    history_mask[:, 0] = arrays["power_observed"]
    history_mask[:, 10:] = arrays["weather_observed"]
    with np.errstate(over="ignore", invalid="ignore"):
        dataset = WindowedPVDataset(
            frame,
            history_columns=REGISTERED_HISTORY_FEATURES,
            future_known_columns=REGISTERED_FUTURE_FEATURES,
            site_metadata=dict(zip(REGISTERED_SITE_FEATURES, arrays["site_vector"], strict=True)),
            history_mask=pd.DataFrame(
                history_mask, index=frame.index, columns=REGISTERED_HISTORY_FEATURES
            ),
            target_observed_mask=arrays["power_observed"],
            lookback=192,
            horizon=24,
            stride=schema["geometry"]["stride"],
            split=selected_bounds,
            require_observed_targets=True,
            causal_state=state,
        )
    return dataset


def _reconstruct(schema, arrays):
    """Reuse the accepted causal and target authorities without entering the formal guard."""
    state = CausalHistoryState.from_arrays(
        labels_ns=arrays["labels_ns"],
        split_ids=arrays["split_ids"],
        power_observed=arrays["power_observed"],
        weather_observed=arrays["weather_observed"],
        clear_sky_norm=arrays["clear_sky_norm"],
        base_values=arrays["base_values_raw"],
        feature_names=REGISTERED_HISTORY_FEATURES,
    )
    plan = freeze_split_plan(
        state.labels_ns,
        {
            name: slice(
                schema["split_plan"]["splits"][name]["start_row"],
                schema["split_plan"]["splits"][name]["stop_row"],
            )
            for name in SPLITS
        },
    )
    _exact(plan, schema["split_plan"], "semantic split plan")
    for split_id, name in enumerate(SPLITS):
        bounds = plan["splits"][name]
        if not np.all(state.split_ids[bounds["start_row"] : bounds["stop_row"]] == split_id):
            raise ValueError("semantic split IDs disagree with frozen partitions")
    selected = plan["splits"][schema["split"]]
    return _dataset_from_causal_state(
        schema,
        arrays,
        state,
        selected_bounds=slice(selected["start_row"], selected["stop_row"]),
    )


def _verify_reconstructed_arrays(schema, causal_arrays, dataset):
    """Compare every stored population, model tensor and UTC audit with causal truth."""
    for name in ("candidate_starts", "history_candidate_starts", "window_starts"):
        _same(causal_arrays[name], getattr(dataset, name), name + " population")
    for start in range(0, len(dataset), 64):
        positions = np.arange(start, min(start + 64, len(dataset)), dtype=np.int64)
        model = dataset.read_model_window_batch(positions)
        for name in MODEL_ARRAYS:
            array = causal_arrays[name]
            actual = (
                array.read_batch(positions)
                if type(array) is ResolverModelArray
                else array[positions]
            )
            if not np.isfinite(actual).all():
                raise ValueError(f"semantic model array must be finite: {name}")
            _same(actual, model[name], name)
    times = origin_times(
        causal_arrays["labels_ns"], dataset.window_starts, lookback=192, horizon=24
    )
    rows = dataset.window_starts[:, None] + 192 + np.arange(24, dtype=np.int64)
    audits = {
        "issue_ns": times.issue_ns,
        "timestamps": times.target_start_ns,
        "day_ids": times.target_start_ns // 86_400_000_000_000,
        "site_ids": np.full(len(dataset), schema["site"]["id"], dtype=np.int64),
        "daylight_mask": causal_arrays["clear_sky_norm"][rows] >= 0.01,
        "target_observed_mask": causal_arrays["power_observed"][rows],
    }
    for name, expected in audits.items():
        _same(causal_arrays[name], expected, name)


def verify_v2_semantics(metadata, causal_arrays):
    """Recheck logical byte identity and causal truth after array loading.

    Accept a full schema or validate_v2_metadata result. This function proves array
    semantics, not source-document interpretation or actual online arrival latency.
    """
    if type(metadata) is dict and "descriptor" in metadata:
        # Returned dictionaries are independent, but intentionally not trusted capabilities.
        current = validate_v2_metadata(metadata["descriptor_path"])
        _exact(metadata, current, "published metadata identity")
        if "views" in current:
            views = {
                f"{view['schema']['site']['id']}/{view['schema']['split']}": view
                for view in current["views"]
            }
            _keys(causal_arrays, views, "selected collection causal arrays")
            for key, view in views.items():
                verify_v2_semantics(view["schema"], causal_arrays[key])
            return
    schema = metadata["schema"] if type(metadata) is dict and "schema" in metadata else metadata
    schema = validate_v2_schema(schema)
    _validate_causal_array_identities(schema, causal_arrays, allow_resolver_model_arrays=True)
    _validate_site_and_weather_provenance(schema, causal_arrays)
    dataset = _reconstruct(schema, causal_arrays)
    _verify_reconstructed_arrays(schema, causal_arrays, dataset)


def validate_preparation_contract(schema, causal_arrays, *, evidence):
    """In-memory seam: full schema + logical arrays + actual referenced evidence bytes.

    The producer need not serialize an artifact to validate its preparation inputs.
    Evidence is a strict path->bytes dict; no eligibility flags or bypasses exist.
    """
    schema = validate_v2_schema(schema)
    items = _schema_evidence(schema)
    _keys(evidence, [item["path"] for item in items], "preparation evidence")
    for item in items:
        if type(evidence[item["path"]]) is not bytes:
            raise TypeError("preparation evidence must contain bytes")
        _verify_bytes(item, evidence[item["path"]])
    state_paths = [
        schema["sources"]["site_metadata"]["path"],
        *(record["file"]["path"] for record in schema["fit_states"].values()),
    ]
    _validate_state_documents(
        schema, {name: _strict_json_value(evidence[name].decode("utf-8")) for name in state_paths}
    )
    verify_v2_semantics(schema, causal_arrays)
    return _identity(schema, semantics=True, content=False)


def require_origin_safe_history_contract(
    dataset_or_descriptor, *, version=2, profile="as-of-origin-v1"
):
    """Explicit expectation guard; never equates metadata identity with loaded semantics."""
    _exact(version, 2, "expected version")
    _exact(profile, PROFILE, "expected profile")
    if isinstance(dataset_or_descriptor, (str, Path)):
        metadata = validate_v2_metadata(dataset_or_descriptor)
        identity = _identity(metadata["schema"], semantics=False, content=True)
        identity.update(
            aggregate_content_sha256=metadata["descriptor"]["aggregate_content_sha256"],
            descriptor_sha256=metadata["receipt"]["descriptor"]["sha256"],
            forecast_population_sha256=metadata["descriptor"]["forecast_population_sha256"],
        )
        if "views" in metadata:
            identity["split"] = metadata["descriptor"]["selected_split"]
            identity["sites"] = [
                copy.deepcopy(view["schema"]["site"]) for view in metadata["views"]
            ]
            identity["populations"] = [
                {
                    "site_id": view["schema"]["site"]["id"],
                    "split": view["schema"]["split"],
                    "forecast_population_sha256": view["schema"]["forecast_population_sha256"],
                }
                for view in metadata["views"]
            ]
        return identity
    from racepv.data.dataset import WindowedPVDataset
    from racepv.io import (
        NpySequenceWindowDataset,
        NpyWindowDataset,
        NpzWindowDataset,
        ShardedWindowDataset,
    )

    if type(dataset_or_descriptor) in (
        NpzWindowDataset,
        NpyWindowDataset,
        NpySequenceWindowDataset,
        ShardedWindowDataset,
    ):
        from racepv.io import _verify_stored_v2

        return _verify_stored_v2(dataset_or_descriptor, version=version, profile=profile)

    if type(dataset_or_descriptor) is not WindowedPVDataset:
        raise TypeError(
            "guard requires a published descriptor path or a validated WindowedPVDataset"
        )
    dataset = dataset_or_descriptor
    _MEMORY_QUALIFICATIONS.pop(dataset, None)
    prepared = dataset.prepared_contract
    _keys(prepared, ("schema", "causal_arrays", "evidence"), "unvalidated prepared_contract")
    identity = validate_preparation_contract(
        prepared["schema"], prepared["causal_arrays"], evidence=prepared["evidence"]
    )
    reference = _reconstruct(prepared["schema"], prepared["causal_arrays"])
    if not dataset.frame.index.equals(reference.frame.index):
        raise ValueError("dataset timestamp audit index differs from the validated origin clock")
    for name in (
        "lookback",
        "horizon",
        "stride",
        "split_bounds",
        "history_columns",
        "future_known_columns",
        "site_metadata_names",
        "require_observed_targets",
    ):
        if getattr(dataset, name) != getattr(reference, name):
            raise ValueError(f"dataset semantic mismatch: {name}")
    if dataset.causal_state is None:
        raise ValueError("dataset has no causal state")
    for name in (
        "labels_ns",
        "split_ids",
        "power_observed",
        "weather_observed",
        "clear_sky_norm",
        "base_values",
        "gap_left",
        "gap_right",
        "_effective_power_observed",
    ):
        _same(
            getattr(dataset.causal_state, name),
            getattr(reference.causal_state, name),
            "dataset " + name,
        )
    for name in (
        "candidate_starts",
        "history_candidate_starts",
        "window_starts",
        "_targets",
        "_states",
        "_target_observed",
        "_base_future",
        "_night",
        "_site_metadata",
    ):
        _same(getattr(dataset, name), getattr(reference, name), "dataset " + name)
    for start in range(0, len(dataset), 64):
        positions = np.arange(start, min(start + 64, len(dataset)), dtype=np.int64)
        actual = dataset.read_model_window_batch(positions)
        for name in MODEL_ARRAYS:
            array = prepared["causal_arrays"][name]
            expected = (
                array.read_batch(positions)
                if type(array) is ResolverModelArray
                else array[positions]
            )
            _same(actual[name], expected, "dataset " + name)
    _register_memory_qualification(dataset)
    return identity


def _identity(schema, *, semantics, content):
    return {
        "schema_kind": SCHEMA_KIND,
        "schema_version": 2,
        "causality_profile": PROFILE,
        "schema_identity_sha256": canonical_json_sha256(schema),
        "forecast_population_sha256": schema["forecast_population_sha256"],
        "split": schema["split"],
        "site": copy.deepcopy(schema["site"]),
        "content_verified": content,
        "array_semantics_verified": semantics,
        "registered_evidence_eligible": False,
        "source_evidence_kind": schema["source_time_contract"]["evidence"]["kind"],
        "source_interpretation_verified": False,
    }
