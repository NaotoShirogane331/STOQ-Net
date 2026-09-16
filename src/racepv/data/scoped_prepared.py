"""Strict in-memory schema-3 reconstruction for one complete split."""

import copy

import numpy as np

from racepv.data import prepared_contract as contract
from racepv.data.causal_history import CausalHistoryState
from racepv.io import REGISTERED_HISTORY_FEATURES

ROW_SCOPE_FIELDS = (
    "kind",
    "split",
    "global_start_row",
    "global_stop_row",
    "stored_start_ns",
    "stored_end_exclusive_ns",
    "index_coordinates",
)


def row_scope_from_plan(split_plan: dict, *, selected_split: str) -> dict:
    """Validate the global plan and derive the exact complete selected split scope."""
    contract._validate_global_split_plan(split_plan, selected_split)
    bounds = split_plan["splits"][selected_split]
    return {
        "kind": "complete_single_split",
        "split": selected_split,
        "global_start_row": bounds["start_row"],
        "global_stop_row": bounds["stop_row"],
        "stored_start_ns": bounds["start_ns"],
        "stored_end_exclusive_ns": bounds["end_exclusive_ns"],
        "index_coordinates": "split_local_zero_based",
    }


def scoped_forecast_population_sha256(schema: dict) -> str:
    """Return schema-3 population identity; computing it performs no validation."""
    return contract.canonical_json_sha256(
        {
            "causality_profile": schema["causality_profile"],
            "clock": schema["clock"],
            "source_time_contract": schema["source_time_contract"],
            "split_plan": schema["split_plan"],
            "split": schema["split"],
            "geometry": schema["geometry"],
            "site": schema["site"],
            "arrays": {name: schema["arrays"][name] for name in contract.POPULATION_ARRAYS},
            "schema_version": 3,
            "row_scope": schema["row_scope"],
        }
    )


def validate_v3_schema(schema: dict) -> dict:
    """Strictly validate scoped metadata only and return a detached copy."""
    contract._keys(schema, (*contract.SCHEMA_FIELDS, "row_scope"), "schema3")
    contract._exact(schema["schema_kind"], contract.SCHEMA_KIND, "schema_kind")
    contract._exact(schema["schema_version"], 3, "schema_version")
    contract._exact(schema["causality_profile"], contract.PROFILE, "causality_profile")
    contract._validate_scientific_schema_fields(schema)
    scope = row_scope_from_plan(schema["split_plan"], selected_split=schema["split"])
    contract._keys(schema["row_scope"], ROW_SCOPE_FIELDS, "row_scope")
    contract._exact(schema["row_scope"], scope, "row_scope")
    row_count = scope["global_stop_row"] - scope["global_start_row"]
    contract._validate_array_schema_fields(
        schema,
        row_count=row_count,
        population_hasher=scoped_forecast_population_sha256,
    )
    return copy.deepcopy(schema)


def _validate_scoped_coordinates(schema, arrays):
    scope = schema["row_scope"]
    labels = arrays["labels_ns"]
    row_count = scope["global_stop_row"] - scope["global_start_row"]
    if labels.shape != (row_count,):
        raise ValueError("scoped labels shape disagrees with row_scope")
    if int(labels[0]) != scope["stored_start_ns"]:
        raise ValueError("scoped labels do not start at stored_start_ns")
    expected_last = scope["stored_start_ns"] + (row_count - 1) * contract.STEP_NS
    if int(labels[-1]) != expected_last:
        raise ValueError("scoped labels do not cover the complete selected partition")
    if expected_last + contract.STEP_NS != scope["stored_end_exclusive_ns"]:
        raise ValueError("scoped labels disagree with stored_end_exclusive_ns")
    split_id = contract.SPLITS.index(schema["split"])
    if not np.all(arrays["split_ids"] == split_id):
        raise ValueError("scoped split IDs disagree with selected partition")


def build_scoped_window_dataset(schema: dict, causal_arrays: dict):
    """Validate local identities and causal truth, then return the reconstructed Dataset."""
    schema = validate_v3_schema(schema)
    contract._validate_causal_array_identities(
        schema,
        causal_arrays,
        allow_resolver_model_arrays=False,
    )
    contract._validate_site_and_weather_provenance(schema, causal_arrays)
    _validate_scoped_coordinates(schema, causal_arrays)
    state = CausalHistoryState.from_arrays(
        labels_ns=causal_arrays["labels_ns"],
        split_ids=causal_arrays["split_ids"],
        power_observed=causal_arrays["power_observed"],
        weather_observed=causal_arrays["weather_observed"],
        clear_sky_norm=causal_arrays["clear_sky_norm"],
        base_values=causal_arrays["base_values_raw"],
        feature_names=REGISTERED_HISTORY_FEATURES,
    )
    dataset = contract._dataset_from_causal_state(
        schema,
        causal_arrays,
        state,
        selected_bounds=slice(0, len(state.labels_ns)),
    )
    contract._verify_reconstructed_arrays(schema, causal_arrays, dataset)
    return dataset


def verify_v3_semantics(schema: dict, causal_arrays: dict) -> None:
    """Fully validate scoped in-memory arrays without granting qualification."""
    build_scoped_window_dataset(schema, causal_arrays)
