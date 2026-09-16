"""Detached v2 scientific metadata; deliberately grants no evidence eligibility."""

import copy

from racepv.data.prepared_contract import (
    SPLITS,
    canonical_json_sha256,
    validate_v2_schema,
)

_COMMON = (
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
    "causal_rules",
)


def build_v2_scientific_view(schemas, *, selected_split):
    """Project supplied leaf schemas; graph completeness is the caller's obligation."""
    if type(selected_split) is not str or selected_split not in SPLITS:
        raise ValueError("selected_split must be a registered split")
    if type(schemas) not in (list, tuple) or not schemas:
        raise ValueError("schemas must be a nonempty list or tuple")
    leaves = copy.deepcopy(schemas)
    for schema in leaves:
        validate_v2_schema(schema)
        if schema["split"] != selected_split:
            raise ValueError("leaf split differs from selected split")
    leaves = sorted(leaves, key=lambda item: item["site"]["id"])
    ids = [item["site"]["id"] for item in leaves]
    names = [item["site"]["name"] for item in leaves]
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise ValueError("site IDs and names must each be unique")
    first = leaves[0]
    signature = {key: first[key] for key in _COMMON}
    method = first["fit_states"]["ramp_label_threshold"]["method"]
    source_clock = {
        key: value for key, value in first["source_time_contract"].items() if key != "evidence"
    }
    for schema in leaves[1:]:
        other = {key: schema[key] for key in _COMMON}
        clock = {
            key: value for key, value in schema["source_time_contract"].items() if key != "evidence"
        }
        if (
            canonical_json_sha256(other) != canonical_json_sha256(signature)
            or canonical_json_sha256(schema["split_plan"])
            != canonical_json_sha256(first["split_plan"])
            or canonical_json_sha256(clock) != canonical_json_sha256(source_clock)
            or schema["fit_states"]["ramp_label_threshold"]["method"] != method
        ):
            raise ValueError("selected leaves have inconsistent scientific contracts")
    signature["ramp_method"] = method
    signature["source_clock"] = source_clock
    keyed = {str(item["site"]["id"]): item for item in leaves}
    return copy.deepcopy(
        {
            "artifact_kind": "racepv.prepared-scientific-view",
            "schema_version": 1,
            "selected_split": selected_split,
            "site_id_map": {key: item["site"]["name"] for key, item in keyed.items()},
            "spatial_roles": {key: item["site"]["spatial_role"] for key, item in keyed.items()},
            "ramp_method": method,
            "ramp_thresholds": {
                key: item["fit_states"]["ramp_label_threshold"]["state"]
                for key, item in keyed.items()
            },
            "fit_states": {key: item["fit_states"] for key, item in keyed.items()},
            "sources": {key: item["sources"] for key, item in keyed.items()},
            "schema_identities": {key: canonical_json_sha256(item) for key, item in keyed.items()},
            "population_identities": {
                key: item["forecast_population_sha256"] for key, item in keyed.items()
            },
            "feature_signature": signature,
            "split_plan": first["split_plan"],
            "content_verified": False,
            "array_semantics_verified": False,
            "source_interpretation_verified": False,
            "registered_evidence_eligible": False,
        }
    )
