"""Schema-3 metadata projection, not source or content qualification.

The caller supplies the complete graph returned by inspect_scoped_store and its
trusted snapshot hash. This module neither opens files nor certifies that graph's
provenance. Formal consumers must independently verify the store and binding.
"""

import hashlib
import json
from dataclasses import dataclass

from racepv.data import prepared_contract as contract
from racepv.data.scientific_view import _COMMON
from racepv.data.scoped_prepared import validate_v3_schema


def _bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


@dataclass(frozen=True)
class ScopedScientificView:
    """Immutable JSON snapshot; properties return detached derived values."""

    input_metadata_json: bytes
    projection_json: bytes

    @property
    def metadata_sha256(self):
        return hashlib.sha256(self.input_metadata_json).hexdigest()

    @property
    def identity_sha256(self):
        return contract.canonical_json_sha256(
            {
                "kind": "racepv.scoped-scientific-view",
                "version": 1,
                "metadata_sha256": self.metadata_sha256,
                "projection": json.loads(self.projection_json),
            }
        )

    @property
    def site_id_map(self):
        return json.loads(self.projection_json)["site_id_map"]

    @property
    def spatial_roles(self):
        return json.loads(self.projection_json)["spatial_roles"]

    @property
    def ramp_method(self):
        return json.loads(self.projection_json)["ramp_method"]

    @property
    def ramp_thresholds(self):
        return json.loads(self.projection_json)["ramp_thresholds"]

    @property
    def feature_signature(self):
        return json.loads(self.projection_json)["feature_signature"]

    @property
    def content_verified(self):
        return False

    @property
    def array_semantics_verified(self):
        return False

    @property
    def source_interpretation_verified(self):
        return False

    @property
    def registered_evidence_eligible(self):
        return False


def _feature_contract(schema):
    return {
        **{name: schema[name] for name in _COMMON},
        "ramp_method": schema["fit_states"]["ramp_label_threshold"]["method"],
        "source_clock": {
            name: value
            for name, value in schema["source_time_contract"].items()
            if name != "evidence"
        },
    }


def scientific_view_from_metadata(metadata, *, selected_split, expected_metadata_sha256):
    """Derive every selected leaf, preserving the entire pinned input snapshot."""
    contract._sha(expected_metadata_sha256, "expected metadata sha256")
    if type(metadata) is not dict:
        raise TypeError("metadata must be a dict")
    blob = _bytes(metadata)
    if hashlib.sha256(blob).hexdigest() != expected_metadata_sha256:
        raise ValueError("metadata identity drift")
    graph = json.loads(blob)
    if type(selected_split) is not str or selected_split not in ("train", "validation"):
        raise ValueError("scientific view supports train/validation only")
    if graph.get("selected_split") != selected_split:
        raise ValueError("metadata selected split mismatch")
    leaves = graph.get("leaves")
    if type(leaves) is not list or not leaves:
        raise ValueError("metadata leaves must be nonempty")
    schemas = []
    for leaf in leaves:
        if type(leaf) is not dict or "schema" not in leaf:
            raise ValueError("metadata leaf schema missing")
        schema = validate_v3_schema(leaf["schema"])
        if schema["split"] != selected_split:
            raise ValueError("leaf split mismatch")
        schemas.append(schema)
    schemas.sort(key=lambda item: item["site"]["id"])
    ids = [item["site"]["id"] for item in schemas]
    names = [item["site"]["name"] for item in schemas]
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise ValueError("site IDs and names must be unique")
    feature = _feature_contract(schemas[0])
    plan = schemas[0]["split_plan"]
    for schema in schemas[1:]:
        if _bytes(_feature_contract(schema)) != _bytes(feature) or _bytes(
            schema["split_plan"]
        ) != _bytes(plan):
            raise ValueError("inconsistent scientific contracts")
    keyed = {str(item["site"]["id"]): item for item in schemas}
    projection = {
        "selected_split": selected_split,
        "site_id_map": {key: item["site"]["name"] for key, item in keyed.items()},
        "spatial_roles": {key: item["site"]["spatial_role"] for key, item in keyed.items()},
        "ramp_method": feature["ramp_method"],
        "ramp_thresholds": {
            key: item["fit_states"]["ramp_label_threshold"]["state"]["threshold"]
            for key, item in keyed.items()
        },
        "feature_signature": feature,
        "split_plan": plan,
        "schemas": keyed,
    }
    return ScopedScientificView(blob, _bytes(projection))


def _revalidate(view):
    if type(view) is not ScopedScientificView:
        raise TypeError("expected ScopedScientificView")
    graph = json.loads(view.input_metadata_json)
    actual = scientific_view_from_metadata(
        graph, selected_split=graph["selected_split"], expected_metadata_sha256=view.metadata_sha256
    )
    if actual != view:
        raise ValueError("scientific projection drift")
    return json.loads(actual.projection_json)


def require_train_validation_compatibility(train, validation):
    """Check frozen per-site contracts exactly; never grants formal eligibility."""
    left, right = _revalidate(train), _revalidate(validation)
    if left["selected_split"] != "train" or right["selected_split"] != "validation":
        raise ValueError("expected train then validation")
    for field in ("site_id_map", "spatial_roles", "feature_signature", "split_plan"):
        if _bytes(left[field]) != _bytes(right[field]):
            raise ValueError(f"cross-split {field} drift")
    for site_id, first in left["schemas"].items():
        second = right["schemas"][site_id]
        for field in ("fit_states", "source_time_contract"):
            if _bytes(first[field]) != _bytes(second[field]):
                raise ValueError(f"cross-split site {site_id} {field} drift")
        for role in ("producer", "time_evidence", "site_metadata"):
            if _bytes(first["sources"][role]) != _bytes(second["sources"][role]):
                raise ValueError(f"cross-split site {site_id} {role} drift")
        train_input = first["sources"]["raw_input"]
        frozen_input = second["sources"].get("frozen_fit_raw_input")
        # Each store validates its own contained path. The pipeline deliberately
        # names these selected.csv and train.csv; compare exact byte identity.
        if frozen_input is None or any(
            train_input[key] != frozen_input[key] for key in ("sha256", "bytes")
        ):
            raise ValueError(f"cross-split site {site_id} frozen train input drift")
