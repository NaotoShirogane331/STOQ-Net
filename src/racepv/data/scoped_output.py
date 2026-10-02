"""Portable schema3 export/recovery identities, never source qualification.

Serialization validation is not authentication. A frozen external protocol and
fresh scoped loading remain required; a dictionary cannot grant formal admission.
"""

import copy

from racepv import io as window_io
from racepv.data import prepared_contract as c
from racepv.data import scoped_consumer, scoped_store
from racepv.data.scoped_scientific_view import (
    require_train_validation_compatibility,
    scientific_view_from_metadata,
)

_PINS = (
    "descriptor_sha256",
    "read_scope_sha256",
    "aggregate_content_sha256",
    "forecast_population_sha256",
)
_FLAGS = (
    "content_verified",
    "array_semantics_verified",
    "source_interpretation_verified",
    "registered_evidence_eligible",
)


def _view(metadata):
    return scientific_view_from_metadata(
        metadata,
        selected_split=metadata["selected_split"],
        expected_metadata_sha256=c.canonical_json_sha256(metadata),
    )


def _derive(metadata, storage_format):
    c._keys(
        metadata, (*_PINS, *_FLAGS, "selected_split", "leaves", "files"), "portable scoped metadata"
    )
    for name in _PINS:
        c._sha(metadata[name], name)
    for name in _FLAGS:
        c._exact(metadata[name], name in ("content_verified", "array_semantics_verified"), name)
    split = metadata["selected_split"]
    if type(split) is not str or split not in ("train", "validation"):
        raise ValueError("scoped output supports train/validation only")
    if storage_format not in (scoped_store.LEAF_FORMAT, scoped_store.INDEX_FORMAT):
        raise ValueError("explicit scoped storage format required")
    leaves = metadata["leaves"]
    if type(leaves) is not list or not leaves:
        raise ValueError("nonempty scoped leaf identities required")
    for leaf in leaves:
        c._keys(leaf, (*_PINS, "schema"), "portable leaf identity")
        for name in _PINS:
            c._sha(leaf[name], name)
        c._exact(
            leaf["forecast_population_sha256"],
            leaf["schema"]["forecast_population_sha256"],
            "leaf population",
        )
    view = _view(metadata)  # Reuses strict v3 schemas and all-site compatibility.
    schemas = [leaf["schema"] for leaf in leaves]
    ids = [schema["site"]["id"] for schema in schemas]
    if ids != sorted(set(ids)):
        raise ValueError("leaf sites must be unique and sorted")
    if storage_format == scoped_store.LEAF_FORMAT:
        if len(leaves) != 1:
            raise ValueError("single-leaf format requires exactly one site")
        for name in _PINS:
            c._exact(metadata[name], leaves[0][name], "leaf/root " + name)
    else:
        population = dict(
            schema_version=3,
            causality_profile=c.PROFILE,
            format=storage_format,
            selected_split=split,
            leaves=[
                dict(
                    site_id=schema["site"]["id"],
                    split=split,
                    forecast_population_sha256=schema["forecast_population_sha256"],
                )
                for schema in schemas
            ],
        )
        c._exact(
            metadata["forecast_population_sha256"],
            c.canonical_json_sha256(population),
            "index population",
        )
    files = metadata["files"]
    if type(files) is not list or not files:
        raise ValueError("complete scoped file identities required")
    for item in files:
        c._keys(item, ("path", "bytes", "sha256", "role", "splits"), "portable file scope")
    identities = scoped_store._file_list(
        [{k: item[k] for k in ("path", "bytes", "sha256")} for item in files], "export files"
    )
    scopes = scoped_store._scope_list(
        [{k: item[k] for k in ("path", "role", "splits")} for item in files],
        identities,
        "export scopes",
    )
    scoped_store._authorize(scopes, ("train",) if split == "train" else ("train", "validation"))
    c._exact(
        metadata["aggregate_content_sha256"],
        window_io._aggregate_file_identities(identities),
        "export aggregate",
    )
    c._exact(
        metadata["read_scope_sha256"],
        scoped_store._read_scope_sha256(identities, scopes),
        "export read scope",
    )
    domains = {schema["source_time_contract"]["evidence"]["kind"] for schema in schemas}
    if len(domains) != 1:
        raise ValueError("mixed source domains")
    # Validate every schema through the view, but retain all schemas, not its first site.
    return dict(
        schema_kind="racepv.prepared-window-contract",
        schema_version=3,
        causality_profile=c.PROFILE,
        split=split,
        storage_format=storage_format,
        **{name: metadata[name] for name in (*_PINS, *_FLAGS)},
        schema_identity_sha256=c.canonical_json_sha256(schemas),
        scoped_metadata=copy.deepcopy(metadata),
        source_evidence_kind=next(iter(domains)),
        clock=copy.deepcopy(schemas[0]["clock"]),
        mask_semantics={
            name: schemas[0]["causal_rules"][name]
            for name in ("missing_values", "imputed_values", "provenance_flag_mask")
        },
        site_id_map=view.site_id_map,
    )


def scoped_output_identity(metadata):
    """Project actual loaded metadata; metadata-only success is insufficient.

    This does not verify arrays itself. Only the loading entry below is an I/O
    boundary; neither serialized flags nor this projection authorize training.
    """
    c._keys(
        metadata,
        (*_PINS, *_FLAGS, "selected_split", "leaves", "files", "descriptor_path"),
        "loaded scoped metadata",
    )
    value = copy.deepcopy(metadata)
    root_path = value.pop("descriptor_path")
    leaves = value["leaves"]
    if type(leaves) is not list or not leaves:
        raise ValueError("nonempty loaded leaves required")
    leaf_format = len(leaves) == 1 and leaves[0]["descriptor_path"] == root_path
    for leaf in leaves:
        c._keys(leaf, (*_PINS, "schema", "arrays", "descriptor_path"), "loaded scoped leaf")
        del leaf["descriptor_path"], leaf["arrays"]
    for item in value["files"]:
        c._keys(item, ("path", "bytes", "sha256", "role", "splits", "resolved_path"), "loaded file")
        del item["resolved_path"]
    return _derive(value, scoped_store.LEAF_FORMAT if leaf_format else scoped_store.INDEX_FORMAT)


def validate_scoped_output_identity(value):
    """Check exact serialized consistency; never trust this as source admission."""
    c._keys(
        value,
        (
            "schema_kind",
            "schema_version",
            "causality_profile",
            "split",
            "storage_format",
            *_PINS,
            *_FLAGS,
            "schema_identity_sha256",
            "scoped_metadata",
            "source_evidence_kind",
            "clock",
            "mask_semantics",
            "site_id_map",
        ),
        "scoped output identity",
    )
    derived = _derive(value["scoped_metadata"], value["storage_format"])
    c._exact(value, derived, "scoped output identity consistency")
    return copy.deepcopy(value)


def require_scoped_output_pair(contracts):
    c._keys(contracts, ("train", "validation"), "scoped output pair")
    checked = {split: validate_scoped_output_identity(value) for split, value in contracts.items()}
    for split, value in checked.items():
        c._exact(value["split"], split, "output split role")
    require_train_validation_compatibility(
        _view(checked["train"]["scoped_metadata"]), _view(checked["validation"]["scoped_metadata"])
    )


def _identities(metadata):
    contracts = {split: scoped_output_identity(value) for split, value in metadata.items()}
    require_scoped_output_pair(contracts)
    return dict(
        descriptor_hashes={split: value["descriptor_sha256"] for split, value in contracts.items()},
        content_hashes={
            split: value["aggregate_content_sha256"] for split, value in contracts.items()
        },
        prepared_contracts=contracts,
    )


def load_scoped_output_identities(args, *, expected_preflight_sha256):
    """Reconstruct verified train/validation bytes, without fit or training."""
    loaded = scoped_consumer.load_scoped_engineering_inputs(
        args, expected_preflight_sha256=expected_preflight_sha256
    )
    return _identities({split: store["metadata"] for split, store in loaded["stores"].items()})


def reverify_scoped_output_identities(args, expected_identities, *, expected_preflight_sha256):
    """Resume/export recheck: metadata drift fails before content; bytes follow.

    The external caller pins expected_identities in its frozen run protocol.
    This method does not accept a checkpoint, run a fit, or authorize execution.
    """
    c._keys(
        expected_identities,
        ("descriptor_hashes", "content_hashes", "prepared_contracts"),
        "expected output identities",
    )
    require_scoped_output_pair(expected_identities["prepared_contracts"])
    report = scoped_consumer.inspect_scoped_consumer_inputs(args)
    c._sha(expected_preflight_sha256, "expected preflight SHA256")
    c._exact(c.canonical_json_sha256(report), expected_preflight_sha256, "current preflight")
    # Compare declarations with the frozen identities before opening any arrays.
    # Temporary True flags are used only to canonicalize the *expected structure*;
    # no verified result is returned until the real full loader succeeds below.
    predicted = {}
    for split, item in report["inputs"].items():
        snapshot = copy.deepcopy(item["metadata"])
        snapshot["content_verified"] = snapshot["array_semantics_verified"] = True
        predicted[split] = snapshot
    c._exact(_identities(predicted), expected_identities, "resume metadata identity drift")
    actual = load_scoped_output_identities(
        args, expected_preflight_sha256=expected_preflight_sha256
    )
    c._exact(actual, expected_identities, "resume content identity drift")
    return actual
