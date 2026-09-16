"""Bind independently recomputed preparation to a pinned physical leaf."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict

import numpy as np

from racepv import io as window_io
from racepv.cli import scoped_prepare
from racepv.cli.prepare import _json_bytes
from racepv.data import prepared_contract as contract
from racepv.data import scoped_store as store


def _layout(acceptance_path, descriptor_path):
    path = store._reject_link_components(acceptance_path)
    descriptor = store._reject_link_components(descriptor_path)
    try:
        relative = descriptor.relative_to(path.parent)
    except ValueError as error:
        raise ValueError("descriptor must be contained beneath acceptance parent") from error
    if len(relative.parts) < 2:
        raise ValueError("acceptance must be outside the leaf file graph")
    contract._relative_name(relative.as_posix())
    return path, descriptor, relative.as_posix()


def _checks(preparation_sha256):
    return {
        "preparation_sha256": preparation_sha256,
        "fit_recomputed": True,
        "preparation_recomputed": True,
        "source_interpretation_verified": False,
        "extraction_completeness_verified": False,
        "formal_ready": False,
    }


def _publication(metadata, relative):
    return {
        "descriptor_path": relative,
        **{
            name: metadata[name]
            for name in (
                "descriptor_sha256",
                "read_scope_sha256",
                "aggregate_content_sha256",
                "forecast_population_sha256",
            )
        },
    }


def _match(identity, entry, metadata, pins, supplied_options):
    options, scope, fit = scoped_prepare._admit(entry, supplied_options)
    for name in ("expected_schema_sha256", "expected_preparation_sha256"):
        contract._sha(pins[name], name)
    if contract.canonical_json_sha256(identity) != pins["expected_preparation_sha256"]:
        raise ValueError("external preparation pin mismatch")
    leaves = metadata["leaves"]
    if len(leaves) != 1 or leaves[0]["descriptor_path"] != metadata["descriptor_path"]:
        raise ValueError("preparation acceptance requires one physical leaf, not an index")
    schema = leaves[0]["schema"]
    if contract.canonical_json_sha256(schema) != pins["expected_schema_sha256"]:
        raise ValueError("external schema pin mismatch")
    expected = {
        "kind": "racepv.scoped-preparation-memory",
        "version": 1,
        "formal_ready": False,
        "manifest_entry": contract._strict_json_value(_json_bytes(asdict(entry))),
        "split_plan": options["frozen_split_contract"],
        "row_scope": scope,
        "source_time_contract": options["source_time_contract"],
        "selected_split": options["selected_split"],
        "allowed_splits": list(options["allowed_splits"]),
        "site_index": options["site_index"],
        "stride": options["stride"],
        "train_raw_input_sha256": options["train_raw_input_sha256"],
        "records_sha256": options["expected_records_sha256"],
        "fit_sha256": options["expected_fit_sha256"],
        "arrays": schema["arrays"],
    }
    contract._exact(identity, expected, "preparation versus publication/context")
    for key, value in {
        "dataset_kind": entry.dataset,
        "simulated": entry.dataset in {"synthetic", "nrel"},
        "site": {
            "id": options["site_index"],
            "name": entry.site_id,
            "spatial_role": entry.spatial_role,
        },
        "geometry": {"lookback": 192, "horizon": 24, "stride": options["stride"]},
        "split_plan": options["frozen_split_contract"],
        "row_scope": scope,
        "split": options["selected_split"],
        "fit_states": fit.roles,
    }.items():
        contract._exact(schema[key], value, "published " + key)
    source = copy.deepcopy(options["source_time_contract"])
    # Relocation is the only permissible source-contract difference.
    source["evidence"]["reference"] = schema["sources"]["time_evidence"]["path"]
    contract._exact(schema["source_time_contract"], source, "published source time contract")
    sources = schema["sources"]
    if options["selected_split"] == "train":
        if "frozen_fit_raw_input" in sources:
            raise ValueError("train publication cannot substitute another raw fit source")
        raw = sources["raw_input"]
    else:
        if "frozen_fit_raw_input" not in sources:
            raise ValueError("validation requires an explicit frozen train raw dependency")
        raw = sources["frozen_fit_raw_input"]
    contract._exact(raw["sha256"], options["train_raw_input_sha256"], "published train raw pin")
    return options


def _inspect(descriptor, pins, supplied_options):
    options = dict(supplied_options)
    return store.inspect_scoped_store(
        descriptor,
        expected_descriptor_sha256=pins["expected_descriptor_sha256"],
        expected_read_scope_sha256=pins["expected_read_scope_sha256"],
        allowed_splits=options["allowed_splits"],
    )


def _reload(descriptor, pins, options):
    loaded = store.load_scoped_store(
        descriptor,
        expected_descriptor_sha256=pins["expected_descriptor_sha256"],
        expected_read_scope_sha256=pins["expected_read_scope_sha256"],
        allowed_splits=options["allowed_splits"],
    )
    metadata = loaded["metadata"]
    for arrays in loaded["causal_arrays"]:
        for array in arrays.values():
            if isinstance(array, np.memmap):
                array._mmap.close()
    return metadata


def accept_published_preparation(
    acceptance_path,
    candidate,
    entry,
    source_frame,
    *,
    train_source_frame,
    expected_train_records_sha256,
    descriptor_path,
    expected_descriptor_sha256,
    expected_read_scope_sha256,
    expected_schema_sha256,
    expected_preparation_sha256,
    **supplied_options,
):
    """Recompute fit/windows and verify current published bytes before writing.

    The caller independently pins records and the preparation. Original-file to
    records extraction and source interpretation are NOT established here.
    """
    path, descriptor, relative = _layout(acceptance_path, descriptor_path)
    window_io._v2_check_destination(path)
    pins = dict(
        expected_descriptor_sha256=expected_descriptor_sha256,
        expected_read_scope_sha256=expected_read_scope_sha256,
        expected_schema_sha256=expected_schema_sha256,
        expected_preparation_sha256=expected_preparation_sha256,
    )
    if type(candidate) is not scoped_prepare.ScopedPreparation:
        raise TypeError("candidate must be a scoped preparation")
    metadata = _inspect(descriptor, pins, supplied_options)
    options = _match(candidate.identity, entry, metadata, pins, supplied_options)
    attempt = window_io._v2_attempt(path, directory=False)
    try:
        checks = scoped_prepare.verify_scoped_preparation(
            candidate,
            entry,
            source_frame,
            train_source_frame=train_source_frame,
            expected_train_records_sha256=expected_train_records_sha256,
            expected_preparation_sha256=expected_preparation_sha256,
            **options,
        )
        contract._exact(checks, _checks(expected_preparation_sha256), "actual recomputation")
        current = _reload(descriptor, pins, options)
        _match(candidate.identity, entry, current, pins, options)
        receipt = {
            "kind": "racepv.scoped-preparation-acceptance",
            "version": 1,
            "preparation": copy.deepcopy(candidate.identity),
            "expected_train_records_sha256": expected_train_records_sha256,
            "schema_sha256": expected_schema_sha256,
            "publication": _publication(current, relative),
            "checks": checks,
            "raw_records_binding_verified": False,
            "content_reloaded": True,
        }
        encoded = window_io._v2_json_bytes(receipt)
        window_io._v2_write_bytes(path, encoded)
        return {
            "acceptance_path": str(path),
            "acceptance_sha256": hashlib.sha256(encoded).hexdigest(),
            "formal_ready": False,
        }
    except BaseException as error:
        window_io._v2_failure(attempt, error)
        raise


def verify_published_preparation_acceptance(
    acceptance_path,
    entry,
    *,
    expected_acceptance_sha256,
    descriptor_path,
    expected_descriptor_sha256,
    expected_read_scope_sha256,
    expected_schema_sha256,
    expected_preparation_sha256,
    **supplied_options,
):
    """Verify external acceptance pin, exact context and current leaf content.

    Historical fit/window recomputation is attested by the independently trusted
    acceptance pin, not repeated or inferred from an arbitrary adjacent JSON.
    """
    contract._sha(expected_acceptance_sha256, "expected_acceptance_sha256")
    path, descriptor, relative = _layout(acceptance_path, descriptor_path)
    receipt, _ = store._read_metadata(path, expected_sha256=expected_acceptance_sha256, cache={})
    contract._keys(
        receipt,
        (
            "kind",
            "version",
            "preparation",
            "expected_train_records_sha256",
            "schema_sha256",
            "publication",
            "checks",
            "raw_records_binding_verified",
            "content_reloaded",
        ),
        "preparation acceptance",
    )
    contract._exact(receipt["kind"], "racepv.scoped-preparation-acceptance", "acceptance kind")
    contract._exact(receipt["version"], 1, "acceptance version")
    contract._sha(receipt["expected_train_records_sha256"], "train records")
    contract._exact(receipt["checks"], _checks(expected_preparation_sha256), "acceptance checks")
    contract._exact(receipt["raw_records_binding_verified"], False, "raw records qualification")
    contract._exact(receipt["content_reloaded"], True, "content reload")
    contract._exact(receipt["schema_sha256"], expected_schema_sha256, "schema pin")
    pins = dict(
        expected_descriptor_sha256=expected_descriptor_sha256,
        expected_read_scope_sha256=expected_read_scope_sha256,
        expected_schema_sha256=expected_schema_sha256,
        expected_preparation_sha256=expected_preparation_sha256,
    )
    metadata = _inspect(descriptor, pins, supplied_options)
    options = _match(receipt["preparation"], entry, metadata, pins, supplied_options)
    contract._exact(receipt["publication"], _publication(metadata, relative), "publication binding")
    current = _reload(descriptor, pins, options)
    _match(receipt["preparation"], entry, current, pins, options)
    return {"acceptance": receipt, "publication": current, "formal_ready": False}
