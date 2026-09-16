"""Pinned preparation-stage receipts, not source/formal qualification.

Metadata inspection never opens original sources, fit bundles or arrays. Full
verification reloads only the declared scoped store. Historical absolute paths
are checked as strings and never used as filesystem targets after relocation.
"""

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

from racepv.cli.prepare import _json_bytes
from racepv.cli.scoped_acceptance import _checks, _reload
from racepv.data import prepared_contract as c
from racepv.data import scoped_store as store
from racepv.data.adapters import ManifestEntry
from racepv.data.scoped_input import _validated_source_and_scope
from racepv.data.source_registration import _signed_int64


@dataclass(frozen=True)
class StageBindingVerification:
    report_json: bytes

    @property
    def report(self):
        return json.loads(self.report_json)


def _record(value, fields, kind, *, version=1):
    c._keys(value, fields.split(), kind)
    c._exact(value["kind"], kind, "receipt kind")
    c._exact(value["version"], version, "receipt version")


def _safe_json(path):
    path = store._reject_link_components(path)
    if path.suffix != ".json" or path.stat().st_nlink != 1:
        raise ValueError("metadata must be a JSON file without hardlink aliases")
    return path


def _read(path, expected_sha256, identity=None):
    path = _safe_json(path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("stage metadata content identity mismatch")
    if identity is not None:
        c._verify_bytes(identity, data)
    return c._strict_json_value(data)


def _reference(root, identity, relative, pin=None):
    c._file_identity(identity)
    c._exact(identity["path"], relative, "stage relative layout")
    if pin is not None:
        c._exact(identity["sha256"], pin, "external stage pin")
    return _safe_json(root / relative)


def _files(value):
    if type(value) is not list or not value:
        raise ValueError("nonempty producer source list required")
    names = set()
    for identity in value:
        c._file_identity(identity)
        name = identity["path"].casefold()
        if name in names:
            raise ValueError("duplicate producer path")
        names.add(name)


def _extraction(value, split, binding):
    _record(
        value,
        "kind version domain original_file registration_sha256 review_sha256 manifest_entry site_index source_time_contract split_plan selected_split row_scope codec source_records selected_records record_ranges record_coordinates source_interval output producer_sources runtime read_scope source_registration_verified formal_ready",
        "racepv.scoped-csv-extraction",
    )
    for key in ("registration_sha256", "review_sha256", "site_index"):
        c._exact(value[key], binding[key], "extraction " + key)
    c._exact(value["selected_split"], split, "extraction split")
    c._exact(value["codec"], "utf8-rfc4180-field-preserving-lf-v1", "extraction codec")
    c._exact(
        value["record_coordinates"],
        "zero_based_data_record_half_open_excluding_header",
        "record coordinates",
    )
    for field in ("formal_ready", "source_registration_verified"):
        c._exact(value[field], False, field)
    c._exact(
        value["read_scope"],
        {
            "full_original_bytes_consumed": True,
            "all_timestamp_fields_scanned": True,
            "numeric_values_parsed": False,
        },
        "actual extraction read scope",
    )
    entry = ManifestEntry.from_dict(value["manifest_entry"])
    c._exact(
        c._strict_json_value(_json_bytes(asdict(entry))),
        value["manifest_entry"],
        "strict manifest entry",
    )
    c._file_identity(value["original_file"])
    c._exact(list(entry.files), [value["original_file"]["path"]], "single original source")
    c._exact(entry.source_file_identities, [value["original_file"]], "original source identity")
    if entry.analysis_start is not None or entry.analysis_end is not None:
        raise ValueError("extraction cannot add analysis cropping")
    if dict(entry.read_options) not in ({}, {"float_precision": "round_trip"}):
        raise ValueError("unsupported extraction CSV options")
    source, scope = _validated_source_and_scope(
        entry,
        frozen_split_contract=value["split_plan"],
        selected_split=split,
        source_time_contract=value["source_time_contract"],
    )
    c._exact(value["row_scope"], scope, "extraction row scope")
    c._exact(value["domain"], source["evidence"]["kind"], "extraction domain")
    c._exact(
        value["domain"] == "synthetic_fixture",
        entry.dataset == "synthetic",
        "extraction dataset domain",
    )
    total = c._integer(value["source_records"], "source records", minimum=1)
    selected = c._integer(value["selected_records"], "selected records", minimum=1)
    if selected > total or total > value["original_file"]["bytes"]:
        raise ValueError("source record counts exceed bounds")
    ranges = value["record_ranges"]
    if type(ranges) is not list or not ranges:
        raise ValueError("nonempty original record ranges required")
    previous, count = -1, 0
    for pair in ranges:
        if type(pair) is not list or len(pair) != 2:
            raise ValueError("record range must be a pair")
        start = c._integer(pair[0], "record start")
        stop = c._integer(pair[1], "record stop", minimum=1)
        if start <= previous or stop <= start or stop > total:
            raise ValueError("noncanonical or out-of-bounds record range")
        previous, count = stop, count + stop - start
    c._exact(count, selected, "selected record count")
    interval = value["source_interval"]
    c._keys(
        interval,
        (
            "minimum_start_ns",
            "maximum_start_ns",
            "maximum_end_exclusive_ns",
            "instantaneous_point_width_ns",
        ),
        "source interval",
    )
    low = _signed_int64(interval["minimum_start_ns"], "minimum source start")
    high = _signed_int64(interval["maximum_start_ns"], "maximum source start")
    end = _signed_int64(interval["maximum_end_exclusive_ns"], "maximum source end")
    c._exact(end, high + (source["interval_duration_ns"] or 1), "source interval end")
    c._exact(
        interval["instantaneous_point_width_ns"],
        1 if source["interval_duration_ns"] is None else None,
        "instantaneous interval",
    )
    if not scope["stored_start_ns"] <= low <= high < end <= scope["stored_end_exclusive_ns"]:
        raise ValueError("source interval outside selected scope")
    c._file_identity(value["output"])
    c._exact(value["output"]["path"], "selected.csv", "extraction output layout")
    _files(value["producer_sources"])
    c._keys(value["runtime"], ("python", "pandas", "numpy"), "extraction runtime")
    if any(type(item) is not str or not item.strip() for item in value["runtime"].values()):
        raise ValueError("runtime versions required")
    return entry


def _historical_root(pipeline, acceptance_identity):
    value = pipeline["preparation_acceptance"]
    c._keys(
        value, ("acceptance_path", "acceptance_sha256", "formal_ready"), "historical acceptance"
    )
    c._exact(value["formal_ready"], False, "historical qualification")
    c._exact(value["acceptance_sha256"], acceptance_identity["sha256"], "acceptance link")
    name = value["acceptance_path"]
    if type(name) is not str:
        raise ValueError("historical path must be text")
    path = PurePosixPath(name)
    if (
        not path.is_absolute()
        or str(path) != name
        or ".." in path.parts
        or "\\" in name
        or path.name != "prepare-acceptance.json"
    ):
        raise ValueError("invalid historical POSIX stage path")
    return path.parent


def _match_prepared(binding, pipeline, acceptance, extractions, schema, allowed):
    _record(
        pipeline,
        "kind version selected_split allowed_splits csv_codec inputs records fit_sha256 schema_sha256 producer_sources publication preparation_acceptance split_payload_binding_verified original_extraction_verified source_interpretation_verified formal_ready",
        "racepv.scoped-csv-preparation",
    )
    _record(
        acceptance,
        "kind version preparation expected_train_records_sha256 schema_sha256 publication checks raw_records_binding_verified content_reloaded",
        "racepv.scoped-preparation-acceptance",
    )
    split = binding["selected_split"]
    train, selected = extractions["train"], extractions[split]
    for field, want in (
        ("selected_split", split),
        ("allowed_splits", list(allowed)),
        ("csv_codec", "utf8-rfc4180-pandas-round-trip-v1"),
        ("split_payload_binding_verified", True),
        ("original_extraction_verified", False),
        ("source_interpretation_verified", False),
        ("formal_ready", False),
    ):
        c._exact(pipeline[field], want, "pipeline " + field)
    _files(pipeline["producer_sources"])
    c._keys(pipeline["records"], ("train", "selected"), "record pins")
    for value in pipeline["records"].values():
        c._sha(value, "records SHA256")
    c._sha(pipeline["fit_sha256"], "fit SHA256")
    schema_sha = c.canonical_json_sha256(schema)
    for value in (binding["schema_sha256"], pipeline["schema_sha256"], acceptance["schema_sha256"]):
        c._exact(value, schema_sha, "schema binding")
    c._keys(pipeline["inputs"], ("train", "selected"), "pipeline inputs")
    for role, extraction in (("train", train), ("selected", selected)):
        identity = pipeline["inputs"][role]
        c._file_identity(identity)
        source_role = (
            "frozen_fit_raw_input" if split == "validation" and role == "train" else "raw_input"
        )
        source = schema["sources"][source_role]
        c._exact(identity["path"], "leaf/" + source["path"], "pipeline input path")
        for field in ("sha256", "bytes"):
            c._exact(identity[field], extraction["output"][field], "extracted input " + field)
            c._exact(identity[field], source[field], "stored input " + field)
    entry = selected["manifest_entry"]
    expected = {
        "kind": "racepv.scoped-preparation-memory",
        "version": 1,
        "formal_ready": False,
        "manifest_entry": entry,
        "split_plan": selected["split_plan"],
        "row_scope": selected["row_scope"],
        "source_time_contract": selected["source_time_contract"],
        "selected_split": split,
        "allowed_splits": list(allowed),
        "site_index": binding["site_index"],
        "stride": schema["geometry"]["stride"],
        "train_raw_input_sha256": train["output"]["sha256"],
        "records_sha256": pipeline["records"]["selected"],
        "fit_sha256": pipeline["fit_sha256"],
        "arrays": schema["arrays"],
    }
    c._exact(acceptance["preparation"], expected, "complete accepted preparation")
    c._exact(acceptance["checks"], _checks(c.canonical_json_sha256(expected)), "acceptance checks")
    c._exact(
        acceptance["expected_train_records_sha256"],
        pipeline["records"]["train"],
        "train records binding",
    )
    c._exact(acceptance["raw_records_binding_verified"], False, "historical source status")
    c._exact(acceptance["content_reloaded"], True, "historical content check")
    for field, want in (
        ("dataset_kind", entry["dataset"]),
        (
            "site",
            {
                "id": binding["site_index"],
                "name": entry["site_id"],
                "spatial_role": entry["spatial_role"],
            },
        ),
        ("split", split),
        ("split_plan", selected["split_plan"]),
        ("row_scope", selected["row_scope"]),
    ):
        c._exact(schema[field], want, "prepared " + field)
    clock = copy.deepcopy(selected["source_time_contract"])
    clock["evidence"]["reference"] = schema["sources"]["time_evidence"]["path"]
    c._exact(schema["source_time_contract"], clock, "prepared source clock")


def inspect_extraction_preparation_binding(
    binding_path,
    *,
    expected_binding_sha256,
    expected_registration_sha256,
    expected_review_sha256,
    expected_train_extraction_sha256,
    expected_selected_extraction_sha256,
    expected_preparation_acceptance_sha256,
    expected_descriptor_sha256,
    expected_read_scope_sha256,
    expected_selected_split,
    allowed_splits,
):
    """Read only strict JSON; external pins attest historical preparation work."""
    pins = (
        expected_binding_sha256,
        expected_registration_sha256,
        expected_review_sha256,
        expected_train_extraction_sha256,
        expected_selected_extraction_sha256,
        expected_preparation_acceptance_sha256,
        expected_descriptor_sha256,
        expected_read_scope_sha256,
    )
    for pin in pins:
        c._sha(pin, "external stage pin")
    if type(expected_selected_split) is not str or expected_selected_split not in (
        "train",
        "validation",
    ):
        raise ValueError("explicit train/validation stage required")
    split = expected_selected_split
    c._exact(
        allowed_splits,
        ("train",) if split == "train" else ("train", "validation"),
        "allowed_splits",
    )
    path = _safe_json(binding_path)
    root = path.parent
    binding = _read(path, expected_binding_sha256)
    _record(
        binding,
        "kind version selected_split site_index registration_sha256 review_sha256 extractions pipeline_receipt preparation_acceptance schema_sha256 extraction_replayed source_registration_verified formal_ready",
        "racepv.extraction-preparation-binding",
    )
    c._integer(binding["site_index"], "site index")
    for field, want in (
        ("selected_split", split),
        ("registration_sha256", expected_registration_sha256),
        ("review_sha256", expected_review_sha256),
        ("extraction_replayed", True),
        ("source_registration_verified", False),
        ("formal_ready", False),
    ):
        c._exact(binding[field], want, "stage " + field)
    c._sha(binding["schema_sha256"], "schema SHA256")
    names = ("train",) if split == "train" else ("train", "validation")
    c._keys(binding["extractions"], names, "extraction references")
    c._exact(
        binding["extractions"][split]["sha256"],
        expected_selected_extraction_sha256,
        "selected extraction external pin",
    )
    references = {
        "pipeline": _reference(root, binding["pipeline_receipt"], "prepared/pipeline.json"),
        "acceptance": _reference(
            root,
            binding["preparation_acceptance"],
            "prepared/prepare-acceptance.json",
            expected_preparation_acceptance_sha256,
        ),
    }
    for name in names:
        pin = (
            expected_train_extraction_sha256
            if name == "train"
            else expected_selected_extraction_sha256
        )
        references[name] = _reference(
            root, binding["extractions"][name], f"extraction/{name}.json", pin
        )
    if len({str(value) for value in (path, *references.values())}) != len(references) + 1:
        raise ValueError("stage metadata role alias")
    extractions = {}
    for name in names:
        identity = binding["extractions"][name]
        value = _read(references[name], identity["sha256"], identity)
        _extraction(value, name, binding)
        extractions[name] = value
    if split == "validation":
        for key in (
            "domain",
            "original_file",
            "manifest_entry",
            "source_time_contract",
            "split_plan",
            "producer_sources",
            "runtime",
            "source_records",
        ):
            c._exact(
                extractions["train"][key], extractions[split][key], "cross-split extraction " + key
            )
    pipeline_identity = binding["pipeline_receipt"]
    acceptance_identity = binding["preparation_acceptance"]
    pipeline = _read(references["pipeline"], pipeline_identity["sha256"], pipeline_identity)
    acceptance = _read(
        references["acceptance"], expected_preparation_acceptance_sha256, acceptance_identity
    )
    # Validate top-level shapes before using nested references. The complete
    # scientific/context checks follow after metadata-only store admission.
    _record(
        pipeline,
        "kind version selected_split allowed_splits csv_codec inputs records fit_sha256 schema_sha256 producer_sources publication preparation_acceptance split_payload_binding_verified original_extraction_verified source_interpretation_verified formal_ready",
        "racepv.scoped-csv-preparation",
    )
    _record(
        acceptance,
        "kind version preparation expected_train_records_sha256 schema_sha256 publication checks raw_records_binding_verified content_reloaded",
        "racepv.scoped-preparation-acceptance",
    )
    accepted_publication = acceptance["publication"]
    publication = pipeline["publication"]
    c._keys(
        accepted_publication,
        (
            "descriptor_path",
            "descriptor_sha256",
            "read_scope_sha256",
            "aggregate_content_sha256",
            "forecast_population_sha256",
        ),
        "accepted publication",
    )
    c._keys(
        publication,
        (
            "acceptance_path",
            "acceptance_sha256",
            "descriptor_path",
            "descriptor_sha256",
            "read_scope_sha256",
            "aggregate_content_sha256",
            "forecast_population_sha256",
            "formal_ready",
            "registered_evidence_eligible",
        ),
        "pipeline publication",
    )
    c._exact(
        accepted_publication["descriptor_path"], "leaf/store.json", "accepted descriptor layout"
    )
    historical_root = _historical_root(pipeline, acceptance_identity)
    c._exact(
        publication["descriptor_path"],
        str(historical_root / "leaf/store.json"),
        "historical descriptor path",
    )
    c._exact(
        publication["acceptance_path"],
        str(historical_root / "stage/acceptance.json"),
        "historical staging path",
    )
    c._sha(publication["acceptance_sha256"], "historical staging SHA256")
    for name in ("formal_ready", "registered_evidence_eligible"):
        c._exact(publication[name], False, "historical publication status")
    for name, pin in (
        ("descriptor_sha256", expected_descriptor_sha256),
        ("read_scope_sha256", expected_read_scope_sha256),
    ):
        c._exact(publication[name], pin, "publication external pin")
        c._exact(accepted_publication[name], pin, "acceptance external pin")
    specification = {
        "descriptor_path": str(root / "prepared/leaf/store.json"),
        "expected_descriptor_sha256": expected_descriptor_sha256,
        "expected_read_scope_sha256": expected_read_scope_sha256,
        "allowed_splits": allowed_splits,
    }
    metadata = store.inspect_scoped_store(**specification)
    c._exact(metadata["selected_split"], split, "actual selected split")
    if (
        len(metadata["leaves"]) != 1
        or metadata["leaves"][0]["descriptor_path"] != metadata["descriptor_path"]
    ):
        raise ValueError("one accepted physical leaf required")
    for name in (
        "descriptor_sha256",
        "read_scope_sha256",
        "aggregate_content_sha256",
        "forecast_population_sha256",
    ):
        c._exact(metadata[name], publication[name], "current versus pipeline " + name)
        c._exact(metadata[name], accepted_publication[name], "current versus acceptance " + name)
    schema = metadata["leaves"][0]["schema"]
    _match_prepared(binding, pipeline, acceptance, extractions, schema, allowed_splits)
    report = {
        "kind": "racepv.stage-binding-verification",
        "version": 1,
        "binding_sha256": expected_binding_sha256,
        "selected_split": split,
        "registration_sha256": expected_registration_sha256,
        "review_sha256": expected_review_sha256,
        "extractions": binding["extractions"],
        "pipeline_receipt": pipeline_identity,
        "preparation_acceptance": acceptance_identity,
        "store_specification": specification,
        "metadata": metadata,
        "historical_extraction_attested_by_pin": True,
        "extraction_replayed": False,
        "content_verified": False,
        "array_semantics_verified": False,
        "source_registration_verified": False,
        "formal_ready": False,
    }
    return StageBindingVerification(_json_bytes(report))


def verify_extraction_preparation_binding(binding_path, **pins):
    """Verify current allowed-split content, then recheck all stage JSON pins.

    Original-source extraction and numeric fits are historical, externally pinned
    attestations. This function never opens their old paths or repeats those stages.
    """
    before = inspect_extraction_preparation_binding(binding_path, **pins)
    report = before.report
    spec = report["store_specification"]
    current = _reload(
        spec["descriptor_path"], spec, {"allowed_splits": tuple(spec["allowed_splits"])}
    )
    snapshot = copy.deepcopy(current)
    snapshot["content_verified"] = snapshot["array_semantics_verified"] = False
    c._exact(snapshot, report["metadata"], "stage metadata changed during content verification")
    after = inspect_extraction_preparation_binding(binding_path, **pins)
    if after != before:
        raise ValueError("stage receipts changed during content verification")
    report["metadata"] = current
    report["content_verified"] = report["array_semantics_verified"] = True
    return StageBindingVerification(_json_bytes(report))
