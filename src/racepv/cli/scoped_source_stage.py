"""Join reviewed source metadata to pinned preparation without granting eligibility.

Only metadata/text files are read. Raw source bytes, producer code and arrays
remain separate stages; this verifier does not execute or follow manifest roots.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from racepv.cli.prepare import _json_bytes
from racepv.cli.scoped_stage_binding import (
    _read,
    _safe_json,
    inspect_extraction_preparation_binding,
)
from racepv.data import prepared_contract as c
from racepv.data.adapters import ManifestEntry
from racepv.data.scoped_source_correspondence import _match_site
from racepv.data.source_registration import verify_source_registration


@dataclass(frozen=True)
class RegisteredPreparationInspection:
    report_json: bytes

    @property
    def report(self):
        return json.loads(self.report_json)


def _same(left, right, name):
    # Canonical numeric equality follows actual normalized ManifestEntry values;
    # strict scalar/container types were already checked by the upstream schemas.
    if _json_bytes(left) != _json_bytes(right):
        raise ValueError("registered preparation mismatch: " + name)


def _manifest_entry(root, identity, entry):
    path = c.contained_artifact_file(root, identity["path"])
    payload = _read(path, identity["sha256"], identity)
    if type(payload) is dict:
        if set(payload) - {"entries", "root"} or "entries" not in payload:
            raise ValueError("unsupported source manifest keys")
        root_name = payload.get("root", ".")
        if root_name != ".":
            c._relative_name(root_name)
        # This is a declaration only, never resolved or opened.
        payload = payload["entries"]
    if type(payload) is not list or not payload:
        raise ValueError("nonempty JSON source manifest entries required")
    matches = []
    keys = {field.name for field in fields(ManifestEntry)}
    for value in payload:
        if type(value) is not dict or set(value) - keys:
            raise ValueError("unknown source manifest entry fields")
        normalized = asdict(ManifestEntry.from_dict(value))
        if (normalized["dataset"], normalized["site_id"]) == (entry["dataset"], entry["site_id"]):
            matches.append(normalized)
    if len(matches) != 1:
        raise ValueError("source manifest must uniquely identify the extracted site")
    _same(matches[0], entry, "source manifest entry")


def _columns(site, entry):
    timestamp = site["timestamp"]
    c._exact(timestamp["column"], entry["timestamp_column"], "timestamp column")
    if entry["timestamp_format"] is None:
        raise ValueError("source qualification requires explicit timestamp parsing format")
    c._exact(timestamp["format"], entry["timestamp_format"], "timestamp format")
    c._exact(entry["timezone"], site["source_time_contract"]["timezone"], "timestamp timezone")
    power = site["columns"]["power"]
    c._exact(power["source_column"], entry["power_column"], "power source column")
    c._exact(power["feature_name"], "power", "power target feature")
    c._exact(power["target_unit"], "kW", "adapter power unit")
    if power["scale"] != entry["power_scale_to_kw"]:
        raise ValueError("adapter power scale differs from reviewed conversion")
    weather = site["columns"]["weather"]
    mapped = {column["feature_name"]: column["source_column"] for column in weather}
    _same(mapped, entry["columns"], "weather columns")
    if set(entry["column_scales"]) - set(mapped):
        raise ValueError("unused weather scale declaration")
    for column in [power, *weather]:
        if column["offset"] != 0:
            raise ValueError("adapter does not implement reviewed nonzero offset")
        if column["measurement"] == "interval_sum":
            raise ValueError("adapter mean-resampling cannot consume interval sums")
    for column in weather:
        if column["scale"] != entry["column_scales"].get(column["feature_name"], 1.0):
            raise ValueError("adapter weather scale differs from reviewed conversion")


def inspect_registered_preparation(
    binding_path, *, registration_root, registration_name, expected_domain, **stage_pins
):
    """Reverify trusted source text and a single site's preparation stage.

    The caller supplies all pins from its external contract. A synthetic result
    remains nonformal. Producer review and full consumer qualification are not
    established by matching declarations here.
    """
    if type(expected_domain) is not str or expected_domain not in (
        "synthetic_fixture",
        "provider_document",
    ):
        raise ValueError("explicit supported source domain required")
    stage = inspect_extraction_preparation_binding(binding_path, **stage_pins)
    report = stage.report
    registration = verify_source_registration(
        registration_root,
        registration_name,
        expected_registration_sha256=stage_pins["expected_registration_sha256"],
        expected_review_sha256=stage_pins["expected_review_sha256"],
    )
    c._exact(registration.domain, expected_domain, "registered source domain")
    schema = report["metadata"]["leaves"][0]["schema"]
    site_id = schema["site"]["id"]
    sites = json.loads(registration.interpretation_json)["sites"]
    selected = [site for site in sites if site["site"]["id"] == site_id]
    if len(selected) != 1:
        raise ValueError("registered site missing for preparation")
    site = selected[0]
    _match_site(site, schema)
    root = _safe_json(binding_path).parent
    for identity in report["extractions"].values():
        extraction = _read(root / identity["path"], identity["sha256"], identity)
        c._exact(extraction["domain"], expected_domain, "extraction source domain")
        if extraction["original_file"] not in site["raw_files"]:
            raise ValueError("original extraction identity is not in reviewed raw files")
        scope = extraction["row_scope"]
        if (
            site["coverage"]["start_ns"] > scope["stored_start_ns"]
            or site["coverage"]["end_exclusive_ns"] < scope["stored_end_exclusive_ns"]
        ):
            raise ValueError("reviewed coverage does not contain required extraction scope")
        entry = extraction["manifest_entry"]
        _columns(site, entry)
        _manifest_entry(Path(registration_root), site["source_manifest"], entry)
    result = {
        "kind": "racepv.registered-preparation-inspection",
        "version": 1,
        "site_id": site_id,
        "domain": expected_domain,
        "registration_sha256": registration.registration_sha256,
        "review_sha256": registration.review_sha256,
        "interpretation_sha256": registration.interpretation_sha256,
        "stage_binding_sha256": report["binding_sha256"],
        "stage_inspection_sha256": c.canonical_json_sha256(report),
        "registration_site_sha256": c.canonical_json_sha256(site),
        "source_manifest": site["source_manifest"],
        "original_identity_matched": True,
        "column_mapping_verified": True,
        "source_manifest_entry_matched": True,
        "source_interpretation_verified": registration.source_interpretation_verified,
        "producer_review_verified": False,
        "content_verified": False,
        "array_semantics_verified": False,
        "formal_ready": False,
    }
    return RegisteredPreparationInspection(_json_bytes(result))
