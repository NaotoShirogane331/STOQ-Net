"""Text/metadata correspondence only, not a prepared qualification binding.

Original extraction, column mapping, producer review, preparation acceptance and
content/array checks remain separate requirements. No success here admits a run.
"""

import json
from dataclasses import dataclass

from racepv.data import prepared_contract as contract
from racepv.data.scoped_consumer import _specifications, inspect_scoped_consumer_inputs
from racepv.data.source_registration import verify_source_registration


@dataclass(frozen=True)
class ScopedSourceCorrespondence:
    report_json: bytes

    @property
    def report(self):
        return json.loads(self.report_json)


def _bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _same(left, right, field):
    if _bytes(left) != _bytes(right):
        raise ValueError(f"source/prepared correspondence {field} mismatch")


def _match_site(registration_site, schema):
    for field in ("site", "dataset_kind"):
        _same(registration_site[field], schema[field], field)
    source_clock = registration_site["source_time_contract"]
    prepared_clock = schema["source_time_contract"]
    # Independently contained roots may relocate documents. Compare all clock
    # semantics and exact document bytes, not the roots' relative path spelling.
    _same(
        {key: value for key, value in source_clock.items() if key != "evidence"},
        {key: value for key, value in prepared_clock.items() if key != "evidence"},
        "source clock",
    )
    for field in ("kind", "sha256"):
        _same(source_clock["evidence"][field], prepared_clock["evidence"][field], "time evidence")
    reference = source_clock["evidence"]["reference"]
    identities = [
        locator["file"]
        for locator in registration_site["evidence"]["timestamp"]
        if locator["file"]["path"] == reference
    ]
    # Registration verification has already required the referenced locator.
    if not identities:
        raise ValueError("source/prepared correspondence timestamp evidence missing")
    for field in ("bytes", "sha256"):
        _same(identities[0][field], schema["sources"]["time_evidence"][field], "time evidence")
    coverage, scope = registration_site["coverage"], schema["row_scope"]
    if (
        coverage["start_ns"] > scope["stored_start_ns"]
        or coverage["end_exclusive_ns"] < scope["stored_end_exclusive_ns"]
    ):
        raise ValueError("source/prepared correspondence coverage does not contain row scope")


def inspect_scoped_source_correspondence(
    args,
    *,
    artifact_root,
    registration_name,
    expected_registration_sha256,
    expected_review_sha256,
    expected_preflight_sha256,
    expected_domain,
):
    """Reopen trusted text pins and both metadata graphs on every invocation.

    Pins must come from the caller's trusted contract, not the artifacts themselves.
    This validates scope correspondence, not the exact original records required to
    extract that scope. It deliberately never equates original raw and subset hashes.
    """
    for value, field in (
        (expected_registration_sha256, "registration"),
        (expected_review_sha256, "review"),
        (expected_preflight_sha256, "preflight"),
    ):
        contract._sha(value, f"expected {field} sha256")
    if type(expected_domain) is not str or expected_domain not in (
        "synthetic_fixture",
        "provider_document",
    ):
        raise ValueError("explicit supported source domain required")
    _specifications(args)  # All consumer pins and forbidden-split checks before any open.
    registration = verify_source_registration(
        artifact_root,
        registration_name,
        expected_registration_sha256=expected_registration_sha256,
        expected_review_sha256=expected_review_sha256,
    )
    _same(registration.domain, expected_domain, "domain")
    preflight = inspect_scoped_consumer_inputs(args)
    if contract.canonical_json_sha256(preflight) != expected_preflight_sha256:
        raise ValueError("source correspondence preflight snapshot identity drift")
    interpreted = json.loads(registration.interpretation_json)
    source_sites = {item["site"]["id"]: item for item in interpreted["sites"]}
    prepared = {}
    for split, entry in preflight["inputs"].items():
        leaves = entry["metadata"]["leaves"]
        _same(
            sorted(source_sites),
            sorted(leaf["schema"]["site"]["id"] for leaf in leaves),
            "site set",
        )
        for leaf in leaves:
            schema = leaf["schema"]
            site_id = schema["site"]["id"]
            _match_site(source_sites[site_id], schema)
            prepared.setdefault(site_id, []).append(
                {
                    "split": split,
                    "descriptor_sha256": leaf["descriptor_sha256"],
                    "schema_sha256": contract.canonical_json_sha256(schema),
                    "forecast_population_sha256": schema["forecast_population_sha256"],
                    "row_scope": schema["row_scope"],
                    "fit_states_sha256": contract.canonical_json_sha256(schema["fit_states"]),
                    "sources_sha256": contract.canonical_json_sha256(schema["sources"]),
                }
            )
    report = {
        "artifact_kind": "racepv.scoped-source-correspondence-preflight",
        "schema_version": 1,
        "prepared_schema_version": 3,
        "domain": registration.domain,
        "registration_sha256": registration.registration_sha256,
        "review_sha256": registration.review_sha256,
        "interpretation_sha256": registration.interpretation_sha256,
        "preflight_sha256": expected_preflight_sha256,
        "site_ids": sorted(source_sites),
        "sites": [
            {
                "site_id": site_id,
                "registration_site_sha256": contract.canonical_json_sha256(source_sites[site_id]),
                "prepared": prepared[site_id],
            }
            for site_id in sorted(source_sites)
        ],
        "metadata_correspondence_verified": True,
        "source_interpretation_verified": False,
        "extraction_verified": False,
        "column_mapping_verified": False,
        "producer_review_verified": False,
        "preparation_acceptance_verified": False,
        "content_verified": False,
        "array_semantics_verified": False,
        "registered_evidence_eligible": False,
        "formal_ready": False,
    }
    return ScopedSourceCorrespondence(_bytes(report))
