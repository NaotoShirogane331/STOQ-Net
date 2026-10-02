"""Externally pinned qualification for the schema3 candidate train/validation pair.

An external pin is a caller-supplied trust root, not proof of the truth of an
attestation. Current producer bytes and current scoped contents are rechecked;
historical extraction/fit attestations are never replayed here. Synthetic
fixtures remain formally ineligible throughout this API.
"""

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from racepv.cli.scoped_source_stage import inspect_registered_preparation
from racepv.cli.scoped_stage_binding import inspect_extraction_preparation_binding
from racepv.data import prepared_contract as c
from racepv.data import scoped_consumer, scoped_output, scoped_store
from racepv.data.scoped_dataset import ScopedCandidateDataset
from racepv.data.source_registration import verify_source_registration

_STAGE_PINS = (
    "expected_binding_sha256",
    "expected_registration_sha256",
    "expected_review_sha256",
    "expected_train_extraction_sha256",
    "expected_selected_extraction_sha256",
    "expected_preparation_acceptance_sha256",
    "expected_descriptor_sha256",
    "expected_read_scope_sha256",
    "expected_selected_split",
    "allowed_splits",
)
_LEAF_PINS = (
    "descriptor_sha256",
    "read_scope_sha256",
    "aggregate_content_sha256",
    "forecast_population_sha256",
)
_TRANSFORMS = dict(
    extraction_codec="utf8-rfc4180-field-preserving-lf-v1",
    preparation_codec="utf8-rfc4180-pandas-round-trip-v1",
    causality_profile="as-of-origin-v1",
    fit_policy="frozen_train_fit_no_refit",
)


def _bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


@dataclass(frozen=True)
class QualifiedCandidateInspection:
    report_json: bytes

    @property
    def report(self):
        return json.loads(self.report_json)


def _read(root, identity, *, expected_sha256=None, as_json=True):
    c._file_identity(identity)
    if expected_sha256 is not None:
        c._sha(expected_sha256, "external qualification pin")
        c._exact(identity["sha256"], expected_sha256, "external qualification pin")
    path = c.contained_artifact_file(root, identity["path"])
    with scoped_store._open_held(path) as (handle, _):
        data = handle.read()
    c._verify_bytes(identity, data)
    return c._strict_json_value(data) if as_json else data


def _bundle(args, domain):
    if type(domain) is not str or domain not in ("synthetic_fixture", "provider_document"):
        raise ValueError("explicit qualification domain required")
    for name in ("qualification_sha256", "producer_review_sha256"):
        c._sha(getattr(args, name, None), "external " + name)
    name = getattr(args, "qualification_name", None)
    c._relative_name(name)
    if not name.endswith(".json"):
        raise ValueError("qualification bundle must be JSON")
    root_value = getattr(args, "qualification_root", None)
    producer_value = getattr(args, "producer_root", None)
    if root_value is None or producer_value is None:
        raise ValueError("explicit artifact and current producer roots required")
    root = scoped_store._reject_link_components(root_value)
    producer_root = scoped_store._reject_link_components(producer_value)
    scoped_consumer._specifications(args)  # Every external consumer pin before file reads.
    path = c.contained_artifact_file(root, name)
    with scoped_store._open_held(path) as (handle, _):
        data = handle.read()
    c._exact(
        hashlib.sha256(data).hexdigest(), args.qualification_sha256, "qualification bundle pin"
    )
    bundle = c._strict_json_value(data)
    c._keys(bundle, ("kind", "version", "domain", "producer_review", "stages"), "qualification")
    c._exact(bundle["kind"], "racepv.scoped-qualification-bundle", "qualification kind")
    c._exact(bundle["version"], 1, "qualification version")
    c._exact(bundle["domain"], domain, "qualification domain")
    review = _read(root, bundle["producer_review"], expected_sha256=args.producer_review_sha256)
    c._keys(
        review,
        ("kind", "version", "domain", "reviewer_id", "method", "decision", "stages"),
        "producer review",
    )
    c._exact(review["kind"], "racepv.producer-transform-review", "producer review kind")
    c._exact(review["version"], 1, "producer review version")
    c._exact(review["domain"], domain, "producer review domain")
    c._exact(review["method"], "independent_producer_semantic_review", "producer review method")
    c._exact(review["decision"], "approved", "producer review decision")
    if type(review["reviewer_id"]) is not str or not review["reviewer_id"].strip():
        raise ValueError("producer reviewer identity required")
    if type(bundle["stages"]) is not list or not bundle["stages"]:
        raise ValueError("complete qualification stage list required")
    if type(review["stages"]) is not list or len(review["stages"]) != len(bundle["stages"]):
        raise ValueError("complete producer review stage list required")
    return root, producer_root, bundle, review


def _producer_files(root, files):
    if type(files) is not list or not files:
        raise ValueError("explicit producer sources required")
    seen = set()
    for identity in files:
        c._file_identity(identity)
        key = identity["path"].casefold()
        if key in seen:
            raise ValueError("duplicate or aliased producer source")
        seen.add(key)
        _read(root, identity, as_json=False)


def inspect_qualified_candidate(args, *, expected_domain="provider_document"):
    """Recheck complete metadata coverage and reviewed producer bytes before arrays."""
    root, producer_root, bundle, review = _bundle(args, expected_domain)
    preflight = scoped_consumer.inspect_scoped_consumer_inputs(args)
    expected_leaves = {
        (leaf["schema"]["site"]["id"], split): leaf
        for split, entry in preflight["inputs"].items()
        for leaf in entry["metadata"]["leaves"]
    }
    seen, file_roles = set(), {}

    def role(identity, name):
        c._file_identity(identity)
        key = identity["path"].casefold()
        existing = file_roles.get(key)
        if existing is not None and existing != (name, identity):
            raise ValueError("qualification file role or path alias")
        file_roles[key] = (name, identity)

    role(
        dict(
            path=args.qualification_name,
            sha256=args.qualification_sha256,
            bytes=(root / args.qualification_name).stat().st_size,
        ),
        "bundle",
    )
    role(bundle["producer_review"], "producer review")
    for stage in bundle["stages"]:
        c._keys(stage, ("site_id", "split", "binding", "registration", "pins"), "qualified stage")
        c._integer(stage["site_id"], "stage site ID")
        if type(stage["split"]) is not str or stage["split"] not in ("train", "validation"):
            raise ValueError("qualification supports exactly train/validation")
        key = (stage["site_id"], stage["split"])
        if key in seen or key not in expected_leaves:
            raise ValueError("duplicate or unselected site/split qualification")
        seen.add(key)
        c._keys(stage["pins"], _STAGE_PINS, "qualified stage external pins")
        for name in _STAGE_PINS[:-2]:
            c._sha(stage["pins"][name], name)
        c._exact(stage["pins"]["expected_selected_split"], stage["split"], "stage split role")
        c._exact(
            stage["pins"]["allowed_splits"],
            ["train"] if stage["split"] == "train" else ["train", "validation"],
            "stage scope",
        )
        role(stage["binding"], "stage binding")
        role(stage["registration"], "source registration")
    if seen != set(expected_leaves):
        raise ValueError("qualification does not cover all selected site/split leaves")
    reviews = {}
    for item in review["stages"]:
        c._keys(
            item,
            (
                "binding_sha256",
                "registered_producer",
                "prepared_producer",
                "extraction_sources",
                "preparation_sources",
                "transforms",
            ),
            "reviewed stage",
        )
        pin = item["binding_sha256"]
        c._sha(pin, "reviewed binding pin")
        if pin in reviews:
            raise ValueError("duplicate reviewed stage")
        c._exact(item["transforms"], _TRANSFORMS, "approved stage transforms")
        reviews[pin] = item
    if set(reviews) != {stage["binding"]["sha256"] for stage in bundle["stages"]}:
        raise ValueError("producer review must cover exact qualification stages")
    stage_reports = []
    for stage in bundle["stages"]:
        pins = dict(stage["pins"])
        pins["allowed_splits"] = tuple(pins["allowed_splits"])
        binding = _read(root, stage["binding"], expected_sha256=pins["expected_binding_sha256"])
        _read(root, stage["registration"], expected_sha256=pins["expected_registration_sha256"])
        binding_path = c.contained_artifact_file(root, stage["binding"]["path"])
        registration_path = c.contained_artifact_file(root, stage["registration"]["path"])
        registered = inspect_registered_preparation(
            binding_path,
            registration_root=registration_path.parent,
            registration_name=registration_path.name,
            expected_domain=expected_domain,
            **pins,
        )
        inspected = inspect_extraction_preparation_binding(binding_path, **pins).report
        leaf = inspected["metadata"]["leaves"][0]
        selected = expected_leaves[(stage["site_id"], stage["split"])]
        for name in (*_LEAF_PINS, "schema"):
            c._exact(leaf[name], selected[name], "consumer versus qualified leaf " + name)
        registration = verify_source_registration(
            registration_path.parent,
            registration_path.name,
            expected_registration_sha256=pins["expected_registration_sha256"],
            expected_review_sha256=pins["expected_review_sha256"],
        )
        sites = json.loads(registration.interpretation_json)["sites"]
        site = next(item for item in sites if item["site"]["id"] == stage["site_id"])
        approved = reviews[stage["binding"]["sha256"]]
        c._exact(approved["registered_producer"], site["producer"], "reviewed registered producer")
        c._exact(
            approved["prepared_producer"],
            leaf["schema"]["sources"]["producer"],
            "reviewed prepared producer",
        )
        _read(registration_path.parent, site["producer"], as_json=False)
        _read(Path(leaf["descriptor_path"]).parent, approved["prepared_producer"], as_json=False)
        pipeline = _read(binding_path.parent, binding["pipeline_receipt"])
        c._exact(
            approved["preparation_sources"],
            pipeline["producer_sources"],
            "reviewed preparation sources",
        )
        _producer_files(producer_root, approved["preparation_sources"])
        for identity in binding["extractions"].values():
            extraction = _read(binding_path.parent, identity)
            c._exact(
                approved["extraction_sources"],
                extraction["producer_sources"],
                "reviewed extraction sources",
            )
        _producer_files(producer_root, approved["extraction_sources"])
        if (
            expected_domain == "provider_document"
            and not registration.source_interpretation_verified
        ):
            raise ValueError("provider source interpretation is not approved")
        stage_reports.append(
            dict(
                site_id=stage["site_id"],
                split=stage["split"],
                binding_sha256=stage["binding"]["sha256"],
                registration_sha256=registration.registration_sha256,
                source_review_sha256=registration.review_sha256,
                registered_preparation=registered.report,
                leaf_identity={name: leaf[name] for name in _LEAF_PINS},
                fit_states_sha256=c.canonical_json_sha256(leaf["schema"]["fit_states"]),
            )
        )
    return QualifiedCandidateInspection(
        _bytes(
            dict(
                kind="racepv.qualified-candidate-inspection",
                version=1,
                domain=expected_domain,
                qualification_sha256=args.qualification_sha256,
                producer_review_sha256=args.producer_review_sha256,
                preflight_sha256=c.canonical_json_sha256(preflight),
                stages=stage_reports,
                producer_review_verified=True,
                content_verified=False,
                array_semantics_verified=False,
                formal_ready=False,
            )
        )
    )


def _qualification_identity(report):
    identity = {
        key: copy.deepcopy(value) for key, value in report.items() if key != "preflight_sha256"
    }
    # Inspection snapshots contain local resolved paths. Portable qualification
    # retains the externally pinned binding, registration and complete leaf/fit
    # identities instead of using a hash of a host-local diagnostic report.
    for stage in identity["stages"]:
        del stage["registered_preparation"]["stage_inspection_sha256"]
    return identity


def load_qualified_candidate(args, *, expected_domain="provider_document"):
    """Load only after all sites, pins, source interpretations and reviews match."""
    before = inspect_qualified_candidate(args, expected_domain=expected_domain)
    report = before.report
    loaded = scoped_consumer.load_scoped_engineering_inputs(
        args, expected_preflight_sha256=report["preflight_sha256"]
    )
    after = inspect_qualified_candidate(args, expected_domain=expected_domain)
    if after != before:
        raise ValueError("qualification changed during scoped content verification")
    report.update(
        content_verified=True,
        array_semantics_verified=True,
        formal_ready=expected_domain == "provider_document",
    )
    qualification = QualifiedCandidateInspection(_bytes(report))
    datasets = {}
    for split, store in loaded["stores"].items():
        metadata = store["metadata"]
        # The engineering view pins the metadata-only flags; the adapter consumes
        # the same independently verified graph with actual content flags.
        from racepv.data.scoped_scientific_view import scientific_view_from_metadata

        view = scientific_view_from_metadata(
            metadata,
            selected_split=split,
            expected_metadata_sha256=c.canonical_json_sha256(metadata),
        )
        datasets[split] = ScopedCandidateDataset(store, view)
    identities = scoped_output._identities(
        {s: value["metadata"] for s, value in loaded["stores"].items()}
    )
    identities.update(
        qualification_schema="racepv.qualified-scoped-output.v1",
        qualification=_qualification_identity(report),
    )
    return dict(
        qualification=qualification, datasets=datasets, engineering=loaded, identities=identities
    )


def validate_qualified_output_identity(value):
    """Validate serialization only; accepting output also requires fresh reverify."""
    c._keys(
        value,
        (
            "descriptor_hashes",
            "content_hashes",
            "prepared_contracts",
            "qualification_schema",
            "qualification",
        ),
        "qualified output",
    )
    c._exact(
        value["qualification_schema"], "racepv.qualified-scoped-output.v1", "qualification schema"
    )
    scoped_output.require_scoped_output_pair(value["prepared_contracts"])
    qualification = value["qualification"]
    c._keys(
        qualification,
        (
            "kind",
            "version",
            "domain",
            "qualification_sha256",
            "producer_review_sha256",
            "stages",
            "producer_review_verified",
            "content_verified",
            "array_semantics_verified",
            "formal_ready",
        ),
        "output qualification",
    )
    c._exact(qualification["kind"], "racepv.qualified-candidate-inspection", "qualification kind")
    c._exact(qualification["version"], 1, "qualification version")
    for name in ("qualification_sha256", "producer_review_sha256"):
        c._sha(qualification[name], name)
    for name in ("producer_review_verified", "content_verified", "array_semantics_verified"):
        c._exact(qualification[name], True, name)
    if type(qualification["domain"]) is not str or qualification["domain"] not in (
        "provider_document",
        "synthetic_fixture",
    ):
        raise ValueError("unsupported output qualification domain")
    c._exact(
        qualification["formal_ready"],
        qualification["domain"] == "provider_document",
        "formal domain",
    )
    c._keys(value["descriptor_hashes"], ("train", "validation"), "descriptor hashes")
    c._keys(value["content_hashes"], ("train", "validation"), "content hashes")
    leaves = {
        (leaf["schema"]["site"]["id"], split): leaf
        for split, base in value["prepared_contracts"].items()
        for leaf in base["scoped_metadata"]["leaves"]
    }
    stages = qualification["stages"]
    if type(stages) is not list or len(stages) != len(leaves):
        raise ValueError("output qualification stage coverage differs")
    seen = set()
    for stage in stages:
        c._keys(
            stage,
            (
                "site_id",
                "split",
                "binding_sha256",
                "registration_sha256",
                "source_review_sha256",
                "registered_preparation",
                "leaf_identity",
                "fit_states_sha256",
            ),
            "output qualification stage",
        )
        c._integer(stage["site_id"], "qualification site ID")
        if type(stage["split"]) is not str or stage["split"] not in ("train", "validation"):
            raise ValueError("output qualification split invalid")
        key = (stage["site_id"], stage["split"])
        if key in seen or key not in leaves:
            raise ValueError("output qualification duplicated or unselected site/split")
        seen.add(key)
        for field in (
            "binding_sha256",
            "registration_sha256",
            "source_review_sha256",
            "fit_states_sha256",
        ):
            c._sha(stage[field], field)
        leaf = leaves[key]
        c._exact(
            stage["leaf_identity"],
            {field: leaf[field] for field in _LEAF_PINS},
            "qualified leaf identity",
        )
        c._exact(
            stage["fit_states_sha256"],
            c.canonical_json_sha256(leaf["schema"]["fit_states"]),
            "qualified fit identity",
        )
        registered = stage["registered_preparation"]
        c._keys(
            registered,
            (
                "kind",
                "version",
                "site_id",
                "domain",
                "registration_sha256",
                "review_sha256",
                "interpretation_sha256",
                "stage_binding_sha256",
                "registration_site_sha256",
                "source_manifest",
                "original_identity_matched",
                "column_mapping_verified",
                "source_manifest_entry_matched",
                "source_interpretation_verified",
                "producer_review_verified",
                "content_verified",
                "array_semantics_verified",
                "formal_ready",
            ),
            "registered stage output",
        )
        for field, want in (
            ("kind", "racepv.registered-preparation-inspection"),
            ("version", 1),
            ("site_id", stage["site_id"]),
            ("domain", qualification["domain"]),
            ("registration_sha256", stage["registration_sha256"]),
            ("review_sha256", stage["source_review_sha256"]),
            ("stage_binding_sha256", stage["binding_sha256"]),
        ):
            c._exact(registered[field], want, "registered stage " + field)
        for field in ("interpretation_sha256", "registration_site_sha256"):
            c._sha(registered[field], field)
        c._file_identity(registered["source_manifest"])
        for field in (
            "original_identity_matched",
            "column_mapping_verified",
            "source_manifest_entry_matched",
        ):
            c._exact(registered[field], True, field)
        c._exact(
            registered["source_interpretation_verified"],
            qualification["domain"] == "provider_document",
            "source interpretation domain",
        )
        for field in (
            "producer_review_verified",
            "content_verified",
            "array_semantics_verified",
            "formal_ready",
        ):
            c._exact(registered[field], False, "historical inspection " + field)
    for split, base in value["prepared_contracts"].items():
        c._exact(
            value["descriptor_hashes"][split], base["descriptor_sha256"], "output descriptor pin"
        )
        c._exact(
            value["content_hashes"][split], base["aggregate_content_sha256"], "output content pin"
        )
        c._exact(base["source_evidence_kind"], qualification["domain"], "output source domain")
    return copy.deepcopy(value)


def reverify_qualified_candidate(args, expected_identities, *, expected_domain="provider_document"):
    """Recovery/export boundary: recheck caller pins, prior identity, then all bytes."""
    expected = validate_qualified_output_identity(expected_identities)
    inspection = inspect_qualified_candidate(args, expected_domain=expected_domain).report
    predicted = {
        **inspection,
        "content_verified": True,
        "array_semantics_verified": True,
        "formal_ready": expected_domain == "provider_document",
    }
    c._exact(
        _qualification_identity(predicted), expected["qualification"], "saved qualification drift"
    )
    base = scoped_output.reverify_scoped_output_identities(
        args,
        {k: expected[k] for k in ("descriptor_hashes", "content_hashes", "prepared_contracts")},
        expected_preflight_sha256=inspection["preflight_sha256"],
    )
    after = inspect_qualified_candidate(args, expected_domain=expected_domain).report
    c._exact(after, inspection, "qualification changed during output reverification")
    actual = {
        **base,
        "qualification_schema": "racepv.qualified-scoped-output.v1",
        "qualification": _qualification_identity(predicted),
    }
    c._exact(actual, expected, "qualified output drift")
    return actual
