"""Pinned, text-only source-interpretation registration verification."""

import hashlib
import json
import math
import re
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from racepv.data.origin_clock import validate_source_time_contract
from racepv.data.prepared_contract import (
    _exact,
    _file_identity,
    _integer,
    _keys,
    _relative_name,
    _sha,
    _strict_json_value,
    _verify_bytes,
    canonical_json_sha256,
    contained_artifact_file,
)
from racepv.io import REGISTERED_DATASET_KINDS, REGISTERED_SPATIAL_ROLES

REGISTRATION_FIELDS = {"artifact_kind", "schema_version", "interpretation", "review"}
INTERPRETATION_FIELDS = {"domain", "sites"}
SITE_FIELDS = {
    "site",
    "dataset_kind",
    "provider_id",
    "provider_site_id",
    "export_id",
    "coverage",
    "timestamp",
    "columns",
    "source_time_contract",
    "source_time_contract_sha256",
    "source_manifest",
    "raw_files",
    "producer",
    "evidence",
}
FACT_GROUPS = ("applicability", "timestamp", "power", "weather", "availability")
REVIEW_FIELDS = {
    "artifact_kind",
    "schema_version",
    "interpretation_sha256",
    "reviewer",
    "reviewed_at",
    "site_decisions",
}
DECISION_FIELDS = {"site_id", "site_sha256", "facts"}
_COLUMN_FIELDS = {
    "source_column",
    "feature_name",
    "measurement",
    "source_unit",
    "target_unit",
    "scale",
    "offset",
}
_TEXT_SUFFIXES = (".txt", ".md", ".json")
_DOMAINS = {"provider_document", "synthetic_fixture"}
_MEASUREMENTS = {"instantaneous", "interval_mean", "interval_sum"}
_UNKNOWN_FORMATS = {"unknown", "unspecified", "conflicting"}
_UTC_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


@dataclass(frozen=True)
class VerifiedSourceRegistration:
    registration_sha256: str
    review_sha256: str
    interpretation_sha256: str
    domain: str
    interpretation_json: bytes
    verified_text_identities_json: bytes

    @property
    def source_interpretation_verified(self):
        return self.domain == "provider_document"

    @property
    def content_verified(self):
        return False

    @property
    def array_semantics_verified(self):
        return False

    @property
    def registered_evidence_eligible(self):
        return False


def verify_source_registration(
    artifact_root,
    registration_name,
    *,
    expected_registration_sha256,
    expected_review_sha256,
) -> VerifiedSourceRegistration:
    """Verify externally trusted text pins; never open declared raw/producer files."""
    registration_sha256 = _sha(expected_registration_sha256, "expected registration sha256")
    review_sha256 = _sha(expected_review_sha256, "expected review sha256")
    registration_name = _relative_name(registration_name)
    if not registration_name.endswith(".json"):
        raise ValueError("JSON publication required")

    root = Path(artifact_root)
    registration_path = contained_artifact_file(root, registration_name)
    registration_data = registration_path.read_bytes()
    if hashlib.sha256(registration_data).hexdigest() != registration_sha256:
        raise ValueError("registration content identity mismatch")
    registration = _strict_json_value(registration_data.decode("utf-8"))

    _keys(registration, REGISTRATION_FIELDS, "source registration")
    _exact(
        registration["artifact_kind"],
        "racepv.source-interpretation-registration",
        "registration.artifact_kind",
    )
    _exact(registration["schema_version"], 1, "registration.schema_version")
    _file_identity(registration["review"])
    if not registration["review"]["path"].endswith(".json"):
        raise ValueError("JSON publication required")
    _exact(registration["review"]["sha256"], review_sha256, "registration.review.sha256")

    interpretation, locators, identities = _validate_interpretation(registration["interpretation"])
    registration_identity = {
        "path": registration_name,
        "bytes": len(registration_data),
        "sha256": registration_sha256,
    }
    role_identities = [
        ("registration", registration_identity),
        ("review", registration["review"]),
        *identities,
    ]
    _validate_identity_graph(root, role_identities)

    review, review_data = read_pinned_json(root, registration["review"])
    _validate_review(review, interpretation)

    evidence_identities = _verify_evidence(root, locators)
    interpretation_json = _canonical_json_bytes(interpretation)
    interpretation_sha256 = hashlib.sha256(interpretation_json).hexdigest()
    verified_identities = {
        "registration": registration_identity,
        "review": registration["review"],
        "evidence": evidence_identities,
    }
    _verify_bytes(registration["review"], review_data)
    return VerifiedSourceRegistration(
        registration_sha256=registration_sha256,
        review_sha256=review_sha256,
        interpretation_sha256=interpretation_sha256,
        domain=interpretation["domain"],
        interpretation_json=interpretation_json,
        verified_text_identities_json=_canonical_json_bytes(verified_identities),
    )


def read_pinned_json(root, identity):
    _file_identity(identity)
    if not identity["path"].endswith(".json"):
        raise ValueError("JSON publication required")
    path = contained_artifact_file(root, identity["path"])
    data = path.read_bytes()
    _verify_bytes(identity, data)
    return _strict_json_value(data.decode("utf-8")), data


def _canonical_json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _nonempty_string(value, name):
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty and stripped")
    return value


def _signed_int64(value, name):
    if type(value) is not int:
        raise TypeError(f"{name} must be a strict integer")
    if value <= -(2**63) or value > 2**63 - 1:
        raise ValueError(f"{name} must be a non-NaT signed int64")
    return value


def _finite_number(value, name, *, positive=False):
    if type(value) is float and not math.isfinite(value):
        raise TypeError(f"{name} must be a finite real JSON number")
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a finite real JSON number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be strictly positive")
    return value


def _validate_site_identity(value, name):
    _keys(value, ("id", "name", "spatial_role"), name)
    site_id = _integer(value["id"], name + ".id")
    site_name = _nonempty_string(value["name"], name + ".name")
    role = value["spatial_role"]
    if type(role) is not str or role not in REGISTERED_SPATIAL_ROLES:
        raise ValueError(f"{name}.spatial_role is not registered")
    return site_id, site_name


def _validate_column(value, name):
    _keys(value, _COLUMN_FIELDS, name)
    for field in ("source_column", "feature_name", "source_unit", "target_unit"):
        _nonempty_string(value[field], name + "." + field)
    if type(value["measurement"]) is not str or value["measurement"] not in _MEASUREMENTS:
        raise ValueError(f"{name}.measurement is invalid")
    _finite_number(value["scale"], name + ".scale", positive=True)
    _finite_number(value["offset"], name + ".offset")


def _validate_columns(value, name):
    _keys(value, ("power", "weather"), name)
    _validate_column(value["power"], name + ".power")
    weather = value["weather"]
    if type(weather) is not list:
        raise TypeError(f"{name}.weather must be a list")
    registered_names = {
        value["power"]["source_column"],
        value["power"]["feature_name"],
    }
    for index, column in enumerate(weather):
        _validate_column(column, f"{name}.weather[{index}]")
        source_column = column["source_column"]
        feature_name = column["feature_name"]
        if (
            source_column == feature_name
            or source_column in registered_names
            or feature_name in registered_names
        ):
            raise ValueError("weather columns must not duplicate or alias power")
        registered_names.update((source_column, feature_name))


def _validate_locator(value, name):
    _keys(value, ("file", "start_line", "end_line"), name)
    _file_identity(value["file"])
    if not value["file"]["path"].endswith(_TEXT_SUFFIXES):
        raise ValueError("evidence must use a UTF-8 text suffix")
    start = _integer(value["start_line"], name + ".start_line", minimum=1)
    end = _integer(value["end_line"], name + ".end_line", minimum=1)
    if end < start:
        raise ValueError("evidence locator line interval is reversed")


def _validate_site(value, domain, index):
    name = f"interpretation.sites[{index}]"
    _keys(value, SITE_FIELDS, name)
    site_id, site_name = _validate_site_identity(value["site"], name + ".site")
    dataset_kind = value["dataset_kind"]
    if type(dataset_kind) is not str or dataset_kind not in REGISTERED_DATASET_KINDS:
        raise ValueError(f"{name}.dataset_kind is not registered")
    if (domain == "synthetic_fixture") != (dataset_kind == "synthetic"):
        raise ValueError("synthetic domain and dataset kind must agree")
    for field in ("provider_id", "provider_site_id", "export_id"):
        _nonempty_string(value[field], name + "." + field)

    _keys(value["coverage"], ("start_ns", "end_exclusive_ns"), name + ".coverage")
    start = _signed_int64(value["coverage"]["start_ns"], name + ".coverage.start_ns")
    end = _signed_int64(value["coverage"]["end_exclusive_ns"], name + ".coverage.end_exclusive_ns")
    if start >= end:
        raise ValueError("coverage must be a nonempty half-open interval")

    _keys(value["timestamp"], ("column", "format"), name + ".timestamp")
    _nonempty_string(value["timestamp"]["column"], name + ".timestamp.column")
    timestamp_format = _nonempty_string(value["timestamp"]["format"], name + ".timestamp.format")
    if timestamp_format.casefold() in _UNKNOWN_FORMATS:
        raise ValueError("timestamp format must name the actual export format")
    _validate_columns(value["columns"], name + ".columns")

    normalized_time = validate_source_time_contract(value["source_time_contract"])
    _exact(normalized_time, value["source_time_contract"], name + ".source_time_contract")
    _exact(
        normalized_time["evidence"]["kind"], domain, name + ".source_time_contract.evidence.kind"
    )
    time_sha = _sha(value["source_time_contract_sha256"], name + ".source_time_contract_sha256")
    _exact(
        time_sha,
        canonical_json_sha256(value["source_time_contract"]),
        name + ".source_time_contract_sha256",
    )

    identities = []
    for role, field in (("manifest", "source_manifest"), ("producer", "producer")):
        _file_identity(value[field])
        identities.append((role, value[field]))
    raw_files = value["raw_files"]
    if type(raw_files) is not list or not raw_files:
        raise ValueError(f"{name}.raw_files must be a nonempty list")
    raw_paths = set()
    for identity in raw_files:
        _file_identity(identity)
        if identity["path"] in raw_paths:
            raise ValueError("raw file identities must be unique")
        raw_paths.add(identity["path"])
        identities.append(("raw", identity))

    _keys(value["evidence"], FACT_GROUPS, name + ".evidence")
    locators = []
    for fact in FACT_GROUPS:
        fact_locators = value["evidence"][fact]
        if type(fact_locators) is not list or not fact_locators:
            raise ValueError(f"{name}.evidence.{fact} must be a nonempty list")
        for locator_index, locator in enumerate(fact_locators):
            _validate_locator(locator, f"{name}.evidence.{fact}[{locator_index}]")
            locators.append((fact, locator))
            identities.append(("evidence", locator["file"]))
    time_evidence = normalized_time["evidence"]
    if not any(
        fact == "timestamp"
        and locator["file"]["path"] == time_evidence["reference"]
        and locator["file"]["sha256"] == time_evidence["sha256"]
        for fact, locator in locators
    ):
        raise ValueError("timestamp evidence must include the source time contract reference")
    return site_id, site_name, locators, identities


def _validate_interpretation(value):
    _keys(value, INTERPRETATION_FIELDS, "interpretation")
    domain = value["domain"]
    if type(domain) is not str or domain not in _DOMAINS:
        raise ValueError("interpretation.domain is invalid")
    sites = value["sites"]
    if type(sites) is not list or not sites:
        raise ValueError("interpretation.sites must be a nonempty list")
    ids = []
    names = set()
    locators = []
    identities = []
    for index, site in enumerate(sites):
        site_id, site_name, site_locators, site_identities = _validate_site(site, domain, index)
        ids.append(site_id)
        if site_name in names:
            raise ValueError("site names must be unique")
        names.add(site_name)
        locators.extend(site_locators)
        identities.extend(site_identities)
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("site ids must be unique and strictly increasing")
    return value, locators, identities


def _validate_review(value, interpretation):
    _keys(value, REVIEW_FIELDS, "source interpretation review")
    _exact(
        value["artifact_kind"],
        "racepv.source-interpretation-review",
        "review.artifact_kind",
    )
    _exact(value["schema_version"], 1, "review.schema_version")
    _sha(value["interpretation_sha256"], "review.interpretation_sha256")
    _exact(
        value["interpretation_sha256"],
        canonical_json_sha256(interpretation),
        "review.interpretation_sha256",
    )
    _keys(value["reviewer"], ("id", "method"), "review.reviewer")
    _nonempty_string(value["reviewer"]["id"], "review.reviewer.id")
    _exact(
        value["reviewer"]["method"],
        "independent_semantic_review",
        "review.reviewer.method",
    )
    reviewed_at = value["reviewed_at"]
    if type(reviewed_at) is not str or _UTC_PATTERN.fullmatch(reviewed_at) is None:
        raise ValueError("review.reviewed_at must be canonical UTC")
    try:
        datetime.strptime(reviewed_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError("review.reviewed_at must be a valid calendar date") from exc

    decisions = value["site_decisions"]
    sites = interpretation["sites"]
    if type(decisions) is not list or len(decisions) != len(sites):
        raise ValueError("review decisions must exactly cover all sites")
    for index, (decision, site) in enumerate(zip(decisions, sites, strict=True)):
        _keys(decision, DECISION_FIELDS, f"review.site_decisions[{index}]")
        _exact(decision["site_id"], site["site"]["id"], f"review.site_decisions[{index}].site_id")
        _sha(decision["site_sha256"], f"review.site_decisions[{index}].site_sha256")
        _exact(
            decision["site_sha256"],
            canonical_json_sha256(site),
            f"review.site_decisions[{index}].site_sha256",
        )
        _keys(decision["facts"], FACT_GROUPS, f"review.site_decisions[{index}].facts")
        for fact in FACT_GROUPS:
            _exact(
                decision["facts"][fact],
                "accepted",
                f"review.site_decisions[{index}].facts.{fact}",
            )


def _existing_identity(root, relative):
    current = root
    parts = relative.split("/")
    for index, part in enumerate(parts):
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("declared file must not contain symlink components")
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("declared file parent must be a directory")
        if index == len(parts) - 1:
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("existing declared path must be a regular file")
            return metadata.st_dev, metadata.st_ino
    raise AssertionError("unreachable")


def _validate_identity_graph(root, role_identities):
    root = Path(root).resolve(strict=True)
    exact_paths = {}
    folded_paths = {}
    hash_roles = {}
    inodes = {}
    for role, identity in role_identities:
        _file_identity(identity)
        path = identity["path"]
        existing = exact_paths.get(path)
        signature = (identity["sha256"], identity["bytes"])
        if existing is not None and existing != (role, signature):
            raise ValueError("file path has conflicting identity or cross-role use")
        exact_paths[path] = role, signature
        folded = path.casefold()
        spelling = folded_paths.get(folded)
        if spelling is not None and spelling != path:
            raise ValueError("casefold file path alias is forbidden")
        folded_paths[folded] = path
        hash_role = hash_roles.get(identity["sha256"])
        if hash_role is not None and hash_role != role:
            raise ValueError("content identity cannot be reused across file roles")
        hash_roles[identity["sha256"]] = role
        inode = _existing_identity(root, path)
        if inode is not None:
            prior = inodes.get(inode)
            if prior is not None and prior != path:
                raise ValueError("existing same-inode aliases are forbidden")
            inodes[inode] = path


def _verify_evidence(root, locators):
    cache = {}
    identities = {}
    for _, locator in locators:
        identity = locator["file"]
        path_name = identity["path"]
        data = cache.get(path_name)
        if data is None:
            path = contained_artifact_file(root, path_name)
            data = path.read_bytes()
            _verify_bytes(identity, data)
            data.decode("utf-8")
            cache[path_name] = data
            identities[path_name] = identity
        lines = data.decode("utf-8").splitlines()
        if locator["end_line"] > len(lines):
            raise ValueError("evidence locator exceeds decoded text lines")
    return [identities[name] for name in sorted(identities)]
