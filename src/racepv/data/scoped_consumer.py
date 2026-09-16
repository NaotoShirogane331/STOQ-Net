"""Pinned schema3 pair preflight and explicitly nonformal engineering loading."""

import copy
import json

from racepv.data import prepared_contract as contract
from racepv.data import scoped_store
from racepv.data.scoped_chronology import require_scoped_train_validation_chronology
from racepv.data.scoped_scientific_view import (
    require_train_validation_compatibility,
    scientific_view_from_metadata,
)


class ScopedQualificationRequired(ValueError):
    """Preserve the completed metadata diagnosis without granting admission."""

    def __init__(self, report):
        super().__init__(
            "schema3 source/extraction qualification is required before formal loading"
        )
        self._report_json = json.dumps(report, sort_keys=True, allow_nan=False)

    @property
    def report(self):
        return json.loads(self._report_json)


def _specifications(args):
    if (
        type(getattr(args, "expected_prepared_schema_version", None)) is not int
        or args.expected_prepared_schema_version != 3
        or getattr(args, "expected_causality_profile", None) != "as-of-origin-v1"
    ):
        raise ValueError("expected schema3 and as-of-origin-v1 must be explicit")
    if getattr(args, "validation_only", False) is not True or any(
        getattr(args, name, None) is not None for name in ("calibration", "test")
    ):
        raise ValueError("schema3 consumer scope is train/validation only")
    if (
        any(getattr(args, name, False) for name in ("allow_ineligible", "allow_development_data"))
        or getattr(args, "verify_content", True) is not True
    ):
        raise ValueError("schema3 consumer forbids evidence/content bypass")
    specifications = {}
    for split in ("train", "validation"):
        path = getattr(args, split, None)
        if path is None:
            raise ValueError(f"missing {split} descriptor")
        descriptor = getattr(args, f"prepared_{split}_descriptor_sha256", None)
        scope = getattr(args, f"prepared_{split}_read_scope_sha256", None)
        contract._sha(descriptor, f"{split} expected descriptor SHA256")
        contract._sha(scope, f"{split} expected read scope SHA256")
        specifications[split] = {
            "descriptor_path": str(path),
            "expected_descriptor_sha256": descriptor,
            "expected_read_scope_sha256": scope,
            "allowed_splits": ("train",) if split == "train" else ("train", "validation"),
        }
    return specifications


def _view(metadata):
    return scientific_view_from_metadata(
        metadata,
        selected_split=metadata["selected_split"],
        expected_metadata_sha256=contract.canonical_json_sha256(metadata),
    )


def inspect_scoped_consumer_inputs(args):
    """Inspect both pinned graphs and scientific compatibility before any content."""
    specs = _specifications(args)  # Validate all four external pins before any file open.
    inputs, views = {}, {}
    for split, spec in specs.items():
        metadata = scoped_store.inspect_scoped_store(**spec)
        if metadata["selected_split"] != split:
            raise ValueError("descriptor selected split differs from requested role")
        current = _view(metadata)
        if not set(current.spatial_roles.values()).issubset({"standard", "source_train"}):
            raise ValueError("training/selection cannot consume target-site roles")
        views[split] = current
        inputs[split] = {
            "specification": {**spec, "allowed_splits": list(spec["allowed_splits"])},
            "metadata": metadata,
            "metadata_sha256": current.metadata_sha256,
            "view_identity_sha256": current.identity_sha256,
            "site_id_map": current.site_id_map,
        }
    require_train_validation_compatibility(views["train"], views["validation"])
    return {
        "artifact_kind": "racepv.scoped-consumer-preflight",
        "schema_version": 1,
        "prepared_schema_version": 3,
        "inputs": inputs,
        "content_verified": False,
        "array_semantics_verified": False,
        "source_interpretation_verified": False,
        "registered_evidence_eligible": False,
        "formal_ready": False,
    }


def load_scoped_engineering_inputs(args, *, expected_preflight_sha256):
    """Full local pair verification; never used to authorize formal training.

    The caller must separately authorize the declared byte scopes. Returned arrays
    belong to the caller. This synthetic-development path cannot grant source or
    extraction qualification, even after every array check succeeds.
    """
    contract._sha(expected_preflight_sha256, "expected preflight SHA256")
    report = inspect_scoped_consumer_inputs(args)
    if contract.canonical_json_sha256(report) != expected_preflight_sha256:
        raise ValueError("preflight snapshot identity drift")
    stores, views, audits = {}, {}, {}
    for split, entry in report["inputs"].items():
        spec = dict(entry["specification"])
        spec["allowed_splits"] = tuple(spec["allowed_splits"])
        loaded = scoped_store.load_scoped_store(**spec)
        snapshot = copy.deepcopy(loaded["metadata"])
        snapshot["content_verified"] = snapshot["array_semantics_verified"] = False
        if contract.canonical_json_sha256(snapshot) != entry["metadata_sha256"]:
            raise ValueError("metadata changed during scoped loading")
        stores[split] = loaded
        views[split] = _view(snapshot)
        audits[split] = {
            str(leaf["schema"]["site"]["id"]): arrays
            for leaf, arrays in zip(snapshot["leaves"], loaded["causal_arrays"], strict=True)
        }
    chronology = require_scoped_train_validation_chronology(
        views["train"], audits["train"], views["validation"], audits["validation"]
    )
    return {
        "preflight": report,
        "preflight_sha256": expected_preflight_sha256,
        "stores": stores,
        "views": views,
        "chronology": chronology,
        "formal_ready": False,
    }
