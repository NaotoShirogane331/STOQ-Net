"""Immutable, externally pinned contracts for four nonformal candidate jobs."""

import copy
import hashlib
import json
import os
import stat
import sys
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

from racepv.cli.scoped_candidate import candidate_model_config_from_signature
from racepv.config import QUANTILE_LEVELS, _construct_config
from racepv.data.exploratory_candidate import NONFORMAL, inspect_inputs, validate_budget
from racepv.data.scoped_store import _reject_link_components
from racepv.training.recoverable_run import (
    _canonical_json,
    _fsync_directory,
    _rename_noreplace,
    _write_exclusive,
    seed_request_lock,
)
from racepv.training.recovery import (
    RecoveryIntegrityError,
    _open_absolute_directory,
    _require_sha256,
    parse_json_bytes,
    validate_relative_path,
)

JOBS = tuple((cell, seed) for cell in ("control", "A") for seed in (13, 29))
SCORE_ID = "qcrps_trapezoid_zero_endpoints_v1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def equal(actual, expected, role):
    if _canonical_json({"v": actual}) != _canonical_json({"v": expected}):
        raise RecoveryIntegrityError("exploratory " + role + " differs")


def safe_read(path, *, limit=32 * 1024 * 1024):
    """Bound reads and reject links, aliases, changed inodes and nonregular files."""
    path = Path(path).absolute()
    directory = _open_absolute_directory(path.parent)
    fd = None
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            raise RecoveryIntegrityError("unsafe or oversized regular file")
        parts, remaining = [], limit + 1
        while remaining:
            part = os.read(fd, min(1024 * 1024, remaining))
            if not part:
                break
            parts.append(part)
            remaining -= len(part)
        after = os.fstat(fd)
        entry = os.stat(path.name, dir_fd=directory, follow_symlinks=False)

        def identity(item):
            return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)

        data = b"".join(parts)
        if (
            len(data) > limit
            or identity(before) != identity(after)
            or identity(after) != identity(entry)
        ):
            raise RecoveryIntegrityError("input changed or exceeded read bound")
        return data
    except OSError as exc:
        raise RecoveryIntegrityError("unsafe or missing file: " + str(path)) from exc
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory)


def strict_json(data, role):
    value = parse_json_bytes(data, role)
    try:
        _canonical_json(value)
    except (ValueError, TypeError) as exc:
        raise RecoveryIntegrityError("nonfinite or invalid " + role) from exc
    return value


def sources():
    root = Path(__file__).resolve().parents[1]
    inventory = [
        {"path": path.relative_to(root).as_posix(), "sha256": digest(safe_read(path))}
        for path in sorted(root.rglob("*.py"))
    ]
    for name, module in tuple(sys.modules.items()):
        if name == "racepv" or name.startswith("racepv."):
            filename = getattr(module, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(root):
                raise RecoveryIntegrityError("imported source root differs")
    return inventory


def portable(metadata):
    result = {
        k: v
        for k, v in metadata.items()
        if k not in ("descriptor_path", "content_verified", "array_semantics_verified")
    }
    result["leaves"] = [
        {k: v for k, v in leaf.items() if k not in ("descriptor_path", "arrays")}
        for leaf in metadata["leaves"]
    ]
    result["files"] = [
        {k: v for k, v in item.items() if k != "resolved_path"} for item in metadata["files"]
    ]
    return result


def assumptions_bytes(report):
    expected, saved = report["evidence"]["sha256"], None
    for entry in report["inputs"].values():
        for leaf in entry["metadata"]["leaves"]:
            identity = leaf["schema"]["sources"]["time_evidence"]
            validate_relative_path(identity["path"])
            data = safe_read(Path(leaf["descriptor_path"]).parent / identity["path"])
            if digest(data) != expected or identity["sha256"] != expected:
                raise RecoveryIntegrityError("assumptions evidence identity differs")
            if len(data) != identity["bytes"]:
                raise RecoveryIntegrityError("assumptions evidence size differs")
            if saved is not None and saved != data:
                raise RecoveryIntegrityError("assumptions evidence bytes differ")
            saved = data
    return saved.decode("utf-8")


def _payload(shared, root, cell, seed):
    return dict(
        schema_version=1,
        purpose="racepv.exploratory-binding.v1",
        campaign_root=str(root),
        cell=cell,
        seed=seed,
        shared=shared,
        **NONFORMAL,
    )


@dataclass(frozen=True)
class Binding:
    raw_bytes: bytes
    sha256: str

    @property
    def raw(self):
        return json.loads(self.raw_bytes)

    @property
    def run_root(self):
        value = self.raw
        return Path(value["campaign_root"]) / f"{value['cell']}-s{value['seed']}"

    @property
    def experiment(self):
        raw = self.raw
        selected = copy.deepcopy(raw["shared"]["experiment"])
        selected["training"]["seed"] = raw["seed"]
        return _construct_config(selected)


def create_campaign(args, campaign_root, *, data_root, experiment, synthetic_test=False):
    """Metadata-only atomic freeze of all four bindings; refuses an existing campaign."""
    report = inspect_inputs(args, experiment=experiment, synthetic_test=synthetic_test)
    root = _reject_link_components(Path(campaign_root).absolute())
    data_root = _reject_link_components(Path(data_root).absolute())
    if root == data_root or root.is_relative_to(data_root) or data_root.is_relative_to(root):
        raise RecoveryIntegrityError("campaign and data roots must not overlap")
    if root.exists():
        raise RecoveryIntegrityError("campaign already exists")
    inputs = {}
    for split, entry in report["inputs"].items():
        path = Path(entry["specification"]["descriptor_path"]).absolute()
        relative = path.relative_to(data_root).as_posix()
        validate_relative_path(relative)
        inputs[split] = dict(
            path=relative,
            descriptor_sha256=entry["metadata"]["descriptor_sha256"],
            read_scope_sha256=entry["specification"]["expected_read_scope_sha256"],
            metadata=portable(entry["metadata"]),
        )
    if inputs["train"]["path"].casefold() == inputs["validation"]["path"].casefold():
        raise RecoveryIntegrityError("descriptor roles alias")
    experiment = replace(experiment, training=replace(experiment.training, seed=13))
    model = candidate_model_config_from_signature(
        report["feature_signature"], experiment, backbone="gru" if synthetic_test else "mamba"
    )
    shared = dict(
        domain="synthetic_fixture" if synthetic_test else "research_assumption",
        data_root=str(data_root),
        inputs=inputs,
        evidence=report["evidence"],
        assumptions_utf8=assumptions_bytes(report),
        feature_signature=report["feature_signature"],
        experiment=experiment.to_dict(),
        model_config=asdict(model),
        source_inventory=sources(),
        score_id=SCORE_ID,
        selection_metric="daylight_qcrps",
        **NONFORMAL,
    )
    # Parent must already exist: no arbitrary recursively created output hierarchy.
    _reject_link_components(root.parent)
    with seed_request_lock(root):
        staging = root.parent / ("." + root.name + ".manifest-" + uuid.uuid4().hex)
        staging.mkdir(mode=0o700)
        (staging / "bindings").mkdir()
        jobs, result = {}, {}
        for cell, seed in JOBS:
            name = f"{cell}-s{seed}"
            raw = _canonical_json(_payload(shared, root, cell, seed))
            filename = name + ".json"
            _write_exclusive(staging / "bindings" / filename, raw)
            jobs[name] = digest(raw)
            result[name] = dict(path=root / "bindings" / filename, sha256=digest(raw))
        manifest = dict(
            schema_version=1,
            purpose="racepv.exploratory-campaign.v1",
            shared_sha256=digest(_canonical_json(shared)),
            jobs=jobs,
        )
        _write_exclusive(staging / "campaign.json", _canonical_json(manifest))
        _fsync_directory(staging / "bindings")
        _fsync_directory(staging)
        _rename_noreplace(staging, root)
        _fsync_directory(root.parent)
    return result


def open_binding(path, expected_sha256, *, synthetic_test=False):
    _require_sha256(expected_sha256, "binding SHA256")
    raw = safe_read(path)
    if digest(raw) != expected_sha256:
        raise RecoveryIntegrityError("external binding SHA256 differs")
    value = strict_json(raw, "exploratory binding")
    keys = {"schema_version", "purpose", "campaign_root", "cell", "seed", "shared", *NONFORMAL}
    if (
        set(value) != keys
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
    ):
        raise RecoveryIntegrityError("binding schema differs")
    if value["purpose"] != "racepv.exploratory-binding.v1":
        raise RecoveryIntegrityError("binding purpose differs")
    if type(value["seed"]) is not int or (value["cell"], value["seed"]) not in JOBS:
        raise RecoveryIntegrityError("job outside four-job campaign budget")
    shared = value["shared"]
    expected_domain = "synthetic_fixture" if synthetic_test else "research_assumption"
    if not isinstance(shared, dict) or shared.get("domain") != expected_domain:
        raise RecoveryIntegrityError("binding evidence domain differs")
    for owner in (value, shared):
        for flag in NONFORMAL:
            if owner.get(flag) is not False:
                raise RecoveryIntegrityError("nonformal identity differs")
    required = {
        "domain",
        "data_root",
        "inputs",
        "evidence",
        "assumptions_utf8",
        "feature_signature",
        "experiment",
        "model_config",
        "source_inventory",
        "score_id",
        "selection_metric",
        *NONFORMAL,
    }
    if set(shared) != required:
        raise RecoveryIntegrityError("shared contract keys differ")
    equal(shared["source_inventory"], sources(), "executing source inventory")
    if shared["score_id"] != SCORE_ID or shared["selection_metric"] != "daylight_qcrps":
        raise RecoveryIntegrityError("score/selection identity differs")
    experiment = _construct_config(copy.deepcopy(shared["experiment"]))
    validate_budget(experiment, synthetic_test=synthetic_test)
    equal(experiment.quantiles, QUANTILE_LEVELS, "quantile geometry")
    equal(experiment.to_dict(), shared["experiment"], "complete experiment")
    model = candidate_model_config_from_signature(
        shared["feature_signature"], experiment, backbone="gru" if synthetic_test else "mamba"
    )
    equal(asdict(model), shared["model_config"], "model configuration")
    if (
        shared["evidence"].get("kind") != expected_domain
        or digest(shared["assumptions_utf8"].encode("utf-8")) != shared["evidence"]["sha256"]
    ):
        raise RecoveryIntegrityError("assumptions evidence differs")
    if set(shared["inputs"]) != {"train", "validation"}:
        raise RecoveryIntegrityError("train/validation scope differs")
    roles = []
    for entry in shared["inputs"].values():
        if set(entry) != {"path", "descriptor_sha256", "read_scope_sha256", "metadata"}:
            raise RecoveryIntegrityError("input pin keys differ")
        validate_relative_path(entry["path"])
        roles.append(entry["path"].casefold())
        for key in ("descriptor_sha256", "read_scope_sha256"):
            _require_sha256(entry[key], key)
    if len(set(roles)) != 2:
        raise RecoveryIntegrityError("input roles alias")
    root = _reject_link_components(Path(value["campaign_root"]))
    if not root.is_absolute():
        raise RecoveryIntegrityError("campaign root must be absolute")
    jobs = {
        f"{cell}-s{seed}": digest(_canonical_json(_payload(shared, root, cell, seed)))
        for cell, seed in JOBS
    }
    expected_manifest = dict(
        schema_version=1,
        purpose="racepv.exploratory-campaign.v1",
        shared_sha256=digest(_canonical_json(shared)),
        jobs=jobs,
    )
    equal(
        strict_json(safe_read(root / "campaign.json"), "campaign"),
        expected_manifest,
        "frozen campaign",
    )
    name = f"{value['cell']}-s{value['seed']}"
    canonical = safe_read(root / "bindings" / (name + ".json"))
    if canonical != raw or jobs[name] != expected_sha256:
        raise RecoveryIntegrityError("fixed campaign binding differs")
    return Binding(raw, expected_sha256)


def preflight(binding, *, data_root=None, synthetic_test=False):
    """Revalidate portable metadata on the original or explicitly migrated root."""
    shared = binding.raw["shared"]
    root = _reject_link_components(Path(data_root or shared["data_root"]).absolute())
    args = SimpleNamespace(
        expected_prepared_schema_version=3,
        expected_causality_profile="as-of-origin-v1",
        validation_only=True,
    )
    for split, entry in shared["inputs"].items():
        setattr(args, split, root / entry["path"])
        setattr(args, f"prepared_{split}_descriptor_sha256", entry["descriptor_sha256"])
        setattr(args, f"prepared_{split}_read_scope_sha256", entry["read_scope_sha256"])
    report = inspect_inputs(args, experiment=binding.experiment, synthetic_test=synthetic_test)
    equal(report["feature_signature"], shared["feature_signature"], "features")
    equal(report["evidence"], shared["evidence"], "evidence")
    equal(assumptions_bytes(report), shared["assumptions_utf8"], "assumptions bytes")
    for split, entry in report["inputs"].items():
        equal(
            portable(entry["metadata"]),
            shared["inputs"][split]["metadata"],
            "portable " + split + " metadata",
        )
    return args, report


def output_metadata(binding):
    value, shared = binding.raw, binding.raw["shared"]
    from racepv.models.reliability_a import MODEL_REVISION
    from racepv.reliability_a_loss import ReliabilityAObjective
    from racepv.reliability_experiment import CONTROL_REVISION, PAIR_PROTOCOL

    return dict(
        binding_sha256=binding.sha256,
        cell=value["cell"],
        seed=value["seed"],
        model_config=shared["model_config"],
        experiment=binding.experiment.to_dict(),
        model_revision=MODEL_REVISION if value["cell"] == "A" else CONTROL_REVISION,
        objective_version=ReliabilityAObjective.objective_id,
        experiment_protocol=PAIR_PROTOCOL,
        source_inventory=shared["source_inventory"],
        inputs=shared["inputs"],
        evidence=shared["evidence"],
        domain=shared["domain"],
        score_id=SCORE_ID,
        selection_metric="daylight_qcrps",
        primary_comparison="overall_qcrps",
        overall_mask_id="overall_eligible_v1",
        daylight_mask_id="daylight_eligible_v1",
        target_cell_identity=["site_ids", "issue_ns", "target_interval_start_ns"],
        **NONFORMAL,
    )
