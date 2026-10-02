"""Independently pinned research-assumption campaigns; never qualified evidence."""

import copy
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

from racepv.baselines.research_registry import get_research_spec
from racepv.cli.scoped_candidate import candidate_model_config_from_signature
from racepv.config import DataConfig, ExperimentConfig
from racepv.data.exploratory_candidate import NONFORMAL
from racepv.data.scoped_store import _reject_link_components
from racepv.training.exploratory_contract import (
    SCORE_ID,
    assumptions_bytes,
    digest,
    equal,
    portable,
    safe_read,
    sources,
    strict_json,
)
from racepv.training.recoverable_run import (
    _canonical_json,
    _fsync_directory,
    _rename_noreplace,
    _write_exclusive,
    seed_request_lock,
)
from racepv.training.recovery import RecoveryIntegrityError, _require_sha256, validate_relative_path
from racepv.training.research_baseline_bridge import build_research_neural, inspect_research_inputs
from racepv.training.research_environment import runtime_inventory

DEFAULT_BUDGET = dict(max_epochs=60, patience=8, min_epochs=10, batch_size=64,
                      num_workers=0, learning_rate=1e-3, weight_decay=1e-4,
                      gradient_clip_norm=1., mixed_precision=True)
CANDIDATES = ("F", "C", "M1", "M3")


def resolve_budget(overrides=None, *, synthetic_test=False):
    supplied = dict(overrides or {})
    if set(supplied) - set(DEFAULT_BUDGET):
        raise ValueError("unknown budget fields")
    value = dict(DEFAULT_BUDGET, **supplied)
    if synthetic_test:
        value["mixed_precision"] = False
    for key in ("max_epochs", "patience", "min_epochs", "batch_size", "num_workers"):
        if type(value[key]) is not int:
            raise ValueError("budget integers required")
    if not (1 <= value["min_epochs"] <= value["max_epochs"] <= 60 and
            1 <= value["patience"] <= 60 and 1 <= value["batch_size"] <= 64 and value["num_workers"] == 0):
        raise ValueError("budget outside allowed range")
    if not synthetic_test and any(value[k] != DEFAULT_BUDGET[k] for k in
                                  ("max_epochs", "patience", "min_epochs", "batch_size", "mixed_precision")):
        raise ValueError("formal research budget requires 60/8/10/64 and AMP")
    if value["learning_rate"] not in (5e-4, 1e-3, 2e-3):
        raise ValueError("learning rate outside protocol candidates")
    if value["weight_decay"] != 1e-4 or value["gradient_clip_norm"] != 1.:
        raise ValueError("optimizer contract differs")
    _canonical_json(value)
    return value


def candidate_config(signature, *, synthetic_test):
    return candidate_model_config_from_signature(signature, ExperimentConfig(data=DataConfig(stride=4)),
        backbone="gru" if synthetic_test else "mamba")


def build_components(job, signature, *, synthetic_test):
    """Construct the requested implementation; unavailable dependencies fail closed."""
    model_id, config = job["model_id"], copy.deepcopy(job["config"])
    if model_id in CANDIDATES:
        if config:
            raise ValueError("candidate overrides are not admitted")
        cfg = candidate_config(signature, synthetic_test=synthetic_test)
        if model_id in ("F", "C"):
            from racepv.reliability_experiment import make_reliability_components
            model, objective = make_reliability_components(cfg, "A" if model_id == "F" else "control")
        else:
            from racepv.models.research_variants import build_research_candidate
            model, objective = build_research_candidate(model_id, cfg)
        identity = dict(model_id=model_id, resolved_config=asdict(model.config), model_class=type(model).__qualname__,
                        model_module=type(model).__module__, model_metadata=model.model_metadata)
        return model, objective, json.loads(_canonical_json(identity))
    spec = get_research_spec(model_id)
    if spec["training_kind"] != "supervised" or model_id in ("quantile_lightgbm", "random_forest_mondrian"):
        raise RuntimeError("NOT_READY: non-neural batch lifecycle is not implemented")
    if model_id in ("nhits", "tft", "xlstm"):
        if "random_seed" in config and config["random_seed"] != job["seed"]:
            raise ValueError("official factory random_seed differs from job seed")
        config["random_seed"] = job["seed"]
    model = build_research_neural(model_id, signature, config)
    return model, None, model.research_identity


@dataclass(frozen=True)
class ResearchBinding:
    path: Path
    raw_bytes: bytes
    sha256: str
    campaign_sha256: str

    @property
    def raw(self):
        return json.loads(self.raw_bytes)

    @property
    def run_root(self):
        return self.path.parent.parent / self.raw["name"]


def create_campaign(args, campaign_root, *, data_root, jobs=None, budget=None,
                    synthetic_test=False, assumptions_sha256=None):
    """Inspect metadata and resolve models; no arrays, forward calls or optimization."""
    budget = resolve_budget(budget, synthetic_test=synthetic_test)
    report = inspect_research_inputs(args, synthetic_test=synthetic_test)
    if assumptions_sha256 is not None:
        equal(report["evidence"]["sha256"], assumptions_sha256, "assumptions external pin")
    elif not synthetic_test:
        raise ValueError("external assumptions SHA256 required")
    root = _reject_link_components(Path(campaign_root).absolute())
    data_root = _reject_link_components(Path(data_root).absolute())
    if root == data_root or root.is_relative_to(data_root) or data_root.is_relative_to(root):
        raise ValueError("campaign and data roots overlap")
    if root.exists():
        raise RecoveryIntegrityError("campaign already exists")
    inputs = {}
    for split, entry in report["inputs"].items():
        relative = Path(entry["specification"]["descriptor_path"]).absolute().relative_to(data_root).as_posix()
        validate_relative_path(relative)
        inputs[split] = dict(path=relative, descriptor_sha256=entry["metadata"]["descriptor_sha256"],
            read_scope_sha256=entry["specification"]["expected_read_scope_sha256"], metadata=portable(entry["metadata"]))
    shared = dict(domain="synthetic_fixture" if synthetic_test else "research_assumption",
        inputs=inputs, evidence=report["evidence"], assumptions_utf8=assumptions_bytes(report),
        feature_signature=report["feature_signature"], source_inventory=sources(), budget=budget,
        runtime_inventory=runtime_inventory(),
        device="cpu" if synthetic_test else "cuda", precision="float32" if synthetic_test else "amp_float16",
        score_id=SCORE_ID, selection_metric="daylight_qcrps", postprocessing="clip_0_1.05_night_zero_ordered_v1",
        **NONFORMAL)
    jobs = jobs or [dict(model_id=m, seed=s, config={}) for m in ("F", "C") for s in (13, 29, 47)]
    payloads = {}
    import torch
    for job in jobs:
        if set(job) != {"model_id", "seed", "config"} or type(job["seed"]) is not int or job["seed"] not in (13, 29, 47):
            raise ValueError("job keys or seed outside research contract")
        if not isinstance(job["config"], dict):
            raise ValueError("job config must be a mapping")
        if job["model_id"] not in CANDIDATES:
            get_research_spec(job["model_id"])
        name = f"{job['model_id']}-s{job['seed']}"
        validate_relative_path(name)
        if name in payloads:
            raise ValueError("duplicate model/seed job")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(job["seed"])
            _, _, identity = build_components(job, report["feature_signature"], synthetic_test=synthetic_test)
        payloads[name] = _canonical_json(dict(schema_version=1, purpose="racepv.research-job.v1",
            name=name, job=job, model_identity=identity, shared=shared, **NONFORMAL))
    with seed_request_lock(root):
        stage = root.parent / ("." + root.name + ".campaign-" + uuid.uuid4().hex)
        stage.mkdir(mode=0o700)
        (stage / "bindings").mkdir()
        pins = {}
        for name, raw in payloads.items():
            _write_exclusive(stage / "bindings" / (name + ".json"), raw)
            pins[name] = dict(path=root / "bindings" / (name + ".json"), sha256=digest(raw))
        manifest = _canonical_json(dict(schema_version=1, purpose="racepv.research-campaign.v1",
            jobs={name: digest(raw) for name, raw in payloads.items()}))
        _write_exclusive(stage / "campaign.json", manifest)
        _fsync_directory(stage / "bindings")
        _fsync_directory(stage)
        _rename_noreplace(stage, root)
        _fsync_directory(root.parent)
    return dict(campaign_sha256=digest(manifest), jobs=pins, metadata_only=True,
                data_root=str(data_root), **NONFORMAL)


def open_binding(path, expected_sha256, *, campaign_sha256, synthetic_test=False):
    path = Path(path).absolute()
    for sha in (expected_sha256, campaign_sha256):
        _require_sha256(sha, "external SHA256")
    raw = safe_read(path)
    if digest(raw) != expected_sha256:
        raise RecoveryIntegrityError("binding external SHA256 differs")
    value = strict_json(raw, "research job")
    if set(value) != {"schema_version", "purpose", "name", "job", "model_identity", "shared", *NONFORMAL} or value["schema_version"] != 1 or value["purpose"] != "racepv.research-job.v1":
        raise RecoveryIntegrityError("job schema differs")
    shared, job = value["shared"], value["job"]
    if set(shared) != {"domain", "inputs", "evidence", "assumptions_utf8", "feature_signature", "source_inventory", "runtime_inventory", "budget", "device", "precision", "score_id", "selection_metric", "postprocessing", *NONFORMAL}:
        raise RecoveryIntegrityError("shared schema differs")
    if shared["domain"] != ("synthetic_fixture" if synthetic_test else "research_assumption"):
        raise ValueError("research domain differs")
    for owner in (value, shared):
        if any(owner[k] is not False for k in NONFORMAL):
            raise RecoveryIntegrityError("nonformal flags differ")
    equal(resolve_budget(shared["budget"], synthetic_test=synthetic_test), shared["budget"], "budget")
    equal(shared["source_inventory"], sources(), "source inventory")
    equal(shared["runtime_inventory"], runtime_inventory(), "runtime inventory")
    if set(job) != {"model_id", "seed", "config"} or type(job["seed"]) is not int or job["seed"] not in (13, 29, 47):
        raise RecoveryIntegrityError("job contract differs")
    if job["model_id"] not in CANDIDATES:
        get_research_spec(job["model_id"])
    equal(value["name"], f"{job['model_id']}-s{job['seed']}", "job name")
    if path.name != value["name"] + ".json" or path.parent.name != "bindings":
        raise RecoveryIntegrityError("canonical binding path differs")
    if set(shared["inputs"]) != {"train", "validation"}:
        raise RecoveryIntegrityError("train/validation scope differs")
    for item in shared["inputs"].values():
        if set(item) != {"path", "descriptor_sha256", "read_scope_sha256", "metadata"}:
            raise RecoveryIntegrityError("input schema differs")
        validate_relative_path(item["path"])
        for key in ("descriptor_sha256", "read_scope_sha256"):
            _require_sha256(item[key], key)
    if digest(shared["assumptions_utf8"].encode()) != shared["evidence"]["sha256"]:
        raise RecoveryIntegrityError("assumptions pin differs")
    equal([shared["device"], shared["precision"], shared["score_id"], shared["selection_metric"], shared["postprocessing"]],
          ["cpu" if synthetic_test else "cuda", "float32" if synthetic_test else "amp_float16", SCORE_ID, "daylight_qcrps", "clip_0_1.05_night_zero_ordered_v1"], "runtime protocol")
    manifest_raw = safe_read(path.parent.parent / "campaign.json")
    if digest(manifest_raw) != campaign_sha256:
        raise RecoveryIntegrityError("campaign external SHA256 differs")
    manifest = strict_json(manifest_raw, "campaign")
    if set(manifest) != {"schema_version", "purpose", "jobs"} or manifest["schema_version"] != 1 or manifest["purpose"] != "racepv.research-campaign.v1" or manifest["jobs"].get(value["name"]) != expected_sha256:
        raise RecoveryIntegrityError("campaign membership differs")
    return ResearchBinding(path, raw, expected_sha256, campaign_sha256)


def refresh(binding, *, synthetic_test=False):
    return open_binding(binding.path, binding.sha256, campaign_sha256=binding.campaign_sha256,
                        synthetic_test=synthetic_test)


def preflight(binding, *, data_root, synthetic_test=False):
    binding = refresh(binding, synthetic_test=synthetic_test)
    root = _reject_link_components(Path(data_root).absolute())
    shared = binding.raw["shared"]
    args = SimpleNamespace(expected_prepared_schema_version=3,
                           expected_causality_profile="as-of-origin-v1", validation_only=True)
    for split, item in shared["inputs"].items():
        setattr(args, split, root / item["path"])
        setattr(args, f"prepared_{split}_descriptor_sha256", item["descriptor_sha256"])
        setattr(args, f"prepared_{split}_read_scope_sha256", item["read_scope_sha256"])
    report = inspect_research_inputs(args, synthetic_test=synthetic_test)
    for key in ("evidence", "feature_signature"):
        equal(report[key], shared[key], key)
    equal(assumptions_bytes(report), shared["assumptions_utf8"], "assumptions")
    for split, entry in report["inputs"].items():
        equal(portable(entry["metadata"]), shared["inputs"][split]["metadata"], "portable metadata")
    return args, report


def output_metadata(binding):
    return dict(binding_sha256=binding.sha256, campaign_sha256=binding.campaign_sha256,
                job=binding.raw["job"], model_identity=binding.raw["model_identity"],
                shared=binding.raw["shared"], **NONFORMAL)
