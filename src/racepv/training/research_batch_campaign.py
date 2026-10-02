"""Metadata-only, independently pinned non-epoch research job contracts."""

import importlib.metadata
import inspect
import uuid
from pathlib import Path
from types import SimpleNamespace

from racepv.baselines.research_registry import QUANTILES, get_research_spec
from racepv.data.exploratory_candidate import NONFORMAL
from racepv.data.scoped_store import _reject_link_components
from racepv.training.exploratory_contract import (
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
from racepv.training.research_baseline_bridge import _signature, inspect_research_inputs
from racepv.training.research_campaign import ResearchBinding

RULES = ("clear_sky_persistence", "smart_persistence", "conditional_empirical_quantiles")
TREES = ("quantile_lightgbm", "random_forest_mondrian")
FOUNDATIONS = ("timesfm3_zs", "sundial_base_zs", "moirai_1_1_r_small_zs")
MODELS = RULES + TREES + FOUNDATIONS
DEFAULT_RESOURCES = dict(batch_size=64, scratch_reserve_bytes=1024**3, model_file_limit_bytes=64 * 1024**3)


def resources(**kwargs):
    value = dict(DEFAULT_RESOURCES, **kwargs)
    if set(value) != set(DEFAULT_RESOURCES) or any(type(v) is not int for v in value.values()):
        raise ValueError("resource fields must be exact integers")
    if not 1 <= value["batch_size"] <= 4096 or value["scratch_reserve_bytes"] < 0 or not 1024 <= value["model_file_limit_bytes"] <= 1024**4:
        raise ValueError("resource limits outside admitted range")
    return value


def dependency_versions(model_id):
    names = ["numpy", "pandas", "torch"]
    if model_id in TREES:
        names += ["joblib", "lightgbm" if model_id == TREES[0] else "scikit-learn"]
    if model_id in FOUNDATIONS:
        names += ["safetensors", "transformers", "timesfm" if model_id == FOUNDATIONS[0]
                  else "uni2ts" if model_id == FOUNDATIONS[2] else "huggingface-hub"]
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def freeze_job(job):
    if not isinstance(job, dict) or set(job) != {"model_id", "seed", "config"}:
        raise ValueError("job requires model_id, seed, config")
    if job["model_id"] not in MODELS or not isinstance(job["config"], dict):
        raise ValueError("unknown, excluded, or non-batch model/config")
    result = dict(job, config=dict(job["config"]))
    if job["model_id"] in FOUNDATIONS:
        config = result["config"]
        expected = {f"{kind}_{field}" for kind in ("source_review", "weight_receipt") for field in ("path", "sha256")}
        if set(config) != expected:
            raise ValueError("foundation requires explicitly pinned source review and weight receipt paths")
        for kind in ("source_review", "weight_receipt"):
            _require_sha256(config[kind + "_sha256"], kind)
            raw = safe_read(Path(config.pop(kind + "_path")).absolute())
            equal(digest(raw), config[kind + "_sha256"], kind + " external pin")
            config[kind + "_utf8"] = raw.decode("utf8")
    return result


def _effective_tree_config(name, options):
    """Read installed constructor defaults without constructing an estimator."""
    if name == "quantile_lightgbm":
        from lightgbm import LGBMRegressor
        factory = LGBMRegressor
    else:
        from sklearn.ensemble import RandomForestRegressor
        factory = RandomForestRegressor
    parameters = inspect.signature(factory).bind(**options)
    parameters.apply_defaults()
    resolved = dict(parameters.arguments)
    resolved.update(resolved.pop("kwargs", {}))
    # Objective and alpha are pinned per quantile subestimator, never overridden.
    if name == "quantile_lightgbm":
        resolved.pop("objective", None)
    return resolved


def resolve_identity(job, signature, *, synthetic_test=False):
    """Pure metadata resolution: never constructs an estimator, adapter or model."""
    if set(job) != {"model_id", "seed", "config"} or job["model_id"] not in MODELS:
        raise ValueError("batch job identity differs")
    name, seed, supplied = job["model_id"], job["seed"], job["config"]
    if type(seed) is not int or seed not in ((13, 29, 47) if name in TREES else (13,)):
        raise ValueError("fixed rule/foundation seed13 or tree seed13/29/47 required")
    if not isinstance(supplied, dict):
        raise ValueError("config must be a mapping")
    _signature(signature)
    spec = get_research_spec(name)
    effective = dict(spec["default_config"])
    identity = dict(model_id=name, prediction_kind=spec["prediction_kind"],
        quantiles=list(QUANTILES), feature_signature=signature,
        dependency_versions=dependency_versions(name),
        input_information=spec["input_information"],
        history_missingness="zero_maskfalse_values",
        postprocessing="finite_clip_0_1.05_night_zero_monotone_rearrangement_v1",
        recovery_boundary="whole_task", **NONFORMAL)
    if name in TREES:
        field, candidates = ("num_leaves", (15, 31, 63)) if name == TREES[0] else ("min_samples_leaf", (1, 2, 4))
        allowed = {field} | ({"n_estimators"} if synthetic_test else set())
        if set(supplied) - allowed:
            raise ValueError("tree config outside fixed protocol candidate fields")
        effective.update(supplied, random_state=seed)
        if type(effective[field]) is not int or effective[field] not in candidates:
            raise ValueError("tree candidate outside protocol")
        if type(effective["n_estimators"]) is not int or not 1 <= effective["n_estimators"] <= spec["default_config"]["n_estimators"]:
            raise ValueError("invalid estimator count")
        effective = _effective_tree_config(name, effective)
        identity.update(recovery_boundary="subestimator", feature_dimension=5391,
            feature_order="history_C_flat,history_mask_C_flat,site_metadata,lead_future_known",
            fit_population="train_target_mask_and_daylight_only", strategy="24_independent_leads",
            selection_metric="daylight_qcrps" if name == TREES[0] else "daylight_mae",
            candidate_selection="external_whole_package; repeat_seeds_reuse_fixed_candidate",
            selection_tiebreak="smaller_min_samples_leaf" if name == TREES[1] else "external_protocol",
            calibration="none", early_stopping=(dict(metric="quantile", patience=50,
                population="validation_target_mask_and_daylight", best_iteration="per_subestimator")
                if name == TREES[0] else None))
    elif name in FOUNDATIONS:
        expected = {f"{kind}_{field}" for kind in ("source_review", "weight_receipt") for field in ("utf8", "sha256")}
        if set(supplied) != expected:
            raise ValueError("frozen foundation receipt schema differs")
        from racepv.baselines.foundation_extension import _authorize
        frozen = _authorize(name, horizon=24, **{
            kind + suffix: supplied[kind + ("_utf8" if suffix == "_bytes" else suffix)].encode("utf8")
            if suffix == "_bytes" else supplied[kind + suffix]
            for kind in ("source_review", "weight_receipt") for suffix in ("_bytes", "_sha256")})
        identity.update(frozen_adapter_identity=frozen, live_weights_verified=False,
            actual_gpu_verified=False, history_missingness="retain_finite_origin_causal_power",
            input_information="history_target_only", recovery_boundary="fixed_validation_batch",
            mapping_version="schema3_origin_history_power_only_v1", device="cpu" if synthetic_test else "cuda")
        effective = dict(sampling_contract=frozen["sampling_contract"], horizon=24, frozen=True, offline=True)
    else:
        if supplied:
            raise ValueError("rule/empirical config is fixed")
        if name == "smart_persistence":
            effective = dict(version="protocol_B0_train_raw_fallback_v1", lookback_steps=96,
                clear_sky_min_exclusive=.001, index_min=0., index_max=2.,
                fallback="train_unique_site_local_hour_median_then_site_median",
                timezone="Australia/Darwin", history_time="issue_ns-(192-index)*900000000000",
                support="both_raw_observation_masks_and_clear_sky_gt_0.001",
                fallback_population="complete_train_split_raw_rows_deduplicated_site_timestamp",
                training_kind="train_reference_statistics_without_optimizer")
    identity["resolved_config"] = effective
    return identity


def _freeze_inputs(report, root):
    inputs = {}
    for split, entry in report["inputs"].items():
        relative = Path(entry["specification"]["descriptor_path"]).absolute().relative_to(root).as_posix()
        validate_relative_path(relative)
        inputs[split] = dict(path=relative, descriptor_sha256=entry["metadata"]["descriptor_sha256"],
            read_scope_sha256=entry["specification"]["expected_read_scope_sha256"], metadata=portable(entry["metadata"]))
    if len({x["path"].casefold() for x in inputs.values()}) != 2:
        raise ValueError("train and validation roles alias")
    return inputs


def create_campaign(args, campaign_root, *, data_root, jobs, synthetic_test=False,
                    assumptions_sha256=None, **resource_options):
    limits = resources(**resource_options)
    report = inspect_research_inputs(args, synthetic_test=synthetic_test)
    if assumptions_sha256 is not None:
        _require_sha256(assumptions_sha256, "assumptions")
        equal(report["evidence"]["sha256"], assumptions_sha256, "assumptions external pin")
    elif not synthetic_test:
        raise ValueError("external assumptions SHA256 required")
    root = _reject_link_components(Path(campaign_root).absolute())
    data = _reject_link_components(Path(data_root).absolute())
    if root == data or root.is_relative_to(data) or data.is_relative_to(root):
        raise ValueError("campaign and data roots overlap")
    if root.exists():
        raise RecoveryIntegrityError("campaign already exists")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("nonempty jobs list required")
    shared = dict(domain="synthetic_fixture" if synthetic_test else "research_assumption",
        inputs=_freeze_inputs(report, data), evidence=report["evidence"], assumptions_utf8=assumptions_bytes(report),
        feature_signature=report["feature_signature"], source_inventory=sources(), resources=limits, **NONFORMAL)
    payloads = {}
    for request in jobs:
        job = freeze_job(request)
        identity = resolve_identity(job, shared["feature_signature"], synthetic_test=synthetic_test)
        name = f"{job['model_id']}-s{job['seed']}"
        if name in payloads:
            raise ValueError("duplicate model/seed: candidates need separate campaigns")
        payloads[name] = _canonical_json(dict(schema_version=1, purpose="racepv.research-batch-job.v1",
            name=name, job=job, model_identity=identity, shared=shared, **NONFORMAL))
        if len(payloads[name]) > 32 * 1024**2:
            raise ValueError("binding receipt metadata exceeds the safe metadata read bound")
    with seed_request_lock(root):
        stage = root.parent / ("." + root.name + ".batch-" + uuid.uuid4().hex)
        stage.mkdir(mode=0o700)
        (stage / "bindings").mkdir()
        pins = {}
        for name, raw in payloads.items():
            _write_exclusive(stage / "bindings" / (name + ".json"), raw)
            pins[name] = dict(path=root / "bindings" / (name + ".json"), sha256=digest(raw))
        manifest = _canonical_json(dict(schema_version=1, purpose="racepv.research-batch-campaign.v1",
            jobs={name: digest(raw) for name, raw in payloads.items()}))
        _write_exclusive(stage / "campaign.json", manifest)
        _fsync_directory(stage / "bindings")
        _fsync_directory(stage)
        _rename_noreplace(stage, root)
        _fsync_directory(root.parent)
    return dict(campaign_sha256=digest(manifest), jobs=pins, metadata_only=True,
                data_root=str(data), **NONFORMAL)


def open_binding(path, expected_sha256, *, campaign_sha256, synthetic_test=False):
    path = _reject_link_components(Path(path).absolute())
    for pin in (expected_sha256, campaign_sha256):
        _require_sha256(pin, "external SHA256")
    raw = safe_read(path)
    equal(digest(raw), expected_sha256, "binding external pin")
    value = strict_json(raw, "batch job")
    if set(value) != {"schema_version", "purpose", "name", "job", "model_identity", "shared", *NONFORMAL} or type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["purpose"] != "racepv.research-batch-job.v1":
        raise RecoveryIntegrityError("batch binding schema differs")
    shared, job = value["shared"], value["job"]
    if set(shared) != {"domain", "inputs", "evidence", "assumptions_utf8", "feature_signature", "source_inventory", "resources", *NONFORMAL}:
        raise RecoveryIntegrityError("batch shared schema differs")
    domain = "synthetic_fixture" if synthetic_test else "research_assumption"
    if shared["domain"] != domain or shared["evidence"]["kind"] != domain:
        raise ValueError("batch evidence domain differs")
    for owner in (value, shared, value["model_identity"]):
        if any(owner[k] is not False for k in NONFORMAL):
            raise RecoveryIntegrityError("nonformal flags differ")
    equal(resources(**shared["resources"]), shared["resources"], "resources")
    equal(shared["source_inventory"], sources(), "executing source inventory")
    equal(resolve_identity(job, shared["feature_signature"], synthetic_test=synthetic_test), value["model_identity"], "resolved batch identity")
    equal(value["name"], f"{job['model_id']}-s{job['seed']}", "job name")
    if path.name != value["name"] + ".json" or path.parent.name != "bindings":
        raise RecoveryIntegrityError("canonical binding path differs")
    if set(shared["inputs"]) != {"train", "validation"}:
        raise RecoveryIntegrityError("train/validation scope differs")
    roles = []
    for item in shared["inputs"].values():
        if set(item) != {"path", "descriptor_sha256", "read_scope_sha256", "metadata"}:
            raise RecoveryIntegrityError("input pin schema differs")
        validate_relative_path(item["path"])
        roles.append(item["path"].casefold())
        for field in ("descriptor_sha256", "read_scope_sha256"):
            _require_sha256(item[field], field)
    if len(set(roles)) != 2:
        raise RecoveryIntegrityError("input roles alias")
    equal(digest(shared["assumptions_utf8"].encode()), shared["evidence"]["sha256"], "assumptions")
    manifest_raw = safe_read(path.parent.parent / "campaign.json")
    equal(digest(manifest_raw), campaign_sha256, "campaign external pin")
    manifest = strict_json(manifest_raw, "batch campaign")
    if set(manifest) != {"schema_version", "purpose", "jobs"} or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or manifest["purpose"] != "racepv.research-batch-campaign.v1" or manifest["jobs"].get(value["name"]) != expected_sha256:
        raise RecoveryIntegrityError("batch campaign membership differs")
    return ResearchBinding(path, raw, expected_sha256, campaign_sha256)


def refresh(binding, *, synthetic_test=False):
    return open_binding(binding.path, binding.sha256, campaign_sha256=binding.campaign_sha256,
                        synthetic_test=synthetic_test)


def preflight(binding, *, data_root, synthetic_test=False):
    binding = refresh(binding, synthetic_test=synthetic_test)
    root = _reject_link_components(Path(data_root).absolute())
    if root == binding.run_root or root.is_relative_to(binding.run_root) or binding.run_root.is_relative_to(root):
        raise ValueError("run and data roots overlap")
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
    equal(assumptions_bytes(report), shared["assumptions_utf8"], "assumptions bytes")
    equal(_freeze_inputs(report, root), shared["inputs"], "frozen portable inputs")
    return args, report
