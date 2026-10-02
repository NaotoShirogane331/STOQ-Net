"""Validation-only, immutable non-epoch lifecycle; numerical work requires opt-in."""

import math
import os
import time
import uuid

import numpy as np

from racepv.baselines.research_registry import QUANTILES
from racepv.data.exploratory_candidate import NONFORMAL
from racepv.training.exploratory_contract import digest, equal, safe_read, strict_json
from racepv.training.recoverable_run import (
    _canonical_json,
    _fsync_directory,
    _rename_noreplace,
    _write_exclusive,
    seed_request_lock,
    writer_lock,
)
from racepv.training.recovery import RecoveryIntegrityError
from racepv.training.research_baseline_bridge import (
    ConditionalEmpiricalQuantiles,
    load_research_inputs,
)
from racepv.training.research_batch_campaign import FOUNDATIONS, TREES, preflight, refresh
from racepv.training.research_batch_rules import ProtocolSmartPersistence
from racepv.training.research_batch_storage import (
    commit_unit,
    file_record,
    read_array,
    read_units,
    require_space,
    stage_directory,
    verified_file,
    write_array,
)
from racepv.training.research_tree_runtime import batches, run_trees

EXPORT_KEYS = ("targets", "target_mask", "daylight_mask", "site_ids", "issue_ns", "target_interval_start_ns")


def population(binding, split):
    return sum(leaf["schema"]["arrays"]["issue_ns"]["shape"][0]
               for leaf in binding.raw["shared"]["inputs"][split]["metadata"]["leaves"])


def unit_names(binding):
    model = binding.raw["job"]["model_id"]
    if model in TREES:
        return [f"l{lead:02d}-q{q:02d}" for lead in range(24)
                for q in range(7 if model == TREES[0] else 1)]
    if model in FOUNDATIONS:
        count = math.ceil(population(binding, "validation") / binding.raw["shared"]["resources"]["batch_size"])
        return [f"b{index:08d}" for index in range(count)]
    return ["reference"]


def _initialize(binding):
    root = binding.run_root
    if root.exists() or root.is_symlink():
        raise RecoveryIntegrityError("run exists; use continue")
    stage = root.parent / ("." + root.name + ".initial-" + uuid.uuid4().hex)
    stage.mkdir(mode=0o700)
    for name in ("units", "staging", "scratch"):
        (stage / name).mkdir()
    _write_exclusive(stage / "binding.json", binding.raw_bytes)
    _fsync_directory(stage)
    _rename_noreplace(stage, root)
    _fsync_directory(root.parent)


def terminal_record(binding, receipts):
    return dict(schema_version=1, binding_sha256=binding.sha256,
        campaign_sha256=binding.campaign_sha256,
        units=[dict(unit=item["unit"], commit_sha256=item["commit_sha256"]) for item in receipts],
        validation_count=population(binding, "validation"), terminal_reason="all_units_committed")


def check_run(binding, *, synthetic_test=False):
    """Inspect owned receipts and stream hashes only; never model/input loading."""
    binding = refresh(binding, synthetic_test=synthetic_test)
    if safe_read(binding.run_root / "binding.json") != binding.raw_bytes:
        raise RecoveryIntegrityError("run binding differs")
    expected = unit_names(binding)
    receipts = read_units(binding, expected)
    terminal = binding.run_root / "terminal.json"
    terminal_sha = None
    status = "units_committed"
    if terminal.exists() or terminal.is_symlink():
        if len(receipts) != len(expected):
            raise RecoveryIntegrityError("terminal before complete unit coverage")
        raw = safe_read(terminal)
        equal(strict_json(raw, "terminal"), terminal_record(binding, receipts), "terminal chain")
        terminal_sha, status = digest(raw), "prediction_complete"
    output = binding.run_root / "exports"
    export_sha = None
    if output.exists() or output.is_symlink():
        if terminal_sha is None:
            raise RecoveryIntegrityError("export before terminal")
        raw = safe_read(output / "commit.json")
        record = strict_json(raw, "export commit")
        if set(record) != {"binding_sha256", "terminal_commit_sha256", "files"}:
            raise RecoveryIntegrityError("export commit schema differs")
        equal(record["binding_sha256"], binding.sha256, "export binding")
        equal(record["terminal_commit_sha256"], terminal_sha, "export terminal")
        if [item["path"] for item in record["files"]] != ["run.json", "validation.npz"]:
            raise RecoveryIntegrityError("export file set differs")
        for item in record["files"]:
            with verified_file(output / item["path"], limit=binding.raw["shared"]["resources"]["model_file_limit_bytes"], expected=item):
                pass
        report = strict_json(safe_read(output / "run.json"), "run report")
        equal(report["binding_sha256"], binding.sha256, "report binding")
        equal(report["terminal_commit_sha256"], terminal_sha, "report terminal")
        export_sha, status = digest(raw), "completed"
    return dict(status=status, committed_units=len(receipts), expected_units=len(expected),
        binding_sha256=binding.sha256, campaign_sha256=binding.campaign_sha256,
        terminal_commit_sha256=terminal_sha, export_commit_sha256=export_sha, **NONFORMAL)


def _reference(binding, datasets, existing, callback, *, train_store):
    if existing:
        return existing
    started = time.perf_counter()
    job, shared = binding.raw["job"], binding.raw["shared"]
    size, signature = shared["resources"]["batch_size"], shared["feature_signature"]
    name = job["model_id"]
    model = None
    if name == "smart_persistence":
        model = ProtocolSmartPersistence(signature).fit_store(train_store, split="train")
    elif name == "conditional_empirical_quantiles":
        model = ConditionalEmpiricalQuantiles().fit(batches(datasets["train"], size), split="train")
    values, fallbacks = [], {}
    for batch in batches(datasets["validation"], size):
        if model is None:
            clear = batch["future_known"][..., signature["future_known_columns"].index("clear_sky_power_norm")].numpy()
            if not np.isfinite(clear).all():
                raise ValueError("nonfinite future clear sky")
            prediction = np.where(batch["daylight_mask"].numpy(), np.clip(clear, 0., 1.05), 0.)
        else:
            prediction = model.predict(batch)
            if name == "conditional_empirical_quantiles":
                unique, counts = np.unique(model.last_fallbacks, return_counts=True)
                for label, count in zip(unique, counts, strict=True):
                    fallbacks[str(label)] = fallbacks.get(str(label), 0) + int(count)
        values.append(prediction)
    prediction = np.concatenate(values)
    if name == "smart_persistence":
        reference = model.fitted_identity()
    elif name == "conditional_empirical_quantiles":
        reference = dict(support_counts={str(k): v for k, v in model.support_counts.items()},
            groups={":".join(map(str, key)): [count, quantiles.tolist()]
                    for key, (count, quantiles) in model._groups.items()}, fallback_counts=fallbacks,
            fit_split="train", minimum_support=200, unique_physical_observations=True)
    else:
        reference = dict(training_kind="none", deterministic_clear_sky=True)
    stage = stage_directory(binding, "reference")
    require_space(stage, prediction.nbytes + len(_canonical_json(reference)) + 65536, shared["resources"]["scratch_reserve_bytes"])
    write_array(stage / "prediction.npy", prediction)
    _write_exclusive(stage / "reference.json", _canonical_json(reference))
    receipt = commit_unit(binding, stage, "reference", previous=None,
        metadata=dict(elapsed_seconds=time.perf_counter() - started, prediction_shape=list(prediction.shape)))
    existing.append(receipt)
    if callback:
        callback(receipt)
    return existing


def _foundation(binding, datasets, existing, callback, *, project_root, synthetic_test, synthetic_backend_factory):
    import torch

    from racepv.baselines.research_foundation import build_research_foundation
    if synthetic_backend_factory is not None and not synthetic_test:
        raise ValueError("injected provider requires synthetic identity")
    if len(existing) == len(unit_names(binding)):
        return existing
    job = binding.raw["job"]
    identity = binding.raw["model_identity"]
    if not synthetic_test:
        if not torch.cuda.is_available():
            raise RuntimeError("NOT_READY: foundation CUDA execution requires target host")
        torch.cuda.reset_peak_memory_stats()
    config = job["config"]
    model = build_research_foundation(job["model_id"], binding.raw["shared"]["feature_signature"],
        source_review_bytes=config["source_review_utf8"].encode(), source_review_sha256=config["source_review_sha256"],
        weight_receipt_bytes=config["weight_receipt_utf8"].encode(), weight_receipt_sha256=config["weight_receipt_sha256"],
        project_root=project_root, device=identity["device"], synthetic_test=synthetic_test,
        synthetic_backend_factory=synthetic_backend_factory)
    equal(model.resolved_identity()["frozen_adapter_identity"], identity["frozen_adapter_identity"], "sealed foundation identity")
    limits = binding.raw["shared"]["resources"]
    previous = existing[-1]["commit_sha256"] if existing else None
    completed = len(existing)
    for index, batch in enumerate(batches(datasets["validation"], limits["batch_size"])):
        if index < completed:
            continue
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = model.predict(batch)
        unit = f"b{index:08d}"
        stage = stage_directory(binding, unit)
        require_space(stage, prediction.nbytes + 65536, limits["scratch_reserve_bytes"])
        write_array(stage / "prediction.npy", prediction)
        _write_exclusive(stage / "reference.json", _canonical_json(model.resolved_identity()))
        receipt = commit_unit(binding, stage, unit, previous=previous, metadata=dict(
            start_index=index * limits["batch_size"], count=len(prediction),
            prediction_shape=list(prediction.shape), elapsed_seconds=time.perf_counter() - started,
            provider="synthetic_mapping_only" if synthetic_backend_factory else "sealed_official_backend",
            live_weights_verified=synthetic_backend_factory is None,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if not synthetic_test else None))
        previous = receipt["commit_sha256"]
        existing.append(receipt)
        if callback:
            callback(receipt)
    return existing


def validate_export(arrays, *, expected_count, prediction_kind):
    key = "point" if prediction_kind == "point" else "quantiles"
    if set(arrays) != {*EXPORT_KEYS, key}:
        raise ValueError("batch export exact fields differ")
    shape = (expected_count, 24)
    for field in ("targets", "target_mask", "daylight_mask", "target_interval_start_ns"):
        if arrays[field].shape != shape:
            raise ValueError("validation shape/coverage differs")
    for field in ("site_ids", "issue_ns"):
        if arrays[field].shape != (expected_count,) or arrays[field].dtype != np.int64:
            raise ValueError("validation key shape/type differs")
    if arrays["target_interval_start_ns"].dtype != np.int64:
        raise ValueError("target time dtype differs")
    for field in ("target_mask", "daylight_mask"):
        if arrays[field].dtype != bool:
            raise ValueError("export masks must be boolean")
    y, output = arrays["targets"], arrays[key]
    if output.shape != (shape if key == "point" else (*shape, 7)) or not np.isfinite(output).all():
        raise ValueError("prediction shape/finite contract differs")
    if (output < 0).any() or (output > 1.05 + 1e-7).any() or (key == "quantiles" and (np.diff(output, axis=-1) < 0).any()):
        raise ValueError("prediction range/order differs")
    if np.any(output[~arrays["daylight_mask"]] != 0):
        raise ValueError("night predictions must be zero")
    if not np.isfinite(y[arrays["target_mask"]]).all():
        raise ValueError("observed targets must be finite")
    expected_times = arrays["issue_ns"][:, None] + np.arange(24, dtype=np.int64)[None] * 900_000_000_000
    if not np.array_equal(arrays["target_interval_start_ns"], expected_times):
        raise ValueError("target time coverage differs from schema issue/lead clock")
    keys = np.stack((arrays["site_ids"], arrays["issue_ns"]), axis=1)
    if len(np.unique(keys, axis=0)) != expected_count:
        raise ValueError("duplicate validation common keys")


def _collect_export(binding, datasets, receipts):
    parts = {key: [] for key in EXPORT_KEYS}
    for batch in batches(datasets["validation"], binding.raw["shared"]["resources"]["batch_size"]):
        for key in parts:
            parts[key].append(batch[key].numpy())
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    model = binding.raw["job"]["model_id"]
    limit = binding.raw["shared"]["resources"]["model_file_limit_bytes"]
    def prediction(receipt):
        record = next(item for item in receipt["files"] if item["path"] == "prediction.npy")
        return read_array(binding.run_root / "units" / receipt["unit"] / "prediction.npy", record, limit)
    if model in TREES:
        raw = np.empty((population(binding, "validation"), 24, 7 if model == TREES[0] else 1), dtype=np.float32)
        for receipt in receipts:
            lead, q = int(receipt["unit"][1:3]), int(receipt["unit"][-2:])
            values = prediction(receipt)
            if values.shape != (len(raw),):
                raise ValueError("subestimator coverage differs")
            raw[:, lead, q] = values
        output = np.sort(raw, axis=-1)
        if model == TREES[1]:
            output = output[..., 0]
    else:
        output = np.concatenate([prediction(receipt) for receipt in receipts])
    if not np.isfinite(output).all():
        raise ValueError("raw committed prediction must be finite")
    day = arrays["daylight_mask"]
    output = np.where(day if output.ndim == 2 else day[..., None], np.clip(output, 0., 1.05), 0.)
    key = "point" if binding.raw["model_identity"]["prediction_kind"] == "point" else "quantiles"
    arrays[key] = output
    validate_export(arrays, expected_count=population(binding, "validation"), prediction_kind=binding.raw["model_identity"]["prediction_kind"])
    return arrays


def _metrics(arrays):
    legal = arrays["target_mask"] & arrays["daylight_mask"]
    if not legal.any():
        raise ValueError("no legal daylight validation population")
    if "point" in arrays:
        return dict(metric="daylight_mae", value=float(np.abs(arrays["targets"][legal] - arrays["point"][legal]).mean()), valid_count=int(legal.sum()))
    errors = arrays["targets"][legal, None] - arrays["quantiles"][legal]
    levels = np.asarray(QUANTILES)
    cells = (np.maximum(levels * errors, (levels - 1) * errors) * np.asarray((.1, .2, .4, .5, .4, .2, .1))).sum(-1)
    return dict(metric="daylight_qcrps", value=float(cells.mean()), valid_count=int(legal.sum()))


def _export(binding, datasets, receipts, session_started):
    arrays = _collect_export(binding, datasets, receipts)
    stage = stage_directory(binding, "export")
    limits = binding.raw["shared"]["resources"]
    require_space(stage, sum(value.nbytes for value in arrays.values()) + 1024**2, limits["scratch_reserve_bytes"])
    with (stage / "validation.npz").open("xb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    import resource
    terminal_sha = digest(safe_read(binding.run_root / "terminal.json"))
    report = dict(binding_sha256=binding.sha256, campaign_sha256=binding.campaign_sha256,
        terminal_commit_sha256=terminal_sha, job=binding.raw["job"], model_identity=binding.raw["model_identity"],
        validation_metrics=_metrics(arrays), validation_rows=len(arrays["targets"]),
        recovery_boundary=binding.raw["model_identity"]["recovery_boundary"],
        committed_unit_seconds=sum(item["metadata"]["elapsed_seconds"] for item in receipts),
        session_seconds=time.perf_counter() - session_started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        peak_rss_scope="entire Linux process lifetime; not isolated per estimator",
        total_seconds_including_uncommitted_attempts=None,
        timing_domain=binding.raw["shared"]["domain"], heldout_access=False, **NONFORMAL)
    _write_exclusive(stage / "run.json", _canonical_json(report))
    records = [file_record(stage / name, limit=limits["model_file_limit_bytes"]) for name in ("run.json", "validation.npz")]
    _write_exclusive(stage / "commit.json", _canonical_json(dict(binding_sha256=binding.sha256,
        terminal_commit_sha256=terminal_sha, files=records)))
    _fsync_directory(stage)
    _rename_noreplace(stage, binding.run_root / "exports")
    _fsync_directory(binding.run_root)


def execute(binding, *, mode, data_root, acknowledge_assumptions=False, acknowledge_scope=False,
            synthetic_test=False, on_unit_committed=None, foundation_project_root=None,
            synthetic_backend_factory=None):
    if mode not in ("start", "continue") or acknowledge_assumptions is not True or acknowledge_scope is not True:
        raise ValueError("start/continue requires explicit assumptions and numeric scope acknowledgements")
    if synthetic_backend_factory is not None and not synthetic_test:
        raise ValueError("injected foundation provider requires synthetic identity")
    binding = refresh(binding, synthetic_test=synthetic_test)
    started = time.perf_counter()
    with seed_request_lock(binding.run_root):
        if mode == "start" and binding.run_root.exists():
            raise RecoveryIntegrityError("run already exists; use continue")
        if mode == "continue":
            result = check_run(binding, synthetic_test=synthetic_test)
            if result["status"] == "completed":
                return dict(result, action="skipped_completed")
        args, report = preflight(binding, data_root=data_root, synthetic_test=synthetic_test)
        if mode == "start":
            _initialize(binding)
        with writer_lock(binding.run_root):
            loaded = load_research_inputs(args, report, synthetic_test=synthetic_test)
            datasets = loaded["datasets"]
            for split in ("train", "validation"):
                equal(len(datasets[split]), population(binding, split), "dataset coverage")
            receipts = read_units(binding, unit_names(binding))
            model = binding.raw["job"]["model_id"]
            if model in TREES:
                receipts = run_trees(binding, datasets, receipts, on_unit_committed=on_unit_committed)
            elif model in FOUNDATIONS:
                receipts = _foundation(binding, datasets, receipts, on_unit_committed,
                    project_root=foundation_project_root, synthetic_test=synthetic_test,
                    synthetic_backend_factory=synthetic_backend_factory)
            else:
                receipts = _reference(binding, datasets, receipts, on_unit_committed,
                                      train_store=loaded["stores"]["train"])
            equal([item["unit"] for item in receipts], unit_names(binding), "complete unit coverage")
            terminal = binding.run_root / "terminal.json"
            terminal_bytes = _canonical_json(terminal_record(binding, receipts))
            if terminal.exists():
                if safe_read(terminal) != terminal_bytes:
                    raise RecoveryIntegrityError("terminal state differs")
            else:
                stage = stage_directory(binding, "terminal")
                _write_exclusive(stage / "terminal.json", terminal_bytes)
                _rename_noreplace(stage / "terminal.json", terminal)
                _fsync_directory(binding.run_root)
            _export(binding, datasets, receipts, started)
        return dict(check_run(binding, synthetic_test=synthetic_test), action="executed")
