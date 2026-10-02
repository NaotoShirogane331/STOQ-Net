"""Research neural lifecycle, durable epochs and fixed-best validation-only export."""

import io
import random
import time
import uuid
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from racepv.data.exploratory_candidate import NONFORMAL
from racepv.training.engine import EarlyStoppingTrainer, _model_inputs
from racepv.training.exploratory_contract import digest, equal, safe_read, strict_json
from racepv.training.recoverable_run import (
    _GENERATION,
    _canonical_json,
    _fsync_directory,
    _rename_noreplace,
    _state_bytes,
    _write_exclusive,
    _write_generation_directory,
    seed_request_lock,
    writer_lock,
)
from racepv.training.recovery import (
    CommittedGeneration,
    RecoveryIntegrityError,
    RecoveryUnsupportedError,
    decode_progress,
    encode_progress,
    restore_rng_state,
    validate_commit,
)
from racepv.training.research_baseline_bridge import (
    ResearchRampWeight,
    load_research_inputs,
    masked_research_loss,
    predict_research_neural,
)
from racepv.training.research_campaign import (
    CANDIDATES,
    build_components,
    output_metadata,
    preflight,
    refresh,
)


class ResearchTrainer(EarlyStoppingTrainer):
    """Weight validation selection by eligible target cells, independent of training loss."""

    def __init__(self, *args, model_id, ramp_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id, self.ramp_weight = model_id, ramp_weight

    def predict_batch(self, batch):
        mask = batch["history_mask"]
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("history mask is not binary")
        batch = dict(batch, history_mask=mask.bool())
        if self.model_id not in CANDIDATES:
            return predict_research_neural(self.model, batch, self.model_id)
        inputs = _model_inputs(batch)
        inputs["history_values"] = torch.where(inputs["history_mask"].bool(), inputs["history_values"], 0.)
        output = self.model(**inputs)
        q = output["quantiles"]
        if q.shape != (*batch["targets"].shape, 7) or not bool(torch.isfinite(q).all()):
            raise ValueError("non-finite or malformed candidate quantiles")
        if bool((q[..., 1:] < q[..., :-1]).any()):
            raise ValueError("candidate quantile order differs")
        output["quantiles"] = torch.where(batch["daylight_mask"].bool()[..., None], q.clamp(0., 1.05), 0.)
        return output

    def _run_epoch(self, loader, training):
        started = time.perf_counter()
        self.model.train(training)
        total, score, count = 0., 0., 0
        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            for raw_batch in loader:
                batch = self._move(raw_batch)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    output = self.predict_batch(batch)
                    result = masked_research_loss(output["quantiles"], batch, model_id="tcn_qr")
                    if result["skip"]:
                        continue
                    if self.model_id in CANDIDATES:
                        loss, terms = self._criterion(output, batch)
                    else:
                        resolved = self.model.research_identity["resolved_config"]
                        terms = masked_research_loss(output, batch, model_id=self.model_id,
                            ramp_training_weight=self.ramp_weight,
                            ramp_weighted_quantile_multiplier=resolved.get("ramp_weighted_quantile_multiplier", 6.),
                            ramp_loss_weight=resolved.get("ramp_loss_weight", 1.))
                        loss = terms["loss"]
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite research loss")
                if training:
                    if self.separate_state_optimizer:
                        self._step_separated_losses(forecast_loss=terms["forecast_loss"], state_loss=terms["state_loss"])
                    else:
                        self.optimizer.zero_grad(set_to_none=True)
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.forecast_parameters, self.gradient_clip_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self._clear_optimizer_gradients()
                        self._assert_finite_training_state(stage="research step")
                n = result["valid_count"]
                total += float(loss.detach()) * n
                score += float(result["daylight_qcrps"].detach()) * n
                count += n
        if count == 0:
            raise ValueError("no legal daylight target population")
        return dict(loss=total / count, daylight_qcrps=score / count,
                    valid_count=float(count), elapsed_seconds=time.perf_counter() - started)


def loader_contract(binding):
    shared = binding.raw["shared"]
    result = dict(num_workers=0, shuffle=True, drop_last=False, batch_size=shared["budget"]["batch_size"])
    for split, entry in shared["inputs"].items():
        result[split + "_length"] = sum(leaf["schema"]["arrays"]["issue_ns"]["shape"][0]
                                           for leaf in entry["metadata"]["leaves"])
    return result


@dataclass(frozen=True)
class ResearchChain:
    latest: CommittedGeneration
    generations: int


def read_chain(binding):
    """Validate every generation, retaining only one full state and compact facts.

    The legacy all-generations reader remains unchanged. Hash the very bytes
    deserialized here, and release each old payload before loading its successor.
    """
    epochs = binding.run_root / "epochs"
    if not epochs.is_dir() or (epochs / "g0000").is_symlink():
        raise RecoveryUnsupportedError("run has no committed g0000 generation")
    names = []
    for entry in epochs.iterdir():
        match = _GENERATION.fullmatch(entry.name)
        if match is None:
            raise RecoveryIntegrityError(f"unexpected epoch entry: {entry.name}")
        if entry.is_symlink() or not entry.is_dir():
            raise RecoveryIntegrityError("epoch generation must be a real directory")
        names.append((int(match.group(1)), entry.name))
    names.sort()
    if [index for index, _ in names] != list(range(len(names))):
        raise RecoveryIntegrityError("epoch generation chain has a gap")
    if not names:
        raise RecoveryIntegrityError("missing initial generation")
    contract, previous_commit, previous_history, previous_metadata = loader_contract(binding), None, None, None
    previous_terminal, state = None, None
    budget = binding.raw["shared"]["budget"]
    for index, name in names:
        state = None  # No previous tensor payload survives into deserialization.
        if previous_terminal is not None:
            raise RecoveryIntegrityError("generation after terminal state")
        directory = epochs / name
        commit_raw = safe_read(directory / "commit.json")
        commit_sha256 = digest(commit_raw)
        commit = strict_json(commit_raw, "epoch commit")
        validate_commit(commit, index, binding.sha256)
        if commit["previous_commit_sha256"] != previous_commit:
            raise RecoveryIntegrityError("epoch commit predecessor mismatch")
        raw = safe_read(directory / "state.pt", limit=2 * 1024**3)
        if digest(raw) != commit["state_sha256"]:
            raise RecoveryIntegrityError("epoch state SHA-256 mismatch")
        try:
            state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RecoveryIntegrityError(f"unsafe or invalid tensor state: {name}") from exc
        del raw
        if not isinstance(state, dict) or state.get("generation") != index or state.get("binding_sha256") != binding.sha256:
            raise RecoveryIntegrityError("epoch state identity mismatch")
        decode_progress(state, binding_sha256=binding.sha256, loader_contract=contract)
        if index:
            equal(digest(_canonical_json(state["history"][:-1])), previous_history, "history prefix")
        if state["next_epoch"] > budget["max_epochs"]:
            raise RecoveryIntegrityError("epoch budget exceeded")
        terminal = ("early_stopping" if state["next_epoch"] >= budget["min_epochs"] and
            state["stale_epochs"] >= budget["patience"] else
            "max_epochs" if state["next_epoch"] == budget["max_epochs"] else None)
        equal(state["terminal_reason"], terminal, "terminal reason")
        if state["best_checkpoint"] is not None:
            equal(state["best_checkpoint"]["metadata"]["binding"], output_metadata(binding), "checkpoint metadata")
            equal(state["best_checkpoint"]["selection_metric"], "daylight_qcrps", "selection")
            metadata_sha256 = digest(_canonical_json(state["best_checkpoint"]["metadata"]))
            if previous_metadata is not None:
                equal(metadata_sha256, previous_metadata, "derived training identity")
            previous_metadata = metadata_sha256
        else:
            previous_metadata = None
        previous_history = digest(_canonical_json(state["history"]))
        previous_terminal, previous_commit = state["terminal_reason"], commit_sha256
    return ResearchChain(CommittedGeneration(index, commit_sha256, commit, state), len(names))


def check_run(binding, *, synthetic_test=False):
    """Internal integrity only; no external input arrays/model construction/device query."""
    binding = refresh(binding, synthetic_test=synthetic_test)
    if safe_read(binding.run_root / "binding.json") != binding.raw_bytes:
        raise RecoveryIntegrityError("run binding differs")
    chain, contract = read_chain(binding), loader_contract(binding)
    latest = chain.latest
    status = "fit_complete" if latest.state["terminal_reason"] else "training_committed"
    output = binding.run_root / "exports"
    if output.exists() or output.is_symlink():
        if status != "fit_complete":
            raise RecoveryIntegrityError("export precedes terminal commit")
        record = strict_json(safe_read(output / "commit.json"), "export commit")
        if set(record) != {"binding_sha256", "terminal_commit_sha256", "files"}:
            raise RecoveryIntegrityError("export commit schema differs")
        equal(record["binding_sha256"], binding.sha256, "export binding")
        equal(record["terminal_commit_sha256"], latest.commit_sha256, "terminal export pin")
        equal([item["path"] for item in record["files"]], ["best.ckpt", "validation.npz", "run.json"], "export files")
        for item in record["files"]:
            data = safe_read(output / item["path"], limit=2 * 1024**3)
            if set(item) != {"path", "sha256", "size_bytes"} or digest(data) != item["sha256"] or len(data) != item["size_bytes"]:
                raise RecoveryIntegrityError("export content corrupted")
        from racepv.training.candidate_recovery import _tensor_same
        best = torch.load(io.BytesIO(safe_read(output / "best.ckpt", limit=2 * 1024**3)), map_location="cpu", weights_only=True)
        _tensor_same(best, latest.state["best_checkpoint"], "fixed best checkpoint")
        report = strict_json(safe_read(output / "run.json"), "run report")
        equal(report["binding"], output_metadata(binding), "run report binding")
        equal(report["terminal_commit_sha256"], latest.commit_sha256, "run report terminal")
        with np.load(io.BytesIO(safe_read(output / "validation.npz", limit=2 * 1024**3)), allow_pickle=False) as data:
            validate_export(dict(data), expected_count=contract["validation_length"], model_identity=binding.raw["model_identity"])
        status = "completed"
    return dict(status=status, generations=chain.generations, binding_sha256=binding.sha256,
                terminal_commit_sha256=latest.commit_sha256, **NONFORMAL)


def _initialize(binding, trainer, contract):
    root = binding.run_root
    if root.exists() or root.is_symlink():
        raise RecoveryIntegrityError("run already exists")
    stage = root.parent / ("." + root.name + ".initial-" + uuid.uuid4().hex)
    stage.mkdir(mode=0o700)
    (stage / "epochs").mkdir()
    (stage / "staging").mkdir()
    _write_exclusive(stage / "binding.json", binding.raw_bytes)
    progress = trainer.capture_progress(next_epoch=0, best_epoch=-1, best_loss=float("inf"),
        stale_epochs=0, history=[], terminal_reason=None, best_checkpoint=None)
    _write_generation_directory(stage / "epochs/g0000", encode_progress(progress, binding.sha256, contract),
        binding_sha256=binding.sha256, generation=0, previous_commit_sha256=None)
    _fsync_directory(stage / "epochs")
    _fsync_directory(stage)
    _rename_noreplace(stage, root)
    _fsync_directory(root.parent)


def _commit(binding, progress, contract, previous):
    chain = read_chain(binding)
    if chain.latest.commit_sha256 != previous or progress.next_epoch != chain.generations or chain.latest.state["terminal_reason"]:
        raise RecoveryIntegrityError("epoch sequence or predecessor differs")
    stage = binding.run_root / "staging" / ("epoch-" + uuid.uuid4().hex)
    _write_generation_directory(stage, encode_progress(progress, binding.sha256, contract),
        binding_sha256=binding.sha256, generation=progress.next_epoch, previous_commit_sha256=previous)
    destination = binding.run_root / "epochs" / f"g{progress.next_epoch:04d}"
    _rename_noreplace(stage, destination)
    _fsync_directory(destination.parent)
    return digest(safe_read(destination / "commit.json"))


def validate_export(arrays, *, expected_count, model_identity=None):
    required = {"quantiles", "targets", "target_mask", "daylight_mask", "site_ids", "issue_ns", "target_interval_start_ns"}
    if (model_identity or {}).get("model_id") == "F":
        required.add("state_logits")
    if not required.issubset(arrays):
        raise RecoveryIntegrityError("validation export fields missing")
    shape = (expected_count, 24)
    if any(arrays[k].shape != shape for k in ("targets", "target_mask", "daylight_mask", "target_interval_start_ns")):
        raise RecoveryIntegrityError("validation shape/coverage differs")
    if any(arrays[k].shape != (expected_count,) for k in ("site_ids", "issue_ns")):
        raise RecoveryIntegrityError("validation key shapes differ")
    q = arrays["quantiles"]
    if q.shape != (*shape, 7) or not np.isfinite(q).all() or (np.diff(q, axis=-1) < 0).any() or (q < 0).any() or (q > 1.05 + 1e-6).any():
        raise RecoveryIntegrityError("invalid quantile export")
    if np.any(q[~arrays["daylight_mask"].astype(bool)] != 0):
        raise RecoveryIntegrityError("night predictions are nonzero")
    if not np.isfinite(arrays["targets"][arrays["target_mask"].astype(bool)]).all():
        raise RecoveryIntegrityError("non-finite legal targets")
    keys = np.stack([np.repeat(arrays["site_ids"], 24), np.repeat(arrays["issue_ns"], 24), arrays["target_interval_start_ns"].reshape(-1)], axis=1)
    if len(np.unique(keys, axis=0)) != expected_count * 24:
        raise RecoveryIntegrityError("duplicate validation common keys")
    for key, value in arrays.items():
        if value.ndim < 1 or value.shape[0] != expected_count:
            raise RecoveryIntegrityError("diagnostic population shape differs")
        if key not in ("targets",) and value.dtype.kind == "f" and not np.isfinite(value).all():
            raise RecoveryIntegrityError("non-finite exported diagnostic")
    for key in ("state_logits", "state_probs"):
        if key in arrays and arrays[key].shape != (*shape, 3):
            raise RecoveryIntegrityError("state diagnostic shape differs")
    active = (model_identity or {}).get("model_metadata", {}).get("active_experts")
    if active is not None:
        count = len(active)
        if arrays.get("expert_quantiles", np.empty(0)).shape != (*shape, count, 7) or arrays.get("gate_weights", np.empty(0)).shape != (*shape, count):
            raise RecoveryIntegrityError("expert diagnostic shape differs from model identity")
        weights = arrays["gate_weights"]
        if (weights < 0).any() or not np.allclose(weights.sum(-1), 1., atol=1e-5):
            raise RecoveryIntegrityError("expert weights are not a simplex")


def _export(binding, trainer, loader, session_started):
    started = time.perf_counter()
    terminal = read_chain(binding).latest
    best = terminal.state["best_checkpoint"]
    if terminal.state["terminal_reason"] is None or best is None:
        raise RecoveryIntegrityError("export requires terminal fixed best")
    trainer.model.load_state_dict(best["model_state_dict"], strict=True)
    trainer.model.eval()
    collected, scalar_outputs = {}, {}
    with torch.inference_mode():
        for raw in loader:
            batch = trainer._move(raw)
            outputs = trainer.predict_batch(batch)
            for key, value in {**outputs, **{k: batch[k] for k in ("targets", "target_mask", "daylight_mask", "site_ids", "issue_ns", "target_interval_start_ns")}}.items():
                if isinstance(value, torch.Tensor):
                    if value.ndim == 0:
                        scalar = float(value.detach().cpu())
                        if not np.isfinite(scalar) or (key in scalar_outputs and scalar_outputs[key] != scalar):
                            raise RecoveryIntegrityError("non-sample output changed across validation batches")
                        scalar_outputs[key] = scalar
                        continue
                    collected.setdefault(key, []).append(value.detach().cpu().numpy())
    arrays = {key: np.concatenate(value) for key, value in collected.items()}
    validate_export(arrays, expected_count=len(loader.dataset), model_identity=binding.raw["model_identity"])
    stage = binding.run_root / "staging" / ("export-" + uuid.uuid4().hex)
    stage.mkdir(mode=0o700)
    _write_exclusive(stage / "best.ckpt", _state_bytes(best))
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    _write_exclusive(stage / "validation.npz", buffer.getvalue())
    history = terminal.state["history"]
    report = dict(binding=output_metadata(binding), terminal_commit_sha256=terminal.commit_sha256,
        non_sample_outputs=scalar_outputs,
        best_epoch=best["epoch"], actual_epochs=terminal.state["next_epoch"],
        actual_precision="amp_float16" if trainer.use_amp else "float32",
        parameter_count=sum(p.numel() for p in trainer.model.parameters()),
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(trainer.device) if trainer.device.type == "cuda" else None,
        train_seconds=sum(x["train_metrics"]["elapsed_seconds"] for x in history),
        validation_seconds=sum(x["validation_metrics"]["elapsed_seconds"] for x in history),
        committed_epoch_seconds=sum(x["seconds"] for x in history),
        export_seconds=time.perf_counter() - started, session_seconds=time.perf_counter() - session_started,
        total_seconds_including_uncommitted_attempts=None, **NONFORMAL)
    _write_exclusive(stage / "run.json", _canonical_json(report))
    files = []
    for filename in ("best.ckpt", "validation.npz", "run.json"):
        data = safe_read(stage / filename, limit=2 * 1024**3)
        files.append(dict(path=filename, sha256=digest(data), size_bytes=len(data)))
    _write_exclusive(stage / "commit.json", _canonical_json(dict(binding_sha256=binding.sha256,
        terminal_commit_sha256=terminal.commit_sha256, files=files)))
    _fsync_directory(stage)
    _rename_noreplace(stage, binding.run_root / "exports")
    _fsync_directory(binding.run_root)


def execute(binding, *, mode, data_root, acknowledge_assumptions=False,
            acknowledge_scope=False, synthetic_test=False, on_epoch_committed=None):
    """Start or continue exactly one job; callback runs only after durable epoch publication."""
    if mode not in ("start", "continue") or acknowledge_assumptions is not True or acknowledge_scope is not True:
        raise ValueError("start/continue requires explicit assumptions and train/validation scope acknowledgements")
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
        job, shared = binding.raw["job"], binding.raw["shared"]
        random.seed(job["seed"])
        np.random.seed(job["seed"])
        torch.manual_seed(job["seed"])
        model, objective, identity = build_components(job, shared["feature_signature"], synthetic_test=synthetic_test)
        equal(identity, binding.raw["model_identity"], "actual model identity")
        if shared["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("NOT_READY: CUDA required; no CPU or GRU fallback")
        loaded = load_research_inputs(args, report, synthetic_test=synthetic_test)
        budget, contract = shared["budget"], loader_contract(binding)
        train = DataLoader(loaded["datasets"]["train"], batch_size=budget["batch_size"], shuffle=True, num_workers=0)
        validation = DataLoader(loaded["datasets"]["validation"], batch_size=budget["batch_size"], shuffle=False, num_workers=0)
        for split, loader in (("train", train), ("validation", validation)):
            equal(len(loader.dataset), contract[split + "_length"], "full loader population")
            equal(loader.dataset.content_identity, shared["inputs"][split]["metadata"]["aggregate_content_sha256"], "loader content")
        weight = None
        if job["model_id"] == "rqr_tft":
            stable = ramp = 0
            for batch in DataLoader(train.dataset, batch_size=budget["batch_size"]):
                legal = batch["target_mask"].bool() & batch["daylight_mask"].bool() & torch.isfinite(batch["targets"])
                stable += int((legal & (batch["state_labels"] == 0)).sum())
                ramp += int((legal & ((batch["state_labels"] == 1) | (batch["state_labels"] == 2))).sum())
            weight = ResearchRampWeight(stable, ramp, (stable / ramp) ** .5 if ramp else 1.)
        trainer = ResearchTrainer(model, objective, model_id=job["model_id"], ramp_weight=weight,
            device=shared["device"], learning_rate=budget["learning_rate"], weight_decay=budget["weight_decay"],
            max_epochs=budget["max_epochs"], patience=budget["patience"], min_epochs=budget["min_epochs"],
            gradient_clip_norm=budget["gradient_clip_norm"], mixed_precision=budget["mixed_precision"],
            checkpoint_metadata=dict(binding=output_metadata(binding), ramp_training_weight=weight.__dict__ if weight else None),
            selection_metric="daylight_qcrps", separate_state_optimizer=job["model_id"] in CANDIDATES)
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)
        if mode == "start":
            _initialize(binding, trainer, contract)
        with writer_lock(binding.run_root):
            check_run(binding, synthetic_test=synthetic_test)
            latest = read_chain(binding).latest
            progress = decode_progress(latest.state, binding_sha256=binding.sha256, loader_contract=contract, trainer=trainer)
            if progress.best_checkpoint is not None:
                equal(progress.best_checkpoint["metadata"], trainer.checkpoint_metadata, "derived train identity")
            trainer.restore_progress(progress)
            restore_rng_state(latest.state["rng_state"])
            previous = latest.commit_sha256
            def commit(current):
                nonlocal previous
                refresh(binding, synthetic_test=synthetic_test)
                previous = _commit(binding, current, contract, previous)
                if on_epoch_committed is not None:
                    on_epoch_committed(current)
            if progress.terminal_reason is None:
                trainer.fit(train, validation, initial_progress=progress, on_epoch_committed=commit)
            _export(binding, trainer, validation, started)
            return dict(check_run(binding, synthetic_test=synthetic_test),
                        action="export_only" if progress.terminal_reason else mode)
