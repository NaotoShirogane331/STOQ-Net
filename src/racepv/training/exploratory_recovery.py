"""Exploratory epoch recovery using the existing trainer and state/lock primitives."""

import io
import json
import random
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data._utils.collate import default_collate

from racepv.cli.scoped_candidate import candidate_model_config_from_signature
from racepv.data import exploratory_candidate
from racepv.data.scoped_dataset import ScopedCandidateDataset
from racepv.models.reliability_a import ReliabilityGateARaCEPV
from racepv.reliability_a_loss import ReliabilityAObjective
from racepv.reliability_experiment import (
    ReliabilityControlRaCEPV,
    make_reliability_components,
)
from racepv.training.engine import EarlyStoppingTrainer
from racepv.training.exploratory_contract import (
    SCORE_ID,
    digest,
    equal,
    open_binding,
    output_metadata,
    preflight,
    safe_read,
    strict_json,
)
from racepv.training.recoverable_run import (
    _canonical_json,
    _fsync_directory,
    _rename_noreplace,
    _state_bytes,
    _write_exclusive,
    _write_generation_directory,
    read_generation_chain,
    seed_request_lock,
    writer_lock,
)
from racepv.training.recovery import (
    RecoveryIntegrityError,
    decode_progress,
    encode_progress,
    restore_rng_state,
)


def refresh(binding, *, synthetic_test=False):
    raw = binding.raw
    return open_binding(
        Path(raw["campaign_root"]) / "bindings" / f"{raw['cell']}-s{raw['seed']}.json",
        binding.sha256,
        synthetic_test=synthetic_test,
    )


def validate_metadata(binding, metadata):
    equal(metadata, output_metadata(binding), "checkpoint metadata")


def validate_trainer(binding, trainer, train_loader, validation_loader):
    """Check actual model, objective, both optimizers and entire sampler populations."""
    raw, config = binding.raw, binding.experiment.training
    expected_type = ReliabilityGateARaCEPV if raw["cell"] == "A" else ReliabilityControlRaCEPV
    if type(trainer) is not EarlyStoppingTrainer or type(trainer.model) is not expected_type:
        raise RecoveryIntegrityError("actual trainer/model class differs")
    equal(asdict(trainer.model.config), raw["shared"]["model_config"], "actual model config")
    validate_metadata(binding, trainer.checkpoint_metadata)
    if type(trainer.criterion) is not ReliabilityAObjective or not trainer.separate_state_optimizer:
        raise RecoveryIntegrityError("objective or separated state optimizer differs")
    equal(
        trainer.criterion.quantiles.cpu().tolist(),
        torch.tensor(binding.experiment.quantiles, dtype=torch.float32).tolist(),
        "objective quantiles",
    )
    for name in ("max_epochs", "patience", "gradient_clip_norm"):
        equal(getattr(trainer, name), getattr(config, name), "trainer " + name)
    if trainer.selection_metric != "daylight_qcrps" or trainer.checkpoint_path is not None:
        raise RecoveryIntegrityError("selection or direct checkpoint side effect differs")
    synthetic = raw["shared"]["domain"] == "synthetic_fixture"
    if trainer.device.type != ("cpu" if synthetic else "cuda") or trainer.use_amp != (
        not synthetic
    ):
        raise RecoveryIntegrityError("trainer device or AMP differs")
    for optimizer in (trainer.forecast_optimizer, trainer.state_optimizer):
        if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
            raise RecoveryIntegrityError("optimizer class/group differs")
        group = optimizer.param_groups[0]
        equal(group["lr"], config.learning_rate, "optimizer learning rate")
        equal(group["weight_decay"], config.weight_decay, "optimizer weight decay")
        for key, value in (
            ("betas", (0.9, 0.999)),
            ("eps", 1e-8),
            ("amsgrad", False),
            ("maximize", False),
        ):
            equal(group[key], value, "optimizer " + key)
    contract = dict(num_workers=0, shuffle=True, drop_last=False, batch_size=config.batch_size)
    for split, loader, sampler in (
        ("train", train_loader, RandomSampler),
        ("validation", validation_loader, SequentialSampler),
    ):
        if (
            type(loader) is not DataLoader
            or type(loader.dataset) is not ScopedCandidateDataset
            or type(loader.sampler) is not sampler
            or type(loader.batch_sampler) is not BatchSampler
            or loader.batch_sampler.sampler is not loader.sampler
            or loader.sampler.data_source is not loader.dataset
            or loader.num_workers != 0
            or loader.drop_last is not False
            or loader.batch_sampler.drop_last is not False
            or loader.generator is not None
            or loader.collate_fn is not default_collate
            or loader.pin_memory
            or loader.persistent_workers
        ):
            raise RecoveryIntegrityError("full dataset/loader/sampler contract differs")
        equal(loader.batch_size, config.batch_size, "loader batch size")
        equal(loader.batch_sampler.batch_size, config.batch_size, "batch sampler size")
        if split == "train" and (
            loader.sampler.replacement
            or loader.sampler.generator is not None
            or loader.sampler.num_samples != len(loader.dataset)
        ):
            raise RecoveryIntegrityError("train population sampling differs")
        equal(
            loader.dataset.feature_signature, raw["shared"]["feature_signature"], "dataset features"
        )
        equal(
            loader.dataset.content_identity,
            raw["shared"]["inputs"][split]["metadata"]["aggregate_content_sha256"],
            "dataset contents",
        )
        expected_length = sum(len(item) for item in loader.dataset._datasets)
        equal(len(loader.dataset), expected_length, "full dataset length")
        contract[split + "_length"] = expected_length
    return contract


def _read_bounded_generation_chain(binding):
    """Bound every commit and state before each use of the legacy chain reader."""
    epochs = binding.run_root / "epochs"
    for path in epochs.glob("g*/commit.json"):
        safe_read(path)
    for path in epochs.glob("g*/state.pt"):
        safe_read(path, limit=512 * 1024 * 1024)
    return read_generation_chain(binding.run_root, binding_sha256=binding.sha256)


def check_run(binding, *, synthetic_test=None):
    """Internal read-only inspection: no data descriptors, arrays, model construction or CUDA."""
    if synthetic_test is None:
        synthetic_test = binding.raw["shared"]["domain"] == "synthetic_fixture"
    binding = refresh(binding, synthetic_test=synthetic_test)
    root = binding.run_root
    if safe_read(root / "binding.json") != binding.raw_bytes:
        raise RecoveryIntegrityError("committed binding differs")
    chain = _read_bounded_generation_chain(binding)
    if not chain:
        raise RecoveryIntegrityError("initial generation is missing")
    expected_loader = dict(
        num_workers=0,
        shuffle=True,
        drop_last=False,
        batch_size=binding.experiment.training.batch_size,
    )
    for split, entry in binding.raw["shared"]["inputs"].items():
        expected_loader[split + "_length"] = sum(
            leaf["schema"]["arrays"]["issue_ns"]["shape"][0] for leaf in entry["metadata"]["leaves"]
        )
    previous = None
    for generation in chain:
        state = generation.state
        equal(state["loader_contract"], expected_loader, "pinned loader contract")
        if previous is not None:
            if previous["terminal_reason"] is not None:
                raise RecoveryIntegrityError("generation follows terminal state")
            equal(state["history"][:-1], previous["history"], "committed history prefix")
            equal(state["loader_contract"], previous["loader_contract"], "committed loader")
        if state["best_checkpoint"] is not None:
            validate_metadata(binding, state["best_checkpoint"].get("metadata"))
            equal(
                state["best_checkpoint"].get("selection_metric"),
                "daylight_qcrps",
                "best checkpoint selection",
            )
        next_epoch = state["next_epoch"]
        config = binding.experiment.training
        if next_epoch > config.max_epochs:
            raise RecoveryIntegrityError("epoch budget exceeded")
        terminal = (
            "early_stopping"
            if state["stale_epochs"] >= config.patience
            else "max_epochs"
            if next_epoch == config.max_epochs
            else None
        )
        if state["terminal_reason"] != terminal:
            raise RecoveryIntegrityError("terminal progress differs")
        previous = state
    latest = chain[-1]
    status = "fit_complete" if latest.state["terminal_reason"] is not None else "training_committed"
    export_root = root / "exports"
    if export_root.exists() or export_root.is_symlink():
        if status != "fit_complete":
            raise RecoveryIntegrityError("exports before terminal generation")
        record = strict_json(safe_read(export_root / "commit.json"), "export commit")
        if set(record) != {"binding_sha256", "terminal_commit_sha256", "files"}:
            raise RecoveryIntegrityError("export commit keys differ")
        equal(record["binding_sha256"], binding.sha256, "export binding")
        equal(record["terminal_commit_sha256"], latest.commit_sha256, "export terminal")
        count = latest.state["loader_contract"]["validation_length"]
        batch = binding.experiment.training.batch_size
        expected_names = ["best.ckpt", "model.pt", "run.json"] + [
            f"validation-{index:06d}.npz" for index in range((count + batch - 1) // batch)
        ]
        if [item.get("path") for item in record["files"]] != expected_names:
            raise RecoveryIntegrityError("export file coverage differs")
        for item in record["files"]:
            if set(item) != {"path", "sha256", "size_bytes"}:
                raise RecoveryIntegrityError("export file record differs")
            data = safe_read(export_root / item["path"], limit=512 * 1024 * 1024)
            if digest(data) != item["sha256"] or len(data) != item["size_bytes"]:
                raise RecoveryIntegrityError("export content corrupted")
        from racepv.training.candidate_recovery import _tensor_same

        best = torch.load(
            io.BytesIO(safe_read(export_root / "best.ckpt")), map_location="cpu", weights_only=True
        )
        _tensor_same(best, latest.state["best_checkpoint"], "exploratory best export")
        model = torch.load(
            io.BytesIO(safe_read(export_root / "model.pt")), map_location="cpu", weights_only=True
        )
        validate_metadata(binding, model["metadata"])
        _tensor_same(model["model_state_dict"], best["model_state_dict"], "exploratory weights")
        report = strict_json(safe_read(export_root / "run.json"), "run report")
        equal(
            {key: report.get(key) for key in output_metadata(binding)},
            output_metadata(binding),
            "report identity",
        )
        status = "completed"
    return dict(
        status=status,
        generations=len(chain),
        binding_sha256=binding.sha256,
        terminal_commit_sha256=latest.commit_sha256,
        **exploratory_candidate.NONFORMAL,
    )


def _initialize(binding, trainer, contract):
    root = binding.run_root
    if root.exists() or root.is_symlink():
        raise RecoveryIntegrityError("run destination already exists")
    staging = root.parent / ("." + root.name + ".initial-" + uuid.uuid4().hex)
    staging.mkdir(mode=0o700)
    (staging / "epochs").mkdir()
    (staging / "staging").mkdir()
    _write_exclusive(staging / "binding.json", binding.raw_bytes)
    progress = trainer.capture_progress(
        next_epoch=0,
        best_epoch=-1,
        best_loss=float("inf"),
        stale_epochs=0,
        history=[],
        terminal_reason=None,
        best_checkpoint=None,
    )
    _write_generation_directory(
        staging / "epochs" / "g0000",
        encode_progress(progress, binding.sha256, contract),
        binding_sha256=binding.sha256,
        generation=0,
        previous_commit_sha256=None,
    )
    _fsync_directory(staging / "epochs")
    _fsync_directory(staging)
    _rename_noreplace(staging, root)
    _fsync_directory(root.parent)


def _commit(binding, progress, contract, previous):
    chain = _read_bounded_generation_chain(binding)
    if chain[-1].commit_sha256 != previous or progress.next_epoch != len(chain):
        raise RecoveryIntegrityError("epoch predecessor or sequence differs")
    if chain[-1].state["terminal_reason"] is not None:
        raise RecoveryIntegrityError("cannot commit after terminal state")
    validate_metadata(binding, progress.best_checkpoint["metadata"])
    stage = binding.run_root / "staging" / ("epoch-" + uuid.uuid4().hex)
    state = encode_progress(progress, binding.sha256, contract)
    _write_generation_directory(
        stage,
        state,
        binding_sha256=binding.sha256,
        generation=progress.next_epoch,
        previous_commit_sha256=previous,
    )
    destination = binding.run_root / "epochs" / f"g{progress.next_epoch:04d}"
    _rename_noreplace(stage, destination)
    _fsync_directory(destination.parent)
    return digest(safe_read(destination / "commit.json"))


def _export(binding, trainer, loader):
    from racepv.metrics.quantile_scoring import quantile_score_cells, quantile_score_summary

    terminal = _read_bounded_generation_chain(binding)[-1]
    best = terminal.state["best_checkpoint"]
    validate_metadata(binding, best["metadata"])
    trainer.model.load_state_dict(best["model_state_dict"], strict=True)
    stage = binding.run_root / "staging" / ("export-" + uuid.uuid4().hex)
    stage.mkdir(mode=0o700)
    metadata = output_metadata(binding)
    _write_exclusive(stage / "best.ckpt", _state_bytes(best))
    _write_exclusive(
        stage / "model.pt",
        _state_bytes(dict(model_state_dict=best["model_state_dict"], metadata=metadata)),
    )
    totals = {name: dict(loss_sum=0.0, count=0) for name in ("overall", "daylight")}
    shards = []
    for index, batch in enumerate(loader):
        arrays = {key: value.numpy() for key, value in trainer.predict([batch]).items()}
        overall = arrays["target_mask"].astype(bool) & np.isfinite(arrays["targets"])
        daylight = overall & arrays["daylight_mask"].astype(bool)
        for name, mask in (("overall", overall), ("daylight", daylight)):
            cells = quantile_score_cells(
                arrays["targets"],
                arrays["quantiles"],
                binding.experiment.quantiles,
                eligible_mask=mask,
                score_id=SCORE_ID,
            )
            summary = quantile_score_summary(cells, eligible_mask=mask)
            totals[name]["loss_sum"] += summary["loss_sum"]
            totals[name]["count"] += summary["count"]
            arrays[name + "_qcrps_cells"] = cells
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        filename = f"validation-{index:06d}.npz"
        _write_exclusive(stage / filename, buffer.getvalue())
        shards.append(filename)
    for summary in totals.values():
        summary["mean"] = summary["loss_sum"] / summary["count"] if summary["count"] else None
    report = dict(
        metadata,
        scores=totals,
        best_epoch=best["epoch"],
        model_metadata=trainer.model.model_metadata,
        prediction_shards=shards,
        validation_windows=len(loader.dataset),
    )
    _write_exclusive(stage / "run.json", _canonical_json(report))
    files = []
    for filename in ["best.ckpt", "model.pt", "run.json", *shards]:
        data = safe_read(stage / filename, limit=512 * 1024 * 1024)
        files.append(dict(path=filename, sha256=digest(data), size_bytes=len(data)))
    _write_exclusive(
        stage / "commit.json",
        _canonical_json(
            dict(
                binding_sha256=binding.sha256,
                terminal_commit_sha256=terminal.commit_sha256,
                files=files,
            )
        ),
    )
    _fsync_directory(stage)
    _rename_noreplace(stage, binding.run_root / "exports")
    _fsync_directory(binding.run_root)


def execute(
    binding,
    *,
    mode,
    acknowledge_assumptions=False,
    acknowledge_scope=False,
    data_root=None,
    synthetic_test=False,
    on_epoch_committed=None,
):
    """Run/resume exactly one frozen destination; callback observes durable epoch commits."""
    if mode not in ("start", "continue"):
        raise RecoveryIntegrityError("execution mode must be start or continue")
    if acknowledge_assumptions is not True or acknowledge_scope is not True:
        raise RecoveryIntegrityError(
            "explicit research-assumption and numeric scope acknowledgements required"
        )
    binding = refresh(binding, synthetic_test=synthetic_test)
    with seed_request_lock(binding.run_root):
        if mode == "start" and (binding.run_root.exists() or binding.run_root.is_symlink()):
            raise RecoveryIntegrityError("run already exists; use continue")
        if mode == "continue":
            inspected = check_run(binding, synthetic_test=synthetic_test)
            if inspected["status"] == "completed":
                return dict(inspected, action="skipped_completed")
        args, report = preflight(binding, data_root=data_root, synthetic_test=synthetic_test)
        loaded = exploratory_candidate.load_inputs(
            args, report, experiment=binding.experiment, synthetic_test=synthetic_test
        )
        seed = binding.raw["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        training = binding.experiment.training
        model_config = candidate_model_config_from_signature(
            binding.raw["shared"]["feature_signature"],
            binding.experiment,
            backbone="gru" if synthetic_test else "mamba",
        )
        model, objective = make_reliability_components(model_config, binding.raw["cell"])
        trainer = EarlyStoppingTrainer(
            model,
            objective,
            device="cpu" if synthetic_test else "cuda",
            learning_rate=training.learning_rate,
            weight_decay=training.weight_decay,
            max_epochs=training.max_epochs,
            patience=training.patience,
            gradient_clip_norm=training.gradient_clip_norm,
            mixed_precision=training.mixed_precision,
            checkpoint_metadata=output_metadata(binding),
            selection_metric="daylight_qcrps",
            separate_state_optimizer=True,
        )
        train_loader = DataLoader(
            loaded["datasets"]["train"], batch_size=training.batch_size, shuffle=True, num_workers=0
        )
        validation_loader = DataLoader(
            loaded["datasets"]["validation"],
            batch_size=training.batch_size,
            shuffle=False,
            num_workers=0,
        )
        contract = validate_trainer(binding, trainer, train_loader, validation_loader)
        if mode == "start":
            _initialize(binding, trainer, contract)
        with writer_lock(binding.run_root):
            inspected = check_run(binding, synthetic_test=synthetic_test)
            latest = _read_bounded_generation_chain(binding)[-1]
            progress = decode_progress(
                latest.state,
                binding_sha256=binding.sha256,
                loader_contract=contract,
                trainer=trainer,
            )
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
                trainer.fit(
                    train_loader,
                    validation_loader,
                    initial_progress=progress,
                    on_epoch_committed=commit,
                )
            _export(binding, trainer, validation_loader)
            return dict(
                check_run(binding, synthetic_test=synthetic_test),
                action="export_only" if progress.terminal_reason is not None else mode,
            )
