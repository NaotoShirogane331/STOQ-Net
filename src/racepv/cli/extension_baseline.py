"""Train one audited supervised baseline-extension model on train/validation only."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path
from typing import Any

import yaml
from torch import Tensor, nn

from racepv.audit import sha256_file
from racepv.baselines.extension_registry import (
    BACKBONE_DEFAULTS,
    SOURCE_IDENTITIES,
    build_extension_model,
)
from racepv.baselines.modern import ModernBaselineLoss
from racepv.config import QUANTILE_LEVELS
from racepv.environment import environment_report
from racepv.training import EarlyStoppingTrainer, TrainingResult, build_window_loader
from racepv.utils import atomic_json_dump, seed_everything, select_device
from scripts.baseline_extension_contract import SEEDS, SUPERVISED_MODELS
from scripts.baseline_extension_remediation_contract import validate_remediation_config
from scripts.baseline_extension_sources import load_source_review_bytes
from scripts.racepv_reference_io import SafeReader

from .modern_baseline import (
    _atomic_save_predictions_npz,
    _atomic_torch_save,
    _atomic_yaml_dump,
    _jsonable,
    _predict_validation,
    _prediction_archive,
    _prepared_output_metadata,
    _reverify_dataset_identities,
    _sample_ramp_thresholds,
    _validate_datasets,
)

REGISTERED_LEVELS = tuple(QUANTILE_LEVELS)
_CONFIG_KEYS = {
    "schema_version",
    "purpose",
    "selection_boundary",
    "selection_metric",
    "quantiles",
    "seeds",
    "common_config",
    "data_protocol",
    "models",
    "training",
    "heldout",
}


def _write_bytes_exclusive(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("artifact write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_exclusive(source: Path, destination: Path) -> None:
    entry = os.lstat(source)
    if stat.S_ISLNK(entry.st_mode) or not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1:
        raise ValueError("artifact copy source must be a single-link regular file")
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        content = bytearray()
        while chunk := os.read(descriptor, min(1024 * 1024, before.st_size - len(content) + 1)):
            content.extend(chunk)
            if len(content) > before.st_size:
                raise ValueError("artifact copy source grew during read")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if len(content) != before.st_size or identity(before) != identity(after) or after.st_nlink != 1:
        raise ValueError("artifact copy source changed during read")
    _write_bytes_exclusive(destination, bytes(content))


def _capture_execution_logs(args: argparse.Namespace, output: Path) -> None:
    paths = (
        (sys.stdout, getattr(args, "execution_stdout_path", None), "stdout.log"),
        (sys.stderr, getattr(args, "execution_stderr_path", None), "stderr.log"),
    )
    for stream, source, name in paths:
        if not isinstance(source, Path):
            raise ValueError("recoverable execution log path is missing")  # noqa: TRY004
        stream.flush()
        descriptor = stream.fileno()
        os.fsync(descriptor)
        _copy_exclusive(source, output / name)


def _terminal_export_result(trainer: Any, progress: Any) -> TrainingResult:
    if progress.terminal_reason not in {"early_stopping", "max_epochs"}:
        raise ValueError("export-only requires a terminal fit commit")
    best = progress.best_checkpoint
    if not isinstance(best, dict) or not isinstance(best.get("model_state_dict"), dict):
        raise ValueError("export-only terminal state has no best checkpoint")  # noqa: TRY004
    trainer.restore_progress(progress)
    trainer.model.load_state_dict(best["model_state_dict"], strict=True)
    return TrainingResult(
        best_epoch=progress.best_epoch,
        best_validation_loss=progress.best_validation_loss,
        best_validation_metric_name=trainer.selection_metric,
        history=progress.history,
        stopped_early=len(progress.history) < trainer.max_epochs,
    )


def _read_frozen_inputs(args: argparse.Namespace) -> dict[str, tuple[bytes, str]]:
    """Read the three metadata inputs through one no-follow project-root reader."""

    project_root = args.extension_config.parent.parent
    expected_common = project_root / "configs/final_day1_4090.yaml"
    paths = {
        "common configuration": args.config,
        "extension configuration": args.extension_config,
        "source review": args.source_review,
    }
    if not project_root.is_absolute() or args.config != expected_common:
        raise ValueError("extension input root differs from the frozen project layout")
    snapshots: dict[str, tuple[bytes, str]] = {}
    with SafeReader(project_root) as reader:
        for label, path in paths.items():
            try:
                relative = path.relative_to(project_root).as_posix()
            except ValueError as exc:
                raise ValueError(f"{label} escapes the project root") from exc
            content, receipt = reader.read(relative, role=label.replace(" ", "_"))
            snapshots[label] = (content, receipt.sha256)
    return snapshots


class _QuantileMappingWrapper(nn.Module):
    """Adapt the extension tensor output to the shared trainer mapping contract."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, **inputs: Tensor) -> dict[str, Tensor]:
        output = self.model(**inputs)
        if not isinstance(output, Tensor):
            raise TypeError("extension model must return one quantile tensor")
        return {"quantiles": output}


def _extension_config(content: bytes) -> dict[str, object]:
    value = yaml.safe_load(content.decode("utf-8"))
    if not isinstance(value, dict) or set(value) != _CONFIG_KEYS:
        raise ValueError("extension configuration must be a mapping")
    if (
        value.get("schema_version") != 1
        or value.get("purpose") != "racepv_six_baseline_supervised_training"
        or value.get("selection_boundary") != "validation_only"
        or value.get("selection_metric") != "daylight_qcrps"
        or value.get("quantiles") != list(REGISTERED_LEVELS)
        or value.get("seeds") != list(SEEDS)
    ):
        raise ValueError("extension configuration semantics differ")
    models = value.get("models")
    if not isinstance(models, dict) or set(models) != set(SUPERVISED_MODELS):
        raise ValueError("extension model configuration grid differs")
    for model_id in SUPERVISED_MODELS:
        record = models[model_id]
        if not isinstance(record, dict) or set(record) != {
            "implementation_kind",
            "upstream_defaults",
        }:
            raise ValueError(f"extension model configuration schema differs: {model_id}")
        if (
            record["implementation_kind"] != "paper_derived_protocol_adaptation"
            or record["upstream_defaults"] != BACKBONE_DEFAULTS[model_id]
        ):
            raise ValueError(f"extension model defaults differ: {model_id}")
    common = value["common_config"]
    if common != {
        "path": "configs/final_day1_4090.yaml",
        "sha256": "8a1d7eb02b4773d200c759050c9e7e11d8f3f2da9698539cbda7608d1101e1b0",
    }:
        raise ValueError("common configuration identity differs")
    data = value["data_protocol"]
    expected_data = {
        "train_descriptor": {
            "path": "data/processed/dkasc-tail-confirm-4site-s4/train.index.json",
            "sha256": "652586ce411a393dfca6e4c1a884debd06c698be8749b893431b4b765bdd2559",
            "content_sha256": "7b51eb990737a4de6b2258c5388d81170f0c8aa104f6c978ccf17cfcb56e5f37",
        },
        "validation_descriptor": {
            "path": "data/processed/dkasc-tail-confirm-4site-s4/validation.index.json",
            "sha256": "11a49549d9bff5a434bfb2bd62fc01595cbaf6e4213e90a80d99c8116be0b8fd",
            "content_sha256": "5fed358f7cf1d0db3fd0f8f5d3013e8757d6a7ccf324225de4289ad4ffc1bcd7",
        },
        "future_target": "forbidden",
    }
    if data != expected_data:
        raise ValueError("future target contract differs")
    training = value["training"]
    expected_training = {
        "batch_size": 64,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "max_epochs": 8,
        "patience": 2,
        "gradient_clip_norm": 1.0,
        "num_workers": 0,
        "mixed_precision": True,
        "hyperparameter_search": False,
        "maximum_fit_count": 15,
        "epoch_recovery": True,
        "wall_clock_timeout_seconds": None,
        "completed_cell_action": "skip_verified",
        "fit_complete_missing_export_action": "export_only",
        "invalid_cell_action": "preserve_and_fail_closed",
    }
    if training != expected_training:
        raise ValueError("training contract differs")
    if value["heldout"] != {"calibration_or_test_loaded": False}:
        raise ValueError("heldout exclusion contract differs")
    return value


def _load_extension_config(content: bytes, *, profile: str) -> dict[str, object]:
    if profile == "v4_amp":
        return _extension_config(content)
    if profile == "fp32_remediation":
        return validate_remediation_config(content)
    raise ValueError("unknown extension configuration profile")


def _precision_metadata(mixed_precision: bool) -> dict[str, object]:
    return {
        "precision_mode": "amp_fp16" if mixed_precision else "fp32",
        "autocast_enabled": mixed_precision,
        "grad_scaler_enabled": mixed_precision,
    }


def _empty_registered_optimizer(payload: Any) -> bool:
    """An untrained single-optimizer commit cannot contain even empty slots."""
    if (
        type(payload) is not dict
        or set(payload) != {"optimizer_schema", "optimizer_state_dict", "parameter_names"}
        or payload["optimizer_schema"] != "single_optimizer_v1"
    ):
        return False
    names = payload["parameter_names"]
    state = payload["optimizer_state_dict"]
    if (
        type(names) is not list
        or not names
        or any(type(name) is not str or not name for name in names)
        or len(set(names)) != len(names)
        or type(state) is not dict
        or set(state) != {"state", "param_groups"}
        or type(state["state"]) is not dict
        or state["state"]
    ):
        return False
    groups = state["param_groups"]
    if type(groups) is not list or len(groups) != 1 or type(groups[0]) is not dict:
        return False
    parameters = groups[0].get("params")
    return (
        type(parameters) is list
        and all(type(parameter) is int for parameter in parameters)
        and parameters == list(range(len(names)))
    )


def _validate_fp32_continuation(progress: Any) -> None:
    checkpoint = getattr(progress, "best_checkpoint", object())
    if checkpoint is None:
        # g0000 has no selected checkpoint: no epoch has yet been committed.
        # Recovery has already verified its binding and generation identity.
        initial_fields = {
            "next_epoch": 0,
            "best_epoch": -1,
            "best_validation_loss": float("inf"),
            "stale_epochs": 0,
            "history": [],
            "terminal_reason": None,
            "scaler_state": {},
        }
        if all(
            type(getattr(progress, field, object())) is type(expected)
            and getattr(progress, field) == expected
            for field, expected in initial_fields.items()
        ) and _empty_registered_optimizer(getattr(progress, "optimizer_state", None)):
            return
    metadata = checkpoint.get("metadata") if isinstance(checkpoint, dict) else None
    scaler_state = getattr(progress, "scaler_state", None)
    expected = _precision_metadata(False)
    if (
        not isinstance(metadata, dict)
        or any(
            type(metadata.get(key)) is not type(value) or metadata.get(key) != value
            for key, value in expected.items()
        )
        or type(scaler_state) is not dict
        or scaler_state
    ):
        raise ValueError("FP32 continuation checkpoint precision is missing or inconsistent")


def _source_record(content: bytes, model_id: str) -> dict[str, object]:
    review = load_source_review_bytes(content)
    if review["status"] != "passed":
        raise ValueError("aggregate source review has not accepted all six models")
    record = review["models"][model_id]
    if record["status"] != "passed" or record["blockers"] != []:
        raise ValueError(f"source review has not accepted {model_id}")
    expected = SOURCE_IDENTITIES[model_id]
    for field in ("repository_url", "git_commit", "source_tree_sha256"):
        if record[field] != expected[field]:
            raise ValueError(f"source review identity differs for {model_id}: {field}")
    return dict(record)


def _resolved_model_config(model_id: str, train_data: Any) -> dict[str, object]:
    schema = train_data.schema
    history_names = list(schema["history_columns"])
    future_names = list(schema["future_known_columns"])
    site_names = list(schema["site_metadata_names"])
    result = {
        "schema_version": "racepv.baseline-extension-supervised-model.v1",
        "model_id": model_id,
        "input_dim": len(history_names),
        "future_dim": len(future_names),
        "site_dim": len(site_names),
        "history_feature_names": history_names,
        "future_feature_names": future_names,
        "site_feature_names": site_names,
        "clear_sky_feature_index": future_names.index("clear_sky_power_norm"),
        "daylight_feature_index": future_names.index("solar_elevation"),
        "history_length": int(train_data.arrays["history_values"].shape[1]),
        "horizon": int(train_data.arrays["targets"].shape[1]),
        "quantiles": list(REGISTERED_LEVELS),
        "feature_schema": {
            "history_values": "registered_history_features_with_target_first",
            "history_mask": "observed_boolean_same_shape",
            "future_known": "registered_weather_without_future_target",
            "site_metadata": "registered_static_features",
            "future_target": "forbidden",
        },
        "source_identity": dict(SOURCE_IDENTITIES[model_id]),
        "backbone_defaults": dict(BACKBONE_DEFAULTS[model_id]),
        "output_head": "direct_ordered_quantiles_v1",
    }
    if schema.get("schema_version") == 2:
        from racepv.baselines.extension_registry import _FEATURE_SCHEMA_V2
        from racepv.cli.modern_baseline import _prepared_dataset_identity

        result.update(
            schema_version="racepv.baseline-extension-supervised-model.v2",
            prepared_contract=_prepared_dataset_identity(train_data),
            feature_schema=dict(_FEATURE_SCHEMA_V2),
        )
    return result


def run_extension_training(
    args: argparse.Namespace,
    *,
    recovery_controller: Any | None = None,
    pre_publish_validator: Any | None = None,
) -> Path:
    """Execute one fit or same-binding continuation and publish audited exports."""

    from racepv.cli.modern_baseline import _preflight_prepared_inputs

    _preflight_prepared_inputs(args)

    if args.name not in SUPERVISED_MODELS or type(args.seed) is not int or args.seed not in SEEDS:
        raise ValueError("extension model/seed identity differs")
    if not args.validation_only:
        raise ValueError("--validation-only is required")
    if recovery_controller is None and (args.output.exists() or args.output.is_symlink()):
        raise FileExistsError(f"output already exists: {args.output}")

    seed_everything(args.seed)
    frozen_inputs = _read_frozen_inputs(args)
    common_content, common_config_sha = frozen_inputs["common configuration"]
    extension_content, extension_config_sha = frozen_inputs["extension configuration"]
    source_content, source_review_sha = frozen_inputs["source review"]
    expected_file_hashes = {
        "extension configuration": getattr(args, "extension_config_sha256", None),
        "source review": getattr(args, "source_review_sha256", None),
    }
    observed_file_hashes = {
        "extension configuration": extension_config_sha,
        "source review": source_review_sha,
    }
    if recovery_controller is not None:
        for label, expected in expected_file_hashes.items():
            if not isinstance(expected, str) or observed_file_hashes[label] != expected:
                raise ValueError(f"{label} hash differs from manifest")
    config_profile = getattr(args, "config_profile", "v4_amp")
    extension_config = _load_extension_config(extension_content, profile=config_profile)
    extension_training = extension_config["training"]
    assert isinstance(extension_training, dict)
    mixed_precision = extension_training["mixed_precision"]
    assert isinstance(mixed_precision, bool)
    if config_profile == "fp32_remediation" and mixed_precision:
        raise ValueError("FP32 remediation requires mixed_precision=false")
    precision_metadata = _precision_metadata(mixed_precision)
    source_record = _source_record(source_content, args.name)

    from racepv.cli.modern_baseline import _load_yaml_mapping
    from racepv.config import _construct_config

    common_config = _construct_config(
        _load_yaml_mapping(common_content, label="common configuration")
    )
    if tuple(common_config.quantiles) != REGISTERED_LEVELS:
        raise ValueError("common configuration quantiles differ")
    common_pin = extension_config["common_config"]
    assert isinstance(common_pin, dict)
    if common_config_sha != common_pin["sha256"]:
        raise ValueError("common configuration hash differs from extension contract")
    datasets, temporal_audit, spatial_protocol, dataset_identities = _validate_datasets(args)
    if recovery_controller is not None:
        protocol = recovery_controller.request.binding.protocol_identity
        from scripts.run_baseline_extension_remediation import validate_prepared_execution_identity

        validate_prepared_execution_identity(dict(protocol), dataset_identities, common_config_sha)
    train_data = datasets["train"]
    validation_data = datasets["validation"]
    resolved_config = _resolved_model_config(args.name, train_data)
    base_model = build_extension_model(args.name, resolved_config)
    model = _QuantileMappingWrapper(base_model)
    criterion = ModernBaselineLoss(REGISTERED_LEVELS)
    training = common_config.training
    resolved_device = select_device(args.device)
    trainer = EarlyStoppingTrainer(
        model,
        criterion,
        device=resolved_device,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        max_epochs=training.max_epochs,
        patience=training.patience,
        gradient_clip_norm=training.gradient_clip_norm,
        mixed_precision=mixed_precision,
        checkpoint_path=None if recovery_controller is not None else args.output / "best.ckpt",
        checkpoint_metadata={
            **_prepared_output_metadata(dataset_identities),
            "baseline": args.name,
            "seed": args.seed,
            "quantiles": list(REGISTERED_LEVELS),
            "source_review_sha256": source_review_sha,
            "source_record": source_record,
            "selection_boundary": "validation_only",
            "calibration_or_test_loaded": False,
            "selection_metric": "daylight_qcrps",
            "train_descriptor_sha256": dataset_identities["descriptor_hashes"]["train"],
            "validation_descriptor_sha256": dataset_identities["descriptor_hashes"]["validation"],
            "train_window_content_sha256": dataset_identities["content_hashes"]["train"],
            "validation_window_content_sha256": dataset_identities["content_hashes"]["validation"],
            **precision_metadata,
        },
        selection_metric="daylight_qcrps",
    )
    train_loader = build_window_loader(train_data, training.batch_size, True, training.num_workers)
    validation_loader = build_window_loader(
        validation_data, training.batch_size, False, training.num_workers
    )

    output = args.output
    fit_kwargs: dict[str, object] = {}
    if recovery_controller is not None:
        output, initial_progress, commit = recovery_controller.prepare(
            trainer, train_loader, validation_loader
        )
        if config_profile == "fp32_remediation" and initial_progress is not None:
            _validate_fp32_continuation(initial_progress)
        fit_kwargs = {
            "initial_progress": initial_progress,
            "on_epoch_committed": commit,
        }
    else:
        output.mkdir(parents=True, exist_ok=False)
    staged_exports = config_profile == "fp32_remediation" and recovery_controller is not None
    restored_stages: set[str] = set()
    if staged_exports and getattr(args, "export_only", False):
        if initial_progress is None or initial_progress.terminal_reason not in {
            "max_epochs",
            "early_stopping",
        }:
            raise ValueError("export-only requires a terminal recoverable continuation")
        for stage in ("model", "validation", "summary"):
            if recovery_controller.restore_export_stage(stage, output):
                restored_stages.add(stage)
        if "summary" in restored_stages:
            if not callable(pre_publish_validator):
                raise ValueError("recoverable publication requires a pre-publish validator")
            pre_publish_validator(output)
            recovery_controller.publish_completed_exports(output)
            return recovery_controller.run_root
    resolved_run = {
        "schema_version": 1,
        "model_id": args.name,
        "seed": args.seed,
        "manifest_sha256": args.manifest_sha256,
        "common_config_sha256": common_config_sha,
        "extension_config_sha256": extension_config_sha,
        "source_review_sha256": source_review_sha,
        "resolved_model_config": resolved_config,
        **precision_metadata,
        **(
            {"prepared_contracts": dataset_identities["prepared_contracts"]}
            if "prepared_contracts" in dataset_identities
            else {}
        ),
    }
    artifact_resolved_config = {**resolved_config, **precision_metadata}
    if "model" not in restored_stages:
        _atomic_yaml_dump(_jsonable(artifact_resolved_config), output / "resolved_config.yaml")
        atomic_json_dump(resolved_run, output / "resolved_run_config.json")
    if getattr(args, "export_only", False):
        if recovery_controller is None or initial_progress is None:
            raise ValueError("export-only requires a terminal recoverable continuation")
        fit = _terminal_export_result(trainer, initial_progress)
    else:
        fit = trainer.fit(train_loader, validation_loader, **fit_kwargs)
    if recovery_controller is not None and "model" not in restored_stages:
        recovery_controller.materialize_best_checkpoint(output)

    bundle = {
        **_prepared_output_metadata(dataset_identities),
        "schema_version": 1,
        "model_state_dict": base_model.state_dict(),
        "baseline": args.name,
        "seed": args.seed,
        "quantiles": list(REGISTERED_LEVELS),
        "resolved_model_config": resolved_config,
        "source_record": source_record,
        "source_review_sha256": source_review_sha,
        "implementation_kind": "paper_derived_protocol_adaptation",
        **precision_metadata,
    }
    if "model" not in restored_stages:
        _atomic_torch_save(bundle, output / "model.pt")
    if "validation" not in restored_stages:
        predicted = _predict_validation(trainer, validation_loader)
        payload, state_semantics = _prediction_archive(
            name=args.name,
            dataset=validation_data,
            predicted=predicted,
            thresholds=_sample_ramp_thresholds(validation_data),
            batch_size=training.batch_size,
            workers=training.num_workers,
        )
        _reverify_dataset_identities(args, dataset_identities)
        prediction_metadata = {
            "schema_version": 1,
            "baseline": args.name,
            "seed": args.seed,
            "split": "validation",
            "selection_boundary": "validation_only",
            "calibration_or_test_loaded": False,
            "quantiles": list(REGISTERED_LEVELS),
            "state_prob_semantics": state_semantics,
            "source_review_sha256": source_review_sha,
            "source_npz_sha256": dataset_identities["descriptor_hashes"]["validation"],
            "source_window_content_sha256": dataset_identities["content_hashes"]["validation"],
            "site_id_map": validation_data.schema["site_id_map"],
            **(
                {"prepared_contract": dataset_identities["prepared_contracts"]["validation"]}
                if "prepared_contracts" in dataset_identities
                else {}
            ),
        }
        _atomic_save_predictions_npz(
            output / "validation_predictions.npz", payload, prediction_metadata
        )
        atomic_json_dump(prediction_metadata, output / "validation_predictions.json")
    descriptor_hashes = dataset_identities["descriptor_hashes"]
    content_hashes = dataset_identities["content_hashes"]
    run = {
        "schema_version": 1,
        "purpose": "training_validation_only_baseline_extension",
        "baseline": args.name,
        **(
            {"prepared_contracts": dataset_identities["prepared_contracts"]}
            if "prepared_contracts" in dataset_identities
            else {}
        ),
        "seed": args.seed,
        "selection_boundary": "validation_only",
        "diagnostic_only": False,
        "calibration_or_test_loaded": False,
        "selection_metric": "daylight_qcrps",
        "best_epoch": fit.best_epoch,
        "best_validation_loss": fit.best_validation_loss,
        "best_validation_metric_name": fit.best_validation_metric_name,
        "stopped_early": fit.stopped_early,
        "global_temporal_audit": temporal_audit,
        "registered_spatial_protocol": spatial_protocol,
        "source_review_sha256": source_review_sha,
        "source_record": source_record,
        "resolved_model_config": resolved_config,
        **precision_metadata,
        "environment": environment_report(),
        "artifacts": {
            "train_npz_sha256": descriptor_hashes["train"],
            "train_window_content_sha256": content_hashes["train"],
            "validation_npz_sha256": descriptor_hashes["validation"],
            "validation_window_content_sha256": content_hashes["validation"],
            "config_sha256": common_config_sha,
            "calibration_npz_sha256": None,
            "calibration_window_content_sha256": None,
            "test_npz_sha256": None,
            "test_window_content_sha256": None,
        },
        "model_sha256": sha256_file(output / "model.pt"),
        "best_checkpoint_sha256": sha256_file(output / "best.ckpt"),
    }
    if recovery_controller is not None:
        atomic_json_dump(_jsonable(artifact_resolved_config), output / "resolved-config.json")
        binding_bytes = recovery_controller.request.binding.raw_bytes
        if binding_bytes is None:
            raise ValueError("recovery binding bytes are unavailable")
        _write_bytes_exclusive(output / "recovery-binding.json", binding_bytes)
        _copy_exclusive(
            output / "validation_predictions.npz", output / "validation-predictions.npz"
        )
        _copy_exclusive(
            output / "validation_predictions.json",
            output / "validation-predictions.json",
        )
        _capture_execution_logs(args, output)
        audit_names = (
            "best.ckpt",
            "model.pt",
            "recovery-binding.json",
            "resolved-config.json",
            "resolved_config.yaml",
            "resolved_run_config.json",
            "stderr.log",
            "stdout.log",
            "validation-predictions.json",
            "validation-predictions.npz",
            "validation_predictions.json",
            "validation_predictions.npz",
        )
        atomic_json_dump(
            {
                "schema_version": 1,
                "purpose": "racepv_baseline_extension_artifact_audit",
                "manifest_sha256": args.manifest_sha256,
                "binding_sha256": recovery_controller.binding_sha256,
                "model_id": args.name,
                "seed": args.seed,
                "files": [
                    {"path": name, "sha256": sha256_file(output / name)} for name in audit_names
                ],
            },
            output / "artifact-audit.json",
        )
        artifact_audit_sha256 = sha256_file(output / "artifact-audit.json")
        atomic_json_dump(
            {
                "schema_version": 1,
                "purpose": "racepv_baseline_extension_execution_state",
                "manifest_sha256": args.manifest_sha256,
                "binding_sha256": recovery_controller.binding_sha256,
                "artifact_audit_sha256": artifact_audit_sha256,
                "model_id": args.name,
                "seed": args.seed,
                "status": "completed",
                "export_only": bool(getattr(args, "export_only", False)),
            },
            output / "execution-state.json",
        )
        execution_state_sha256 = sha256_file(output / "execution-state.json")
        run["extension_completion"] = {
            "artifact_audit_sha256": artifact_audit_sha256,
            "execution_state_sha256": execution_state_sha256,
        }
        atomic_json_dump(run, output / "run.json")
        if not callable(pre_publish_validator):
            raise ValueError("recoverable publication requires a pre-publish validator")
        pre_publish_validator(output)
        recovery_controller.publish_completed_exports(output)
        return recovery_controller.run_root
    atomic_json_dump(run, output / "run.json")
    return output
