"""Train one fixed CycleNet-QR cell on train/validation only."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from torch import Tensor, nn

from racepv.baselines.modern import ModernBaselineLoss
from racepv.config import QUANTILE_LEVELS
from racepv.training import EarlyStoppingTrainer
from racepv.utils import seed_everything, select_device
from scripts.cyclenet_validation_contract import (
    MODEL_REVISION,
    SEEDS,
    canonical_json_bytes,
    load_cyclenet_campaign_bytes,
    validate_cyclenet_model_config,
)
from scripts.racepv_reference_io import SafeReader, strict_json

REGISTERED_LEVELS = tuple(QUANTILE_LEVELS)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class _QuantileMappingWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, **inputs: Tensor) -> dict[str, Tensor]:
        output = self.model(**inputs)
        if not isinstance(output, Tensor):
            raise TypeError("CycleNet adapter must return one quantile tensor")
        return {"quantiles": output}


def _digest(value: object, *, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{role} must be a lowercase SHA-256")
    return value


def write_cyclenet_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    """Write one canonical CycleNet JSON artifact without replacement."""

    from racepv.cli.extension_baseline import _write_bytes_exclusive

    _write_bytes_exclusive(path, canonical_json_bytes(payload))


def write_cyclenet_resolved_config_pair_exclusive(
    output: Path, payload: Mapping[str, object]
) -> None:
    """Write the intake JSON and recovery-stage YAML-compatible config aliases."""

    write_cyclenet_json_exclusive(output / "resolved-config.json", payload)
    write_cyclenet_json_exclusive(output / "resolved_config.yaml", payload)


def save_cyclenet_predictions_npz_exclusive(
    destination: Path,
    tensors: Mapping[str, Tensor | np.ndarray],
    metadata: Mapping[str, object],
) -> None:
    """Publish one NPZ and its canonical JSON sidecar without replacement."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_suffix(".json")
    npz_temporary: Path | None = None
    json_temporary: Path | None = None
    npz_published = False
    json_published = False
    try:
        npz_fd, npz_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        npz_temporary = Path(npz_name)
        arrays = {
            name: value.detach().cpu().numpy() if isinstance(value, Tensor) else np.asarray(value)
            for name, value in tensors.items()
        }
        with os.fdopen(npz_fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            getattr(os, "fdatasync", os.fsync)(handle.fileno())

        json_fd, json_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{sidecar.name}.",
            suffix=".tmp",
        )
        json_temporary = Path(json_name)
        with os.fdopen(json_fd, "wb") as handle:
            handle.write(canonical_json_bytes(metadata))
            handle.flush()
            getattr(os, "fdatasync", os.fsync)(handle.fileno())

        os.link(npz_temporary, destination, follow_symlinks=False)
        npz_published = True
        os.link(json_temporary, sidecar, follow_symlinks=False)
        json_published = True
    except Exception:
        if json_published:
            sidecar.unlink(missing_ok=True)
        if npz_published:
            destination.unlink(missing_ok=True)
        raise
    finally:
        if npz_temporary is not None:
            npz_temporary.unlink(missing_ok=True)
        if json_temporary is not None:
            json_temporary.unlink(missing_ok=True)


def cyclenet_run_artifacts(protocol: Mapping[str, str]) -> dict[str, str | None]:
    """Expose CycleNet and shared recovery names for the same sealed identities."""

    return {
        **protocol,
        "train_npz_sha256": protocol["train_descriptor_sha256"],
        "train_window_content_sha256": protocol["train_content_sha256"],
        "validation_npz_sha256": protocol["validation_descriptor_sha256"],
        "validation_window_content_sha256": protocol["validation_content_sha256"],
        "config_sha256": protocol["common_config_sha256"],
        "calibration_npz_sha256": None,
        "calibration_window_content_sha256": None,
        "test_npz_sha256": None,
        "test_window_content_sha256": None,
    }


def cyclenet_checkpoint_metadata(
    *,
    seed: int,
    source_capsule_sha256: str,
    model_config_sha256: str,
    manifest_sha256: str,
    binding_sha256: str,
    dataset_identities=None,
) -> dict[str, object]:
    """Build the exact identity attached to every checkpoint and export."""

    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("CycleNet seed differs")
    result = {
        "baseline": "cyclenet_qr",
        "model_revision": MODEL_REVISION,
        "seed": seed,
        "source_capsule_sha256": _digest(source_capsule_sha256, role="source capsule"),
        "model_config_sha256": _digest(model_config_sha256, role="model config"),
        "manifest_sha256": _digest(manifest_sha256, role="manifest"),
        "binding_sha256": _digest(binding_sha256, role="binding"),
        "train_descriptor_sha256": (
            "652586ce411a393dfca6e4c1a884debd06c698be8749b893431b4b765bdd2559"
        ),
        "train_window_content_sha256": (
            "7b51eb990737a4de6b2258c5388d81170f0c8aa104f6c978ccf17cfcb56e5f37"
        ),
        "validation_descriptor_sha256": (
            "11a49549d9bff5a434bfb2bd62fc01595cbaf6e4213e90a80d99c8116be0b8fd"
        ),
        "validation_window_content_sha256": (
            "5fed358f7cf1d0db3fd0f8f5d3013e8757d6a7ccf324225de4289ad4ffc1bcd7"
        ),
        "precision_mode": "fp32",
        "autocast_enabled": False,
        "grad_scaler_enabled": False,
        "selection_boundary": "validation_only",
        "calibration_or_test_loaded": False,
        "selection_metric": "daylight_qcrps",
        "quantiles": list(REGISTERED_LEVELS),
    }
    if dataset_identities is not None:
        from racepv.baselines.extension_registry import validate_prepared_identity

        contracts = dataset_identities.get("prepared_contracts")
        if type(contracts) is not dict or set(contracts) != {"train", "validation"}:
            raise ValueError("CycleNet v2 checkpoint requires verified prepared identities")
        result["prepared_contracts"] = {}
        for split in ("train", "validation"):
            identity = validate_prepared_identity(contracts[split])
            if identity["split"] != split or identity["array_semantics_verified"] is not True:
                raise ValueError("CycleNet checkpoint requires current verified dataset semantics")
            result["prepared_contracts"][split] = identity
            result[f"{split}_descriptor_sha256"] = identity["descriptor_sha256"]
            result[f"{split}_window_content_sha256"] = identity["aggregate_content_sha256"]
        result["prepared_schema_version"] = 2
    return result


def build_cyclenet_trainer(
    model: nn.Module | object,
    *,
    device: str,
    checkpoint_metadata: Mapping[str, object],
) -> Any:
    """Construct only the frozen FP32 trainer contract."""

    return EarlyStoppingTrainer(
        model,
        ModernBaselineLoss(REGISTERED_LEVELS),
        device=device,
        learning_rate=0.001,
        weight_decay=0.0001,
        max_epochs=8,
        patience=2,
        gradient_clip_norm=1.0,
        mixed_precision=False,
        checkpoint_path=None,
        checkpoint_metadata=dict(checkpoint_metadata),
        selection_metric="daylight_qcrps",
    )


def run_cyclenet_training(
    args: argparse.Namespace,
    *,
    recovery_controller: Any | None = None,
    pre_publish_validator: Any | None = None,
) -> Path:
    """Execute one validated cell through the shared recoverable trainer."""

    from racepv.cli.modern_baseline import _preflight_prepared_inputs

    _preflight_prepared_inputs(args)

    if getattr(args, "name", None) != "cyclenet_qr":
        raise ValueError("CycleNet model identity differs")
    if type(getattr(args, "seed", None)) is not int or args.seed not in SEEDS:
        raise ValueError("CycleNet seed differs")
    if getattr(args, "validation_only", None) is not True:
        raise ValueError("--validation-only is required")
    if recovery_controller is None:
        raise ValueError("CycleNet formal training requires a recovery controller")
    project_root = getattr(args, "project_root", None)
    if not isinstance(project_root, Path) or not project_root.is_absolute():
        raise ValueError("absolute project root is required")
    expected_paths = {
        "model_config": "runs/cyclenet-validation-20260912-v1/model-config.json",
        "manifest": "runs/cyclenet-validation-20260912-v1/manifest.json",
        "config": "configs/final_day1_4090.yaml",
    }
    contents: dict[str, bytes] = {}
    observed: dict[str, str] = {}
    with SafeReader(project_root) as reader:
        for role, relative in expected_paths.items():
            supplied = getattr(args, role, None)
            if supplied != project_root.joinpath(*relative.split("/")):
                raise ValueError(f"CycleNet {role} path differs")
            content, receipt = reader.read(relative, role=f"CycleNet {role}")
            contents[role] = content
            observed[role] = receipt.sha256
    for role in ("model_config", "manifest"):
        if observed[role] != _digest(getattr(args, f"{role}_sha256", None), role=role):
            raise ValueError(f"CycleNet {role} hash differs")
    if observed["config"] != ("8a1d7eb02b4773d200c759050c9e7e11d8f3f2da9698539cbda7608d1101e1b0"):
        raise ValueError("CycleNet common config hash differs")
    model_config = validate_cyclenet_model_config(
        strict_json(contents["model_config"], role="CycleNet model config")
    )
    if contents["model_config"] != canonical_json_bytes(model_config):
        raise ValueError("CycleNet model config is not canonical JSON")
    manifest = load_cyclenet_campaign_bytes(contents["manifest"])
    if args.seed not in manifest["seeds"]:
        raise ValueError("CycleNet manifest seed differs")
    binding = recovery_controller.request.binding
    if (
        binding.seed != args.seed
        or binding.raw.get("model_revision") != MODEL_REVISION
        or binding.raw.get("manifest_sha256") != observed["manifest"]
        or binding.raw.get("model_config_sha256") != observed["model_config"]
    ):
        raise ValueError("CycleNet recovery binding identity differs")

    from racepv.audit import sha256_file
    from racepv.baselines.cyclenet_qr import build_cyclenet_qr
    from racepv.cli.extension_baseline import (
        _capture_execution_logs,
        _copy_exclusive,
        _terminal_export_result,
        _write_bytes_exclusive,
    )
    from racepv.cli.modern_baseline import (
        _atomic_torch_save,
        _jsonable,
        _predict_validation,
        _prediction_archive,
        _reverify_dataset_identities,
        _sample_ramp_thresholds,
        _validate_datasets,
    )
    from racepv.environment import environment_report
    from racepv.training import build_window_loader

    seed_everything(args.seed)
    datasets, temporal_audit, spatial_protocol, dataset_identities = _validate_datasets(args)
    from racepv.baselines.extension_registry import prepared_protocol_from_identities

    protocol = prepared_protocol_from_identities(dataset_identities, observed["config"])
    if dict(binding.protocol_identity) != protocol:
        raise ValueError("loaded data/config identity differs from recovery binding")
    source_sha = model_config["source_capsule_pin"]["sha256"]
    metadata = cyclenet_checkpoint_metadata(
        seed=args.seed,
        source_capsule_sha256=str(source_sha),
        model_config_sha256=observed["model_config"],
        manifest_sha256=observed["manifest"],
        binding_sha256=binding.raw_sha256,
        dataset_identities=dataset_identities
        if "prepared_contracts" in dataset_identities
        else None,
    )
    base_model = build_cyclenet_qr(project_root, model_config)
    trainer = build_cyclenet_trainer(
        _QuantileMappingWrapper(base_model),
        device=select_device(args.device),
        checkpoint_metadata=metadata,
    )
    train_data = datasets["train"]
    validation_data = datasets["validation"]
    train_loader = build_window_loader(train_data, 64, True, 0)
    validation_loader = build_window_loader(validation_data, 64, False, 0)
    output, initial_progress, commit = recovery_controller.prepare(
        trainer, train_loader, validation_loader
    )
    restored_stages: set[str] = set()
    if getattr(args, "export_only", False):
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
                raise ValueError("recoverable publication requires validation")
            pre_publish_validator(output)
            recovery_controller.publish_completed_exports(output)
            return recovery_controller.run_root

    if "model" not in restored_stages:
        write_cyclenet_resolved_config_pair_exclusive(output, {**model_config, **metadata})
        write_cyclenet_json_exclusive(
            output / "resolved_run_config.json",
            {
                "schema_version": 1,
                "model_id": "cyclenet_qr",
                "model_revision": MODEL_REVISION,
                "seed": args.seed,
                **metadata,
            },
        )
    if getattr(args, "export_only", False):
        fit = _terminal_export_result(trainer, initial_progress)
    else:
        fit = trainer.fit(
            train_loader,
            validation_loader,
            initial_progress=initial_progress,
            on_epoch_committed=commit,
        )
    if "model" not in restored_stages:
        recovery_controller.materialize_best_checkpoint(output)
        _atomic_torch_save(
            {
                "schema_version": 1,
                "model_state_dict": base_model.state_dict(),
                "model_id": "cyclenet_qr",
                "model_revision": MODEL_REVISION,
                "seed": args.seed,
                "resolved_model_config": model_config,
                **metadata,
            },
            output / "model.pt",
        )
    if "validation" not in restored_stages:
        predicted = _predict_validation(trainer, validation_loader)
        payload, state_semantics = _prediction_archive(
            name="cyclenet_qr",
            dataset=validation_data,
            predicted=predicted,
            thresholds=_sample_ramp_thresholds(validation_data),
            batch_size=64,
            workers=0,
        )
        _reverify_dataset_identities(args, dataset_identities)
        prediction_metadata = {
            **(
                {"prepared_contract": protocol["prepared_contracts"]["validation"]}
                if "prepared_contracts" in protocol
                else {}
            ),
            "schema_version": 1,
            "model_id": "cyclenet_qr",
            "model_revision": MODEL_REVISION,
            "seed": args.seed,
            "split": "validation",
            "state_prob_semantics": state_semantics,
            "source_npz_sha256": protocol["validation_descriptor_sha256"],
            "source_window_content_sha256": protocol["validation_content_sha256"],
            "site_id_map": validation_data.schema["site_id_map"],
            **metadata,
        }
        save_cyclenet_predictions_npz_exclusive(
            output / "validation_predictions.npz", payload, prediction_metadata
        )
    if "summary" not in restored_stages:
        binding_bytes = binding.raw_bytes
        if binding_bytes is None:
            raise ValueError("recovery binding bytes are unavailable")
        _write_bytes_exclusive(output / "recovery-binding.json", binding_bytes)
        _copy_exclusive(
            output / "validation_predictions.npz", output / "validation-predictions.npz"
        )
        _copy_exclusive(
            output / "validation_predictions.json", output / "validation-predictions.json"
        )
        _capture_execution_logs(args, output)
        artifact_names = (
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
        audit = {
            "schema_version": 1,
            "purpose": "racepv_cyclenet_validation_artifact_audit",
            "manifest_sha256": observed["manifest"],
            "binding_sha256": binding.raw_sha256,
            "model_id": "cyclenet_qr",
            "model_revision": MODEL_REVISION,
            "seed": args.seed,
            "files": [
                {"path": name, "sha256": sha256_file(output / name)} for name in artifact_names
            ],
        }
        write_cyclenet_json_exclusive(output / "artifact-audit.json", audit)
        write_cyclenet_json_exclusive(
            output / "execution-state.json",
            {
                "schema_version": 1,
                "purpose": "racepv_cyclenet_validation_execution_state",
                "manifest_sha256": observed["manifest"],
                "binding_sha256": binding.raw_sha256,
                "artifact_audit_sha256": sha256_file(output / "artifact-audit.json"),
                "model_id": "cyclenet_qr",
                "model_revision": MODEL_REVISION,
                "seed": args.seed,
                "status": "completed",
                "export_only": bool(getattr(args, "export_only", False)),
                **metadata,
            },
        )
        run = {
            "schema_version": 1,
            "purpose": "training_validation_only_cyclenet_qr",
            "model_id": "cyclenet_qr",
            "model_revision": MODEL_REVISION,
            "seed": args.seed,
            "best_epoch": fit.best_epoch,
            "best_validation_loss": fit.best_validation_loss,
            "best_validation_metric_name": fit.best_validation_metric_name,
            "stopped_early": fit.stopped_early,
            "global_temporal_audit": temporal_audit,
            "registered_spatial_protocol": spatial_protocol,
            "environment": environment_report(),
            "diagnostic_only": False,
            "artifacts": cyclenet_run_artifacts(protocol),
            "model_sha256": sha256_file(output / "model.pt"),
            "best_checkpoint_sha256": sha256_file(output / "best.ckpt"),
            **metadata,
        }
        write_cyclenet_json_exclusive(output / "run.json", _jsonable(run))
    if not callable(pre_publish_validator):
        raise ValueError("recoverable publication requires validation")
    pre_publish_validator(output)
    recovery_controller.publish_completed_exports(output)
    return recovery_controller.run_root
