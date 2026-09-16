"""Fail-closed serialization and inspection primitives for recoverable training."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import stat
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from .engine import EarlyStoppingTrainer, EpochRecord, TrainerProgress

_HEX = frozenset("0123456789abcdef")
_HELDOUT_COMPONENTS = frozenset({"calibration", "test"})
_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "purpose",
        "run_id",
        "seed",
        "selection_boundary",
        "heldout_evidence_used",
        "reference_policy",
        "protocol_identity",
        "source_inventory",
        "resolved_training_contract",
        "feature_contract",
        "runtime_contract",
    }
)
_PIN_KEYS = frozenset({"path", "sha256"})
_PROTOCOL_KEYS = frozenset(
    {
        "train_descriptor_sha256",
        "train_content_sha256",
        "validation_descriptor_sha256",
        "validation_content_sha256",
        "common_config_sha256",
    }
)
_RUNTIME_KEYS = frozenset(
    {
        "python_version",
        "torch_version",
        "cuda_version",
        "mamba_ssm_version",
        "dependency_freeze_sha256",
        "device_type",
        "gpu_model",
        "compute_capability",
        "driver_version",
        "num_devices",
        "num_workers",
        "sampler",
        "drop_last",
        "amp_enabled",
        "determinism_settings",
    }
)
_DETERMINISM_KEYS = frozenset(
    {
        "deterministic_algorithms",
        "deterministic_warn_only",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "matmul_allow_tf32",
        "cudnn_allow_tf32",
        "cublas_workspace_config",
        "pythonhashseed",
    }
)
_COMMIT_KEYS = frozenset(
    {
        "schema_version",
        "purpose",
        "generation",
        "binding_sha256",
        "previous_commit_sha256",
        "state_sha256",
    }
)
_STATE_KEYS = frozenset(
    {
        "schema_version",
        "purpose",
        "binding_sha256",
        "generation",
        "next_epoch",
        "terminal_reason",
        "current_model_state",
        "best_checkpoint",
        "optimizer_state",
        "scaler_state",
        "rng_state",
        "history",
        "stale_epochs",
        "loader_contract",
    }
)
_HISTORY_KEYS = frozenset(
    {
        "epoch",
        "train_loss",
        "validation_loss",
        "seconds",
        "train_metrics",
        "validation_metrics",
        "validation_selection_metric",
    }
)


class RecoveryError(RuntimeError):
    """Base error for the recoverable-run subsystem."""


class RecoveryUnsupportedError(RecoveryError):
    """The supplied run cannot be represented by the recovery protocol."""


class RecoveryIntegrityError(RecoveryError):
    """A schema, identity, or filesystem integrity check failed."""


class RecoveryBusyError(RecoveryError):
    """Another writer currently owns the run."""


@dataclass(frozen=True)
class SafeBytes:
    data: bytes
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class RunBinding:
    raw_sha256: str
    run_id: str
    seed: int
    protocol_identity: Mapping[str, str]
    runtime_contract: Mapping[str, object]
    raw: Mapping[str, object]
    binding_root: Path | None = None
    raw_bytes: bytes | None = None


@dataclass(frozen=True)
class CommittedGeneration:
    generation: int
    commit_sha256: str
    commit: Mapping[str, object]
    state: Mapping[str, object]


def capture_rng_state() -> dict[str, object]:
    """Capture RNG state without initializing CUDA in a CPU-only control path."""

    algorithm, values, position, has_gauss, cached_gaussian = np.random.get_state()
    cuda_states: list[torch.Tensor] = []
    if torch.cuda.is_initialized():
        cuda_states = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": algorithm,
            "values": torch.tensor(values.astype("int64"), dtype=torch.int64),
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": cuda_states,
    }


def restore_rng_state(payload: Mapping[str, object]) -> None:
    """Restore validated Python, NumPy, CPU Torch and initialized CUDA RNG state."""

    if frozenset(payload) != frozenset({"python", "numpy", "torch_cpu", "torch_cuda"}):
        raise RecoveryIntegrityError("RNG state keys differ")
    numpy_state = payload["numpy"]
    if not isinstance(numpy_state, dict) or frozenset(numpy_state) != frozenset(
        {"algorithm", "values", "position", "has_gauss", "cached_gaussian"}
    ):
        raise RecoveryIntegrityError("NumPy RNG state is invalid")
    values = numpy_state["values"]
    torch_cpu = payload["torch_cpu"]
    cuda_states = payload["torch_cuda"]
    if not isinstance(values, torch.Tensor) or values.dtype != torch.int64 or values.ndim != 1:
        raise RecoveryIntegrityError("NumPy RNG values must be a one-dimensional int64 tensor")
    if not isinstance(torch_cpu, torch.Tensor) or torch_cpu.dtype != torch.uint8:
        raise RecoveryIntegrityError("Torch CPU RNG state must be a uint8 tensor")
    if not isinstance(cuda_states, list) or not all(
        isinstance(state, torch.Tensor) and state.dtype == torch.uint8 for state in cuda_states
    ):
        raise RecoveryIntegrityError("Torch CUDA RNG states must be uint8 tensors")
    random.setstate(payload["python"])
    np.random.set_state(
        (
            str(numpy_state["algorithm"]),
            values.numpy().astype("uint32"),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(torch_cpu.cpu())
    if cuda_states:
        if not torch.cuda.is_available():
            raise RecoveryUnsupportedError("saved CUDA RNG state cannot be restored")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def encode_progress(
    progress: TrainerProgress,
    binding_sha256: str,
    loader_contract: Mapping[str, object],
) -> dict[str, object]:
    _require_sha256(binding_sha256, "binding SHA-256")
    if progress.next_epoch < 0 or progress.stale_epochs < 0:
        raise RecoveryIntegrityError("trainer progress counters must be non-negative")
    return {
        "schema_version": 1,
        "purpose": "racepv_epoch_state",
        "binding_sha256": binding_sha256,
        "generation": progress.next_epoch,
        "next_epoch": progress.next_epoch,
        "terminal_reason": progress.terminal_reason,
        "current_model_state": progress.current_model_state,
        "best_checkpoint": progress.best_checkpoint,
        "optimizer_state": progress.optimizer_state,
        "scaler_state": progress.scaler_state,
        "rng_state": capture_rng_state(),
        "history": [asdict(record) for record in progress.history],
        "stale_epochs": progress.stale_epochs,
        "loader_contract": dict(loader_contract),
    }


def decode_progress(
    payload: Mapping[str, object],
    *,
    binding_sha256: str,
    loader_contract: Mapping[str, object],
    trainer: EarlyStoppingTrainer | None = None,
) -> TrainerProgress:
    if not isinstance(payload, dict):
        raise RecoveryIntegrityError("epoch state must be a mapping")
    _require_exact_keys(payload, _STATE_KEYS, "epoch state")
    if payload["schema_version"] != 1 or isinstance(payload["schema_version"], bool):
        raise RecoveryIntegrityError("epoch state schema_version must be integer 1")
    if payload["purpose"] != "racepv_epoch_state":
        raise RecoveryIntegrityError("epoch state purpose is invalid")
    if payload["binding_sha256"] != binding_sha256:
        raise RecoveryIntegrityError("epoch state binding mismatch")
    generation = payload["generation"]
    next_epoch = payload["next_epoch"]
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or isinstance(next_epoch, bool)
        or not isinstance(next_epoch, int)
        or generation < 0
        or next_epoch != generation
    ):
        raise RecoveryIntegrityError("epoch state generation is invalid")
    if payload["loader_contract"] != dict(loader_contract):
        raise RecoveryIntegrityError("loader contract drifted")
    terminal = payload["terminal_reason"]
    if terminal not in {None, "early_stopping", "max_epochs"}:
        raise RecoveryIntegrityError("terminal reason is invalid")
    stale_epochs = payload["stale_epochs"]
    if isinstance(stale_epochs, bool) or not isinstance(stale_epochs, int) or stale_epochs < 0:
        raise RecoveryIntegrityError("stale epoch count is invalid")
    raw_history = payload["history"]
    if not isinstance(raw_history, list) or len(raw_history) != next_epoch:
        raise RecoveryIntegrityError("history length differs from next epoch")
    history: list[EpochRecord] = []
    for expected_epoch, item in enumerate(raw_history):
        if not isinstance(item, dict):
            raise RecoveryIntegrityError("history item must be a mapping")
        _require_exact_keys(item, _HISTORY_KEYS, "history item")
        if item["epoch"] != expected_epoch or isinstance(item["epoch"], bool):
            raise RecoveryIntegrityError("history epochs must be consecutive")
        for field in (
            "train_loss",
            "validation_loss",
            "seconds",
            "validation_selection_metric",
        ):
            _require_finite_number(item[field], f"history.{field}")
        if float(item["seconds"]) < 0:
            raise RecoveryIntegrityError("history seconds must be non-negative")
        for metrics_field in ("train_metrics", "validation_metrics"):
            metrics = item[metrics_field]
            if not isinstance(metrics, dict) or not all(
                isinstance(name, str) and _is_finite_number(value)
                for name, value in metrics.items()
            ):
                raise RecoveryIntegrityError(f"history {metrics_field} is invalid")
        history.append(EpochRecord(**item))
    model_state = payload["current_model_state"]
    optimizer_state = payload["optimizer_state"]
    scaler_state = payload["scaler_state"]
    best_checkpoint = payload["best_checkpoint"]
    if not isinstance(model_state, dict) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in model_state.items()
    ):
        raise RecoveryIntegrityError("current model state is invalid")
    if not isinstance(optimizer_state, dict) or not isinstance(scaler_state, dict):
        raise RecoveryIntegrityError("optimizer or scaler state is invalid")
    if best_checkpoint is not None and not isinstance(best_checkpoint, dict):
        raise RecoveryIntegrityError("best checkpoint is invalid")
    best_epoch = -1
    best_loss = float("inf")
    if best_checkpoint is not None:
        try:
            best_epoch = int(best_checkpoint["epoch"])
            best_loss = float(best_checkpoint["validation_selection_metric"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RecoveryIntegrityError("best checkpoint selection is invalid") from exc
        if not math.isfinite(best_loss):
            raise RecoveryIntegrityError("best checkpoint selection is non-finite")
    if history:
        selected_epoch = min(
            range(len(history)),
            key=lambda index: history[index].validation_selection_metric,
        )
        selected_loss = history[selected_epoch].validation_selection_metric
        expected_stale = len(history) - selected_epoch - 1
        if best_epoch != selected_epoch or best_loss != selected_loss:
            raise RecoveryIntegrityError("best checkpoint differs from strict history selection")
        if stale_epochs != expected_stale:
            raise RecoveryIntegrityError("stale epoch count differs from history")
    elif best_checkpoint is not None or stale_epochs != 0 or terminal is not None:
        raise RecoveryIntegrityError("empty history has inconsistent selection state")
    if trainer is not None:
        _validate_progress_against_trainer(
            trainer,
            model_state=model_state,
            optimizer_state=optimizer_state,
            scaler_state=scaler_state,
            best_checkpoint=best_checkpoint,
            history=history,
            terminal_reason=terminal,
            stale_epochs=stale_epochs,
        )
    return TrainerProgress(
        next_epoch=next_epoch,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        stale_epochs=stale_epochs,
        history=history,
        terminal_reason=terminal,
        current_model_state=model_state,
        best_checkpoint=best_checkpoint,
        optimizer_state=optimizer_state,
        scaler_state=scaler_state,
    )


def _is_finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _require_finite_number(value: object, role: str) -> float:
    if not _is_finite_number(value):
        raise RecoveryIntegrityError(f"{role} must be finite")
    return float(value)


def _validate_tensor_state(
    observed: Mapping[str, object],
    expected: Mapping[str, torch.Tensor],
    role: str,
) -> None:
    if set(observed) != set(expected):
        raise RecoveryIntegrityError(f"{role} keys differ")
    for name, expected_tensor in expected.items():
        tensor = observed[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != expected_tensor.shape
            or tensor.dtype != expected_tensor.dtype
        ):
            raise RecoveryIntegrityError(f"{role} tensor identity differs: {name}")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise RecoveryIntegrityError(f"{role} tensor is non-finite: {name}")


def _validate_optimizer_state_dict(
    observed: object,
    expected: Mapping[str, object],
    role: str,
) -> None:
    if not isinstance(observed, dict) or frozenset(observed) != {"state", "param_groups"}:
        raise RecoveryIntegrityError(f"{role} optimizer state is invalid")
    groups = observed["param_groups"]
    expected_groups = expected["param_groups"]
    if not isinstance(groups, list) or len(groups) != len(expected_groups):
        raise RecoveryIntegrityError(f"{role} optimizer parameter groups differ")
    valid_parameter_ids: set[int] = set()
    for group, expected_group in zip(groups, expected_groups, strict=True):
        if not isinstance(group, dict) or set(group) != set(expected_group):
            raise RecoveryIntegrityError(f"{role} optimizer group schema differs")
        if len(group["params"]) != len(expected_group["params"]):
            raise RecoveryIntegrityError(f"{role} optimizer parameter groups differ")
        valid_parameter_ids.update(group["params"])
        for key in set(group) - {"params"}:
            if group[key] != expected_group[key]:
                raise RecoveryIntegrityError(f"{role} optimizer hyperparameters differ")
    state = observed["state"]
    if not isinstance(state, dict) or any(key not in valid_parameter_ids for key in state):
        raise RecoveryIntegrityError(f"{role} optimizer state membership differs")
    for parameter_state in state.values():
        if not isinstance(parameter_state, dict):
            raise RecoveryIntegrityError(f"{role} optimizer parameter state is invalid")
        for value in parameter_state.values():
            if isinstance(value, torch.Tensor) and (
                value.is_floating_point() or value.is_complex()
            ):
                if not bool(torch.isfinite(value).all()):
                    raise RecoveryIntegrityError(f"{role} optimizer tensor is non-finite")


def _validate_progress_against_trainer(
    trainer: EarlyStoppingTrainer,
    *,
    model_state: Mapping[str, object],
    optimizer_state: Mapping[str, object],
    scaler_state: Mapping[str, object],
    best_checkpoint: Mapping[str, object] | None,
    history: list[EpochRecord],
    terminal_reason: str | None,
    stale_epochs: int,
) -> None:
    _validate_tensor_state(model_state, trainer.model.state_dict(), "model state")
    expected_optimizer = trainer._optimizer_payload()
    if trainer.separate_state_optimizer:
        expected_keys = {
            "optimizer_schema",
            "forecast_optimizer_state_dict",
            "state_optimizer_state_dict",
            "forecast_parameter_names",
            "state_parameter_names",
        }
        if set(optimizer_state) != expected_keys:
            raise RecoveryIntegrityError("optimizer schema keys differ")
        for names in ("forecast_parameter_names", "state_parameter_names"):
            if optimizer_state[names] != expected_optimizer[names]:
                raise RecoveryIntegrityError("optimizer named parameter groups differ")
        for role in ("forecast", "state"):
            key = f"{role}_optimizer_state_dict"
            _validate_optimizer_state_dict(optimizer_state[key], expected_optimizer[key], role)
    else:
        expected_keys = {"optimizer_schema", "optimizer_state_dict", "parameter_names"}
        if set(optimizer_state) != expected_keys:
            raise RecoveryIntegrityError("optimizer schema keys differ")
        if optimizer_state["parameter_names"] != expected_optimizer["parameter_names"]:
            raise RecoveryIntegrityError("optimizer named parameter groups differ")
        _validate_optimizer_state_dict(
            optimizer_state["optimizer_state_dict"],
            expected_optimizer["optimizer_state_dict"],
            "single",
        )
    if optimizer_state["optimizer_schema"] != expected_optimizer["optimizer_schema"]:
        raise RecoveryIntegrityError("optimizer schema differs")
    expected_scaler = trainer.scaler.state_dict()
    if set(scaler_state) != set(expected_scaler):
        raise RecoveryIntegrityError("scaler state mode differs")
    if best_checkpoint is not None:
        checkpoint_model = best_checkpoint.get("model_state_dict")
        if not isinstance(checkpoint_model, dict):
            raise RecoveryIntegrityError("best checkpoint model state is invalid")
        _validate_tensor_state(
            checkpoint_model, trainer.model.state_dict(), "best checkpoint model"
        )
        if best_checkpoint.get("selection_metric") != trainer.selection_metric:
            raise RecoveryIntegrityError("best checkpoint selection metric differs")
    for record in history:
        if trainer.selection_metric not in record.validation_metrics:
            raise RecoveryIntegrityError("history selection metric is missing")
        if (
            record.validation_metrics[trainer.selection_metric]
            != record.validation_selection_metric
        ):
            raise RecoveryIntegrityError("history selection metric differs")
    if len(history) > trainer.max_epochs:
        raise RecoveryIntegrityError("history exceeds trainer epoch budget")
    expected_terminal: str | None = None
    if history and stale_epochs >= trainer.patience:
        expected_terminal = "early_stopping"
    elif len(history) == trainer.max_epochs:
        expected_terminal = "max_epochs"
    if terminal_reason != expected_terminal:
        raise RecoveryIntegrityError("terminal reason differs from trainer state")


def _require_exact_keys(value: Mapping[str, object], keys: frozenset[str], role: str) -> None:
    observed = frozenset(value)
    if observed != keys:
        raise RecoveryIntegrityError(
            f"{role} keys differ: missing={sorted(keys - observed)}, extra={sorted(observed - keys)}"
        )


def _require_sha256(value: object, role: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in _HEX for ch in value):
        raise RecoveryIntegrityError(f"{role} must be a lowercase SHA-256")
    return value


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RecoveryIntegrityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise RecoveryIntegrityError(f"non-finite JSON value: {value}")


def parse_json_bytes(data: bytes, role: str) -> Mapping[str, object]:
    try:
        decoded = data.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except RecoveryIntegrityError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryIntegrityError(f"invalid {role} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RecoveryIntegrityError(f"{role} must be a JSON object")
    return value


def validate_relative_path(value: str | Path) -> tuple[str, ...]:
    if not isinstance(value, (str, Path)):
        raise RecoveryIntegrityError("path must be a string or Path")
    raw = str(value)
    if not raw or "\\" in raw or "\x00" in raw:
        raise RecoveryIntegrityError("path contains a forbidden separator or byte")
    if raw.startswith("./") or "//" in raw:
        raise RecoveryIntegrityError("path contains a non-canonical separator")
    path = PurePosixPath(raw)
    if path.is_absolute():
        raise RecoveryIntegrityError("path must be relative")
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise RecoveryIntegrityError("path contains an unsafe component")
    if any(part.casefold() in _HELDOUT_COMPONENTS for part in parts):
        raise RecoveryIntegrityError("held-out path component is forbidden")
    return parts


def _open_absolute_directory(path: Path) -> int:
    absolute = path.absolute()
    if not absolute.is_absolute():
        raise RecoveryIntegrityError("root must be absolute")
    current = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in absolute.parts[1:]:
            next_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current,
            )
            os.close(current)
            current = next_fd
        return current
    except OSError as exc:
        os.close(current)
        raise RecoveryIntegrityError(f"unsafe root path: {path}") from exc


def read_regular_file_nofollow(root: Path, relative: Path) -> SafeBytes:
    parts = validate_relative_path(relative)
    directory = _open_absolute_directory(root)
    try:
        for component in parts[:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory,
            )
            os.close(directory)
            directory = next_fd
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        file_fd = os.open(parts[-1], flags, dir_fd=directory)
        try:
            before = os.fstat(file_fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise RecoveryIntegrityError("input must be a single-link regular file")
            chunks: list[bytes] = []
            while chunk := os.read(file_fd, 1024 * 1024):
                chunks.append(chunk)
            after = os.fstat(file_fd)
            entry = os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)

            def identity(item: os.stat_result) -> tuple[int, int, int, int, int]:
                return (
                    item.st_dev,
                    item.st_ino,
                    item.st_size,
                    item.st_mtime_ns,
                    item.st_ctime_ns,
                )

            if identity(before) != identity(after) or identity(after) != identity(entry):
                raise RecoveryIntegrityError("input changed while being read")
            data = b"".join(chunks)
            return SafeBytes(data, hashlib.sha256(data).hexdigest(), len(data))
        finally:
            os.close(file_fd)
    except RecoveryIntegrityError:
        raise
    except OSError as exc:
        raise RecoveryIntegrityError(f"unsafe or missing input: {relative}") from exc
    finally:
        os.close(directory)


def _validate_pin(value: object, role: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise RecoveryIntegrityError(f"{role} must be an object")
    _require_exact_keys(value, _PIN_KEYS, role)
    validate_relative_path(value["path"])
    _require_sha256(value["sha256"], f"{role}.sha256")
    return value


def load_binding(root: Path, path: Path, expected_sha256: str) -> RunBinding:
    raw = read_regular_file_nofollow(root, path)
    payload = parse_json_bytes(raw.data, "binding")
    _require_exact_keys(payload, _BINDING_KEYS, "binding")
    _require_sha256(expected_sha256, "expected binding SHA-256")
    if raw.sha256 != expected_sha256:
        raise RecoveryIntegrityError("binding SHA-256 mismatch")
    if payload["schema_version"] != 1 or isinstance(payload["schema_version"], bool):
        raise RecoveryIntegrityError("binding schema_version must be integer 1")
    if payload["purpose"] != "racepv_epoch_recovery_binding":
        raise RecoveryIntegrityError("binding purpose is invalid")
    if payload["selection_boundary"] != "validation_only":
        raise RecoveryIntegrityError("binding selection boundary must be validation_only")
    if payload["heldout_evidence_used"] is not False:
        raise RecoveryIntegrityError("binding must exclude held-out evidence")
    seed = payload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in {13, 29, 47, 71, 101}:
        raise RecoveryIntegrityError("binding seed is not a registered integer")
    run_id = payload["run_id"]
    if not isinstance(run_id, str) or not run_id:
        raise RecoveryIntegrityError("binding run_id must be non-empty")
    for role in (
        "reference_policy",
        "source_inventory",
        "resolved_training_contract",
        "feature_contract",
    ):
        _validate_pin(payload[role], role)
    protocol = payload["protocol_identity"]
    if not isinstance(protocol, dict):
        raise RecoveryIntegrityError("protocol_identity must be an object")
    _require_exact_keys(protocol, _PROTOCOL_KEYS, "protocol_identity")
    for key, value in protocol.items():
        _require_sha256(value, f"protocol_identity.{key}")
    runtime = payload["runtime_contract"]
    if not isinstance(runtime, dict):
        raise RecoveryIntegrityError("runtime_contract must be an object")
    _require_exact_keys(runtime, _RUNTIME_KEYS, "runtime_contract")
    for key in ("num_devices", "num_workers"):
        if isinstance(runtime[key], bool) or not isinstance(runtime[key], int):
            raise RecoveryIntegrityError(f"runtime_contract.{key} must be an integer")
    for key in (
        "python_version",
        "torch_version",
        "cuda_version",
        "mamba_ssm_version",
        "gpu_model",
        "compute_capability",
        "driver_version",
    ):
        if not isinstance(runtime[key], str) or not runtime[key]:
            raise RecoveryIntegrityError(f"runtime_contract.{key} must be non-empty")
    _require_sha256(runtime["dependency_freeze_sha256"], "dependency freeze SHA-256")
    if (
        runtime["device_type"] != "cuda"
        or runtime["num_devices"] != 1
        or runtime["num_workers"] != 0
        or runtime["sampler"] != "pytorch_default_random_sampler"
        or runtime["drop_last"] is not False
        or runtime["amp_enabled"] is not True
    ):
        raise RecoveryIntegrityError("runtime contract is not the sealed formal CUDA mode")
    determinism = runtime["determinism_settings"]
    if not isinstance(determinism, dict):
        raise RecoveryIntegrityError("determinism_settings must be an object")
    _require_exact_keys(determinism, _DETERMINISM_KEYS, "determinism_settings")
    expected_determinism = {
        "deterministic_algorithms": True,
        "deterministic_warn_only": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "cublas_workspace_config": ":4096:8",
        "pythonhashseed": str(seed),
    }
    if determinism != expected_determinism:
        raise RecoveryIntegrityError("runtime contract determinism settings differ")
    return RunBinding(
        raw_sha256=raw.sha256,
        run_id=run_id,
        seed=seed,
        protocol_identity=dict(protocol),
        runtime_contract=dict(runtime),
        raw=dict(payload),
        binding_root=root,
        raw_bytes=raw.data,
    )


def validate_runtime_contract(
    binding: RunBinding,
    observed: Mapping[str, object],
) -> None:
    """Require an execution environment to equal the externally approved binding."""

    if not isinstance(observed, dict) or frozenset(observed) != _RUNTIME_KEYS:
        raise RecoveryIntegrityError("observed runtime contract keys differ")
    if observed != dict(binding.runtime_contract):
        differing = sorted(
            key for key in _RUNTIME_KEYS if observed.get(key) != binding.runtime_contract.get(key)
        )
        raise RecoveryIntegrityError(f"runtime contract drifted: {','.join(differing)}")


def read_pinned_binding_inputs(binding: RunBinding) -> dict[str, SafeBytes]:
    """Read every scientific contract pin from its fixed binding root."""

    if binding.binding_root is None:
        raise RecoveryIntegrityError("binding root is unavailable")
    result: dict[str, SafeBytes] = {}
    for role in (
        "reference_policy",
        "source_inventory",
        "resolved_training_contract",
        "feature_contract",
    ):
        pin = binding.raw[role]
        if not isinstance(pin, dict):  # narrowed by load_binding
            raise RecoveryIntegrityError(f"{role} pin is invalid")
        relative = str(pin["path"])
        parts = validate_relative_path(relative)
        if parts[0] != "inputs":
            raise RecoveryIntegrityError("pinned inputs must be contained by inputs/")
        if relative in result:
            raise RecoveryIntegrityError("pinned input paths must be unique")
        observed = read_regular_file_nofollow(binding.binding_root, Path(relative))
        if observed.sha256 != pin["sha256"]:
            raise RecoveryIntegrityError(f"pinned input SHA-256 mismatch: {role}")
        result[relative] = observed
    reference_path = str(binding.raw["reference_policy"]["path"])
    for role, pin in _manifest_pins(result[reference_path].data).items():
        relative = str(pin["path"])
        if relative in result:
            raise RecoveryIntegrityError("policy dependency paths must be unique")
        observed = read_regular_file_nofollow(binding.binding_root, Path(relative))
        if observed.sha256 != pin["sha256"]:
            raise RecoveryIntegrityError(f"pinned input SHA-256 mismatch: {role}")
        result[relative] = observed
    return result


def _manifest_pins(policy_bytes: bytes) -> dict[str, Mapping[str, object]]:
    policy = parse_json_bytes(policy_bytes, "reference policy")
    manifests = policy.get("modern_manifests")
    if not isinstance(manifests, dict) or frozenset(manifests) != {"pilot", "multiseed"}:
        raise RecoveryIntegrityError("reference policy manifest pins differ")
    result: dict[str, Mapping[str, object]] = {}
    for phase in ("pilot", "multiseed"):
        pin = _validate_pin(manifests[phase], f"modern_manifests.{phase}")
        parts = validate_relative_path(str(pin["path"]))
        if parts[0] != "inputs":
            raise RecoveryIntegrityError("policy dependencies must be contained by inputs/")
        result[phase] = pin
    return result


def load_tensor_mapping(root: Path, relative: Path) -> Mapping[str, object]:
    raw = read_regular_file_nofollow(root, relative)
    try:
        payload = torch.load(io.BytesIO(raw.data), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RecoveryIntegrityError(f"unsafe or invalid tensor state: {relative}") from exc
    if not isinstance(payload, dict):
        raise RecoveryIntegrityError("tensor state must be a mapping")
    return payload


def validate_commit(payload: Mapping[str, object], generation: int, binding_sha256: str) -> None:
    _require_exact_keys(payload, _COMMIT_KEYS, "epoch commit")
    if payload["schema_version"] != 1 or isinstance(payload["schema_version"], bool):
        raise RecoveryIntegrityError("epoch commit schema_version must be integer 1")
    if payload["purpose"] != "racepv_epoch_commit":
        raise RecoveryIntegrityError("epoch commit purpose is invalid")
    if payload["generation"] != generation or isinstance(payload["generation"], bool):
        raise RecoveryIntegrityError("epoch commit generation mismatch")
    if payload["binding_sha256"] != binding_sha256:
        raise RecoveryIntegrityError("epoch commit binding mismatch")
    _require_sha256(payload["state_sha256"], "epoch state SHA-256")
