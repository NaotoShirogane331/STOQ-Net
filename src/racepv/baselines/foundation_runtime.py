"""Lazy, offline-only runtime backends for the sealed foundation baselines."""

from __future__ import annotations

import hashlib
import importlib
import os
import stat
import sys
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import numpy as np

from scripts.baseline_extension_sources import audit_checkout

_CHECKOUTS = {
    "timesfm3_zs": "third-party/baseline-extension/timesfm3_zs/src",
    "moirai_1_1_r_small_zs": "third-party/baseline-extension/moirai_1_1_r_small_zs/src",
}
_SOURCE_ROOTS = {
    model_id: f"third-party/baseline-extension/{model_id}"
    for model_id in ("timesfm3_zs", "sundial_base_zs", "moirai_1_1_r_small_zs")
}
_WEIGHT_ROOT = "model-cache/baseline-extension"


def _safe_parts(relative: str) -> tuple[str, ...]:
    if not isinstance(relative, str) or "\\" in relative or "\0" in relative:
        raise ValueError("foundation runtime path is unsafe")
    parts = tuple(relative.split("/"))
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("foundation runtime path is unsafe")
    return parts


def _open_directory_chain(root: Path, parts: tuple[str, ...]) -> tuple[int, list[int]]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    descriptors = [os.open(root, os.O_RDONLY | directory | nofollow)]
    current = descriptors[0]
    try:
        for component in parts:
            current = os.open(component, os.O_RDONLY | directory | nofollow, dir_fd=current)
            descriptors.append(current)
        return current, descriptors
    except OSError as exc:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise ValueError("foundation runtime path ancestor is unsafe") from exc


def _regular_sha256_at(project_root: Path, relative: str) -> str:
    parts = _safe_parts(relative)
    parent, descriptors = _open_directory_chain(project_root, parts[:-1])
    try:
        try:
            descriptor = os.open(
                parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent
            )
        except OSError as exc:
            raise ValueError("foundation runtime input is unsafe") from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError("foundation runtime input is linked or non-regular")
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    finally:
        for opened in reversed(descriptors):
            os.close(opened)

    def identity(item: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    if identity(before) != identity(after):
        raise ValueError("foundation runtime input changed while reading")
    return digest.hexdigest()


def verify_runtime_source(
    project_root: Path, model_id: str, identity: Mapping[str, object]
) -> None:
    """Re-audit the accepted checkout before any third-party module can load."""

    parts = _safe_parts(_SOURCE_ROOTS[model_id])
    _checkout_descriptor, descriptors = _open_directory_chain(project_root, parts)
    for descriptor in reversed(descriptors):
        os.close(descriptor)
    checkout = project_root.joinpath(*parts)
    observed = audit_checkout(model_id, checkout)
    expected_fields = ("repository_url", "git_commit", "source_tree_sha256")
    for field in expected_fields:
        if observed.get(field) != identity.get(field):
            label = "source tree" if field == "source_tree_sha256" else field
            raise ValueError(f"foundation {label} drifted")
    if observed.get("git_clean") is not True:
        raise ValueError("foundation source checkout is dirty")


def verify_runtime_snapshot(
    project_root: Path,
    model_id: str,
    identity: Mapping[str, object],
    *,
    snapshot_relative: str,
) -> Path:
    """Revalidate every runtime byte declared by the accepted weight receipt."""

    root = Path(project_root).resolve(strict=True)
    expected_relative = f"{_WEIGHT_ROOT}/{model_id}"
    if snapshot_relative != expected_relative:
        raise ValueError("foundation snapshot path differs from the manifest contract")
    parts = _safe_parts(snapshot_relative)
    _snapshot_descriptor, descriptors = _open_directory_chain(root, parts)
    for descriptor in reversed(descriptors):
        os.close(descriptor)
    snapshot = root.joinpath(*parts)
    if _regular_sha256_at(root, f"{snapshot_relative}/config.json") != identity["config_sha256"]:
        raise ValueError("foundation config drifted")
    license_name = "LICENSE" if model_id == "timesfm3_zs" else "README.md"
    if (
        _regular_sha256_at(root, f"{snapshot_relative}/{license_name}")
        != identity["weight_license_sha256"]
    ):
        raise ValueError("foundation weight license evidence drifted")
    if _regular_sha256_at(root, f"{snapshot_relative}/README.md") != identity["model_card_sha256"]:
        raise ValueError("foundation model card drifted")
    declared = identity["checkpoint_files"]
    if type(declared) is not list or not declared:
        raise ValueError("foundation checkpoint inventory differs")
    for record in declared:
        if type(record) is not dict or set(record) != {"path", "sha256"}:
            raise ValueError("foundation checkpoint record differs")
        path = str(record["path"])
        if (
            len(_safe_parts(path)) != 1
            or _regular_sha256_at(root, f"{snapshot_relative}/{path}") != record["sha256"]
        ):
            raise ValueError("foundation checkpoint drifted")
    runtime_files = identity["runtime_code_files"]
    if type(runtime_files) is not list:
        raise ValueError("foundation runtime-code inventory differs")
    for record in runtime_files:
        if type(record) is not dict or set(record) != {"path", "sha256"}:
            raise ValueError("foundation runtime-code record differs")
        path = str(record["path"])
        if (
            len(_safe_parts(path)) != 1
            or _regular_sha256_at(root, f"{snapshot_relative}/{path}") != record["sha256"]
        ):
            raise ValueError("foundation runtime code drifted")
    return snapshot


@contextmanager
def _fixed_import(project_root: Path, model_id: str):
    relative = _CHECKOUTS[model_id]
    root = Path(project_root).resolve(strict=True)
    parts = _safe_parts(relative)
    _source_descriptor, descriptors = _open_directory_chain(root, parts)
    for descriptor in reversed(descriptors):
        os.close(descriptor)
    source = root.joinpath(*parts)
    original = list(sys.path)
    sys.path.insert(0, str(source))
    try:
        yield
    finally:
        sys.path[:] = original


def _import_fixed(project_root: Path, model_id: str, module: str) -> ModuleType:
    with _fixed_import(project_root, model_id):
        imported = importlib.import_module(module)
    module_file = Path(str(getattr(imported, "__file__", ""))).resolve(strict=True)
    checkout = Path(project_root).resolve(strict=True).joinpath(*_CHECKOUTS[model_id].split("/"))
    if not module_file.is_relative_to(checkout):
        raise ValueError("foundation module resolved outside its fixed checkout")
    return imported


class _TimesFM3Backend:
    def __init__(self, project_root: Path, snapshot: Path, device: str) -> None:
        package = _import_fixed(project_root, "timesfm3_zs", "timesfm3")
        config = package.ModelConfig(
            checkpoint_path=str(snapshot), per_core_batch_size=32, device=device
        )
        self._forecaster = package.TimesFM3Evaluator(config)

    def forecast_point(self, past_target: np.ndarray, *, horizon: int) -> np.ndarray:
        outputs = list(
            self._forecaster.predict_batch(
                [row for row in past_target],
                horizon=horizon,
                return_quantiles=False,
                use_symmetric_averaging=False,
                make_positive=True,
                padding_mode="none",
            )
        )
        values = [np.asarray(output.forecast) for output in outputs]
        return np.stack(values, axis=0).astype(np.float32, copy=False)


class _SundialBackend:
    def __init__(self, snapshot: Path, device: str) -> None:
        transformers = importlib.import_module("transformers")
        self._device = device
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            trust_remote_code=True,
            local_files_only=True,
        ).to(device)
        self._model.eval()

    def sample(
        self,
        past_target: np.ndarray,
        *,
        horizon: int,
        num_samples: int,
        rng_seed: int,
        patch_size: int | None = None,
    ) -> np.ndarray:
        if patch_size is not None:
            raise ValueError("Sundial does not accept a patch-size override")
        torch = importlib.import_module("torch")
        devices = [torch.cuda.current_device()] if self._device.startswith("cuda") else []
        with torch.inference_mode(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(rng_seed)
            raw = self._model.generate(
                torch.as_tensor(past_target, dtype=torch.float32, device=self._device),
                max_new_tokens=horizon,
                num_samples=num_samples,
            )
        values = raw.detach().cpu().numpy()
        if values.shape == (past_target.shape[0], horizon, num_samples):
            values = values.transpose(0, 2, 1)
        if values.shape != (past_target.shape[0], num_samples, horizon):
            raise ValueError("Sundial sample shape differs from the sealed contract")
        return values


class _MoiraiBackend:
    def __init__(self, project_root: Path, snapshot: Path, device: str) -> None:
        module = _import_fixed(project_root, "moirai_1_1_r_small_zs", "uni2ts.model.moirai")
        self._forecast_type = module.MoiraiForecast
        self._module = module.MoiraiModule.from_pretrained(str(snapshot), local_files_only=True)
        self._device = device

    def sample(
        self,
        past_target: np.ndarray,
        *,
        horizon: int,
        num_samples: int,
        rng_seed: int,
        patch_size: int | None = None,
    ) -> np.ndarray:
        if patch_size != 32:
            raise ValueError("Moirai patch_size must remain fixed at 32")
        torch = importlib.import_module("torch")
        model = self._forecast_type(
            module=self._module,
            prediction_length=horizon,
            context_length=past_target.shape[1],
            patch_size=patch_size,
            num_samples=num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(self._device)
        model.eval()
        target = torch.as_tensor(past_target[..., None], dtype=torch.float32, device=self._device)
        observed = torch.ones_like(target, dtype=torch.bool)
        is_pad = torch.zeros(target.shape[:2], dtype=torch.bool, device=self._device)
        devices = [torch.cuda.current_device()] if self._device.startswith("cuda") else []
        with torch.inference_mode(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(rng_seed)
            raw = model(
                past_target=target,
                past_observed_target=observed,
                past_is_pad=is_pad,
                num_samples=num_samples,
            )
        values = raw.detach().cpu().numpy()
        if values.shape == (past_target.shape[0], num_samples, horizon, 1):
            values = values[..., 0]
        if values.shape != (past_target.shape[0], num_samples, horizon):
            raise ValueError("Moirai sample shape differs from the sealed contract")
        return values


def build_runtime_backend_factory(
    project_root: Path,
    model_id: str,
    *,
    device: str,
    snapshot_relative: str,
) -> Callable[[dict[str, object]], object]:
    """Return a lazy factory; no third-party module is imported by this function."""

    if model_id not in {"timesfm3_zs", "sundial_base_zs", "moirai_1_1_r_small_zs"}:
        raise ValueError("unknown foundation runtime model")

    def factory(identity: dict[str, object]) -> object:
        if identity.get("model_id") != model_id:
            raise ValueError("foundation runtime identity differs")
        snapshot = verify_runtime_snapshot(
            project_root,
            model_id,
            identity,
            snapshot_relative=snapshot_relative,
        )
        verify_runtime_source(project_root, model_id, identity)
        if model_id == "timesfm3_zs":
            return _TimesFM3Backend(project_root, snapshot, device)
        if model_id == "sundial_base_zs":
            return _SundialBackend(snapshot, device)
        return _MoiraiBackend(project_root, snapshot, device)

    return factory
