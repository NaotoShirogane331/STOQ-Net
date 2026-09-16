"""Train one audited modern baseline without loading held-out data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from racepv.audit import sha256_file
from racepv.config import QUANTILE_LEVELS, _construct_config
from racepv.environment import environment_report
from racepv.io import (
    WindowDataset,
    load_window_dataset,
    require_global_split_chronology,
    require_registered_spatial_protocol,
    window_dataset_content_identity,
)
from racepv.training import (
    EFFICIENCY_BENCHMARK_BATCH_SIZE,
    EFFICIENCY_BENCHMARK_REPETITIONS,
    EFFICIENCY_BENCHMARK_WARMUP,
    EarlyStoppingTrainer,
    benchmark_inference,
    build_window_loader,
    count_parameters,
    efficiency_audit_record,
)
from racepv.utils import atomic_json_dump, seed_everything, select_device

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODERN_BASELINES = ("timemixerpp_qr", "timer_xl_qr", "rqr_tft")
REGISTERED_LEVELS = QUANTILE_LEVELS
_IMPLEMENTATION_PATH = Path("src/racepv/baselines/modern.py")
_FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HELD_OUT_TOKENS = {"calibration", "test"}
_EXPECTED_IMPLEMENTATION_KIND = {
    "timemixerpp_qr": "paper_derived_protocol_reproduction",
    "timer_xl_qr": "official_backbone_protocol_adaptation",
    "rqr_tft": "paper_derived_protocol_reproduction",
}
_EXPECTED_SOURCE_URL = {
    "timemixerpp_qr": (
        "https://proceedings.iclr.cc/paper_files/paper/2025/hash/"
        "2b187165e28fdfdc0ffb34d1bfff2b0c-Abstract-Conference.html"
    ),
    "timer_xl_qr": "https://github.com/thuml/OpenLTM.git",
    "rqr_tft": "https://doi.org/10.3390/app16168261",
}
_GENERAL_STATE_SEMANTICS = "median_change_heuristic_not_model_output"
_RQR_STATE_SEMANTICS = (
    "binary_ramp_compatibility_split_equally_between_up_down_no_learned_direction"
)
_RQR_RAMP_SEMANTICS = "learned_binary_ramp_probability"


def _add_prepared_expectations(parser):
    parser.add_argument("--expected-prepared-schema-version", type=int)
    parser.add_argument("--expected-causality-profile")
    for split in ("train", "validation"):
        parser.add_argument(f"--prepared-{split}-descriptor-sha256")
        parser.add_argument(f"--prepared-{split}-read-scope-sha256")


def _prepared_expectations(args):
    from racepv.data.weather import expected_origin_contract

    version = getattr(args, "expected_prepared_schema_version", None)
    profile = getattr(args, "expected_causality_profile", None)
    if type(version) is int and version == 3:
        if profile != "as-of-origin-v1":
            raise ValueError("expected schema3 and profile as-of-origin-v1 must be paired")
    else:
        expected_origin_contract(version, profile)
        if any(
            getattr(args, f"prepared_{split}_{kind}_sha256", None) is not None
            for split in ("train", "validation")
            for kind in ("descriptor", "read_scope")
        ):
            raise ValueError("scoped descriptor pins require explicit schema3")
    if version is not None and (
        getattr(args, "allow_ineligible", False)
        or getattr(args, "verify_content", True) is not True
        or getattr(args, "allow_development_data", False)
    ):
        raise ValueError("prepared v2 admission forbids content/evidence bypasses")
    return {"expected_version": version, "expected_profile": profile}


def _prepared_dataset_identity(dataset):
    from racepv.data.prepared_contract import require_origin_safe_history_contract

    identity = require_origin_safe_history_contract(dataset, version=2, profile="as-of-origin-v1")
    schema = dataset.schema
    return {
        **identity,
        "clock": dict(schema["clock"]),
        "mask_semantics": {
            name: schema["causal_rules"][name]
            for name in ("missing_values", "imputed_values", "provenance_flag_mask")
        },
    }


def _require_prepared_scientific_evidence(identity):
    if (
        identity.get("content_verified") is not True
        or identity.get("array_semantics_verified") is not True
        or identity.get("source_interpretation_verified") is not True
        or identity.get("registered_evidence_eligible") is not True
    ):
        raise ValueError(
            "prepared v2 source interpretation and registered evidence are not qualified"
        )


def _prepared_output_metadata(identities, *, split=None):
    """Project verified identities into outputs without changing qualification claims."""
    from racepv.baselines.extension_registry import validate_prepared_identity

    if not identities or "prepared_contracts" not in identities:
        return {}
    contracts = {
        name: validate_prepared_identity(value)
        for name, value in identities["prepared_contracts"].items()
    }
    if not contracts or any(value["split"] != name for name, value in contracts.items()):
        raise ValueError("prepared output split identity differs")
    result = {"prepared_schema_version": 2, "prepared_contracts": contracts}
    if split is not None:
        if split not in contracts:
            raise ValueError("prepared output source split is missing")
        result["prepared_contract"] = contracts[split]
    return result


def _preflight_prepared_inputs(args, *, continuous_panel=False, legacy_only=False):
    """Explicit protocol admission runs before config/model/optimizer/panel factories."""
    expectations = _prepared_expectations(args)
    if expectations["expected_version"] is None:
        return
    if continuous_panel:
        raise ValueError("origin-dynamic prepared contracts cannot form a continuous panel")
    if legacy_only:
        raise ValueError("baseline supports legacy prepared contracts only")
    if expectations["expected_version"] == 3:
        from racepv.data.scoped_consumer import (
            ScopedQualificationRequired,
            inspect_scoped_consumer_inputs,
        )

        report = inspect_scoped_consumer_inputs(args)
        # Metadata success is not source/extraction qualification. Never fall
        # through to legacy loaders or factories while the binding is missing.
        raise ScopedQualificationRequired(report)
    identities = {"prepared_contracts": {}}
    for split in ("train", "validation", "calibration", "test"):
        path = getattr(args, split, None)
        if path is not None:
            dataset = load_window_dataset(path, verify_content=True, **expectations)
            identity = _prepared_dataset_identity(dataset)
            _require_prepared_scientific_evidence(identity)
            identities["prepared_contracts"][split] = identity
    return identities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_prepared_expectations(parser)
    parser.add_argument("--name", choices=MODERN_BASELINES, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        required=True,
        help="required seal: fit on train and predict validation only",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--source-identities", type=Path, required=True)
    parser.add_argument("--openltm-root", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _path_tokens(path: Path) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", str(path).lower()) if token}


def _validate_arguments(args: argparse.Namespace) -> None:
    _prepared_expectations(args)
    if not args.validation_only:  # defensive when called with a hand-built namespace
        raise ValueError("--validation-only is required")
    for split in ("train", "validation"):
        path = getattr(args, split)
        forbidden = sorted(_path_tokens(path).intersection(_HELD_OUT_TOKENS))
        if forbidden:
            raise ValueError(
                f"{split} contains a held-out path token {forbidden}; "
                "calibration/test paths are forbidden"
            )
    if args.name == "timer_xl_qr" and args.openltm_root is None:
        raise ValueError("timer_xl_qr requires --openltm-root")
    if args.name != "timer_xl_qr" and args.openltm_root is not None:
        raise ValueError("--openltm-root is only valid for timer_xl_qr")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    _validate_arguments(args)
    return args


def _read_hashed_bytes(path: Path, *, label: str) -> tuple[bytes, str]:
    try:
        with path.open("rb") as handle:
            content = handle.read()
    except OSError as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    return content, hashlib.sha256(content).hexdigest()


def _load_json_mapping(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _load_yaml_mapping(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(content.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 YAML") from exc
    if not isinstance(raw, dict):
        raise TypeError(f"{label} root must be a mapping")
    return raw


def _load_common_config(content: bytes):
    return _construct_config(_load_yaml_mapping(content, label="common configuration"))


def _load_model_hyperparameters(content: bytes, name: str) -> dict[str, object]:
    raw = _load_yaml_mapping(content, label="model configuration")
    selected = raw.get(name)
    if not isinstance(selected, dict):
        raise KeyError(f"model configuration has no mapping for {name!r}")
    result = dict(selected)
    expected_kind = _EXPECTED_IMPLEMENTATION_KIND[name]
    if result.get("implementation_kind") != expected_kind:
        raise ValueError(
            f"{name} implementation_kind must be {expected_kind!r} before construction"
        )
    return result


def _validated_constructor_hyperparameters(
    name: str,
    model_hyperparameters: Mapping[str, object],
) -> dict[str, object]:
    """Remove disclosure-only keys only after their frozen values are checked."""

    result = dict(model_hyperparameters)
    implementation_kind = result.pop("implementation_kind")
    expected_kind = _EXPECTED_IMPLEMENTATION_KIND[name]
    if implementation_kind != expected_kind:  # defensive after the load-time check
        raise ValueError(f"{name} implementation_kind changed before construction")
    if name == "rqr_tft" and "ramp_pos_weight" in result:
        raise ValueError("rqr_tft ramp_pos_weight is derived from daylight training labels only")
    if name == "timer_xl_qr":
        initialization = result.pop("initialization", None)
        if initialization != "task_specific_from_scratch":
            raise ValueError("timer_xl_qr initialization must be task_specific_from_scratch")
    return result


def _validate_paper_source_identity(
    name: str,
    identity: Mapping[str, object],
) -> None:
    expected_url = _EXPECTED_SOURCE_URL[name]
    if identity.get("name") != name:
        raise ValueError(f"source identity name must be {name!r}")
    if identity.get("repository_url") != expected_url or identity.get("paper_url") != expected_url:
        raise ValueError(f"source identity URL does not match the reviewed {name} paper")
    if identity.get("official_code_used") is not False:
        raise ValueError(f"{name} must disclose official_code_used=false")
    if identity.get("implementation_kind") != _EXPECTED_IMPLEMENTATION_KIND[name]:
        raise ValueError(f"source identity implementation_kind does not match {name}")
    files = identity.get("implementation_files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{name} source identity has no implementation file hashes")
    seen_required = False
    for record in files:
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"{name} implementation file identity has an invalid schema")
        raw_path = record["path"]
        expected_hash = record["sha256"]
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{name} implementation identity has an invalid path")
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{name} implementation identity path escapes the repository")
        implementation = (_PROJECT_ROOT / relative).resolve()
        if (
            not implementation.is_relative_to(_PROJECT_ROOT.resolve())
            or not implementation.is_file()
        ):
            raise ValueError(f"missing reviewed implementation file: {relative.as_posix()}")
        if not isinstance(expected_hash, str) or _FULL_SHA256.fullmatch(expected_hash) is None:
            raise ValueError(f"{name} implementation identity has an invalid SHA-256")
        actual_hash = sha256_file(implementation)
        if actual_hash != expected_hash:
            raise ValueError(
                f"implementation SHA-256 changed for {relative.as_posix()}: "
                f"{expected_hash} != {actual_hash}"
            )
        seen_required |= relative == _IMPLEMENTATION_PATH
    if not seen_required:
        raise ValueError(f"{name} identity does not cover {_IMPLEMENTATION_PATH.as_posix()}")


def _load_selected_source_identity(
    manifest_content: bytes,
    *,
    name: str,
    openltm_root: Path | None,
) -> dict[str, object]:
    manifest = _load_json_mapping(manifest_content, label="source-identities manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("source-identities manifest must use schema_version=1")
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("source-identities manifest has no sources mapping")
    selected = sources.get(name)
    if not isinstance(selected, dict):
        raise ValueError(f"source-identities manifest has no reviewed identity for {name}")
    identity = dict(selected)
    if name == "timer_xl_qr":
        if openltm_root is None:  # narrowed by argument validation
            raise ValueError("timer_xl_qr requires --openltm-root")
        from racepv.baselines.modern import _validated_openltm_source_identity

        _validated_openltm_source_identity(openltm_root, identity)
    else:
        _validate_paper_source_identity(name, identity)
    return identity


def _descriptor_sha256(path: Path) -> str:
    descriptor = path / "store.json" if path.is_dir() else path
    return sha256_file(descriptor)


def _sample_ramp_thresholds(dataset: WindowDataset) -> np.ndarray:
    from racepv.cli.baseline import _sample_ramp_thresholds as implementation

    return implementation(dataset)


def _prediction_payload(
    dataset: WindowDataset,
    quantiles: np.ndarray,
    levels: Sequence[float],
    thresholds: np.ndarray,
    *,
    batch_size: int,
    workers: int,
) -> dict[str, np.ndarray]:
    from racepv.cli.baseline import _prediction_payload as implementation

    return implementation(
        dataset,
        quantiles,
        levels,
        thresholds,
        batch_size=batch_size,
        workers=workers,
    )


def _validate_datasets(
    args: argparse.Namespace,
) -> tuple[
    dict[str, WindowDataset],
    dict[str, object],
    dict[str, list[str]],
    dict[str, dict[str, str]],
]:
    from racepv.cli.baseline import _enforce_fit_boundaries, _require_collection_split

    paths = {"train": args.train, "validation": args.validation}
    expectations = _prepared_expectations(args)
    if expectations["expected_version"] == 3:
        _preflight_prepared_inputs(args)
    descriptor_hashes = {split: _descriptor_sha256(path) for split, path in paths.items()}
    datasets = {}
    prepared_contracts = {}
    content_hashes: dict[str, str] = {}
    for split, path in paths.items():
        dataset = load_window_dataset(path, verify_content=True, **expectations)
        if expectations["expected_version"] == 2:
            prepared_contracts[split] = _prepared_dataset_identity(dataset)
            _require_prepared_scientific_evidence(prepared_contracts[split])
        if _descriptor_sha256(path) != descriptor_hashes[split]:
            raise RuntimeError(f"{split} dataset descriptor drifted while loading")
        content_identity = getattr(dataset, "content_identity", None)
        if (
            not isinstance(content_identity, str)
            or _FULL_SHA256.fullmatch(content_identity) is None
        ):
            raise ValueError(f"{split} verified dataset has no content identity")
        datasets[split] = dataset
        content_hashes[split] = content_identity
    if prepared_contracts:
        spatial = {}
        plans = {}
        signature = None
        for split, dataset in datasets.items():
            identity = prepared_contracts[split]
            if identity["split"] != split:
                raise ValueError("prepared dataset split differs from requested role")
            schema = dataset.schema
            current = {
                key: schema[key]
                for key in (
                    "history_columns",
                    "future_known_columns",
                    "site_metadata_names",
                    "geometry",
                    "clock",
                    "causal_rules",
                )
            }
            if signature is not None and current != signature:
                raise ValueError("prepared train/validation feature and origin contracts differ")
            signature = current
            sites = identity.get("sites", [identity["site"]])
            spatial[split] = sorted({site["spatial_role"] for site in sites})
            if "zero_parameter_test" in spatial[split]:
                raise ValueError("prepared fitting cannot use zero_parameter_test sites")
            plans[split] = schema["split_plan"]["splits"][split]
        if plans["train"]["end_exclusive_ns"] > plans["validation"]["start_ns"]:
            raise ValueError("prepared train and validation split chronology overlaps")
        return (
            datasets,
            {"prepared_schema_version": 2, "splits": plans},
            spatial,
            {
                "descriptor_hashes": descriptor_hashes,
                "content_hashes": content_hashes,
                "prepared_contracts": prepared_contracts,
            },
        )
    train_signature: dict[str, Any] | None = None
    for split, dataset in datasets.items():
        dataset.require_registered_schema()
        dataset.require_registered_evidence()
        _require_collection_split(dataset, split)
        signature = dataset.registered_schema_signature()
        if train_signature is None:
            train_signature = signature
        elif signature != train_signature:
            raise ValueError("train and validation must share the registered feature schema")
    chronology = require_global_split_chronology(datasets)
    spatial_protocol = require_registered_spatial_protocol(datasets)
    _enforce_fit_boundaries(datasets)
    return (
        datasets,
        chronology,
        spatial_protocol,
        {
            "descriptor_hashes": descriptor_hashes,
            "content_hashes": content_hashes,
        },
    )


def _reverify_dataset_identities(
    args: argparse.Namespace,
    identities: Mapping[str, Mapping[str, str]],
) -> None:
    paths = {"train": args.train, "validation": args.validation}
    expectations = _prepared_expectations(args)
    if expectations["expected_version"] == 3:
        _preflight_prepared_inputs(args)
    for split, path in paths.items():
        descriptor = _descriptor_sha256(path)
        content = window_dataset_content_identity(path, verify=True)
        if descriptor != identities["descriptor_hashes"][split]:
            raise RuntimeError(f"{split} dataset identity drift: descriptor SHA-256 changed")
        if content != identities["content_hashes"][split]:
            raise RuntimeError(f"{split} dataset identity drift: verified content changed")
        if "prepared_contracts" in identities:
            current = _prepared_dataset_identity(
                load_window_dataset(path, verify_content=True, **expectations)
            )
            if current != identities["prepared_contracts"][split]:
                raise RuntimeError(f"{split} prepared origin/population identity drift")


def _rqr_training_weight(
    dataset: WindowDataset,
    *,
    batch_size: int,
    workers: int,
) -> dict[str, object]:
    from racepv.baselines.modern import ramp_class_weight_from_training_labels

    stable_count = 0
    ramp_count = 0
    loader = build_window_loader(dataset, batch_size, False, workers)
    for batch in loader:
        measured = ramp_class_weight_from_training_labels(
            batch["state_labels"],
            daylight_mask=batch["daylight_mask"],
        )
        stable_count += measured.stable_count
        ramp_count += measured.ramp_count
    unbounded = math.sqrt(stable_count / ramp_count) if ramp_count else 1.0
    weight = float(np.clip(unbounded, 1.0, 4.0))
    return {
        "stable_count": stable_count,
        "ramp_count": ramp_count,
        "unclipped_ramp_pos_weight": float(unbounded),
        "ramp_pos_weight": weight,
        "source": "valid_daylight_training_labels_only",
        "clip_range": [1.0, 4.0],
    }


def _atomic_yaml_dump(payload: Mapping[str, object], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False)
    temporary.replace(destination)


def _atomic_torch_save(payload: Mapping[str, object], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(destination)


def _sync_file(handle: Any) -> None:
    handle.flush()
    getattr(os, "fdatasync", os.fsync)(handle.fileno())


def _atomic_save_predictions_npz(
    destination: Path,
    tensors: Mapping[str, torch.Tensor | np.ndarray],
    metadata: Mapping[str, object],
) -> None:
    """Publish complete NPZ and JSON files without exposing partial bytes."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_suffix(".json")
    npz_temporary: Path | None = None
    json_temporary: Path | None = None
    try:
        npz_fd, npz_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        npz_temporary = Path(npz_name)
        arrays = {
            name: value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
            for name, value in tensors.items()
        }
        with os.fdopen(npz_fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            _sync_file(handle)

        json_fd, json_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{sidecar.name}.",
            suffix=".tmp",
        )
        json_temporary = Path(json_name)
        with os.fdopen(json_fd, "w", encoding="utf-8") as handle:
            json.dump(dict(metadata), handle, ensure_ascii=False, indent=2)
            _sync_file(handle)

        os.replace(npz_temporary, destination)
        npz_temporary = None
        os.replace(json_temporary, sidecar)
        json_temporary = None
    finally:
        if npz_temporary is not None:
            npz_temporary.unlink(missing_ok=True)
        if json_temporary is not None:
            json_temporary.unlink(missing_ok=True)


def _model_inputs(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: batch[key]
        for key in ("history_values", "history_mask", "future_known", "site_metadata")
    }


@torch.inference_mode()
def _predict_validation(
    trainer: EarlyStoppingTrainer,
    loader: torch.utils.data.DataLoader,
) -> dict[str, torch.Tensor]:
    """Collect only batch-shaped inference/audit fields from a modern model.

    RQR-TFT also returns its persisted scalar class weight.  That scalar belongs
    in model/run metadata, not in a per-window prediction archive.
    """

    trainer.model.eval()
    collected: dict[str, list[torch.Tensor]] = {}
    output_fields = {"quantiles", "ramp_probs"}
    audit_fields = {
        "targets",
        "state_labels",
        "daylight_mask",
        "timestamps",
        "day_ids",
        "site_ids",
    }
    for raw_batch in loader:
        batch = trainer._move(raw_batch)
        outputs = trainer.model(**_model_inputs(batch))
        for key in output_fields:
            value = outputs.get(key)
            if isinstance(value, torch.Tensor):
                if value.ndim == 0 or value.shape[0] != batch["targets"].shape[0]:
                    raise ValueError(f"prediction output {key!r} is not batch-shaped")
                collected.setdefault(key, []).append(value.detach().cpu())
        for key in audit_fields:
            if key in batch:
                collected.setdefault(key, []).append(batch[key].detach().cpu())
    if "quantiles" not in collected:
        raise RuntimeError("validation prediction produced no quantiles")
    return {key: torch.cat(values, dim=0) for key, values in collected.items()}


def _validate_quantiles(payload: Mapping[str, np.ndarray]) -> None:
    quantiles = np.asarray(payload["quantiles"])
    daylight = np.asarray(payload["daylight_mask"]).astype(bool)
    if not np.isfinite(quantiles).all():
        raise FloatingPointError("modern baseline produced non-finite quantiles")
    if np.any(quantiles[..., 1:] < quantiles[..., :-1]):
        raise ValueError("modern baseline produced crossed quantiles")
    if np.any((quantiles < 0.0) | (quantiles > 1.05)):
        raise ValueError("modern baseline produced quantiles outside [0,1.05]")
    if np.any(quantiles[~daylight] != 0.0):
        raise ValueError("modern baseline produced non-zero night quantiles")


def _prediction_archive(
    *,
    name: str,
    dataset: WindowDataset,
    predicted: Mapping[str, torch.Tensor],
    thresholds: np.ndarray,
    batch_size: int,
    workers: int,
) -> tuple[dict[str, np.ndarray], str]:
    raw_quantiles = predicted["quantiles"].numpy()
    if not np.isfinite(raw_quantiles).all():
        raise FloatingPointError("modern baseline produced non-finite quantiles")
    payload = _prediction_payload(
        dataset,
        raw_quantiles,
        REGISTERED_LEVELS,
        thresholds,
        batch_size=max(batch_size, 512),
        workers=workers,
    )
    if name == "rqr_tft":
        if "ramp_probs" not in predicted:
            raise KeyError("RQR-TFT prediction output has no ramp_probs")
        ramp_probs = predicted["ramp_probs"].numpy().astype(np.float32, copy=False)
        if ramp_probs.shape != payload["targets"].shape:
            raise ValueError("RQR-TFT ramp probabilities must match validation targets")
        if not np.isfinite(ramp_probs).all():
            raise FloatingPointError("RQR-TFT produced non-finite ramp probabilities")
        if np.any((ramp_probs < 0.0) | (ramp_probs > 1.0)):
            raise ValueError("RQR-TFT produced raw ramp probabilities outside [0,1]")
        ramp_probs = ramp_probs.astype(np.float32, copy=False)
        payload["ramp_probs"] = ramp_probs
        payload["state_probs"] = np.stack(
            (1.0 - ramp_probs, ramp_probs / 2.0, ramp_probs / 2.0),
            axis=-1,
        ).astype(np.float32)
        semantics = _RQR_STATE_SEMANTICS
    else:
        semantics = _GENERAL_STATE_SEMANTICS
    _validate_quantiles(payload)
    probabilities = np.asarray(payload["state_probs"])
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("compatibility state probabilities must be finite in [0,1]")
    if not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-6):
        raise ValueError("compatibility state probabilities must sum to one")
    expected = {
        "quantiles",
        "state_probs",
        "targets",
        "state_labels",
        "daylight_mask",
        "timestamps",
        "day_ids",
        "site_ids",
    }
    if name == "rqr_tft":
        expected.add("ramp_probs")
    if set(payload) != expected:
        raise RuntimeError(
            f"modern prediction archive schema drifted: {set(payload)} != {expected}"
        )
    return payload, semantics


def _training_peak_memory(device: torch.device) -> tuple[int | None, str | None]:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device)), None
    if device.type == "mps":
        return None, "training executed on MPS; PyTorch has no resettable GPU peak counter"
    return None, "training executed on CPU; GPU peak memory is not applicable"


def _jsonable(value: object) -> object:
    return json.loads(json.dumps(value))


def run(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    _preflight_prepared_inputs(args)
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    seed_everything(args.seed)
    common_content, common_config_hash = _read_hashed_bytes(
        args.config, label="common configuration"
    )
    model_content, model_config_hash = _read_hashed_bytes(
        args.model_config, label="model configuration"
    )
    source_content, source_manifest_hash = _read_hashed_bytes(
        args.source_identities, label="source-identities manifest"
    )
    common_config = _load_common_config(common_content)
    if tuple(common_config.quantiles) != REGISTERED_LEVELS:
        raise ValueError("modern baselines require the registered seven quantiles")
    model_hyperparameters = _load_model_hyperparameters(model_content, args.name)
    constructor_hyperparameters = _validated_constructor_hyperparameters(
        args.name, model_hyperparameters
    )
    source_identity = _load_selected_source_identity(
        source_content,
        name=args.name,
        openltm_root=args.openltm_root,
    )
    datasets, temporal_audit, spatial_protocol, dataset_identities = _validate_datasets(args)
    descriptor_hashes = dataset_identities["descriptor_hashes"]
    content_hashes = dataset_identities["content_hashes"]

    train_data = datasets["train"]
    validation_data = datasets["validation"]
    training = common_config.training
    ramp_weight: dict[str, object] | None = None
    if args.name == "rqr_tft":
        ramp_weight = _rqr_training_weight(
            train_data,
            batch_size=training.batch_size,
            workers=training.num_workers,
        )
        constructor_hyperparameters["ramp_pos_weight"] = ramp_weight["ramp_pos_weight"]

    dimensions = {
        "input_dim": int(train_data.arrays["history_values"].shape[-1]),
        "future_dim": int(train_data.arrays["future_known"].shape[-1]),
        "site_dim": int(train_data.arrays["site_metadata"].shape[-1]),
    }
    from racepv.baselines.modern import (
        ModernBaselineLoss,
        ModernBaselineSpec,
        build_modern_baseline,
    )

    model, spec = build_modern_baseline(
        args.name,
        **dimensions,
        quantiles=REGISTERED_LEVELS,
        hyperparameters=constructor_hyperparameters,
        openltm_root=args.openltm_root,
        source_identity=source_identity if args.name == "timer_xl_qr" else None,
    )
    if not isinstance(spec, ModernBaselineSpec) or spec.name != args.name:
        raise TypeError("constructed modern baseline identity does not match the requested model")

    loss_hyperparameters: dict[str, object] = {"levels": list(REGISTERED_LEVELS)}
    if args.name == "rqr_tft":
        loss_hyperparameters.update(
            ramp_pos_weight=constructor_hyperparameters["ramp_pos_weight"],
            ramp_weighted_quantile_multiplier=constructor_hyperparameters[
                "ramp_weighted_quantile_multiplier"
            ],
            ramp_loss_weight=constructor_hyperparameters["ramp_loss_weight"],
        )
    criterion = ModernBaselineLoss(
        REGISTERED_LEVELS,
        ramp_pos_weight=float(loss_hyperparameters.get("ramp_pos_weight", 1.0)),
        ramp_weighted_quantile_multiplier=float(
            loss_hyperparameters.get("ramp_weighted_quantile_multiplier", 6.0)
        ),
        ramp_loss_weight=float(loss_hyperparameters.get("ramp_loss_weight", 1.0)),
    )

    resolved_device = select_device(args.device)
    device = torch.device(resolved_device)
    resolved_model_hyperparameters = dict(model_hyperparameters)
    if ramp_weight is not None:
        resolved_model_hyperparameters["ramp_pos_weight"] = ramp_weight["ramp_pos_weight"]
    common_hyperparameters = common_config.to_dict()
    implementation_metadata_method = getattr(model, "implementation_metadata", None)
    implementation_metadata = (
        implementation_metadata_method()
        if callable(implementation_metadata_method)
        else {
            "actual_backbone": spec.actual_backbone,
            "official_code_used": spec.official_code_used,
            "implementation_kind": spec.implementation_kind,
        }
    )

    checkpoint_metadata = {
        **(
            {"prepared_contracts": dataset_identities["prepared_contracts"]}
            if "prepared_contracts" in dataset_identities
            else {}
        ),
        "baseline": args.name,
        "seed": args.seed,
        "quantiles": list(REGISTERED_LEVELS),
        **dimensions,
        "model_spec": asdict(spec),
        "source_identity": source_identity,
        "source_identities_manifest_sha256": source_manifest_hash,
        "selection_boundary": "validation_only",
        "calibration_or_test_loaded": False,
        "selection_metric": "daylight_qcrps",
        "train_descriptor_sha256": descriptor_hashes["train"],
        "validation_descriptor_sha256": descriptor_hashes["validation"],
        "train_window_content_sha256": content_hashes["train"],
        "validation_window_content_sha256": content_hashes["validation"],
    }
    trainer = EarlyStoppingTrainer(
        model,
        criterion,
        device=resolved_device,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        max_epochs=training.max_epochs,
        patience=training.patience,
        gradient_clip_norm=training.gradient_clip_norm,
        mixed_precision=training.mixed_precision,
        checkpoint_path=args.output / "best.ckpt",
        checkpoint_metadata=checkpoint_metadata,
        selection_metric="daylight_qcrps",
    )
    train_loader = build_window_loader(train_data, training.batch_size, True, training.num_workers)
    validation_loader = build_window_loader(
        validation_data, training.batch_size, False, training.num_workers
    )

    args.output.mkdir(parents=True, exist_ok=False)
    _atomic_yaml_dump(
        {
            **_prepared_output_metadata(dataset_identities),
            "schema_version": 1,
            "model_name": args.name,
            "seed": args.seed,
            "common": _jsonable(common_hyperparameters),
            "model": _jsonable(resolved_model_hyperparameters),
        },
        args.output / "resolved_config.yaml",
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fit_started = time.perf_counter()
    fit = trainer.fit(train_loader, validation_loader)
    fit_seconds = time.perf_counter() - fit_started
    training_peak, training_peak_reason = _training_peak_memory(device)
    checkpoint_path = args.output / "best.ckpt"
    checkpoint_hash = sha256_file(checkpoint_path)

    bundle = {
        **_prepared_output_metadata(dataset_identities),
        "schema_version": 1,
        "model_state_dict": model.state_dict(),
        "baseline": args.name,
        "seed": args.seed,
        "quantiles": list(REGISTERED_LEVELS),
        "constructor_dimensions": dimensions,
        "constructor_hyperparameters": constructor_hyperparameters,
        "implementation_kind": spec.implementation_kind,
        "model_spec": asdict(spec),
        "source_identity": source_identity,
        "source_identities_manifest_sha256": source_manifest_hash,
        "implementation_metadata": implementation_metadata,
    }
    model_path = args.output / "model.pt"
    _atomic_torch_save(bundle, model_path)
    model_hash = sha256_file(model_path)
    model_bytes = model_path.stat().st_size

    predicted = _predict_validation(trainer, validation_loader)
    thresholds = _sample_ramp_thresholds(validation_data)
    payload, state_semantics = _prediction_archive(
        name=args.name,
        dataset=validation_data,
        predicted=predicted,
        thresholds=thresholds,
        batch_size=training.batch_size,
        workers=training.num_workers,
    )
    benchmark_loader = build_window_loader(
        validation_data,
        EFFICIENCY_BENCHMARK_BATCH_SIZE,
        False,
        training.num_workers,
    )
    benchmark_batch = next(iter(benchmark_loader))
    batch_inference = benchmark_inference(
        model,
        _model_inputs(benchmark_batch),
        warmup=EFFICIENCY_BENCHMARK_WARMUP,
        repetitions=EFFICIENCY_BENCHMARK_REPETITIONS,
    )
    trainable_parameters = count_parameters(model)
    efficiency = efficiency_audit_record(
        trainable_parameters=trainable_parameters,
        trainable_parameters_unavailable_reason=None,
        model_artifact_bytes=model_bytes,
        model_artifact_definition="serialized portable inference model.pt file size",
        batch_inference=batch_inference,
    )
    precision = {
        "configured_mixed_precision": bool(training.mixed_precision),
        "effective_mixed_precision": bool(trainer.use_amp),
    }
    shared_metadata: dict[str, object] = {
        **(
            {"prepared_contracts": dataset_identities["prepared_contracts"]}
            if "prepared_contracts" in dataset_identities
            else {}
        ),
        "baseline": args.name,
        "selection_boundary": "validation_only",
        "calibration_or_test_loaded": False,
        "selection_metric": "daylight_qcrps",
        "checkpoint_selection_uses_unweighted_daylight_qcrps": True,
        "model_spec": asdict(spec),
        "source_identity": source_identity,
        "source_identities_manifest_sha256": source_manifest_hash,
        "train_descriptor_sha256": descriptor_hashes["train"],
        "validation_descriptor_sha256": descriptor_hashes["validation"],
        "train_window_content_sha256": content_hashes["train"],
        "validation_window_content_sha256": content_hashes["validation"],
        "model_sha256": model_hash,
        "best_checkpoint_sha256": checkpoint_hash,
        "environment": environment_report(),
        "trainable_parameters": trainable_parameters,
        "model_bytes": model_bytes,
        "fit_seconds": fit_seconds,
        "device": resolved_device,
        "precision": precision,
        "training_peak_gpu_memory_bytes": training_peak,
        "training_peak_gpu_memory_unavailable_reason": training_peak_reason,
        "efficiency": efficiency,
        "common_hyperparameters": common_hyperparameters,
        "model_hyperparameters": resolved_model_hyperparameters,
        "loss_hyperparameters": loss_hyperparameters,
        "seed": args.seed,
        "quantiles": list(REGISTERED_LEVELS),
        "actual_backbone": spec.actual_backbone,
        "official_code_used": spec.official_code_used,
        "implementation_kind": spec.implementation_kind,
        "implementation_metadata": implementation_metadata,
        "state_prob_semantics": state_semantics,
    }
    if ramp_weight is not None:
        shared_metadata.update(
            ramp_class_weight=ramp_weight,
            ramp_probability_semantics=_RQR_RAMP_SEMANTICS,
            ramp_decision_threshold_semantics="diagnostic_only",
        )
    prediction_metadata = {
        **(
            {"prepared_contract": dataset_identities["prepared_contracts"]["validation"]}
            if "prepared_contracts" in dataset_identities
            else {}
        ),
        **shared_metadata,
        "split": "validation",
        "source_npz_sha256": descriptor_hashes["validation"],
        "source_window_content_sha256": content_hashes["validation"],
        "site_id_map": validation_data.schema["site_id_map"],
    }
    _reverify_dataset_identities(args, dataset_identities)
    _atomic_save_predictions_npz(
        args.output / "validation_predictions.npz",
        payload,
        prediction_metadata,
    )

    run_metadata = {
        **shared_metadata,
        "purpose": "training_validation_only_modern_baseline",
        "prediction_splits": ["validation"],
        "best_epoch": fit.best_epoch,
        "best_validation_loss": fit.best_validation_loss,
        "best_validation_metric_name": fit.best_validation_metric_name,
        "stopped_early": fit.stopped_early,
        "global_temporal_audit": temporal_audit,
        "registered_spatial_protocol": spatial_protocol,
        "config_sha256": common_config_hash,
        "model_config_sha256": model_config_hash,
    }
    atomic_json_dump(run_metadata, args.output / "run.json")
    return args.output


def main() -> None:
    output = run()
    print(f"Completed modern baseline; outputs are in {output}")


if __name__ == "__main__":
    main()
