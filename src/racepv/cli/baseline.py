"""Train leakage-controlled RaCE-PV baselines on the registered NPZ interface."""

from __future__ import annotations

import argparse
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from racepv.audit import sha256_file
from racepv.baselines import (
    ClearSkyPersistence,
    LSTMQuantile,
    PatchTSTQuantile,
    QuantileLightGBM,
    RandomForestPointForecaster,
    SMambaQuantile,
    SmartPersistence,
    TCNQuantile,
    ramp_state_probabilities_from_median,
)
from racepv.config import load_config, save_config
from racepv.environment import environment_report
from racepv.io import (
    WindowDataset,
    load_window_dataset,
    require_global_split_chronology,
    require_registered_spatial_protocol,
    save_predictions_npz,
    window_dataset_content_identity,
)
from racepv.losses import pinball_loss
from racepv.training import (
    EFFICIENCY_BENCHMARK_BATCH_SIZE,
    EFFICIENCY_BENCHMARK_REPETITIONS,
    EFFICIENCY_BENCHMARK_WARMUP,
    EarlyStoppingTrainer,
    benchmark_callable,
    benchmark_inference,
    build_window_loader,
    count_parameters,
    efficiency_audit_record,
)
from racepv.utils import atomic_json_dump, seed_everything, select_device

LOCAL_BASELINES = (
    "smart_persistence",
    "clear_sky_persistence",
    "quantile_lightgbm",
    "random_forest_mondrian",
    "lstm_qr",
    "tcn_qr",
    "patchtst_qr",
    "s_mamba_qr",
)


class _QuantileOnlyLoss(nn.Module):
    def __init__(self, quantiles: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer("levels", torch.as_tensor(tuple(quantiles), dtype=torch.float32))

    def forward(
        self,
        outputs: Mapping[str, Tensor],
        targets: Tensor,
        state_labels: Tensor,
        *,
        daylight_mask: Tensor | None = None,
    ) -> Tensor:
        del state_labels, daylight_mask
        return pinball_loss(outputs["quantiles"], targets, self.levels)


def _loader(dataset: WindowDataset, batch_size: int, shuffle: bool, workers: int):
    return build_window_loader(dataset, batch_size, shuffle, workers)


TREE_SIGNAL_NAMES = (
    "power_norm",
    "clear_sky_residual",
    "power_diff_1",
    "power_diff_2",
    "rolling_volatility_1h",
    "rolling_volatility_3h",
    "clear_sky_index",
)
TREE_LAGS = (1, 2, 4, 8, 12, 24, 48, 96, 192)
TREE_WINDOWS = (4, 12, 24, 48, 96, 192)


def _tree_batch_features(
    batch: Mapping[str, Tensor],
    *,
    history_names: Sequence[str],
) -> np.ndarray:
    """Compact, fully causal tree features without flattening every window cell.

    A literal flattening of 192 x (values + masks) for millions of overlapping
    windows is prohibitively large.  This registered representation retains
    recent multiscale values, missingness, causal moments, every future-known
    value, and static site metadata.  Its exact specification is written to the
    run artifact so it cannot be silently changed between models.
    """

    history = np.nan_to_num(batch["history_values"].numpy(), nan=0.0).astype(np.float32, copy=False)
    mask = batch["history_mask"].numpy().astype(np.float32, copy=False)
    # Defence in depth for legacy NPZ inputs: no tree feature may expose a
    # bidirectionally interpolated value whose issue-time availability is zero.
    history = history * mask
    future = np.nan_to_num(batch["future_known"].numpy(), nan=0.0).astype(np.float32, copy=False)
    site = np.nan_to_num(batch["site_metadata"].numpy(), nan=0.0).astype(np.float32, copy=False)
    selected = [history_names.index(name) for name in TREE_SIGNAL_NAMES]
    pieces = [history[:, -1, :], mask[:, -1, :]]
    for lag in TREE_LAGS:
        offset = min(lag, history.shape[1])
        pieces.extend(
            (
                history[:, -offset, selected],
                mask[:, -offset, selected],
            )
        )
    for width in TREE_WINDOWS:
        width = min(width, history.shape[1])
        values = history[:, -width:, :][:, :, selected]
        available = mask[:, -width:, :][:, :, selected]
        denominator = np.maximum(available.sum(axis=1), 1.0)
        mean = (values * available).sum(axis=1) / denominator
        variance = (((values - mean[:, None, :]) ** 2) * available).sum(axis=1) / denominator
        pieces.extend((mean, np.sqrt(np.maximum(variance, 0.0))))
    for width in (24, 192):
        width = min(width, history.shape[1])
        pieces.append(mask[:, -width:, :].mean(axis=1))
    pieces.extend((future.reshape(future.shape[0], -1), site))
    return np.concatenate(pieces, axis=1).astype(np.float32, copy=False)


def _tree_feature_store(
    dataset: WindowDataset,
    destination: Path,
    *,
    batch_size: int,
    workers: int,
) -> tuple[np.memmap, np.memmap, dict[str, object]]:
    if dataset.schema is None:
        raise ValueError("registered feature names are required for tree baselines")
    history_names = tuple(dataset.schema["history_columns"])
    sample_loader = _loader(dataset, min(batch_size, len(dataset)), False, workers)
    first = next(iter(sample_loader))
    first_features = _tree_batch_features(first, history_names=history_names)
    features = np.lib.format.open_memmap(
        destination / "features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), first_features.shape[1]),
    )
    targets = np.lib.format.open_memmap(
        destination / "targets.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(dataset), int(first["targets"].shape[1])),
    )
    cursor = 0
    # Recreate the loader so the first batch is not treated specially and the
    # extraction order exactly matches prediction/audit arrays.
    for batch in _loader(dataset, batch_size, False, workers):
        values = _tree_batch_features(batch, history_names=history_names)
        stop = cursor + len(values)
        features[cursor:stop] = values
        targets[cursor:stop] = batch["targets"].numpy().astype(np.float32, copy=False)
        cursor = stop
    if cursor != len(dataset):
        raise RuntimeError("tree feature extraction did not visit every sample")
    features.flush()
    targets.flush()
    specification: dict[str, object] = {
        "representation": "registered_causal_multiscale_v1",
        "history_signal_names": list(TREE_SIGNAL_NAMES),
        "lag_steps": list(TREE_LAGS),
        "moment_windows_steps": list(TREE_WINDOWS),
        "availability_windows_steps": [24, 192],
        "includes_last_value_and_mask_for_all_history_channels": True,
        "masked_history_values_are_zeroed": True,
        "includes_all_future_known_cells": True,
        "includes_all_site_metadata": True,
        "feature_count": int(features.shape[1]),
    }
    return features, targets, specification


def _close_memmaps(*values: np.memmap) -> None:
    """Release Windows file handles before the temporary feature cache is removed."""

    for value in values:
        value.flush()
        handle = getattr(value, "_mmap", None)
        if handle is not None:
            handle.close()


def _sample_ramp_thresholds(dataset: WindowDataset) -> np.ndarray:
    """Return each origin's registered train-only site threshold.

    Computing this statistic from overlapping windows would over-weight interior
    timestamps and pooling sites would disagree with the label definition.  The
    preparation stage therefore persists the fitted per-site value in the schema.
    """

    if dataset.schema is None:  # narrowed by require_registered_schema
        raise ValueError("registered schema is required for ramp-state inference")
    raw = dataset.schema.get("ramp_thresholds")
    if not isinstance(raw, dict):
        raise ValueError("schema is missing train-only site ramp thresholds")
    lookup: dict[int, float] = {}
    for key, state in raw.items():
        state = raw.get(key)
        if not isinstance(state, dict) or "threshold" not in state:
            raise ValueError(f"schema has no fitted ramp threshold for site {key}")
        threshold = float(state["threshold"])
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError(f"invalid ramp threshold for site {key}: {threshold}")
        lookup[int(key)] = threshold
    values = np.empty(len(dataset), dtype=np.float32)
    site_ids = dataset.arrays["site_ids"]
    for index in range(len(dataset)):
        site_id = int(np.asarray(site_ids[index]).reshape(-1)[0])
        if site_id not in lookup:
            raise ValueError(f"schema has no fitted ramp threshold for site {site_id}")
        values[index] = lookup[site_id]
    return values


def _prediction_payload(
    dataset: WindowDataset,
    quantiles: np.ndarray,
    levels: Sequence[float],
    thresholds: np.ndarray,
    *,
    batch_size: int = 4096,
    workers: int = 0,
) -> dict[str, np.ndarray]:
    quantiles = np.sort(np.clip(np.asarray(quantiles), 0.0, 1.05), axis=-1)
    expected = (len(dataset), 24, len(levels))
    if quantiles.shape != expected:
        raise ValueError(f"quantile predictions have shape {quantiles.shape}, expected {expected}")
    targets = np.empty((len(dataset), 24), dtype=np.float32)
    labels = np.empty((len(dataset), 24), dtype=np.int64)
    daylight = np.empty((len(dataset), 24), dtype=bool)
    timestamps = np.empty((len(dataset), 24), dtype=np.int64)
    day_ids = np.empty((len(dataset), 24), dtype=np.int32)
    site_ids = np.empty(len(dataset), dtype=np.int64)
    state_probs = np.empty((len(dataset), 24, 3), dtype=np.float32)
    median = quantiles[..., tuple(levels).index(0.5)]
    cursor = 0
    for batch in _loader(dataset, batch_size, False, workers):
        count = int(batch["targets"].shape[0])
        stop = cursor + count
        batch_daylight = batch["daylight_mask"].numpy().astype(bool, copy=False)
        daylight[cursor:stop] = batch_daylight
        targets[cursor:stop] = batch["targets"].numpy()
        labels[cursor:stop] = batch["state_labels"].numpy()
        timestamps[cursor:stop] = batch["timestamps"].numpy()
        day_ids[cursor:stop] = batch["day_ids"].numpy()
        site_ids[cursor:stop] = batch["site_ids"].numpy().reshape(-1)
        quantiles[cursor:stop][~batch_daylight] = 0.0
        history_mask = batch["history_mask"].numpy().astype(bool, copy=False)[..., 0]
        state_probs[cursor:stop] = ramp_state_probabilities_from_median(
            median[cursor:stop],
            batch["history_values"].numpy()[..., 0],
            thresholds[cursor:stop],
            batch_daylight,
            history_mask,
        )
        cursor = stop
    if cursor != len(dataset):
        raise RuntimeError("prediction payload extraction did not visit every sample")
    output = {
        "quantiles": quantiles.astype(np.float32),
        "state_probs": state_probs,
        "targets": targets,
        "state_labels": labels,
        "daylight_mask": daylight.astype(np.float32),
        "timestamps": timestamps,
        "day_ids": day_ids,
        "site_ids": site_ids,
    }
    return output


def _persistence_quantiles(
    name: str,
    dataset: WindowDataset,
    quantiles: Sequence[float],
    *,
    batch_size: int = 4096,
    workers: int = 0,
) -> np.ndarray:
    output = np.empty((len(dataset), 24, len(quantiles)), dtype=np.float32)
    cursor = 0
    for batch in _loader(dataset, batch_size, False, workers):
        values = _persistence_batch_quantiles(name, batch, quantiles)
        stop = cursor + len(values)
        output[cursor:stop] = values
        cursor = stop
    if cursor != len(dataset):
        raise RuntimeError("persistence prediction did not visit every sample")
    return output


def _persistence_batch_quantiles(
    name: str,
    batch: Mapping[str, Tensor],
    quantiles: Sequence[float],
) -> np.ndarray:
    history = batch["history_values"].numpy()
    future = batch["future_known"].numpy()
    if name == "smart_persistence":
        history_mask = batch["history_mask"].numpy().astype(bool, copy=False)
        valid = history_mask[..., 0] & history_mask[..., 1]
        point = SmartPersistence().predict(history[..., 0], history[..., 1], future[..., 0], valid)
    elif name == "clear_sky_persistence":
        point = ClearSkyPersistence().predict(future[..., 0])
    else:
        raise KeyError(name)
    return np.repeat(np.asarray(point, dtype=np.float32)[..., None], len(quantiles), axis=-1)


def _tree_batch_quantiles(
    name: str,
    model: object,
    features: np.ndarray,
    quantile_count: int,
) -> np.ndarray:
    if name == "quantile_lightgbm":
        return np.asarray(model.predict(features), dtype=np.float32)
    if name == "random_forest_mondrian":
        point = np.asarray(model.predict(features), dtype=np.float32)
        return np.repeat(point[..., None], quantile_count, axis=-1)
    raise KeyError(name)


def _build_neural(name: str, sample: Mapping[str, Tensor], quantiles: Sequence[float]) -> nn.Module:
    common = {
        "input_dim": int(sample["history_values"].shape[-1]),
        "future_dim": int(sample["future_known"].shape[-1]),
        "site_dim": int(sample["site_metadata"].shape[-1]),
        "quantiles": quantiles,
    }
    if name == "lstm_qr":
        return LSTMQuantile(**common)
    if name == "tcn_qr":
        return TCNQuantile(**common)
    if name == "patchtst_qr":
        return PatchTSTQuantile(**common)
    if name == "s_mamba_qr":
        return SMambaQuantile(**common)
    raise KeyError(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    from racepv.cli.modern_baseline import _add_prepared_expectations

    _add_prepared_expectations(parser)
    parser.add_argument("--name", choices=LOCAL_BASELINES, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "fit on train and write validation predictions only; calibration/test "
            "arguments are forbidden in this screening mode"
        ),
    )
    parser.add_argument(
        "--allow-development-data",
        action="store_true",
        help="allow provenance-complete synthetic/debug data; never for reported evidence",
    )
    parser.add_argument(
        "--disable-mixed-precision",
        action="store_true",
        help=(
            "run the neural baseline in FP32; currently registered only for "
            "S-Mamba after a full-panel AMP non-finite-loss failure"
        ),
    )
    return parser


def _effective_mixed_precision(name: str, *, configured: bool, disabled: bool) -> bool:
    if disabled and name != "s_mamba_qr":
        raise ValueError("--disable-mixed-precision is registered only for s_mamba_qr")
    return bool(configured and not disabled)


def _resolve_input_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve an explicit formal or validation-only split contract.

    Requiring ``--validation-only`` prevents an accidentally omitted held-out
    path from silently changing a registered four-split run into a pilot.
    Conversely, forbidding held-out paths in pilot mode makes it impossible for
    the screening runner to load calibration or test archives.
    """

    if args.validation_only:
        forbidden = {
            name: path
            for name, path in (("calibration", args.calibration), ("test", args.test))
            if path is not None
        }
        if forbidden:
            raise ValueError(
                f"--validation-only forbids calibration/test inputs; received {sorted(forbidden)}"
            )
        return {"train": args.train, "validation": args.validation}
    missing = [
        name
        for name, path in (("calibration", args.calibration), ("test", args.test))
        if path is None
    ]
    if missing:
        raise ValueError(
            "formal baseline runs require --calibration and --test; "
            f"missing {missing}. Use --validation-only for screening."
        )
    return {
        "train": args.train,
        "validation": args.validation,
        "calibration": args.calibration,
        "test": args.test,
    }


def _selection_boundary_metadata(validation_only: bool) -> dict[str, object]:
    return {
        "selection_boundary": (
            "validation_only" if validation_only else "registered_split_protocol"
        ),
        "validation_only_screening": bool(validation_only),
        "calibration_or_test_loaded": not validation_only,
    }


def _descriptor_hash(path: Path) -> str:
    """Hash the NPZ/index descriptor or a single-store manifest."""

    source = path / "store.json" if path.is_dir() else path
    return sha256_file(source)


def _spatial_roles(dataset: WindowDataset) -> set[str]:
    if dataset.schema is None:
        raise ValueError("registered schema is required for spatial-role enforcement")
    roles = dataset.schema.get("spatial_roles")
    if not isinstance(roles, dict):
        raise ValueError("registered schema is missing spatial_roles")
    return {str(value) for value in roles.values()}


def _require_collection_split(dataset: WindowDataset, expected: str) -> None:
    declared = getattr(dataset, "collection_split", None)
    if declared is not None and declared != expected:
        raise ValueError(f"{expected} input points to a collection declared as split={declared!r}")


def _enforce_fit_boundaries(datasets: Mapping[str, WindowDataset]) -> None:
    train_roles = _spatial_roles(datasets["train"])
    if not train_roles.issubset({"standard", "source_train"}):
        raise ValueError(
            "baseline fitting may only use standard/source_train archives; "
            f"received {sorted(train_roles)}"
        )
    for split in (name for name in ("validation", "calibration") if name in datasets):
        if "zero_parameter_test" in _spatial_roles(datasets[split]):
            raise ValueError(
                f"{split} cannot use zero_parameter_test sites for model selection "
                "or conformal fitting"
            )


def main() -> None:
    args = build_parser().parse_args()
    from racepv.cli.modern_baseline import _preflight_prepared_inputs

    _preflight_prepared_inputs(args, legacy_only=True)
    config = load_config(args.config)
    effective_mixed_precision = _effective_mixed_precision(
        args.name,
        configured=config.training.mixed_precision,
        disabled=args.disable_mixed_precision,
    )
    seed_everything(args.seed)
    paths = _resolve_input_paths(args)
    datasets = {
        name: load_window_dataset(path, verify_content=True) for name, path in paths.items()
    }
    for split, dataset in datasets.items():
        dataset.require_registered_schema()
        dataset.require_registered_evidence(allow_ineligible=args.allow_development_data)
        _require_collection_split(dataset, split)
        if dataset.registered_schema_signature() != datasets["train"].registered_schema_signature():
            raise ValueError("all baseline archives must share the registered feature schema")
    temporal_audit = require_global_split_chronology(datasets)
    spatial_protocol = require_registered_spatial_protocol(datasets)
    _enforce_fit_boundaries(datasets)
    split_thresholds = {
        split: _sample_ramp_thresholds(dataset) for split, dataset in datasets.items()
    }
    prediction_splits = tuple(name for name in paths if name != "train")
    descriptor_hashes = {name: _descriptor_hash(path) for name, path in paths.items()}
    content_hashes = {
        name: window_dataset_content_identity(path, verify=False) for name, path in paths.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    save_config(config, args.output / "resolved_config.yaml")
    run: dict[str, object] = {
        "baseline": args.name,
        "seed": args.seed,
        "purpose": (
            "training_validation_only_baseline_screening"
            if args.validation_only
            else "registered_four_split_baseline"
        ),
        "prediction_splits": list(prediction_splits),
        **_selection_boundary_metadata(args.validation_only),
        "precision": {
            "configured_mixed_precision": config.training.mixed_precision,
            "disable_mixed_precision_cli": args.disable_mixed_precision,
            "effective_mixed_precision": effective_mixed_precision,
        },
        "ramp_threshold_source": "prepared train-only per-site schema",
        "state_probability_method": (
            "continuous softmax of threshold-normalized 30-minute median-forecast "
            "change margins; masked history is causally forward-filled"
        ),
        "evidence_mode": {
            "allow_development_data": args.allow_development_data,
            "dataset_kind": datasets["train"].schema["dataset_kind"],
            "simulated": datasets["train"].schema["simulated"],
            "registered_evidence_eligible": datasets["train"].schema[
                "registered_evidence_eligible"
            ],
        },
        "global_temporal_audit": temporal_audit,
        "registered_spatial_protocol": spatial_protocol,
        "ramp_thresholds_by_split_and_site": {
            split: datasets[split].schema["ramp_thresholds"] for split in datasets
        },
        "environment": environment_report(),
        "artifacts": {
            **{f"{split}_descriptor_sha256": descriptor_hashes[split] for split in paths},
            **{f"{split}_window_content_sha256": content_hashes[split] for split in paths},
            "config_sha256": sha256_file(args.config),
        },
    }
    artifact_identity: dict[str, object] = {
        "config_sha256": sha256_file(args.config),
        "effective_mixed_precision": effective_mixed_precision,
    }

    def metadata(split: str) -> dict[str, object]:
        return {
            "baseline": args.name,
            "split": split,
            "seed": args.seed,
            "quantiles": list(config.quantiles),
            "source_npz_sha256": descriptor_hashes[split],
            "source_window_content_sha256": content_hashes[split],
            "site_id_map": datasets[split].schema["site_id_map"],
            "dataset_kind": datasets[split].schema["dataset_kind"],
            "simulated": datasets[split].schema["simulated"],
            "ramp_method": datasets[split].schema["ramp_method"],
            "registered_evidence_eligible": datasets[split].schema["registered_evidence_eligible"],
            "development_data_override": args.allow_development_data,
            **_selection_boundary_metadata(args.validation_only),
            "state_probability_method": (
                "continuous_threshold_margin_softmax_from_median_forecast"
            ),
            **artifact_identity,
        }

    if args.name in {"smart_persistence", "clear_sky_persistence"}:
        benchmark_batch = next(
            iter(
                _loader(
                    datasets["validation"],
                    EFFICIENCY_BENCHMARK_BATCH_SIZE,
                    False,
                    config.training.num_workers,
                )
            )
        )
        benchmark_size = int(benchmark_batch["targets"].shape[0])
        batch_inference = benchmark_callable(
            lambda: _persistence_batch_quantiles(args.name, benchmark_batch, config.quantiles),
            batch_size=benchmark_size,
            device="cpu",
            warmup=EFFICIENCY_BENCHMARK_WARMUP,
            repetitions=EFFICIENCY_BENCHMARK_REPETITIONS,
            scope=(
                "persistence_predict_on_preloaded_registered_windows_excluding_"
                "payload_postprocessing_and_artifact_io"
            ),
        )
        started = time.perf_counter()
        for split in prediction_splits:
            values = _persistence_quantiles(
                args.name,
                datasets[split],
                config.quantiles,
                batch_size=max(config.training.batch_size, 512),
                workers=config.training.num_workers,
            )
            save_predictions_npz(
                args.output / f"{split}_predictions.npz",
                _prediction_payload(
                    datasets[split],
                    values,
                    config.quantiles,
                    split_thresholds[split],
                    batch_size=max(config.training.batch_size, 512),
                    workers=config.training.num_workers,
                ),
                metadata=metadata(split),
            )
        run["trainable_parameters"] = 0
        run["efficiency"] = efficiency_audit_record(
            trainable_parameters=0,
            trainable_parameters_unavailable_reason=None,
            model_artifact_bytes=0,
            model_artifact_definition=("no fitted model artifact; deterministic persistence rule"),
            batch_inference=batch_inference,
        )
        generation_seconds = time.perf_counter() - started
        run["prediction_generation_and_save_seconds"] = generation_seconds
        if not args.validation_only:
            run["validation_calibration_test_generation_and_save_seconds"] = generation_seconds
    elif args.name in {"quantile_lightgbm", "random_forest_mondrian"}:
        import joblib

        with tempfile.TemporaryDirectory(
            dir=args.output,
            prefix="tree-feature-cache-",
            ignore_cleanup_errors=True,
        ) as scratch_text:
            scratch = Path(scratch_text)
            train_cache = scratch / "train"
            train_cache.mkdir()
            train_x, train_y, feature_specification = _tree_feature_store(
                datasets["train"],
                train_cache,
                batch_size=max(config.training.batch_size, 512),
                workers=config.training.num_workers,
            )
            if args.name == "quantile_lightgbm":
                model = QuantileLightGBM(config.quantiles, random_state=args.seed).fit(
                    train_x, train_y
                )
            else:
                model = RandomForestPointForecaster(random_state=args.seed).fit(train_x, train_y)

            model_path = args.output / "model.joblib"
            joblib.dump(model, model_path)
            model_hash = sha256_file(model_path)
            artifact_identity["model_sha256"] = model_hash
            _close_memmaps(train_x, train_y)
            prediction_seconds = 0.0
            benchmark_features: np.ndarray | None = None
            for split in prediction_splits:
                split_cache = scratch / split
                split_cache.mkdir()
                features, split_targets, split_specification = _tree_feature_store(
                    datasets[split],
                    split_cache,
                    batch_size=max(config.training.batch_size, 512),
                    workers=config.training.num_workers,
                )
                if split_specification != feature_specification:
                    raise ValueError("tree feature specification changed between splits")
                if split == "validation":
                    benchmark_size = min(EFFICIENCY_BENCHMARK_BATCH_SIZE, len(features))
                    benchmark_features = np.asarray(
                        features[:benchmark_size], dtype=np.float32
                    ).copy()
                started = time.perf_counter()
                values = _tree_batch_quantiles(args.name, model, features, len(config.quantiles))
                prediction_seconds += time.perf_counter() - started
                save_predictions_npz(
                    args.output / f"{split}_predictions.npz",
                    _prediction_payload(
                        datasets[split],
                        values,
                        config.quantiles,
                        split_thresholds[split],
                        batch_size=max(config.training.batch_size, 512),
                        workers=config.training.num_workers,
                    ),
                    metadata=metadata(split),
                )
                _close_memmaps(features, split_targets)
                del features, split_targets, values
            del train_x, train_y
            if benchmark_features is None:
                raise RuntimeError("validation tree features are unavailable for benchmarking")
            batch_inference = benchmark_callable(
                lambda: _tree_batch_quantiles(
                    args.name,
                    model,
                    benchmark_features,
                    len(config.quantiles),
                ),
                batch_size=len(benchmark_features),
                device="cpu",
                warmup=EFFICIENCY_BENCHMARK_WARMUP,
                repetitions=EFFICIENCY_BENCHMARK_REPETITIONS,
                scope=(
                    "tree_predict_on_precomputed_registered_features_excluding_"
                    "feature_construction_payload_postprocessing_and_artifact_io"
                ),
            )
        run["tree_feature_specification"] = feature_specification
        run["feature_count"] = int(feature_specification["feature_count"])
        run["prediction_seconds"] = prediction_seconds
        if not args.validation_only:
            run["validation_calibration_test_prediction_seconds"] = prediction_seconds
        run["model_sha256"] = model_hash
        run["model_bytes"] = model_path.stat().st_size
        run["trainable_parameters"] = None
        run["efficiency"] = efficiency_audit_record(
            trainable_parameters=None,
            trainable_parameters_unavailable_reason=(
                "fitted tree ensembles have no framework-independent count of "
                "differentiable trainable scalars"
            ),
            model_artifact_bytes=model_path.stat().st_size,
            model_artifact_definition="serialized model.joblib file size",
            batch_inference=batch_inference,
        )
    else:
        device = select_device(args.device)
        if args.name == "s_mamba_qr" and torch.device(device).type != "cuda":
            raise RuntimeError(
                "S-Mamba-QR is a registered official mamba_ssm CUDA/WSL baseline; "
                "select a CUDA device instead of running it on native Windows/CPU"
            )
        model = _build_neural(args.name, datasets["train"][0], config.quantiles)
        criterion = _QuantileOnlyLoss(config.quantiles)
        trainer = EarlyStoppingTrainer(
            model,
            criterion,
            device=device,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            max_epochs=config.training.max_epochs,
            patience=config.training.patience,
            gradient_clip_norm=config.training.gradient_clip_norm,
            mixed_precision=effective_mixed_precision,
            checkpoint_path=args.output / "best.ckpt",
            checkpoint_metadata={
                "baseline": args.name,
                "seed": args.seed,
                "quantiles": list(config.quantiles),
                "input_dim": int(datasets["train"].arrays["history_values"].shape[-1]),
                "future_dim": int(datasets["train"].arrays["future_known"].shape[-1]),
                "site_dim": int(datasets["train"].arrays["site_metadata"].shape[-1]),
                "actual_backbone": ("mamba_ssm" if args.name == "s_mamba_qr" else args.name),
                "effective_mixed_precision": effective_mixed_precision,
                "train_descriptor_sha256": descriptor_hashes["train"],
                "validation_descriptor_sha256": descriptor_hashes["validation"],
                "train_window_content_sha256": content_hashes["train"],
                "validation_window_content_sha256": content_hashes["validation"],
            },
        )
        train_loader = _loader(
            datasets["train"], config.training.batch_size, True, config.training.num_workers
        )
        validation_loader = _loader(
            datasets["validation"],
            config.training.batch_size,
            False,
            config.training.num_workers,
        )
        fit = trainer.fit(train_loader, validation_loader)
        checkpoint_hash = sha256_file(args.output / "best.ckpt")
        artifact_identity["best_checkpoint_sha256"] = checkpoint_hash
        run["best_checkpoint_sha256"] = checkpoint_hash
        run["actual_backbone"] = "mamba_ssm" if args.name == "s_mamba_qr" else args.name
        model_path = args.output / "model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "baseline": args.name,
                "seed": args.seed,
                "quantiles": list(config.quantiles),
                "input_dim": int(datasets["train"].arrays["history_values"].shape[-1]),
                "future_dim": int(datasets["train"].arrays["future_known"].shape[-1]),
                "site_dim": int(datasets["train"].arrays["site_metadata"].shape[-1]),
                "actual_backbone": run["actual_backbone"],
                "effective_mixed_precision": effective_mixed_precision,
            },
            model_path,
        )
        model_hash = sha256_file(model_path)
        artifact_identity["model_sha256"] = model_hash
        run["model_sha256"] = model_hash
        for split in prediction_splits:
            loader = _loader(
                datasets[split],
                config.training.batch_size,
                False,
                config.training.num_workers,
            )
            predicted = trainer.predict(loader)
            payload = _prediction_payload(
                datasets[split],
                predicted["quantiles"].numpy(),
                config.quantiles,
                split_thresholds[split],
                batch_size=max(config.training.batch_size, 512),
                workers=config.training.num_workers,
            )
            save_predictions_npz(
                args.output / f"{split}_predictions.npz",
                payload,
                metadata=metadata(split),
            )
        benchmark_loader = _loader(
            datasets["validation"],
            EFFICIENCY_BENCHMARK_BATCH_SIZE,
            False,
            config.training.num_workers,
        )
        benchmark_batch = next(iter(benchmark_loader))
        batch_inference = benchmark_inference(
            model,
            {
                key: benchmark_batch[key]
                for key in (
                    "history_values",
                    "history_mask",
                    "future_known",
                    "site_metadata",
                )
            },
            warmup=EFFICIENCY_BENCHMARK_WARMUP,
            repetitions=EFFICIENCY_BENCHMARK_REPETITIONS,
        )
        trainable_parameters = count_parameters(model)
        model_bytes = model_path.stat().st_size
        run.update(
            {
                "device": device,
                "best_epoch": fit.best_epoch,
                "best_validation_loss": fit.best_validation_loss,
                "trainable_parameters": trainable_parameters,
                "efficiency": efficiency_audit_record(
                    trainable_parameters=trainable_parameters,
                    trainable_parameters_unavailable_reason=None,
                    model_artifact_bytes=model_bytes,
                    model_artifact_definition="serialized inference model.pt file size",
                    batch_inference=batch_inference,
                ),
            }
        )
    atomic_json_dump(run, args.output / "run.json")
    print(f"Completed baseline {args.name}; outputs are in {args.output}")


if __name__ == "__main__":
    main()
