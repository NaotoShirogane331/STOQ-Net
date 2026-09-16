"""Run official NeuralForecast N-HiTS/TFT/xLSTM on registered RaCE-PV windows."""

from __future__ import annotations

import argparse
import gc
import hashlib
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from racepv.audit import sha256_file
from racepv.baselines.classical import ramp_state_probabilities_from_median
from racepv.baselines.external import (
    FREQUENCY,
    HISTORICAL_EXOGENOUS,
    HORIZON,
    LOOKBACK,
    OFFICIAL_RUNNER_MODELS,
    STATIC_EXOGENOUS,
    FutureExogenousPlan,
    build_neuralforecast_model,
    build_prediction_frames,
    build_training_panel,
    extract_official_quantiles,
    load_neuralforecast_class,
    official_model_metadata,
    read_window_batch,
    resolve_future_exogenous_plan,
    validate_archive_contract,
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
from racepv.training import (
    EFFICIENCY_BENCHMARK_BATCH_SIZE,
    EFFICIENCY_BENCHMARK_REPETITIONS,
    EFFICIENCY_BENCHMARK_WARMUP,
    benchmark_callable,
    efficiency_audit_record,
)
from racepv.utils import atomic_json_dump, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    from racepv.cli.modern_baseline import _add_prepared_expectations

    _add_prepared_expectations(parser)
    parser.add_argument("--name", choices=OFFICIAL_RUNNER_MODELS, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--max-steps",
        type=int,
        required=True,
        help=(
            "Official NeuralForecast optimizer-step budget. This is not an epoch count "
            "and is recorded as a deviation from the nominal 80-epoch protocol."
        ),
    )
    parser.add_argument("--val-check-steps", type=int, default=100)
    parser.add_argument(
        "--series-batch-size",
        type=int,
        default=1,
        help="Number of site series loaded before window sampling; fixed to 1 for long registered panels.",
    )
    parser.add_argument(
        "--windows-batch-size",
        type=int,
        default=256,
        help="Training windows sampled per optimizer step; 256 is the registered 24GB-GPU setting.",
    )
    parser.add_argument("--inference-windows-batch-size", type=int, default=256)
    parser.add_argument(
        "--prediction-batch-windows",
        type=int,
        default=256,
        help="Number of lazy issue-time contexts materialised per predict call.",
    )
    parser.add_argument(
        "--payload-batch-windows",
        type=int,
        default=2048,
        help="Bounded batch used to copy audit arrays into each prediction archive.",
    )
    parser.add_argument(
        "--max-prediction-context-rows",
        type=int,
        default=100_000,
        help="Hard guard on prediction_batch_windows * 192.",
    )
    parser.add_argument(
        "--max-panel-rows",
        type=int,
        default=3_100_000,
        help=(
            "Hard guard checked before the deduplicated panel is allocated; the default "
            "covers the registered DKASC panel (3,024,345 rows) and the smaller "
            "60-site NREL training panel."
        ),
    )
    parser.add_argument(
        "--origin-stride-steps",
        type=int,
        default=None,
        help="Prepared window stride; omitted means infer the GCD from train origins.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "fit on train, save the official model, and predict validation only; "
            "calibration/test arguments are forbidden in this screening mode"
        ),
    )
    parser.add_argument(
        "--allow-development-data",
        action="store_true",
        help="allow provenance-complete synthetic/debug data; never for reported evidence",
    )
    return parser


def _resolve_input_paths(args: argparse.Namespace) -> dict[str, Path]:
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
            "formal official-baseline runs require --calibration and --test; "
            f"missing {missing}. Use --validation-only for screening."
        )
    return {
        "train": args.train,
        "validation": args.validation,
        "calibration": args.calibration,
        "test": args.test,
    }


def _validate_cli(args: argparse.Namespace) -> None:
    positive = {
        "max_steps": args.max_steps,
        "val_check_steps": args.val_check_steps,
        "series_batch_size": args.series_batch_size,
        "windows_batch_size": args.windows_batch_size,
        "inference_windows_batch_size": args.inference_windows_batch_size,
        "prediction_batch_windows": args.prediction_batch_windows,
        "payload_batch_windows": args.payload_batch_windows,
        "max_prediction_context_rows": args.max_prediction_context_rows,
        "max_panel_rows": args.max_panel_rows,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"official baseline limits must be positive: {invalid}")
    if args.origin_stride_steps is not None and not 1 <= args.origin_stride_steps <= LOOKBACK:
        raise ValueError(f"origin_stride_steps must be in [1,{LOOKBACK}]")
    context_rows = args.prediction_batch_windows * LOOKBACK
    if context_rows > args.max_prediction_context_rows:
        raise MemoryError(
            f"prediction chunks would materialise {context_rows:,} history rows, above "
            f"--max-prediction-context-rows={args.max_prediction_context_rows:,}; "
            "lower --prediction-batch-windows or raise the guard explicitly"
        )


def _trainer_device_kwargs(device: str) -> dict[str, object]:
    accelerator = {"cuda": "gpu"}.get(device, device)
    return {"accelerator": accelerator, "devices": 1}


def _state_dict_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        try:
            payload = tensor.numpy().tobytes(order="C")
        except TypeError:  # e.g. bfloat16 has no direct NumPy representation
            payload = tensor.float().numpy().tobytes(order="C")
        digest.update(payload)
    return digest.hexdigest()


def _json_metrics(model: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    raw = getattr(model, "metrics", {}) or {}
    for name, value in raw.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        try:
            result[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _artifact_manifest(root: Path) -> list[dict[str, object]]:
    artifacts = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=str):
        artifacts.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not artifacts:
        raise RuntimeError("NeuralForecast.save produced no model artifacts")
    return artifacts


def _descriptor_path(path: Path) -> Path:
    return path / "store.json" if path.is_dir() else path


def _descriptor_sha256(path: Path) -> str:
    return sha256_file(_descriptor_path(path))


def _schema_path(path: Path) -> Path | None:
    candidate = path / "schema.json" if path.is_dir() else path.with_suffix(".schema.json")
    return candidate if candidate.exists() else None


def _input_artifacts(
    args: argparse.Namespace,
    *,
    descriptor_hashes: Mapping[str, str],
    content_hashes: Mapping[str, str],
) -> dict[str, object]:
    result: dict[str, object] = {"config_sha256": sha256_file(args.config)}
    for split in descriptor_hashes:
        path: Path = getattr(args, split)
        descriptor = _descriptor_path(path)
        state: dict[str, object] = {
            "path": str(path),
            "descriptor_path": str(descriptor),
            "descriptor_bytes": descriptor.stat().st_size,
            "descriptor_sha256": descriptor_hashes[split],
            "window_content_sha256": content_hashes[split],
        }
        schema = _schema_path(path)
        if schema is not None:
            state["schema_path"] = str(schema)
            state["schema_sha256"] = sha256_file(schema)
        result[split] = state
    return result


def _spatial_roles(dataset: WindowDataset) -> set[str]:
    if dataset.schema is None:
        raise ValueError("registered schema is required for spatial-role enforcement")
    roles = dataset.schema.get("spatial_roles")
    if not isinstance(roles, dict) or not roles:
        raise ValueError("registered schema is missing spatial_roles")
    return {str(value) for value in roles.values()}


def _enforce_fit_roles(datasets: Mapping[str, WindowDataset]) -> dict[str, list[str]]:
    roles = {split: _spatial_roles(dataset) for split, dataset in datasets.items()}
    if not roles["train"].issubset({"standard", "source_train"}):
        raise ValueError(
            "official baseline training may only use standard/source_train archives; "
            f"received {sorted(roles['train'])}"
        )
    checked_splits = (
        (split, purpose)
        for split, purpose in (
            ("validation", "post-fit validation/model preselection"),
            ("calibration", "later conformal fitting"),
        )
        if split in roles
    )
    for split, purpose in checked_splits:
        if "zero_parameter_test" in roles[split]:
            raise ValueError(
                f"{purpose} cannot use a zero_parameter_test archive; target sites are "
                "reserved for prediction without model or conformal updates"
            )
    return {split: sorted(values) for split, values in roles.items()}


def _threshold_lookup(dataset: WindowDataset) -> dict[int, float]:
    if dataset.schema is None:
        raise ValueError("registered schema is required for ramp-state inference")
    raw = dataset.schema.get("ramp_thresholds")
    if not isinstance(raw, dict):
        raise ValueError("schema is missing train-only ramp thresholds")
    lookup: dict[int, float] = {}
    for key, state in raw.items():
        if not isinstance(state, dict) or "threshold" not in state:
            raise ValueError(f"schema has no fitted ramp threshold for site {key}")
        value = float(state["threshold"])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"invalid ramp threshold for site {key}: {value}")
        lookup[int(key)] = value
    return lookup


def _prediction_payload(
    dataset: WindowDataset,
    raw_quantiles: np.ndarray,
    levels: Sequence[float],
    *,
    batch_windows: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    raw = np.asarray(raw_quantiles, dtype=np.float32)
    expected_shape = (len(dataset), HORIZON, len(levels))
    if raw.shape != expected_shape:
        raise ValueError(f"official quantiles have shape {raw.shape}, expected {expected_shape}")
    crossing = np.any(np.diff(raw, axis=-1) < -1e-7, axis=-1)
    out_of_bounds = (raw < 0.0) | (raw > 1.05)
    values = np.sort(np.clip(raw, 0.0, 1.05), axis=-1).astype(np.float32)
    targets = np.empty((len(dataset), HORIZON), dtype=np.float32)
    labels = np.empty((len(dataset), HORIZON), dtype=np.int64)
    daylight = np.empty((len(dataset), HORIZON), dtype=bool)
    timestamps = np.empty((len(dataset), HORIZON), dtype=np.int64)
    day_ids = np.empty((len(dataset), HORIZON), dtype=np.int32)
    site_ids = np.empty(len(dataset), dtype=np.int64)
    state_probs = np.empty((len(dataset), HORIZON, 3), dtype=np.float32)
    lookup = _threshold_lookup(dataset)
    median_index = tuple(float(value) for value in levels).index(0.5)
    for start in range(0, len(dataset), batch_windows):
        stop = min(start + batch_windows, len(dataset))
        batch = read_window_batch(
            dataset,
            start,
            stop,
            (
                "history_values",
                "history_mask",
                "targets",
                "state_labels",
                "daylight_mask",
                "timestamps",
                "day_ids",
                "site_ids",
            ),
        )
        batch_daylight = np.asarray(batch["daylight_mask"], dtype=bool)
        batch_site_ids = np.asarray(batch["site_ids"], dtype=np.int64).reshape(-1)
        thresholds = np.empty(stop - start, dtype=np.float32)
        for item, site_id in enumerate(batch_site_ids):
            if int(site_id) not in lookup:
                raise ValueError(f"schema has no fitted ramp threshold for site {site_id}")
            thresholds[item] = lookup[int(site_id)]
        values[start:stop][~batch_daylight] = 0.0
        targets[start:stop] = np.asarray(batch["targets"], dtype=np.float32)
        labels[start:stop] = np.asarray(batch["state_labels"], dtype=np.int64)
        daylight[start:stop] = batch_daylight
        timestamps[start:stop] = np.asarray(batch["timestamps"], dtype=np.int64)
        day_ids[start:stop] = np.asarray(batch["day_ids"], dtype=np.int32)
        site_ids[start:stop] = batch_site_ids
        state_probs[start:stop] = ramp_state_probabilities_from_median(
            values[start:stop, :, median_index],
            np.asarray(batch["history_values"], dtype=np.float32)[..., 0],
            thresholds,
            batch_daylight,
            history_mask=np.asarray(batch["history_mask"], dtype=np.float32)[..., 0],
        )

    errors = targets[..., None] - values
    tau = np.asarray(levels, dtype=np.float32).reshape(1, 1, -1)
    pinball = np.maximum(tau * errors, (tau - 1.0) * errors)
    audit: dict[str, object] = {
        "quantile_crossing_rate_before_rearrangement": float(crossing.mean()),
        "out_of_physical_bounds_rate_before_clipping": float(out_of_bounds.mean()),
        "quantile_rearrangement": "ascending sort after official MQLoss prediction",
        "physical_clipping": [0.0, 1.05],
        "night_rule": "all quantiles forced to zero under the registered daylight mask",
        "mean_pinball_all": float(pinball.mean()),
        "mean_pinball_daylight": (
            float(pinball[daylight].mean()) if daylight.any() else float("nan")
        ),
        "payload_batch_windows": int(batch_windows),
        "state_probability_method": (
            "derived continuous threshold-margin softmax from the median trajectory; "
            "missing history is causally forward-filled and night is stable"
        ),
    }
    payload = {
        "quantiles": values,
        "state_probs": state_probs,
        "targets": targets,
        "state_labels": labels,
        "daylight_mask": daylight.astype(np.float32),
        "timestamps": timestamps,
        "day_ids": day_ids,
        "site_ids": site_ids,
    }
    return payload, audit


def _predict_split(
    neural_forecast: Any,
    model: Any,
    dataset: WindowDataset,
    *,
    split: str,
    levels: Sequence[float],
    future_plan: FutureExogenousPlan,
    chunk_windows: int,
) -> tuple[np.ndarray, dict[str, object]]:
    output = np.empty((len(dataset), HORIZON, len(levels)), dtype=np.float32)
    started = time.perf_counter()
    calls = 0
    for start in range(0, len(dataset), chunk_windows):
        stop = min(start + chunk_windows, len(dataset))
        frames = build_prediction_frames(
            dataset,
            split=split,
            start=start,
            stop=stop,
            future_plan=future_plan,
        )
        forecasts = neural_forecast.predict(
            df=frames.history,
            static_df=frames.static,
            futr_df=frames.future,
            verbose=False,
        )
        output[start:stop] = extract_official_quantiles(forecasts, model, frames, levels)
        calls += 1
    elapsed = time.perf_counter() - started
    return output, {
        "prediction_seconds": elapsed,
        "prediction_calls": calls,
        "prediction_batch_windows": chunk_windows,
        "windows_per_second": float(len(dataset) / elapsed) if elapsed > 0 else float("inf"),
    }


def _benchmark_official_inference(
    neural_forecast: Any,
    model: Any,
    dataset: WindowDataset,
    *,
    split: str,
    levels: Sequence[float],
    future_plan: FutureExogenousPlan,
    framework_internal_batch_size: int,
) -> dict[str, object]:
    """Measure one fixed validation batch through the public predict API."""

    batch_size = min(EFFICIENCY_BENCHMARK_BATCH_SIZE, len(dataset))
    if batch_size < 1:
        raise ValueError("validation data must contain at least one benchmark window")
    frames = build_prediction_frames(
        dataset,
        split=split,
        start=0,
        stop=batch_size,
        future_plan=future_plan,
    )

    def predict() -> np.ndarray:
        forecasts = neural_forecast.predict(
            df=frames.history,
            static_df=frames.static,
            futr_df=frames.future,
            verbose=False,
        )
        return extract_official_quantiles(forecasts, model, frames, levels)

    first_parameter = next(iter(model.parameters()), None)
    device = first_parameter.device if first_parameter is not None else torch.device("cpu")
    result = benchmark_callable(
        predict,
        batch_size=batch_size,
        device=device,
        warmup=EFFICIENCY_BENCHMARK_WARMUP,
        repetitions=EFFICIENCY_BENCHMARK_REPETITIONS,
        scope=(
            "official_neuralforecast_predict_on_prebuilt_registered_frames_"
            "excluding_frame_construction_payload_postprocessing_and_artifact_io"
        ),
    )
    result.update(
        {
            "framework_internal_windows_batch_size": int(framework_internal_batch_size),
            "prepared_history_rows": int(len(frames.history)),
            "prepared_future_rows": int(len(frames.future)),
        }
    )
    return result


def _prediction_metadata(
    *,
    args: argparse.Namespace,
    split: str,
    dataset: WindowDataset,
    quantiles: Sequence[float],
    model_metadata: Mapping[str, object],
    model_state_sha256: str,
    future_plan: FutureExogenousPlan,
    descriptor_hashes: Mapping[str, str],
    content_hashes: Mapping[str, str],
    resolved_run_config_sha256: str,
    audit: Mapping[str, object],
) -> dict[str, object]:
    if dataset.schema is None:
        raise RuntimeError("registered schema unexpectedly missing")
    return {
        "baseline": f"official_neuralforecast_{args.name}",
        "implementation": model_metadata,
        "split": split,
        "seed": args.seed,
        "quantiles": [float(value) for value in quantiles],
        "resolved_run_config_sha256": resolved_run_config_sha256,
        # Kept for evaluator compatibility: this is the NPZ/index/store descriptor,
        # while the next identity covers all underlying shard files.
        "source_npz_sha256": descriptor_hashes[split],
        "source_window_content_sha256": content_hashes[split],
        "train_npz_sha256": descriptor_hashes["train"],
        "train_window_content_sha256": content_hashes["train"],
        "model_state_sha256": model_state_sha256,
        "site_id_map": dataset.schema["site_id_map"],
        "dataset_kind": dataset.schema["dataset_kind"],
        "simulated": dataset.schema["simulated"],
        "ramp_method": dataset.schema["ramp_method"],
        "registered_evidence_eligible": dataset.schema["registered_evidence_eligible"],
        "development_data_override": args.allow_development_data,
        "validation_only_screening": args.validation_only,
        "calibration_or_test_loaded": not args.validation_only,
        "ramp_threshold_source": "prepared train-only per-site schema",
        "ramp_thresholds": dataset.schema["ramp_thresholds"],
        "state_probabilities": {
            "type": "derived_continuous_threshold_margin_softmax",
            "trained_state_classifier": False,
            "history_missingness": "causal_forward_fill_then_leading_zero",
            "night_rule": "stable_probability_one",
        },
        "future_information_contract": {
            "source_features": list(future_plan.source_features),
            "neuralforecast_columns": list(future_plan.columns),
            "storage_mode": future_plan.storage_mode,
            "omitted_features": dict(future_plan.omitted_features),
            "future_dataframe_contains_target": False,
            "future_measured_weather_used": False,
        },
        "postprocessing": dict(audit),
        "parameter_update_on_this_split": False,
    }


def main() -> None:
    args = build_parser().parse_args()
    from racepv.cli.modern_baseline import _preflight_prepared_inputs

    _preflight_prepared_inputs(args, continuous_panel=True)
    _validate_cli(args)
    config = load_config(args.config)
    seed_everything(args.seed)
    paths = _resolve_input_paths(args)
    prediction_splits = tuple(split for split in paths if split != "train")
    datasets = {
        split: load_window_dataset(path, verify_content=True) for split, path in paths.items()
    }
    for dataset in datasets.values():
        dataset.require_registered_evidence(allow_ineligible=args.allow_development_data)
    global_temporal_audit = require_global_split_chronology(datasets)
    registered_spatial_protocol = require_registered_spatial_protocol(datasets)
    roles = _enforce_fit_roles(datasets)
    descriptor_hashes = {split: _descriptor_sha256(path) for split, path in paths.items()}
    content_hashes = {
        split: window_dataset_content_identity(path, verify=False) for split, path in paths.items()
    }
    future_plan = resolve_future_exogenous_plan(
        datasets, allowed_future_features=config.data.allowed_future_features
    )
    archive_audit = validate_archive_contract(
        datasets,
        allowed_future_features=config.data.allowed_future_features,
        future_plan=future_plan,
        validation_only=args.validation_only,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    # The immutable configuration deliberately excludes the random seed so the
    # selector can verify that all five seed replicates use one model protocol.
    # Seed identity remains explicit in every prediction sidecar and run.json.
    resolved_run_config = {
        "schema_version": 1,
        "baseline": f"official_neuralforecast_{args.name}",
        "purpose": (
            "training_validation_only_official_baseline_screening"
            if args.validation_only
            else "registered_four_split_official_baseline"
        ),
        "prediction_splits": list(prediction_splits),
        "experiment_config_sha256": sha256_file(args.config),
        "optimization": {
            "unit": "optimizer_steps",
            "max_steps": args.max_steps,
            "val_check_steps": args.val_check_steps,
            "learning_rate": config.training.learning_rate,
            "series_batch_size": args.series_batch_size,
            "windows_batch_size": args.windows_batch_size,
            "inference_windows_batch_size": args.inference_windows_batch_size,
            "early_stop_patience_steps": -1,
        },
        "panel": {
            "origin_stride_steps": args.origin_stride_steps,
            "max_panel_rows": args.max_panel_rows,
        },
        "prediction": {
            "prediction_batch_windows": args.prediction_batch_windows,
            "payload_batch_windows": args.payload_batch_windows,
            "max_prediction_context_rows": args.max_prediction_context_rows,
        },
        "efficiency_benchmark": {
            "batch_size_windows": EFFICIENCY_BENCHMARK_BATCH_SIZE,
            "warmup_repetitions": EFFICIENCY_BENCHMARK_WARMUP,
            "timed_repetitions": EFFICIENCY_BENCHMARK_REPETITIONS,
            "population": "first validation windows in registered order",
        },
        "device_request": args.device,
        "allow_development_data": args.allow_development_data,
        "quantiles": list(config.quantiles),
        "history_length": LOOKBACK,
        "horizon": HORIZON,
        "historical_exogenous": list(HISTORICAL_EXOGENOUS),
        "future_exogenous": list(future_plan.columns),
        "future_source_features": list(future_plan.source_features),
        "future_storage_mode": future_plan.storage_mode,
        "omitted_future_features": dict(future_plan.omitted_features),
        "static_exogenous": list(STATIC_EXOGENOUS),
        "input_descriptor_sha256": descriptor_hashes,
        "input_window_content_sha256": content_hashes,
    }
    resolved_run_config_path = args.output / "resolved_run_config.json"
    atomic_json_dump(resolved_run_config, resolved_run_config_path)
    resolved_run_config_sha256 = sha256_file(resolved_run_config_path)
    panel = build_training_panel(
        datasets["train"],
        future_plan=future_plan,
        origin_stride_steps=args.origin_stride_steps,
        max_panel_rows=args.max_panel_rows,
    )

    save_config(config, args.output / "resolved_config.yaml")
    NeuralForecast = load_neuralforecast_class()
    trainer_kwargs: dict[str, object] = {
        "learning_rate": config.training.learning_rate,
        "batch_size": args.series_batch_size,
        "valid_batch_size": args.series_batch_size,
        "windows_batch_size": args.windows_batch_size,
        "inference_windows_batch_size": args.inference_windows_batch_size,
        "step_size": int(panel.audit["origin_stride_steps"]),
        "scaler_type": "identity",
        "start_padding_enabled": False,
        # Requiring a complete target horizon prevents padded segment tails from
        # becoming additional, unregistered training origins.
        "training_data_availability_threshold": (0.0, 1.0),
        "drop_last_loader": False,
        "enable_checkpointing": False,
        # N-HiTS uses CUDA linear-interpolation backward kernels that currently
        # have no deterministic implementation. Preserve the warning while
        # keeping the registered multi-seed run executable, matching
        # seed_everything(..., warn_only=True).
        "deterministic": "warn",
        "gradient_clip_val": config.training.gradient_clip_norm,
        "optimizer": torch.optim.Adam,
        "optimizer_kwargs": {"weight_decay": config.training.weight_decay},
        "logger": False,
        "enable_progress_bar": not args.quiet,
        **_trainer_device_kwargs(args.device),
    }
    model = build_neuralforecast_model(
        args.name,
        quantiles=config.quantiles,
        historical_exogenous=HISTORICAL_EXOGENOUS,
        future_exogenous=future_plan.columns,
        static_exogenous=STATIC_EXOGENOUS,
        max_steps=args.max_steps,
        val_check_steps=args.val_check_steps,
        # Independent 48-hour validation contexts cannot be mapped to NF's
        # contiguous val_size tail without changing their registered context.
        early_stop_patience_steps=-1,
        random_seed=args.seed,
        **trainer_kwargs,
    )
    neural_forecast = NeuralForecast(models=[model], freq=FREQUENCY)
    fit_started = time.perf_counter()
    neural_forecast.fit(
        df=panel.frame,
        static_df=panel.static_frame,
        val_size=0,
        verbose=not args.quiet,
    )
    fit_seconds = time.perf_counter() - fit_started
    fitted_model = neural_forecast.models[0]
    model_metadata = official_model_metadata(fitted_model)
    if model_metadata["early_stop_patience_steps"] != -1:
        raise RuntimeError("official runner unexpectedly enabled internal early stopping")
    state_before_predictions = _state_dict_sha256(fitted_model)

    model_dir = args.output / "neuralforecast_model"
    neural_forecast.save(
        path=str(model_dir),
        save_dataset=False,
        overwrite=True,
    )
    model_artifacts = _artifact_manifest(model_dir)

    # Explicit predict dataframes make the fitted training tensor unnecessary.
    # Retaining ID/date metadata is cheap and avoids relying on private predict
    # implementation details in future NeuralForecast 3.x patch releases.
    if hasattr(neural_forecast, "dataset"):
        delattr(neural_forecast, "dataset")
    panel_audit = panel.audit
    del panel
    gc.collect()

    split_reports: dict[str, object] = {}
    for split in prediction_splits:
        raw, timing = _predict_split(
            neural_forecast,
            fitted_model,
            datasets[split],
            split=split,
            levels=config.quantiles,
            future_plan=future_plan,
            chunk_windows=args.prediction_batch_windows,
        )
        payload, prediction_audit = _prediction_payload(
            datasets[split],
            raw,
            config.quantiles,
            batch_windows=args.payload_batch_windows,
        )
        metadata = _prediction_metadata(
            args=args,
            split=split,
            dataset=datasets[split],
            quantiles=config.quantiles,
            model_metadata=model_metadata,
            model_state_sha256=state_before_predictions,
            future_plan=future_plan,
            descriptor_hashes=descriptor_hashes,
            content_hashes=content_hashes,
            resolved_run_config_sha256=resolved_run_config_sha256,
            audit=prediction_audit,
        )
        save_predictions_npz(
            args.output / f"{split}_predictions.npz",
            payload,
            metadata=metadata,
        )
        split_reports[split] = {**timing, **prediction_audit}

    batch_inference = _benchmark_official_inference(
        neural_forecast,
        fitted_model,
        datasets["validation"],
        split="validation",
        levels=config.quantiles,
        future_plan=future_plan,
        framework_internal_batch_size=args.inference_windows_batch_size,
    )
    state_after_predictions = _state_dict_sha256(fitted_model)
    if state_after_predictions != state_before_predictions:
        raise RuntimeError("official predict calls changed model parameters")

    trainable_parameters = int(
        sum(parameter.numel() for parameter in fitted_model.parameters() if parameter.requires_grad)
    )
    trajectories = list(getattr(fitted_model, "train_trajectories", ()) or ())
    observed_steps = max((int(step) for step, _ in trajectories), default=None)
    model_artifact_bytes = sum(int(item["bytes"]) for item in model_artifacts)
    run = {
        "baseline": f"official_neuralforecast_{args.name}",
        "seed": args.seed,
        "purpose": (
            "training_validation_only_official_baseline_screening"
            if args.validation_only
            else "registered_four_split_official_baseline"
        ),
        "prediction_splits": list(prediction_splits),
        "calibration_or_test_loaded": not args.validation_only,
        "resolved_run_config_sha256": resolved_run_config_sha256,
        "implementation": model_metadata,
        "trainable_parameters": trainable_parameters,
        "efficiency": efficiency_audit_record(
            trainable_parameters=trainable_parameters,
            trainable_parameters_unavailable_reason=None,
            model_artifact_bytes=model_artifact_bytes,
            model_artifact_definition=(
                "sum of all files written by NeuralForecast.save in model_directory"
            ),
            batch_inference=batch_inference,
        ),
        "fit_seconds": fit_seconds,
        "observed_last_logged_optimizer_step": observed_steps,
        "trainer_metrics": _json_metrics(fitted_model),
        "optimization_budget": {
            "framework_unit": "optimizer_steps",
            "max_steps": args.max_steps,
            "val_check_steps": args.val_check_steps,
            "optimizer": "torch.optim.Adam",
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "is_equivalent_to_80_epochs": False,
            "registered_plan_nominal_max_epochs": config.training.max_epochs,
            "protocol_deviation": (
                "NeuralForecast 3.x exposes max_steps and rejects max_epochs; this runner "
                "does not relabel optimizer steps as epochs"
            ),
        },
        "validation_selection": {
            "validation_predictions_saved": True,
            "validation_usage": "post-fit diagnostic and external baseline preselection",
            "used_by_fit": False,
            "early_stopping_enabled": False,
            "reason": (
                "mapping independent pre-windowed validation contexts to NeuralForecast's "
                "contiguous val_size/val_df tail would change the registered 48-hour context"
            ),
        },
        "spatial_roles_by_split": roles,
        "information_boundary": {
            "historical_exogenous": list(HISTORICAL_EXOGENOUS),
            "future_source_features": list(future_plan.source_features),
            "future_exogenous": list(future_plan.columns),
            "future_storage_mode": future_plan.storage_mode,
            "omitted_registered_future_features": dict(future_plan.omitted_features),
            "static_exogenous": list(STATIC_EXOGENOUS),
            "future_measured_weather_used": False,
        },
        "training_panel": panel_audit,
        "archive_audit": archive_audit,
        "global_temporal_audit": global_temporal_audit,
        "registered_spatial_protocol": registered_spatial_protocol,
        "evidence_mode": {
            "allow_development_data": args.allow_development_data,
            "dataset_kind": datasets["train"].schema["dataset_kind"],
            "simulated": datasets["train"].schema["simulated"],
            "registered_evidence_eligible": datasets["train"].schema[
                "registered_evidence_eligible"
            ],
        },
        "prediction_reports": split_reports,
        "parameter_state_audit": {
            "before_predictions_sha256": state_before_predictions,
            "after_predictions_sha256": state_after_predictions,
            "unchanged": True,
        },
        "resource_guards": {
            "max_panel_rows": args.max_panel_rows,
            "estimated_panel_rows": panel_audit["deduplicated_rows"],
            "guard_checked_before_dataframe_construction": True,
            "prediction_batch_windows": args.prediction_batch_windows,
            "payload_batch_windows": args.payload_batch_windows,
            "max_prediction_context_rows": args.max_prediction_context_rows,
            "actual_prediction_context_rows_per_full_chunk": (
                args.prediction_batch_windows * LOOKBACK
            ),
            "full_100_site_scalability_claimed": False,
        },
        "artifacts": {
            "resolved_run_config": {
                "path": str(resolved_run_config_path),
                "sha256": resolved_run_config_sha256,
            },
            "inputs": _input_artifacts(
                args,
                descriptor_hashes=descriptor_hashes,
                content_hashes=content_hashes,
            ),
            "model_directory": str(model_dir),
            "model_files": model_artifacts,
        },
        "environment": environment_report(),
    }
    atomic_json_dump(run, args.output / "run.json")
    print(f"Completed official NeuralForecast {args.name}; outputs are in {args.output}")


if __name__ == "__main__":
    main()
