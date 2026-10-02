"""Train RaCE-PV on prepared chronological window indexes or legacy NPZ archives."""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from racepv.audit import sha256_file
from racepv.config import load_config, save_config
from racepv.data import (
    FORECAST_WEATHER_FEATURES,
    IssuedWeatherStore,
    WeatherAugmentedDataset,
)
from racepv.diagnostics import SPECIALIST_DIAGNOSTIC_WEIGHTS
from racepv.environment import environment_report
from racepv.io import (
    WindowDataset,
    load_window_dataset,
    require_global_split_chronology,
    require_registered_spatial_protocol,
    save_predictions_npz,
    window_dataset_temporal_bounds,
)
from racepv.losses import RaCEPVLoss
from racepv.models import RaCEPV, RaCEPVConfig
from racepv.models.config import (
    FORECAST_SELECTION_METRIC,
    REGISTERED_SPECIALIST_LOSS_WEIGHT,
)
from racepv.reliability_experiment import (
    PAIR_PROTOCOL,
    require_reliability_recovery_binding,
    resolve_reliability_request,
)
from racepv.reliability_experiment import (
    make_reliability_components as _build_reliability_components,
)
from racepv.training import (
    EFFICIENCY_BENCHMARK_BATCH_SIZE,
    EFFICIENCY_BENCHMARK_REPETITIONS,
    EFFICIENCY_BENCHMARK_WARMUP,
    FROZEN_WEATHER_ADAPTER_SCHEMA,
    EarlyStoppingTrainer,
    benchmark_inference,
    build_window_loader,
    chronological_development_split,
    count_parameters,
    efficiency_audit_record,
    initialize_frozen_weather_adapter,
)
from racepv.utils import atomic_json_dump, seed_everything, select_device

MOE_AUDIT_ARRAYS = frozenset(
    {
        "expert_quantiles",
        "gate_weights",
        "fusion_logits",
        "state_logits",
        "router_logits",
        "ramp_presence_logits",
        "ramp_direction_logits",
    }
)

ADJUSTMENT_CELLS = {
    "F00": ("flat", 0.0),
    "F10": ("hierarchical", 0.0),
    "F01": ("flat", 0.05),
    "F11": ("hierarchical", 0.05),
}
SPECIALIST_RAMP_CELLS = {
    "S00": 0.0,
    "S02": 0.02,
    "S05": 0.05,
}
RAMP_REFINEMENT_CELLS = {
    "G00": ("oracle_gate_regret", 0.0),
    "G02": ("oracle_gate_regret", 0.02),
    "T00": ("direction_expert_tail_mse", 0.0),
    "T02": ("direction_expert_tail_mse", 0.02),
    "I00": ("isolated_oracle_gate_regret", 0.0),
    "I02": ("isolated_oracle_gate_regret", 0.02),
}
INDEPENDENT_STATE_CELLS = {"D00": 0.0, "D02": 0.02}
STATE_BOUNDARY_CELLS = {"E00": 1.0, "E20": 2.0}
TAIL_RESIDUAL_CELLS = {
    "R00": ("off", 0.0, 0.0),
    "R02": ("sparse_ramp_up", 0.02, 0.001),
}
REGISTERED_TAIL_TRAIN_SITES = frozenset(
    {
        "dkasc_07_first_solar_fixed",
        "dkasc_03_bp_solar_roof",
        "dkasc_01a_trina_dual",
        "dkasc_02_ecokinetics_dual",
    }
)
REGISTERED_TAIL_VALIDATION_SITES = frozenset(
    {
        "dkasc_04_kyocera_dual",
        "dkasc_05_kyocera_single",
        "dkasc_06_kyocera_dual",
        "dkasc_08_kaneka_fixed",
    }
)
RAMP_AUXILIARY_BETA = 0.1


def _benchmark_model_inputs(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Select the same atomic model inputs used by training and prediction."""

    selected = {
        name: batch[name]
        for name in (
            "history_values",
            "history_mask",
            "future_known",
            "site_metadata",
        )
    }
    weather_keys = ("future_weather", "future_weather_mask")
    weather_present = tuple(name in batch for name in weather_keys)
    if any(weather_present) and not all(weather_present):
        raise KeyError("future weather values and mask must be provided as an atomic pair")
    if all(weather_present):
        selected.update({name: batch[name] for name in weather_keys})
    return selected


def _required_future_feature_indices(columns: tuple[str, ...]) -> dict[str, int]:
    """Resolve v2 physical features by audited schema names, never positions."""

    required = {
        "future_clear_sky_index": "clear_sky_power_norm",
        "future_hour_sin_index": "hour_sin",
        "future_hour_cos_index": "hour_cos",
    }
    missing = [feature for feature in required.values() if feature not in columns]
    if missing:
        raise ValueError(f"v2 future-known schema is missing required features: {missing}")
    return {name: columns.index(feature) for name, feature in required.items()}


def _v2_state_loss_weight(configured: float, override: float | None) -> float:
    """Reject legacy state-loss coefficients instead of silently reinterpreting them."""

    selected = float(configured if override is None else override)
    if selected != 1.0:
        raise ValueError(
            "RaCE-PV fixes the state cross-entropy coefficient at 1.0; "
            "legacy state-loss-weight ablations are incompatible with "
            "architecture_version=2"
        )
    return selected


def _v2_specialist_loss_weight(configured: float, override: float | None) -> float:
    """Keep the registered specialist coefficient fixed at its frozen value."""

    selected = float(configured if override is None else override)
    if abs(selected - REGISTERED_SPECIALIST_LOSS_WEIGHT) > 1e-12:
        raise ValueError(
            "RaCE-PV fixes the balanced specialist qCRPS coefficient at "
            f"{REGISTERED_SPECIALIST_LOSS_WEIGHT:g}; "
            "other specialist-loss weights are incompatible with "
            "architecture_version=2"
        )
    return selected


def _resolve_input_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve the train input boundary before any output directory is created."""

    if args.validation_only:
        forbidden = {
            name: path
            for name, path in (("calibration", args.calibration), ("test", args.test))
            if path is not None
        }
        if forbidden:
            raise ValueError(
                "validation-only training forbids calibration/test inputs; "
                f"received {sorted(forbidden)}"
            )
        return {"train": args.train, "validation": args.validation}
    return {
        name: path
        for name, path in (
            ("train", args.train),
            ("validation", args.validation),
            ("calibration", args.calibration),
            ("test", args.test),
        )
        if path is not None
    }


def _resolve_prediction_state_coupling(
    args: argparse.Namespace,
) -> dict[str, object]:
    """Validate and describe the sealed prediction/state coupling ablation."""

    requested_mode = getattr(args, "prediction_state_coupling", None)
    if requested_mode is None:
        return {
            "prediction_state_coupling_requested": False,
            "prediction_state_coupling": "decoupled",
            "diagnostic_only": False,
            "diagnostic_ablation": None,
            "selection_boundary": (
                "validation_only"
                if bool(getattr(args, "validation_only", False))
                else "registered_split_protocol"
            ),
        }
    if not bool(getattr(args, "validation_only", False)):
        raise ValueError(
            "prediction-state coupling ablation is only available with --validation-only"
        )
    confounded = {
        name: getattr(args, name, None)
        for name in (
            "diagnostic_ablation",
            "adjustment_cell",
            "specialist_ramp_cell",
            "ramp_refinement_cell",
            "independent_state_cell",
            "state_boundary_cell",
            "tail_residual_cell",
            "weather_store",
            "frozen_weather_adapter_from",
            "internal_development_fraction",
        )
        if getattr(args, name, None) is not None
    }
    if confounded:
        raise ValueError(
            "prediction-state coupling ablation cannot be combined with other treatments: "
            f"{sorted(confounded)}"
        )
    if getattr(args, "forecast_weather_mode", "off") != "off":
        raise ValueError(
            "prediction-state coupling ablation cannot be combined with forecast weather"
        )
    if bool(getattr(args, "allow_development_data", False)):
        raise ValueError("prediction-state coupling ablation requires registered evidence data")
    if args.router_mode != "contextual" or args.fusion_mode != "learned":
        raise ValueError("prediction-state coupling ablation requires contextual learned fusion")
    if tuple(args.active_experts) != ("stable", "ramp_up", "ramp_down"):
        raise ValueError("prediction-state coupling ablation requires all three experts")
    if args.no_clear_sky_prior:
        raise ValueError("prediction-state coupling ablation requires the clear-sky prior")
    return {
        "prediction_state_coupling_requested": True,
        "prediction_state_coupling": str(requested_mode),
        "diagnostic_only": True,
        "diagnostic_ablation": "prediction_state_coupling",
        "selection_boundary": "validation_only",
    }


def _resolve_weather_protocol(args: argparse.Namespace) -> dict[str, object]:
    """Validate the sealed validation-only weather experiment boundary."""

    mode = str(getattr(args, "forecast_weather_mode", "off"))
    store = getattr(args, "weather_store", None)
    if mode not in {"off", "experts", "experts_fusion", "transient_isolated"}:
        raise ValueError("unknown forecast weather mode")
    if mode != "off" and store is None:
        raise ValueError(f"forecast weather mode {mode!r} requires --weather-store")
    weather_experiment = store is not None
    if weather_experiment and not bool(args.validation_only):
        raise ValueError("forecast weather experiments are only available with --validation-only")
    protocol: dict[str, object] = {
        "forecast_weather_mode": mode,
        "weather_store": store,
        "weather_experiment": weather_experiment,
        "diagnostic_only": weather_experiment,
        "selection_boundary": (
            "validation_only" if args.validation_only else "registered_split_protocol"
        ),
        "state_head_weather_access": False,
    }
    if mode == "transient_isolated":
        protocol.update(
            {
                "weather_direct_stable_expert_access": False,
                "weather_direct_transient_expert_access": True,
                "weather_direct_fusion_access": False,
                "weather_indirect_fusion_access": False,
                "fusion_representation_semantics": ("weather_free_same_run_expert_representations"),
            }
        )
    return protocol


def _resolve_frozen_weather_adapter_request(
    args: argparse.Namespace,
) -> dict[str, object]:
    """Validate the explicit W00-initialized adapter-only training boundary."""

    source = getattr(args, "frozen_weather_adapter_from", None)
    if source is None:
        return {
            "enabled": False,
            "training_protocol": "full_model_v2",
            "source_checkpoint": None,
            "state_optimizer_enabled": True,
            "state_loss_gradient_target": "ramp_state_head",
        }
    if not bool(getattr(args, "validation_only", False)):
        raise ValueError("frozen weather adapter requires --validation-only")
    if getattr(args, "forecast_weather_mode", "off") != "transient_isolated":
        raise ValueError("frozen weather adapter requires transient_isolated mode")
    if getattr(args, "weather_store", None) is None:
        raise ValueError("frozen weather adapter requires --weather-store")
    conflicting = {
        name: getattr(args, name, None)
        for name in (
            "diagnostic_ablation",
            "adjustment_cell",
            "specialist_ramp_cell",
            "ramp_refinement_cell",
            "independent_state_cell",
            "state_boundary_cell",
            "tail_residual_cell",
            "internal_development_fraction",
        )
        if getattr(args, name, None) is not None
    }
    if conflicting:
        raise ValueError(
            "frozen weather adapter cannot be combined with other treatments: "
            f"{sorted(conflicting)}"
        )
    return {
        "enabled": True,
        "training_protocol": FROZEN_WEATHER_ADAPTER_SCHEMA,
        "source_checkpoint": Path(source),
        "state_optimizer_enabled": False,
        "state_loss_gradient_target": "none_frozen",
    }


def _serializable_weather_protocol(
    protocol: Mapping[str, object],
    *,
    weather_store_sha256: str | None,
    weather_feature_names: tuple[str, ...],
) -> dict[str, object]:
    """Build one JSON-safe weather provenance payload for every run artifact."""

    store = protocol.get("weather_store")
    return {
        **protocol,
        "weather_store": str(store) if store is not None else None,
        "weather_store_sha256": weather_store_sha256,
        "weather_feature_names": list(weather_feature_names),
    }


def _resolve_specialist_objective(
    *,
    configured: float,
    override: float | None,
    diagnostic_ablation: str | None,
    validation_only: bool,
) -> tuple[float, dict[str, object]]:
    """Keep the main objective fixed while allowing one labelled diagnostic."""

    if diagnostic_ablation not in {None, *SPECIALIST_DIAGNOSTIC_WEIGHTS}:
        raise ValueError(f"unsupported diagnostic ablation: {diagnostic_ablation!r}")
    if diagnostic_ablation is None:
        registered_weight = _v2_specialist_loss_weight(configured, override)
        return registered_weight, {
            "diagnostic_only": False,
            "diagnostic_ablation": None,
        }
    if not validation_only:
        raise ValueError(
            f"--diagnostic-ablation {diagnostic_ablation} is only available with --validation-only"
        )
    if override is not None:
        raise ValueError("named specialist diagnostics must not also pass --specialist-loss-weight")
    return SPECIALIST_DIAGNOSTIC_WEIGHTS[diagnostic_ablation], {
        "diagnostic_only": True,
        "diagnostic_ablation": diagnostic_ablation,
    }


def _resolve_adjustment_cell(
    adjustment_cell: str | None,
    *,
    validation_only: bool,
) -> dict[str, object]:
    """Resolve the frozen one-day factorial treatments without free-form tuning."""

    if adjustment_cell is None:
        return {
            "adjustment_cell": None,
            "state_head_mode": "flat",
            "ramp_auxiliary_weight": 0.0,
            "ramp_auxiliary_beta": RAMP_AUXILIARY_BETA,
            "diagnostic_only": False,
        }
    if adjustment_cell not in ADJUSTMENT_CELLS:
        raise ValueError(f"unsupported adjustment cell: {adjustment_cell!r}")
    if not validation_only:
        raise ValueError("adjustment cells are only available with --validation-only")
    state_head_mode, ramp_auxiliary_weight = ADJUSTMENT_CELLS[adjustment_cell]
    return {
        "adjustment_cell": adjustment_cell,
        "state_head_mode": state_head_mode,
        "ramp_auxiliary_weight": ramp_auxiliary_weight,
        "ramp_auxiliary_beta": RAMP_AUXILIARY_BETA,
        "diagnostic_only": True,
    }


def _resolve_specialist_ramp_cell(
    specialist_ramp_cell: str | None,
    *,
    validation_only: bool,
) -> dict[str, object]:
    """Resolve the bounded direction-expert ramp auxiliary treatments."""

    if specialist_ramp_cell is None:
        return {
            "specialist_ramp_cell": None,
            "state_head_mode": "flat",
            "ramp_auxiliary_weight": 0.0,
            "ramp_auxiliary_beta": RAMP_AUXILIARY_BETA,
            "ramp_auxiliary_scope": "direction_experts",
            "diagnostic_only": False,
        }
    if specialist_ramp_cell not in SPECIALIST_RAMP_CELLS:
        raise ValueError(f"unsupported specialist ramp cell: {specialist_ramp_cell!r}")
    if not validation_only:
        raise ValueError("specialist ramp cells are only available with --validation-only")
    return {
        "specialist_ramp_cell": specialist_ramp_cell,
        "state_head_mode": "flat",
        "ramp_auxiliary_weight": SPECIALIST_RAMP_CELLS[specialist_ramp_cell],
        "ramp_auxiliary_beta": RAMP_AUXILIARY_BETA,
        "ramp_auxiliary_scope": "direction_experts",
        "diagnostic_only": True,
    }


def _resolve_ramp_refinement_cell(
    ramp_refinement_cell: str | None,
    *,
    validation_only: bool,
) -> dict[str, object]:
    """Resolve one bounded mechanism chosen by the archive diagnostics."""

    if ramp_refinement_cell is None:
        return {
            "ramp_refinement_cell": None,
            "state_head_mode": "flat",
            "ramp_auxiliary_weight": 0.0,
            "ramp_auxiliary_beta": RAMP_AUXILIARY_BETA,
            "ramp_auxiliary_scope": "oracle_gate_regret",
            "diagnostic_only": False,
        }
    if ramp_refinement_cell not in RAMP_REFINEMENT_CELLS:
        raise ValueError(f"unsupported ramp refinement cell: {ramp_refinement_cell!r}")
    if not validation_only:
        raise ValueError("ramp refinement cells are only available with --validation-only")
    scope, weight = RAMP_REFINEMENT_CELLS[ramp_refinement_cell]
    return {
        "ramp_refinement_cell": ramp_refinement_cell,
        "state_head_mode": "flat",
        "ramp_auxiliary_weight": weight,
        "ramp_auxiliary_beta": RAMP_AUXILIARY_BETA,
        "ramp_auxiliary_scope": scope,
        "diagnostic_only": True,
    }


def _resolve_independent_state_cell(
    independent_state_cell: str | None,
    *,
    validation_only: bool,
) -> dict[str, object]:
    """Resolve the sealed raw-input state-head pair without free-form tuning."""

    if independent_state_cell is None:
        return {
            "independent_state_cell": None,
            "state_head_mode": "flat",
            "state_representation_mode": "expert_detached",
            "enable_auxiliary_gate_path": False,
            "ramp_auxiliary_weight": 0.0,
            "ramp_auxiliary_beta": RAMP_AUXILIARY_BETA,
            "ramp_auxiliary_scope": "fusion_median",
            "diagnostic_only": False,
        }
    if independent_state_cell not in INDEPENDENT_STATE_CELLS:
        raise ValueError(f"unsupported independent state cell: {independent_state_cell!r}")
    if not validation_only:
        raise ValueError("independent state cells are only available with --validation-only")
    weight = INDEPENDENT_STATE_CELLS[independent_state_cell]
    return {
        "independent_state_cell": independent_state_cell,
        "state_head_mode": "flat",
        "state_representation_mode": "independent_raw",
        "enable_auxiliary_gate_path": weight > 0.0,
        "ramp_auxiliary_weight": weight,
        "ramp_auxiliary_beta": RAMP_AUXILIARY_BETA,
        "ramp_auxiliary_scope": "isolated_oracle_gate_regret",
        "diagnostic_only": True,
    }


def _resolve_state_boundary_cell(
    state_boundary_cell: str | None,
    *,
    validation_only: bool,
) -> dict[str, object]:
    """Resolve the sealed flat-head event-boundary weighting pair."""

    if state_boundary_cell is None:
        return {
            "state_boundary_cell": None,
            "state_boundary_weight": 1.0,
            "diagnostic_only": False,
        }
    if state_boundary_cell not in STATE_BOUNDARY_CELLS:
        raise ValueError(f"unsupported state boundary cell: {state_boundary_cell!r}")
    if not validation_only:
        raise ValueError("state boundary cells are only available with --validation-only")
    return {
        "state_boundary_cell": state_boundary_cell,
        "state_boundary_weight": STATE_BOUNDARY_CELLS[state_boundary_cell],
        "diagnostic_only": True,
    }


def _resolve_tail_residual_cell(
    tail_residual_cell: str | None,
    *,
    validation_only: bool,
    internal_development_fraction: float | None,
) -> dict[str, object]:
    """Resolve the sealed sparse tail pair without exposing free weights."""

    if tail_residual_cell is None:
        return {
            "tail_residual_cell": None,
            "tail_residual_mode": "off",
            "tail_auxiliary_weight": 0.0,
            "tail_sparsity_weight": 0.0,
            "diagnostic_only": False,
        }
    if tail_residual_cell not in TAIL_RESIDUAL_CELLS:
        raise ValueError(f"unsupported tail residual cell: {tail_residual_cell!r}")
    if not validation_only:
        raise ValueError("tail residual cells are only available with --validation-only")
    if (
        internal_development_fraction is None
        or abs(float(internal_development_fraction) - 0.10) > 1e-12
    ):
        raise ValueError("tail residual cells require internal development fraction 0.10")
    mode, auxiliary_weight, sparsity_weight = TAIL_RESIDUAL_CELLS[tail_residual_cell]
    return {
        "tail_residual_cell": tail_residual_cell,
        "tail_residual_mode": mode,
        "tail_auxiliary_weight": auxiliary_weight,
        "tail_sparsity_weight": sparsity_weight,
        "diagnostic_only": True,
    }


def _validate_tail_residual_request(args: argparse.Namespace) -> None:
    """Fail closed on any request that would confound R00/R02."""

    cell = getattr(args, "tail_residual_cell", None)
    if cell is None:
        return
    _resolve_tail_residual_cell(
        str(cell),
        validation_only=bool(getattr(args, "validation_only", False)),
        internal_development_fraction=getattr(
            args,
            "internal_development_fraction",
            None,
        ),
    )
    conflicting = {
        name: getattr(args, name, None)
        for name in (
            "calibration",
            "test",
            "weather_store",
            "frozen_weather_adapter_from",
            "prediction_state_coupling",
            "diagnostic_ablation",
            "adjustment_cell",
            "specialist_ramp_cell",
            "ramp_refinement_cell",
            "independent_state_cell",
            "state_boundary_cell",
            "state_loss_weight",
            "specialist_loss_weight",
        )
        if getattr(args, name, None) is not None
    }
    if conflicting:
        raise ValueError(
            "tail residual treatment cannot be combined with other treatments or "
            f"held-out/free-loss inputs: {sorted(conflicting)}"
        )
    if getattr(args, "forecast_weather_mode", "off") != "off":
        raise ValueError("tail residual treatment cannot use forecast weather")
    if bool(getattr(args, "allow_development_data", False)):
        raise ValueError("tail residual treatment requires registered evidence data")
    if (
        getattr(args, "router_mode", "contextual") != "contextual"
        or getattr(
            args,
            "fusion_mode",
            "learned",
        )
        != "learned"
    ):
        raise ValueError("tail residual treatment requires contextual learned fusion")
    if tuple(getattr(args, "active_experts", ())) != (
        "stable",
        "ramp_up",
        "ramp_down",
    ):
        raise ValueError("tail residual treatment requires all three experts")
    if bool(getattr(args, "no_clear_sky_prior", False)):
        raise ValueError("tail residual treatment requires the clear-sky prior")


def _validate_sparse_tail_dataset_boundary(
    *,
    train_sites: set[str] | frozenset[str],
    validation_sites: set[str] | frozenset[str],
) -> None:
    """Require the frozen four training and four unseen validation devices."""

    observed_train = frozenset(str(site) for site in train_sites)
    observed_validation = frozenset(str(site) for site in validation_sites)
    if (
        observed_train != REGISTERED_TAIL_TRAIN_SITES
        or observed_validation != REGISTERED_TAIL_VALIDATION_SITES
        or not observed_train.isdisjoint(observed_validation)
    ):
        raise ValueError(
            "tail residual new-site validation requires the exact registered, "
            "disjoint train and validation site sets"
        )


def _audit_sparse_tail_spatial_boundary(
    datasets: Mapping[str, WindowDataset],
) -> dict[str, object]:
    """Audit calendar overlap while preserving an explicitly spatial claim."""

    if set(datasets) != {"train", "validation"}:
        raise ValueError("spatial validation audit requires only train and validation")
    bounds = {
        split: window_dataset_temporal_bounds(datasets[split]) for split in ("train", "validation")
    }
    train_end = bounds["train"]["target_end_ns"]
    validation_history_start = bounds["validation"]["history_start_ns"]
    historical_boundary_passed = train_end < validation_history_start
    return {
        "boundary_semantics": "unseen_site_spatial_validation",
        "historical_extrapolation": False,
        "calendar_overlap_policy": "allowed_audited_not_historical",
        "global_historical_boundary_passed": historical_boundary_passed,
        "bounds": bounds,
        "checks": [
            {
                "earlier_split": "train",
                "later_split": "validation",
                "earlier_target_end_ns": train_end,
                "later_history_start_ns": validation_history_start,
                "passed": historical_boundary_passed,
                "enforced": False,
                "reason": "registered_disjoint_site_spatial_validation",
            }
        ],
    }


def _resolve_temporal_boundary_audit(
    datasets: Mapping[str, WindowDataset],
    *,
    tail_residual_cell: str | None,
) -> dict[str, object]:
    if tail_residual_cell is not None:
        return _audit_sparse_tail_spatial_boundary(datasets)
    return require_global_split_chronology(datasets)


def _resolve_training_treatment(
    *,
    adjustment_cell: str | None,
    specialist_ramp_cell: str | None,
    ramp_refinement_cell: str | None = None,
    validation_only: bool,
    independent_state_cell: str | None = None,
    state_boundary_cell: str | None = None,
    tail_residual_cell: str | None = None,
    internal_development_fraction: float | None = None,
) -> dict[str, object]:
    """Resolve one auditable treatment family without changing legacy cells."""

    selected = [
        value
        for value in (
            adjustment_cell,
            specialist_ramp_cell,
            ramp_refinement_cell,
            independent_state_cell,
            state_boundary_cell,
            tail_residual_cell,
        )
        if value is not None
    ]
    if len(selected) > 1:
        raise ValueError(
            "adjustment, specialist-ramp, ramp-refinement, and independent-state "
            "state-boundary, and tail-residual cells are mutually exclusive"
        )
    tail_treatment = _resolve_tail_residual_cell(
        tail_residual_cell,
        validation_only=validation_only,
        internal_development_fraction=internal_development_fraction,
    )
    if tail_residual_cell is not None:
        return {
            **tail_treatment,
            "adjustment_cell": None,
            "specialist_ramp_cell": None,
            "ramp_refinement_cell": None,
            "independent_state_cell": None,
            "state_boundary_cell": None,
            "state_head_mode": "flat",
            "state_representation_mode": "expert_detached",
            "enable_auxiliary_gate_path": False,
            "ramp_auxiliary_weight": 0.0,
            "ramp_auxiliary_beta": RAMP_AUXILIARY_BETA,
            "ramp_auxiliary_scope": "fusion_median",
            "state_boundary_weight": 1.0,
        }
    if state_boundary_cell is not None:
        return {
            **tail_treatment,
            **_resolve_state_boundary_cell(
                state_boundary_cell,
                validation_only=validation_only,
            ),
            "adjustment_cell": None,
            "specialist_ramp_cell": None,
            "ramp_refinement_cell": None,
            "independent_state_cell": None,
            "state_head_mode": "flat",
            "state_representation_mode": "expert_detached",
            "enable_auxiliary_gate_path": False,
            "ramp_auxiliary_weight": 0.0,
            "ramp_auxiliary_beta": RAMP_AUXILIARY_BETA,
            "ramp_auxiliary_scope": "fusion_median",
        }
    if independent_state_cell is not None:
        return {
            **tail_treatment,
            **_resolve_independent_state_cell(
                independent_state_cell,
                validation_only=validation_only,
            ),
            "adjustment_cell": None,
            "specialist_ramp_cell": None,
            "ramp_refinement_cell": None,
            "state_boundary_cell": None,
            "state_boundary_weight": 1.0,
        }
    if ramp_refinement_cell is not None:
        return {
            **tail_treatment,
            **_resolve_ramp_refinement_cell(
                ramp_refinement_cell,
                validation_only=validation_only,
            ),
            "adjustment_cell": None,
            "specialist_ramp_cell": None,
            "independent_state_cell": None,
            "state_boundary_cell": None,
            "state_boundary_weight": 1.0,
            "state_representation_mode": "expert_detached",
            "enable_auxiliary_gate_path": (ramp_refinement_cell == "I02"),
        }
    if specialist_ramp_cell is not None:
        return {
            **tail_treatment,
            **_resolve_specialist_ramp_cell(
                specialist_ramp_cell,
                validation_only=validation_only,
            ),
            "adjustment_cell": None,
            "ramp_refinement_cell": None,
            "independent_state_cell": None,
            "state_boundary_cell": None,
            "state_boundary_weight": 1.0,
            "state_representation_mode": "expert_detached",
            "enable_auxiliary_gate_path": False,
        }
    return {
        **tail_treatment,
        **_resolve_adjustment_cell(
            adjustment_cell,
            validation_only=validation_only,
        ),
        "specialist_ramp_cell": None,
        "ramp_refinement_cell": None,
        "independent_state_cell": None,
        "state_boundary_cell": None,
        "state_boundary_weight": 1.0,
        "state_representation_mode": "expert_detached",
        "enable_auxiliary_gate_path": False,
        "ramp_auxiliary_scope": "fusion_median",
    }


def _resolve_selection_datasets(
    train_data,
    external_validation_data,
    *,
    internal_development_fraction: float | None,
    validation_only: bool,
    specialist_ramp_cell: str | None,
    ramp_refinement_cell: str | None = None,
    independent_state_cell: str | None = None,
    state_boundary_cell: str | None = None,
    tail_residual_cell: str | None = None,
    horizon_steps: int,
    origin_stride_steps: int,
):
    """Select early-stopping data without consuming external validation."""

    if internal_development_fraction is None:
        return (
            train_data,
            external_validation_data,
            {
                "method": "external_validation_early_stopping",
                "external_validation_usage": "early_stopping_and_prediction",
            },
        )
    if not validation_only or (
        specialist_ramp_cell is None
        and ramp_refinement_cell is None
        and independent_state_cell is None
        and state_boundary_cell is None
        and tail_residual_cell is None
    ):
        raise ValueError(
            "internal development splitting is only available for validation-only "
            "specialist ramp, ramp refinement, independent state, or state boundary cells"
        )
    split = chronological_development_split(
        train_data,
        fraction=internal_development_fraction,
        horizon_steps=horizon_steps,
        origin_stride_steps=origin_stride_steps,
    )
    return (
        split.train,
        split.development,
        {
            **split.audit,
            "external_validation_usage": "prediction_once_after_fit",
        },
    )


def _ramp_positive_weight_from_counts(stable_count: int, ramp_count: int) -> float:
    """Return the registered train-only square-root imbalance correction."""

    if stable_count < 0 or ramp_count < 0:
        raise ValueError("training label counts cannot be negative")
    if stable_count + ramp_count <= 0:
        raise ValueError("positive training labels are required")
    if ramp_count == 0:
        return 4.0
    ratio = math.sqrt(stable_count / ramp_count)
    return float(min(4.0, max(1.0, ratio)))


def _training_ramp_positive_weight(loader) -> tuple[float, dict[str, int]]:
    """Count only valid daylight train labels and derive the hierarchical weight."""

    stable_count = 0
    ramp_count = 0
    for batch in loader:
        labels = batch["state_labels"]
        valid = (labels >= 0) & (labels <= 2)
        daylight = batch.get("daylight_mask")
        if daylight is not None:
            valid = valid & daylight.bool()
        stable_count += int((valid & (labels == 0)).sum())
        ramp_count += int((valid & ((labels == 1) | (labels == 2))).sum())
    return _ramp_positive_weight_from_counts(stable_count, ramp_count), {
        "stable": stable_count,
        "ramp": ramp_count,
    }


def _run_mode_metadata(
    *,
    validation_only: bool,
    diagnostic_ablation: str | None,
    adjustment_cell: str | None = None,
    specialist_ramp_cell: str | None = None,
    ramp_refinement_cell: str | None = None,
    independent_state_cell: str | None = None,
    state_boundary_cell: str | None = None,
    tail_residual_cell: str | None = None,
) -> dict[str, object]:
    treatments = [
        value
        for value in (
            diagnostic_ablation,
            adjustment_cell,
            specialist_ramp_cell,
            ramp_refinement_cell,
            independent_state_cell,
            state_boundary_cell,
            tail_residual_cell,
        )
        if value is not None
    ]
    if treatments and not validation_only:
        raise ValueError("diagnostic ablations are only available with --validation-only")
    if len(treatments) > 1:
        raise ValueError("diagnostic and development treatments cannot be combined")
    metadata: dict[str, object] = {
        "selection_boundary": (
            "validation_only" if validation_only else "registered_split_protocol"
        ),
        "diagnostic_only": bool(treatments),
        "diagnostic_ablation": (
            "sparse_tail_residual" if tail_residual_cell is not None else diagnostic_ablation
        ),
        "adjustment_cell": adjustment_cell,
        "specialist_ramp_cell": specialist_ramp_cell,
        "ramp_refinement_cell": ramp_refinement_cell,
        "independent_state_cell": independent_state_cell,
        "state_boundary_cell": state_boundary_cell,
    }
    if tail_residual_cell is not None:
        metadata["tail_residual_cell"] = tail_residual_cell
        metadata.update(
            {
                "evaluation_boundary": "unseen_site_spatial_validation",
                "historical_extrapolation": False,
                "calendar_overlap_policy": "allowed_audited_not_historical",
            }
        )
    return metadata


def _prediction_payload_for_archive(
    predictions: dict[str, torch.Tensor], *, include_moe_audit: bool
) -> dict[str, torch.Tensor]:
    persistent = {name: values for name, values in predictions.items() if not name.startswith("_")}
    if include_moe_audit:
        return persistent
    return {name: values for name, values in persistent.items() if name not in MOE_AUDIT_ARRAYS}


def _loader(dataset: WindowDataset, batch_size: int, shuffle: bool, workers: int):
    return build_window_loader(dataset, batch_size, shuffle, workers)


def _fit_training_stage(
    trainer: Any,
    train_loader: Any,
    validation_loader: Any,
    *,
    output_dir: Path,
    resolved_run: dict[str, object],
    experiment: Any,
    recovery_controller: Any | None,
) -> tuple[Any, Path]:
    """Persist resolved inputs and run fit through an optional recovery controller."""

    fit_kwargs: dict[str, object] = {}
    binding = getattr(getattr(recovery_controller, "request", None), "binding", None)
    if binding is not None and binding.raw.get("schema_version") == 2:
        from racepv.training.candidate_recovery import validate_candidate_resolved

        validate_candidate_resolved(binding, resolved_run)
    if recovery_controller is not None:
        output_dir, initial_progress, on_epoch_committed = recovery_controller.prepare(
            trainer,
            train_loader,
            validation_loader,
        )
        fit_kwargs = {
            "initial_progress": initial_progress,
            "on_epoch_committed": on_epoch_committed,
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(experiment, output_dir / "resolved_config.yaml")
    atomic_json_dump(resolved_run, output_dir / "resolved_run_config.json")
    result = trainer.fit(train_loader, validation_loader, **fit_kwargs)
    if recovery_controller is not None:
        recovery_controller.materialize_best_checkpoint(output_dir)
    return result, output_dir


def _container_hash(path: Path) -> str:
    """Legacy container identity; full underlying identity is recorded separately."""

    source = path / "store.json" if path.is_dir() else path
    return sha256_file(source)


def _spatial_roles(dataset: WindowDataset) -> set[str]:
    if dataset.schema is None:
        raise ValueError("registered schema is required for spatial-role enforcement")
    roles = dataset.schema.get("spatial_roles")
    if not isinstance(roles, dict):
        raise ValueError("registered schema is missing spatial_roles")
    return {str(value) for value in roles.values()}


def _require_non_target_fit_split(dataset: WindowDataset, *, purpose: str) -> None:
    roles = _spatial_roles(dataset)
    if "zero_parameter_test" in roles:
        raise ValueError(
            f"{purpose} cannot use a zero_parameter_test archive; target sites are "
            "reserved for prediction with no model update or conformal fitting"
        )


def _require_collection_split(dataset: WindowDataset, expected: str) -> None:
    declared = getattr(dataset, "collection_split", None)
    if declared is not None and declared != expected:
        raise ValueError(f"{expected} input points to a collection declared as split={declared!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    from racepv.cli.modern_baseline import _add_prepared_expectations

    _add_prepared_expectations(parser)
    from racepv.cli.scoped_candidate import add_candidate_qualification_arguments

    add_candidate_qualification_arguments(parser)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "fit on train and write validation predictions only; calibration/test "
            "arguments are forbidden in this screening mode"
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backbone", choices=("auto", "mamba", "gru", "tcn"), default="mamba")
    parser.add_argument("--no-clear-sky-prior", action="store_true")
    parser.add_argument(
        "--router-mode", choices=("contextual", "amplitude_only"), default="contextual"
    )
    parser.add_argument("--fusion-mode", choices=("learned", "equal"), default="learned")
    parser.add_argument(
        "--prediction-state-coupling",
        choices=("decoupled", "joint_head", "state_as_gate"),
        help=("validation-only sealed ablation of state/forecast gradient and routing semantics"),
    )
    parser.add_argument(
        "--weather-store",
        type=Path,
        help="read-only issued-weather NPZ sidecar shared by train and validation",
    )
    parser.add_argument(
        "--forecast-weather-mode",
        choices=("off", "experts", "experts_fusion", "transient_isolated"),
        default="off",
        help="validation-only direct weather injection treatment",
    )
    parser.add_argument(
        "--frozen-weather-adapter-from",
        type=Path,
        help=("validation-only W00 checkpoint used to initialize a frozen-core weather adapter"),
    )
    parser.add_argument(
        "--active-experts",
        nargs="+",
        choices=("stable", "ramp_up", "ramp_down"),
        default=("stable", "ramp_up", "ramp_down"),
    )
    parser.add_argument("--state-loss-weight", type=float)
    parser.add_argument("--specialist-loss-weight", type=float)
    parser.add_argument(
        "--diagnostic-ablation",
        choices=tuple(SPECIALIST_DIAGNOSTIC_WEIGHTS),
        help=(
            "validation-only diagnostic that uses a named specialist qCRPS "
            "coefficient without changing the registered RaCE-PV objective"
        ),
    )
    parser.add_argument(
        "--adjustment-cell",
        choices=tuple(ADJUSTMENT_CELLS),
        help="validation-only fixed RaCE-PV one-day adjustment treatment",
    )
    parser.add_argument(
        "--specialist-ramp-cell",
        choices=tuple(SPECIALIST_RAMP_CELLS),
        help=(
            "validation-only bounded treatment applying SmoothL1 to the true "
            "ramp-direction expert median"
        ),
    )
    parser.add_argument(
        "--ramp-refinement-cell",
        choices=tuple(RAMP_REFINEMENT_CELLS),
        help=(
            "validation-only bounded gate-regret or direction-expert tail-MSE "
            "treatment selected by the sealed ramp diagnostics"
        ),
    )
    parser.add_argument(
        "--independent-state-cell",
        choices=tuple(INDEPENDENT_STATE_CELLS),
        help=(
            "validation-only raw-input state-head pair with an optional "
            "RNG-neutral isolated gate auxiliary"
        ),
    )
    parser.add_argument(
        "--state-boundary-cell",
        choices=tuple(STATE_BOUNDARY_CELLS),
        help=(
            "validation-only flat state-head pair that upweights valid "
            "within-horizon onset, offset, and direction-reversal labels"
        ),
    )
    parser.add_argument(
        "--tail-residual-cell",
        choices=tuple(TAIL_RESIDUAL_CELLS),
        help=(
            "validation-only sparse bounded residual treatment evaluated on "
            "the registered new-site boundary"
        ),
    )
    parser.add_argument(
        "--internal-development-fraction",
        type=float,
        help=(
            "reserve each training shard's chronological tail for early stopping; "
            "only valid with --validation-only and a specialist/refinement cell"
        ),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--reliability-cell",
        choices=("control", "A"),
        help="matched no-specialist candidate pair; requires qualified corrected schema3 inputs",
    )
    parser.add_argument(
        "--allow-development-data",
        action="store_true",
        help=(
            "permit a provenance-complete but evidence-ineligible synthetic/debug "
            "archive; never use this option for reported experiments"
        ),
    )
    return parser


def run_training(args: argparse.Namespace, *, recovery_controller: Any | None = None) -> None:
    from racepv.cli.modern_baseline import (
        _preflight_prepared_inputs,
        _prepared_expectations,
        _prepared_output_metadata,
    )

    reliability_protocol = resolve_reliability_request(args)
    if reliability_protocol is not None and recovery_controller is not None:
        if getattr(getattr(recovery_controller, "request", None), "binding", None) is None:
            raise ValueError("candidate recovery executor requires a pinned schema2 binding")
        from racepv.training.candidate_recovery import inspect_candidate_recovery
        from racepv.training.recovery import validate_runtime_contract
        from scripts.run_racepv_reference_seed import _collect_runtime_contract

        pinned_candidate = inspect_candidate_recovery(recovery_controller.request.binding, args)
        seed_everything(args.seed)
        validate_runtime_contract(
            recovery_controller.request.binding,
            _collect_runtime_contract(
                args.seed,
                num_workers=pinned_candidate["experiment"]["training"]["num_workers"],
                amp_enabled=pinned_candidate["experiment"]["training"]["mixed_precision"],
            ),
        )
    elif (
        recovery_controller is not None
        and recovery_controller.request.binding.raw.get("schema_version") == 2
    ):
        raise ValueError("candidate binding requires explicit candidate cell")
    require_reliability_recovery_binding(
        recovery_controller, reliability_protocol, seed=getattr(args, "seed", None)
    )
    candidate = None
    if reliability_protocol is not None:
        from racepv.data.scoped_qualification import load_qualified_candidate

        if not any(
            getattr(args, name, None) is not None
            for name in (
                "qualification_root",
                "qualification_name",
                "qualification_sha256",
                "producer_root",
                "producer_review_sha256",
            )
        ):
            _preflight_prepared_inputs(
                args
            )  # Preserve the existing missing-qualification diagnosis.
            raise ValueError("schema3 qualification is required for the candidate")
        candidate = load_qualified_candidate(args)
        prepared_identities = candidate["identities"]
        if recovery_controller is not None:
            from racepv.training.candidate_recovery import _equal

            _equal(
                prepared_identities,
                pinned_candidate["qualified_prepared_inputs"],
                "fresh prepared inputs",
            )
    else:
        prepared_identities = _preflight_prepared_inputs(args)
    prepared_expectations = _prepared_expectations(args)
    if recovery_controller is not None and (
        args.calibration is not None
        or args.test is not None
        or bool(args.allow_development_data)
        or not bool(args.validation_only)
    ):
        raise ValueError(
            "recoverable reference runs require registered validation-only data "
            "without held-out or development inputs"
        )
    input_paths = _resolve_input_paths(args)
    _validate_tail_residual_request(args)
    coupling_protocol = _resolve_prediction_state_coupling(args)
    weather_protocol = _resolve_weather_protocol(args)
    adapter_request = _resolve_frozen_weather_adapter_request(args)
    weather_store = (
        IssuedWeatherStore(args.weather_store, **prepared_expectations)
        if args.weather_store is not None
        else None
    )
    run_mode = _run_mode_metadata(
        validation_only=args.validation_only,
        diagnostic_ablation=args.diagnostic_ablation,
        adjustment_cell=args.adjustment_cell,
        specialist_ramp_cell=args.specialist_ramp_cell,
        ramp_refinement_cell=args.ramp_refinement_cell,
        independent_state_cell=args.independent_state_cell,
        state_boundary_cell=args.state_boundary_cell,
        tail_residual_cell=args.tail_residual_cell,
    )
    if bool(coupling_protocol["prediction_state_coupling_requested"]):
        run_mode = {**run_mode, **coupling_protocol}
    if reliability_protocol is not None:
        run_mode = {**run_mode, **reliability_protocol}
    treatment = _resolve_training_treatment(
        adjustment_cell=args.adjustment_cell,
        specialist_ramp_cell=args.specialist_ramp_cell,
        ramp_refinement_cell=args.ramp_refinement_cell,
        validation_only=args.validation_only,
        independent_state_cell=args.independent_state_cell,
        state_boundary_cell=args.state_boundary_cell,
        tail_residual_cell=args.tail_residual_cell,
        internal_development_fraction=args.internal_development_fraction,
    )
    experiment = load_config(args.config)
    seed = experiment.training.seed if args.seed is None else args.seed
    seed_everything(seed)
    device = select_device(args.device)
    output_dir = args.output or Path(experiment.output_dir)
    if args.tail_residual_cell is None and recovery_controller is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(experiment, output_dir / "resolved_config.yaml")

    if candidate is not None:
        train_data = candidate["datasets"]["train"]
        validation_data = candidate["datasets"]["validation"]
    else:
        train_data = load_window_dataset(
            input_paths["train"], verify_content=True, **prepared_expectations
        )
        validation_data = load_window_dataset(
            input_paths["validation"], verify_content=True, **prepared_expectations
        )
        train_data.require_registered_schema()
        validation_data.require_registered_schema()
        train_data.require_registered_evidence(allow_ineligible=args.allow_development_data)
        validation_data.require_registered_evidence(allow_ineligible=args.allow_development_data)
        _require_collection_split(train_data, "train")
        _require_collection_split(validation_data, "validation")
        train_roles = _spatial_roles(train_data)
        if not train_roles.issubset({"standard", "source_train"}):
            raise ValueError(
                "training weights may only use standard/source_train archives; "
                f"received roles {sorted(train_roles)}"
            )
        _require_non_target_fit_split(validation_data, purpose="validation/early stopping")
        if (
            train_data.registered_schema_signature()
            != validation_data.registered_schema_signature()
        ):
            raise ValueError("train and validation archives do not share the same feature schema")
    registered_datasets: dict[str, WindowDataset] = {
        "train": train_data,
        "validation": validation_data,
    }
    calibration_data: WindowDataset | None = None
    test_data: WindowDataset | None = None
    if args.calibration:
        calibration_data = load_window_dataset(
            args.calibration, verify_content=True, **prepared_expectations
        )
        calibration_data.require_registered_schema()
        calibration_data.require_registered_evidence(allow_ineligible=args.allow_development_data)
        _require_collection_split(calibration_data, "calibration")
        _require_non_target_fit_split(calibration_data, purpose="conformal calibration")
        if (
            calibration_data.registered_schema_signature()
            != train_data.registered_schema_signature()
        ):
            raise ValueError("calibration archive feature schema differs from training")
        registered_datasets["calibration"] = calibration_data
    if args.test:
        test_data = load_window_dataset(args.test, verify_content=True, **prepared_expectations)
        test_data.require_registered_schema()
        test_data.require_registered_evidence(allow_ineligible=args.allow_development_data)
        _require_collection_split(test_data, "test")
        if test_data.registered_schema_signature() != train_data.registered_schema_signature():
            raise ValueError("test archive feature schema differs from training")
        registered_datasets["test"] = test_data
    spatial_protocol = (
        {
            split: sorted(set(dataset.view.spatial_roles.values()))
            for split, dataset in registered_datasets.items()
        }
        if candidate is not None
        else require_registered_spatial_protocol(registered_datasets)
    )
    if args.tail_residual_cell is not None:
        if train_data.schema is None or validation_data.schema is None:
            raise ValueError("tail residual treatment requires registered schemas")
        _validate_sparse_tail_dataset_boundary(
            train_sites={str(value) for value in train_data.schema["site_id_map"].values()},
            validation_sites={
                str(value) for value in validation_data.schema["site_id_map"].values()
            },
        )
        if recovery_controller is None:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_config(experiment, output_dir / "resolved_config.yaml")
    temporal_audit = (
        candidate["engineering"]["chronology"]
        if candidate is not None
        else _resolve_temporal_boundary_audit(
            registered_datasets, tail_residual_cell=args.tail_residual_cell
        )
    )
    if weather_store is not None:
        train_data = WeatherAugmentedDataset(train_data, weather_store, **prepared_expectations)
        validation_data = WeatherAugmentedDataset(
            validation_data, weather_store, **prepared_expectations
        )
    fit_train_data, early_stopping_data, selection_data_audit = _resolve_selection_datasets(
        train_data,
        validation_data,
        internal_development_fraction=args.internal_development_fraction,
        validation_only=args.validation_only,
        specialist_ramp_cell=args.specialist_ramp_cell,
        ramp_refinement_cell=args.ramp_refinement_cell,
        independent_state_cell=args.independent_state_cell,
        state_boundary_cell=args.state_boundary_cell,
        tail_residual_cell=args.tail_residual_cell,
        horizon_steps=experiment.data.horizon,
        origin_stride_steps=experiment.data.stride,
    )
    if candidate is not None:
        from racepv.cli.scoped_candidate import candidate_model_config

        model_config = candidate_model_config(train_data, experiment, backbone=args.backbone)
        model, reliability_objective = _build_reliability_components(
            model_config, args.reliability_cell
        )
    else:
        sample = train_data[0]
        if train_data.schema is None:  # pragma: no cover - narrowed by schema validation
            raise RuntimeError("registered training schema unexpectedly missing")
        feature_indices = _required_future_feature_indices(
            tuple(str(name) for name in train_data.schema["future_known_columns"])
        )
        model_config = RaCEPVConfig(
            history_features=sample["history_values"].shape[-1],
            future_features=sample["future_known"].shape[-1],
            site_features=sample["site_metadata"].shape[-1],
            history_length=experiment.data.history_length,
            horizon=experiment.data.horizon,
            sampling_minutes=experiment.data.frequency_minutes,
            quantiles=experiment.quantiles,
            power_index=0,
            history_clear_sky_index=1,
            **feature_indices,
            backbone=args.backbone,
            use_clear_sky_prior=not args.no_clear_sky_prior,
            clear_sky_history_feature_indices=(2, 7),
            router_mode=args.router_mode,
            fusion_mode=args.fusion_mode,
            state_head_mode=str(treatment["state_head_mode"]),
            state_representation_mode=str(treatment["state_representation_mode"]),
            prediction_state_coupling=str(coupling_protocol["prediction_state_coupling"]),
            enable_auxiliary_gate_path=bool(treatment["enable_auxiliary_gate_path"]),
            tail_residual_mode=str(treatment["tail_residual_mode"]),
            future_weather_features=(
                len(FORECAST_WEATHER_FEATURES) if weather_store is not None else 0
            ),
            forecast_weather_mode=str(weather_protocol["forecast_weather_mode"]),
            active_experts=tuple(args.active_experts),
        )
        model = RaCEPV(model_config)
        reliability_objective = None
    if model.backbone_name == "mamba_ssm" and torch.device(device).type != "cuda":
        raise RuntimeError(
            "official mamba_ssm is a registered CUDA/WSL path; choose a CUDA device "
            "or use --backbone gru/tcn for a development run"
        )
    selected_state_weight = _v2_state_loss_weight(
        experiment.training.state_loss_weight,
        args.state_loss_weight,
    )
    if reliability_protocol is None:
        selected_specialist_weight, diagnostic_metadata = _resolve_specialist_objective(
            configured=experiment.training.specialist_loss_weight,
            override=args.specialist_loss_weight,
            diagnostic_ablation=args.diagnostic_ablation,
            validation_only=args.validation_only,
        )
    else:
        selected_specialist_weight = None
        diagnostic_metadata = {"diagnostic_ablation": None}
    if (
        not bool(coupling_protocol["prediction_state_coupling_requested"])
        and args.tail_residual_cell is None
        and diagnostic_metadata["diagnostic_ablation"] != run_mode["diagnostic_ablation"]
    ):
        raise RuntimeError("diagnostic training mode metadata is inconsistent")
    ramp_positive_weight = 1.0
    ramp_label_counts: dict[str, int] | None = None
    if model_config.state_head_mode == "hierarchical":
        ramp_positive_weight, ramp_label_counts = _training_ramp_positive_weight(
            _loader(
                fit_train_data,
                experiment.training.batch_size,
                False,
                experiment.training.num_workers,
            )
        )
    criterion = (
        reliability_objective
        if reliability_objective is not None
        else RaCEPVLoss(
            quantiles=experiment.quantiles,
            specialist_weight=selected_specialist_weight,
            active_experts=model_config.active_experts,
            state_head_mode=model_config.state_head_mode,
            ramp_positive_weight=ramp_positive_weight,
            ramp_auxiliary_weight=float(treatment["ramp_auxiliary_weight"]),
            ramp_auxiliary_beta=float(treatment["ramp_auxiliary_beta"]),
            ramp_auxiliary_scope=str(treatment["ramp_auxiliary_scope"]),
            state_boundary_weight=float(treatment["state_boundary_weight"]),
            tail_auxiliary_weight=float(treatment["tail_auxiliary_weight"]),
            tail_sparsity_weight=float(treatment["tail_sparsity_weight"]),
        )
    )
    loss_weights = {
        "state": selected_state_weight,
        "specialist": selected_specialist_weight,
        "ramp_auxiliary": float(treatment["ramp_auxiliary_weight"]),
        "state_boundary": float(treatment["state_boundary_weight"]),
        "tail_auxiliary": float(treatment["tail_auxiliary_weight"]),
        "tail_sparsity": float(treatment["tail_sparsity_weight"]),
    }
    if reliability_protocol is not None:
        loss_weights = {"forecast": 1.0, "state": 1.0}
    if model_config.state_head_mode == "hierarchical":
        state_objective = "hierarchical_train_only_sqrt_balanced"
    elif float(treatment["state_boundary_weight"]) > 1.0:
        state_objective = "transition_weighted_flat_cross_entropy"
    else:
        state_objective = "unweighted_cross_entropy"
    train_index_hash = _container_hash(args.train)
    validation_index_hash = _container_hash(args.validation)
    train_content_hash = getattr(train_data, "content_identity", train_index_hash)
    validation_content_hash = getattr(validation_data, "content_identity", validation_index_hash)
    archive_weather_protocol = _serializable_weather_protocol(
        weather_protocol,
        weather_store_sha256=(weather_store.content_sha256 if weather_store is not None else None),
        weather_feature_names=(
            tuple(weather_store.feature_names) if weather_store is not None else ()
        ),
    )
    training_protocol = (
        PAIR_PROTOCOL
        if reliability_protocol is not None
        else str(adapter_request["training_protocol"])
    )
    optimizer_schema = (
        FROZEN_WEATHER_ADAPTER_SCHEMA
        if bool(adapter_request["enabled"])
        else "separated_forecast_state_v1"
    )
    adapter_audit: dict[str, object] = {}
    if bool(adapter_request["enabled"]):
        source_checkpoint = adapter_request["source_checkpoint"]
        if not isinstance(source_checkpoint, Path):
            raise TypeError("frozen weather adapter source checkpoint must be a path")
        if weather_store is None:  # guarded before data loading
            raise RuntimeError("frozen weather adapter lost its weather store")
        adapter_audit = initialize_frozen_weather_adapter(
            model,
            source_checkpoint,
            expected_seed=seed,
            expected_train_content_sha256=str(train_content_hash),
            expected_validation_content_sha256=str(validation_content_hash),
            expected_weather_store_sha256=weather_store.content_sha256,
            require_official_mamba=model.backbone_name == "mamba_ssm",
        )
        if adapter_audit.get("optimizer_schema") != optimizer_schema:
            raise RuntimeError("frozen weather adapter optimizer schema drifted")
    resolved_run = {
        **_prepared_output_metadata(prepared_identities),
        **run_mode,
        "training_protocol": training_protocol,
        "optimizer_schema": optimizer_schema,
        "adapter_initialization": adapter_audit or None,
        "experiment": experiment.to_dict(),
        "cli_overrides": {
            "seed": seed,
            "device": device,
            "backbone": args.backbone,
            "no_clear_sky_prior": args.no_clear_sky_prior,
            "router_mode": args.router_mode,
            "fusion_mode": args.fusion_mode,
            "prediction_state_coupling_requested": coupling_protocol[
                "prediction_state_coupling_requested"
            ],
            "prediction_state_coupling": model_config.prediction_state_coupling,
            "active_experts": list(args.active_experts),
            "state_loss_weight": selected_state_weight,
            "specialist_loss_weight": selected_specialist_weight,
            "validation_only": args.validation_only,
            "diagnostic_ablation": args.diagnostic_ablation,
            "adjustment_cell": args.adjustment_cell,
            "state_head_mode": model_config.state_head_mode,
            "ramp_positive_weight": ramp_positive_weight,
            "ramp_label_counts": ramp_label_counts,
            "specialist_ramp_cell": args.specialist_ramp_cell,
            "ramp_refinement_cell": args.ramp_refinement_cell,
            "independent_state_cell": args.independent_state_cell,
            "state_boundary_cell": args.state_boundary_cell,
            "tail_residual_cell": args.tail_residual_cell,
            "tail_residual_mode": treatment["tail_residual_mode"],
            "tail_auxiliary_weight": treatment["tail_auxiliary_weight"],
            "tail_sparsity_weight": treatment["tail_sparsity_weight"],
            "state_boundary_weight": treatment["state_boundary_weight"],
            "state_representation_mode": model_config.state_representation_mode,
            "enable_auxiliary_gate_path": model_config.enable_auxiliary_gate_path,
            "ramp_auxiliary_weight": treatment["ramp_auxiliary_weight"],
            "ramp_auxiliary_beta": treatment["ramp_auxiliary_beta"],
            "ramp_auxiliary_scope": treatment["ramp_auxiliary_scope"],
            "internal_development_fraction": args.internal_development_fraction,
            "selection_data_audit": selection_data_audit,
            "allow_development_data": args.allow_development_data,
            "weather_protocol": archive_weather_protocol,
            "training_protocol": training_protocol,
            "optimizer_schema": optimizer_schema,
            "adapter_initialization": adapter_audit or None,
        },
        "model_config": asdict(model_config),
        "model_metadata": model.model_metadata,
        "efficiency_benchmark": {
            "batch_size_windows": EFFICIENCY_BENCHMARK_BATCH_SIZE,
            "warmup_repetitions": EFFICIENCY_BENCHMARK_WARMUP,
            "timed_repetitions": EFFICIENCY_BENCHMARK_REPETITIONS,
            "population": "first validation windows in registered order",
        },
    }
    if candidate is not None and recovery_controller is not None:
        from racepv.training.candidate_recovery import validate_candidate_resolved

        validate_candidate_resolved(recovery_controller.request.binding, resolved_run)
    trainer = EarlyStoppingTrainer(
        model,
        criterion,
        device=device,
        learning_rate=experiment.training.learning_rate,
        weight_decay=experiment.training.weight_decay,
        max_epochs=experiment.training.max_epochs,
        patience=experiment.training.patience,
        gradient_clip_norm=experiment.training.gradient_clip_norm,
        mixed_precision=experiment.training.mixed_precision,
        checkpoint_path=(output_dir / "best.ckpt" if recovery_controller is None else None),
        checkpoint_metadata={
            **_prepared_output_metadata(prepared_identities),
            "model_config": asdict(model_config),
            "model_metadata": model.model_metadata,
            "seed": seed,
            "quantiles": list(experiment.quantiles),
            "loss_weights": loss_weights,
            "state_loss": state_objective,
            "optimizer_schema": optimizer_schema,
            "training_protocol": training_protocol,
            **adapter_audit,
            "ramp_positive_weight": ramp_positive_weight,
            "ramp_label_counts": ramp_label_counts,
            "ramp_auxiliary_scope": treatment["ramp_auxiliary_scope"],
            "ramp_decision_threshold": model_config.ramp_decision_threshold,
            "selection_data_audit": selection_data_audit,
            "weather_protocol": {
                "forecast_weather_mode": model_config.forecast_weather_mode,
                "weather_store_sha256": (
                    weather_store.content_sha256 if weather_store is not None else None
                ),
                "weather_feature_names": (
                    list(weather_store.feature_names) if weather_store is not None else []
                ),
                "forecast_weather_contract_version": (
                    model_config.forecast_weather_contract_version
                ),
                "state_head_weather_access": False,
            },
            **run_mode,
            **model.model_metadata,
            "train_npz_sha256": train_index_hash,
            "validation_npz_sha256": validation_index_hash,
            "train_window_content_sha256": train_content_hash,
            "validation_window_content_sha256": validation_content_hash,
        },
        selection_metric=FORECAST_SELECTION_METRIC,
        separate_state_optimizer=not bool(adapter_request["enabled"]),
        single_optimizer_schema=optimizer_schema,
    )
    train_loader = _loader(
        fit_train_data,
        experiment.training.batch_size,
        True,
        experiment.training.num_workers,
    )
    selection_loader = _loader(
        early_stopping_data,
        experiment.training.batch_size,
        False,
        experiment.training.num_workers,
    )
    if candidate is not None:
        from racepv.data.scoped_qualification import reverify_qualified_candidate

        reverify_qualified_candidate(args, prepared_identities)
    result, output_dir = _fit_training_stage(
        trainer,
        train_loader,
        selection_loader,
        output_dir=output_dir,
        resolved_run=resolved_run,
        experiment=experiment,
        recovery_controller=recovery_controller,
    )
    if candidate is not None:
        reverify_qualified_candidate(args, prepared_identities)
    model_stage_restored = False
    validation_stage_restored = False
    summary_stage_restored = False
    if recovery_controller is not None:
        model_stage_restored = recovery_controller.restore_export_stage("model", output_dir)
        validation_stage_restored = recovery_controller.restore_export_stage(
            "validation", output_dir
        )
        summary_stage_restored = recovery_controller.restore_export_stage("summary", output_dir)

    def prediction_metadata(
        split: str,
        source: Path,
        dataset: WindowDataset,
        *,
        include_moe_audit: bool,
    ) -> dict[str, object]:
        if candidate is not None:
            from racepv.cli.scoped_candidate import candidate_dataset_metadata

            dataset_metadata = candidate_dataset_metadata(dataset, prepared_identities)
        else:
            dataset_metadata = {
                name: dataset.schema[name]
                for name in (
                    "site_id_map",
                    "dataset_kind",
                    "simulated",
                    "ramp_method",
                    "registered_evidence_eligible",
                )
            }
        return {
            **_prepared_output_metadata(prepared_identities, split=split),
            "split": split,
            "run_name": experiment.run_name,
            "seed": seed,
            "quantiles": list(experiment.quantiles),
            "source_npz_sha256": _container_hash(source),
            "source_window_content_sha256": getattr(
                dataset, "content_identity", _container_hash(source)
            ),
            **dataset_metadata,
            **({"model_config": asdict(model_config)} if candidate is not None else {}),
            "development_data_override": args.allow_development_data,
            "model_sha256": model_hash,
            "best_checkpoint_sha256": checkpoint_hash,
            "resolved_run_config_sha256": resolved_run_hash,
            "experiment_config_sha256": sha256_file(args.config),
            "loss_weights": loss_weights,
            "state_loss": state_objective,
            "optimizer_schema": optimizer_schema,
            "training_protocol": training_protocol,
            **adapter_audit,
            "ramp_positive_weight": ramp_positive_weight,
            "ramp_label_counts": ramp_label_counts,
            "ramp_auxiliary_scope": treatment["ramp_auxiliary_scope"],
            "selection_data_audit": selection_data_audit,
            "weather_protocol": archive_weather_protocol,
            "moe_audit_payload": (
                "included" if include_moe_audit else "omitted_non_test_storage_optimization"
            ),
            **run_mode,
            **model.model_metadata,
        }

    model_path = output_dir / "model.pt"
    if not model_stage_restored:
        torch.save(
            {
                **(_prepared_output_metadata(prepared_identities) if candidate is not None else {}),
                "model_state_dict": model.state_dict(),
                "model_config": asdict(model_config),
                "quantiles": list(experiment.quantiles),
                "seed": seed,
                "model_metadata": model.model_metadata,
                "loss_weights": loss_weights,
                "state_loss": state_objective,
                "optimizer_schema": optimizer_schema,
                "training_protocol": training_protocol,
                **adapter_audit,
                "ramp_positive_weight": ramp_positive_weight,
                "ramp_label_counts": ramp_label_counts,
                "ramp_auxiliary_scope": treatment["ramp_auxiliary_scope"],
                "selection_data_audit": selection_data_audit,
                **run_mode,
                "train_npz_sha256": train_index_hash,
                "validation_npz_sha256": validation_index_hash,
                "train_window_content_sha256": train_content_hash,
                "validation_window_content_sha256": validation_content_hash,
            },
            model_path,
        )
    model_hash = sha256_file(model_path)
    checkpoint_hash = sha256_file(output_dir / "best.ckpt")
    resolved_run_hash = sha256_file(output_dir / "resolved_run_config.json")
    if recovery_controller is not None and not model_stage_restored:
        recovery_controller.publish_export_stage("model", output_dir)
    if not validation_stage_restored:
        external_validation_loader = _loader(
            validation_data,
            experiment.training.batch_size,
            False,
            experiment.training.num_workers,
        )
        validation_predictions = trainer.predict(external_validation_loader)
        # V2 archives are self-auditing: state and fusion logits are always retained
        # so evaluators can fail closed on semantic mismatches.
        validation_moe_audit = True
        save_predictions_npz(
            output_dir / "validation_predictions.npz",
            _prediction_payload_for_archive(
                validation_predictions, include_moe_audit=validation_moe_audit
            ),
            metadata=prediction_metadata(
                "validation",
                args.validation,
                validation_data,
                include_moe_audit=validation_moe_audit,
            ),
        )
    if args.calibration:
        assert calibration_data is not None
        calibration_loader = _loader(
            calibration_data,
            experiment.training.batch_size,
            False,
            experiment.training.num_workers,
        )
        save_predictions_npz(
            output_dir / "calibration_predictions.npz",
            _prediction_payload_for_archive(
                trainer.predict(calibration_loader), include_moe_audit=True
            ),
            metadata=prediction_metadata(
                "calibration",
                args.calibration,
                calibration_data,
                include_moe_audit=True,
            ),
        )
    if args.test:
        assert test_data is not None
        test_loader = _loader(
            test_data,
            experiment.training.batch_size,
            False,
            experiment.training.num_workers,
        )
        save_predictions_npz(
            output_dir / "test_predictions.npz",
            trainer.predict(test_loader),
            metadata=prediction_metadata("test", args.test, test_data, include_moe_audit=True),
        )
    if recovery_controller is not None and not validation_stage_restored:
        recovery_controller.publish_export_stage("validation", output_dir)

    if not summary_stage_restored:
        benchmark_loader = _loader(
            validation_data,
            EFFICIENCY_BENCHMARK_BATCH_SIZE,
            False,
            experiment.training.num_workers,
        )
        benchmark_batch = next(iter(benchmark_loader))
        benchmark_inputs = _benchmark_model_inputs(benchmark_batch)
        batch_inference = benchmark_inference(
            model,
            benchmark_inputs,
            warmup=EFFICIENCY_BENCHMARK_WARMUP,
            repetitions=EFFICIENCY_BENCHMARK_REPETITIONS,
        )
        trainable_parameters = count_parameters(model)
        efficiency = efficiency_audit_record(
            trainable_parameters=trainable_parameters,
            trainable_parameters_unavailable_reason=None,
            model_artifact_bytes=model_path.stat().st_size,
            model_artifact_definition="serialized inference model.pt file size",
            batch_inference=batch_inference,
        )
        atomic_json_dump(
            {
                **run_mode,
                "run_name": experiment.run_name,
                **_prepared_output_metadata(prepared_identities),
                "seed": seed,
                "device": device,
                "model": asdict(model_config),
                "model_metadata": model.model_metadata,
                "loss_weights": loss_weights,
                "state_loss": state_objective,
                "optimizer_schema": optimizer_schema,
                "training_protocol": training_protocol,
                **adapter_audit,
                "ramp_positive_weight": ramp_positive_weight,
                "ramp_label_counts": ramp_label_counts,
                "ramp_auxiliary_scope": treatment["ramp_auxiliary_scope"],
                "ramp_decision_threshold": model_config.ramp_decision_threshold,
                "selection_data_audit": selection_data_audit,
                "evidence_mode": {
                    "allow_development_data": args.allow_development_data,
                    **(
                        train_data.export_metadata
                        if candidate is not None
                        else {
                            name: train_data.schema[name] for name in ("dataset_kind", "simulated")
                        }
                    ),
                    "registered_evidence_eligible": (
                        candidate["qualification"].report["formal_ready"]
                        if candidate is not None
                        else train_data.schema["registered_evidence_eligible"]
                    ),
                },
                "global_temporal_audit": temporal_audit,
                "registered_spatial_protocol": spatial_protocol,
                "best_epoch": result.best_epoch,
                "best_validation_loss": result.best_validation_loss,
                "stopped_early": result.stopped_early,
                "history": [asdict(record) for record in result.history],
                "trainable_parameters": trainable_parameters,
                "efficiency": efficiency,
                "environment": environment_report(),
                "artifacts": {
                    "train_npz_sha256": train_index_hash,
                    "validation_npz_sha256": validation_index_hash,
                    "train_window_content_sha256": train_content_hash,
                    "validation_window_content_sha256": validation_content_hash,
                    "calibration_npz_sha256": (
                        _container_hash(args.calibration) if args.calibration else None
                    ),
                    "calibration_window_content_sha256": (
                        getattr(calibration_data, "content_identity", None)
                        if args.calibration
                        else None
                    ),
                    "test_npz_sha256": _container_hash(args.test) if args.test else None,
                    "test_window_content_sha256": (
                        getattr(test_data, "content_identity", None) if args.test else None
                    ),
                    "config_sha256": sha256_file(args.config),
                    "model_sha256": model_hash,
                    "best_checkpoint_sha256": checkpoint_hash,
                    "resolved_run_config_sha256": resolved_run_hash,
                },
            },
            output_dir / "run.json",
        )
    if recovery_controller is not None:
        if not summary_stage_restored:
            recovery_controller.publish_export_stage("summary", output_dir)
        recovery_controller.publish_completed_exports(output_dir)
    print(
        f"Completed {experiment.run_name}; best validation daylight qCRPS="
        f"{result.best_validation_loss:.6f}"
    )
    print(f"Actual transient backend: {model.backbone_name}")


def main() -> None:
    run_training(build_parser().parse_args())


if __name__ == "__main__":
    main()
