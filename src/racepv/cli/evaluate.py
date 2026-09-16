"""Calibrate and evaluate RaCE-PV prediction archives without retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from racepv.audit import sha256_file
from racepv.baselines import ClearSkyPersistence, SmartPersistence
from racepv.calibration import GlobalCQR, TimeBlockedConditionalCQR
from racepv.calibration.cqr import LeadBandCQR
from racepv.config import QUANTILE_LEVELS, SEEDS, load_config
from racepv.evidence import (
    require_registered_input,
    require_registered_json_input,
    require_registered_prediction_input,
)
from racepv.io import (
    WindowDataset,
    load_window_dataset,
    read_model_window_batch,
    require_global_split_chronology,
    require_registered_spatial_protocol,
)
from racepv.metrics import (
    benjamini_hochberg,
    binary_ramp_metrics,
    conditional_coverage_error,
    diebold_mariano_test,
    interval_score,
    point_forecast_metrics,
    prediction_interval_coverage,
    prediction_interval_normalized_width,
    probabilistic_forecast_metrics,
    ramp_classification_metrics,
    ramp_event_errors,
)
from racepv.models.config import (
    FORECAST_SELECTION_METRIC,
    FUSION_GATE_SEMANTICS,
    OBJECTIVE_VERSION,
    STATE_HEAD_SEMANTICS,
)
from racepv.utils import atomic_json_dump

REPORT_LEADS = (1, 2, 4, 8, 16, 24)
LEAD_BANDS = (("0-1h", 0, 4), ("1-3h", 4, 12), ("3-6h", 12, 24))
REFERENCE_SELECTION_METRICS = {
    "point": {
        "daylight_nrmse": "minimize",
        "daylight_nmae": "minimize",
    },
    "probabilistic": {
        "mean_pinball_loss": "minimize",
        "quantile_crps": "minimize",
    },
    "ramp_point": {
        "ramp_nrmse": "minimize",
        "ramp_nmae": "minimize",
    },
    "ramp_probabilistic": {
        "ramp_mean_pinball_loss": "minimize",
        "ramp_quantile_crps": "minimize",
    },
    "ramp": {
        "binary_auprc": "maximize",
        "binary_f1": "maximize",
        "binary_csi": "maximize",
        "macro_auprc": "maximize",
        "macro_f1": "maximize",
    },
}
DEFAULT_REFERENCE_SELECTION_METRICS = {
    "point": "daylight_nrmse",
    "probabilistic": "mean_pinball_loss",
    "ramp_point": "ramp_nrmse",
    "ramp_probabilistic": "ramp_quantile_crps",
    "ramp": "binary_auprc",
}


def _ramp_decision_threshold(metadata: Mapping[str, Any]) -> float:
    """Read the frozen binary-ramp threshold, preserving legacy archives."""

    raw = metadata.get("ramp_decision_threshold", 0.5)
    if isinstance(raw, bool):
        raise ValueError("ramp_decision_threshold must be numeric")
    try:
        threshold = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("ramp_decision_threshold must be numeric") from exc
    if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("ramp_decision_threshold must lie in (0, 1]")
    return threshold


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _load_metadata(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        raise FileNotFoundError(f"prediction metadata sidecar is required: {sidecar}")
    with sidecar.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{sidecar} must contain a JSON object")
    return value


def _window_container_sha256(path: Path) -> str:
    """Hash an NPZ/index or the manifest of a legacy directory store."""

    return sha256_file(path / "store.json" if path.is_dir() else path)


def _prepared_metadata_claim(metadata):
    return bool(
        {
            "prepared_contract",
            "prepared_contracts",
            "prepared_schema_version",
            "causality_profile",
        }.intersection(metadata)
    )


def _metadata_window_identities_match(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    """Compare full content identities when both exist, else legacy containers."""

    if _prepared_metadata_claim(first) or _prepared_metadata_claim(second):
        from racepv.baselines.extension_registry import validate_prepared_identity

        try:
            left = validate_prepared_identity(first.get("prepared_contract"))
            right = validate_prepared_identity(second.get("prepared_contract"))
        except (TypeError, ValueError):
            return False
        return left == right
    first_content = first.get("source_window_content_sha256")
    second_content = second.get("source_window_content_sha256")
    if first_content is not None and second_content is not None:
        return bool(first_content == second_content)
    first_container = first.get("source_npz_sha256")
    second_container = second.get("source_npz_sha256")
    return bool(first_container is not None and first_container == second_container)


def _require(archive: Mapping[str, np.ndarray], names: Sequence[str], label: str) -> None:
    missing = [name for name in names if name not in archive]
    if missing:
        raise KeyError(f"{label} is missing arrays: {missing}")


def _validate_prediction_archive(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    split: str,
    levels: Sequence[float],
    horizon: int,
    power_upper_bound: float,
) -> None:
    _require(
        arrays,
        (
            "quantiles",
            "state_probs",
            "targets",
            "state_labels",
            "daylight_mask",
            "timestamps",
            "day_ids",
            "site_ids",
        ),
        f"{split} predictions",
    )
    targets = arrays["targets"]
    if targets.ndim != 2 or targets.shape[0] == 0 or targets.shape[1] != horizon:
        raise ValueError(f"{split} targets must have non-empty shape [N,{horizon}]")
    expected = targets.shape
    if arrays["quantiles"].shape != expected + (len(levels),):
        raise ValueError(f"{split} quantiles have the wrong shape")
    if arrays["state_probs"].shape != expected + (3,):
        raise ValueError(f"{split} state probabilities have the wrong shape")
    for name in ("state_labels", "daylight_mask", "timestamps", "day_ids"):
        if arrays[name].shape != expected:
            raise ValueError(f"{split} {name} must match targets")
    if arrays["site_ids"].shape not in {(expected[0],), (expected[0], 1)}:
        raise ValueError(f"{split} site_ids must have one value per forecast origin")
    if np.any(~np.isfinite(targets)) or np.any(
        (targets < -1e-7) | (targets > power_upper_bound + 1e-7)
    ):
        raise ValueError(f"{split} targets must be finite normalized power values")
    quantiles = arrays["quantiles"]
    if np.any(~np.isfinite(quantiles)) or np.any(
        (quantiles < -1e-7) | (quantiles > power_upper_bound + 1e-7)
    ):
        raise ValueError(f"{split} quantiles must be finite and within physical bounds")
    if np.any(np.diff(arrays["quantiles"], axis=-1) < -1e-7):
        raise ValueError(f"{split} quantile predictions cross")
    probabilities = arrays["state_probs"]
    if (
        np.any(~np.isfinite(probabilities))
        or np.any((probabilities < -1e-7) | (probabilities > 1.0 + 1e-7))
        or not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-4)
    ):
        raise ValueError(f"{split} state probabilities are not finite simplex weights")
    raw_version = metadata.get("architecture_version", 1)
    if isinstance(raw_version, bool):
        raise ValueError(f"{split} architecture_version must be an integer")
    try:
        architecture_version = int(raw_version)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{split} architecture_version must be an integer") from exc
    if architecture_version not in {1, 2}:
        raise ValueError(f"{split} uses unsupported architecture_version={architecture_version}")

    v1_moe_arrays = {"expert_quantiles", "gate_weights", "router_logits"}
    v2_moe_arrays = {
        "expert_quantiles",
        "gate_weights",
        "fusion_logits",
        "state_logits",
        "router_logits",
    }
    if architecture_version == 2:
        expected_semantics = {
            "objective_version": OBJECTIVE_VERSION,
            "forecast_selection_metric": FORECAST_SELECTION_METRIC,
            "state_head_semantics": STATE_HEAD_SEMANTICS,
            "fusion_gate_semantics": FUSION_GATE_SEMANTICS,
        }
        for name, expected_value in expected_semantics.items():
            if metadata.get(name) != expected_value:
                raise ValueError(f"{split} v2 metadata {name} must equal {expected_value!r}")
        missing_v2 = sorted(v2_moe_arrays.difference(arrays))
        if missing_v2:
            raise ValueError(f"{split} v2 archive is missing required MoE arrays: {missing_v2}")
        moe_arrays = v2_moe_arrays
    else:
        unexpected_v2 = {"fusion_logits", "state_logits"}.intersection(arrays)
        if unexpected_v2:
            raise ValueError(
                f"{split} archive contains v2 arrays without architecture_version=2: "
                f"{sorted(unexpected_v2)}"
            )
        moe_arrays = v1_moe_arrays
    present_moe = moe_arrays.intersection(arrays)
    declared_moe = metadata.get("moe_audit_payload")
    if declared_moe == "included" and present_moe != moe_arrays:
        raise ValueError(f"{split} archive declares but lacks its complete MoE audit payload")
    if declared_moe == "omitted_non_test_storage_optimization" and present_moe:
        raise ValueError(f"{split} archive declares an omitted MoE audit payload")
    if split == "test" and "run_name" in metadata and present_moe != moe_arrays:
        raise ValueError(f"{split} RaCE-PV archive is missing its complete MoE audit payload")
    if present_moe and present_moe != moe_arrays:
        raise ValueError(f"{split} archive contains an incomplete MoE audit payload")
    if present_moe:
        experts = arrays["expert_quantiles"]
        gates = arrays["gate_weights"]
        if experts.shape != expected + (3, len(levels)):
            raise ValueError(f"{split} expert quantiles have the wrong shape")
        if np.any(~np.isfinite(experts)) or np.any(
            (experts < -1e-7) | (experts > power_upper_bound + 1e-7)
        ):
            raise ValueError(f"{split} expert quantiles are non-finite or out of bounds")
        if np.any(np.diff(experts, axis=-1) < -1e-7):
            raise ValueError(f"{split} contains crossing expert quantiles")
        if gates.shape != expected + (3,):
            raise ValueError(f"{split} gate weights have the wrong shape")
        if (
            np.any(~np.isfinite(gates))
            or np.any((gates < -1e-7) | (gates > 1.0 + 1e-7))
            or not np.allclose(gates.sum(axis=-1), 1.0, atol=1e-4)
        ):
            raise ValueError(f"{split} gate weights are not convex")
        router_logits = arrays["router_logits"]
        if router_logits.shape != expected + (3,) or np.any(~np.isfinite(router_logits)):
            raise ValueError(f"{split} router logits must be finite with shape [N,H,3]")

        def softmax(logits: np.ndarray) -> np.ndarray:
            shifted = logits - np.max(logits, axis=-1, keepdims=True)
            exponentiated = np.exp(shifted)
            return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)

        if architecture_version == 2:
            state_logits = arrays["state_logits"]
            fusion_logits = arrays["fusion_logits"]
            if state_logits.shape != expected + (3,) or np.any(~np.isfinite(state_logits)):
                raise ValueError(f"{split} state logits must be finite with shape [N,H,3]")
            if fusion_logits.shape != expected + (3,) or np.any(~np.isfinite(fusion_logits)):
                raise ValueError(f"{split} fusion logits must be finite with shape [N,H,3]")
            if not np.allclose(router_logits, state_logits, atol=0.0, rtol=0.0):
                raise ValueError(
                    f"{split} router_logits must be an exact compatibility alias of state_logits"
                )
            if not np.allclose(softmax(state_logits), probabilities, atol=2e-5):
                raise ValueError(f"{split} state probabilities do not equal softmax(state_logits)")
            if not np.allclose(softmax(fusion_logits), gates, atol=2e-5):
                raise ValueError(f"{split} fusion gate weights do not equal softmax(fusion_logits)")
            state_head_mode = metadata.get("state_head_mode", "flat")
            if state_head_mode not in {"flat", "hierarchical"}:
                raise ValueError(f"{split} has unsupported state_head_mode={state_head_mode!r}")
            hierarchical_names = {
                "ramp_presence_logits",
                "ramp_direction_logits",
            }
            present_hierarchical = hierarchical_names.intersection(arrays)
            if state_head_mode == "hierarchical" and present_hierarchical != hierarchical_names:
                missing = sorted(hierarchical_names.difference(arrays))
                raise ValueError(
                    f"{split} hierarchical archive is missing required arrays: {missing}"
                )
            if present_hierarchical and present_hierarchical != hierarchical_names:
                raise ValueError(
                    f"{split} archive contains incomplete hierarchical state diagnostics"
                )
            if present_hierarchical:
                presence_logits = arrays["ramp_presence_logits"]
                direction_logits = arrays["ramp_direction_logits"]
                if presence_logits.shape != expected or np.any(~np.isfinite(presence_logits)):
                    raise ValueError(
                        f"{split} ramp_presence_logits must be finite with shape [N,H]"
                    )
                if direction_logits.shape != expected + (2,) or np.any(
                    ~np.isfinite(direction_logits)
                ):
                    raise ValueError(
                        f"{split} ramp_direction_logits must be finite with shape [N,H,2]"
                    )
                clipped = np.clip(presence_logits, -80.0, 80.0)
                ramp_probability = 1.0 / (1.0 + np.exp(-clipped))
                direction_probability = softmax(direction_logits)
                composed = np.concatenate(
                    (
                        (1.0 - ramp_probability)[..., None],
                        ramp_probability[..., None] * direction_probability,
                    ),
                    axis=-1,
                )
                if not np.allclose(composed, probabilities, atol=2e-5):
                    raise ValueError(
                        f"{split} ramp presence probabilities and direction logits "
                        "do not compose to state_probs"
                    )
        elif not np.allclose(softmax(router_logits), probabilities, atol=2e-5):
            raise ValueError(f"{split} state probabilities do not equal softmax(router_logits)")
        fused = np.sum(gates[..., None] * experts, axis=2)
        if not np.allclose(fused, arrays["quantiles"], atol=2e-5):
            raise ValueError(f"{split} fused quantiles do not equal the shared-gate expert sum")
    daylight = arrays["daylight_mask"].astype(bool)
    if not np.all(np.isin(arrays["daylight_mask"], (0, 1))):
        raise ValueError(f"{split} daylight_mask must contain only boolean/0/1 values")
    if not np.issubdtype(arrays["state_labels"].dtype, np.integer):
        raise ValueError(f"{split} state labels must use an integer dtype")
    if not np.all(np.isin(arrays["state_labels"], (-100, 0, 1, 2))):
        raise ValueError(f"{split} state labels contain values outside the registered mapping")
    if np.any(arrays["state_labels"][~daylight] != -100):
        raise ValueError(f"{split} night labels must be -100")
    if np.any(np.abs(arrays["targets"][~daylight]) > 1e-7):
        raise ValueError(f"{split} night targets must be zero")
    if np.any(np.abs(quantiles[~daylight]) > 1e-7):
        raise ValueError(f"{split} night quantile forecasts must be zero")
    if present_moe and np.any(np.abs(arrays["expert_quantiles"][~daylight]) > 1e-7):
        raise ValueError(f"{split} night expert quantiles must be zero")
    night_probabilities = probabilities[~daylight]
    if np.any(np.abs(night_probabilities[:, 0] - 1.0) > 1e-4) or np.any(
        np.abs(night_probabilities[:, 1:]) > 1e-4
    ):
        raise ValueError(f"{split} night state probabilities must select stable")
    for name in ("timestamps", "day_ids", "site_ids"):
        values = np.asarray(arrays[name])
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{split} {name} must use an integer audit representation")
        if np.issubdtype(values.dtype, np.signedinteger) and np.any(
            values == np.iinfo(values.dtype).min
        ):
            raise ValueError(f"{split} {name} contains an invalid integer sentinel")
    expected_step_ns = 15 * 60 * 1_000_000_000
    if horizon > 1 and np.any(np.diff(arrays["timestamps"], axis=1) != expected_step_ns):
        raise ValueError(f"{split} target timestamps must advance at 15-minute intervals")
    site_values = np.asarray(arrays["site_ids"]).reshape(-1)
    for site in np.unique(site_values):
        origin_order = arrays["timestamps"][site_values == site, 0]
        if origin_order.size > 1 and np.any(np.diff(origin_order) <= 0):
            raise ValueError(f"{split} forecast origins must increase strictly within each site")
    if metadata.get("split") != split:
        raise ValueError(f"metadata split is {metadata.get('split')!r}, expected {split!r}")
    identity_names = [
        key
        for key in ("run_name", "baseline")
        if isinstance(metadata.get(key), str) and str(metadata[key]).strip()
    ]
    if len(identity_names) != 1:
        raise ValueError(f"{split} metadata must identify exactly one model run or baseline")
    if isinstance(metadata.get("seed"), bool) or not isinstance(metadata.get("seed"), int):
        raise ValueError(f"{split} metadata must contain an integer random seed")
    recorded_levels = tuple(float(value) for value in metadata.get("quantiles", ()))
    if recorded_levels != tuple(float(value) for value in levels):
        raise ValueError(f"{split} metadata quantiles differ from the evaluation config")
    site_map = metadata.get("site_id_map")
    if not isinstance(site_map, dict) or not site_map:
        raise ValueError(f"{split} metadata must contain site_id_map")
    canonical_site_map: dict[str, str] = {}
    for raw_key, raw_value in site_map.items():
        try:
            key = str(int(raw_key))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{split} site_id_map keys must be integers") from exc
        if str(raw_key) != key or not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(f"{split} site_id_map must use canonical keys and names")
        canonical_site_map[key] = raw_value
    if len(set(canonical_site_map.values())) != len(canonical_site_map):
        raise ValueError(f"{split} site_id_map names must be unique")
    observed = {str(int(value)) for value in np.unique(arrays["site_ids"])}
    if not observed.issubset(canonical_site_map):
        raise ValueError(f"{split} site_ids are not described by site_id_map")


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return value


def _resolve_reference_path_from_manifest(
    manifest_path: Path,
    *,
    expected_role: str,
    seed: int,
) -> Path:
    """Resolve the immutable selected test artifact for one evaluation seed."""

    manifest = _load_json_object(manifest_path, "reference selection manifest")
    if manifest.get("schema_version") != 3 or manifest.get("role") != expected_role:
        raise ValueError(f"reference manifest must use schema 3 with role={expected_role!r}")
    test_predictions = manifest.get("test_predictions")
    artifact = test_predictions.get(str(int(seed))) if isinstance(test_predictions, dict) else None
    if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
        raise ValueError(
            f"{expected_role} reference manifest has no selected test artifact for seed {seed}"
        )
    path = Path(artifact["path"])
    return path if path.is_absolute() else manifest_path.parent / path


def _validate_reference_selection(
    manifest_path: Path,
    reference_path: Path,
    reference_metadata: Mapping[str, Any],
    *,
    expected_role: str,
) -> dict[str, Any]:
    """Validate a validation-only baseline selection and its immutable evidence."""

    manifest = _load_json_object(manifest_path, "reference selection manifest")
    required = {
        "schema_version",
        "role",
        "selection_split",
        "selection_metric",
        "selection_direction",
        "selection_aggregation",
        "selection_seeds",
        "selected_baseline",
        "validation_results",
        "validation_results_sha256",
        "test_predictions",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"reference selection manifest is missing: {missing}")
    if manifest["schema_version"] != 3 or manifest["selection_split"] != "validation":
        raise ValueError("reference selection must use schema 3 and split='validation'")
    role = str(manifest["role"])
    if role != expected_role or role not in REFERENCE_SELECTION_METRICS:
        raise ValueError(f"reference role is {role!r}, expected {expected_role!r}")
    metric = str(manifest["selection_metric"])
    direction = str(manifest["selection_direction"])
    expected_direction = REFERENCE_SELECTION_METRICS[role].get(metric)
    if expected_direction is None or direction != expected_direction:
        raise ValueError(f"unsupported {role} reference metric/direction: {metric}/{direction}")
    if manifest["selection_aggregation"] != "equal_seed_mean":
        raise ValueError("reference selection must aggregate validation scores by equal seed mean")
    raw_seeds = manifest["selection_seeds"]
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("reference selection must contain a non-empty seed set")
    try:
        selection_seeds = sorted({int(value) for value in raw_seeds})
    except (TypeError, ValueError) as exc:
        raise ValueError("reference selection seeds must be integers") from exc
    if (
        len(selection_seeds) != len(raw_seeds)
        or [int(value) for value in raw_seeds] != selection_seeds
    ):
        raise ValueError("reference selection seeds must be unique and sorted")
    if selection_seeds != list(SEEDS):
        raise ValueError(
            f"registered reference selection requires the exact seed set {list(SEEDS)}"
        )
    selected = str(manifest["selected_baseline"])
    if reference_metadata.get("baseline") != selected:
        raise ValueError("reference metadata baseline differs from the selected baseline")
    try:
        reference_seed = int(reference_metadata["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("reference metadata must contain its selected integer seed") from exc
    if reference_seed not in selection_seeds:
        raise ValueError("reference seed is absent from the group selection seed set")
    test_predictions = manifest["test_predictions"]
    if not isinstance(test_predictions, dict) or set(test_predictions) != {
        str(seed) for seed in selection_seeds
    }:
        raise ValueError("group selection must lock one test artifact for every selected seed")
    selected_artifact = test_predictions[str(reference_seed)]
    if not isinstance(selected_artifact, dict):
        raise ValueError("selected seed test artifact must be an object")
    artifact_fields = {"path", "sha256", "metadata_sha256"}
    if not artifact_fields.issubset(selected_artifact):
        raise ValueError("selected seed test artifact is missing path or SHA256 fields")
    registered_reference_path = Path(str(selected_artifact["path"]))
    if not registered_reference_path.is_absolute():
        registered_reference_path = manifest_path.parent / registered_reference_path
    if registered_reference_path.resolve() != reference_path.resolve():
        raise ValueError("reference path differs from the selected test prediction path")
    if sha256_file(reference_path) != selected_artifact["sha256"]:
        raise ValueError("reference prediction SHA256 differs from the selection manifest")
    metadata_path = reference_path.with_suffix(".json")
    if (
        not metadata_path.is_file()
        or sha256_file(metadata_path) != selected_artifact["metadata_sha256"]
    ):
        raise ValueError("reference metadata SHA256 differs from the selection manifest")

    evidence_path = Path(str(manifest["validation_results"]))
    if not evidence_path.is_absolute():
        evidence_path = manifest_path.parent / evidence_path
    if (
        not evidence_path.is_file()
        or sha256_file(evidence_path) != manifest["validation_results_sha256"]
    ):
        raise ValueError("validation selection evidence is missing or has a different SHA256")
    evidence = _load_json_object(evidence_path, "validation selection evidence")
    if (
        evidence.get("schema_version") != 2
        or evidence.get("split") != "validation"
        or evidence.get("role") != role
        or evidence.get("metric") != metric
        or evidence.get("direction") != direction
        or evidence.get("aggregation") != "equal_seed_mean"
        or evidence.get("seed_set") != selection_seeds
        or evidence.get("selected_baseline") != selected
    ):
        raise ValueError("validation evidence role, metric, or direction differs from manifest")
    scores = evidence.get("scores")
    if not isinstance(scores, dict) or len(scores) < 2 or selected not in scores:
        raise ValueError(
            "validation evidence must score at least two candidates including selection"
        )
    numeric_scores = {str(name): float(value) for name, value in scores.items()}
    if any(not np.isfinite(value) for value in numeric_scores.values()):
        raise ValueError("validation candidate scores must be finite")
    seed_scores = evidence.get("seed_scores")
    if not isinstance(seed_scores, dict) or set(seed_scores) != set(numeric_scores):
        raise ValueError("validation evidence must contain per-seed scores for every baseline")
    expected_seed_keys = {str(seed) for seed in selection_seeds}
    for name, aggregate in numeric_scores.items():
        values = seed_scores[name]
        if not isinstance(values, dict) or set(values) != expected_seed_keys:
            raise ValueError("all baseline candidates must use the identical validation seed set")
        per_seed = np.asarray([float(values[str(seed)]) for seed in selection_seeds])
        if np.any(~np.isfinite(per_seed)) or not np.isclose(
            aggregate, np.mean(per_seed), rtol=0.0, atol=1e-12
        ):
            raise ValueError("aggregate validation score differs from equal seed mean")
    best_score = (
        min(numeric_scores.values()) if direction == "minimize" else max(numeric_scores.values())
    )
    tied_best = sorted(name for name, value in numeric_scores.items() if value == best_score)
    if not tied_best or selected != tied_best[0]:
        raise ValueError(
            "selected baseline is not the validation optimum with lexical exact-tie break"
        )
    return {
        "selected_baseline": selected,
        "role": role,
        "selection_metric": metric,
        "selection_direction": direction,
        "selection_aggregation": "equal_seed_mean",
        "selection_seeds": selection_seeds,
        "selected_seed": reference_seed,
        "selected_validation_score": numeric_scores[selected],
        "candidate_count": len(numeric_scores),
        "tied_best": tied_best,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "validation_results_path": str(evidence_path),
        "validation_results_sha256": sha256_file(evidence_path),
        "reference_prediction_sha256": sha256_file(reference_path),
        "reference_prediction_metadata_sha256": sha256_file(metadata_path),
    }


def _validate_prediction_pair(
    calibration: Mapping[str, np.ndarray],
    calibration_metadata: Mapping[str, Any],
    test: Mapping[str, np.ndarray],
    test_metadata: Mapping[str, Any],
) -> None:
    if calibration_metadata.get("seed") != test_metadata.get("seed"):
        raise ValueError("calibration and test predictions use different seeds")
    identity_keys = (
        "run_name",
        "baseline",
        "model_sha256",
        "model_state_sha256",
        "checkpoint_sha256",
        "best_checkpoint_sha256",
        "resolved_run_config_sha256",
        "config_sha256",
        "loss_weights",
        "requested_backbone",
        "actual_backbone",
        "use_clear_sky_prior",
        "router_mode",
        "fusion_mode",
        "active_experts",
        "architecture_version",
        "objective_version",
        "forecast_selection_metric",
        "state_head_semantics",
        "fusion_gate_semantics",
    )
    compared = 0
    for key in identity_keys:
        if key in calibration_metadata or key in test_metadata:
            compared += 1
            if calibration_metadata.get(key) != test_metadata.get(key):
                raise ValueError(f"calibration/test model identity differs at {key}")
    if compared == 0:
        raise ValueError("prediction metadata contains no auditable model identity")

    cal_map = {str(k): str(v) for k, v in calibration_metadata["site_id_map"].items()}
    test_map = {str(k): str(v) for k, v in test_metadata["site_id_map"].items()}
    common_names = set(cal_map.values()).intersection(test_map.values())
    for name in common_names:
        cal_ids = [int(key) for key, value in cal_map.items() if value == name]
        test_ids = [int(key) for key, value in test_map.items() if value == name]
        cal_selection = np.isin(calibration["site_ids"].reshape(-1), cal_ids)
        test_selection = np.isin(test["site_ids"].reshape(-1), test_ids)
        if np.any(cal_selection) and np.any(test_selection):
            cal_times = calibration["timestamps"][cal_selection]
            test_times = test["timestamps"][test_selection]
            if np.max(cal_times) >= np.min(test_times):
                raise ValueError(
                    f"calibration and test timestamps overlap or reverse for site {name}"
                )


def _site_day_ids(
    day_ids: np.ndarray,
    site_ids: np.ndarray,
    site_id_map: Mapping[str, Any],
) -> np.ndarray:
    """Return stable, compact site-day hashes, not dataset-dependent ranks."""

    days = np.asarray(day_ids)
    ids = np.asarray(site_ids).reshape(-1)
    if days.ndim != 2 or days.shape[0] != ids.size:
        raise ValueError("day_ids must have shape [N, H] aligned with site_ids [N]")
    output = np.empty(days.shape, dtype=np.uint64)
    seen: dict[int, str] = {}
    for site_id in np.unique(ids):
        row_mask = ids == site_id
        name = str(site_id_map[str(int(site_id))])
        encoded_name = name.encode("utf-8")
        name_token = f"{len(encoded_name)}:".encode("ascii") + encoded_name
        site_days = days[row_mask]
        unique_days, inverse = np.unique(site_days, return_inverse=True)
        hashes = []
        for raw_day in unique_days:
            day = raw_day.item() if isinstance(raw_day, np.generic) else raw_day
            identity = name_token + b"|day=" + repr(day).encode("utf-8")
            digest = hashlib.blake2b(b"racepv-site-day-v1|" + identity, digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            text = identity.decode("utf-8")
            previous = seen.get(value)
            if previous is not None and previous != text:
                raise RuntimeError("site-day hash collision; use a wider registered identity")
            seen[value] = text
            hashes.append(value)
        output[row_mask] = np.asarray(hashes, dtype=np.uint64)[inverse].reshape(site_days.shape)
    return output


def _forecast_cell_ids(timestamps: np.ndarray, step_minutes: int) -> np.ndarray:
    """Build stable target-time/lead IDs used to order hash candidates.

    A target timestamp alone is duplicated across overlapping forecast windows.
    Combining its grid slot with the lead index uniquely identifies the forecast
    cell while remaining invariant to archive row permutations.
    """

    raw = np.asarray(timestamps)
    if raw.ndim != 2:
        raise ValueError("target timestamps must have shape [N, H]")
    if step_minutes <= 0:
        raise ValueError("step_minutes must be positive")
    if np.issubdtype(raw.dtype, np.datetime64):
        values = raw.astype("datetime64[ns]").astype(np.int64)
    elif np.issubdtype(raw.dtype, np.integer):
        values = raw.astype(np.int64, copy=False)
    else:
        try:
            values = raw.astype("datetime64[ns]").astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "target timestamps must be datetime-like or epoch nanoseconds"
            ) from exc
    if np.any(values == np.iinfo(np.int64).min):
        raise ValueError("target timestamps contain NaT/iNaT values")
    step_ns = int(step_minutes) * 60 * 1_000_000_000
    slots, remainder = np.divmod(values, step_ns)
    if np.any(remainder != 0):
        raise ValueError("target timestamps are not aligned to the configured time grid")
    horizon = raw.shape[1]
    if horizon < 1:
        raise ValueError("target timestamps must contain at least one forecast lead")
    limit = np.iinfo(np.int64).max // horizon
    if np.any(np.abs(slots) > limit):
        raise OverflowError("target timestamps are too large to encode stable forecast-cell IDs")
    return slots * horizon + np.arange(horizon, dtype=np.int64)[None, :]


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _raw_intervals(
    predictions: np.ndarray,
    quantile_levels: Sequence[float],
    coverages: Sequence[float],
) -> dict[float, np.ndarray]:
    levels = np.asarray(quantile_levels, dtype=float)
    intervals: dict[float, np.ndarray] = {}
    for coverage in coverages:
        lower_level = (1.0 - coverage) / 2.0
        upper_level = 1.0 - lower_level
        lower = np.flatnonzero(np.isclose(levels, lower_level, atol=1e-8))
        upper = np.flatnonzero(np.isclose(levels, upper_level, atol=1e-8))
        if lower.size != 1 or upper.size != 1:
            raise ValueError(
                f"coverage {coverage:g} requires quantiles {lower_level:g}/{upper_level:g}"
            )
        intervals[float(coverage)] = np.stack(
            (predictions[..., int(lower[0])], predictions[..., int(upper[0])]),
            axis=-1,
        )
    return intervals


def _probability_metrics_with_valid_states(
    targets: np.ndarray,
    quantiles: np.ndarray,
    levels: Sequence[float],
    *,
    intervals: Mapping[float, np.ndarray],
    labels: np.ndarray,
    daylight: np.ndarray,
) -> dict[str, object]:
    """Score all daylight cells but condition coverage only on known states."""

    result = probabilistic_forecast_metrics(
        targets,
        quantiles,
        levels,
        intervals=dict(intervals),
        mask=daylight,
    )
    valid_state = daylight & (labels >= 0) & (labels <= 2)
    groups = np.where(labels == 0, "stable", "ramp")
    for coverage, values in intervals.items():
        array = np.asarray(values)
        result["intervals"][float(coverage)]["conditional_coverage"] = (  # type: ignore[index]
            conditional_coverage_error(
                targets,
                array[..., 0],
                array[..., 1],
                groups,
                coverage=float(coverage),
                mask=valid_state,
            )
        )
    return result


def _cqr_acceptance_report(
    raw_probability: Mapping[str, Any],
    conditional_probability: Mapping[str, Any],
    calibration_rows: Sequence[Mapping[str, Any]],
    *,
    coverages: Sequence[float],
    min_group_size: int,
    coverage_tolerance: float = 0.03,
    maximum_state_gap: float = 0.05,
) -> dict[str, object]:
    """Fail-closed gate for treating static conditional CQR as more than a diagnostic."""

    if min_group_size < 1:
        raise ValueError("min_group_size must be positive")
    if coverage_tolerance < 0 or maximum_state_gap < 0:
        raise ValueError("CQR acceptance tolerances cannot be negative")

    raw_intervals = raw_probability.get("intervals")
    conditional_intervals = conditional_probability.get("intervals")
    if not isinstance(raw_intervals, Mapping) or not isinstance(conditional_intervals, Mapping):
        raise ValueError("raw and conditional probability summaries require intervals")

    def coverage_row(rows: Mapping[Any, Any], coverage: float) -> Mapping[str, Any] | None:
        for key, value in rows.items():
            try:
                matches = bool(np.isclose(float(key), coverage, atol=1e-10))
            except (TypeError, ValueError):
                matches = False
            if matches and isinstance(value, Mapping):
                return value
        return None

    failures: list[str] = []
    criteria: dict[str, object] = {}
    for raw_coverage in coverages:
        coverage = float(raw_coverage)
        label = f"{coverage:g}"
        raw_row = coverage_row(raw_intervals, coverage)
        conditional_row = coverage_row(conditional_intervals, coverage)
        if raw_row is None or conditional_row is None:
            failures.append(f"{label} interval metrics are missing")
            continue
        conditional_coverage = conditional_row.get("conditional_coverage")
        groups = (
            conditional_coverage.get("groups")
            if isinstance(conditional_coverage, Mapping)
            else None
        )
        if not isinstance(groups, Mapping):
            failures.append(f"{label} stable/ramp conditional coverage is missing")
            continue

        observed: dict[str, float] = {}
        for name, value in (
            ("overall", conditional_row.get("picp")),
            (
                "stable",
                groups.get("stable", {}).get("coverage")
                if isinstance(groups.get("stable"), Mapping)
                else None,
            ),
            (
                "ramp",
                groups.get("ramp", {}).get("coverage")
                if isinstance(groups.get("ramp"), Mapping)
                else None,
            ),
        ):
            try:
                observed_value = float(value)
            except (TypeError, ValueError):
                observed_value = float("nan")
            observed[name] = observed_value
            if not np.isfinite(observed_value) or abs(observed_value - coverage) > (
                coverage_tolerance + 1e-12
            ):
                failures.append(
                    f"{label} {name} PICP={observed_value!r} is outside "
                    f"target ±{coverage_tolerance:g}"
                )
        state_gap = abs(observed["stable"] - observed["ramp"])
        if not np.isfinite(state_gap) or state_gap > maximum_state_gap + 1e-12:
            failures.append(
                f"{label} stable/ramp coverage gap={state_gap!r} exceeds {maximum_state_gap:g}"
            )

        try:
            raw_score = float(raw_row.get("interval_score"))
            conditional_score = float(conditional_row.get("interval_score"))
        except (TypeError, ValueError):
            raw_score = conditional_score = float("nan")
        if (
            not np.isfinite(raw_score)
            or not np.isfinite(conditional_score)
            or conditional_score > raw_score + 1e-12
        ):
            failures.append(
                f"{label} conditional interval score={conditional_score!r} is worse "
                f"than raw={raw_score!r}"
            )
        criteria[label] = {
            "target": coverage,
            "overall_picp": observed["overall"],
            "stable_picp": observed["stable"],
            "ramp_picp": observed["ramp"],
            "stable_ramp_gap": state_gap,
            "raw_interval_score": raw_score,
            "conditional_interval_score": conditional_score,
        }

        coverage_calibration_rows = [
            row
            for row in calibration_rows
            if np.isclose(float(row.get("coverage", np.nan)), coverage, atol=1e-10)
        ]
        for state in ("stable", "ramp"):
            state_rows = [
                row for row in coverage_calibration_rows if row.get("state_group") == state
            ]
            if not state_rows:
                failures.append(f"{label} calibration rows are missing state group {state}")
                continue
            for row in state_rows:
                band = str(row.get("lead_band", "unknown"))
                try:
                    correction = float(row.get("correction"))
                except (TypeError, ValueError):
                    correction = float("nan")
                if not np.isfinite(correction):
                    failures.append(
                        f"{label} {state}/{band} has non-finite correction={correction!r}"
                    )
                try:
                    group_n_scores = int(row.get("group_n_scores", -1))
                except (TypeError, ValueError):
                    group_n_scores = -1
                if group_n_scores < min_group_size:
                    failures.append(
                        f"{label} {state}/{band} group_n_scores={group_n_scores} is below "
                        f"{min_group_size}"
                    )
                if bool(row.get("used_fallback")):
                    failures.append(f"{label} {state}/{band} used an all-state fallback correction")

    accepted = not failures
    return {
        "accepted": accepted,
        "diagnostic_only": not accepted,
        "failure_reasons": failures,
        "criteria_by_coverage": criteria,
        "coverage_tolerance": coverage_tolerance,
        "maximum_stable_ramp_gap": maximum_state_gap,
        "minimum_group_scores": min_group_size,
    }


def _dataset_array_batch(
    dataset: WindowDataset,
    name: str,
    start: int,
    stop: int,
) -> np.ndarray:
    """Read one dataset array slice without materializing a lazy global array."""

    if hasattr(dataset, "read_model_window_batch"):
        return read_model_window_batch(dataset, slice(start, stop), [name])[name]
    # Preserve the existing integer-only legacy array adapter.
    values = dataset.arrays[name]
    if isinstance(values, np.ndarray):
        return np.asarray(values[start:stop])
    return np.stack([np.asarray(values[index]) for index in range(start, stop)], axis=0)


def _smart_persistence(
    dataset: WindowDataset,
    *,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    required = ("history_values", "history_mask", "future_known")
    missing = [name for name in required if name not in dataset.arrays]
    if missing:
        raise KeyError(f"test windows are missing arrays: {missing}")
    history_shape = dataset.arrays["history_values"].shape
    future_shape = dataset.arrays["future_known"].shape
    if history_shape[-1] < 2 or future_shape[-1] < 1:
        raise ValueError("prepared features do not contain registered power/clear-sky columns")
    smart = np.empty((len(dataset), future_shape[1]), dtype=np.float32)
    clear_sky = np.empty_like(smart)
    smart_model = SmartPersistence()
    clear_sky_model = ClearSkyPersistence()
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        history = _dataset_array_batch(dataset, "history_values", start, stop)
        history_mask = _dataset_array_batch(dataset, "history_mask", start, stop).astype(bool)
        future = _dataset_array_batch(dataset, "future_known", start, stop)
        valid = history_mask[..., 0] & history_mask[..., 1]
        smart[start:stop] = smart_model.predict(
            history[..., 0], history[..., 1], future[..., 0], valid
        )
        clear_sky[start:stop] = clear_sky_model.predict(future[..., 0])
    return smart, clear_sky


def _validate_source_windows(
    path: Path,
    predictions: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    label: str,
    batch_size: int = 512,
    allow_development_data: bool = False,
    expected_version=None,
    expected_profile=None,
) -> WindowDataset:
    if (
        expected_version is not None
        or expected_profile is not None
        or _prepared_metadata_claim(metadata)
    ):
        _admit_v2_prediction_source(
            path,
            metadata,
            expected_version=expected_version,
            expected_profile=expected_profile,
            allow_development_data=allow_development_data,
        )
    expectations = (
        {}
        if expected_version is None and expected_profile is None
        else {"expected_version": expected_version, "expected_profile": expected_profile}
    )
    dataset = load_window_dataset(path, verify_content=True, **expectations)
    dataset.require_registered_schema()
    dataset.require_registered_evidence(allow_ineligible=allow_development_data)
    if dataset.schema is None or {
        str(key): str(value) for key, value in dataset.schema["site_id_map"].items()
    } != {str(key): str(value) for key, value in metadata["site_id_map"].items()}:
        raise ValueError(f"{label} window site map differs from prediction metadata")
    content_identity = getattr(dataset, "content_identity", None)
    expected_content = metadata.get("source_window_content_sha256")
    if expected_content is not None:
        if content_identity != expected_content:
            raise ValueError(f"{label} window content identity differs from prediction metadata")
    else:
        container = path / "store.json" if path.is_dir() else path
        if sha256_file(container) != metadata.get("source_npz_sha256"):
            raise ValueError(f"{label} window checksum differs from prediction metadata")
    if len(dataset) != predictions["targets"].shape[0]:
        raise ValueError(f"{label} windows and predictions have different sample counts")
    for name in (
        "targets",
        "state_labels",
        "daylight_mask",
        "timestamps",
        "day_ids",
        "site_ids",
    ):
        if name not in dataset.arrays:
            raise ValueError(f"{label} window dataset is missing audit array {name}")
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            actual = _dataset_array_batch(dataset, name, start, stop)
            expected = np.asarray(predictions[name][start:stop])
            if name == "site_ids":
                actual = actual.reshape(-1)
                expected = expected.reshape(-1)
            if not np.array_equal(actual, expected):
                raise ValueError(
                    f"{label} windows and prediction archive differ at {name} "
                    f"in rows [{start}, {stop})"
                )
    return dataset


def _admit_v2_prediction_source(
    path, metadata, *, expected_version, expected_profile, allow_development_data=False
):
    """Verify metadata and source graph before prediction arrays may be opened."""
    from racepv.baselines.extension_registry import validate_prepared_identity
    from racepv.data.prepared_contract import (
        V2_WINDOW_SEQUENCE_STORE_FORMAT,
        require_origin_safe_history_contract,
        validate_v2_metadata,
    )
    from racepv.data.weather import expected_origin_contract

    expected = expected_origin_contract(expected_version, expected_profile)
    if expected is None or allow_development_data:
        raise ValueError(
            "prepared v2 evaluation requires explicit expectations without development bypass"
        )
    identity = validate_prepared_identity(metadata.get("prepared_contract"))
    descriptor = (
        path / "store.json"
        if path.is_dir()
        else path
        if path.suffix == ".json"
        else path.with_suffix(".descriptor.json")
    )
    actual = require_origin_safe_history_contract(
        descriptor, version=expected_version, profile=expected_profile
    )
    source = validate_v2_metadata(descriptor)
    schema = source["schema"]
    if source["descriptor"]["format"] == V2_WINDOW_SEQUENCE_STORE_FORMAT:
        # Match the accepted stored-reader projection without loading any arrays.
        selected = [
            view for view in source["views"] if view["schema"]["split"] == identity["split"]
        ]
        if len(selected) != 1:
            raise ValueError("prepared prediction source selection identity differs")
        actual["split"] = identity["split"]
        actual["selected_forecast_population_sha256"] = selected[0]["schema"][
            "forecast_population_sha256"
        ]
        schema = selected[0]["schema"]
    actual["clock"] = dict(schema["clock"])
    actual["mask_semantics"] = {
        name: schema["causal_rules"][name]
        for name in ("missing_values", "imputed_values", "provenance_flag_mask")
    }
    # A descriptor proves bytes, not array semantics. Require the prediction's
    # prior semantic receipt, compare every other canonical identity field,
    # and leave the independent source-scientific qualification check below.
    if identity["array_semantics_verified"] is not True:
        raise ValueError("prepared prediction identity requires verified array semantics")
    proof_field = "array_semantics_verified"
    if {key: value for key, value in identity.items() if key != proof_field} != {
        key: value for key, value in actual.items() if key != proof_field
    }:
        raise ValueError("prepared prediction source identity differs")
    if metadata.get("source_window_content_sha256") != actual["aggregate_content_sha256"]:
        raise ValueError("prepared prediction requires exact complete content identity")
    # The accepted source authority has no scientific qualification receipt yet.
    # Metadata flags cannot substitute for that independent proof.
    if (
        actual["source_interpretation_verified"] is not True
        or actual["registered_evidence_eligible"] is not True
    ):
        raise ValueError("prepared prediction source interpretation/evidence is not qualified")
    return identity


def _scoped_metrics(
    targets: np.ndarray,
    median: np.ndarray,
    daylight: np.ndarray,
    reference: np.ndarray | None,
) -> dict[str, object]:
    output: dict[str, object] = {
        "overall": point_forecast_metrics(
            targets,
            median,
            daylight_mask=daylight,
            reference_pred=reference,
        ),
        "reported_leads": {},
        "lead_bands": {},
    }
    for lead in REPORT_LEADS:
        if lead > targets.shape[1]:
            continue
        selection = slice(lead - 1, lead)
        output["reported_leads"][f"{lead * 15}min"] = point_forecast_metrics(  # type: ignore[index]
            targets[:, selection],
            median[:, selection],
            daylight_mask=daylight[:, selection],
            reference_pred=None if reference is None else reference[:, selection],
        )
    for name, start, end in LEAD_BANDS:
        if start >= targets.shape[1]:
            continue
        selection = slice(start, min(end, targets.shape[1]))
        output["lead_bands"][name] = point_forecast_metrics(  # type: ignore[index]
            targets[:, selection],
            median[:, selection],
            daylight_mask=daylight[:, selection],
            reference_pred=None if reference is None else reference[:, selection],
        )
    return output


def _forecast_origin_timestamps(
    timestamps: np.ndarray,
    *,
    n_origins: int,
    horizon: int,
    step_minutes: int,
    contract=None,
) -> tuple[np.ndarray, str]:
    """Return epoch-nanosecond forecast origins from origins or target times.

    Legacy archives use ``timestamps[:, 0] - step``; explicit v2 contracts use
    the shared interval-start label clock. A one-dimensional array is accepted
    as an already explicit forecast-origin time axis for focused tests and reuse.
    """

    raw = np.asarray(timestamps)
    if contract is not None:
        from racepv.data.weather import forecast_origin_timestamps

        # Validate the contract even for an already explicit one-dimensional axis.
        forecast_origin_timestamps(np.empty((0, 1), dtype=np.int64), contract=contract)
        if step_minutes != 15:
            raise ValueError("prepared v2 clock requires 15 minutes")
    if step_minutes <= 0:
        raise ValueError("step_minutes must be positive")
    step_ns = int(step_minutes) * 60 * 1_000_000_000
    if np.issubdtype(raw.dtype, np.datetime64):
        values = raw.astype("datetime64[ns]").astype(np.int64)
    elif np.issubdtype(raw.dtype, np.integer):
        values = raw.astype(np.int64, copy=False)
    else:
        try:
            values = raw.astype("datetime64[ns]").astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError("DM timestamps must be datetime-like or epoch nanoseconds") from exc
    if np.any(values == np.iinfo(np.int64).min):
        raise ValueError("DM timestamps contain NaT/iNaT values")

    if values.ndim == 1:
        if values.shape[0] != n_origins:
            raise ValueError("forecast-origin timestamps must have one value per loss row")
        origins = values
        timestamp_input = "forecast_origin_timestamps"
    elif values.ndim == 2:
        if values.shape != (n_origins, horizon):
            raise ValueError(f"target timestamps must have shape [{n_origins},{horizon}] for DM")
        if horizon > 1 and np.any(np.diff(values, axis=1) != step_ns):
            raise ValueError("DM target timestamps must advance by the registered step")
        if np.any(values[:, 0] < np.iinfo(np.int64).min + step_ns):
            raise OverflowError("cannot derive forecast origins from target timestamps")
        if contract is not None:
            origins = forecast_origin_timestamps(values, contract=contract)
        else:
            origins = values[:, 0] - step_ns
        timestamp_input = "target_timestamps_derived_to_forecast_origins"
    else:
        raise ValueError("DM timestamps must have shape [N] or [N,H]")
    if np.any(np.mod(origins, step_ns) != 0):
        raise ValueError("DM forecast origins are not aligned to the registered time grid")
    return np.asarray(origins, dtype=np.int64), timestamp_input


def _infer_origin_step_minutes(
    timestamps: np.ndarray,
    site_ids: np.ndarray | None,
    *,
    target_step_minutes: int,
    contract=None,
) -> int:
    """Infer the fundamental forecast-origin cadence from within-site differences."""

    raw = np.asarray(timestamps)
    if raw.ndim not in {1, 2} or raw.shape[0] < 2:
        raise ValueError("origin-step inference requires at least two forecast origins")
    horizon = 1 if raw.ndim == 1 else raw.shape[1]
    origins, _ = _forecast_origin_timestamps(
        raw,
        n_origins=raw.shape[0],
        horizon=horizon,
        step_minutes=target_step_minutes,
        contract=contract,
    )
    sites = (
        np.zeros(raw.shape[0], dtype=int) if site_ids is None else np.asarray(site_ids).reshape(-1)
    )
    if sites.shape[0] != raw.shape[0]:
        raise ValueError("site_ids must provide one value per forecast origin")
    minute_ns = 60 * 1_000_000_000
    differences: list[np.ndarray] = []
    for site in np.unique(sites):
        selected = np.sort(origins[sites == site])
        if selected.size > 1:
            delta = np.diff(selected)
            if np.any(delta <= 0) or np.any(np.mod(delta, minute_ns) != 0):
                raise ValueError("forecast-origin differences must be positive whole minutes")
            differences.append(delta // minute_ns)
    if not differences:
        raise ValueError("origin-step inference has no within-site timestamp differences")
    step = int(np.gcd.reduce(np.concatenate(differences)))
    if step < target_step_minutes or step % target_step_minutes:
        raise ValueError("inferred origin step is incompatible with the target grid")
    return step


def _time_aware_diebold_mariano_test(
    model_loss: np.ndarray,
    reference_loss: np.ndarray,
    timestamps: np.ndarray,
    *,
    horizon: int,
    lag: int | None = None,
    step_minutes: int = 15,
    origin_step_minutes: int | None = None,
    contract=None,
) -> tuple[Any, dict[str, object]]:
    """Run DM/HAC without treating opposite sides of a time gap as adjacent.

    The loss rows are ordered by forecast issue time.  Every non-``step_minutes``
    jump starts a new continuous segment.  ``selected_lag`` missing positions are
    inserted between segments before the existing Newey--West implementation is
    called, which prevents covariance pairs at *all* registered lags from crossing
    a gap while retaining the protocol's ``H-1`` lag inside continuous segments.
    """

    first, second = np.broadcast_arrays(
        np.asarray(model_loss, dtype=float), np.asarray(reference_loss, dtype=float)
    )
    if first.ndim == 0:
        raise ValueError("DM loss arrays must contain a forecast-origin dimension")
    n_origins = first.shape[0]
    if horizon < 1:
        raise ValueError("horizon must be positive")
    origins, timestamp_input = _forecast_origin_timestamps(
        timestamps,
        n_origins=n_origins,
        horizon=horizon,
        step_minutes=step_minutes,
        contract=contract,
    )
    selected_origin_step = (
        _infer_origin_step_minutes(
            timestamps,
            np.zeros(n_origins, dtype=int),
            target_step_minutes=step_minutes,
            contract=contract,
        )
        if origin_step_minutes is None
        else int(origin_step_minutes)
    )
    if selected_origin_step < step_minutes or selected_origin_step % step_minutes:
        raise ValueError("origin_step_minutes must be a target-step multiple")
    requested_lag = (
        max(0, int(np.ceil(horizon * step_minutes / selected_origin_step)) - 1)
        if lag is None
        else int(lag)
    )
    if requested_lag < 0:
        raise ValueError("lag cannot be negative")
    selected_lag = min(requested_lag, max(0, n_origins - 1))
    order = np.argsort(origins, kind="stable")
    ordered_origins = origins[order]
    if ordered_origins.size > 1 and np.any(np.diff(ordered_origins) <= 0):
        raise ValueError("DM forecast origins must be unique")
    ordered_first = first[order]
    ordered_second = second[order]
    step_ns = selected_origin_step * 60 * 1_000_000_000
    split_points = (
        np.flatnonzero(np.diff(ordered_origins) != step_ns) + 1
        if ordered_origins.size > 1
        else np.asarray([], dtype=int)
    )
    segment_bounds = np.r_[0, split_points, n_origins]
    segment_sizes = np.diff(segment_bounds)

    if split_points.size and selected_lag > 0:
        first_parts: list[np.ndarray] = []
        second_parts: list[np.ndarray] = []
        separator_shape = (selected_lag,) + first.shape[1:]
        separator = np.full(separator_shape, np.nan, dtype=float)
        for segment_index, (start, stop) in enumerate(
            zip(segment_bounds[:-1], segment_bounds[1:], strict=True)
        ):
            if segment_index:
                first_parts.append(separator)
                second_parts.append(separator.copy())
            first_parts.append(ordered_first[start:stop])
            second_parts.append(ordered_second[start:stop])
        ordered_first = np.concatenate(first_parts, axis=0)
        ordered_second = np.concatenate(second_parts, axis=0)

    result = diebold_mariano_test(
        ordered_first,
        ordered_second,
        horizon=horizon,
        lag=selected_lag,
    )
    paired = np.isfinite(first - second)
    if paired.ndim > 1:
        paired = np.any(paired.reshape(n_origins, -1), axis=1)
    paired = paired[order]
    effective_segments = sum(
        bool(np.any(paired[start:stop]))
        for start, stop in zip(segment_bounds[:-1], segment_bounds[1:], strict=True)
    )
    audit = {
        "hac_time_axis": "forecast_origin",
        "first_forecast_origin_ns": int(ordered_origins[0]),
        "last_forecast_origin_ns": int(ordered_origins[-1]),
        "timestamp_input": timestamp_input,
        "expected_origin_step_minutes": selected_origin_step,
        "target_step_minutes": int(step_minutes),
        "requested_hac_lag": requested_lag,
        "origin_segment_count": int(segment_sizes.size),
        "effective_origin_segment_count": int(effective_segments),
        "origin_gap_count": int(split_points.size),
        "longest_contiguous_origin_segment": int(np.max(segment_sizes)),
        "gap_covariance_policy": "no Newey-West covariance pair crosses a time gap",
    }
    return result, audit


def _dm_report(
    targets: np.ndarray,
    median: np.ndarray,
    reference: np.ndarray,
    daylight: np.ndarray,
    site_ids: np.ndarray,
    timestamps: np.ndarray,
    site_id_map: Mapping[str, Any],
    reference_name: str,
    *,
    step_minutes: int = 15,
    contract=None,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    sites = np.asarray(site_ids).reshape(-1)
    origin_step_minutes = _infer_origin_step_minutes(
        timestamps,
        sites,
        target_step_minutes=step_minutes,
        contract=contract,
    )
    registered_lag = max(0, int(np.ceil(targets.shape[1] * step_minutes / origin_step_minutes)) - 1)
    for site in np.unique(sites):
        selected = sites == site
        for lead in REPORT_LEADS:
            if lead > targets.shape[1]:
                continue
            index = lead - 1
            model_loss = np.where(
                daylight[selected, index],
                (median[selected, index] - targets[selected, index]) ** 2,
                np.nan,
            )
            reference_loss = np.where(
                daylight[selected, index],
                (reference[selected, index] - targets[selected, index]) ** 2,
                np.nan,
            )
            try:
                result, temporal_audit = _time_aware_diebold_mariano_test(
                    model_loss,
                    reference_loss,
                    timestamps[selected],
                    horizon=targets.shape[1],
                    step_minutes=step_minutes,
                    origin_step_minutes=origin_step_minutes,
                    contract=contract,
                )
            except ValueError:
                continue
            rows.append(
                {
                    "site_id": int(site),
                    "site_name": str(site_id_map.get(str(int(site)), site)),
                    "lead_minutes": lead * 15,
                    "statistic": result.statistic,
                    "pvalue": result.pvalue,
                    "mean_loss_difference": result.mean_loss_difference,
                    "lag": result.lag,
                    "n_observations": result.n_observations,
                    **temporal_audit,
                }
            )
    if not rows:
        return {
            "loss": "squared_error",
            "rows": [],
            "reason": "insufficient observations",
            "newey_west_lag": registered_lag,
            "hac_time_axis": "forecast_origin",
            "expected_origin_step_minutes": origin_step_minutes,
            "origin_gap_policy": (
                "forecast origins are partitioned at every non-registered-step jump; "
                "Newey-West covariance is estimated only within continuous segments"
            ),
        }
    adjusted, rejected = benjamini_hochberg([float(row["pvalue"]) for row in rows])
    for row, value, decision in zip(rows, adjusted, rejected, strict=True):
        row["bh_adjusted_pvalue"] = float(value)
        row["reject_equal_accuracy_at_0.05"] = bool(decision)
        row["evaluated_model_better"] = bool(float(row["mean_loss_difference"]) < 0)
    return {
        "loss": "squared_error",
        "reference": reference_name,
        "comparison_family": f"all registered site-by-lead comparisons against {reference_name}",
        "newey_west_lag": registered_lag,
        "hac_time_axis": "forecast_origin",
        "expected_origin_step_minutes": origin_step_minutes,
        "origin_gap_policy": (
            "forecast origins are partitioned at every non-registered-step jump; "
            "Newey-West covariance is estimated only within continuous segments"
        ),
        "rows": rows,
        "significantly_better_fraction": float(
            np.mean(
                [
                    bool(row["reject_equal_accuracy_at_0.05"] and row["evaluated_model_better"])
                    for row in rows
                ]
            )
        ),
    }


def _segmented_event_metrics(
    times: np.ndarray,
    true_states: np.ndarray,
    predicted_states: np.ndarray,
    true_power: np.ndarray,
    predicted_power: np.ndarray,
) -> dict[str, object]:
    order = np.argsort(times, kind="stable")
    times = times[order].astype("datetime64[ns]")
    true_states = true_states[order]
    predicted_states = predicted_states[order]
    true_power = true_power[order]
    predicted_power = predicted_power[order]
    valid = (true_states >= 0) & (true_states <= 2)
    elapsed = np.diff(times.astype(np.int64))
    expected = 15 * 60 * 1_000_000_000
    starts: list[int] = []
    stops: list[int] = []
    active_start: int | None = None
    for index in range(len(times)):
        contiguous = index == 0 or elapsed[index - 1] == expected
        if valid[index] and (active_start is None or contiguous):
            if active_start is None:
                active_start = index
            continue
        if active_start is not None:
            starts.append(active_start)
            stops.append(index)
            active_start = index if valid[index] else None
    if active_start is not None:
        starts.append(active_start)
        stops.append(len(times))
    segments = []
    for start, stop in zip(starts, stops, strict=True):
        row = ramp_event_errors(
            true_states[start:stop],
            predicted_states[start:stop],
            y_true_power=true_power[start:stop],
            y_pred_power=predicted_power[start:stop],
            step_minutes=15,
        )
        segments.append(
            {
                "start": str(times[start]),
                "end": str(times[stop - 1]),
                **row,
            }
        )
    summary: dict[str, object] = {
        key: sum(int(row[key]) for row in segments)
        for key in (
            "true_events",
            "predicted_events",
            "matched_events",
            "false_alarms",
            "missed_events",
        )
    }
    for key in (
        "mean_signed_timing_error_minutes",
        "mean_absolute_timing_error_minutes",
        "amplitude_mae",
    ):
        weighted = [
            (float(row[key]), int(row["matched_events"]))
            for row in segments
            if np.isfinite(float(row[key])) and int(row["matched_events"]) > 0
        ]
        summary[key] = (
            float(
                sum(value * weight for value, weight in weighted)
                / sum(weight for _, weight in weighted)
            )
            if weighted
            else float("nan")
        )
    return {"summary": summary, "contiguous_segments": segments}


def _event_report(
    targets: np.ndarray,
    labels: np.ndarray,
    median: np.ndarray,
    state_probs: np.ndarray,
    timestamps: np.ndarray | None,
    site_ids: np.ndarray | None,
    site_id_map: Mapping[str, Any],
) -> dict[str, object]:
    if timestamps is None or timestamps.shape != targets.shape:
        return {"available": False, "reason": "target timestamps were not saved"}
    sites = np.zeros(targets.shape[0], dtype=int) if site_ids is None else site_ids.reshape(-1)
    rows = []
    for site in np.unique(sites):
        selected = np.flatnonzero(sites == site)
        origin_order = np.argsort(timestamps[selected, 0], kind="stable")
        selected = selected[origin_order]
        unique_time = timestamps[selected, 0]
        true_states = labels[selected, 0]
        if not np.any((true_states >= 0) & (true_states <= 2)):
            continue
        predicted_states = np.argmax(state_probs[selected, 0], axis=-1)
        true_power = targets[selected, 0]
        predicted_power = median[selected, 0]
        shortest = _segmented_event_metrics(
            unique_time,
            true_states,
            predicted_states,
            true_power,
            predicted_power,
        )
        by_lead = {}
        for lead in REPORT_LEADS:
            if lead > targets.shape[1]:
                continue
            index = lead - 1
            by_lead[f"{lead * 15}min"] = _segmented_event_metrics(
                timestamps[selected, index],
                labels[selected, index],
                np.argmax(state_probs[selected, index], axis=-1),
                targets[selected, index],
                median[selected, index],
            )
        rows.append(
            {
                "site_id": int(site),
                "site_name": str(site_id_map.get(str(int(site)), site)),
                "forecast_lead_policy": (
                    "fixed 15-minute lead at every forecast origin; longer-lead tail "
                    "cells are not appended"
                ),
                "shortest_lead_sequence": shortest,
                "by_reported_lead": by_lead,
            }
        )
    return {"available": bool(rows), "sites": rows}


def _event_daily_block_bootstrap_report(
    targets: np.ndarray,
    labels: np.ndarray,
    median: np.ndarray,
    state_probs: np.ndarray,
    timestamps: np.ndarray,
    site_ids: np.ndarray | None,
    site_id_map: Mapping[str, Any],
    *,
    n_bootstrap: int,
    seed: int,
    step_minutes: int = 15,
    origin_step_minutes: int | None = None,
    minimum_finite_fraction: float = 0.95,
    maximum_candidate_blocks: int = 512,
    contract=None,
) -> dict[str, object]:
    """Percentile CIs for matched-event errors using one-day moving blocks.

    Candidate blocks contain a contiguous day of *forecast origins*.  Event matching
    is evaluated at the fixed first lead inside every block with the same greedy,
    same-direction matcher used by :func:`_segmented_event_metrics`.  Bootstrap
    replicates combine block summaries with matched-event weights; they are not means
    of classification metrics and are not described as pooled cell estimates.
    """

    if n_bootstrap < 1:
        raise ValueError("bootstrap replicate count must be positive")
    if step_minutes != 15:
        raise ValueError("registered event bootstrap requires the 15-minute target grid")
    if not 0.0 < minimum_finite_fraction <= 1.0:
        raise ValueError("minimum_finite_fraction must be in (0, 1]")
    if maximum_candidate_blocks < 2:
        raise ValueError("maximum_candidate_blocks must be at least two")
    if targets.shape != labels.shape or median.shape != targets.shape:
        raise ValueError("event targets, labels, and median must have equal [N,H] shapes")
    if state_probs.shape != labels.shape + (3,) or timestamps.shape != labels.shape:
        raise ValueError("event probabilities and timestamps have inconsistent shapes")
    if targets.ndim != 2 or targets.shape[0] == 0:
        raise ValueError("event arrays must be non-empty [N,H] arrays")

    origins, timestamp_input = _forecast_origin_timestamps(
        timestamps,
        n_origins=targets.shape[0],
        horizon=targets.shape[1],
        step_minutes=step_minutes,
        contract=contract,
    )
    sites = (
        np.zeros(targets.shape[0], dtype=int)
        if site_ids is None
        else np.asarray(site_ids).reshape(-1)
    )
    if sites.shape[0] != targets.shape[0]:
        raise ValueError("site_ids must have one value per forecast origin")
    selected_origin_step = (
        _infer_origin_step_minutes(
            timestamps,
            sites,
            target_step_minutes=step_minutes,
            contract=contract,
        )
        if origin_step_minutes is None
        else int(origin_step_minutes)
    )
    if selected_origin_step <= 0 or 1440 % selected_origin_step:
        raise ValueError("origin_step_minutes must divide one day")

    metric_descriptions = {
        "mean_signed_timing_error_minutes": (
            "signed start-time error in minutes over greedily matched same-direction events"
        ),
        "mean_absolute_timing_error_minutes": (
            "absolute start-time error in minutes over greedily matched same-direction events"
        ),
        "amplitude_mae": (
            "matched-event-count-weighted segment amplitude MAE returned by the current "
            "greedy same-direction matcher"
        ),
    }
    block_length = 1440 // selected_origin_step
    expected_step_ns = selected_origin_step * 60 * 1_000_000_000
    minimum_replicates = int(np.ceil(minimum_finite_fraction * n_bootstrap))
    predicted_states = np.argmax(state_probs, axis=-1)
    rows: list[dict[str, object]] = []

    for offset, site in enumerate(np.unique(sites)):
        indices = np.flatnonzero(sites == site)
        order = np.argsort(origins[indices], kind="stable")
        indices = indices[order]
        site_origins = origins[indices]
        if site_origins.size > 1 and np.any(np.diff(site_origins) <= 0):
            raise ValueError(f"event bootstrap requires unique forecast origins within site {site}")
        origin_differences = np.diff(site_origins)
        gap_locations = np.flatnonzero(origin_differences != expected_step_ns)
        segment_bounds = np.r_[0, gap_locations + 1, len(indices)]
        segment_lengths = np.diff(segment_bounds)

        valid_starts = np.asarray(
            [
                start
                for start in range(max(0, len(indices) - block_length + 1))
                if np.all(origin_differences[start : start + block_length - 1] == expected_step_ns)
            ],
            dtype=int,
        )
        total_candidate_blocks = int(valid_starts.size)
        if valid_starts.size > maximum_candidate_blocks:
            positions = np.linspace(0, valid_starts.size - 1, maximum_candidate_blocks, dtype=int)
            valid_starts = valid_starts[np.unique(positions)]

        temporal_audit = {
            "time_axis": "forecast_origin",
            "first_forecast_origin_ns": int(site_origins[0]),
            "last_forecast_origin_ns": int(site_origins[-1]),
            "timestamp_input": timestamp_input,
            "expected_origin_step_minutes": selected_origin_step,
            "target_step_minutes": int(step_minutes),
            "origins_per_day": int(block_length),
            "origin_count": int(site_origins.size),
            "origin_segment_count": int(segment_lengths.size),
            "origin_gap_count": int(gap_locations.size),
            "longest_contiguous_origin_segment": (
                int(np.max(segment_lengths)) if segment_lengths.size else 0
            ),
            "block_gap_policy": (
                f"a candidate 24-hour block is eligible only when all {block_length} "
                "consecutive forecast-origin steps are present; no block crosses a "
                "time gap"
            ),
        }
        blocks_per_replicate = int(np.ceil(len(indices) / block_length))
        draws = (
            np.random.default_rng(seed + offset).integers(
                0,
                len(valid_starts),
                size=(n_bootstrap, blocks_per_replicate),
            )
            if valid_starts.size
            else None
        )

        def lead_report(
            lead_index: int,
            site_indices: np.ndarray,
            block_starts: np.ndarray,
            block_draws: np.ndarray | None,
        ) -> dict[str, object]:
            lead_minutes = (lead_index + 1) * step_minutes
            full_summary = _segmented_event_metrics(
                timestamps[site_indices, lead_index],
                labels[site_indices, lead_index],
                predicted_states[site_indices, lead_index],
                targets[site_indices, lead_index],
                median[site_indices, lead_index],
            )["summary"]
            point_matched_events = int(full_summary["matched_events"])
            metric_rows = [
                {
                    "metric": name,
                    "estimate": float(full_summary[name]),
                    "estimand": (
                        f"site-specific {description}; point estimate is computed on "
                        f"the complete fixed-{lead_minutes}-minute-lead sequence after "
                        "splitting every forecast-origin time gap; percentile replicates "
                        "resample contiguous 24-hour origin blocks and combine current-"
                        "matcher block summaries using matched-event counts"
                    ),
                    "point_matched_events": point_matched_events,
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "finite_replicates": 0,
                    "minimum_required_finite_replicates": minimum_replicates,
                    "ci_available": False,
                }
                for name, description in metric_descriptions.items()
            ]
            if block_draws is None:
                return {
                    "lead_minutes": lead_minutes,
                    "available": False,
                    "metrics": metric_rows,
                }

            block_summaries = []
            for start in block_starts:
                block_indices = site_indices[start : start + block_length]
                block_summaries.append(
                    _segmented_event_metrics(
                        timestamps[block_indices, lead_index],
                        labels[block_indices, lead_index],
                        predicted_states[block_indices, lead_index],
                        targets[block_indices, lead_index],
                        median[block_indices, lead_index],
                    )["summary"]
                )
            block_weights = np.asarray(
                [summary["matched_events"] for summary in block_summaries],
                dtype=float,
            )
            for row in metric_rows:
                name = str(row["metric"])
                values = np.asarray([summary[name] for summary in block_summaries], dtype=float)
                sampled_values = values[block_draws]
                sampled_weights = block_weights[block_draws]
                usable = np.isfinite(sampled_values) & (sampled_weights > 0)
                weight_sums = np.sum(np.where(usable, sampled_weights, 0.0), axis=1)
                weighted_sums = np.sum(
                    np.where(usable, sampled_values * sampled_weights, 0.0), axis=1
                )
                distribution = np.full(n_bootstrap, np.nan, dtype=float)
                np.divide(
                    weighted_sums,
                    weight_sums,
                    out=distribution,
                    where=weight_sums > 0,
                )
                finite = distribution[np.isfinite(distribution)]
                sufficient = bool(
                    np.isfinite(float(row["estimate"])) and finite.size >= minimum_replicates
                )
                low, high = (
                    np.quantile(finite, (0.025, 0.975))
                    if sufficient
                    else (float("nan"), float("nan"))
                )
                row.update(
                    {
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "finite_replicates": int(finite.size),
                        "ci_available": sufficient,
                        "finite_candidate_blocks": int(
                            np.sum(np.isfinite(values) & (block_weights > 0))
                        ),
                    }
                )
            return {
                "lead_minutes": lead_minutes,
                "available": any(bool(row["ci_available"]) for row in metric_rows),
                "metrics": metric_rows,
            }

        by_reported_lead = {
            f"{lead * step_minutes}min": lead_report(lead - 1, indices, valid_starts, draws)
            for lead in REPORT_LEADS
            if lead <= targets.shape[1]
        }
        primary = by_reported_lead[f"{step_minutes}min"]

        rows.append(
            {
                "site_id": int(site),
                "site_name": str(site_id_map.get(str(int(site)), site)),
                "available": any(
                    bool(lead_row["available"]) for lead_row in by_reported_lead.values()
                ),
                "reason": (
                    None
                    if valid_starts.size
                    else "no contiguous 24-hour forecast-origin moving block"
                ),
                "block_length_origins": block_length,
                "total_candidate_blocks": total_candidate_blocks,
                "evaluated_candidate_blocks": int(valid_starts.size),
                "candidate_subsampling": (
                    f"all if <={maximum_candidate_blocks}, otherwise evenly spaced over test time"
                ),
                "blocks_drawn_per_replicate": blocks_per_replicate,
                "n_bootstrap": n_bootstrap,
                "forecast_lead_policy": (
                    "fixed lead per sequence; the top-level metrics alias is 15 minutes"
                ),
                "metrics": primary["metrics"],
                "by_reported_lead": by_reported_lead,
                "temporal_audit": temporal_audit,
            }
        )

    return {
        "unit": "site",
        "n_bootstrap": n_bootstrap,
        "confidence_level": 0.95,
        "interval_method": "percentile",
        "method": (
            "site-stratified percentile moving-block bootstrap with contiguous "
            "24-hour forecast-origin blocks; event matching is recomputed within each "
            "candidate block before vectorized matched-event-weighted resampling"
        ),
        "classification_metric_distinction": (
            "These event-error replicates use matched-event weights. The separate ramp "
            "classification bootstrap estimates a mean of precomputed daily block "
            "classification metrics and is not a pooled-cell or pooled-event estimand."
        ),
        "amplitude_aggregation_boundary": (
            "To preserve the registered matcher exactly, segment-level amplitude MAE is "
            "weighted by that segment's matched-event count. A match beginning at a "
            "segment or resampled-block boundary can lack the preceding power value and "
            "therefore contributes support weight but no amplitude difference; this "
            "boundary behavior is not silently redefined in evaluation."
        ),
        "minimum_finite_replicate_fraction": minimum_finite_fraction,
        "rows": rows,
    }


def _bootstrap_report(
    labels: np.ndarray,
    state_probs: np.ndarray,
    site_ids: np.ndarray | None,
    timestamps: np.ndarray,
    site_id_map: Mapping[str, Any],
    reference_probs: np.ndarray | None = None,
    reference_name: str | None = None,
    *,
    n_bootstrap: int,
    seed: int,
    minimum_finite_fraction: float = 0.95,
    maximum_candidate_blocks: int = 512,
    origin_step_minutes: int = 15,
) -> dict[str, object]:
    if n_bootstrap < 1:
        raise ValueError("bootstrap replicate count must be positive")
    if not 0.0 < minimum_finite_fraction <= 1.0:
        raise ValueError("minimum_finite_fraction must be in (0, 1]")
    if maximum_candidate_blocks < 2:
        raise ValueError("maximum_candidate_blocks must be at least two")
    if origin_step_minutes <= 0 or 1440 % origin_step_minutes:
        raise ValueError("origin_step_minutes must divide one day")
    if reference_probs is not None and reference_probs.shape != state_probs.shape:
        raise ValueError("reference state probabilities must match evaluated probabilities")
    if (reference_probs is None) != (reference_name is None):
        raise ValueError("reference probabilities and name must be supplied together")
    if state_probs.shape != labels.shape + (3,) or timestamps.shape != labels.shape:
        raise ValueError("labels, probabilities, and timestamps have inconsistent shapes")
    sites = np.zeros(labels.shape[0], dtype=int) if site_ids is None else site_ids.reshape(-1)
    rows = []
    for offset, site in enumerate(np.unique(sites)):
        indices = np.flatnonzero(sites == site)
        origin_time = timestamps[indices, 0].astype("datetime64[ns]").astype(np.int64)
        order = np.argsort(origin_time, kind="stable")
        indices, origin_time = indices[order], origin_time[order]
        block_length = 1440 // origin_step_minutes
        if len(indices) < block_length:
            rows.append(
                {
                    "site_id": int(site),
                    "site_name": str(site_id_map.get(str(int(site)), site)),
                    "available": False,
                    "reason": f"fewer than {block_length} forecast origins",
                }
            )
            continue
        differences = np.diff(origin_time)
        expected = origin_step_minutes * 60 * 1_000_000_000
        valid_starts = np.asarray(
            [
                start
                for start in range(len(indices) - block_length + 1)
                if np.all(differences[start : start + block_length - 1] == expected)
            ],
            dtype=int,
        )
        if not valid_starts.size:
            rows.append(
                {
                    "site_id": int(site),
                    "site_name": str(site_id_map.get(str(int(site)), site)),
                    "available": False,
                    "reason": "no contiguous 24-hour moving block",
                }
            )
            continue

        total_candidate_blocks = int(valid_starts.size)
        if valid_starts.size > maximum_candidate_blocks:
            positions = np.linspace(0, valid_starts.size - 1, maximum_candidate_blocks, dtype=int)
            valid_starts = valid_starts[np.unique(positions)]

        selected_labels = labels[indices]
        selected_probs = state_probs[indices]
        selected_reference = reference_probs[indices] if reference_probs is not None else None
        selected_eligible = np.isin(selected_labels, (0, 1, 2)) & np.all(
            np.isfinite(selected_probs), axis=-1
        )
        if selected_reference is not None:
            selected_eligible &= np.all(np.isfinite(selected_reference), axis=-1)

        metric_names = tuple(
            [f"binary_{key}" for key in ("precision", "recall", "f1", "csi", "auprc")]
            + [
                f"{state_name}_{key}"
                for state_name in ("stable", "ramp-up", "ramp-down")
                for key in ("precision", "recall", "f1", "csi", "auprc")
            ]
        )

        def metric_values(
            y: np.ndarray,
            p: np.ndarray,
            eligible: np.ndarray,
            names: tuple[str, ...],
        ) -> dict[str, float]:
            y = y[eligible]
            p = p[eligible]
            if y.size == 0:
                return {name: float("nan") for name in names}
            binary = binary_ramp_metrics(y, p)
            multi = ramp_classification_metrics(y, p)
            values = {
                f"binary_{key}": float(binary[key])
                for key in ("precision", "recall", "f1", "csi", "auprc")
            }
            for state_name in ("stable", "ramp-up", "ramp-down"):
                for key in ("precision", "recall", "f1", "csi", "auprc"):
                    values[f"{state_name}_{key}"] = float(
                        multi["per_class"][state_name][key]  # type: ignore[index]
                    )
            return values

        model_blocks = [
            metric_values(
                selected_labels[start : start + block_length],
                selected_probs[start : start + block_length],
                selected_eligible[start : start + block_length],
                metric_names,
            )
            for start in valid_starts
        ]
        reference_blocks = (
            [
                metric_values(
                    selected_labels[start : start + block_length],
                    selected_reference[start : start + block_length],
                    selected_eligible[start : start + block_length],
                    metric_names,
                )
                for start in valid_starts
            ]
            if selected_reference is not None
            else None
        )
        model_values = {
            name: np.asarray([block[name] for block in model_blocks], dtype=float)
            for name in metric_names
        }
        reference_values = (
            {
                name: np.asarray([block[name] for block in reference_blocks], dtype=float)
                for name in metric_names
            }
            if reference_blocks is not None
            else None
        )
        rng = np.random.default_rng(seed + offset)
        blocks_needed = int(np.ceil(len(indices) / block_length))
        draws = rng.integers(
            0,
            len(valid_starts),
            size=(n_bootstrap, blocks_needed),
        )

        def bootstrap_means(values: np.ndarray, block_draws: np.ndarray) -> np.ndarray:
            sampled = values[block_draws]
            finite = np.isfinite(sampled)
            counts = np.sum(finite, axis=1)
            sums = np.sum(np.where(finite, sampled, 0.0), axis=1)
            result = np.full(n_bootstrap, np.nan, dtype=float)
            np.divide(sums, counts, out=result, where=counts > 0)
            return result

        metrics = []
        for name in metric_names:
            values = model_values[name]
            valid_values = values[np.isfinite(values)]
            estimate = float(np.mean(valid_values)) if valid_values.size else float("nan")
            distribution = bootstrap_means(values, draws)
            finite = distribution[np.isfinite(distribution)]
            sufficient = finite.size >= int(np.ceil(minimum_finite_fraction * n_bootstrap))
            low, high = (
                np.quantile(finite, (0.025, 0.975)) if sufficient else (float("nan"), float("nan"))
            )
            row: dict[str, object] = {
                "metric": name,
                "estimand": "mean metric across contiguous 24-hour forecast-origin blocks",
                "estimate": estimate,
                "finite_candidate_blocks": int(valid_values.size),
                "total_evaluated_candidate_blocks": int(values.size),
                "ci_low": float(low),
                "ci_high": float(high),
                "finite_replicates": int(finite.size),
                "minimum_required_finite_replicates": int(
                    np.ceil(minimum_finite_fraction * n_bootstrap)
                ),
                "ci_available": bool(sufficient),
            }
            if reference_values is not None:
                reference_metric = reference_values[name]
                pair_valid = np.isfinite(values) & np.isfinite(reference_metric)
                differences = np.where(pair_valid, values - reference_metric, np.nan)
                valid_differences = differences[np.isfinite(differences)]
                difference_estimate = (
                    float(np.mean(valid_differences)) if valid_differences.size else float("nan")
                )
                difference_distribution = bootstrap_means(differences, draws)
                finite_difference = difference_distribution[np.isfinite(difference_distribution)]
                paired_sufficient = finite_difference.size >= int(
                    np.ceil(minimum_finite_fraction * n_bootstrap)
                )
                difference_low, difference_high = (
                    np.quantile(finite_difference, (0.025, 0.975))
                    if paired_sufficient
                    else (float("nan"), float("nan"))
                )
                finite_reference = reference_metric[np.isfinite(reference_metric)]
                row["paired_reference"] = {
                    "name": reference_name,
                    "reference_estimate": (
                        float(np.mean(finite_reference)) if finite_reference.size else float("nan")
                    ),
                    "difference_model_minus_reference": difference_estimate,
                    "finite_paired_candidate_blocks": int(valid_differences.size),
                    "difference_ci_low": float(difference_low),
                    "difference_ci_high": float(difference_high),
                    "finite_replicates": int(finite_difference.size),
                    "ci_available": bool(paired_sufficient),
                    "significantly_better_at_0.05": bool(
                        paired_sufficient and float(difference_low) > 0.0
                    ),
                }
            metrics.append(row)
        rows.append(
            {
                "site_id": int(site),
                "site_name": str(site_id_map.get(str(int(site)), site)),
                "available": True,
                "block_length_origins": block_length,
                "total_candidate_blocks": total_candidate_blocks,
                "evaluated_candidate_blocks": int(valid_starts.size),
                "common_eligible_cells": int(np.sum(selected_eligible)),
                "eligible_population": (
                    "labels in {stable,ramp-up,ramp-down} with finite model and, when "
                    "present, reference probabilities; identical for every metric"
                ),
                "candidate_subsampling": (
                    f"all if <={maximum_candidate_blocks}, otherwise evenly spaced over test time"
                ),
                "blocks_drawn_per_replicate": blocks_needed,
                "n_bootstrap": n_bootstrap,
                "temporal_audit": {
                    "expected_origin_step_minutes": int(origin_step_minutes),
                    "origins_per_day": int(block_length),
                },
                "metrics": metrics,
            }
        )
    return {
        "unit": "site",
        "method": (
            "bootstrap of precomputed metrics on overlapping contiguous 24-hour origin blocks; "
            "paired references use identical block draws"
        ),
        "minimum_finite_replicate_fraction": minimum_finite_fraction,
        "rows": rows,
    }


def _expert_specialization_report(
    arrays: Mapping[str, np.ndarray],
    levels: Sequence[float],
    metadata: Mapping[str, Any],
) -> dict[str, object]:
    """Audit whether the named experts actually specialize and whether routing collapses."""

    required = {"expert_quantiles", "gate_weights"}
    if not required.issubset(arrays):
        return {
            "available": False,
            "reason": "prediction archive is not a three-expert RaCE-PV output",
        }
    names = ("stable", "ramp_up", "ramp_down")
    experts = np.asarray(arrays["expert_quantiles"], dtype=float)
    gates = np.asarray(arrays["gate_weights"], dtype=float)
    targets = np.asarray(arrays["targets"], dtype=float)
    labels = np.asarray(arrays["state_labels"], dtype=int)
    daylight = np.asarray(arrays["daylight_mask"], dtype=bool)
    median_index = int(np.flatnonzero(np.isclose(np.asarray(levels), 0.5))[0])
    active_raw = metadata.get("active_experts", names)
    active = (
        tuple(str(value) for value in active_raw)
        if isinstance(active_raw, (list, tuple))
        else names
    )
    invalid_active = sorted(set(active).difference(names))
    if invalid_active:
        raise ValueError(f"prediction metadata has unknown active experts: {invalid_active}")

    state_masks = {
        "stable": labels == 0,
        "ramp_up": labels == 1,
        "ramp_down": labels == 2,
        "ramp_any": np.isin(labels, (1, 2)),
    }
    scopes: list[tuple[str, int, int]] = [
        ("all_leads", 0, targets.shape[1]),
        *LEAD_BANDS,
    ]
    performance: dict[str, object] = {}
    gate_by_truth_and_lead: dict[str, object] = {}
    for state_name, state_mask in state_masks.items():
        state_performance: dict[str, object] = {}
        state_gates: dict[str, object] = {}
        for scope_name, start, end in scopes:
            if start >= targets.shape[1]:
                continue
            lead_mask = np.zeros_like(daylight, dtype=bool)
            lead_mask[:, start : min(end, targets.shape[1])] = True
            mask = daylight & state_mask & lead_mask
            expert_rows: dict[str, object] = {}
            for expert_index, expert_name in enumerate(names):
                expert_quantiles = experts[..., expert_index, :]
                expert_rows[expert_name] = {
                    "active": expert_name in active,
                    "point": point_forecast_metrics(
                        targets,
                        expert_quantiles[..., median_index],
                        daylight_mask=mask,
                    )["daylight"],
                    "probabilistic": probabilistic_forecast_metrics(
                        targets,
                        expert_quantiles,
                        levels,
                        mask=mask,
                    ),
                }
            state_performance[scope_name] = {
                "count": int(np.sum(mask)),
                "experts": expert_rows,
            }
            selected_gates = gates[mask]
            if selected_gates.size:
                hard = np.argmax(selected_gates, axis=-1)
                mean_weights = selected_gates.mean(axis=0)
                hard_shares = np.asarray(
                    [np.mean(hard == index) for index in range(len(names))],
                    dtype=float,
                )
                entropy = -np.sum(
                    np.where(
                        selected_gates > 0,
                        selected_gates * np.log(np.clip(selected_gates, 1e-12, 1.0)),
                        0.0,
                    ),
                    axis=-1,
                )
                mean_entropy = float(np.mean(entropy))
            else:
                mean_weights = np.full(len(names), np.nan)
                hard_shares = np.full(len(names), np.nan)
                mean_entropy = float("nan")
            state_gates[scope_name] = {
                "count": int(np.sum(mask)),
                "mean_gate_weight": {
                    name: float(mean_weights[index]) for index, name in enumerate(names)
                },
                "hard_routing_share": {
                    name: float(hard_shares[index]) for index, name in enumerate(names)
                },
                "mean_gate_entropy_nats": mean_entropy,
                "normalized_gate_entropy": (
                    mean_entropy / np.log(len(active))
                    if len(active) > 1 and np.isfinite(mean_entropy)
                    else float("nan")
                ),
            }
        performance[state_name] = state_performance
        gate_by_truth_and_lead[state_name] = state_gates

    selected_all = gates[daylight]
    overall_mean = selected_all.mean(axis=0)
    overall_hard = np.argmax(selected_all, axis=-1)
    hard_share = np.asarray(
        [np.mean(overall_hard == index) for index in range(len(names))], dtype=float
    )
    active_indices = [names.index(name) for name in active]
    active_mean = overall_mean[active_indices]
    active_hard = hard_share[active_indices]
    collapse_threshold = 0.01
    pairwise = {}
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            key = f"{names[left]}__vs__{names[right]}"
            pairwise[key] = {
                "mean_absolute_median_difference": float(
                    np.mean(
                        np.abs(
                            experts[..., left, median_index] - experts[..., right, median_index]
                        )[daylight]
                    )
                ),
                "mean_absolute_all_quantile_difference": float(
                    np.mean(np.abs(experts[..., left, :] - experts[..., right, :])[daylight])
                ),
            }
    valid_state_mask = daylight & np.isin(labels, (0, 1, 2))
    return {
        "available": True,
        "expert_order": list(names),
        "active_experts": list(active),
        "performance_by_true_state_and_lead": performance,
        "gate_by_true_state_and_lead": gate_by_truth_and_lead,
        "router_vs_true_state": ramp_classification_metrics(labels, gates, mask=valid_state_mask),
        "overall_utilization": {
            "mean_gate_weight": {
                name: float(overall_mean[index]) for index, name in enumerate(names)
            },
            "hard_routing_share": {
                name: float(hard_share[index]) for index, name in enumerate(names)
            },
            "collapse_diagnostic_threshold": collapse_threshold,
            "collapsed_active_expert": bool(
                np.any(active_mean < collapse_threshold) or np.any(active_hard < collapse_threshold)
            ),
            "note": (
                "The 1% threshold is a transparent routing-collapse diagnostic, not a "
                "statistical acceptance test."
            ),
        },
        "expert_prediction_diversity": pairwise,
        "interpretation_boundary": (
            "Named-expert specialization is supported only when the state/lead matrix, "
            "gate utilization, diversity and ablations agree; names alone are not evidence."
        ),
    }


def _per_site_report(
    site_ids: np.ndarray | None,
    targets: np.ndarray,
    median: np.ndarray,
    quantiles: np.ndarray,
    levels: Sequence[float],
    labels: np.ndarray,
    state_probs: np.ndarray,
    daylight: np.ndarray,
    conditional_intervals: Mapping[float, np.ndarray],
    lead_band_intervals: Mapping[float, np.ndarray],
    global_intervals: Mapping[float, np.ndarray],
    reference: np.ndarray | None,
    selected_point_reference: np.ndarray | None,
    selected_point_reference_name: str | None,
    reference_quantiles: np.ndarray | None,
    reference_name: str | None,
    ramp_reference_quantiles: np.ndarray | None,
    ramp_reference_name: str | None,
    site_id_map: Mapping[str, Any],
) -> list[dict[str, object]]:
    sites = np.zeros(targets.shape[0], dtype=int) if site_ids is None else site_ids.reshape(-1)
    rows = []
    for site in np.unique(sites):
        selected = sites == site
        intervals = {
            coverage: values[selected] for coverage, values in conditional_intervals.items()
        }
        lead_band_site_intervals = {
            coverage: values[selected] for coverage, values in lead_band_intervals.items()
        }
        global_site_intervals = {
            coverage: values[selected] for coverage, values in global_intervals.items()
        }
        model_probability = _probability_metrics_with_valid_states(
            targets[selected],
            quantiles[selected],
            levels,
            intervals=intervals,
            labels=labels[selected],
            daylight=daylight[selected],
        )
        lead_band_probability = _probability_metrics_with_valid_states(
            targets[selected],
            quantiles[selected],
            levels,
            intervals=lead_band_site_intervals,
            labels=labels[selected],
            daylight=daylight[selected],
        )
        global_probability = _probability_metrics_with_valid_states(
            targets[selected],
            quantiles[selected],
            levels,
            intervals=global_site_intervals,
            labels=labels[selected],
            daylight=daylight[selected],
        )
        reference_probability = (
            probabilistic_forecast_metrics(
                targets[selected],
                reference_quantiles[selected],
                levels,
                mask=daylight[selected],
            )
            if reference_quantiles is not None
            else None
        )
        ramp_mask = daylight[selected] & np.isin(labels[selected], (1, 2))
        site_point = _scoped_metrics(
            targets[selected],
            median[selected],
            daylight[selected],
            None if reference is None else reference[selected],
        )
        ramp_point_model = point_forecast_metrics(
            targets[selected], median[selected], daylight_mask=ramp_mask
        )["daylight"]
        site_point["ramp_periods"] = {"model": ramp_point_model}
        if selected_point_reference is not None:
            ramp_point_reference = point_forecast_metrics(
                targets[selected],
                selected_point_reference[selected],
                daylight_mask=ramp_mask,
            )["daylight"]
            site_point["ramp_periods"] = {
                "model": ramp_point_model,
                "validation_selected_reference": {
                    "name": selected_point_reference_name,
                    **ramp_point_reference,
                },
                "improvement_reference_minus_model": {
                    "nmae": float(ramp_point_reference["nmae"]) - float(ramp_point_model["nmae"]),
                    "nrmse": float(ramp_point_reference["nrmse"])
                    - float(ramp_point_model["nrmse"]),
                },
            }
        ramp_probability = probabilistic_forecast_metrics(
            targets[selected], quantiles[selected], levels, mask=ramp_mask
        )
        ramp_reference_probability = (
            probabilistic_forecast_metrics(
                targets[selected],
                ramp_reference_quantiles[selected],
                levels,
                mask=ramp_mask,
            )
            if ramp_reference_quantiles is not None
            else None
        )
        by_lead: dict[str, object] = {}
        for lead in REPORT_LEADS:
            if lead > targets.shape[1]:
                continue
            lead_name = f"{lead * 15}min"
            selection = (selected, slice(lead - 1, lead))
            lead_intervals = {
                coverage: values[selection] for coverage, values in conditional_intervals.items()
            }
            lead_global_intervals = {
                coverage: values[selection] for coverage, values in global_intervals.items()
            }
            lead_band_only_intervals = {
                coverage: values[selection] for coverage, values in lead_band_intervals.items()
            }
            lead_model = _probability_metrics_with_valid_states(
                targets[selection],
                quantiles[selection],
                levels,
                intervals=lead_intervals,
                labels=labels[selection],
                daylight=daylight[selection],
            )
            lead_global = _probability_metrics_with_valid_states(
                targets[selection],
                quantiles[selection],
                levels,
                intervals=lead_global_intervals,
                labels=labels[selection],
                daylight=daylight[selection],
            )
            lead_band_only = _probability_metrics_with_valid_states(
                targets[selection],
                quantiles[selection],
                levels,
                intervals=lead_band_only_intervals,
                labels=labels[selection],
                daylight=daylight[selection],
            )
            entry: dict[str, object] = {
                "model": lead_model,
                "lead_band_only_cqr_intervals": lead_band_only["intervals"],
                "global_cqr_intervals": lead_global["intervals"],
                "conditional_minus_lead_band_only_interval_score": {
                    str(coverage): float(
                        lead_model["intervals"][coverage]["interval_score"]  # type: ignore[index]
                        - lead_band_only["intervals"][coverage]["interval_score"]  # type: ignore[index]
                    )
                    for coverage in conditional_intervals
                },
                "conditional_minus_global_interval_score": {
                    str(coverage): float(
                        lead_model["intervals"][coverage]["interval_score"]  # type: ignore[index]
                        - lead_global["intervals"][coverage]["interval_score"]  # type: ignore[index]
                    )
                    for coverage in conditional_intervals
                },
            }
            if reference_quantiles is not None:
                lead_reference = probabilistic_forecast_metrics(
                    targets[selection],
                    reference_quantiles[selection],
                    levels,
                    mask=daylight[selection],
                )
                entry["registered_reference"] = {
                    "name": reference_name,
                    **lead_reference,
                }
                entry["improvement_reference_minus_model"] = {
                    "mean_pinball_loss": float(lead_reference["mean_pinball_loss"])
                    - float(lead_model["mean_pinball_loss"]),
                    "quantile_crps": float(lead_reference["quantile_crps"])
                    - float(lead_model["quantile_crps"]),
                }
            by_lead[lead_name] = entry
        rows.append(
            {
                "site_id": int(site),
                "site_name": str(site_id_map.get(str(int(site)), site)),
                "sample_count": int(np.sum(selected)),
                "point": site_point,
                "probabilistic": model_probability,
                "ramp_period_probabilistic": {
                    "model": ramp_probability,
                    "validation_selected_reference": (
                        {
                            "name": ramp_reference_name,
                            **ramp_reference_probability,
                        }
                        if ramp_reference_probability is not None
                        else None
                    ),
                    "improvement_reference_minus_model": (
                        {
                            "mean_pinball_loss": float(
                                ramp_reference_probability["mean_pinball_loss"]
                            )
                            - float(ramp_probability["mean_pinball_loss"]),
                            "quantile_crps": float(ramp_reference_probability["quantile_crps"])
                            - float(ramp_probability["quantile_crps"]),
                        }
                        if ramp_reference_probability is not None
                        else None
                    ),
                },
                "lead_band_only_cqr_intervals": lead_band_probability["intervals"],
                "global_cqr_intervals": global_probability["intervals"],
                "conditional_minus_lead_band_only_interval_score": {
                    str(coverage): float(
                        model_probability["intervals"][coverage]["interval_score"]  # type: ignore[index]
                        - lead_band_probability["intervals"][coverage]["interval_score"]  # type: ignore[index]
                    )
                    for coverage in conditional_intervals
                },
                "lead_band_only_minus_global_interval_score": {
                    str(coverage): float(
                        lead_band_probability["intervals"][coverage]["interval_score"]  # type: ignore[index]
                        - global_probability["intervals"][coverage]["interval_score"]  # type: ignore[index]
                    )
                    for coverage in conditional_intervals
                },
                "registered_reference_probabilistic": (
                    {"name": reference_name, **reference_probability}
                    if reference_probability is not None
                    else None
                ),
                "improvement_reference_minus_model": (
                    {
                        "mean_pinball_loss": float(reference_probability["mean_pinball_loss"])
                        - float(model_probability["mean_pinball_loss"]),
                        "quantile_crps": float(reference_probability["quantile_crps"])
                        - float(model_probability["quantile_crps"]),
                    }
                    if reference_probability is not None
                    else None
                ),
                "probabilistic_by_reported_lead": by_lead,
                "ramp": ramp_classification_metrics(
                    labels[selected], state_probs[selected], mask=daylight[selected]
                ),
            }
        )
    return rows


def _true_state_lead_interval_report(
    targets: np.ndarray,
    labels: np.ndarray,
    intervals: Mapping[float, np.ndarray],
    daylight: np.ndarray,
) -> dict[str, object]:
    output: dict[str, object] = {}
    state_groups = {"stable": labels == 0, "ramp": labels > 0}
    for coverage, values in intervals.items():
        rows: dict[str, object] = {}
        for state_name, state_mask in state_groups.items():
            for band_name, start, end in LEAD_BANDS:
                if start >= targets.shape[1]:
                    continue
                band_mask = np.zeros_like(daylight, dtype=bool)
                band_mask[:, start : min(end, targets.shape[1])] = True
                mask = daylight & state_mask & band_mask
                low, high = values[..., 0], values[..., 1]
                rows[f"{state_name}/{band_name}"] = {
                    "count": int(np.sum(mask)),
                    "picp": prediction_interval_coverage(targets, low, high, mask=mask),
                    "pinaw": prediction_interval_normalized_width(targets, low, high, mask=mask),
                    "interval_score": interval_score(
                        targets, low, high, coverage=coverage, mask=mask
                    ),
                }
        output[str(coverage)] = rows
    return output


def _true_state_quantile_score_report(
    targets: np.ndarray,
    labels: np.ndarray,
    quantiles: np.ndarray,
    levels: Sequence[float],
    daylight: np.ndarray,
    *,
    reference_quantiles: np.ndarray | None = None,
    reference_name: str | None = None,
    ramp_reference_quantiles: np.ndarray | None = None,
    ramp_reference_name: str | None = None,
) -> dict[str, object]:
    """Report pinball/CRPS on stable and ramp truth, including lead bands."""

    if (reference_quantiles is None) != (reference_name is None):
        raise ValueError("reference quantiles and name must be supplied together")
    if (ramp_reference_quantiles is None) != (ramp_reference_name is None):
        raise ValueError("ramp reference quantiles and name must be supplied together")
    state_groups = {
        "stable": labels == 0,
        "ramp": np.isin(labels, (1, 2)),
        "ramp_up": labels == 1,
        "ramp_down": labels == 2,
    }
    output: dict[str, object] = {}
    for state_name, state_mask in state_groups.items():
        state_reference_quantiles = (
            ramp_reference_quantiles if state_name.startswith("ramp") else reference_quantiles
        )
        state_reference_name = (
            ramp_reference_name if state_name.startswith("ramp") else reference_name
        )
        scopes = [("all_leads", 0, targets.shape[1]), *LEAD_BANDS]
        state_rows: dict[str, object] = {}
        for band_name, start, end in scopes:
            if start >= targets.shape[1]:
                continue
            band_mask = np.zeros_like(daylight, dtype=bool)
            band_mask[:, start : min(end, targets.shape[1])] = True
            mask = daylight & state_mask & band_mask
            model = probabilistic_forecast_metrics(targets, quantiles, levels, mask=mask)
            row: dict[str, object] = {
                "count": int(np.sum(mask)),
                "model": model,
            }
            if state_reference_quantiles is not None:
                reference = probabilistic_forecast_metrics(
                    targets, state_reference_quantiles, levels, mask=mask
                )
                row["validation_selected_reference"] = {
                    "name": state_reference_name,
                    **reference,
                }
                row["improvement_reference_minus_model"] = {
                    "mean_pinball_loss": float(reference["mean_pinball_loss"])
                    - float(model["mean_pinball_loss"]),
                    "quantile_crps": float(reference["quantile_crps"])
                    - float(model["quantile_crps"]),
                }
            state_rows[band_name] = row
        output[state_name] = state_rows
    return output


def _quantile_crps_cells(
    targets: np.ndarray,
    quantiles: np.ndarray,
    levels: Sequence[float],
) -> np.ndarray:
    """Return the repository's trapezoidal quantile-CRPS approximation per cell."""

    truth = np.asarray(targets, dtype=float)
    predictions = np.asarray(quantiles, dtype=float)
    taus = np.asarray(levels, dtype=float)
    residual = truth[..., None] - predictions
    pinball = np.maximum(taus * residual, (taus - 1.0) * residual)
    integration_levels = np.r_[0.0, taus, 1.0]
    integration_losses = np.concatenate(
        (np.zeros_like(pinball[..., :1]), pinball, np.zeros_like(pinball[..., :1])),
        axis=-1,
    )
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return 2.0 * integrate(integration_losses, integration_levels, axis=-1)


def _dm_quantile_report(
    targets: np.ndarray,
    quantiles: np.ndarray,
    reference_quantiles: np.ndarray,
    levels: Sequence[float],
    eligibility: np.ndarray,
    site_ids: np.ndarray,
    timestamps: np.ndarray,
    site_id_map: Mapping[str, Any],
    reference_name: str,
    *,
    step_minutes: int = 15,
    contract=None,
) -> dict[str, object]:
    """DM/BH comparison of per-cell quantile-CRPS on one fixed population."""

    model_scores = _quantile_crps_cells(targets, quantiles, levels)
    reference_scores = _quantile_crps_cells(targets, reference_quantiles, levels)
    sites = np.asarray(site_ids).reshape(-1)
    origin_step_minutes = _infer_origin_step_minutes(
        timestamps,
        sites,
        target_step_minutes=step_minutes,
        contract=contract,
    )
    registered_lag = max(0, int(np.ceil(targets.shape[1] * step_minutes / origin_step_minutes)) - 1)
    rows: list[dict[str, object]] = []
    for site in np.unique(sites):
        selected = sites == site
        for lead in REPORT_LEADS:
            if lead > targets.shape[1]:
                continue
            index = lead - 1
            model_loss = np.where(
                eligibility[selected, index], model_scores[selected, index], np.nan
            )
            reference_loss = np.where(
                eligibility[selected, index], reference_scores[selected, index], np.nan
            )
            try:
                result, temporal_audit = _time_aware_diebold_mariano_test(
                    model_loss,
                    reference_loss,
                    timestamps[selected],
                    horizon=targets.shape[1],
                    step_minutes=step_minutes,
                    origin_step_minutes=origin_step_minutes,
                    contract=contract,
                )
            except ValueError:
                continue
            rows.append(
                {
                    "site_id": int(site),
                    "site_name": str(site_id_map.get(str(int(site)), site)),
                    "lead_minutes": lead * 15,
                    "statistic": result.statistic,
                    "pvalue": result.pvalue,
                    "mean_loss_difference": result.mean_loss_difference,
                    "lag": result.lag,
                    "n_observations": result.n_observations,
                    **temporal_audit,
                }
            )
    if not rows:
        return {
            "loss": "trapezoidal_quantile_crps",
            "rows": [],
            "reason": "insufficient ramp observations",
            "newey_west_lag": registered_lag,
            "hac_time_axis": "forecast_origin",
            "expected_origin_step_minutes": origin_step_minutes,
            "origin_gap_policy": (
                "forecast origins are partitioned at every non-registered-step jump; "
                "Newey-West covariance is estimated only within continuous segments"
            ),
        }
    adjusted, rejected = benjamini_hochberg([float(row["pvalue"]) for row in rows])
    for row, value, decision in zip(rows, adjusted, rejected, strict=True):
        row["bh_adjusted_pvalue"] = float(value)
        row["reject_equal_accuracy_at_0.05"] = bool(decision)
        row["evaluated_model_better"] = bool(float(row["mean_loss_difference"]) < 0)
    return {
        "loss": "trapezoidal_quantile_crps",
        "reference": reference_name,
        "population": "true ramp-up/down daylight cells only",
        "comparison_family": (
            f"all registered site-by-lead ramp comparisons against {reference_name}"
        ),
        "newey_west_lag": registered_lag,
        "hac_time_axis": "forecast_origin",
        "expected_origin_step_minutes": origin_step_minutes,
        "origin_gap_policy": (
            "forecast origins are partitioned at every non-registered-step jump; "
            "Newey-West covariance is estimated only within continuous segments"
        ),
        "rows": rows,
        "significantly_better_fraction": float(
            np.mean(
                [
                    bool(row["reject_equal_accuracy_at_0.05"] and row["evaluated_model_better"])
                    for row in rows
                ]
            )
        ),
    }


def _balanced_site_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    daylight_nrmse = np.asarray(
        [row["point"]["overall"]["daylight"]["nrmse"] for row in rows],  # type: ignore[index]
        dtype=float,
    )
    skills = np.asarray(
        [
            row["point"]["overall"]["daylight"].get("forecast_skill_rmse", np.nan)  # type: ignore[index]
            for row in rows
        ],
        dtype=float,
    )
    finite_skill = skills[np.isfinite(skills)]
    by_lead: dict[str, object] = {}
    probability_by_lead: dict[str, object] = {}
    for lead in REPORT_LEADS:
        name = f"{lead * 15}min"
        lead_skills = np.asarray(
            [
                row["point"]["reported_leads"][name]["daylight"].get(  # type: ignore[index]
                    "forecast_skill_rmse", np.nan
                )
                for row in rows
                if name in row["point"]["reported_leads"]  # type: ignore[operator,index]
            ],
            dtype=float,
        )
        finite = lead_skills[np.isfinite(lead_skills)]
        by_lead[name] = {
            "mean_forecast_skill_rmse": (float(np.mean(finite)) if finite.size else float("nan")),
            "positive_skill_sites": int(np.sum(finite > 0)),
            "evaluated_sites": int(finite.size),
            "positive_skill_fraction": (
                float(np.mean(finite > 0)) if finite.size else float("nan")
            ),
        }
        probabilistic_rows = [
            row["probabilistic_by_reported_lead"][name]  # type: ignore[index]
            for row in rows
            if name in row["probabilistic_by_reported_lead"]  # type: ignore[operator,index]
        ]
        pinball = np.asarray(
            [entry["model"]["mean_pinball_loss"] for entry in probabilistic_rows],  # type: ignore[index]
            dtype=float,
        )
        crps = np.asarray(
            [entry["model"]["quantile_crps"] for entry in probabilistic_rows],  # type: ignore[index]
            dtype=float,
        )
        improvements_pinball = np.asarray(
            [
                entry.get("improvement_reference_minus_model", {}).get(  # type: ignore[union-attr]
                    "mean_pinball_loss", np.nan
                )
                for entry in probabilistic_rows
            ],
            dtype=float,
        )
        improvements_crps = np.asarray(
            [
                entry.get("improvement_reference_minus_model", {}).get(  # type: ignore[union-attr]
                    "quantile_crps", np.nan
                )
                for entry in probabilistic_rows
            ],
            dtype=float,
        )
        conditional_better = []
        state_conditioning_better = []
        for entry in probabilistic_rows:
            differences = np.asarray(
                list(entry["conditional_minus_global_interval_score"].values()),  # type: ignore[index,union-attr]
                dtype=float,
            )
            finite_differences = differences[np.isfinite(differences)]
            conditional_better.append(
                bool(finite_differences.size and np.all(finite_differences < 0))
            )
            state_differences = np.asarray(
                list(
                    entry["conditional_minus_lead_band_only_interval_score"].values()  # type: ignore[index,union-attr]
                ),
                dtype=float,
            )
            finite_state_differences = state_differences[np.isfinite(state_differences)]
            state_conditioning_better.append(
                bool(finite_state_differences.size and np.all(finite_state_differences < 0))
            )
        finite_pinball_gain = improvements_pinball[np.isfinite(improvements_pinball)]
        finite_crps_gain = improvements_crps[np.isfinite(improvements_crps)]
        probability_by_lead[name] = {
            "equal_weight_mean_pinball_loss": float(np.nanmean(pinball)),
            "equal_weight_mean_quantile_crps": float(np.nanmean(crps)),
            "sites_better_than_reference_pinball": int(np.sum(finite_pinball_gain > 0)),
            "fraction_sites_better_than_reference_pinball": (
                float(np.mean(finite_pinball_gain > 0))
                if finite_pinball_gain.size
                else float("nan")
            ),
            "sites_better_than_reference_crps": int(np.sum(finite_crps_gain > 0)),
            "fraction_sites_better_than_reference_crps": (
                float(np.mean(finite_crps_gain > 0)) if finite_crps_gain.size else float("nan")
            ),
            "fraction_sites_conditional_interval_score_better_than_global": (
                float(np.mean(conditional_better)) if conditional_better else float("nan")
            ),
            "fraction_sites_state_conditioned_interval_score_better_than_lead_band_only": (
                float(np.mean(state_conditioning_better))
                if state_conditioning_better
                else float("nan")
            ),
        }
    ramp_nrmse_gains = np.asarray(
        [
            row["point"]["ramp_periods"]
            .get(  # type: ignore[index]
                "improvement_reference_minus_model", {}
            )
            .get("nrmse", np.nan)
            for row in rows
        ],
        dtype=float,
    )
    ramp_pinball_gains = np.asarray(
        [
            (
                row["ramp_period_probabilistic"].get(  # type: ignore[index]
                    "improvement_reference_minus_model"
                )
                or {}
            ).get("mean_pinball_loss", np.nan)
            for row in rows
        ],
        dtype=float,
    )
    ramp_crps_gains = np.asarray(
        [
            (
                row["ramp_period_probabilistic"].get(  # type: ignore[index]
                    "improvement_reference_minus_model"
                )
                or {}
            ).get("quantile_crps", np.nan)
            for row in rows
        ],
        dtype=float,
    )
    finite_ramp_nrmse = ramp_nrmse_gains[np.isfinite(ramp_nrmse_gains)]
    finite_ramp_pinball = ramp_pinball_gains[np.isfinite(ramp_pinball_gains)]
    finite_ramp_crps = ramp_crps_gains[np.isfinite(ramp_crps_gains)]
    return {
        "site_count": len(rows),
        "equal_weight_mean_daylight_nrmse": float(np.nanmean(daylight_nrmse)),
        "equal_weight_std_daylight_nrmse": (
            float(np.nanstd(daylight_nrmse, ddof=1)) if len(daylight_nrmse) > 1 else 0.0
        ),
        "sites_with_positive_smart_persistence_skill": int(np.sum(finite_skill > 0)),
        "fraction_sites_with_positive_smart_persistence_skill": (
            float(np.mean(finite_skill > 0)) if finite_skill.size else float("nan")
        ),
        "by_reported_lead": by_lead,
        "probabilistic_by_reported_lead": probability_by_lead,
        "ramp_periods_vs_validation_selected_references": {
            "fraction_sites_better_nrmse": (
                float(np.mean(finite_ramp_nrmse > 0)) if finite_ramp_nrmse.size else float("nan")
            ),
            "fraction_sites_better_pinball": (
                float(np.mean(finite_ramp_pinball > 0))
                if finite_ramp_pinball.size
                else float("nan")
            ),
            "fraction_sites_better_quantile_crps": (
                float(np.mean(finite_ramp_crps > 0)) if finite_ramp_crps.size else float("nan")
            ),
            "evaluated_sites_nrmse": int(finite_ramp_nrmse.size),
            "evaluated_sites_probability": int(finite_ramp_crps.size),
        },
        "note": "Each site receives equal weight; pooled cell metrics are reported separately.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    from racepv.cli.modern_baseline import _add_prepared_expectations

    _add_prepared_expectations(parser)
    parser.add_argument("--calibration-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--calibration-windows", type=Path, required=True)
    parser.add_argument("--test-windows", type=Path, required=True)
    for role in REFERENCE_SELECTION_METRICS:
        option_role = role.replace("_", "-")
        parser.add_argument(
            f"--{option_role}-reference-predictions",
            dest=f"{role}_reference_predictions",
            type=Path,
            help=(
                f"optional explicit {role} reference path; must equal the seed-specific "
                "artifact resolved from its manifest"
            ),
        )
        parser.add_argument(
            f"--{option_role}-reference-selection-manifest",
            dest=f"{role}_reference_selection_manifest",
            type=Path,
            help=(
                f"validation-only {role} group-selection manifest; automatically resolves "
                "the current test seed"
            ),
        )
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--max-bootstrap-candidate-blocks",
        type=int,
        default=512,
        help=(
            "maximum evenly spaced contiguous-day candidate blocks evaluated per site; "
            "fixed before test execution to bound event/bootstrap cost"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--calibration-block-seed",
        type=int,
        default=20260804,
        help="registered value-independent site-day subsampling seed",
    )
    parser.add_argument(
        "--block-reducer",
        choices=("subsample", "none", "max"),
        default="subsample",
        help=(
            "subsample is registered site-day-balanced marginal CQR; none is an "
            "unblocked cell diagnostic; max targets simultaneous coverage within each "
            "conditional site-day stratum (or the whole site-day for global CQR)"
        ),
    )
    parser.add_argument(
        "--min-group-scores",
        type=int,
        help=(
            "override conditional fallback threshold in effective scores (site-days for "
            "subsample/max); required explicitly with daily max blocks"
        ),
    )
    parser.add_argument(
        "--allow-development-data",
        action="store_true",
        help="allow provenance-complete synthetic/debug data; never for reported evidence",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from racepv.cli.modern_baseline import _prepared_expectations
    from racepv.data.weather import expected_origin_contract

    expectations = _prepared_expectations(args)
    origin_contract = expected_origin_contract(
        expectations["expected_version"], expectations["expected_profile"]
    )
    calibration_metadata = _load_metadata(args.calibration_predictions)
    test_metadata = _load_metadata(args.test_predictions)
    for path, metadata in (
        (args.calibration_windows, calibration_metadata),
        (args.test_windows, test_metadata),
    ):
        if expectations["expected_version"] is not None or _prepared_metadata_claim(metadata):
            _admit_v2_prediction_source(
                path, metadata, **expectations, allow_development_data=args.allow_development_data
            )
    require_registered_prediction_input(args.calibration_predictions)
    require_registered_prediction_input(args.test_predictions)
    require_registered_input(args.calibration_windows)
    require_registered_input(args.test_windows)
    for role in REFERENCE_SELECTION_METRICS:
        predictions = getattr(args, f"{role}_reference_predictions")
        manifest = getattr(args, f"{role}_reference_selection_manifest")
        if predictions is not None:
            require_registered_prediction_input(predictions)
        if manifest is not None:
            require_registered_json_input(manifest)
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")
    if args.max_bootstrap_candidate_blocks < 2:
        raise ValueError("--max-bootstrap-candidate-blocks must be at least two")
    for role in REFERENCE_SELECTION_METRICS:
        predictions = getattr(args, f"{role}_reference_predictions")
        manifest = getattr(args, f"{role}_reference_selection_manifest")
        if predictions is not None and manifest is None:
            raise ValueError(
                f"--{role}-reference-predictions is only an override check and requires "
                f"--{role}-reference-selection-manifest"
            )
    if args.block_reducer == "max" and args.min_group_scores is None:
        raise ValueError(
            "--block-reducer max reduces to one score per site-day; specify a preregistered "
            "--min-group-scores block threshold instead of reusing the 200-cell threshold"
        )
    config = load_config(args.config)
    levels = tuple(float(value) for value in config.quantiles or QUANTILE_LEVELS)
    calibration = _load_npz(args.calibration_predictions)
    test = _load_npz(args.test_predictions)
    require_registered_input(args.calibration_predictions, metadata=calibration_metadata)
    require_registered_input(args.test_predictions, metadata=test_metadata)
    _validate_prediction_archive(
        calibration,
        calibration_metadata,
        split="calibration",
        levels=levels,
        horizon=config.data.horizon,
        power_upper_bound=config.data.power_upper_bound,
    )
    _validate_prediction_archive(
        test,
        test_metadata,
        split="test",
        levels=levels,
        horizon=config.data.horizon,
        power_upper_bound=config.data.power_upper_bound,
    )
    _validate_prediction_pair(calibration, calibration_metadata, test, test_metadata)

    cal_daylight = calibration["daylight_mask"].astype(bool)
    test_daylight = test["daylight_mask"].astype(bool)
    targets = test["targets"]
    quantiles = test["quantiles"]
    state_probs = test["state_probs"]
    labels = test["state_labels"].astype(int)
    median_index = levels.index(0.5)
    median = quantiles[..., median_index]
    test_origin_step_minutes = _infer_origin_step_minutes(
        test["timestamps"],
        test.get("site_ids"),
        target_step_minutes=config.data.frequency_minutes,
        contract=origin_contract,
    )

    calibration_window_dataset = _validate_source_windows(
        args.calibration_windows,
        calibration,
        calibration_metadata,
        label="calibration",
        allow_development_data=args.allow_development_data,
        **expectations,
    )
    require_registered_input(args.calibration_windows, metadata=calibration_window_dataset.schema)
    if calibration_window_dataset.schema is None:
        raise RuntimeError("registered calibration schema unexpectedly missing")
    calibration_roles = set(
        str(value) for value in calibration_window_dataset.schema["spatial_roles"].values()
    )
    if "zero_parameter_test" in calibration_roles:
        raise ValueError("conditional calibration cannot be fitted on zero_parameter_test sites")
    test_window_dataset = _validate_source_windows(
        args.test_windows,
        test,
        test_metadata,
        label="test",
        allow_development_data=args.allow_development_data,
        **expectations,
    )
    require_registered_input(args.test_windows, metadata=test_window_dataset.schema)
    evaluation_temporal_audit = require_global_split_chronology(
        {
            "calibration": calibration_window_dataset,
            "test": test_window_dataset,
        }
    )
    evaluation_spatial_protocol = require_registered_spatial_protocol(
        {
            "calibration": calibration_window_dataset,
            "test": test_window_dataset,
        }
    )
    smart, clear_sky = _smart_persistence(test_window_dataset)
    references: dict[str, dict[str, Any]] = {}
    reference_cache: dict[Path, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
    try:
        evaluation_seed = int(test_metadata["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("evaluated test prediction metadata must contain an integer seed") from exc
    for role in REFERENCE_SELECTION_METRICS:
        explicit_reference_path = getattr(args, f"{role}_reference_predictions")
        manifest_path = getattr(args, f"{role}_reference_selection_manifest")
        if manifest_path is None:
            continue
        reference_path = _resolve_reference_path_from_manifest(
            manifest_path,
            expected_role=role,
            seed=evaluation_seed,
        )
        require_registered_prediction_input(reference_path)
        if (
            explicit_reference_path is not None
            and explicit_reference_path.resolve() != reference_path.resolve()
        ):
            raise ValueError(
                f"explicit {role} reference path differs from the manifest-selected "
                f"artifact for seed {evaluation_seed}"
            )
        cache_key = reference_path.resolve()
        if cache_key not in reference_cache:
            loaded_reference = _load_npz(reference_path)
            loaded_metadata = _load_metadata(reference_path)
            _validate_prediction_archive(
                loaded_reference,
                loaded_metadata,
                split="test",
                levels=levels,
                horizon=config.data.horizon,
                power_upper_bound=config.data.power_upper_bound,
            )
            reference_cache[cache_key] = (loaded_reference, loaded_metadata)
        reference, reference_metadata = reference_cache[cache_key]
        if int(reference_metadata.get("seed", -1)) != evaluation_seed:
            raise ValueError(f"{role} reference seed differs from evaluated model seed")
        reference_selection = _validate_reference_selection(
            manifest_path,
            reference_path,
            reference_metadata,
            expected_role=role,
        )
        if not _metadata_window_identities_match(reference_metadata, test_metadata):
            raise ValueError(
                f"{role} reference predictions were not generated from the same test archive"
            )
        if {str(key): str(value) for key, value in reference_metadata["site_id_map"].items()} != {
            str(key): str(value) for key, value in test_metadata["site_id_map"].items()
        }:
            raise ValueError(f"{role} reference site map differs from evaluated test data")
        for name in (
            "targets",
            "state_labels",
            "daylight_mask",
            "timestamps",
            "day_ids",
            "site_ids",
        ):
            if not np.array_equal(reference[name], test[name]):
                raise ValueError(f"{role} reference and model predictions differ at {name}")
        references[role] = {
            "name": str(reference_selection["selected_baseline"]),
            "selection": reference_selection,
            "path": str(reference_path),
            "prediction_sha256": sha256_file(reference_path),
            "metadata": reference_metadata,
            "quantiles": reference["quantiles"],
            "median": reference["quantiles"][..., median_index],
            "state_probs": reference["state_probs"],
        }

    point_reference = references.get("point")
    probabilistic_reference = references.get("probabilistic")
    ramp_point_reference = references.get("ramp_point")
    ramp_probabilistic_reference = references.get("ramp_probabilistic")
    ramp_reference = references.get("ramp")

    raw_intervals = _raw_intervals(quantiles, levels, config.calibration.coverages)
    calibration_blocks = _site_day_ids(
        calibration["day_ids"],
        calibration["site_ids"],
        calibration_metadata["site_id_map"],
    )
    calibration_cell_ids = _forecast_cell_ids(
        calibration["timestamps"], config.data.frequency_minutes
    )
    test_blocks = _site_day_ids(
        test["day_ids"],
        test["site_ids"],
        test_metadata["site_id_map"],
    )
    test_cell_ids = _forecast_cell_ids(test["timestamps"], config.data.frequency_minutes)
    calibration_min_group_scores = (
        config.calibration.min_group_scores
        if args.min_group_scores is None
        else args.min_group_scores
    )
    calibrator = TimeBlockedConditionalCQR(
        quantile_levels=levels,
        coverages=config.calibration.coverages,
        step_minutes=config.data.frequency_minutes,
        min_group_size=calibration_min_group_scores,
        block_reducer=args.block_reducer,
        block_seed=args.calibration_block_seed,
        bounds=(0.0, config.data.power_upper_bound),
        samples_per_day=config.calibration.steps_per_day,
    ).fit(
        calibration["targets"],
        calibration["quantiles"],
        calibration["state_probs"],
        timestamps=calibration_blocks,
        sample_ids=calibration_cell_ids,
        mask=cal_daylight,
    )
    conditional_intervals = calibrator.transform(
        quantiles,
        state_probs,
        night_mask=~test_daylight,
    )
    lead_band_calibrator = LeadBandCQR(
        quantile_levels=levels,
        coverages=config.calibration.coverages,
        step_minutes=config.data.frequency_minutes,
        block_reducer=args.block_reducer,
        block_seed=args.calibration_block_seed,
        bounds=(0.0, config.data.power_upper_bound),
        samples_per_day=config.calibration.steps_per_day,
    ).fit(
        calibration["targets"],
        calibration["quantiles"],
        timestamps=calibration_blocks,
        sample_ids=calibration_cell_ids,
        mask=cal_daylight,
    )
    lead_band_intervals = lead_band_calibrator.transform(
        quantiles,
        night_mask=~test_daylight,
    )
    global_calibrator = GlobalCQR(
        quantile_levels=levels,
        coverages=config.calibration.coverages,
        step_minutes=config.data.frequency_minutes,
        block_reducer=args.block_reducer,
        block_seed=args.calibration_block_seed,
        bounds=(0.0, config.data.power_upper_bound),
        samples_per_day=config.calibration.steps_per_day,
    ).fit(
        calibration["targets"],
        calibration["quantiles"],
        timestamps=calibration_blocks,
        sample_ids=calibration_cell_ids,
        mask=cal_daylight,
    )
    global_intervals = global_calibrator.transform(
        quantiles,
        night_mask=~test_daylight,
    )
    point = _scoped_metrics(targets, median, test_daylight, smart)
    valid_ramp = test_daylight & (labels != 0) & (labels >= 0)
    ramp_point_values = smart if ramp_point_reference is None else ramp_point_reference["median"]
    ramp_point_reference_name = (
        "smart_persistence" if ramp_point_reference is None else str(ramp_point_reference["name"])
    )
    ramp_point_model = point_forecast_metrics(
        targets,
        median,
        daylight_mask=valid_ramp,
    )["daylight"]
    ramp_point_reference_metrics = point_forecast_metrics(
        targets,
        ramp_point_values,
        daylight_mask=valid_ramp,
    )["daylight"]
    point["ramp_periods"] = {
        "nmae_nrmse": ramp_point_model,
        "validation_selected_reference": {
            "name": ramp_point_reference_name,
            **ramp_point_reference_metrics,
        },
        "improvement_reference_minus_model": {
            "nmae": float(ramp_point_reference_metrics["nmae"]) - float(ramp_point_model["nmae"]),
            "nrmse": float(ramp_point_reference_metrics["nrmse"])
            - float(ramp_point_model["nrmse"]),
        },
    }
    raw_probability = _probability_metrics_with_valid_states(
        targets,
        quantiles,
        levels,
        intervals=raw_intervals,
        labels=labels,
        daylight=test_daylight,
    )
    conditional_probability = _probability_metrics_with_valid_states(
        targets,
        quantiles,
        levels,
        intervals=conditional_intervals,
        labels=labels,
        daylight=test_daylight,
    )
    lead_band_probability = _probability_metrics_with_valid_states(
        targets,
        quantiles,
        levels,
        intervals=lead_band_intervals,
        labels=labels,
        daylight=test_daylight,
    )
    global_probability = _probability_metrics_with_valid_states(
        targets,
        quantiles,
        levels,
        intervals=global_intervals,
        labels=labels,
        daylight=test_daylight,
    )
    conditional_calibration_audit = calibrator.calibration_report()
    cqr_acceptance = _cqr_acceptance_report(
        raw_probability,
        conditional_probability,
        conditional_calibration_audit,
        coverages=config.calibration.coverages,
        min_group_size=calibration_min_group_scores,
    )
    probability = {
        "primary_result": "raw_quantiles_and_intervals",
        "raw_quantiles_and_intervals": raw_probability,
        "conditional_cqr_intervals": {
            "intervals": conditional_probability["intervals"],
            "diagnostic_only": cqr_acceptance["diagnostic_only"],
            "acceptance": cqr_acceptance,
            "note": "CQR changes interval bounds, not the underlying pinball/CRPS values.",
        },
        "lead_band_only_cqr_ablation_intervals": {
            "intervals": lead_band_probability["intervals"],
            "note": (
                "All-state CQR retains the same lead bands and block sampler; comparison "
                "with conditional CQR isolates predicted-state conditioning."
            ),
        },
        "global_cqr_ablation_intervals": {
            "intervals": global_probability["intervals"],
            "note": "Global CQR changes interval bounds, not the underlying pinball/CRPS values.",
        },
        "true_state_by_lead_band": _true_state_lead_interval_report(
            targets, labels, conditional_intervals, test_daylight
        ),
        "lead_band_only_true_state_by_lead_band": _true_state_lead_interval_report(
            targets, labels, lead_band_intervals, test_daylight
        ),
        "global_true_state_by_lead_band": _true_state_lead_interval_report(
            targets, labels, global_intervals, test_daylight
        ),
        "true_state_quantile_scores": _true_state_quantile_score_report(
            targets,
            labels,
            quantiles,
            levels,
            test_daylight,
            reference_quantiles=(
                None if probabilistic_reference is None else probabilistic_reference["quantiles"]
            ),
            reference_name=(
                None if probabilistic_reference is None else str(probabilistic_reference["name"])
            ),
            ramp_reference_quantiles=(
                None
                if ramp_probabilistic_reference is None
                else ramp_probabilistic_reference["quantiles"]
            ),
            ramp_reference_name=(
                None
                if ramp_probabilistic_reference is None
                else str(ramp_probabilistic_reference["name"])
            ),
        ),
        "predicted_condition_report": calibrator.coverage_report(
            targets,
            conditional_intervals,
            state_probs,
            timestamps=test_blocks,
            sample_ids=test_cell_ids,
            mask=test_daylight,
        ),
        "global_registered_estimand_report": global_calibrator.coverage_report(
            targets,
            global_intervals,
            timestamps=test_blocks,
            sample_ids=test_cell_ids,
            mask=test_daylight,
        ),
        "lead_band_only_registered_estimand_report": lead_band_calibrator.coverage_report(
            targets,
            lead_band_intervals,
            timestamps=test_blocks,
            sample_ids=test_cell_ids,
            mask=test_daylight,
        ),
        "interval_score_ablation_differences": {
            str(coverage): {
                "conditional_minus_lead_band_only": float(
                    conditional_probability["intervals"][coverage]["interval_score"]  # type: ignore[index]
                    - lead_band_probability["intervals"][coverage]["interval_score"]  # type: ignore[index]
                ),
                "lead_band_only_minus_global": float(
                    lead_band_probability["intervals"][coverage]["interval_score"]  # type: ignore[index]
                    - global_probability["intervals"][coverage]["interval_score"]  # type: ignore[index]
                ),
                "interpretation": (
                    "The first contrast isolates predicted-state conditioning; the second "
                    "isolates physical lead-band conditioning. Negative favors the first "
                    "method named in each contrast."
                ),
            }
            for coverage in conditional_intervals
        },
    }
    if probabilistic_reference is not None:
        reference_probability = probabilistic_forecast_metrics(
            targets,
            probabilistic_reference["quantiles"],
            levels,
            mask=test_daylight,
        )
        probability["validation_selected_probabilistic_reference"] = {
            "name": probabilistic_reference["name"],
            "selection": probabilistic_reference["selection"],
            **reference_probability,
        }
        probability["improvement_reference_minus_model"] = {
            "mean_pinball_loss": float(reference_probability["mean_pinball_loss"])
            - float(raw_probability["mean_pinball_loss"]),
            "quantile_crps": float(reference_probability["quantile_crps"])
            - float(raw_probability["quantile_crps"]),
        }
    ramp = {
        "three_state": ramp_classification_metrics(labels, state_probs, mask=test_daylight),
        "binary": binary_ramp_metrics(
            labels,
            state_probs,
            mask=test_daylight,
            threshold=_ramp_decision_threshold(test_metadata),
        ),
        "events": _event_report(
            targets,
            labels,
            median,
            state_probs,
            test.get("timestamps"),
            test.get("site_ids"),
            test_metadata["site_id_map"],
        ),
        "event_daily_moving_block_bootstrap": _event_daily_block_bootstrap_report(
            targets,
            labels,
            median,
            state_probs,
            test["timestamps"],
            test.get("site_ids"),
            test_metadata["site_id_map"],
            n_bootstrap=args.bootstrap,
            seed=args.seed,
            step_minutes=config.data.frequency_minutes,
            origin_step_minutes=test_origin_step_minutes,
            maximum_candidate_blocks=args.max_bootstrap_candidate_blocks,
            contract=origin_contract,
        ),
        "moving_block_bootstrap": _bootstrap_report(
            labels,
            state_probs,
            test.get("site_ids"),
            test["timestamps"],
            test_metadata["site_id_map"],
            None if ramp_reference is None else ramp_reference["state_probs"],
            None if ramp_reference is None else str(ramp_reference["name"]),
            n_bootstrap=args.bootstrap,
            seed=args.seed,
            origin_step_minutes=test_origin_step_minutes,
            maximum_candidate_blocks=args.max_bootstrap_candidate_blocks,
        ),
    }

    per_site = _per_site_report(
        test.get("site_ids"),
        targets,
        median,
        quantiles,
        levels,
        labels,
        state_probs,
        test_daylight,
        conditional_intervals,
        lead_band_intervals,
        global_intervals,
        smart,
        None if ramp_point_reference is None else ramp_point_reference["median"],
        None if ramp_point_reference is None else str(ramp_point_reference["name"]),
        None if probabilistic_reference is None else probabilistic_reference["quantiles"],
        None if probabilistic_reference is None else str(probabilistic_reference["name"]),
        None if ramp_probabilistic_reference is None else ramp_probabilistic_reference["quantiles"],
        None if ramp_probabilistic_reference is None else str(ramp_probabilistic_reference["name"]),
        test_metadata["site_id_map"],
    )
    report: dict[str, object] = {
        "provenance": {
            "calibration_predictions": str(args.calibration_predictions),
            "test_predictions": str(args.test_predictions),
            "calibration_prediction_sha256": sha256_file(args.calibration_predictions),
            "test_prediction_sha256": sha256_file(args.test_predictions),
            "calibration_window_container_sha256": _window_container_sha256(
                args.calibration_windows
            ),
            "test_window_container_sha256": _window_container_sha256(args.test_windows),
            "calibration_window_content_sha256": getattr(
                calibration_window_dataset, "content_identity", None
            ),
            "test_window_content_sha256": getattr(test_window_dataset, "content_identity", None),
            "calibration_metadata": calibration_metadata,
            "test_metadata": test_metadata,
            "validation_selected_references": {
                role: {
                    "path": value["path"],
                    "prediction_sha256": value["prediction_sha256"],
                    "selection": value["selection"],
                    "metadata": value["metadata"],
                }
                for role, value in references.items()
            },
        },
        "protocol": {
            "evidence_mode": {
                "allow_development_data": args.allow_development_data,
                "calibration_registered_evidence_eligible": (
                    calibration_window_dataset.schema["registered_evidence_eligible"]
                ),
                "test_registered_evidence_eligible": test_window_dataset.schema[
                    "registered_evidence_eligible"
                ],
                "test_dataset_kind": test_window_dataset.schema["dataset_kind"],
                "test_simulated": test_window_dataset.schema["simulated"],
            },
            "global_temporal_audit": evaluation_temporal_audit,
            "registered_spatial_protocol": evaluation_spatial_protocol,
            "quantiles": levels,
            "coverages": config.calibration.coverages,
            "calibration_condition": "predicted stable/ramp and physical lead band",
            "calibration_block_reducer": args.block_reducer,
            "calibration_block_seed": args.calibration_block_seed,
            "calibration_coverage_interpretation": (
                "For each predicted-state/lead-band stratum: marginal coverage over "
                "eligible site-days with one value-independent forecast cell selected "
                "per site-day from stable target-time/lead IDs. Site-days, not sites or "
                "forecast cells, receive equal weight."
                if args.block_reducer == "subsample"
                else (
                    "unblocked marginal cell diagnostic; overlapping cells are not "
                    "claimed as independent conformal evidence"
                    if args.block_reducer == "none"
                    else (
                        "For conditional CQR, simultaneous coverage of all eligible cells "
                        "inside each site-day/predicted-state/lead-band stratum using its "
                        "maximum score; lead-band-only CQR pools states within each "
                        "site-day/lead-band stratum; global CQR uses all eligible cells in "
                        "the whole site-day."
                    )
                )
            ),
            "calibration_block_identity": (
                "BLAKE2b-64 digest of the length-prefixed site name and target-day ID "
                "under a fixed version tag, with in-archive collision checking; identities "
                "are never renumbered when other site-days are added or removed"
            ),
            "calibration_subsample_indexing": (
                "blake2b(seed, canonical site-day identity) modulo eligible count after "
                "canonical ordering by stable target-time/lead cell ID; no target, "
                "forecast, state probability, or conformity-score value enters the hash "
                "or ordering (predicted state only defines stratum eligibility)"
                if args.block_reducer == "subsample"
                else None
            ),
            "calibration_fallback_policy": (
                "If a requested predicted-state/lead-band group has fewer than the "
                "registered effective-score threshold, use the all-state correction from "
                "the same lead band. Audit rows report requested-group and actual-source "
                "counts separately."
            ),
            "lead_band_only_cqr_ablation_scope": (
                "all predicted states pooled within each of the same three physical lead "
                "bands, using the identical block reducer, seed, block identities, and "
                "canonical cell ordering as conditional CQR; conditional minus this "
                "control isolates predicted-state conditioning. It is fitted directly and "
                "does not use the conditional min-group fallback threshold."
            ),
            "global_cqr_ablation_scope": (
                "one unconditional correction across all predicted states and all lead "
                "bands; subsample uses one cell per eligible site-day, so this ablation "
                "intentionally removes both state and lead conditioning"
            ),
            "calibration_min_group_scores": calibration_min_group_scores,
            "calibration_min_group_score_unit": (
                "forecast cells"
                if args.block_reducer == "none"
                else "eligible site-day effective scores within the requested stratum"
            ),
            "bootstrap_unit": (
                "overlapping candidate blocks of "
                f"{1440 // test_origin_step_minutes} contiguous forecast origins per site"
            ),
            "maximum_bootstrap_candidate_blocks_per_site": (args.max_bootstrap_candidate_blocks),
            "bootstrap_estimands": {
                "classification": (
                    "mean of precomputed 24-hour block classification metrics within "
                    "each site; not a pooled-cell metric"
                ),
                "event_timing_and_amplitude": (
                    "full continuous fixed-15-minute-lead event-error point estimate "
                    "within each site, with matched-event-weighted percentile replicates "
                    "from resampled 24-hour forecast-origin blocks"
                ),
            },
            "dm_hac_time_axis": "forecast_origin",
            "dm_hac_expected_origin_step_minutes": test_origin_step_minutes,
            "dm_hac_target_step_minutes": config.data.frequency_minutes,
            "dm_hac_overlap_lag": max(
                0,
                int(
                    np.ceil(
                        config.data.horizon
                        * config.data.frequency_minutes
                        / test_origin_step_minutes
                    )
                )
                - 1,
            ),
            "dm_hac_gap_policy": (
                "derive issue time from the saved target-time matrix, split each site at "
                "every non-inferred origin jump, and exclude all cross-segment covariance "
                "pairs while using the overlap lag implied by horizon duration and origin cadence"
            ),
            "reference_policy": (
                "point, probabilistic, and ramp comparators are independently selected on "
                "validation data by equal-seed mean of their manifest metric; one model "
                "class is fixed across the common seed set for each role, and is not a "
                "post-test best baseline for every reported metric"
            ),
        },
        "point": point,
        "probabilistic": probability,
        "ramp": ramp,
        "cqr_acceptance": cqr_acceptance,
        "expert_specialization": _expert_specialization_report(test, levels, test_metadata),
        "calibration_audit": conditional_calibration_audit,
        "lead_band_only_calibration_audit": lead_band_calibrator.calibration_report(),
        "global_calibration_audit": global_calibrator.calibration_report(),
        "per_site": per_site,
        "site_balanced_summary": _balanced_site_summary(per_site),
    }

    target_coverage = 0.90
    if target_coverage not in config.calibration.coverages:
        raise ValueError("registered acceptance diagnostics require a 0.90 interval")
    conditional_at_90 = probability["conditional_cqr_intervals"]["intervals"][target_coverage]  # type: ignore[index]
    lead_band_at_90 = probability["lead_band_only_cqr_ablation_intervals"]["intervals"][
        target_coverage
    ]  # type: ignore[index]
    global_at_90 = probability["global_cqr_ablation_intervals"]["intervals"][target_coverage]  # type: ignore[index]
    group_rows = conditional_at_90["conditional_coverage"]["groups"]  # type: ignore[index]
    stable_coverage = float(group_rows.get("stable", {}).get("coverage", np.nan))
    ramp_coverage = float(group_rows.get("ramp", {}).get("coverage", np.nan))
    conditional_score = float(conditional_at_90["interval_score"])  # type: ignore[index]
    lead_band_score = float(lead_band_at_90["interval_score"])  # type: ignore[index]
    global_score = float(global_at_90["interval_score"])  # type: ignore[index]
    site_coverage_rows = []
    for row in per_site:
        interval_metrics = row["probabilistic"]["intervals"][target_coverage]  # type: ignore[index]
        conditions = interval_metrics["conditional_coverage"]["groups"]  # type: ignore[index]
        stable = float(conditions.get("stable", {}).get("coverage", np.nan))
        ramp_value = float(conditions.get("ramp", {}).get("coverage", np.nan))
        site_coverage_rows.append(
            {
                "site_id": row["site_id"],
                "site_name": row["site_name"],
                "stable_coverage": stable,
                "ramp_coverage": ramp_value,
                "both_in_87_93": bool(0.87 <= stable <= 0.93 and 0.87 <= ramp_value <= 0.93),
                "gap_at_most_0.05": bool(
                    np.isfinite(stable)
                    and np.isfinite(ramp_value)
                    and abs(stable - ramp_value) <= 0.05
                ),
            }
        )
    report["registered_acceptance_diagnostics"] = {
        "coverage_90_stable_in_87_93": bool(0.87 <= stable_coverage <= 0.93),
        "coverage_90_ramp_in_87_93": bool(0.87 <= ramp_coverage <= 0.93),
        "stable_ramp_coverage_gap_at_most_0.05": bool(
            np.isfinite(stable_coverage)
            and np.isfinite(ramp_coverage)
            and abs(stable_coverage - ramp_coverage) <= 0.05
        ),
        "conditional_interval_score_better_than_global": bool(
            np.isfinite(conditional_score)
            and np.isfinite(global_score)
            and conditional_score < global_score
        ),
        "conditional_interval_score_better_than_lead_band_only": bool(
            np.isfinite(conditional_score)
            and np.isfinite(lead_band_score)
            and conditional_score < lead_band_score
        ),
        "observed_stable_coverage": stable_coverage,
        "observed_ramp_coverage": ramp_coverage,
        "conditional_interval_score": conditional_score,
        "lead_band_only_interval_score": lead_band_score,
        "global_interval_score": global_score,
        "per_site_coverage": site_coverage_rows,
        "fraction_sites_meeting_both_coverage_bands": float(
            np.mean([row["both_in_87_93"] for row in site_coverage_rows])
        ),
        "note": "These are diagnostics, not automatic publication claims.",
    }
    report["smart_persistence"] = point_forecast_metrics(
        targets, smart, daylight_mask=test_daylight
    )
    report["clear_sky_persistence"] = point_forecast_metrics(
        targets, clear_sky, daylight_mask=test_daylight
    )
    report["diebold_mariano_vs_smart_persistence"] = _dm_report(
        targets,
        median,
        smart,
        test_daylight,
        test["site_ids"],
        test["timestamps"],
        test_metadata["site_id_map"],
        "smart_persistence",
        step_minutes=config.data.frequency_minutes,
        contract=origin_contract,
    )
    if point_reference is not None:
        report["diebold_mariano_vs_validation_selected_point_reference"] = _dm_report(
            targets,
            median,
            point_reference["median"],
            test_daylight,
            test["site_ids"],
            test["timestamps"],
            test_metadata["site_id_map"],
            str(point_reference["name"]),
            step_minutes=config.data.frequency_minutes,
            contract=origin_contract,
        )
    if ramp_point_reference is not None:
        report["diebold_mariano_ramp_periods_vs_validation_selected_ramp_point_reference"] = (
            _dm_report(
                targets,
                median,
                ramp_point_reference["median"],
                valid_ramp,
                test["site_ids"],
                test["timestamps"],
                test_metadata["site_id_map"],
                str(ramp_point_reference["name"]),
                step_minutes=config.data.frequency_minutes,
                contract=origin_contract,
            )
        )
    if ramp_probabilistic_reference is not None:
        report["diebold_mariano_ramp_quantile_crps_vs_validation_selected_ramp_reference"] = (
            _dm_quantile_report(
                targets,
                quantiles,
                ramp_probabilistic_reference["quantiles"],
                levels,
                valid_ramp,
                test["site_ids"],
                test["timestamps"],
                test_metadata["site_id_map"],
                str(ramp_probabilistic_reference["name"]),
                step_minutes=config.data.frequency_minutes,
                contract=origin_contract,
            )
        )

    args.output.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"conditional_{int(round(coverage * 100))}": interval
        for coverage, interval in conditional_intervals.items()
    }
    arrays.update(
        {
            f"lead_band_only_{int(round(coverage * 100))}": interval
            for coverage, interval in lead_band_intervals.items()
        }
    )
    arrays.update(
        {
            f"global_{int(round(coverage * 100))}": interval
            for coverage, interval in global_intervals.items()
        }
    )
    np.savez_compressed(args.output / "calibrated_intervals.npz", **arrays)
    atomic_json_dump(_json_ready(report), args.output / "evaluation.json")
    atomic_json_dump(
        {
            "block_reducer": args.block_reducer,
            "block_seed": args.calibration_block_seed,
            "block_identity": ("BLAKE2b-64 digest of length-prefixed site name and target-day ID"),
            "subsample_order_basis": (
                "stable target-time/lead cell ID" if args.block_reducer == "subsample" else None
            ),
            "conditional_groups": _json_ready(calibrator.calibration_report()),
            "lead_band_only_groups": _json_ready(lead_band_calibrator.calibration_report()),
            "global_groups": _json_ready(global_calibrator.calibration_report()),
        },
        args.output / "calibration.json",
    )
    print(json.dumps({"output": str(args.output), "evaluated_samples": len(targets)}))


if __name__ == "__main__":
    main()
