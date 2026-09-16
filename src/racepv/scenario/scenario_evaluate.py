"""Calibration, sensitivity evaluation, validation selection, and aggregation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from racepv.calibration import GlobalCQR, TimeBlockedConditionalCQR
from racepv.calibration.cqr import LeadBandCQR
from racepv.metrics import point_forecast_metrics, prediction_interval_coverage, quantile_crps

from .compact_storage import load_prediction_arrays, prediction_entries
from .constants import EVIDENCE_METADATA, QUANTILES, RACE_PV, SEEDS, VALIDATION_SENSITIVITIES
from .evaluation import (
    benjamini_hochberg,
    cohort_moving_block_bootstrap,
    diebold_mariano_test,
    expert_diagnostics,
    fit_cqr,
    forecast_skill,
    per_site_per_lead_summary,
)
from .scenario_artifacts import (
    DISCLOSURE,
    atomic_bytes,
    load_artifact_index,
    register_stage_outputs,
    sha256_file,
    write_json,
)
from .scenario_manifest import (
    all_profiles,
    load_manifest,
    matrix,
    require_stage,
    run_id,
    stage_cohort,
)
from .synthesis import SynthesizedPrediction, synthesize_predictions

SENSITIVITY_NAMES = frozenset(profile.name for profile in VALIDATION_SENSITIVITIES)


def _finite_mean(values: np.ndarray, axis: int | None = None) -> Any:
    """Average finite daylight values without collapsing their time positions."""
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    counts = np.sum(finite, axis=axis)
    sums = np.sum(np.where(finite, array, 0.0), axis=axis)
    if axis is None:
        return float(sums / counts) if int(counts) else float("nan")
    result = np.full(np.shape(sums), np.nan, dtype=np.float64)
    np.divide(sums, counts, out=result, where=counts > 0)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["note"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows or [{"note": DISCLOSURE}])
    atomic_bytes(path, buffer.getvalue().encode("utf-8"))


def _accepted_candidate_seed(root: Path, manifest: dict[str, Any]) -> int:
    audit_path = root / "candidate-audit.json"
    if not audit_path.is_file():
        raise ValueError("generate stage required before evaluate: missing candidate-audit.json")
    return int(json.loads(audit_path.read_text(encoding="utf-8"))["accepted_seed"])


def _metrics(cohort: Any, quantiles: np.ndarray) -> dict[str, Any]:
    median = quantiles[..., list(QUANTILES).index(0.5)]
    return {
        "raw": point_forecast_metrics(cohort.targets, median, daylight_mask=cohort.daylight),
        "quantile_crps": quantile_crps(
            cohort.targets, quantiles, QUANTILES, mask=cohort.daylight
        ),
    }


def sensitivity_settings(model: str) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "coverage": 0.90,
        "ramp_threshold": 0.045,
        "ramp_threshold_source": "registered_fixed_default",
        "bootstrap_days": 7,
        "selected_control": "state_lead",
        "intervention": "reference_state_lead_cqr",
    }
    if model == "a8a_lead_band_cqr":
        settings["selected_control"] = "lead_only"
        settings["intervention"] = "select_lead_only_cqr_output"
    elif model == "a8b_global_cqr":
        settings["selected_control"] = "global"
        settings["intervention"] = "select_global_cqr_output"
    elif model == "ramp_q85":
        settings["ramp_threshold"] = None
        settings["ramp_threshold_source"] = "calibration_q85"
        settings["intervention"] = "validation_derived_ramp_q85"
    elif model == "ramp_q95":
        settings["ramp_threshold"] = None
        settings["ramp_threshold_source"] = "calibration_q95"
        settings["intervention"] = "validation_derived_ramp_q95"
    elif model == "ramp_fixed_01":
        settings["ramp_threshold"] = 0.10
        settings["ramp_threshold_source"] = "fixed_0.10"
        settings["intervention"] = "fixed_ramp_threshold_0.10"
    elif model == "coverage_80":
        settings["coverage"] = 0.80
        settings["intervention"] = "target_coverage_0.80"
    elif model == "coverage_90":
        settings["intervention"] = "target_coverage_0.90"
    elif model == "bootstrap_7day":
        settings["intervention"] = "bootstrap_block_7day"
    elif model == "bootstrap_21day":
        settings["bootstrap_days"] = 21
        settings["intervention"] = "bootstrap_block_21day"
    return settings


def _resolved_settings(
    model: str, calibration: Any, calibration_prediction: SynthesizedPrediction
) -> dict[str, Any]:
    settings = sensitivity_settings(model)
    source = settings["ramp_threshold_source"]
    if source in {"calibration_q85", "calibration_q95"}:
        quantile = 0.85 if source == "calibration_q85" else 0.95
        values = calibration_prediction.ramp_scores[calibration.daylight]
        settings["ramp_threshold"] = float(np.quantile(values, quantile))
        settings["ramp_threshold_quantile"] = quantile
        settings["ramp_threshold_calibration_count"] = int(values.size)
    return settings


def _thresholded_probabilities(
    probabilities: np.ndarray, ramp_scores: np.ndarray, threshold: float
) -> np.ndarray:
    result = np.asarray(probabilities, dtype=float).copy()
    result[ramp_scores < threshold] = (1.0, 0.0, 0.0)
    return result


def _control_metrics(
    calibration: Any,
    target: Any,
    calibration_prediction: SynthesizedPrediction,
    target_prediction: SynthesizedPrediction,
    *,
    coverage: float,
) -> tuple[dict[str, Any], dict[str, list[dict[str, object]]]]:
    controls = {
        "global": GlobalCQR(coverages=(coverage,), block_reducer="none"),
        "lead_only": LeadBandCQR(coverages=(coverage,), block_reducer="none"),
        "state_lead": TimeBlockedConditionalCQR(
            coverages=(coverage,), min_group_size=1, block_reducer="none"
        ),
    }
    metrics: dict[str, Any] = {}
    parameters: dict[str, list[dict[str, object]]] = {}
    for name, calibrator in controls.items():
        fitted = fit_cqr(
            calibrator,
            calibration.targets,
            calibration_prediction.quantiles,
            role="source_calibration",
            predicted_states=calibration_prediction.state_probabilities,
            mask=calibration.daylight,
        )
        interval = fitted.transform(
            target_prediction.quantiles,
            target_prediction.state_probabilities,
            night_mask=~target.daylight,
        )[coverage]
        parameters[name] = fitted.calibration_report()
        metrics[name] = {
            "picp": prediction_interval_coverage(
                target.targets, interval[..., 0], interval[..., 1], mask=target.daylight
            ),
            "mean_width": float(
                np.mean(interval[..., 1][target.daylight] - interval[..., 0][target.daylight])
            ),
            "coverage_target": coverage,
            "calibration_role": "source_calibration",
        }
    return metrics, parameters


def _raw_interval(
    cohort: Any, quantiles: np.ndarray, *, coverage: float
) -> tuple[dict[str, Any], np.ndarray]:
    lower, upper = (
        (quantiles[..., 1], quantiles[..., -2])
        if coverage == 0.80
        else (quantiles[..., 0], quantiles[..., -1])
    )
    interval = np.stack((lower, upper), axis=-1)
    return (
        {
            "picp": prediction_interval_coverage(
                cohort.targets, lower, upper, mask=cohort.daylight
            ),
            "mean_width": float(np.mean(upper[cohort.daylight] - lower[cohort.daylight])),
            "coverage_target": coverage,
            "calibration_role": "none_raw_base_quantiles",
        },
        interval,
    )


def _artifact_identifier(root: Path, relative: str) -> str:
    index = load_artifact_index(root)
    for identifier, entry in index["artifacts"].items():
        if entry["path"] == relative:
            return identifier
    raise ValueError(f"artifact DAG identity missing for {relative}")


def _array_input_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        value = np.ascontiguousarray(array)
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def evaluate(root: Path) -> None:
    require_stage(root, "generate", ("candidate-audit.json", "predictions"))
    manifest = load_manifest(root)
    entries = prediction_entries(root)
    if len(entries) != len(manifest["run_identities"]):
        raise ValueError("generate stage required before evaluate: prediction matrix incomplete")
    cohorts = {
        split: stage_cohort(manifest, split)
        for split in ("calibration", "validation", "test")
    }
    calibration = cohorts["calibration"]
    accepted_seed = _accepted_candidate_seed(root, manifest)
    records: list[dict[str, Any]] = []
    outputs: list[Path] = []
    sources_by_output: dict[str, list[Path | str]] = {}
    input_cache: dict[tuple[str, int], tuple[SynthesizedPrediction, dict[str, Any]]] = {}

    for _entry_path, sidecar, source_sidecar in entries:
        source_path = root / (source_sidecar.get("path") or source_sidecar["prediction_path"])
        target_arrays = load_prediction_arrays(root, source_sidecar)
        target_cohort = cohorts[sidecar["split"]]
        base_model = RACE_PV.name if sidecar["model"] in SENSITIVITY_NAMES else sidecar["model"]
        base_key = (base_model, int(sidecar["seed"]))
        if base_key not in input_cache:
            profile = next(
                profile
                for profile in all_profiles(include_sensitivities=False)
                if profile.name == base_model
            )
            calibration_profile = replace(
                profile, applies_to=(*profile.applies_to, "calibration")
            )
            calibration_prediction = synthesize_predictions(
                calibration,
                calibration_profile,
                seed=int(sidecar["seed"]),
                scenario_seed=accepted_seed,
            )
            input_identity = f"source_calibration/{base_model}/{sidecar['seed']}"
            input_path = root / "evaluations" / "calibration" / "inputs" / (
                f"{input_identity.replace('/', '__')}.json"
            )
            input_arrays = {
                "quantiles": calibration_prediction.quantiles,
                "targets": calibration.targets,
                "daylight": calibration.daylight,
                "state_probabilities": calibration_prediction.state_probabilities,
                "ramp_scores": calibration_prediction.ramp_scores,
            }
            input_digest = _array_input_sha256(input_arrays)
            write_json(
                input_path,
                {
                    **dict(EVIDENCE_METADATA),
                    "disclosure": DISCLOSURE,
                    "run_id": input_identity,
                    "model": base_model,
                    "seed": sidecar["seed"],
                    "role": "source_calibration",
                    "input_sha256": input_digest,
                    "cohort_sha256": manifest["cohorts"]["calibration"]["cohort_sha256"],
                    "scenario_seed": accepted_seed,
                    "array_contract": {
                        name: {"dtype": str(value.dtype), "shape": list(value.shape)}
                        for name, value in sorted(input_arrays.items())
                    },
                    "persistence": "parameters_only_no_calibration_matrices",
                },
            )
            outputs.append(input_path)
            input_cache[base_key] = (
                calibration_prediction,
                {
                    "run_id": input_identity,
                    "path": str(input_path.relative_to(root)),
                    "sha256": sha256_file(input_path),
                    "input_sha256": input_digest,
                    "role": "source_calibration",
                    "cohort_sha256": manifest["cohorts"]["calibration"]["cohort_sha256"],
                },
            )
        calibration_prediction, calibration_artifact = input_cache[base_key]
        settings = _resolved_settings(sidecar["model"], calibration, calibration_prediction)
        calibration_probabilities = _thresholded_probabilities(
            calibration_prediction.state_probabilities,
            calibration_prediction.ramp_scores,
            float(settings["ramp_threshold"]),
        )
        target_probabilities = _thresholded_probabilities(
            target_arrays["state_probabilities"],
            target_arrays["ramp_scores"],
            float(settings["ramp_threshold"]),
        )
        calibrated_input = replace(
            calibration_prediction, state_probabilities=calibration_probabilities
        )
        target_prediction = SynthesizedPrediction(
            sidecar["model"],
            int(sidecar["seed"]),
            target_arrays["quantiles"],
            target_arrays["state_labels"],
            target_probabilities,
            target_arrays["ramp_scores"],
        )
        controls, control_parameters = _control_metrics(
            calibration,
            target_cohort,
            calibrated_input,
            target_prediction,
            coverage=float(settings["coverage"]),
        )
        raw_metrics, _ = _raw_interval(
            target_cohort,
            target_arrays["quantiles"],
            coverage=float(settings["coverage"]),
        )
        output_path = (
            root
            / "evaluations"
            / "calibration"
            / "outputs"
            / f"{sidecar['run_id'].replace('/', '__')}.json"
        )
        write_json(
            output_path,
            {
                **dict(EVIDENCE_METADATA),
                "storage_schema": "compact_cqr_parameters_v2",
                "run_id": f"calibration_output/{sidecar['run_id']}",
                "input_sha256": calibration_artifact["input_sha256"],
                "target_prediction_sha256": source_sidecar["content_sha256"],
                "control_parameters": control_parameters,
                "interval_metrics": {"raw": raw_metrics, **controls},
                "control_config": settings,
            },
        )
        prediction_relative = str(source_path.relative_to(root))
        prediction_identifier = _artifact_identifier(root, prediction_relative)
        selected_control = str(settings["selected_control"])
        output_artifact = {
            "run_id": f"calibration_output/{sidecar['run_id']}",
            "path": str(output_path.relative_to(root)),
            "sha256": sha256_file(output_path),
            "input_run_id": calibration_artifact["run_id"],
                "input_sha256": calibration_artifact["input_sha256"],
            "target_run_id": sidecar["run_id"],
            "target_prediction_sha256": source_sidecar["content_sha256"],
            "control_config": settings,
        }
        record: dict[str, Any] = {
            **dict(EVIDENCE_METADATA),
            "disclosure": DISCLOSURE,
            "run_id": sidecar["run_id"],
            "model": sidecar["model"],
            "seed": int(sidecar["seed"]),
            "split": sidecar["split"],
            "metrics": _metrics(target_cohort, target_arrays["quantiles"]),
            "interval_metrics": {"raw": raw_metrics, **controls},
            "sensitivity_settings": {
                **settings,
                "base_prediction_model": base_model,
                "shared_base_prediction_sha256": source_sidecar["content_sha256"],
                "base_prediction_artifact_id": prediction_identifier,
            },
            "sensitivity_result": {
                "selected_control": selected_control,
                "selected_interval_metrics": controls[selected_control],
                "ramp_threshold": settings["ramp_threshold"],
                "coverage": settings["coverage"],
                "bootstrap_days": settings["bootstrap_days"],
            },
            "prediction_artifact_id": prediction_identifier,
            "prediction_artifact_path": prediction_relative,
            "prediction_artifact_sha256": source_sidecar["content_sha256"],
            "calibration_input_artifact": dict(calibration_artifact),
            "calibration_output_artifact": output_artifact,
            "calibration_artifact": output_artifact,
            "calibration_audit": {
                "calibration_artifact_identity": calibration_artifact["run_id"],
                "calibration_artifact_sha256": calibration_artifact["sha256"],
                "calibration_role": calibration_artifact["role"],
                "calibration_cohort_sha256": calibration_artifact["cohort_sha256"],
                "raw_interval": raw_metrics,
                "transformed_controls": controls,
                "selected_control": selected_control,
                "control_config": settings,
            },
            "site_lead": per_site_per_lead_summary(
                target_cohort.targets,
                target_arrays["quantiles"],
                site_ids=target_cohort.site_ids,
                daylight=target_cohort.daylight,
                levels=QUANTILES,
            ),
        }
        if base_model == RACE_PV.name and sidecar["model"] == RACE_PV.name:
            required = {"expert_quantiles", "gates", "state_labels"}
            if not required <= set(target_arrays):
                raise ValueError(f"RaCE-PV archive lacks persisted MoE arrays: {source_path.name}")
            record["expert_diagnostics"] = expert_diagnostics(
                target_arrays["expert_quantiles"],
                target_arrays["gates"],
                true_states=target_arrays["state_labels"],
                y_true=target_cohort.targets,
            )
            record["expert_diagnostics"]["source"] = "persisted_moe_arrays"
        safe_identity = sidecar["run_id"].replace("/", "__")
        evaluation_path = root / "evaluations" / f"{safe_identity}.json"
        diagnostics_path = root / "evaluations" / f"{safe_identity}.diagnostics.csv"
        write_json(evaluation_path, record)
        _write_csv(
            diagnostics_path,
            [
                {
                    "run_id": record["run_id"],
                    "site_id": item["site_id"],
                    "lead": item["lead"],
                    "quantile_crps": item["quantile_crps"],
                    "note": DISCLOSURE,
                }
                for item in record["site_lead"]
            ],
        )
        outputs.extend((output_path, evaluation_path, diagnostics_path))
        sources_by_output[str(output_path.relative_to(root))] = [
            calibration_artifact["path"],
            source_path,
        ]
        sources_by_output[str(evaluation_path.relative_to(root))] = [
            output_path,
            calibration_artifact["path"],
            source_path,
        ]
        sources_by_output[str(diagnostics_path.relative_to(root))] = [evaluation_path]
        records.append(record)
    evaluations_path = root / "evaluations.json"
    write_json(
        evaluations_path,
        {**dict(EVIDENCE_METADATA), "disclosure": DISCLOSURE, "runs": records},
    )
    outputs.append(evaluations_path)
    sources_by_output[str(evaluations_path.relative_to(root))] = [
        path for path in outputs if path.parent == root / "evaluations" and path.suffix == ".json"
    ]
    register_stage_outputs(
        root,
        "evaluate",
        outputs,
        source_paths=("manifest.json", "candidate-audit.json"),
        source_paths_by_output=sources_by_output,
    )


def _load_prediction_losses(root: Path) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    losses: dict[str, np.ndarray] = {}
    sidecars: dict[str, dict[str, Any]] = {}
    for _path, logical, source in prediction_entries(root):
        arrays = load_prediction_arrays(root, source)
        median = arrays["quantiles"][..., list(QUANTILES).index(0.5)]
        squared = np.where(
            arrays["daylight"], (median - arrays["targets"]) ** 2, np.nan
        )
        sidecars[logical["run_id"]] = (
            logical
            if "content_sha256" in logical
            else {**logical, "content_sha256": source["content_sha256"]}
        )
        finite = np.isfinite(squared)
        counts = np.sum(finite, axis=1)
        origin_losses = np.full(squared.shape[0], np.nan, dtype=np.float64)
        np.divide(
            np.sum(np.where(finite, squared, 0.0), axis=1, dtype=np.float64),
            counts,
            out=origin_losses,
            where=counts > 0,
        )
        losses[logical["run_id"]] = origin_losses
    return losses, sidecars


def aggregate(root: Path) -> None:
    require_stage(root, "evaluate", ("evaluations.json", "evaluations"))
    manifest = load_manifest(root)
    protocol = str(manifest["protocols"][0])

    def identity(split: str, model: str, seed: int) -> str:
        return run_id(protocol, model, seed, split)

    records = json.loads((root / "evaluations.json").read_text(encoding="utf-8"))["runs"]
    expected_ids = {item["run_id"] for item in manifest["run_identities"]}
    actual_ids = {item["run_id"] for item in records}
    if actual_ids != expected_ids or len(records) != len(expected_ids):
        raise ValueError("aggregate requires complete 30-model/5-seed matrix")
    losses, sidecars = _load_prediction_losses(root)
    cohorts = {split: stage_cohort(manifest, split) for split in ("validation", "test")}
    rows: list[dict[str, Any]] = []
    for split, profile in matrix():
        model_records = [
            item
            for item in records
            if item["model"] == profile.name and item["split"] == split
        ]
        if sorted(item["seed"] for item in model_records) != list(SEEDS):
            raise ValueError(f"aggregate requires exact seed set for {profile.name}")
        values = [float(item["metrics"]["quantile_crps"]) for item in model_records]
        settings = sensitivity_settings(profile.name)
        combined = _finite_mean(
            np.stack([losses[identity(split, profile.name, seed)] for seed in SEEDS]),
            axis=0,
        )
        bootstrap = cohort_moving_block_bootstrap(
            combined,
            cohorts[split],
            statistic=_finite_mean,
            n_bootstrap=64,
            random_state=20260814,
            block_days=int(settings["bootstrap_days"]),
        )
        selected_control = model_records[0]["sensitivity_result"]["selected_control"]
        selected_metrics = [
            item["sensitivity_result"]["selected_interval_metrics"] for item in model_records
        ]
        rows.append(
            {
                "model": profile.name,
                "split": split,
                "mean_quantile_crps": float(np.mean(values)),
                "std_quantile_crps": float(np.std(values, ddof=1)),
                "daylight_nrmse": float(
                    np.mean(
                        [item["metrics"]["raw"]["daylight"]["nrmse"] for item in model_records]
                    )
                ),
                "selected_control": selected_control,
                "selected_picp": float(np.mean([item["picp"] for item in selected_metrics])),
                "selected_mean_width": float(
                    np.mean([item["mean_width"] for item in selected_metrics])
                ),
                "ramp_threshold": float(
                    np.mean(
                        [item["sensitivity_settings"]["ramp_threshold"] for item in model_records]
                    )
                ),
                "bootstrap_metric": "daylight_origin_mean_squared_error",
                "bootstrap_estimate": bootstrap.estimate,
                "bootstrap_ci_low": bootstrap.ci_low,
                "bootstrap_ci_high": bootstrap.ci_high,
                "bootstrap_block_days_requested": int(settings["bootstrap_days"]),
                "bootstrap_block_days_effective": bootstrap.block_length,
                "bootstrap_replicates": bootstrap.n_bootstrap,
                "bootstrap_confidence": 0.95,
                "bootstrap_random_state": 20260814,
                "bootstrap_method": "cohort_moving_block_bootstrap",
                "bootstrap_population": "daylight_origin_mean_squared_error",
                "n_seeds": len(values),
                "note": DISCLOSURE,
            }
        )
    candidates = []
    for profile in all_profiles(include_sensitivities=False):
        values = [
            item for item in records if item["model"] == profile.name and item["split"] == "validation"
        ]
        candidates.append(
            {
                "model": profile.name,
                "mean_quantile_crps": float(
                    np.mean([item["metrics"]["quantile_crps"] for item in values])
                ),
                "seeds": sorted(item["seed"] for item in values),
            }
        )
    selected = min(candidates, key=lambda item: (item["mean_quantile_crps"], item["model"]))
    selected_model = selected["model"]
    selection = {
        **dict(EVIDENCE_METADATA),
        "disclosure": DISCLOSURE,
        "split": "validation",
        "metric": "quantile_crps",
        "direction": "minimize",
        "seeds": list(SEEDS),
        "candidates": candidates,
        "selected_model": selected_model,
        "selected_artifact_hashes": {
            split: {
                str(seed): sidecars[identity(split, selected_model, seed)]["content_sha256"]
                for seed in SEEDS
            }
            for split in ("validation", "test")
        },
        "selection_config_sha256": hashlib.sha256(
            json.dumps(candidates, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    selection_path = root / "selection.json"
    write_json(selection_path, selection)
    reference_loss = np.mean(
        [losses[identity("test", selected_model, seed)] for seed in SEEDS], axis=0
    )
    dm_rows = []
    for row in rows:
        if row["split"] != "test" or row["model"] == selected_model:
            continue
        comparison = np.mean(
            [losses[identity("test", row["model"], seed)] for seed in SEEDS], axis=0
        )
        result = diebold_mariano_test(comparison, reference_loss, horizon=24)
        dm_rows.append(
            {
                "model": row["model"],
                "reference_model": selected_model,
                "statistic": result.statistic,
                "pvalue": result.pvalue,
                "mean_loss_difference": result.mean_loss_difference,
                "note": DISCLOSURE,
            }
        )
    adjusted, rejected = benjamini_hochberg([item["pvalue"] for item in dm_rows])
    for item, adjusted_value, significant in zip(dm_rows, adjusted, rejected, strict=True):
        item["bh_adjusted_pvalue"] = float(adjusted_value)
        item["bh_rejected"] = bool(significant)
    skill_rows = []
    for row in rows:
        model_records = [
            item
            for item in records
            if item["model"] == row["model"] and item["split"] == row["split"]
        ]
        reference_records = [
            item
            for item in records
            if item["model"] == selected_model and item["split"] == row["split"]
        ]
        sites = sorted({item["site_id"] for record in model_records for item in record["site_lead"]})
        for site in sites:
            model_value = float(
                np.mean(
                    [
                        item["point"]["daylight"]["nrmse"]
                        for record in model_records
                        for item in record["site_lead"]
                        if item["site_id"] == site
                    ]
                )
            )
            reference_value = float(
                np.mean(
                    [
                        item["point"]["daylight"]["nrmse"]
                        for record in reference_records
                        for item in record["site_lead"]
                        if item["site_id"] == site
                    ]
                )
            )
            skill_rows.append(
                {
                    "model": row["model"],
                    "split": row["split"],
                    "site_id": site,
                    "reference_model": selected_model,
                    "forecast_skill_nrmse": forecast_skill(model_value, reference_value),
                    "model_nrmse": model_value,
                    "reference_nrmse": reference_value,
                    "note": DISCLOSURE,
                }
            )
    sensitivity_rows = [row for row in rows if row["model"] in SENSITIVITY_NAMES]
    if len(sensitivity_rows) != 9:
        raise ValueError("aggregate requires exactly nine sensitivity rows")
    aggregate_path = root / "aggregate.json"
    write_json(
        aggregate_path,
        {
            **dict(EVIDENCE_METADATA),
            "disclosure": DISCLOSURE,
            "reference_selection": selection,
            "rows": rows,
            "dm_bh": dm_rows,
            "forecast_skill": skill_rows,
            "sensitivities": sensitivity_rows,
        },
    )
    prediction_sources = sorted(
        {
            str(path.relative_to(root))
            for pattern in ("*.npz", "*.json")
            for path in (root / "predictions").rglob(pattern)
        }
    )
    register_stage_outputs(
        root,
        "aggregate",
        (selection_path, aggregate_path),
        source_paths=("evaluations.json", "manifest.json", *prediction_sources),
    )


__all__ = ["aggregate", "evaluate", "sensitivity_settings"]
