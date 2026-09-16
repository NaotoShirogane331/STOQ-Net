"""Select one role-specific baseline class across seeds and lock its test artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from racepv.audit import sha256_file
from racepv.config import QUANTILE_LEVELS, SEEDS, load_config
from racepv.evidence import require_registered_prediction_input
from racepv.metrics import (
    binary_ramp_metrics,
    point_forecast_metrics,
    probabilistic_forecast_metrics,
    ramp_classification_metrics,
)
from racepv.utils import atomic_json_dump

from .evaluate import (
    DEFAULT_REFERENCE_SELECTION_METRICS,
    REFERENCE_SELECTION_METRICS,
    _load_metadata,
    _load_npz,
    _validate_prediction_archive,
    _validate_prediction_pair,
)


def _score_candidate(
    role: str,
    metric: str,
    arrays: Mapping[str, np.ndarray],
    levels: Sequence[float],
) -> tuple[float, int]:
    targets = arrays["targets"]
    daylight = arrays["daylight_mask"].astype(bool)
    true_ramp = daylight & np.isin(arrays["state_labels"], (1, 2))
    valid_states = daylight & np.isin(arrays["state_labels"], (0, 1, 2))
    if role in {"point", "ramp_point"}:
        median_matches = np.flatnonzero(np.isclose(np.asarray(levels), 0.5))
        if median_matches.size != 1:
            raise ValueError("point reference selection requires exactly one median quantile")
        median = arrays["quantiles"][..., int(median_matches[0])]
        selection = daylight if role == "point" else true_ramp
        values = point_forecast_metrics(
            targets,
            median,
            daylight_mask=selection,
        )["daylight"]
        key = {
            "daylight_nrmse": "nrmse",
            "daylight_nmae": "nmae",
            "ramp_nrmse": "nrmse",
            "ramp_nmae": "nmae",
        }[metric]
        return float(values[key]), int(np.sum(selection))
    if role in {"probabilistic", "ramp_probabilistic"}:
        selection = daylight if role == "probabilistic" else true_ramp
        values = probabilistic_forecast_metrics(
            targets,
            arrays["quantiles"],
            levels,
            mask=selection,
        )
        key = metric.removeprefix("ramp_")
        return float(values[key]), int(np.sum(selection))
    if role == "ramp":
        binary = binary_ramp_metrics(
            arrays["state_labels"], arrays["state_probs"], mask=valid_states
        )
        multiclass = ramp_classification_metrics(
            arrays["state_labels"], arrays["state_probs"], mask=valid_states
        )
        if metric.startswith("binary_"):
            value = binary[metric.removeprefix("binary_")]
        else:
            value = multiclass["macro"][metric.removeprefix("macro_")]  # type: ignore[index]
        return float(value), int(np.sum(valid_states))
    raise ValueError(f"unsupported reference role: {role}")


def _population_signature(candidate: Mapping[str, np.ndarray]) -> str:
    """Hash the common evaluation population without retaining another archive."""

    digest = hashlib.sha256()
    for name in (
        "targets",
        "state_labels",
        "daylight_mask",
        "timestamps",
        "day_ids",
        "site_ids",
    ):
        values = np.asarray(candidate[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        contiguous = np.ascontiguousarray(values)
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def select_reference(
    validation_paths: Sequence[Path],
    test_paths: Sequence[Path],
    *,
    role: str,
    metric: str | None,
    config_path: Path,
    output_path: Path,
    evidence_path: Path | None = None,
) -> dict[str, Any]:
    """Select the equal-seed-mean validation optimum without a supplied winner."""

    if role not in REFERENCE_SELECTION_METRICS:
        raise ValueError(f"role must be one of {tuple(REFERENCE_SELECTION_METRICS)}")
    selected_metric = metric or DEFAULT_REFERENCE_SELECTION_METRICS[role]
    direction = REFERENCE_SELECTION_METRICS[role].get(selected_metric)
    if direction is None:
        raise ValueError(f"metric {selected_metric!r} is not registered for role {role!r}")
    if len(validation_paths) != len(test_paths) or len(validation_paths) < 2:
        raise ValueError("at least two paired validation/test prediction archives are required")

    for path in (*validation_paths, *test_paths):
        require_registered_prediction_input(path)

    config = load_config(config_path)
    levels = tuple(float(value) for value in config.quantiles or QUANTILE_LEVELS)
    evidence_destination = evidence_path or output_path.with_name(
        f"{output_path.stem}.validation_results.json"
    )
    evidence_parent = evidence_destination.parent.resolve()
    rows: list[dict[str, Any]] = []
    validation_population_signature: str | None = None
    test_population_signature: str | None = None
    seen_instances: set[tuple[str, int]] = set()
    for validation_path, test_path in zip(validation_paths, test_paths, strict=True):
        validation_path = Path(validation_path)
        test_path = Path(test_path)
        validation = _load_npz(validation_path)
        test = _load_npz(test_path)
        validation_metadata = _load_metadata(validation_path)
        test_metadata = _load_metadata(test_path)
        _validate_prediction_archive(
            validation,
            validation_metadata,
            split="validation",
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
        _validate_prediction_pair(
            validation,
            validation_metadata,
            test,
            test_metadata,
        )
        name = validation_metadata.get("baseline")
        if not isinstance(name, str) or not name or test_metadata.get("baseline") != name:
            raise ValueError("reference candidates must be named baselines with matching identities")
        try:
            candidate_seed = int(validation_metadata["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"baseline {name} has no integer seed identity") from exc
        if test_metadata.get("seed") != validation_metadata["seed"]:
            raise ValueError(f"baseline {name} validation/test seed identities differ")
        configuration_identity = validation_metadata.get(
            "resolved_run_config_sha256", validation_metadata.get("config_sha256")
        )
        if not isinstance(configuration_identity, str) or not configuration_identity:
            raise ValueError(f"baseline {name} has no immutable configuration identity")
        instance = (name, candidate_seed)
        if instance in seen_instances:
            raise ValueError(f"duplicate baseline/seed candidate: {name}/{candidate_seed}")
        seen_instances.add(instance)
        candidate_validation_signature = _population_signature(validation)
        candidate_test_signature = _population_signature(test)
        if validation_population_signature is None:
            validation_population_signature = candidate_validation_signature
            test_population_signature = candidate_test_signature
        elif (
            candidate_validation_signature != validation_population_signature
            or candidate_test_signature != test_population_signature
        ):
            raise ValueError("candidate baselines use different validation or test populations")
        score, eligible_count = _score_candidate(role, selected_metric, validation, levels)
        if not np.isfinite(score):
            raise ValueError(f"baseline {name} produced a non-finite validation score")
        rows.append(
            {
                "baseline": name,
                "seed": candidate_seed,
                "configuration_sha256": configuration_identity,
                "score": score,
                "eligible_count": eligible_count,
                "validation_prediction_path": os.path.relpath(
                    validation_path.resolve(), evidence_parent
                ),
                "validation_prediction_sha256": sha256_file(validation_path),
                "validation_metadata_sha256": sha256_file(
                    validation_path.with_suffix(".json")
                ),
                "test_prediction_path": os.path.relpath(test_path.resolve(), evidence_parent),
                "test_prediction_sha256": sha256_file(test_path),
                "test_prediction_metadata_sha256": sha256_file(
                    test_path.with_suffix(".json")
                ),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["baseline"]), []).append(row)
    if len(grouped) < 2:
        raise ValueError("reference selection requires at least two baseline model classes")
    seed_sets = {
        name: {int(row["seed"]) for row in candidates}
        for name, candidates in grouped.items()
    }
    first_seed_set = next(iter(seed_sets.values()))
    if not first_seed_set or any(seeds != first_seed_set for seeds in seed_sets.values()):
        raise ValueError("every baseline model class must use the identical validation seed set")
    if first_seed_set != set(SEEDS):
        raise ValueError(
            f"registered reference selection requires the exact seed set {list(SEEDS)}"
        )
    for name, candidates in grouped.items():
        configurations = {str(row["configuration_sha256"]) for row in candidates}
        if len(configurations) != 1:
            raise ValueError(f"baseline {name} mixes configurations across selection seeds")
    selection_seeds = sorted(first_seed_set)
    aggregate_rows = [
        {
            "baseline": name,
            "score": float(
                np.mean(
                    [
                        float(row["score"])
                        for row in sorted(candidates, key=lambda value: int(value["seed"]))
                    ]
                )
            ),
        }
        for name, candidates in grouped.items()
    ]
    aggregate_rows.sort(
        key=(
            (lambda row: (float(row["score"]), str(row["baseline"])))
            if direction == "minimize"
            else (lambda row: (-float(row["score"]), str(row["baseline"])))
        )
    )
    winner_name = str(aggregate_rows[0]["baseline"])
    rows.sort(key=lambda row: (str(row["baseline"]), int(row["seed"])))
    evidence = {
        "schema_version": 2,
        "split": "validation",
        "role": role,
        "metric": selected_metric,
        "direction": direction,
        "aggregation": "equal_seed_mean",
        "seed_set": selection_seeds,
        "scores": {
            str(row["baseline"]): float(row["score"]) for row in aggregate_rows
        },
        "seed_scores": {
            name: {
                str(int(row["seed"])): float(row["score"])
                for row in sorted(candidates, key=lambda value: int(value["seed"]))
            }
            for name, candidates in sorted(grouped.items())
        },
        "candidates": rows,
        "selected_baseline": winner_name,
        "validation_population_sha256": validation_population_signature,
        "test_population_sha256": test_population_signature,
        "tie_break": "lexicographically smallest baseline name after exact score tie",
    }
    atomic_json_dump(evidence, evidence_destination)

    output_parent = output_path.parent.resolve()
    selected_rows = {
        int(row["seed"]): row for row in grouped[winner_name]
    }

    def selected_test_path(seed: int) -> Path:
        path = Path(str(selected_rows[seed]["test_prediction_path"]))
        return path if path.is_absolute() else evidence_parent / path

    manifest = {
        "schema_version": 3,
        "role": role,
        "selection_split": "validation",
        "selection_metric": selected_metric,
        "selection_direction": direction,
        "selection_aggregation": "equal_seed_mean",
        "selection_seeds": selection_seeds,
        "selected_baseline": winner_name,
        "validation_results": os.path.relpath(evidence_destination.resolve(), output_parent),
        "validation_results_sha256": sha256_file(evidence_destination),
        "test_predictions": {
            str(seed): {
                "path": os.path.relpath(
                    selected_test_path(seed).resolve(),
                    output_parent,
                ),
                "sha256": selected_rows[seed]["test_prediction_sha256"],
                "metadata_sha256": selected_rows[seed][
                    "test_prediction_metadata_sha256"
                ],
            }
            for seed in selection_seeds
        },
    }
    atomic_json_dump(manifest, output_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-predictions",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "candidate validation NPZ files; repeat each baseline for the identical seed set "
            "and order them like --test-predictions"
        ),
    )
    parser.add_argument(
        "--test-predictions",
        type=Path,
        nargs="+",
        required=True,
        help="candidate test prediction NPZ files paired by position",
    )
    parser.add_argument("--role", choices=tuple(REFERENCE_SELECTION_METRICS), required=True)
    parser.add_argument("--metric", help="registered role-specific metric; role default if omitted")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="schema-v3 group manifest locking the selected class test artifact for every seed",
    )
    parser.add_argument("--evidence-output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = select_reference(
        args.validation_predictions,
        args.test_predictions,
        role=args.role,
        metric=args.metric,
        config_path=args.config,
        output_path=args.output,
        evidence_path=args.evidence_output,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
