"""Aggregate repeated RaCE-PV JSON evaluations without hiding seed variability."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from racepv.config import SEEDS
from racepv.evidence import require_registered_json_input
from racepv.utils import atomic_json_dump

AGGREGATABLE_LEAVES = {
    "mae",
    "rmse",
    "nmae",
    "nrmse",
    "forecast_skill_mae",
    "forecast_skill_rmse",
    "mean_pinball_loss",
    "quantile_crps",
    "picp",
    "pinaw",
    "interval_score",
    "conditional_coverage_error",
    "coverage",
    "precision",
    "recall",
    "f1",
    "csi",
    "auprc",
    "timing_mae_minutes",
    "timing_bias_minutes",
    "amplitude_mae",
    "amplitude_bias",
}


def _is_aggregatable_metric(path: str) -> bool:
    leaf = path.rsplit(".", 1)[-1]
    return leaf in AGGREGATABLE_LEAVES or leaf.startswith(
        (
            "mean_forecast_skill_",
            "equal_weight_mean_",
            "fraction_sites_better_",
            "positive_skill_fraction",
        )
    )


def _flatten(value: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            output.update(_flatten(item, child))
    # Decisions, significance flags and p-values must not be averaged as if they
    # were forecast metrics. Only preregistered metric leaves are admitted.
    elif (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and np.isfinite(float(value))
        and _is_aggregatable_metric(prefix)
    ):
        output[prefix] = float(value)
    return output


def _run_identity(report: Mapping[str, Any], path: Path) -> dict[str, Any]:
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{path} has no evaluation provenance")
    metadata = provenance.get("test_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{path} has no test prediction metadata")
    try:
        seed = int(metadata["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} has no integer seed") from error
    experiment = metadata.get("baseline", metadata.get("run_name"))
    if not isinstance(experiment, str) or not experiment:
        raise ValueError(f"{path} has no baseline/run_name identity")
    test_content = provenance.get(
        "test_window_content_sha256", metadata.get("source_window_content_sha256")
    )
    if not isinstance(test_content, str) or len(test_content) != 64:
        raise ValueError(f"{path} has no full test-window content identity")
    configuration = metadata.get(
        "experiment_config_sha256",
        metadata.get("config_sha256", metadata.get("resolved_run_config_sha256")),
    )
    if not isinstance(configuration, str) or not configuration:
        raise ValueError(f"{path} has no immutable configuration identity")
    protocol_fields = {
        key: metadata.get(key)
        for key in (
            "requested_backbone",
            "actual_backbone",
            "use_clear_sky_prior",
            "router_mode",
            "fusion_mode",
            "active_experts",
            "loss_weights",
            "ramp_method",
        )
        if key in metadata
    }
    return {
        "seed": seed,
        "experiment": experiment,
        "test_window_content_sha256": test_content,
        "configuration_identity": configuration,
        "protocol_fields": protocol_fields,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-runs", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.require_runs != len(SEEDS):
        raise ValueError(
            f"registered aggregation requires --require-runs={len(SEEDS)}"
        )
    if len(args.inputs) != args.require_runs:
        raise ValueError(
            f"received {len(args.inputs)} runs, expected exactly {args.require_runs}"
        )
    if len({path.resolve() for path in args.inputs}) != len(args.inputs):
        raise ValueError("aggregation inputs must be unique")
    flattened = []
    identities = []
    for path in args.inputs:
        value = require_registered_json_input(path)
        identities.append(_run_identity(value, path))
        flattened.append(_flatten(value))
    seeds = [int(identity["seed"]) for identity in identities]
    if sorted(seeds) != list(SEEDS) or len(set(seeds)) != len(SEEDS):
        raise ValueError(f"aggregation requires the exact unique seed set {list(SEEDS)}")
    for key in (
        "experiment",
        "test_window_content_sha256",
        "configuration_identity",
        "protocol_fields",
    ):
        if any(identity[key] != identities[0][key] for identity in identities[1:]):
            raise ValueError(f"aggregation inputs differ at registered identity {key}")
    all_keys = sorted(set().union(*(row.keys() for row in flattened)))
    rows = []
    for key in all_keys:
        values = np.asarray([row[key] for row in flattened if key in row], dtype=float)
        rows.append(
            {
                "metric": key,
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "median": float(np.median(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json_dump(
        {
            "run_count": len(args.inputs),
            "seed_set": sorted(seeds),
            "registered_identity": {
                key: identities[0][key]
                for key in (
                    "experiment",
                    "test_window_content_sha256",
                    "configuration_identity",
                    "protocol_fields",
                )
            },
            "aggregation_policy": (
                "metric whitelist only; booleans, p-values, significance decisions, "
                "hashes and counts are not averaged"
            ),
            "inputs": [str(path) for path in args.inputs],
            "metrics": rows,
        },
        args.output / "summary.json",
    )
    print(f"Aggregated {len(args.inputs)} runs into {args.output}")


if __name__ == "__main__":
    main()
