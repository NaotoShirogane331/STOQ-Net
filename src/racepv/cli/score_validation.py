"""Score one prediction archive on validation data without touching held-out test data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from racepv.audit import sha256_file
from racepv.config import QUANTILE_LEVELS, load_config
from racepv.evidence import require_registered_prediction_input
from racepv.metrics import (
    binary_ramp_metrics,
    point_forecast_metrics,
    probabilistic_forecast_metrics,
    ramp_classification_metrics,
)
from racepv.utils import atomic_json_dump

from .evaluate import (
    _json_ready,
    _load_metadata,
    _load_npz,
    _validate_prediction_archive,
)

REPORTED_LEAD_STEPS = (0, 1, 3, 7, 15, 23)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _score_slice(
    targets: np.ndarray,
    quantiles: np.ndarray,
    state_labels: np.ndarray,
    state_probs: np.ndarray,
    daylight: np.ndarray,
    levels: tuple[float, ...],
) -> dict[str, Any]:
    median_index = levels.index(0.5)
    valid_states = daylight & np.isin(state_labels, (0, 1, 2))
    result: dict[str, Any] = {
        "count": int(targets.size),
        "daylight_count": int(daylight.sum()),
        "point": point_forecast_metrics(
            targets,
            quantiles[..., median_index],
            daylight_mask=daylight,
        ),
        "probabilistic_daylight": probabilistic_forecast_metrics(
            targets,
            quantiles,
            levels,
            mask=daylight,
        ),
    }
    if np.any(valid_states):
        result["ramp_binary"] = binary_ramp_metrics(
            state_labels, state_probs, mask=valid_states
        )
        result["ramp_multiclass"] = ramp_classification_metrics(
            state_labels, state_probs, mask=valid_states
        )
    for state_name, state_mask in (
        ("stable", daylight & (state_labels == 0)),
        ("ramp", daylight & np.isin(state_labels, (1, 2))),
        ("ramp_up", daylight & (state_labels == 1)),
        ("ramp_down", daylight & (state_labels == 2)),
    ):
        result[f"probabilistic_{state_name}"] = probabilistic_forecast_metrics(
            targets,
            quantiles,
            levels,
            mask=state_mask,
        )
    return result


def main() -> None:
    args = build_parser().parse_args()
    require_registered_prediction_input(args.predictions)
    config = load_config(args.config)
    levels = tuple(float(value) for value in config.quantiles or QUANTILE_LEVELS)
    arrays = _load_npz(args.predictions)
    metadata = _load_metadata(args.predictions)
    _validate_prediction_archive(
        arrays,
        metadata,
        split="validation",
        levels=levels,
        horizon=config.data.horizon,
        power_upper_bound=config.data.power_upper_bound,
    )

    targets = arrays["targets"]
    quantiles = arrays["quantiles"]
    labels = arrays["state_labels"].astype(int)
    probabilities = arrays["state_probs"]
    daylight = arrays["daylight_mask"].astype(bool)
    by_lead: dict[str, Any] = {}
    for step in REPORTED_LEAD_STEPS:
        if step >= targets.shape[1]:
            raise ValueError(
                f"registered lead step {step} is outside horizon {targets.shape[1]}"
            )
        by_lead[str((step + 1) * config.data.frequency_minutes)] = _score_slice(
            targets[:, step],
            quantiles[:, step],
            labels[:, step],
            probabilities[:, step],
            daylight[:, step],
            levels,
        )

    identity = {
        key: metadata[key]
        for key in ("run_name", "baseline", "seed")
        if key in metadata
    }
    report = {
        "schema_version": 1,
        "split": "validation",
        "selection_boundary": (
            "Validation-only diagnostic. No calibration or test archive is read by this CLI."
        ),
        "prediction_path": str(args.predictions.resolve()),
        "prediction_sha256": sha256_file(args.predictions),
        "prediction_metadata_sha256": sha256_file(args.predictions.with_suffix(".json")),
        "identity": identity,
        "overall": _score_slice(
            targets, quantiles, labels, probabilities, daylight, levels
        ),
        "by_lead_minutes": by_lead,
    }
    atomic_json_dump(_json_ready(report), args.output)
    print(f"Wrote validation-only scores to {args.output}")


if __name__ == "__main__":
    main()
