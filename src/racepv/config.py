"""Experiment configuration with strict validation of information boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

QUANTILE_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
SEEDS = (13, 29, 47, 71, 101)


@dataclass(frozen=True)
class DataConfig:
    frequency_minutes: int = 15
    history_length: int = 192
    horizon: int = 24
    stride: int = 1
    max_interpolation_minutes: int = 30
    daylight_threshold: float = 0.01
    power_upper_bound: float = 1.05
    capacity_fallback_quantile: float = 0.995
    ramp_window_steps: int = 2
    ramp_quantile: float = 0.90
    split_ratios: tuple[float, float, float, float] = (0.60, 0.15, 0.10, 0.15)
    allowed_future_features: tuple[str, ...] = (
        "clear_sky_power_norm",
        "solar_elevation",
        "solar_azimuth_sin",
        "solar_azimuth_cos",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "lead_time_hours",
    )

    def validate(self) -> None:
        if self.frequency_minutes != 15:
            raise ValueError("The registered RaCE-PV protocol uses a 15-minute frequency.")
        if self.history_length != 192 or self.horizon != 24:
            raise ValueError("The registered protocol requires L=192 and H=24.")
        if abs(sum(self.split_ratios) - 1.0) > 1e-9:
            raise ValueError("split_ratios must sum to one")
        if not 0 < self.ramp_quantile < 1:
            raise ValueError("ramp_quantile must be in (0, 1)")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 80
    patience: int = 10
    gradient_clip_norm: float = 1.0
    # V2 fixes the unweighted state cross-entropy coefficient at one.  The
    # field remains parseable so legacy experiment YAMLs fail explicitly in
    # the train entry point instead of being silently reinterpreted.
    state_loss_weight: float = 1.0
    specialist_loss_weight: float = 0.15
    num_workers: int = 0
    seed: int = 13
    mixed_precision: bool = True


@dataclass(frozen=True)
class CalibrationConfig:
    coverages: tuple[float, ...] = (0.80, 0.90)
    min_group_scores: int = 200
    steps_per_day: int = 96
    horizon_bands: tuple[tuple[int, int], ...] = ((0, 4), (4, 12), (12, 24))


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    quantiles: tuple[float, ...] = QUANTILE_LEVELS
    output_dir: str = "runs/default"
    run_name: str = "racepv"

    def validate(self) -> None:
        self.data.validate()
        if tuple(sorted(self.quantiles)) != self.quantiles:
            raise ValueError("quantiles must be strictly ordered")
        if 0.5 not in self.quantiles:
            raise ValueError("quantiles must include the median (0.5)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct_config(raw: dict[str, Any]) -> ExperimentConfig:
    data_raw = raw.get("data", {})
    train_raw = raw.get("training", {})
    cal_raw = raw.get("calibration", {})
    for key in ("allowed_future_features", "split_ratios"):
        if key in data_raw:
            data_raw[key] = tuple(data_raw[key])
    if "coverages" in cal_raw:
        cal_raw["coverages"] = tuple(cal_raw["coverages"])
    if "horizon_bands" in cal_raw:
        cal_raw["horizon_bands"] = tuple(tuple(v) for v in cal_raw["horizon_bands"])
    quantiles = tuple(raw.get("quantiles", QUANTILE_LEVELS))
    cfg = ExperimentConfig(
        data=DataConfig(**data_raw),
        training=TrainingConfig(**train_raw),
        calibration=CalibrationConfig(**cal_raw),
        quantiles=quantiles,
        output_dir=raw.get("output_dir", "runs/default"),
        run_name=raw.get("run_name", "racepv"),
    )
    cfg.validate()
    return cfg


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError("configuration root must be a mapping")
    return _construct_config(raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
