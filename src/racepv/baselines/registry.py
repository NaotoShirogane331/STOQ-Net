"""Baseline registry used by experiment configuration files."""

from __future__ import annotations

from typing import Any

from .classical import (
    ClearSkyPersistence,
    QuantileLightGBM,
    RandomForestPointForecaster,
    SmartPersistence,
)
from .neural import LSTMQuantile, PatchTSTQuantile, TCNQuantile
from .s_mamba import SMambaQuantile

_BASELINES = {
    "smart_persistence": SmartPersistence,
    "clear_sky_persistence": ClearSkyPersistence,
    "quantile_lightgbm": QuantileLightGBM,
    "random_forest": RandomForestPointForecaster,
    "lstm_qr": LSTMQuantile,
    "tcn_qr": TCNQuantile,
    "patchtst_qr": PatchTSTQuantile,
    "s_mamba_qr": SMambaQuantile,
}


def list_baselines() -> tuple[str, ...]:
    return tuple(sorted(_BASELINES))


def build_baseline(name: str, **kwargs: Any):
    try:
        baseline = _BASELINES[name]
    except KeyError as exc:
        raise KeyError(f"unknown baseline {name!r}; choose from {list_baselines()}") from exc
    return baseline(**kwargs)
