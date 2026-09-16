"""Deterministic and tree-based reference forecasters.

These classes deliberately receive already prepared features. They never inspect
future target columns and therefore can be used in the same leakage-controlled
protocol as RaCE-PV.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


def _last_valid(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must have shape [batch, history]")
    if mask is None:
        mask = np.isfinite(values)
    mask = np.asarray(mask, dtype=bool)
    reverse_index = np.argmax(mask[:, ::-1], axis=1)
    has_value = mask.any(axis=1)
    index = values.shape[1] - 1 - reverse_index
    result = values[np.arange(values.shape[0]), index]
    result[~has_value] = 0.0
    return result


def ramp_state_probabilities_from_median(
    median: np.ndarray,
    history_power: np.ndarray,
    thresholds: np.ndarray,
    daylight: np.ndarray,
    history_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Turn a baseline median trajectory into comparable continuous ramp scores.

    Stable/up/down logits are distances from the registered site threshold,
    scaled by that same threshold and passed through softmax.  Thus AUCPR uses a
    genuine ranking score rather than a one-hot label. Missing history is filled
    forward using only observations available before the issue time.
    """

    forecast = np.asarray(median, dtype=np.float32)
    history = np.asarray(history_power, dtype=np.float32)
    if forecast.ndim != 2 or history.ndim != 2 or history.shape[0] != forecast.shape[0]:
        raise ValueError("median=[N,H] and history_power=[N,L] are required")
    if history.shape[1] < 2:
        raise ValueError("at least two historical power steps are required")
    if history_mask is None:
        available = np.isfinite(history)
    else:
        available = np.asarray(history_mask, dtype=bool) & np.isfinite(history)
        if available.shape != history.shape:
            raise ValueError("history_mask must match history_power")
    causal = np.where(available, history, np.nan).astype(np.float32)
    # Forward filling is issue-time causal. Leading missing values use zero,
    # which is explicit and cannot import a later interpolation endpoint.
    for step in range(1, causal.shape[1]):
        missing = ~np.isfinite(causal[:, step])
        causal[missing, step] = causal[missing, step - 1]
    causal = np.nan_to_num(causal, nan=0.0, posinf=0.0, neginf=0.0)

    boundary = np.asarray(thresholds, dtype=np.float32).reshape(-1, 1)
    if boundary.shape[0] != forecast.shape[0] or np.any(~np.isfinite(boundary)):
        raise ValueError("one finite ramp threshold is required per forecast origin")
    if np.any(boundary <= 0):
        raise ValueError("ramp thresholds must be positive")
    day = np.asarray(daylight, dtype=bool)
    if day.shape != forecast.shape:
        raise ValueError("daylight must match median")

    extended = np.concatenate((causal[:, -2:], forecast), axis=1)
    changes = extended[:, 2:] - extended[:, :-2]
    scale = np.maximum(boundary, np.float32(1e-4))
    tie = np.float32(1e-6)
    logits = np.stack(
        (
            np.zeros_like(changes),
            (changes - boundary) / scale + tie,
            (-changes - boundary) / scale + tie,
        ),
        axis=-1,
    )
    logits -= np.max(logits, axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    probabilities[~day] = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
    return probabilities.astype(np.float32, copy=False)


@dataclass
class SmartPersistence:
    """Persist the latest clear-sky index over the future clear-sky curve."""

    epsilon: float = 1e-4
    upper_bound: float = 1.05

    def predict(
        self,
        history_power: np.ndarray,
        history_clear_sky: np.ndarray,
        future_clear_sky: np.ndarray,
        history_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        power_last = _last_valid(history_power, history_mask)
        clear_last = _last_valid(history_clear_sky, history_mask)
        clear_index = power_last / np.maximum(clear_last, self.epsilon)
        clear_index = np.clip(clear_index, 0.0, 1.5)
        forecast = clear_index[:, None] * np.asarray(future_clear_sky, dtype=np.float64)
        return np.clip(forecast, 0.0, self.upper_bound)


@dataclass
class ClearSkyPersistence:
    """Use the known future clear-sky power as a deterministic baseline."""

    upper_bound: float = 1.05

    def predict(self, future_clear_sky: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(future_clear_sky, dtype=np.float64), 0.0, self.upper_bound)


class QuantileLightGBM:
    """One LightGBM quantile regressor for every lead/quantile pair.

    The estimator is intentionally explicit rather than using a multi-output
    wrapper, making each horizon's training data and quantile objective auditable.
    """

    def __init__(
        self,
        quantiles: Iterable[float] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
        **model_kwargs: object,
    ) -> None:
        self.quantiles = tuple(float(q) for q in quantiles)
        self.model_kwargs = {
            "n_estimators": 300,
            "learning_rate": 0.04,
            "num_leaves": 31,
            "verbosity": -1,
            "random_state": 13,
            **model_kwargs,
        }
        self.models: list[list[object]] = []

    @staticmethod
    def _model_class():
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install race-pv[trees] to use QuantileLightGBM") from exc
        return LGBMRegressor

    def fit(self, features: np.ndarray, targets: np.ndarray) -> QuantileLightGBM:
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("features=[N,F] and targets=[N,H] are required")
        model_class = self._model_class()
        self.models = []
        for horizon in range(y.shape[1]):
            horizon_models = []
            for quantile in self.quantiles:
                model = model_class(
                    objective="quantile",
                    alpha=quantile,
                    **self.model_kwargs,
                )
                model.fit(x, y[:, horizon])
                horizon_models.append(model)
            self.models.append(horizon_models)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("fit must be called before predict")
        x = np.asarray(features, dtype=np.float32)
        raw = np.stack(
            [
                np.stack([model.predict(x) for model in models], axis=-1)
                for models in self.models
            ],
            axis=1,
        )
        # Finite tree ensembles can cross; sorting is the standard transparent
        # rearrangement baseline and is reported as such in experiment metadata.
        return np.sort(raw, axis=-1)


class RandomForestPointForecaster:
    """Multi-output random forest intended for external Mondrian calibration."""

    def __init__(self, **kwargs: object) -> None:
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:  # pragma: no cover
            raise ImportError("scikit-learn is required") from exc
        defaults = {
            "n_estimators": 500,
            "min_samples_leaf": 2,
            "n_jobs": -1,
            "random_state": 13,
        }
        defaults.update(kwargs)
        self.model = RandomForestRegressor(**defaults)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> RandomForestPointForecaster:
        self.model.fit(np.asarray(features), np.asarray(targets))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(features)))
