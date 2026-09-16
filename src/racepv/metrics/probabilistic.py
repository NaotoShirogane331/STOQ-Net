"""Proper scores and interval diagnostics for quantile forecasts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _quantile_inputs(
    y_true: np.ndarray,
    quantile_predictions: np.ndarray,
    quantile_levels: Sequence[float],
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=float)
    predictions = np.asarray(quantile_predictions, dtype=float)
    levels = np.asarray(quantile_levels, dtype=float)
    if levels.ndim != 1 or np.any((levels <= 0) | (levels >= 1)):
        raise ValueError("quantile_levels must be a one-dimensional array inside (0, 1)")
    if np.any(np.diff(levels) <= 0):
        raise ValueError("quantile_levels must be strictly increasing")
    if predictions.shape != truth.shape + (levels.size,):
        raise ValueError("quantile_predictions must have shape y_true.shape + [Q]")
    valid = np.isfinite(truth) & np.all(np.isfinite(predictions), axis=-1)
    if mask is not None:
        try:
            valid &= np.broadcast_to(np.asarray(mask, dtype=bool), truth.shape)
        except ValueError as exc:
            raise ValueError("mask is not broadcastable to y_true") from exc
    return truth, predictions, levels, valid


def pinball_loss(
    y_true: np.ndarray,
    quantile_predictions: np.ndarray,
    quantile_levels: Sequence[float],
    *,
    mask: np.ndarray | None = None,
    reduction: str = "mean",
) -> float | np.ndarray:
    """Compute pinball loss, optionally retaining sample/quantile dimensions."""

    truth, predictions, levels, valid = _quantile_inputs(
        y_true, quantile_predictions, quantile_levels, mask
    )
    residual = truth[..., None] - predictions
    losses = np.maximum(levels * residual, (levels - 1.0) * residual)
    if reduction == "none":
        return np.where(valid[..., None], losses, np.nan)
    if reduction == "quantile":
        if not np.any(valid):
            return np.full(levels.shape, np.nan)
        return np.mean(losses[valid], axis=0)
    if reduction != "mean":
        raise ValueError("reduction must be mean, quantile, or none")
    return float(np.mean(losses[valid])) if np.any(valid) else float("nan")


def quantile_crps(
    y_true: np.ndarray,
    quantile_predictions: np.ndarray,
    quantile_levels: Sequence[float],
    *,
    mask: np.ndarray | None = None,
    integration: str = "trapezoid",
) -> float:
    """Approximate CRPS as twice the integrated pinball loss.

    ``integration="trapezoid"`` is the default because the RaCE-PV quantile
    grid is not equally spaced.  It integrates over quantile level and adds the
    zero-loss endpoints at 0 and 1, appropriate for bounded normalized PV
    power. ``integration="mean"`` retains the conventional equal-weight
    multi-quantile approximation for comparisons with software reporting it.
    """

    if not isinstance(integration, str) or integration not in ("mean", "trapezoid"):
        raise ValueError("integration must be mean or trapezoid")
    per_quantile = np.asarray(
        pinball_loss(
            y_true,
            quantile_predictions,
            quantile_levels,
            mask=mask,
            reduction="quantile",
        )
    )
    if not np.any(np.isfinite(per_quantile)):
        return float("nan")
    levels = np.asarray(quantile_levels, dtype=float)
    if integration == "mean":
        return float(2.0 * np.mean(per_quantile))
    integration_levels = np.r_[0.0, levels, 1.0]
    integration_losses = np.r_[0.0, per_quantile, 0.0]
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return float(2.0 * integrate(integration_losses, integration_levels))


def _interval_inputs(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    truth, low, high = np.broadcast_arrays(
        np.asarray(y_true, dtype=float),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )
    if np.any((low > high) & np.isfinite(low) & np.isfinite(high)):
        raise ValueError("lower interval bounds cannot exceed upper bounds")
    valid = np.isfinite(truth) & np.isfinite(low) & np.isfinite(high)
    if mask is not None:
        try:
            valid &= np.broadcast_to(np.asarray(mask, dtype=bool), truth.shape)
        except ValueError as exc:
            raise ValueError("mask is not broadcastable to interval shape") from exc
    return truth, low, high, valid


def prediction_interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> float:
    """Prediction interval coverage probability (PICP), inclusive at endpoints."""

    truth, low, high, valid = _interval_inputs(y_true, lower, upper, mask)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(((truth >= low) & (truth <= high))[valid]))


def prediction_interval_normalized_width(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    normalizer: float | np.ndarray = 1.0,
    mask: np.ndarray | None = None,
) -> float:
    """Mean interval width divided by rated capacity (PINAW)."""

    truth, low, high, valid = _interval_inputs(y_true, lower, upper, mask)
    try:
        scale = np.broadcast_to(np.asarray(normalizer, dtype=float), truth.shape)
    except ValueError as exc:
        raise ValueError("normalizer is not broadcastable to interval shape") from exc
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("normalizer must contain positive finite values")
    if not np.any(valid):
        return float("nan")
    return float(np.mean((high[valid] - low[valid]) / scale[valid]))


def interval_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    coverage: float,
    mask: np.ndarray | None = None,
    reduction: str = "mean",
) -> float | np.ndarray:
    """Gneiting--Raftery central interval score (smaller is better)."""

    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between 0 and 1")
    truth, low, high, valid = _interval_inputs(y_true, lower, upper, mask)
    alpha = 1.0 - coverage
    scores = high - low
    scores = scores + (2.0 / alpha) * (low - truth) * (truth < low)
    scores = scores + (2.0 / alpha) * (truth - high) * (truth > high)
    if reduction == "none":
        return np.where(valid, scores, np.nan)
    if reduction != "mean":
        raise ValueError("reduction must be mean or none")
    return float(np.mean(scores[valid])) if np.any(valid) else float("nan")


def conditional_coverage_error(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    groups: np.ndarray,
    *,
    coverage: float,
    mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Absolute coverage error for each supplied condition and its aggregates."""

    truth, low, high, valid = _interval_inputs(y_true, lower, upper, mask)
    condition = np.asarray(groups)
    try:
        condition = np.broadcast_to(condition, truth.shape)
    except ValueError as exc:
        raise ValueError("groups is not broadcastable to interval shape") from exc
    covered = (truth >= low) & (truth <= high)
    rows: dict[str, dict[str, float | int]] = {}
    errors = []
    for group in np.unique(condition[valid]):
        selection = valid & (condition == group)
        empirical = float(np.mean(covered[selection]))
        error = abs(empirical - coverage)
        errors.append(error)
        rows[str(group)] = {
            "count": int(np.sum(selection)),
            "coverage": empirical,
            "absolute_error": error,
        }
    return {
        "target": coverage,
        "groups": rows,
        "mean_absolute_error": float(np.mean(errors)) if errors else float("nan"),
        "maximum_absolute_error": float(np.max(errors)) if errors else float("nan"),
    }


def probabilistic_forecast_metrics(
    y_true: np.ndarray,
    quantile_predictions: np.ndarray,
    quantile_levels: Sequence[float],
    *,
    intervals: dict[float, np.ndarray] | None = None,
    groups: np.ndarray | None = None,
    normalizer: float | np.ndarray = 1.0,
    mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Return pinball/CRPS and optional interval diagnostics in one mapping."""

    result: dict[str, object] = {
        "mean_pinball_loss": pinball_loss(y_true, quantile_predictions, quantile_levels, mask=mask),
        "quantile_crps": quantile_crps(y_true, quantile_predictions, quantile_levels, mask=mask),
    }
    if intervals is None:
        return result
    interval_results = {}
    for coverage, values in intervals.items():
        array = np.asarray(values, dtype=float)
        if array.shape != np.asarray(y_true).shape + (2,):
            raise ValueError("interval arrays must have shape y_true.shape + [2]")
        low, high = array[..., 0], array[..., 1]
        metrics: dict[str, object] = {
            "picp": prediction_interval_coverage(y_true, low, high, mask=mask),
            "pinaw": prediction_interval_normalized_width(
                y_true, low, high, normalizer=normalizer, mask=mask
            ),
            "interval_score": interval_score(y_true, low, high, coverage=coverage, mask=mask),
        }
        if groups is not None:
            metrics["conditional_coverage"] = conditional_coverage_error(
                y_true, low, high, groups, coverage=coverage, mask=mask
            )
        interval_results[float(coverage)] = metrics
    result["intervals"] = interval_results
    return result


picp = prediction_interval_coverage
pinaw = prediction_interval_normalized_width
approximate_crps = quantile_crps
