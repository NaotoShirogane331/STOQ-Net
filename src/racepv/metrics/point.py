"""Deterministic forecasting metrics for normalized photovoltaic power."""

from __future__ import annotations

import numpy as np


def _paired(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth, prediction = np.broadcast_arrays(
        np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    )
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if mask is not None:
        try:
            valid &= np.broadcast_to(np.asarray(mask, dtype=bool), truth.shape)
        except ValueError as exc:
            raise ValueError("mask is not broadcastable to the target shape") from exc
    return truth, prediction, valid


def _normalizer_array(normalizer: float | np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    try:
        values = np.broadcast_to(np.asarray(normalizer, dtype=float), shape)
    except ValueError as exc:
        raise ValueError("normalizer is not broadcastable to the target shape") from exc
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("normalizer must contain positive finite values")
    return values


def normalized_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalizer: float | np.ndarray = 1.0,
    mask: np.ndarray | None = None,
) -> float:
    """Mean absolute error divided elementwise by capacity/normalizer."""

    truth, prediction, valid = _paired(y_true, y_pred, mask)
    if not np.any(valid):
        return float("nan")
    scale = _normalizer_array(normalizer, truth.shape)
    return float(np.mean(np.abs(prediction[valid] - truth[valid]) / scale[valid]))


def normalized_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalizer: float | np.ndarray = 1.0,
    mask: np.ndarray | None = None,
) -> float:
    """Root mean squared normalized error."""

    truth, prediction, valid = _paired(y_true, y_pred, mask)
    if not np.any(valid):
        return float("nan")
    scale = _normalizer_array(normalizer, truth.shape)
    errors = (prediction[valid] - truth[valid]) / scale[valid]
    return float(np.sqrt(np.mean(np.square(errors))))


def forecast_skill(
    y_true: np.ndarray | float,
    y_pred: np.ndarray | float,
    reference_pred: np.ndarray | float | None = None,
    *,
    metric: str = "rmse",
    normalizer: float | np.ndarray = 1.0,
    mask: np.ndarray | None = None,
) -> float:
    """Return ``1 - model_error / reference_error``.

    If ``reference_pred`` is omitted, the first two arguments are interpreted as
    already-computed model and reference errors.  This convenience keeps result
    aggregation independent of the raw prediction arrays.
    """

    if reference_pred is None:
        model_error = float(np.asarray(y_true))
        reference_error = float(np.asarray(y_pred))
    else:
        if metric.lower() in {"mae", "nmae"}:
            function = normalized_mae
        elif metric.lower() in {"rmse", "nrmse"}:
            function = normalized_rmse
        else:
            raise ValueError("metric must be mae/nmae or rmse/nrmse")
        truth, model_values, reference_values = np.broadcast_arrays(
            np.asarray(y_true, dtype=float),
            np.asarray(y_pred, dtype=float),
            np.asarray(reference_pred, dtype=float),
        )
        common = (
            np.isfinite(truth)
            & np.isfinite(model_values)
            & np.isfinite(reference_values)
        )
        if mask is not None:
            try:
                common &= np.broadcast_to(np.asarray(mask, dtype=bool), truth.shape)
            except ValueError as exc:
                raise ValueError("mask is not broadcastable to the target shape") from exc
        model_error = function(
            truth, model_values, normalizer=normalizer, mask=common
        )
        reference_error = function(
            truth, reference_values, normalizer=normalizer, mask=common
        )
    if not np.isfinite(reference_error) or reference_error <= 0:
        return float("nan")
    return float(1.0 - model_error / reference_error)


def point_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    daylight_mask: np.ndarray | None = None,
    normalizer: float | np.ndarray = 1.0,
    reference_pred: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Compute all-day and daylight-only nMAE/nRMSE/forecast skill."""

    truth, _, valid = _paired(y_true, y_pred)
    if daylight_mask is None:
        daylight = np.ones(truth.shape, dtype=bool)
    else:
        try:
            daylight = np.broadcast_to(np.asarray(daylight_mask, dtype=bool), truth.shape)
        except ValueError as exc:
            raise ValueError("daylight_mask is not broadcastable to the target shape") from exc

    output: dict[str, dict[str, float]] = {}
    for name, selection in (("all_day", valid), ("daylight", valid & daylight)):
        values = {
            "nmae": normalized_mae(
                y_true, y_pred, normalizer=normalizer, mask=selection
            ),
            "nrmse": normalized_rmse(
                y_true, y_pred, normalizer=normalizer, mask=selection
            ),
        }
        if reference_pred is not None:
            values["forecast_skill_mae"] = forecast_skill(
                y_true,
                y_pred,
                reference_pred,
                metric="mae",
                normalizer=normalizer,
                mask=selection,
            )
            values["forecast_skill_rmse"] = forecast_skill(
                y_true,
                y_pred,
                reference_pred,
                metric="rmse",
                normalizer=normalizer,
                mask=selection,
            )
        output[name] = values
    return output


nmae = normalized_mae
nrmse = normalized_rmse
