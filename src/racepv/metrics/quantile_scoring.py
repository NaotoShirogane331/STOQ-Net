"""Pure, versioned per-cell quantile scoring and aggregation."""

import numpy as np

_SCORE_IDS = (
    "qcrps_trapezoid_zero_endpoints_v1",
    "pinball_twice_equal_weight_v1",
)


def _real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real numeric data")
    return array


def _mask(value, shape):
    array = np.asarray(value)
    if array.dtype.kind != "b" or array.shape != shape:
        raise ValueError("eligible_mask must be boolean with exact cell shape")
    return array


def quantile_score_cells(targets, predictions, levels, *, eligible_mask, score_id):
    """Return one float64 score per target cell under an explicit score version."""

    if not isinstance(score_id, str) or score_id not in _SCORE_IDS:
        raise ValueError("unknown score_id")
    truth = _real_array(targets, "targets")
    forecasts = _real_array(predictions, "predictions")
    quantiles = _real_array(levels, "levels")
    selection = _mask(eligible_mask, truth.shape)
    if quantiles.ndim != 1 or quantiles.size == 0:
        raise ValueError("levels must be a nonempty vector")
    if not np.all(np.isfinite(quantiles)) or np.any((quantiles <= 0) | (quantiles >= 1)):
        raise ValueError("levels must be finite and inside (0,1)")
    quantiles = quantiles.astype(np.float64)
    if np.any((quantiles <= 0) | (quantiles >= 1)) or np.any(quantiles[1:] <= quantiles[:-1]):
        raise ValueError("levels must remain inside (0,1) and strictly increasing in float64")
    if forecasts.shape != truth.shape + (quantiles.size,):
        raise ValueError("prediction shape must equal target shape plus Q")
    output = np.full(truth.shape, np.nan, dtype=np.float64)
    selected_truth = truth[selection]
    selected_forecasts = forecasts[selection]
    if not np.all(np.isfinite(selected_truth)) or not np.all(np.isfinite(selected_forecasts)):
        raise ValueError("eligible observations must be finite")
    if np.any(selected_forecasts[:, 1:] < selected_forecasts[:, :-1]):
        raise ValueError("eligible quantiles cross")
    try:
        with np.errstate(over="raise", invalid="raise"):
            y = selected_truth.astype(np.float64)
            q = selected_forecasts.astype(np.float64)
            if not np.all(np.isfinite(y)) or not np.all(np.isfinite(q)):
                raise ValueError("eligible observations must be finite")
            if np.any(q[:, 1:] < q[:, :-1]):
                raise ValueError("eligible quantiles cross")
            residual = y[:, None] - q
            losses = np.maximum(quantiles * residual, (quantiles - 1) * residual)
            if score_id == _SCORE_IDS[0]:
                extended = np.r_[0.0, quantiles, 1.0]
                weights = extended[2:] - extended[:-2]
            else:
                weights = np.full(quantiles.shape, 2.0 / quantiles.size)
            values = np.sum(losses * weights, axis=-1, dtype=np.float64)
    except FloatingPointError as exc:
        raise ValueError("eligible score overflow or invalid arithmetic") from exc
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("eligible scores must be finite and nonnegative")
    output[selection] = values
    return output


def quantile_score_summary(cells, *, eligible_mask):
    """Summarize selected cell losses without treating unavailable cells as zero."""

    array = _real_array(cells, "cells")
    selection = _mask(eligible_mask, array.shape)
    selected_input = array[selection]
    if not np.all(np.isfinite(selected_input)) or np.any(selected_input < 0):
        raise ValueError("eligible losses must be finite and nonnegative")
    try:
        with np.errstate(over="raise", invalid="raise"):
            selected = selected_input.astype(np.float64)
            if not np.all(np.isfinite(selected)) or np.any(selected < 0):
                raise ValueError("eligible losses must be finite and nonnegative")
            total = float(np.sum(selected, dtype=np.float64))
    except FloatingPointError as exc:
        raise ValueError("loss_sum overflow or invalid arithmetic") from exc
    if not np.isfinite(total):
        raise ValueError("loss_sum must be finite")
    count = int(selected.size)
    return {
        "loss_sum": total,
        "count": count,
        "mean": total / count if count else None,
        "unavailable_reason": None if count else "no_eligible_cells",
    }
