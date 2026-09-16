"""Statistical comparison utilities used by the RaCE-PV experiments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import erfc, sqrt
from typing import NamedTuple

import numpy as np


class DMResult(NamedTuple):
    """Diebold--Mariano result; negative mean favors the first/model loss."""

    statistic: float
    pvalue: float
    mean_loss_difference: float
    lag: int
    n_observations: int


def _normal_cdf(value: float) -> float:
    return 0.5 * erfc(-value / sqrt(2.0))


def diebold_mariano_test(
    model_loss: np.ndarray,
    baseline_loss: np.ndarray,
    *,
    horizon: int = 24,
    lag: int | None = None,
    alternative: str = "two-sided",
    harvey_correction: bool = False,
) -> DMResult:
    """Compare paired forecast losses using a Newey--West HAC variance.

    Arrays may be ``[time]`` or ``[time, ...]``.  In the latter case losses are
    averaged over non-time dimensions before testing, so the temporal sample
    size is the number of forecast origins rather than ``N * H``.  The default
    HAC lag is ``horizon - 1`` as specified in the RaCE-PV protocol.
    """

    first, second = np.broadcast_arrays(
        np.asarray(model_loss, dtype=float), np.asarray(baseline_loss, dtype=float)
    )
    if first.ndim == 0:
        raise ValueError("loss arrays must contain a temporal dimension")
    differences = first - second
    if differences.ndim > 1:
        flattened = differences.reshape(differences.shape[0], -1)
        finite = np.isfinite(flattened)
        counts = np.sum(finite, axis=1)
        sums = np.sum(np.where(finite, flattened, 0.0), axis=1)
        differences = np.full(flattened.shape[0], np.nan, dtype=float)
        np.divide(sums, counts, out=differences, where=counts > 0)
    differences = differences.reshape(-1)
    finite = np.isfinite(differences)
    n = int(np.sum(finite))
    if n < 2:
        raise ValueError("at least two finite paired loss observations are required")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    selected_lag = horizon - 1 if lag is None else int(lag)
    if selected_lag < 0:
        raise ValueError("lag cannot be negative")
    # Keep missing/night positions in the time axis. Compressing them would make
    # sunset and the following sunrise artificial neighbors in the HAC series.
    selected_lag = min(selected_lag, differences.size - 1)

    mean_difference = float(np.mean(differences[finite]))
    centered = np.zeros(differences.shape, dtype=float)
    centered[finite] = differences[finite] - mean_difference
    gamma_zero = float(np.dot(centered, centered) / n)
    long_run_variance = gamma_zero
    for order in range(1, selected_lag + 1):
        valid_pairs = finite[order:] & finite[:-order]
        autocovariance = float(
            np.dot(centered[order:][valid_pairs], centered[:-order][valid_pairs]) / n
        )
        bartlett_weight = 1.0 - order / (selected_lag + 1.0)
        long_run_variance += 2.0 * bartlett_weight * autocovariance
    # Numerical cancellation can make a theoretically non-negative HAC estimate
    # very slightly negative.
    long_run_variance = max(0.0, long_run_variance)
    if long_run_variance <= np.finfo(float).eps:
        statistic = 0.0 if abs(mean_difference) <= np.finfo(float).eps else float(
            np.sign(mean_difference) * np.inf
        )
    else:
        statistic = mean_difference / np.sqrt(long_run_variance / n)

    if harvey_correction:
        factor_squared = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
        if factor_squared > 0 and np.isfinite(statistic):
            statistic *= np.sqrt(factor_squared)

    if alternative == "two-sided":
        pvalue = min(1.0, 2.0 * (1.0 - _normal_cdf(abs(statistic))))
    elif alternative == "less":
        pvalue = _normal_cdf(statistic)
    elif alternative == "greater":
        pvalue = 1.0 - _normal_cdf(statistic)
    else:
        raise ValueError("alternative must be two-sided, less, or greater")
    return DMResult(
        float(statistic),
        float(pvalue),
        mean_difference,
        selected_lag,
        n,
    )


def benjamini_hochberg(
    pvalues: Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Return BH-adjusted p-values and rejection decisions.

    Non-finite input positions are retained as ``NaN`` and are excluded from the
    number of tested hypotheses.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    values = np.asarray(pvalues, dtype=float)
    if np.any((values[np.isfinite(values)] < 0) | (values[np.isfinite(values)] > 1)):
        raise ValueError("finite p-values must be inside [0, 1]")
    flat = values.reshape(-1)
    finite_positions = np.flatnonzero(np.isfinite(flat))
    adjusted_flat = np.full(flat.shape, np.nan, dtype=float)
    rejected_flat = np.zeros(flat.shape, dtype=bool)
    if finite_positions.size == 0:
        return adjusted_flat.reshape(values.shape), rejected_flat.reshape(values.shape)

    finite_values = flat[finite_positions]
    order = np.argsort(finite_values, kind="mergesort")
    ordered = finite_values[order]
    m = ordered.size
    ranks = np.arange(1, m + 1)
    adjusted_ordered = ordered * m / ranks
    adjusted_ordered = np.minimum.accumulate(adjusted_ordered[::-1])[::-1]
    adjusted_ordered = np.clip(adjusted_ordered, 0.0, 1.0)
    adjusted_finite = np.empty(m, dtype=float)
    adjusted_finite[order] = adjusted_ordered
    adjusted_flat[finite_positions] = adjusted_finite
    rejected_flat[finite_positions] = adjusted_finite <= alpha
    return adjusted_flat.reshape(values.shape), rejected_flat.reshape(values.shape)


def bh_adjust(pvalues: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convenience wrapper returning only BH-adjusted p-values."""

    return benjamini_hochberg(pvalues)[0]


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate, percentile interval, and bootstrap replicates."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    block_length: int
    n_bootstrap: int
    distribution: np.ndarray


def moving_block_bootstrap(
    data: np.ndarray | Sequence[np.ndarray],
    statistic: Callable[..., float] | None = None,
    *,
    block_length: int = 96,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int | np.random.Generator | None = None,
) -> BootstrapResult:
    """Percentile moving-block bootstrap along the first (time) dimension.

    Pass a sequence of arrays for paired resampling, for example true labels and
    ramp probabilities.  Blocks are sampled serially in the current process;
    no ``fork`` or platform-specific multiprocessing behavior is used.
    """

    if isinstance(data, np.ndarray):
        arrays = (np.asarray(data),)
    else:
        arrays = tuple(np.asarray(value) for value in data)
        if not arrays:
            raise ValueError("data must contain at least one array")
    n = int(arrays[0].shape[0]) if arrays[0].ndim else 0
    if n < 1 or any(array.ndim == 0 or array.shape[0] != n for array in arrays):
        raise ValueError("all arrays must share a non-empty first dimension")
    if block_length < 1 or n_bootstrap < 1:
        raise ValueError("block_length and n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    length = min(int(block_length), n)
    function = statistic or (lambda values: float(np.mean(values)))
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    blocks_needed = int(np.ceil(n / length))
    last_start = n - length
    distribution = np.empty(int(n_bootstrap), dtype=float)
    for iteration in range(int(n_bootstrap)):
        starts = rng.integers(0, last_start + 1, size=blocks_needed)
        indices = np.concatenate(
            [np.arange(start, start + length, dtype=int) for start in starts]
        )[:n]
        samples = tuple(array[indices] for array in arrays)
        distribution[iteration] = float(function(*samples))
    estimate = float(function(*arrays))
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(distribution, [tail, 1.0 - tail])
    return BootstrapResult(
        estimate=estimate,
        ci_low=float(low),
        ci_high=float(high),
        confidence=float(confidence),
        block_length=length,
        n_bootstrap=int(n_bootstrap),
        distribution=distribution,
    )


dm_test = diebold_mariano_test
moving_block_bootstrap_ci = moving_block_bootstrap
