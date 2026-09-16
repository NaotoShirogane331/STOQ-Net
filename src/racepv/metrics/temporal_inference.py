"""Hourly-grid loss aggregation and fixed-lag HAC inference."""

from math import erfc, sqrt
from numbers import Integral

import numpy as np

HOUR_NS = 3_600_000_000_000


def _ns(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("time must be integer nanoseconds")
    result = int(value)
    if not -(2**63) < result < 2**63:
        raise ValueError("time outside supported range or NaT")
    return result


def _times(value: object) -> list[int]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError("origin_ns must be a one-dimensional integer array")
    return [_ns(item) for item in array]


def _real(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError("real numeric array required")
    return array


def build_hourly_origin_losses(
    loss_sum: object,
    valid_count: object,
    origin_ns: object,
    *,
    hour_anchor_ns: object,
    range_start_ns: object,
    range_stop_ns: object,
) -> dict[str, object]:
    """Build an absolute hourly grid from per-seed, per-origin sufficient statistics."""

    anchor, start, stop = map(_ns, (hour_anchor_ns, range_start_ns, range_stop_ns))
    if stop <= start:
        raise ValueError("empty or reversed evaluation range")

    times = _times(origin_ns)
    if len(set(times)) != len(times) or any((time - anchor) % (HOUR_NS // 4) for time in times):
        raise ValueError("duplicate or off-quarter-hour origin")

    sums = _real(loss_sum)
    counts = np.asarray(valid_count)
    if sums.ndim != 2 or sums.shape[0] == 0 or sums.shape[1] != len(times):
        raise ValueError("sum must have nonempty seed axis and aligned origin axis")
    if counts.shape != sums.shape or counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise ValueError("count must be aligned nonnegative integer data")
    if np.any(counts != counts[:1]):
        raise ValueError("seed support counts differ")
    if not np.isfinite(sums).all() or np.any(sums < 0) or np.any(sums[counts == 0] != 0):
        raise ValueError("invalid sufficient statistics")

    first = start + ((anchor - start) % HOUR_NS)
    grid = list(range(first, stop, HOUR_NS))
    lookup = {time: index for index, time in enumerate(times)}
    present = np.array([time in lookup for time in grid], dtype=bool)
    valid = np.zeros(len(grid), dtype=bool)
    seed_losses = np.full((sums.shape[0], len(grid)), np.nan)
    losses = np.full(len(grid), np.nan)

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for index, time in enumerate(grid):
                if time not in lookup:
                    continue
                source_index = lookup[time]
                if counts[0, source_index] == 0:
                    continue
                numerator = sums[:, source_index].astype(np.float64)
                denominator = counts[:, source_index].astype(np.float64)
                values = numerator / denominator
                mean = float(np.mean(values))
                if not np.isfinite(values).all() or not np.isfinite(mean):
                    raise ValueError("nonfinite hourly loss")
                seed_losses[:, index] = values
                losses[index] = mean
                valid[index] = True
    except FloatingPointError as exc:
        raise ValueError("hourly arithmetic overflow or invalid value") from exc

    return {
        "origin_ns": np.array(grid, dtype=np.int64),
        "present": present,
        "valid": valid,
        "seed_losses": seed_losses,
        "loss": losses,
        "diagnostics": {
            "hour_anchor_ns": anchor,
            "range_start_ns": start,
            "range_stop_ns": stop,
            "input_origins": len(times),
            "grid_origins": len(grid),
            "valid_origins": int(valid.sum()),
            "excluded_out_of_range": sum(not start <= time < stop for time in times),
            "excluded_nonhour_phase": sum(
                start <= time < stop and (time - anchor) % HOUR_NS != 0 for time in times
            ),
            "missing_hourly_origins": int((~present).sum()),
            "present_zero_support_origins": int((present & ~valid).sum()),
        },
    }


def dm_hac_on_hourly_grid(
    difference: object,
    valid: object,
    origin_ns: object,
    *,
    lag: object = 5,
) -> dict[str, object]:
    """Compute the registered observed-pair Bartlett HAC on a complete hourly grid."""

    if isinstance(lag, (bool, np.bool_)) or not isinstance(lag, Integral) or lag != 5:
        raise ValueError("registered lag is exactly 5")

    times = _times(origin_ns)
    if any(right - left != HOUR_NS for left, right in zip(times, times[1:], strict=False)):
        raise ValueError("complete increasing hourly grid required")

    raw = _real(difference)
    mask = np.asarray(valid)
    if raw.shape != (len(times),) or mask.shape != raw.shape or mask.dtype.kind != "b":
        raise ValueError("difference and boolean validity must match the grid exactly")

    selected = raw[mask]
    if not np.isfinite(selected).all():
        raise ValueError("eligible differences must be finite")

    positions = np.flatnonzero(mask)
    gaps: dict[str, int] = {}
    for left, right in zip(positions, positions[1:], strict=False):
        key = str(int(right - left))
        gaps[key] = gaps.get(key, 0) + 1

    n = len(positions)
    result: dict[str, object] = {
        "hac_id": "hourly_observed_pair_bartlett_v1",
        "estimand": "equal_hourly_origin_mean_loss_difference",
        "requested_lag": 5,
        "effective_lag": 5,
        "n_observations": n,
        "grid_origins": len(times),
        "grid_start_ns": times[0] if times else None,
        "grid_stop_ns_exclusive": _ns(times[-1] + HOUR_NS) if times else None,
        "valid_span_ns": times[int(positions[-1])] - times[int(positions[0])] if n else 0,
        "valid_utc_days": len({times[int(index)] // (24 * HOUR_NS) for index in positions}),
        "adjacent_valid_gap_hours": gaps,
        "weights": [1.0] + [1 - order / 6 for order in range(1, 6)],
        "pair_counts": [],
        "gamma": [],
        "mean_difference": None,
        "long_run_variance": None,
        "variance_of_mean": None,
        "variance_roundoff_clamped": False,
        "variance_roundoff_tolerance": None,
        "statistic": None,
        "p_value": None,
        "unavailable_reason": None,
    }

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            values = selected.astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError("float64 difference overflow")
            if np.any((selected != 0) & (values == 0)):
                raise ValueError("float64 difference underflow")

            if n and np.all(values == values[0]):
                mean = float(values[0])
            else:
                mean = float(np.mean(values)) if n else None
            if n and not np.isfinite(mean):
                raise ValueError("mean overflow")
            result["mean_difference"] = mean

            centered = np.zeros(len(times), dtype=np.float64)
            if n:
                centered[mask] = values - mean

            gamma: list[float | None] = []
            pair_counts: list[int] = []
            for order in range(6):
                pairs = mask if order == 0 else mask[order:] & mask[:-order]
                left = centered if order == 0 else centered[order:]
                right = centered if order == 0 else centered[:-order]
                pair_counts.append(int(pairs.sum()))
                gamma.append(float(np.dot(left[pairs], right[pairs]) / n) if n else None)
            result["pair_counts"] = pair_counts
            result["gamma"] = gamma

            if n and not np.isfinite(gamma).all():
                raise ValueError("autocovariance overflow")
            if n < 7 or result["valid_span_ns"] < 6 * HOUR_NS:
                result["unavailable_reason"] = "insufficient_support"
                return result

            numeric_gamma = [float(value) for value in gamma]
            terms = np.array(
                [numeric_gamma[0]]
                + [2 * (1 - order / 6) * numeric_gamma[order] for order in range(1, 6)]
            )
            long_run_variance = float(np.sum(terms))
            tolerance = float(64 * np.finfo(np.float64).eps * np.sum(np.abs(terms)))
            if (
                not np.isfinite(long_run_variance)
                or not np.isfinite(tolerance)
                or long_run_variance < -tolerance
            ):
                raise ValueError("invalid long-run variance")
            result["variance_roundoff_tolerance"] = tolerance
            if long_run_variance < 0:
                long_run_variance = 0.0
                result["variance_roundoff_clamped"] = True

            variance = long_run_variance / n
            result["long_run_variance"] = long_run_variance
            result["variance_of_mean"] = variance
            if variance == 0:
                if np.all(values == 0):
                    result["statistic"] = 0.0
                    result["p_value"] = 1.0
                else:
                    result["unavailable_reason"] = (
                        "degenerate_nonzero_constant"
                        if np.all(values == values[0])
                        else "degenerate_zero_variance"
                    )
                return result

            statistic = mean / sqrt(variance)
            if not np.isfinite(statistic):
                raise ValueError("statistic overflow")
            result["statistic"] = statistic
            result["p_value"] = erfc(abs(statistic) / sqrt(2.0))
            return result
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("HAC arithmetic overflow or invalid value") from exc
