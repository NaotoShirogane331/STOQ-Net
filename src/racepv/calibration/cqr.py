"""Time-blocked, state-conditional conformalized quantile regression.

The calibrator in this module is deliberately model agnostic: it consumes NumPy
arrays emitted by any quantile forecaster.  Calibration groups are defined by
the *predicted* state (stable versus either ramp direction) and by physical lead
time.  True future states must never be passed as routing groups at inference.

The registered ``block_reducer="subsample"`` draws one score, independently of
its value, from every eligible non-overlapping site-day block.  It therefore
estimates a day-balanced marginal correction without pretending that
overlapping forecast cells are independent. In the conditional calibrator this
target is defined separately inside every predicted-state/lead-band stratum.
``block_reducer="max"`` targets the stronger simultaneous event *within each
fitted stratum* (and within the whole site-day for :class:`GlobalCQR`). All modes
use the finite-sample conformal rank ``ceil((n + 1) * coverage)``; no
unconditional finite-sample guarantee is claimed for a non-stationary time
series.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
DEFAULT_COVERAGES = (0.80, 0.90)


@dataclass(frozen=True)
class CalibrationEntry:
    """Audit information for one fitted calibration correction.

    ``n_*`` always describes the scores that actually determine ``correction``.
    ``group_n_*`` describes the requested conditional group before fallback.
    Keeping both populations explicit prevents a fallback row from mixing a
    conditional raw count with an all-state effective-score count.
    """

    correction: float
    source: str
    n_raw: int
    n_scores: int
    n_blocks: int
    conformal_rank: int
    group_n_raw: int | None = None
    group_n_scores: int | None = None
    group_n_blocks: int | None = None
    used_fallback: bool = False


def finite_sample_quantile(scores: np.ndarray, coverage: float) -> tuple[float, int]:
    """Return the split-conformal upper quantile and its one-based rank.

    The empirical rank is ``ceil((n + 1) * coverage)``.  If that rank is
    ``n + 1``, the augmented conformal order statistic is infinity: returning
    the largest observed score instead would silently claim an unattainable
    finite-sample coverage level.  For ordinary RaCE-PV groups (at least 200
    effective scores) the rank is always finite for the requested 80/90% intervals.
    """

    values = np.asarray(scores, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("at least one finite conformity score is required")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between 0 and 1")
    rank = int(np.ceil((values.size + 1) * coverage))
    if rank > values.size:
        return float("inf"), rank
    value = float(np.partition(values, rank - 1)[rank - 1])
    return value, rank


def lead_band_names(horizon: int, step_minutes: int = 15) -> np.ndarray:
    """Map each one-based forecast lead to the three RaCE-PV lead bands."""

    if horizon < 1 or step_minutes < 1:
        raise ValueError("horizon and step_minutes must be positive")
    minutes = (np.arange(horizon) + 1) * step_minutes
    if np.any(minutes > 360):
        raise ValueError("RaCE-PV lead bands are defined only through 6 hours")
    return np.where(
        minutes <= 60,
        "0-1h",
        np.where(minutes <= 180, "1-3h", "3-6h"),
    )


def _as_targets(values: np.ndarray, name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=float)
    squeezed = array.ndim == 1
    if squeezed:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [N, H] (or [N] for H=1)")
    return array, squeezed


def _as_quantile_cube(values: np.ndarray, n: int, h: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and h == 1 and array.shape[0] == n:
        array = array[:, None, :]
    if array.ndim != 3 or array.shape[:2] != (n, h):
        raise ValueError("quantile_predictions must have shape [N, H, Q]")
    return array


def _broadcast_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    array = np.asarray(mask, dtype=bool)
    try:
        return np.broadcast_to(array, shape).copy()
    except ValueError as exc:
        raise ValueError(f"mask cannot be broadcast to {shape}") from exc


def _state_labels(states: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(states)
    if array.ndim == 3:
        if array.shape[:2] != shape or array.shape[-1] != 3:
            raise ValueError("state probabilities must have shape [N, H, 3]")
        if not np.all(np.isfinite(array)):
            raise ValueError("state probabilities contain non-finite values")
        array = np.argmax(array, axis=-1)
    elif array.ndim == 1 and shape[1] == 1 and array.shape[0] == shape[0]:
        array = array[:, None]
    if array.shape != shape:
        raise ValueError("predicted_states must have shape [N, H] or [N, H, 3]")

    if array.dtype.kind in "OUS":
        normalized = np.char.lower(array.astype(str))
        output = np.full(shape, -1, dtype=int)
        output[np.isin(normalized, ["stable", "0"])] = 0
        output[np.isin(normalized, ["ramp-up", "ramp_up", "up", "1"])] = 1
        output[np.isin(normalized, ["ramp-down", "ramp_down", "down", "2"])] = 2
        array = output
    else:
        array = array.astype(int, copy=False)
    if np.any((array < 0) | (array > 2)):
        raise ValueError("state labels must be stable/0, ramp-up/1, or ramp-down/2")
    return array


def _state_groups(states: np.ndarray) -> np.ndarray:
    return np.where(states == 0, "stable", "ramp")


def _target_day_ids(
    timestamps: np.ndarray | Sequence | None,
    n: int,
    h: int,
    step_minutes: int,
    samples_per_day: int | None,
) -> np.ndarray:
    """Build non-overlapping target-day identifiers for every forecast cell."""

    def numeric_is_epoch_nanoseconds(values: np.ndarray) -> bool:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        # Explicit day IDs (typically around 20,000 since 1970, or small study
        # indices) are safely below this threshold. pandas ``.asi8`` timestamps
        # are around 1e18 and must first be interpreted as datetime64[ns].
        return bool(finite.size and np.nanmedian(np.abs(finite)) >= 1e14)

    def nanoseconds_to_days(values: np.ndarray) -> np.ndarray:
        integer_values = np.asarray(values).astype(np.int64)
        datetimes = integer_values.astype("datetime64[ns]")
        if np.any(np.isnat(datetimes)):
            raise ValueError("timestamps contain pandas NaT/iNaT values")
        return datetimes.astype("datetime64[D]").astype(np.int64)

    if timestamps is None:
        per_day = samples_per_day or max(1, int(round(24 * 60 / step_minutes)))
        return (np.arange(n)[:, None] + np.arange(1, h + 1)[None, :]) // per_day

    raw = np.asarray(timestamps)
    if raw.shape == (n, h):
        if np.issubdtype(raw.dtype, np.number):
            # Registered evaluation uses stable uint64 hashes for explicit
            # site-day block identities. They are identifiers, never epochs.
            if raw.dtype == np.dtype(np.uint64):
                return raw.copy()
            if numeric_is_epoch_nanoseconds(raw):
                return nanoseconds_to_days(raw)
            # Small two-dimensional numeric arrays are explicit target-day IDs.
            return raw.copy()
        try:
            return raw.astype("datetime64[D]").astype(np.int64)
        except (TypeError, ValueError):
            return raw.astype(str)
    if raw.shape != (n,):
        raise ValueError("timestamps must have shape [N] (origins) or [N, H] (targets)")

    if np.issubdtype(raw.dtype, np.number):
        if not numeric_is_epoch_nanoseconds(raw):
            # Small numeric one-dimensional values are explicit origin-day IDs.
            return np.broadcast_to(raw[:, None], (n, h)).copy()
        origins = np.asarray(raw).astype(np.int64).astype("datetime64[ns]")
        if np.any(np.isnat(origins)):
            raise ValueError("timestamps contain pandas NaT/iNaT values")
        leads = (np.arange(1, h + 1) * step_minutes).astype("timedelta64[m]")
        targets = origins[:, None] + leads[None, :]
        return targets.astype("datetime64[D]").astype(np.int64)
    try:
        origins = raw.astype("datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamps must be datetime-like or numeric day IDs") from exc
    leads = (np.arange(1, h + 1) * step_minutes).astype("timedelta64[m]")
    targets = origins[:, None] + leads[None, :]
    return targets.astype("datetime64[D]").astype(np.int64)


def _sample_ids(
    values: np.ndarray | Sequence | None,
    n: int,
    h: int,
) -> np.ndarray | None:
    """Validate optional value-independent identifiers for forecast cells."""

    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim == 1 and h == 1 and array.shape[0] == n:
        array = array[:, None]
    if array.shape != (n, h):
        raise ValueError("sample_ids must have shape [N, H] (or [N] for H=1)")
    return array


def _stable_scalar_token(value: object) -> str:
    """Serialize a scalar deterministically for registered hashing/sorting."""

    if isinstance(value, np.generic):
        if isinstance(value, np.datetime64):
            return f"datetime64:{value.astype('datetime64[ns]').astype(np.int64)}"
        if isinstance(value, np.timedelta64):
            return f"timedelta64:{value.astype('timedelta64[ns]').astype(np.int64)}"
        value = value.item()
    if isinstance(value, bytes):
        return f"bytes:{value.hex()}"
    if isinstance(value, str):
        return f"str:{value}"
    if isinstance(value, (bool, np.bool_)):
        return f"bool:{int(value)}"
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("block and sample identifiers must be finite")
        return f"float:{value.hex()}"
    return f"{type(value).__qualname__}:{value!r}"


def _reduce_blocks(
    scores: np.ndarray,
    blocks: np.ndarray,
    reducer: str,
    *,
    seed: int,
    sample_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    values = np.asarray(scores, dtype=float).reshape(-1)
    ids = np.asarray(blocks).reshape(-1)
    if ids.size != values.size:
        raise ValueError("scores and blocks must contain the same number of cells")
    keys = None if sample_ids is None else np.asarray(sample_ids).reshape(-1)
    if keys is not None and keys.size != values.size:
        raise ValueError("sample_ids and scores must contain the same number of cells")
    if (
        keys is not None
        and np.issubdtype(keys.dtype, np.number)
        and not np.all(np.isfinite(keys))
    ):
        raise ValueError("sample identifiers must be finite")
    if np.issubdtype(ids.dtype, np.number) and not np.all(np.isfinite(ids)):
        raise ValueError("block identifiers must be finite")
    finite = np.isfinite(values)
    values, ids = values[finite], ids[finite]
    if keys is not None:
        keys = keys[finite]
    if values.size == 0:
        return values, 0
    unique, inverse = np.unique(ids, return_inverse=True)
    if reducer == "none":
        return values, int(unique.size)
    # Group once in O(N log N), rather than rescanning all N cells for each of
    # B site-days (the previous O(N*B) path is prohibitive on multi-site data).
    group_order = np.argsort(inverse, kind="stable")
    group_sizes = np.bincount(inverse, minlength=unique.size)
    stops = np.cumsum(group_sizes)
    starts = np.concatenate((np.asarray((0,), dtype=int), stops[:-1]))
    reduced = []
    for group_index, block in enumerate(unique):
        positions = group_order[starts[group_index] : stops[group_index]]
        block_values = values[positions]
        if reducer == "subsample":
            # The selected position depends only on the registered seed and
            # stable block identity. When stable sample IDs are supplied, the
            # eligible cells are first put into a canonical order, so archive
            # row permutations cannot change the draw. Neither ordering nor
            # indexing uses targets, forecasts, or conformity-score values.
            if keys is not None:
                block_keys = keys[positions]
                if np.issubdtype(block_keys.dtype, np.number) or block_keys.dtype.kind in {
                    "S",
                    "U",
                }:
                    order = np.argsort(block_keys, kind="stable")
                    ordered_keys = block_keys[order]
                    duplicate = bool(
                        ordered_keys.size > 1
                        and np.any(ordered_keys[1:] == ordered_keys[:-1])
                    )
                else:
                    tokens = [_stable_scalar_token(value) for value in block_keys]
                    duplicate = len(set(tokens)) != len(tokens)
                    order = np.asarray(
                        sorted(range(len(tokens)), key=tokens.__getitem__), dtype=int
                    )
                if duplicate:
                    raise ValueError(
                        "sample_ids must be unique within every calibration block"
                    )
                block_values = block_values[order]
            digest = hashlib.blake2b(
                f"{seed}|{_stable_scalar_token(block)}".encode(),
                digest_size=8,
            ).digest()
            index = int.from_bytes(digest, "little") % block_values.size
            reduced.append(block_values[index])
        elif reducer == "max":
            reduced.append(np.max(block_values))
        elif reducer == "mean":
            reduced.append(np.mean(block_values))
        elif reducer == "median":
            reduced.append(np.median(block_values))
        else:
            raise ValueError(
                "block_reducer must be one of: subsample, none, max, mean, median"
            )
    return np.asarray(reduced, dtype=float), int(unique.size)


def _registered_coverage_summary(
    covered: np.ndarray,
    widths: np.ndarray,
    valid: np.ndarray,
    blocks: np.ndarray,
    reducer: str,
    *,
    seed: int,
    target: float,
    sample_ids: np.ndarray | None,
) -> dict[str, object]:
    """Score empirical coverage on the same effective unit as calibration."""

    if not np.any(valid):
        return {
            "unit": "none",
            "count": 0,
            "picp": float("nan"),
            "mean_width": float("nan"),
            "coverage_error": float("nan"),
        }
    selected_blocks = blocks[valid]
    selected_ids = None if sample_ids is None else sample_ids[valid]
    if reducer == "subsample":
        coverage_values, n_blocks = _reduce_blocks(
            covered[valid].astype(float),
            selected_blocks,
            "subsample",
            seed=seed,
            sample_ids=selected_ids,
        )
        width_values, width_n_blocks = _reduce_blocks(
            widths[valid],
            selected_blocks,
            "subsample",
            seed=seed,
            sample_ids=selected_ids,
        )
        if width_n_blocks != n_blocks:
            raise RuntimeError("coverage and width block reductions disagree")
        picp = float(np.mean(coverage_values))
        mean_width = float(np.mean(width_values))
        unit = "one_hash_selected_cell_per_eligible_site_day"
        count = int(n_blocks)
    elif reducer == "max":
        _, inverse = np.unique(selected_blocks, return_inverse=True)
        n_blocks = int(np.max(inverse)) + 1
        block_coverage = np.ones(n_blocks, dtype=bool)
        np.logical_and.at(block_coverage, inverse, covered[valid])
        block_counts = np.bincount(inverse, minlength=n_blocks)
        block_width = np.bincount(
            inverse, weights=widths[valid], minlength=n_blocks
        ) / block_counts
        picp = float(np.mean(block_coverage))
        mean_width = float(np.mean(block_width))
        unit = "joint_event_within_each_eligible_site_day_stratum"
        count = n_blocks
    elif reducer == "none":
        picp = float(np.mean(covered[valid]))
        mean_width = float(np.mean(widths[valid]))
        unit = "forecast_cell_diagnostic"
        count = int(np.sum(valid))
    else:
        return {
            "unit": f"undefined_for_{reducer}_diagnostic",
            "count": 0,
            "picp": float("nan"),
            "mean_width": float("nan"),
            "coverage_error": float("nan"),
        }
    return {
        "unit": unit,
        "count": count,
        "picp": picp,
        "mean_width": mean_width,
        "coverage_error": abs(picp - target),
    }


class TimeBlockedConditionalCQR:
    """State- and lead-conditional CQR with auditable daily blocks.

    Parameters
    ----------
    quantile_levels:
        Levels corresponding to the last prediction dimension.
    coverages:
        Central interval coverage levels. RaCE-PV uses 0.80 and 0.90.
    min_group_size:
        Minimum number of effective finite conformity scores required for a
        state-specific group (one site-day score for the registered
        ``subsample`` mode). Smaller groups use the all-state correction from the
        same lead band.
    block_reducer:
        ``"subsample"`` selects one score per non-overlapping site-day using a
        value-independent registered hash and targets day-balanced marginal
        coverage inside each fitted state/lead-band stratum. ``"none"`` is an
        unblocked cell-level diagnostic; ``"max"`` uses one maximum score per
        site-day and stratum for simultaneous coverage within that stratum.
        ``"mean"`` and ``"median"`` are diagnostic options and do not carry a
        standard conformal coverage interpretation.
    """

    def __init__(
        self,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
        coverages: Sequence[float] = DEFAULT_COVERAGES,
        *,
        step_minutes: int = 15,
        min_group_size: int = 200,
        block_reducer: str = "subsample",
        block_seed: int = 20260804,
        clip_scores_at_zero: bool = True,
        bounds: tuple[float, float] | None = (0.0, 1.05),
        samples_per_day: int | None = None,
    ) -> None:
        levels = np.asarray(quantile_levels, dtype=float)
        if levels.ndim != 1 or levels.size < 2:
            raise ValueError("quantile_levels must be a one-dimensional sequence")
        if np.any(np.diff(levels) <= 0) or levels[0] <= 0 or levels[-1] >= 1:
            raise ValueError("quantile_levels must be strictly increasing inside (0, 1)")
        requested = tuple(float(value) for value in coverages)
        if not requested or any(not 0 < value < 1 for value in requested):
            raise ValueError("coverages must contain values strictly inside (0, 1)")
        if step_minutes <= 0 or min_group_size < 1:
            raise ValueError("step_minutes and min_group_size must be positive")
        if block_reducer not in {"subsample", "none", "max", "mean", "median"}:
            raise ValueError("invalid block_reducer")
        if bounds is not None and not bounds[0] < bounds[1]:
            raise ValueError("bounds must be an increasing (lower, upper) pair")

        self.quantile_levels = levels
        self.coverages = requested
        self.step_minutes = int(step_minutes)
        self.min_group_size = int(min_group_size)
        self.block_reducer = block_reducer
        self.block_seed = int(block_seed)
        self.clip_scores_at_zero = bool(clip_scores_at_zero)
        self.bounds = bounds
        self.samples_per_day = samples_per_day
        self.calibration_: dict[tuple[float, str, str], CalibrationEntry] = {}
        self.horizon_: int | None = None
        self.subsample_order_basis_: str | None = None

        self._interval_indices: dict[float, tuple[int, int]] = {}
        for coverage in self.coverages:
            lower, upper = (1 - coverage) / 2, 1 - (1 - coverage) / 2
            lower_matches = np.flatnonzero(np.isclose(levels, lower, atol=1e-8))
            upper_matches = np.flatnonzero(np.isclose(levels, upper, atol=1e-8))
            if lower_matches.size == 0 or upper_matches.size == 0:
                raise ValueError(
                    f"coverage {coverage:g} requires quantiles {lower:g} and {upper:g}"
                )
            self._interval_indices[coverage] = (
                int(lower_matches[0]),
                int(upper_matches[0]),
            )

    def fit(
        self,
        y_true: np.ndarray,
        quantile_predictions: np.ndarray,
        predicted_states: np.ndarray,
        *,
        timestamps: np.ndarray | Sequence | None = None,
        sample_ids: np.ndarray | Sequence | None = None,
        mask: np.ndarray | None = None,
    ) -> TimeBlockedConditionalCQR:
        """Estimate conditional conformal corrections on calibration data.

        ``sample_ids`` should be stable, value-independent identifiers for each
        forecast cell (for example site/target-time/lead IDs). They define the
        canonical within-block order used by registered subsampling. If omitted,
        input order is used and reproducibility therefore requires an identical
        archived row order.
        """

        targets, _ = _as_targets(y_true, "y_true")
        n, h = targets.shape
        predictions = _as_quantile_cube(quantile_predictions, n, h)
        if predictions.shape[-1] != self.quantile_levels.size:
            raise ValueError("prediction quantile dimension does not match quantile_levels")
        if np.any(np.diff(predictions, axis=-1) < -1e-10):
            raise ValueError("base quantile predictions must be nondecreasing")
        states = _state_labels(predicted_states, (n, h))
        groups = _state_groups(states)
        valid = _broadcast_mask(mask, (n, h)) & np.isfinite(targets)
        valid &= np.all(np.isfinite(predictions), axis=-1)
        blocks = _target_day_ids(
            timestamps, n, h, self.step_minutes, self.samples_per_day
        )
        cell_ids = _sample_ids(sample_ids, n, h)
        bands = lead_band_names(h, self.step_minutes)

        calibration: dict[tuple[float, str, str], CalibrationEntry] = {}
        for coverage, (lower_index, upper_index) in self._interval_indices.items():
            lower = predictions[..., lower_index]
            upper = predictions[..., upper_index]
            raw_scores = np.maximum(lower - targets, targets - upper)
            if self.clip_scores_at_zero:
                raw_scores = np.maximum(raw_scores, 0.0)

            for band in np.unique(bands):
                band_mask = valid & (bands[None, :] == band)
                fallback_raw = raw_scores[band_mask]
                fallback_blocks = blocks[band_mask]
                fallback_scores, fallback_n_blocks = _reduce_blocks(
                    fallback_raw,
                    fallback_blocks,
                    self.block_reducer,
                    seed=self.block_seed,
                    sample_ids=None if cell_ids is None else cell_ids[band_mask],
                )
                if fallback_scores.size == 0:
                    raise ValueError(f"no valid calibration scores for lead band {band}")
                fallback_correction, fallback_rank = finite_sample_quantile(
                    fallback_scores, coverage
                )

                for group in ("stable", "ramp"):
                    group_mask = band_mask & (groups == group)
                    group_raw = raw_scores[group_mask]
                    group_blocks = blocks[group_mask]
                    selected, n_blocks = _reduce_blocks(
                        group_raw,
                        group_blocks,
                        self.block_reducer,
                        seed=self.block_seed,
                        sample_ids=None if cell_ids is None else cell_ids[group_mask],
                    )
                    if selected.size >= self.min_group_size:
                        correction, rank = finite_sample_quantile(selected, coverage)
                        entry = CalibrationEntry(
                            correction=correction,
                            source=f"{group}/{band}",
                            n_raw=int(group_raw.size),
                            n_scores=int(selected.size),
                            n_blocks=n_blocks,
                            conformal_rank=rank,
                            group_n_raw=int(group_raw.size),
                            group_n_scores=int(selected.size),
                            group_n_blocks=n_blocks,
                        )
                    else:
                        entry = CalibrationEntry(
                            correction=fallback_correction,
                            source=f"all-states/{band}",
                            n_raw=int(fallback_raw.size),
                            n_scores=int(fallback_scores.size),
                            n_blocks=fallback_n_blocks,
                            conformal_rank=fallback_rank,
                            group_n_raw=int(group_raw.size),
                            group_n_scores=int(selected.size),
                            group_n_blocks=n_blocks,
                            used_fallback=True,
                        )
                    calibration[(coverage, group, str(band))] = entry

        self.calibration_ = calibration
        self.horizon_ = h
        self.subsample_order_basis_ = (
            "stable_sample_ids" if cell_ids is not None else "archived_input_order"
        )
        return self

    def transform(
        self,
        quantile_predictions: np.ndarray,
        predicted_states: np.ndarray,
        *,
        night_mask: np.ndarray | None = None,
    ) -> dict[float, np.ndarray]:
        """Apply learned corrections and return ``[N, H, 2]`` intervals."""

        if self.horizon_ is None or not self.calibration_:
            raise RuntimeError("fit must be called before transform")
        predictions = np.asarray(quantile_predictions, dtype=float)
        squeezed = False
        if predictions.ndim == 2 and self.horizon_ == 1:
            predictions = predictions[:, None, :]
            squeezed = True
        if predictions.ndim != 3 or predictions.shape[1] != self.horizon_:
            raise ValueError("quantile_predictions have a different fitted horizon")
        if predictions.shape[-1] != self.quantile_levels.size:
            raise ValueError("prediction quantile dimension does not match quantile_levels")
        n, h, _ = predictions.shape
        states = _state_labels(predicted_states, (n, h))
        groups = _state_groups(states)
        bands = lead_band_names(h, self.step_minutes)
        night = _broadcast_mask(night_mask, (n, h)) if night_mask is not None else None

        output: dict[float, np.ndarray] = {}
        for coverage, (lower_index, upper_index) in self._interval_indices.items():
            lower = predictions[..., lower_index].copy()
            upper = predictions[..., upper_index].copy()
            for band in np.unique(bands):
                for group in ("stable", "ramp"):
                    selection = (bands[None, :] == band) & (groups == group)
                    correction = self.calibration_[(coverage, group, str(band))].correction
                    lower[selection] -= correction
                    upper[selection] += correction
            if self.bounds is not None:
                lower = np.clip(lower, self.bounds[0], self.bounds[1])
                upper = np.clip(upper, self.bounds[0], self.bounds[1])
            if night is not None:
                lower[night] = 0.0
                upper[night] = 0.0
            interval = np.stack((lower, upper), axis=-1)
            output[coverage] = interval[:, 0] if squeezed else interval
        return output

    def fit_transform(
        self,
        y_true: np.ndarray,
        quantile_predictions: np.ndarray,
        predicted_states: np.ndarray,
        *,
        timestamps: np.ndarray | Sequence | None = None,
        sample_ids: np.ndarray | Sequence | None = None,
        mask: np.ndarray | None = None,
        night_mask: np.ndarray | None = None,
    ) -> dict[float, np.ndarray]:
        """Fit and transform the calibration predictions (mainly for diagnostics)."""

        return self.fit(
            y_true,
            quantile_predictions,
            predicted_states,
            timestamps=timestamps,
            sample_ids=sample_ids,
            mask=mask,
        ).transform(quantile_predictions, predicted_states, night_mask=night_mask)

    def calibration_report(self) -> list[dict[str, object]]:
        """Return JSON/table-friendly audit rows for all fitted groups."""

        if not self.calibration_:
            raise RuntimeError("fit must be called before requesting a report")
        rows = []
        for (coverage, group, band), entry in sorted(self.calibration_.items()):
            rows.append(
                {
                    "coverage": coverage,
                    "state_group": group,
                    "lead_band": band,
                    "correction": entry.correction,
                    "source": entry.source,
                    "used_fallback": entry.used_fallback,
                    # n_* is always the actual correction-source population.
                    "n_raw": entry.n_raw,
                    "n_scores": entry.n_scores,
                    "n_blocks": entry.n_blocks,
                    "source_n_raw": entry.n_raw,
                    "source_n_scores": entry.n_scores,
                    "source_n_blocks": entry.n_blocks,
                    "group_n_raw": entry.group_n_raw,
                    "group_n_scores": entry.group_n_scores,
                    "group_n_blocks": entry.group_n_blocks,
                    "conformal_rank": entry.conformal_rank,
                    "block_reducer": self.block_reducer,
                    "block_seed": (
                        self.block_seed if self.block_reducer == "subsample" else None
                    ),
                    "subsample_order_basis": (
                        self.subsample_order_basis_
                        if self.block_reducer == "subsample"
                        else None
                    ),
                    "subsample_hash": (
                        "blake2b-64" if self.block_reducer == "subsample" else None
                    ),
                }
            )
        return rows

    def coverage_report(
        self,
        y_true: np.ndarray,
        calibrated_intervals: Mapping[float, np.ndarray],
        predicted_states: np.ndarray,
        *,
        timestamps: np.ndarray | Sequence | None = None,
        sample_ids: np.ndarray | Sequence | None = None,
        mask: np.ndarray | None = None,
    ) -> dict[float, dict[str, object]]:
        """Report pooled coverage plus coverage on the registered block unit."""

        targets, squeezed = _as_targets(y_true, "y_true")
        n, h = targets.shape
        if self.horizon_ is not None and h != self.horizon_:
            raise ValueError("y_true has a different fitted horizon")
        states = _state_labels(predicted_states, (n, h))
        groups = _state_groups(states)
        bands = lead_band_names(h, self.step_minutes)
        valid_base = _broadcast_mask(mask, (n, h)) & np.isfinite(targets)
        blocks = _target_day_ids(
            timestamps, n, h, self.step_minutes, self.samples_per_day
        )
        cell_ids = _sample_ids(sample_ids, n, h)

        def summarize(
            interval: np.ndarray, selection: np.ndarray, target: float
        ) -> dict[str, object]:
            valid = selection & valid_base
            valid &= np.all(np.isfinite(interval), axis=-1)
            count = int(np.sum(valid))
            if count == 0:
                return {
                    "count": 0,
                    "picp": float("nan"),
                    "mean_width": float("nan"),
                    "coverage_error": float("nan"),
                    "registered_estimand": _registered_coverage_summary(
                        np.zeros((n, h), dtype=bool),
                        np.zeros((n, h), dtype=float),
                        valid,
                        blocks,
                        self.block_reducer,
                        seed=self.block_seed,
                        target=target,
                        sample_ids=cell_ids,
                    ),
                }
            covered = (targets >= interval[..., 0]) & (targets <= interval[..., 1])
            picp = float(np.mean(covered[valid]))
            widths = interval[..., 1] - interval[..., 0]
            return {
                "count": count,
                "picp": picp,
                "mean_width": float(np.mean(widths[valid])),
                "coverage_error": abs(picp - target),
                "registered_estimand": _registered_coverage_summary(
                    covered,
                    widths,
                    valid,
                    blocks,
                    self.block_reducer,
                    seed=self.block_seed,
                    target=target,
                    sample_ids=cell_ids,
                ),
            }

        report: dict[float, dict[str, object]] = {}
        for coverage in self.coverages:
            if coverage not in calibrated_intervals:
                raise ValueError(f"calibrated_intervals is missing coverage {coverage:g}")
            interval = np.asarray(calibrated_intervals[coverage], dtype=float)
            if squeezed and interval.shape == (n, 2):
                interval = interval[:, None, :]
            if interval.shape != (n, h, 2):
                raise ValueError("each calibrated interval must have shape [N, H, 2]")
            by_state = {
                group: summarize(interval, groups == group, coverage)
                for group in ("stable", "ramp")
            }
            by_band = {
                str(band): summarize(interval, bands[None, :] == band, coverage)
                for band in np.unique(bands)
            }
            by_state_band: dict[str, object] = {}
            for group in ("stable", "ramp"):
                for band in np.unique(bands):
                    row = summarize(
                        interval,
                        (groups == group) & (bands[None, :] == band),
                        coverage,
                    )
                    entry = self.calibration_[(coverage, group, str(band))]
                    row["calibration_source"] = entry.source
                    row["conditional_target_supported"] = not entry.used_fallback
                    row["coverage_scope_note"] = (
                        "direct predicted-state/lead-band calibration target"
                        if not entry.used_fallback
                        else (
                            "conditional diagnostic only: correction targets all predicted "
                            "states in this lead band because the requested group fell back"
                        )
                    )
                    by_state_band[f"{group}/{band}"] = row
            report[coverage] = {
                "target": coverage,
                "overall": summarize(interval, np.ones((n, h), dtype=bool), coverage),
                "by_state": by_state,
                "by_lead_band": by_band,
                "by_state_and_lead_band": by_state_band,
                "registered_estimand_note": (
                    "A predicted-state/lead-band row is the registered conditional target "
                    "only when conditional_target_supported is true. Fallback rows are "
                    "conditional diagnostics for an all-state same-band correction. "
                    "Overall, state-only, and band-only rows pool separately calibrated "
                    "strata and are secondary summaries."
                ),
            }
        return report


# Concise alias used by experiment configurations.
BlockedConditionalCQR = TimeBlockedConditionalCQR


class LeadBandCQR:
    """All-state, lead-band-only CQR ablation.

    This control retains the registered physical lead bands but removes
    predicted-state conditioning. Comparing conditional CQR against this class
    therefore isolates the contribution of stable/ramp conditioning, while the
    comparison against :class:`GlobalCQR` also removes lead conditioning.
    """

    def __init__(
        self,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
        coverages: Sequence[float] = DEFAULT_COVERAGES,
        *,
        step_minutes: int = 15,
        block_reducer: str = "subsample",
        block_seed: int = 20260804,
        clip_scores_at_zero: bool = True,
        bounds: tuple[float, float] | None = (0.0, 1.05),
        samples_per_day: int | None = None,
    ) -> None:
        # Reuse the conditional implementation's parameter validation and
        # interval-index construction; fitting below uses a direct all-state
        # band pool and does not invoke conditional fallback.
        self._delegate = TimeBlockedConditionalCQR(
            quantile_levels=quantile_levels,
            coverages=coverages,
            step_minutes=step_minutes,
            min_group_size=1,
            block_reducer=block_reducer,
            block_seed=block_seed,
            clip_scores_at_zero=clip_scores_at_zero,
            bounds=bounds,
            samples_per_day=samples_per_day,
        )
        self.quantile_levels = self._delegate.quantile_levels
        self.coverages = self._delegate.coverages
        self.step_minutes = self._delegate.step_minutes
        self.block_reducer = self._delegate.block_reducer
        self.block_seed = self._delegate.block_seed
        self.bounds = self._delegate.bounds
        self.samples_per_day = self._delegate.samples_per_day
        self._interval_indices = self._delegate._interval_indices
        self.calibration_: dict[tuple[float, str], CalibrationEntry] = {}
        self.horizon_: int | None = None
        self.subsample_order_basis_: str | None = None

    def fit(
        self,
        y_true: np.ndarray,
        quantile_predictions: np.ndarray,
        predicted_states: np.ndarray | None = None,
        *,
        timestamps: np.ndarray | Sequence | None = None,
        sample_ids: np.ndarray | Sequence | None = None,
        mask: np.ndarray | None = None,
    ) -> LeadBandCQR:
        """Fit one all-state correction for each requested coverage/lead band."""

        del predicted_states
        targets, _ = _as_targets(y_true, "y_true")
        n, h = targets.shape
        predictions = _as_quantile_cube(quantile_predictions, n, h)
        if predictions.shape[-1] != self.quantile_levels.size:
            raise ValueError("prediction quantile dimension does not match quantile_levels")
        if np.any(np.diff(predictions, axis=-1) < -1e-10):
            raise ValueError("base quantile predictions must be nondecreasing")
        valid = _broadcast_mask(mask, (n, h)) & np.isfinite(targets)
        valid &= np.all(np.isfinite(predictions), axis=-1)
        blocks = _target_day_ids(
            timestamps, n, h, self.step_minutes, self.samples_per_day
        )
        cell_ids = _sample_ids(sample_ids, n, h)
        bands = lead_band_names(h, self.step_minutes)
        calibration: dict[tuple[float, str], CalibrationEntry] = {}
        for coverage, (lower_index, upper_index) in self._interval_indices.items():
            scores = np.maximum(
                predictions[..., lower_index] - targets,
                targets - predictions[..., upper_index],
            )
            if self._delegate.clip_scores_at_zero:
                scores = np.maximum(scores, 0.0)
            for band in np.unique(bands):
                band_mask = valid & (bands[None, :] == band)
                raw = scores[band_mask]
                selected, n_blocks = _reduce_blocks(
                    raw,
                    blocks[band_mask],
                    self.block_reducer,
                    seed=self.block_seed,
                    sample_ids=None if cell_ids is None else cell_ids[band_mask],
                )
                if selected.size == 0:
                    raise ValueError(f"no valid calibration scores for lead band {band}")
                correction, rank = finite_sample_quantile(selected, coverage)
                calibration[(coverage, str(band))] = CalibrationEntry(
                    correction=correction,
                    source=f"all-states/{band}",
                    n_raw=int(raw.size),
                    n_scores=int(selected.size),
                    n_blocks=n_blocks,
                    conformal_rank=rank,
                    group_n_raw=int(raw.size),
                    group_n_scores=int(selected.size),
                    group_n_blocks=n_blocks,
                )
        self.calibration_ = calibration
        self.horizon_ = h
        self.subsample_order_basis_ = (
            "stable_sample_ids" if cell_ids is not None else "archived_input_order"
        )
        return self

    def transform(
        self,
        quantile_predictions: np.ndarray,
        predicted_states: np.ndarray | None = None,
        *,
        night_mask: np.ndarray | None = None,
    ) -> dict[float, np.ndarray]:
        """Apply one all-state correction within each physical lead band."""

        del predicted_states
        if self.horizon_ is None or not self.calibration_:
            raise RuntimeError("fit must be called before transform")
        predictions = np.asarray(quantile_predictions, dtype=float)
        squeezed = False
        if predictions.ndim == 2 and self.horizon_ == 1:
            predictions = predictions[:, None, :]
            squeezed = True
        if predictions.ndim != 3 or predictions.shape[1] != self.horizon_:
            raise ValueError("quantile_predictions have a different fitted horizon")
        if predictions.shape[-1] != self.quantile_levels.size:
            raise ValueError("prediction quantile dimension does not match quantile_levels")
        n, h, _ = predictions.shape
        bands = lead_band_names(h, self.step_minutes)
        night = _broadcast_mask(night_mask, (n, h)) if night_mask is not None else None
        output: dict[float, np.ndarray] = {}
        for coverage, (lower_index, upper_index) in self._interval_indices.items():
            lower = predictions[..., lower_index].copy()
            upper = predictions[..., upper_index].copy()
            for band in np.unique(bands):
                selection = bands[None, :] == band
                correction = self.calibration_[(coverage, str(band))].correction
                lower[:, selection[0]] -= correction
                upper[:, selection[0]] += correction
            if self.bounds is not None:
                lower = np.clip(lower, self.bounds[0], self.bounds[1])
                upper = np.clip(upper, self.bounds[0], self.bounds[1])
            if night is not None:
                lower[night], upper[night] = 0.0, 0.0
            interval = np.stack((lower, upper), axis=-1)
            output[coverage] = interval[:, 0] if squeezed else interval
        return output

    def fit_transform(
        self,
        y_true: np.ndarray,
        quantile_predictions: np.ndarray,
        predicted_states: np.ndarray | None = None,
        *,
        timestamps: np.ndarray | Sequence | None = None,
        sample_ids: np.ndarray | Sequence | None = None,
        mask: np.ndarray | None = None,
        night_mask: np.ndarray | None = None,
    ) -> dict[float, np.ndarray]:
        return self.fit(
            y_true,
            quantile_predictions,
            predicted_states,
            timestamps=timestamps,
            sample_ids=sample_ids,
            mask=mask,
        ).transform(quantile_predictions, night_mask=night_mask)

    def calibration_report(self) -> list[dict[str, object]]:
        """Return one auditable all-state row per coverage/lead band."""

        if not self.calibration_:
            raise RuntimeError("fit must be called before requesting a report")
        return [
            {
                "coverage": coverage,
                "state_group": "all-states",
                "lead_band": band,
                "correction": entry.correction,
                "source": entry.source,
                "used_fallback": False,
                "n_raw": entry.n_raw,
                "n_scores": entry.n_scores,
                "n_blocks": entry.n_blocks,
                "source_n_raw": entry.n_raw,
                "source_n_scores": entry.n_scores,
                "source_n_blocks": entry.n_blocks,
                "group_n_raw": entry.group_n_raw,
                "group_n_scores": entry.group_n_scores,
                "group_n_blocks": entry.group_n_blocks,
                "conformal_rank": entry.conformal_rank,
                "block_reducer": self.block_reducer,
                "block_seed": (
                    self.block_seed if self.block_reducer == "subsample" else None
                ),
                "subsample_order_basis": (
                    self.subsample_order_basis_
                    if self.block_reducer == "subsample"
                    else None
                ),
                "subsample_hash": (
                    "blake2b-64" if self.block_reducer == "subsample" else None
                ),
            }
            for (coverage, band), entry in sorted(self.calibration_.items())
        ]

    def coverage_report(
        self,
        y_true: np.ndarray,
        calibrated_intervals: Mapping[float, np.ndarray],
        predicted_states: np.ndarray | None = None,
        *,
        timestamps: np.ndarray | Sequence | None = None,
        sample_ids: np.ndarray | Sequence | None = None,
        mask: np.ndarray | None = None,
    ) -> dict[float, dict[str, object]]:
        """Report pooled coverage and the exact all-state/lead-band estimands."""

        del predicted_states
        targets, squeezed = _as_targets(y_true, "y_true")
        n, h = targets.shape
        if self.horizon_ is not None and h != self.horizon_:
            raise ValueError("y_true has a different fitted horizon")
        bands = lead_band_names(h, self.step_minutes)
        valid_base = _broadcast_mask(mask, (n, h)) & np.isfinite(targets)
        blocks = _target_day_ids(
            timestamps, n, h, self.step_minutes, self.samples_per_day
        )
        cell_ids = _sample_ids(sample_ids, n, h)

        def summarize(
            interval: np.ndarray, selection: np.ndarray, target: float
        ) -> dict[str, object]:
            valid = selection & valid_base & np.all(np.isfinite(interval), axis=-1)
            count = int(np.sum(valid))
            if count == 0:
                covered = np.zeros((n, h), dtype=bool)
                widths = np.zeros((n, h), dtype=float)
                picp = mean_width = float("nan")
            else:
                covered = (targets >= interval[..., 0]) & (targets <= interval[..., 1])
                widths = interval[..., 1] - interval[..., 0]
                picp = float(np.mean(covered[valid]))
                mean_width = float(np.mean(widths[valid]))
            return {
                "count": count,
                "picp": picp,
                "mean_width": mean_width,
                "coverage_error": abs(picp - target),
                "registered_estimand": _registered_coverage_summary(
                    covered,
                    widths,
                    valid,
                    blocks,
                    self.block_reducer,
                    seed=self.block_seed,
                    target=target,
                    sample_ids=cell_ids,
                ),
            }

        report: dict[float, dict[str, object]] = {}
        for coverage in self.coverages:
            if coverage not in calibrated_intervals:
                raise ValueError(f"calibrated_intervals is missing coverage {coverage:g}")
            interval = np.asarray(calibrated_intervals[coverage], dtype=float)
            if squeezed and interval.shape == (n, 2):
                interval = interval[:, None, :]
            if interval.shape != (n, h, 2):
                raise ValueError("each calibrated interval must have shape [N, H, 2]")
            by_band = {}
            for band in np.unique(bands):
                row = summarize(interval, bands[None, :] == band, coverage)
                row["calibration_source"] = self.calibration_[
                    (coverage, str(band))
                ].source
                by_band[str(band)] = row
            report[coverage] = {
                "target": coverage,
                "overall": summarize(
                    interval, np.ones((n, h), dtype=bool), coverage
                ),
                "by_lead_band": by_band,
                "registered_estimand_note": (
                    "Each lead-band row is the registered all-state lead-band target; "
                    "overall coverage pools separately calibrated bands."
                ),
            }
        return report


class GlobalCQR:
    """Unconditional CQR ablation using one correction across states and leads.

    ``predicted_states`` is accepted by :meth:`fit`, :meth:`transform`, and
    :meth:`coverage_report` for drop-in compatibility with
    :class:`TimeBlockedConditionalCQR`, but it is intentionally ignored.
    """

    def __init__(
        self,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
        coverages: Sequence[float] = DEFAULT_COVERAGES,
        *,
        step_minutes: int = 15,
        block_reducer: str = "subsample",
        block_seed: int = 20260804,
        clip_scores_at_zero: bool = True,
        bounds: tuple[float, float] | None = (0.0, 1.05),
        samples_per_day: int | None = None,
    ) -> None:
        self.quantile_levels = np.asarray(quantile_levels, dtype=float)
        if (
            self.quantile_levels.ndim != 1
            or self.quantile_levels.size < 2
            or np.any(np.diff(self.quantile_levels) <= 0)
            or self.quantile_levels[0] <= 0
            or self.quantile_levels[-1] >= 1
        ):
            raise ValueError("quantile_levels must be strictly increasing inside (0, 1)")
        self.coverages = tuple(float(value) for value in coverages)
        if not self.coverages or any(not 0 < value < 1 for value in self.coverages):
            raise ValueError("coverages must contain values strictly inside (0, 1)")
        if step_minutes <= 0:
            raise ValueError("step_minutes must be positive")
        if block_reducer not in {"subsample", "none", "max", "mean", "median"}:
            raise ValueError("invalid block_reducer")
        if bounds is not None and not bounds[0] < bounds[1]:
            raise ValueError("bounds must be an increasing (lower, upper) pair")
        self.step_minutes = int(step_minutes)
        self.block_reducer = block_reducer
        self.block_seed = int(block_seed)
        self.clip_scores_at_zero = bool(clip_scores_at_zero)
        self.bounds = bounds
        self.samples_per_day = samples_per_day
        self.calibration_: dict[float, CalibrationEntry] = {}
        self.horizon_: int | None = None
        self.subsample_order_basis_: str | None = None
        self._interval_indices: dict[float, tuple[int, int]] = {}
        for coverage in self.coverages:
            lower, upper = (1 - coverage) / 2, 1 - (1 - coverage) / 2
            lower_matches = np.flatnonzero(
                np.isclose(self.quantile_levels, lower, atol=1e-8)
            )
            upper_matches = np.flatnonzero(
                np.isclose(self.quantile_levels, upper, atol=1e-8)
            )
            if not lower_matches.size or not upper_matches.size:
                raise ValueError(
                    f"coverage {coverage:g} requires quantiles {lower:g} and {upper:g}"
                )
            self._interval_indices[coverage] = (
                int(lower_matches[0]),
                int(upper_matches[0]),
            )

    def fit(
        self,
        y_true: np.ndarray,
        quantile_predictions: np.ndarray,
        predicted_states: np.ndarray | None = None,
        *,
        timestamps: np.ndarray | Sequence | None = None,
        sample_ids: np.ndarray | Sequence | None = None,
        mask: np.ndarray | None = None,
    ) -> GlobalCQR:
        """Fit one correction per coverage from the all-state/all-lead pool."""

        del predicted_states
        targets, _ = _as_targets(y_true, "y_true")
        n, h = targets.shape
        predictions = _as_quantile_cube(quantile_predictions, n, h)
        if predictions.shape[-1] != self.quantile_levels.size:
            raise ValueError("prediction quantile dimension does not match quantile_levels")
        if np.any(np.diff(predictions, axis=-1) < -1e-10):
            raise ValueError("base quantile predictions must be nondecreasing")
        valid = _broadcast_mask(mask, (n, h)) & np.isfinite(targets)
        valid &= np.all(np.isfinite(predictions), axis=-1)
        blocks = _target_day_ids(
            timestamps, n, h, self.step_minutes, self.samples_per_day
        )
        cell_ids = _sample_ids(sample_ids, n, h)
        calibration: dict[float, CalibrationEntry] = {}
        for coverage, (lower_index, upper_index) in self._interval_indices.items():
            scores = np.maximum(
                predictions[..., lower_index] - targets,
                targets - predictions[..., upper_index],
            )
            if self.clip_scores_at_zero:
                scores = np.maximum(scores, 0.0)
            raw = scores[valid]
            selected, n_blocks = _reduce_blocks(
                raw,
                blocks[valid],
                self.block_reducer,
                seed=self.block_seed,
                sample_ids=None if cell_ids is None else cell_ids[valid],
            )
            correction, rank = finite_sample_quantile(selected, coverage)
            calibration[coverage] = CalibrationEntry(
                correction=correction,
                source="global/all",
                n_raw=int(raw.size),
                n_scores=int(selected.size),
                n_blocks=n_blocks,
                conformal_rank=rank,
                group_n_raw=int(raw.size),
                group_n_scores=int(selected.size),
                group_n_blocks=n_blocks,
            )
        self.calibration_ = calibration
        self.horizon_ = h
        self.subsample_order_basis_ = (
            "stable_sample_ids" if cell_ids is not None else "archived_input_order"
        )
        return self

    def transform(
        self,
        quantile_predictions: np.ndarray,
        predicted_states: np.ndarray | None = None,
        *,
        night_mask: np.ndarray | None = None,
    ) -> dict[float, np.ndarray]:
        """Apply the same correction to all states and forecast lead times."""

        del predicted_states
        if self.horizon_ is None or not self.calibration_:
            raise RuntimeError("fit must be called before transform")
        predictions = np.asarray(quantile_predictions, dtype=float)
        squeezed = False
        if predictions.ndim == 2 and self.horizon_ == 1:
            predictions = predictions[:, None, :]
            squeezed = True
        if predictions.ndim != 3 or predictions.shape[1] != self.horizon_:
            raise ValueError("quantile_predictions have a different fitted horizon")
        if predictions.shape[-1] != self.quantile_levels.size:
            raise ValueError("prediction quantile dimension does not match quantile_levels")
        n, h, _ = predictions.shape
        night = _broadcast_mask(night_mask, (n, h)) if night_mask is not None else None
        output: dict[float, np.ndarray] = {}
        for coverage, (lower_index, upper_index) in self._interval_indices.items():
            correction = self.calibration_[coverage].correction
            lower = predictions[..., lower_index] - correction
            upper = predictions[..., upper_index] + correction
            if self.bounds is not None:
                lower = np.clip(lower, self.bounds[0], self.bounds[1])
                upper = np.clip(upper, self.bounds[0], self.bounds[1])
            if night is not None:
                lower[night], upper[night] = 0.0, 0.0
            interval = np.stack((lower, upper), axis=-1)
            output[coverage] = interval[:, 0] if squeezed else interval
        return output

    def fit_transform(
        self,
        y_true: np.ndarray,
        quantile_predictions: np.ndarray,
        predicted_states: np.ndarray | None = None,
        *,
        timestamps: np.ndarray | Sequence | None = None,
        sample_ids: np.ndarray | Sequence | None = None,
        mask: np.ndarray | None = None,
        night_mask: np.ndarray | None = None,
    ) -> dict[float, np.ndarray]:
        return self.fit(
            y_true,
            quantile_predictions,
            predicted_states,
            timestamps=timestamps,
            sample_ids=sample_ids,
            mask=mask,
        ).transform(
            quantile_predictions,
            predicted_states,
            night_mask=night_mask,
        )

    def calibration_report(self) -> list[dict[str, object]]:
        """Return one audit row for each interval coverage."""

        if not self.calibration_:
            raise RuntimeError("fit must be called before requesting a report")
        return [
            {
                "coverage": coverage,
                "state_group": "global",
                "lead_band": "all",
                "correction": entry.correction,
                "source": entry.source,
                "used_fallback": False,
                "n_raw": entry.n_raw,
                "n_scores": entry.n_scores,
                "n_blocks": entry.n_blocks,
                "source_n_raw": entry.n_raw,
                "source_n_scores": entry.n_scores,
                "source_n_blocks": entry.n_blocks,
                "group_n_raw": entry.group_n_raw,
                "group_n_scores": entry.group_n_scores,
                "group_n_blocks": entry.group_n_blocks,
                "conformal_rank": entry.conformal_rank,
                "block_reducer": self.block_reducer,
                "block_seed": (
                    self.block_seed if self.block_reducer == "subsample" else None
                ),
                "subsample_order_basis": (
                    self.subsample_order_basis_
                    if self.block_reducer == "subsample"
                    else None
                ),
                "subsample_hash": (
                    "blake2b-64" if self.block_reducer == "subsample" else None
                ),
            }
            for coverage, entry in sorted(self.calibration_.items())
        ]

    def coverage_report(
        self,
        y_true: np.ndarray,
        calibrated_intervals: Mapping[float, np.ndarray],
        predicted_states: np.ndarray | None = None,
        *,
        timestamps: np.ndarray | Sequence | None = None,
        sample_ids: np.ndarray | Sequence | None = None,
        mask: np.ndarray | None = None,
    ) -> dict[float, dict[str, object]]:
        """Return pooled and registered-unit coverage; state input is ignored."""

        del predicted_states
        targets, squeezed = _as_targets(y_true, "y_true")
        n, h = targets.shape
        valid_base = _broadcast_mask(mask, (n, h)) & np.isfinite(targets)
        blocks = _target_day_ids(
            timestamps, n, h, self.step_minutes, self.samples_per_day
        )
        cell_ids = _sample_ids(sample_ids, n, h)
        report: dict[float, dict[str, object]] = {}
        for coverage in self.coverages:
            if coverage not in calibrated_intervals:
                raise ValueError(f"calibrated_intervals is missing coverage {coverage:g}")
            interval = np.asarray(calibrated_intervals[coverage], dtype=float)
            if squeezed and interval.shape == (n, 2):
                interval = interval[:, None, :]
            if interval.shape != (n, h, 2):
                raise ValueError("each calibrated interval must have shape [N, H, 2]")
            valid = valid_base & np.all(np.isfinite(interval), axis=-1)
            if np.any(valid):
                covered = (targets >= interval[..., 0]) & (targets <= interval[..., 1])
                picp = float(np.mean(covered[valid]))
                widths = interval[..., 1] - interval[..., 0]
                width = float(np.mean(widths[valid]))
            else:
                covered = np.zeros((n, h), dtype=bool)
                widths = np.zeros((n, h), dtype=float)
                picp, width = float("nan"), float("nan")
            report[coverage] = {
                "target": coverage,
                "overall": {
                    "count": int(np.sum(valid)),
                    "picp": picp,
                    "mean_width": width,
                    "coverage_error": abs(picp - coverage),
                    "registered_estimand": _registered_coverage_summary(
                        covered,
                        widths,
                        valid,
                        blocks,
                        self.block_reducer,
                        seed=self.block_seed,
                        target=coverage,
                        sample_ids=cell_ids,
                    ),
                },
            }
        return report
