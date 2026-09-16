"""Pure numerical calendar-loss summaries and paired block bootstrap."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

_MODEL_FIELDS = {"model_id", "kind", "seed_ids", "loss_sum", "valid_count"}
_NEURAL_SEEDS = (13, 29, 47, 71, 101)
_DETERMINISTIC_SEEDS = ("deterministic",)
_ESTIMAND_ID = "equal_site_equal_seed_cell_weighted_qcrps"
_DIFFERENCE_ESTIMAND_ID = "equal_site_equal_seed_cell_weighted_qcrps_difference"
_BLOCK_DAYS = (1, 3, 7)
_REPLICATES = 10000
_RNG_SEED = 20260906


@dataclass(frozen=True)
class _Model:
    model_id: str
    kind: str
    seed_ids: tuple[int, ...] | tuple[str]
    sums: np.ndarray
    counts: np.ndarray
    original_counts: np.ndarray


def _validate_axes(
    site_ids: object, calendar_days: object
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(site_ids, tuple) or not site_ids:
        raise ValueError("site_ids must be a non-empty tuple")
    if any(not isinstance(item, str) or not item for item in site_ids):
        raise ValueError("site_ids must contain non-empty strings")
    if len(set(site_ids)) != len(site_ids):
        raise ValueError("site_ids must be unique")

    if not isinstance(calendar_days, tuple) or not calendar_days:
        raise ValueError("calendar_days must be a non-empty tuple")
    parsed_days: list[date] = []
    for item in calendar_days:
        if not isinstance(item, str):
            raise ValueError("calendar days must be canonical ISO date strings")
        try:
            parsed = date.fromisoformat(item)
        except (TypeError, ValueError) as exc:
            raise ValueError("calendar days must be canonical ISO date strings") from exc
        if parsed.isoformat() != item:
            raise ValueError("calendar days must be canonical ISO date strings")
        parsed_days.append(parsed)
    if any(
        following.toordinal() - previous.toordinal() != 1
        for previous, following in zip(parsed_days, parsed_days[1:], strict=False)
    ):
        raise ValueError("calendar_days must be strictly consecutive")
    return site_ids, calendar_days


def _validate_models(models: object, *, site_count: int, day_count: int) -> tuple[_Model, ...]:
    if not isinstance(models, (list, tuple)) or not models:
        raise ValueError("models must be a non-empty list or tuple")

    normalized: list[_Model] = []
    model_ids: set[str] = set()
    shared_counts: np.ndarray | None = None
    for record in models:
        if type(record) is not dict or set(record) != _MODEL_FIELDS:
            raise ValueError("each model must be an exact five-field dict")

        model_id = record["model_id"]
        kind = record["kind"]
        seed_ids = record["seed_ids"]
        if not isinstance(model_id, str) or not model_id or model_id in model_ids:
            raise ValueError("model_id must be a unique non-empty string")
        model_ids.add(model_id)
        if not isinstance(kind, str) or kind not in {"neural", "deterministic"}:
            raise ValueError("invalid model kind")
        expected_seeds = _NEURAL_SEEDS if kind == "neural" else _DETERMINISTIC_SEEDS
        if not isinstance(seed_ids, tuple) or seed_ids != expected_seeds:
            raise ValueError("seed_ids do not match model kind")
        if kind == "neural" and any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in seed_ids
        ):
            raise ValueError("neural seed_ids must be integers")

        loss_sum = record["loss_sum"]
        valid_count = record["valid_count"]
        if not isinstance(loss_sum, np.ndarray):
            raise ValueError("loss_sum must be a NumPy array")
        if not isinstance(valid_count, np.ndarray):
            raise ValueError("valid_count must be a NumPy array")
        if np.issubdtype(loss_sum.dtype, np.bool_) or not (
            np.issubdtype(loss_sum.dtype, np.integer) or np.issubdtype(loss_sum.dtype, np.floating)
        ):
            raise ValueError("loss_sum must have a non-bool real numeric dtype")
        if np.issubdtype(valid_count.dtype, np.bool_) or not np.issubdtype(
            valid_count.dtype, np.integer
        ):
            raise ValueError("valid_count must have a non-bool integer dtype")

        expected_shape = (len(seed_ids), site_count, day_count)
        if loss_sum.shape != expected_shape or valid_count.shape != expected_shape:
            raise ValueError("calendar arrays have the wrong shape")
        if not np.isfinite(loss_sum).all() or np.any(loss_sum < 0):
            raise ValueError("loss_sum must be finite and non-negative")
        if np.any(valid_count < 0):
            raise ValueError("valid_count must be non-negative")
        if np.any((valid_count == 0) & (loss_sum != 0)):
            raise ValueError("zero valid_count requires zero loss_sum")

        try:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                sums = loss_sum.astype(np.float64, copy=True)
                counts = valid_count.astype(np.float64, copy=True)
        except (FloatingPointError, OverflowError, RuntimeWarning) as exc:
            raise ValueError("invalid calendar array conversion") from exc
        if not np.isfinite(sums).all() or not np.isfinite(counts).all():
            raise ValueError("calendar array conversion is non-finite")
        if np.any((loss_sum != 0) & (sums == 0)):
            raise ValueError("non-zero loss_sum underflowed during float64 conversion")

        for seed_counts in valid_count:
            if shared_counts is None:
                shared_counts = seed_counts.copy()
            elif not np.array_equal(seed_counts, shared_counts):
                raise ValueError("valid_count must match for every model and seed")

        normalized.append(
            _Model(
                model_id=model_id,
                kind=kind,
                seed_ids=seed_ids,
                sums=sums,
                counts=counts,
                original_counts=valid_count.copy(),
            )
        )
    return tuple(normalized)


def _validate(
    models: object, *, site_ids: object, calendar_days: object
) -> tuple[tuple[_Model, ...], tuple[str, ...], tuple[str, ...]]:
    normalized_sites, normalized_days = _validate_axes(site_ids, calendar_days)
    normalized_models = _validate_models(
        models, site_count=len(normalized_sites), day_count=len(normalized_days)
    )
    return normalized_models, normalized_sites, normalized_days


def _weighted_scores(sums: np.ndarray, counts: np.ndarray, weights: np.ndarray):
    """Return per-seed/site ratio-of-sums, or ``None`` for missing support."""
    try:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            numerator = np.sum(sums * weights[None, None, :], axis=-1)
            denominator = np.sum(counts * weights[None, None, :], axis=-1)
            if not np.isfinite(numerator).all() or not np.isfinite(denominator).all():
                raise ValueError("non-finite calendar aggregate")
            if np.any(denominator <= 0):
                return None
            values = numerator / denominator
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("invalid calendar arithmetic") from exc
    if not np.isfinite(values).all():
        raise ValueError("non-finite calendar score")
    return values


def _integer_site_seed_totals(counts: np.ndarray) -> list[list[int]]:
    return [
        [
            sum(int(value) for value in counts[seed_index, site_index, :])
            for site_index in range(counts.shape[1])
        ]
        for seed_index in range(counts.shape[0])
    ]


def _summarize_validated(
    models: tuple[_Model, ...], site_ids: tuple[str, ...], calendar_days: tuple[str, ...]
) -> dict[str, Any]:
    weights = np.ones(len(calendar_days), dtype=np.float64)
    summaries: list[dict[str, Any]] = []
    for model in models:
        values = _weighted_scores(model.sums, model.counts, weights)
        with np.errstate(over="ignore", invalid="ignore"):
            total_sums = np.sum(model.sums, axis=-1)
        if not np.isfinite(total_sums).all():
            raise ValueError("non-finite calendar aggregate")
        total_counts = _integer_site_seed_totals(model.original_counts)

        site_seed_scores = []
        for site_index, site_id in enumerate(site_ids):
            for seed_index, seed_id in enumerate(model.seed_ids):
                count = total_counts[seed_index][site_index]
                site_seed_scores.append(
                    {
                        "site_id": site_id,
                        "seed_id": seed_id,
                        "loss_sum": float(total_sums[seed_index, site_index]),
                        "valid_count": count,
                        "score": (
                            None
                            if count <= 0
                            else float(total_sums[seed_index, site_index] / float(count))
                        ),
                    }
                )

        if values is None:
            score = None
            reason = "zero_support_site"
            seed_scores = None
            seed_sd = None
            seed_sd_reason = (
                "not_applicable" if model.kind == "deterministic" else "zero_support_site"
            )
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                seed_values = values.mean(axis=1)
                score = float(values.mean(axis=0).mean())
                candidate_seed_sd = (
                    float(np.std(seed_values, ddof=1)) if model.kind == "neural" else None
                )
            if (
                not np.isfinite(seed_values).all()
                or not np.isfinite(score)
                or (candidate_seed_sd is not None and not np.isfinite(candidate_seed_sd))
            ):
                raise ValueError("non-finite calendar summary")
            reason = None
            seed_scores = [float(value) for value in seed_values]
            if model.kind == "neural":
                seed_sd = candidate_seed_sd
                seed_sd_reason = None
            else:
                seed_sd = None
                seed_sd_reason = "not_applicable"

        summaries.append(
            {
                "model_id": model.model_id,
                "kind": model.kind,
                "seed_ids": list(model.seed_ids),
                "site_seed_scores": site_seed_scores,
                "seed_scores": seed_scores,
                "seed_sd": seed_sd,
                "seed_sd_reason": seed_sd_reason,
                "score": score,
                "reason": reason,
            }
        )
    return {
        "kind": "racepv.calendar-loss-summary",
        "version": 1,
        "site_ids": list(site_ids),
        "calendar_days": list(calendar_days),
        "estimand_id": _ESTIMAND_ID,
        "qualification": False,
        "models": summaries,
    }


def summarize_calendar_losses(models, *, site_ids, calendar_days) -> dict:
    """Summarize sufficient calendar statistics using the registered estimand."""
    normalized, sites, days = _validate(models, site_ids=site_ids, calendar_days=calendar_days)
    return _summarize_validated(normalized, sites, days)


def _iter_block_indices(day_count: int, block_days: int):
    rng = np.random.Generator(np.random.PCG64(_RNG_SEED))
    number_of_blocks = (day_count + block_days - 1) // block_days
    offsets = np.arange(block_days, dtype=np.int64)
    for _ in range(_REPLICATES):
        starts = rng.integers(
            0,
            day_count - block_days + 1,
            size=number_of_blocks,
            dtype=np.int64,
        )
        yield (starts[:, None] + offsets[None, :]).reshape(-1)[:day_count]


def _point_comparisons(point_summary: dict[str, Any], reference_model: str) -> list[dict[str, Any]]:
    summaries = point_summary["models"]
    by_id = {item["model_id"]: item for item in summaries}
    all_supported = all(item["score"] is not None for item in summaries)
    comparisons = []
    for item in summaries:
        if item["model_id"] == reference_model:
            continue
        comparisons.append(
            {
                "reference_model": reference_model,
                "baseline_model": item["model_id"],
                "difference": (
                    float(by_id[reference_model]["score"] - item["score"])
                    if all_supported
                    else None
                ),
                "reason": None if all_supported else "zero_support_site",
            }
        )
    return comparisons


def _insufficient_block(
    block_days: int, day_count: int, comparisons: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "block_days": block_days,
        "requested_replicates": _REPLICATES,
        "executed_replicates": 0,
        "rng_seed": _RNG_SEED,
        "bit_generator": "PCG64",
        "numpy_version": np.__version__,
        "index_sha256": None,
        "index_encoding": "replicate-major little-endian int64 C-order",
        "index_shape": [0, day_count],
        "sampling": "non-circular moving blocks, concatenated then trimmed",
        "limitation": "finite-sample non-circular endpoint weighting is unequal",
        "reason": "insufficient_calendar_days",
        "comparisons": [
            {
                "reference_model": item["reference_model"],
                "baseline_model": item["baseline_model"],
                "replicate_differences": [],
                "invalid_replicates": 0,
                "ci95": None,
                "reason": "insufficient_calendar_days",
            }
            for item in comparisons
        ],
    }


def _bootstrap_block(
    models: tuple[_Model, ...],
    reference_index: int,
    point_comparisons: list[dict[str, Any]],
    *,
    day_count: int,
    block_days: int,
) -> dict[str, Any]:
    if day_count < block_days:
        return _insufficient_block(block_days, day_count, point_comparisons)

    digest = hashlib.sha256()
    replicate_differences: list[list[float | None]] = [[] for _ in point_comparisons]
    invalid_replicates = 0
    baseline_indices = [index for index in range(len(models)) if index != reference_index]
    for indices in _iter_block_indices(day_count, block_days):
        digest.update(indices.astype("<i8", copy=False).tobytes(order="C"))
        weights = np.bincount(indices, minlength=day_count)
        values = [_weighted_scores(model.sums, model.counts, weights) for model in models]
        if any(item is None for item in values):
            invalid_replicates += 1
            for differences in replicate_differences:
                differences.append(None)
            continue
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                scores = [float(item.mean(axis=0).mean()) for item in values]
                reference_score = scores[reference_index]
                replicate_values = [
                    float(reference_score - scores[baseline_index])
                    for baseline_index in baseline_indices
                ]
        except (FloatingPointError, OverflowError, RuntimeWarning) as exc:
            raise ValueError("invalid calendar bootstrap reduction") from exc
        if not np.isfinite(scores).all() or not np.isfinite(replicate_values).all():
            raise ValueError("non-finite calendar bootstrap reduction")
        for differences, difference in zip(replicate_differences, replicate_values, strict=True):
            differences.append(difference)

    comparisons = []
    for point, differences in zip(point_comparisons, replicate_differences, strict=True):
        if invalid_replicates:
            ci95 = None
            reason = "zero_support_replicate"
        else:
            quantiles = np.quantile(
                np.asarray(differences, dtype=np.float64),
                [0.025, 0.975],
                method="linear",
            )
            if not np.isfinite(quantiles).all():
                raise ValueError("non-finite calendar interval")
            ci95 = [float(quantiles[0]), float(quantiles[1])]
            reason = None
        comparisons.append(
            {
                "reference_model": point["reference_model"],
                "baseline_model": point["baseline_model"],
                "replicate_differences": differences,
                "invalid_replicates": invalid_replicates,
                "ci95": ci95,
                "reason": reason,
            }
        )
    return {
        "block_days": block_days,
        "requested_replicates": _REPLICATES,
        "executed_replicates": _REPLICATES,
        "rng_seed": _RNG_SEED,
        "bit_generator": "PCG64",
        "numpy_version": np.__version__,
        "index_sha256": digest.hexdigest(),
        "index_encoding": "replicate-major little-endian int64 C-order",
        "index_shape": [_REPLICATES, day_count],
        "sampling": "non-circular moving blocks, concatenated then trimmed",
        "limitation": "finite-sample non-circular endpoint weighting is unequal",
        "reason": None,
        "comparisons": comparisons,
    }


def paired_calendar_block_bootstrap(models, *, site_ids, calendar_days, reference_model) -> dict:
    """Compute registered paired calendar moving-block bootstrap intervals."""
    normalized, sites, days = _validate(models, site_ids=site_ids, calendar_days=calendar_days)
    if len(normalized) < 2:
        raise ValueError("paired bootstrap requires at least two models")
    reference_indices = [
        index for index, model in enumerate(normalized) if model.model_id == reference_model
    ]
    if len(reference_indices) != 1:
        raise ValueError("reference_model must identify exactly one model")
    reference_index = reference_indices[0]

    point_summary = _summarize_validated(normalized, sites, days)
    comparisons = _point_comparisons(point_summary, reference_model)
    blocks = [
        _bootstrap_block(
            normalized,
            reference_index,
            comparisons,
            day_count=len(days),
            block_days=block_days,
        )
        for block_days in _BLOCK_DAYS
    ]
    return {
        "kind": "racepv.paired-calendar-bootstrap",
        "version": 1,
        "site_ids": list(sites),
        "calendar_days": list(days),
        "estimand_id": _DIFFERENCE_ESTIMAND_ID,
        "qualification": False,
        "reference_model": reference_model,
        "point_summary": point_summary,
        "comparisons": comparisons,
        "seed_sd_interpretation": (
            "sample SD across frozen neural fits after equal site weighting; "
            "not a retraining-population interval"
        ),
        "blocks": blocks,
    }
