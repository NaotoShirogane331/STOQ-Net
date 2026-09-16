"""Scenario-level evaluation and leakage guards built on registered metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from racepv.metrics import (
    binary_ramp_metrics,
    forecast_skill,
    point_forecast_metrics,
    quantile_crps,
)
from racepv.metrics.statistics import (
    benjamini_hochberg,
    diebold_mariano_test,
    moving_block_bootstrap,
)

from .cohort import Cohort


@dataclass(frozen=True)
class BootstrapSampleBlock:
    """One auditable contiguous selected site/block day draw."""

    site_id: str
    block_id: str
    day_ids: np.ndarray
    origin_indices: np.ndarray


@dataclass(frozen=True)
class CohortBootstrapResult:
    """Bootstrap statistics plus the exact site/block-safe sampled origins."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    block_length: int
    n_bootstrap: int
    distribution: np.ndarray
    sampled_blocks: tuple[tuple[BootstrapSampleBlock, ...], ...]


def fit_cqr(
    calibrator, y_true: np.ndarray, quantile_predictions: np.ndarray, *, role: str, **kwargs
):
    """Fit only on an explicit source calibration role, never a held-out target."""
    if role != "source_calibration":
        raise ValueError("CQR fitting accepts only the source calibration role")
    return calibrator.fit(y_true, quantile_predictions, **kwargs)


def per_site_per_lead_summary(
    y_true: np.ndarray,
    quantiles: np.ndarray,
    *,
    site_ids: np.ndarray,
    daylight: np.ndarray,
    levels: Sequence[float],
) -> list[dict[str, object]]:
    """Return compact comparable summaries without pooling sites or leads."""
    truth = np.asarray(y_true)
    predictions = np.asarray(quantiles)
    sites = np.asarray(site_ids)
    output = []
    for site in sorted(set(sites[:, 0].tolist())):
        for lead in range(truth.shape[1]):
            site_mask = sites[:, lead] == site
            local_truth = truth[site_mask, lead]
            local_predictions = predictions[site_mask, lead]
            local_daylight = np.asarray(daylight)[site_mask, lead]
            output.append(
                {
                    "site_id": site,
                    "lead": lead + 1,
                    "quantile_crps": quantile_crps(
                        local_truth, local_predictions, levels, mask=local_daylight
                    ),
                    "point": point_forecast_metrics(
                        local_truth,
                        local_predictions[:, list(levels).index(0.5)],
                        daylight_mask=local_daylight,
                    ),
                }
            )
    return output


def expert_diagnostics(
    expert_quantiles: np.ndarray,
    gates: np.ndarray,
    *,
    true_states: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
) -> dict[str, object]:
    median = expert_quantiles[..., expert_quantiles.shape[-1] // 2]
    mean_gates = np.mean(gates, axis=(0, 1))
    entropy = -np.sum(gates * np.log(np.clip(gates, 1e-12, 1.0)), axis=-1)
    diversity = float(np.mean(np.std(median, axis=-1)))
    result: dict[str, object] = {
        "utilization": mean_gates,
        "mean_entropy": float(np.mean(entropy)),
        "diversity": diversity,
        "collapse": bool(np.any(mean_gates < 0.02) or diversity < 1e-5),
    }
    if true_states is not None:
        states = np.asarray(true_states)
        if states.shape != gates.shape[:2]:
            raise ValueError("true_states must have shape [N, H]")
        by_state: dict[str, dict[str, object]] = {}
        if y_true is not None and np.asarray(y_true).shape != states.shape:
            raise ValueError("y_true must have shape [N, H]")
        for name, mask in (
            ("stable", states == 0),
            ("rise", states == 1),
            ("fall", states == 2),
        ):
            if np.any(mask):
                values: dict[str, object] = {
                    "utilization": np.mean(gates[mask], axis=0),
                    "expert_median": np.mean(median[mask], axis=0),
                    "count": int(np.sum(mask)),
                }
                if y_true is not None:
                    truth = np.asarray(y_true, dtype=float)[mask]
                    values["expert_nmae"] = np.mean(np.abs(median[mask] - truth[:, None]), axis=0)
                by_state[name] = values
        result["by_true_state"] = by_state
    return result


def cohort_moving_block_bootstrap(
    data: np.ndarray,
    cohort: Cohort,
    *,
    statistic=None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int | np.random.Generator | None = None,
    block_days: int = 7,
):
    """Bootstrap fixed-size moving multi-day blocks within site/block boundaries."""
    values = np.asarray(data)
    if values.ndim == 0 or values.shape[0] != cohort.targets.shape[0]:
        raise ValueError("data must have the cohort's forecast-origin first dimension")
    if block_days < 1:
        raise ValueError("block_days must be positive")
    if cohort.block_ids is None or cohort.block_ids.shape != (len(values),):
        raise ValueError("cohort must carry one selected block ID for every forecast origin")
    groups: list[tuple[str, str, list[np.ndarray]]] = []
    pairs = zip(cohort.site_ids[:, 0], cohort.block_ids, strict=True)
    for site, block in sorted(set(pairs), key=lambda item: (str(item[0]), str(item[1]))):
        rows = np.flatnonzero((cohort.site_ids[:, 0] == site) & (cohort.block_ids == block))
        days = cohort.day_ids[rows, 0]
        unique_days = np.unique(days)
        groups.append(
            (str(site), str(block), [rows[days == day] for day in unique_days])
        )
    function = statistic or (lambda selected: float(np.mean(selected)))
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    distribution = np.empty(n_bootstrap, dtype=float)
    audit: list[tuple[BootstrapSampleBlock, ...]] = []
    for iteration in range(n_bootstrap):
        sampled: list[np.ndarray] = []
        replica: list[BootstrapSampleBlock] = []
        for site, block, days in groups:
            length = min(block_days, len(days))
            remaining = sum(len(day) for day in days)
            while remaining:
                start = int(rng.integers(0, len(days) - length + 1))
                selected_days = days[start : start + length]
                origin_indices = np.concatenate(selected_days)[:remaining]
                sampled_day_ids = np.unique(cohort.day_ids[origin_indices, 0])
                sampled.append(origin_indices)
                replica.append(
                    BootstrapSampleBlock(
                        site,
                        block,
                        sampled_day_ids,
                        origin_indices,
                    )
                )
                remaining -= len(origin_indices)
        distribution[iteration] = float(function(values[np.concatenate(sampled)]))
        audit.append(tuple(replica))
    tail = (1 - confidence) / 2
    return CohortBootstrapResult(
        estimate=float(function(values)),
        ci_low=float(np.quantile(distribution, tail)),
        ci_high=float(np.quantile(distribution, 1 - tail)),
        confidence=float(confidence),
        block_length=block_days,
        n_bootstrap=n_bootstrap,
        distribution=distribution,
        sampled_blocks=tuple(audit),
    )


__all__ = [
    "benjamini_hochberg",
    "binary_ramp_metrics",
    "BootstrapSampleBlock",
    "CohortBootstrapResult",
    "cohort_moving_block_bootstrap",
    "diebold_mariano_test",
    "expert_diagnostics",
    "fit_cqr",
    "forecast_skill",
    "moving_block_bootstrap",
    "per_site_per_lead_summary",
]
