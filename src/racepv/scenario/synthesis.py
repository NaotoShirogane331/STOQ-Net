"""Seeded prediction-level oracle-conditioned simulation (never registered evidence)."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from .cohort import Cohort
from .constants import HORIZON, ModelProfile, model_profile
from .realism import ScenarioGenerationConfig, canonical_config_sha256

SCENARIO_WEATHER_SEED = 20260814


@dataclass(frozen=True)
class SynthesizedPrediction:
    model: str
    seed: int
    quantiles: np.ndarray
    state_labels: np.ndarray
    state_probabilities: np.ndarray
    ramp_scores: np.ndarray
    expert_quantiles: np.ndarray | None = None
    router_logits: np.ndarray | None = None
    gates: np.ndarray | None = None
    shared_weather: np.ndarray | None = None
    generation_config_id: str | None = None
    generation_config_sha256: str | None = None


@dataclass(frozen=True)
class SharedScenarioShocks:
    """Model-independent shocks keyed only by shared cohort identity and seed."""

    weather: np.ndarray
    ar_error: np.ndarray
    heavy_tail: np.ndarray


def _rng(model: str, seed: int) -> np.random.Generator:
    value = int(sha256(f"racepv-synthesis-v1|{model}|{seed}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(value)


def _cohort_token(cohort: Cohort) -> str:
    return "|".join(record.block_id for record in cohort.blocks)


def _temporally_correlated_noise(cohort: Cohort, rng: np.random.Generator) -> np.ndarray:
    n = cohort.targets.shape[0]
    output = np.empty(n, dtype=float)
    output[0] = rng.normal()
    for index in range(1, n):
        contiguous = (
            cohort.site_ids[index, 0] == cohort.site_ids[index - 1, 0]
            and cohort.source_indices[index, 0] == cohort.source_indices[index - 1, 0] + 1
        )
        output[index] = (
            0.80 * output[index - 1] + 0.60 * rng.normal() if contiguous else rng.normal()
        )
    return output


def shared_scenario_shocks(cohort: Cohort, seed: int) -> SharedScenarioShocks:
    """Build temporal/shared weather draws once for every compared model."""
    rng = _rng(f"shared|{_cohort_token(cohort)}", seed)
    n, h = cohort.targets.shape
    weather = _temporally_correlated_noise(cohort, rng)
    lead_weather = rng.normal(size=h)
    weather = weather[:, None] + 0.35 * lead_weather[None, :]
    ar = np.empty((n, h))
    ar[:, 0] = _temporally_correlated_noise(cohort, rng)
    for lead in range(1, h):
        ar[:, lead] = 0.72 * ar[:, lead - 1] + 0.69 * _temporally_correlated_noise(cohort, rng)
    heavy = rng.standard_t(df=4, size=(n, h))
    return SharedScenarioShocks(weather, ar, heavy)


def _softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(centered)
    return values / np.sum(values, axis=-1, keepdims=True)


def _state_targets(cohort: Cohort) -> np.ndarray:
    targets = cohort.targets
    prior = cohort.previous_targets
    if prior is None:
        prior = np.concatenate((targets[:, :1], targets[:, :-1]), axis=1)
    delta = targets - prior
    labels = np.zeros(targets.shape, dtype=int)
    labels[delta >= 0.045] = 1
    labels[delta <= -0.045] = 2
    labels[~cohort.daylight | ~np.isfinite(prior)] = -100
    return labels


def _ordered_quantiles(center: np.ndarray, spread: np.ndarray, *, bias: float = 0.0) -> np.ndarray:
    # Deliberately asymmetric, ordered tails. Sorting after clipping is a safety
    # invariant, not a distributional assumption.
    multipliers = np.array([-1.95, -1.25, -0.62, 0.0, 0.72, 1.42, 2.25])
    values = center[..., None] + bias + spread[..., None] * multipliers
    return np.sort(np.clip(values, 0.0, 1.05), axis=-1)


def _stable_site_scale(site_ids: np.ndarray, modulation: float) -> np.ndarray:
    """Return a deterministic, bounded per-site scale multiplier."""
    values = np.array(
        [
            int(sha256(f"racepv-realism-v2|{value}".encode()).hexdigest()[:16], 16)
            / (16**16 - 1)
            for value in site_ids[:, 0]
        ],
        dtype=float,
    )
    return (1.0 + modulation * (2.0 * values - 1.0))[:, None]


def _seed_scale(seed: int, modulation: float) -> float:
    value = int(sha256(f"racepv-realism-v2|seed|{seed}".encode()).hexdigest()[:16], 16)
    return 1.0 + modulation * (2.0 * value / (16**16 - 1) - 1.0)


def _six_band_lead_scale(horizon: int, modulation: float) -> np.ndarray:
    bands = np.arange(horizon) * 6 // horizon
    centered = (bands - 2.5) / 2.5
    return 1.0 + modulation * centered[None, :]


def _synthesize_predictions_v2(
    cohort: Cohort,
    profile: ModelProfile,
    *,
    seed: int,
    scenario_seed: int,
    generation_config: ScenarioGenerationConfig,
) -> SynthesizedPrediction:
    """Generate a realism-v2 forecast without modifying legacy calculations."""
    model_seed = seed if profile.seed_behavior == "stochastic" else 0
    rng = _rng(profile.name, model_seed)
    n, h = cohort.targets.shape
    shared = shared_scenario_shocks(cohort, scenario_seed)
    model_residual = np.empty((n, h))
    model_residual[:, 0] = _temporally_correlated_noise(cohort, rng)
    for lead_index in range(1, h):
        model_residual[:, lead_index] = (
            0.65 * model_residual[:, lead_index - 1] + 0.76 * rng.normal(size=n)
        )
    site_effect = (
        np.array(
            [
                int(sha256(str(value).encode()).hexdigest()[:8], 16) % 5 - 2
                for value in cohort.site_ids[:, 0]
            ],
            dtype=float,
        )[:, None]
        * 0.006
    )
    lead = np.arange(1, h + 1)[None, :] / HORIZON
    prior = cohort.previous_targets
    if prior is None:
        raise ValueError("scenario cohorts require finite in-block prior targets")
    delta = cohort.targets - prior
    rampiness = np.abs(delta)
    error_scale = generation_config.race_error_scale if profile.is_main else profile.error_scale
    lead_growth = generation_config.race_lead_growth if profile.is_main else profile.lead_growth
    scale = error_scale * (0.55 + 0.85 * rampiness) * (1 + lead_growth * lead)
    if profile.is_main:
        scale *= _stable_site_scale(cohort.site_ids, generation_config.race_site_modulation)
        scale *= _six_band_lead_scale(h, generation_config.race_lead_modulation)
        scale *= _seed_scale(model_seed, generation_config.seed_scale_modulation)
    heavy = shared.heavy_tail * (rampiness > 0.04)
    residual_weight = (
        generation_config.stochastic_residual_weight
        if profile.seed_behavior == "stochastic"
        else 1.0
    )
    center = (
        cohort.targets
        + profile.bias
        + scale
        * (
            0.36 * shared.weather
            + 0.26 * shared.ar_error
            + 0.22 * heavy
            + 0.16 * residual_weight * model_residual
            if profile.seed_behavior == "stochastic"
            else 0.16 * model_residual
        )
        + site_effect
    )
    labels = _state_targets(cohort)
    if profile.is_main and generation_config.ramp_error_multiplier != 1.0:
        ramp = np.isin(labels, (1, 2))
        residual = center - cohort.targets
        center = np.where(
            ramp,
            cohort.targets + generation_config.ramp_error_multiplier * residual,
            center,
        )
    spread = scale * (0.65 + 0.7 * rampiness) * generation_config.spread_scale
    spread *= np.where(rampiness >= 0.045, generation_config.ramp_spread_multiplier, 1.0)
    spread = np.maximum(0.012, spread)
    logits = np.full((n, h, 3), -0.8, dtype=float)
    logits[..., 0] = 1.1 - 6 * rampiness
    logits[..., 1] = 5.5 * np.maximum(delta, 0)
    logits[..., 2] = 5.5 * np.maximum(-delta, 0)
    if profile.seed_behavior == "stochastic":
        logits += 0.12 * shared.weather[..., None] + rng.normal(
            0, generation_config.router_noise_scale, size=logits.shape
        )
    if profile.is_main:
        daylight = labels >= 0
        row_indices, lead_indices = np.nonzero(daylight)
        logits[row_indices, lead_indices, labels[daylight]] += (
            generation_config.main_router_correct_bonus
        )
    probabilities = _softmax(logits)
    config_sha256 = canonical_config_sha256(generation_config)
    if profile.is_main:
        expert_centers = np.stack((center - 0.012, center + 0.020, center - 0.020), axis=-1)
        expert_spreads = np.stack((spread * 0.72, spread * 1.30, spread * 1.35), axis=-1)
        experts = np.stack(
            [
                _ordered_quantiles(
                    expert_centers[..., index],
                    expert_spreads[..., index],
                    bias=(-0.002, 0.006, -0.008)[index],
                )
                for index in range(3)
            ],
            axis=-2,
        )
        gates = probabilities.copy()
        quantiles = np.clip(np.einsum("nhe,nheq->nhq", gates, experts), 0.0, 1.05)
        experts[~cohort.daylight] = 0.0
        gates[~cohort.daylight] = (1.0, 0.0, 0.0)
        probabilities[~cohort.daylight] = (1.0, 0.0, 0.0)
        logits[~cohort.daylight] = (0.0, -1000.0, -1000.0)
        quantiles[~cohort.daylight] = 0.0
        return SynthesizedPrediction(
            profile.name,
            model_seed,
            quantiles,
            labels,
            probabilities,
            probabilities[..., 1:].sum(-1),
            experts,
            logits,
            gates,
            shared.weather,
            generation_config.schema,
            config_sha256,
        )
    quantiles = _ordered_quantiles(center, spread)
    quantiles[~cohort.daylight] = 0.0
    probabilities[~cohort.daylight] = (1.0, 0.0, 0.0)
    return SynthesizedPrediction(
        profile.name,
        model_seed,
        quantiles,
        labels,
        probabilities,
        probabilities[..., 1:].sum(-1),
        shared_weather=shared.weather,
        generation_config_id=generation_config.schema,
        generation_config_sha256=config_sha256,
    )


def synthesize_predictions(
    cohort: Cohort,
    model: str | ModelProfile,
    *,
    seed: int,
    scenario_seed: int = SCENARIO_WEATHER_SEED,
    generation_config: ScenarioGenerationConfig | None = None,
) -> SynthesizedPrediction:
    """Generate correlated synthetic forecasts for a fixed shared cohort.

    The target-conditioned state signal is intentionally explicit: this package
    models an *oracle-conditioned hypothetical* scenario and attaches metadata
    that makes it ineligible as registered evidence.
    """
    profile = model_profile(model)
    if cohort.split not in profile.applies_to:
        raise ValueError(f"model {profile.name!r} does not apply to the {cohort.split!r} split")
    if generation_config is not None:
        return _synthesize_predictions_v2(
            cohort,
            profile,
            seed=seed,
            scenario_seed=scenario_seed,
            generation_config=generation_config,
        )
    # Scenario seed controls common weather across *all* compared models. The
    # profile seed controls only model-specific residuals; deterministic
    # persistence therefore remains byte-identical across protocol seeds.
    model_seed = seed if profile.seed_behavior == "stochastic" else 0
    rng = _rng(profile.name, model_seed)
    n, h = cohort.targets.shape
    shared = shared_scenario_shocks(cohort, scenario_seed)
    model_residual = np.empty((n, h))
    model_residual[:, 0] = _temporally_correlated_noise(cohort, rng)
    for lead in range(1, h):
        model_residual[:, lead] = 0.65 * model_residual[:, lead - 1] + 0.76 * rng.normal(size=n)
    site_effect = (
        np.array(
            [
                int(sha256(str(value).encode()).hexdigest()[:8], 16) % 5 - 2
                for value in cohort.site_ids[:, 0]
            ],
            dtype=float,
        )[:, None]
        * 0.006
    )
    lead = np.arange(1, h + 1)[None, :] / HORIZON
    prior = cohort.previous_targets
    if prior is None:
        raise ValueError("scenario cohorts require finite in-block prior targets")
    delta = cohort.targets - prior
    rampiness = np.abs(delta)
    scale = profile.error_scale * (0.55 + 0.85 * rampiness) * (1 + profile.lead_growth * lead)
    # Shared weather, AR residuals, site effects and heavy ramp tails provide
    # correlated cross-model errors while each profile remains separated.
    heavy = shared.heavy_tail * (rampiness > 0.04)
    center = (
        cohort.targets
        + profile.bias
        + scale
        * (
            0.36 * shared.weather + 0.26 * shared.ar_error + 0.22 * heavy + 0.16 * model_residual
            if profile.seed_behavior == "stochastic"
            else 0.16 * model_residual
        )
        + site_effect
    )
    spread = np.maximum(0.012, scale * (0.65 + 0.7 * rampiness))
    labels = _state_targets(cohort)
    logits = np.full((n, h, 3), -0.8, dtype=float)
    logits[..., 0] = 1.1 - 6 * rampiness
    logits[..., 1] = 5.5 * np.maximum(delta, 0)
    logits[..., 2] = 5.5 * np.maximum(-delta, 0)
    if profile.seed_behavior == "stochastic":
        logits += 0.12 * shared.weather[..., None] + rng.normal(0, 0.20, size=logits.shape)
    probabilities = _softmax(logits)
    if profile.is_main:
        expert_centers = np.stack((center - 0.012, center + 0.020, center - 0.020), axis=-1)
        expert_spreads = np.stack((spread * 0.72, spread * 1.30, spread * 1.35), axis=-1)
        experts = np.stack(
            [
                _ordered_quantiles(
                    expert_centers[..., i], expert_spreads[..., i], bias=(-0.002, 0.006, -0.008)[i]
                )
                for i in range(3)
            ],
            axis=-2,
        )
        gates = probabilities.copy()
        quantiles = np.clip(np.einsum("nhe,nheq->nhq", gates, experts), 0.0, 1.05)
        # Reapply the same night invariant to every forecast representation.
        experts[~cohort.daylight] = 0.0
        gates[~cohort.daylight] = (1.0, 0.0, 0.0)
        probabilities[~cohort.daylight] = (1.0, 0.0, 0.0)
        logits[~cohort.daylight] = (0.0, -1000.0, -1000.0)
        quantiles[~cohort.daylight] = 0.0
        return SynthesizedPrediction(
            profile.name,
            model_seed,
            quantiles,
            labels,
            probabilities,
            probabilities[..., 1:].sum(-1),
            experts,
            logits,
            gates,
            shared.weather,
        )
    quantiles = _ordered_quantiles(center, spread)
    quantiles[~cohort.daylight] = 0.0
    probabilities[~cohort.daylight] = (1.0, 0.0, 0.0)
    return SynthesizedPrediction(
        profile.name,
        model_seed,
        quantiles,
        labels,
        probabilities,
        probabilities[..., 1:].sum(-1),
        shared_weather=shared.weather,
    )
