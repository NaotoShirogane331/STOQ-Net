"""Immutable configuration and candidate catalog for the realism-v2 scenario."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import product

from .constants import RACE_PV, ModelProfile


@dataclass(frozen=True)
class ScenarioGenerationConfig:
    """Bounded, serializable controls for one realism-v2 generation."""

    schema: str = "realism_v2"
    revision: str = "r1"
    race_error_scale: float = 0.0750
    ramp_error_multiplier: float = 1.0
    race_lead_growth: float = 0.95
    stochastic_residual_weight: float = 0.40
    spread_scale: float = 0.48
    ramp_spread_multiplier: float = 1.25
    main_router_correct_bonus: float = 1.40
    router_noise_scale: float = 0.16
    race_site_modulation: float = 0.10
    race_lead_modulation: float = 0.04
    seed_scale_modulation: float = 0.025

    def __post_init__(self) -> None:
        if self.schema != "realism_v2":
            raise ValueError("schema must be 'realism_v2'")
        if self.revision not in {"r1", "r2", "r3", "r4", "r5", "r6", "r6b"}:
            raise ValueError(
                "revision must be 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', or 'r6b'"
            )
        bounds = {
            "race_error_scale": (0.0, 1.0),
            "race_lead_growth": (0.0, 2.0),
            "stochastic_residual_weight": (0.0, 1.0),
            "spread_scale": (0.0, 2.0),
            "ramp_spread_multiplier": (0.0, 4.0),
            "main_router_correct_bonus": (0.0, 4.0),
            "router_noise_scale": (0.0, 1.0),
            "race_site_modulation": (0.0, 1.0),
            "race_lead_modulation": (0.0, 1.0),
            "seed_scale_modulation": (0.0, 1.0),
        }
        for name, (lower, upper) in bounds.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a real number")
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be in [{lower}, {upper}]")
        multiplier = self.ramp_error_multiplier
        if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)):
            raise ValueError("ramp_error_multiplier must be a real number")
        if not math.isfinite(multiplier) or not 0.0 < multiplier <= 1.0:
            raise ValueError("ramp_error_multiplier must be in (0.0, 1.0]")

    @classmethod
    def realism_v2_default(cls) -> ScenarioGenerationConfig:
        return cls()


@dataclass(frozen=True)
class RealismCandidate:
    """One deterministic model-level candidate from the predeclared catalog."""

    candidate_id: str
    config: ScenarioGenerationConfig
    config_sha256: str
    scenario_seed: int
    model_profile: ModelProfile = RACE_PV

    @property
    def generation_config(self) -> ScenarioGenerationConfig:
        """Compatibility name for consumers that call the config a generation config."""

        return self.config

    @property
    def model(self) -> ModelProfile:
        """The model profile evaluated by this model-level candidate."""

        return self.model_profile


def _config_payload(config: ScenarioGenerationConfig) -> dict[str, object]:
    payload = asdict(config)
    if config.revision == "r1" and config.ramp_error_multiplier == 1.0:
        payload.pop("revision")
        payload.pop("ramp_error_multiplier")
    return payload


def canonical_config_sha256(config: ScenarioGenerationConfig) -> str:
    """Hash a configuration using the repository's sorted-key JSON convention."""

    payload = _config_payload(config)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _candidate_catalog_r1() -> tuple[RealismCandidate, ...]:
    """Reconstruct the exact rejected revision-1 catalog."""

    race_error_scales = (0.0735, 0.0750, 0.0765)
    race_lead_growths = (0.90, 1.00)
    residual_weights = (0.32, 0.44)
    spread_scales = (0.42, 0.50)
    router_bonuses = (1.20, 1.55)
    scenario_seeds = (20260814, 20260821, 20260828, 20260904)

    rows: list[RealismCandidate] = []
    for index in range(32):
        race_index = index % 3
        lead_index = index // 3 % 2
        residual_index = index // 6 % 2
        spread_index = index // 12 % 2
        bonus_index = index // 24 % 2
        seed_index = index % 4
        scenario_seed = scenario_seeds[seed_index]
        config = ScenarioGenerationConfig(
            race_error_scale=race_error_scales[race_index],
            race_lead_growth=race_lead_growths[lead_index],
            stochastic_residual_weight=residual_weights[residual_index],
            spread_scale=spread_scales[spread_index],
            main_router_correct_bonus=router_bonuses[bonus_index],
        )
        rows.append(
            RealismCandidate(
                candidate_id=f"rv2-{index:03d}-{scenario_seed}",
                config=config,
                config_sha256=canonical_config_sha256(config),
                scenario_seed=scenario_seed,
            )
        )
    return tuple(rows)


def _candidate_catalog_r2() -> tuple[RealismCandidate, ...]:
    """Return the predeclared 32-row whole-model revision-2 design."""

    design_rows = product(
        (0.0705, 0.0720),
        (0.84, 0.90),
        (0.60, 0.64),
        (0.88, 0.94),
        (0.14, 0.22),
    )
    scenario_seeds = (20260814, 20260821, 20260828, 20260904)
    rows: list[RealismCandidate] = []
    for index, design in enumerate(design_rows):
        (
            race_error_scale,
            ramp_error_multiplier,
            spread_scale,
            ramp_spread_multiplier,
            seed_scale_modulation,
        ) = design
        scenario_seed = scenario_seeds[index % len(scenario_seeds)]
        config = ScenarioGenerationConfig(
            revision="r2",
            race_error_scale=race_error_scale,
            ramp_error_multiplier=ramp_error_multiplier,
            race_lead_growth=0.90,
            stochastic_residual_weight=0.44,
            spread_scale=spread_scale,
            ramp_spread_multiplier=ramp_spread_multiplier,
            main_router_correct_bonus=1.00,
            router_noise_scale=0.16,
            race_site_modulation=0.10,
            race_lead_modulation=0.04,
            seed_scale_modulation=seed_scale_modulation,
        )
        rows.append(
            RealismCandidate(
                candidate_id=f"rv2-r2-{index:03d}-{scenario_seed}",
                config=config,
                config_sha256=canonical_config_sha256(config),
                scenario_seed=scenario_seed,
            )
        )
    return tuple(rows)


def _candidate_catalog_r3() -> tuple[RealismCandidate, ...]:
    """Return the predeclared 32-row whole-model revision-3 design."""

    design_rows = product(
        (0.0728, 0.0731),
        (0.88, 0.92),
        (0.68, 0.72),
        (0.76, 0.82),
        (0.14, 0.22),
    )
    scenario_seeds = (20260814, 20260821, 20260828, 20260904)
    rows: list[RealismCandidate] = []
    for index, design in enumerate(design_rows):
        (
            race_error_scale,
            ramp_error_multiplier,
            spread_scale,
            ramp_spread_multiplier,
            seed_scale_modulation,
        ) = design
        scenario_seed = scenario_seeds[index % len(scenario_seeds)]
        config = ScenarioGenerationConfig(
            revision="r3",
            race_error_scale=race_error_scale,
            ramp_error_multiplier=ramp_error_multiplier,
            race_lead_growth=0.90,
            stochastic_residual_weight=0.44,
            spread_scale=spread_scale,
            ramp_spread_multiplier=ramp_spread_multiplier,
            main_router_correct_bonus=1.00,
            router_noise_scale=0.16,
            race_site_modulation=0.10,
            race_lead_modulation=0.04,
            seed_scale_modulation=seed_scale_modulation,
        )
        rows.append(
            RealismCandidate(
                candidate_id=f"rv2-r3-{index:03d}-{scenario_seed}",
                config=config,
                config_sha256=canonical_config_sha256(config),
                scenario_seed=scenario_seed,
            )
        )
    return tuple(rows)


def _candidate_catalog_r4() -> tuple[RealismCandidate, ...]:
    """Return the predeclared 32-row whole-model revision-4 design."""

    design_rows = product(
        (0.07335, 0.07355),
        (0.86, 0.90),
        (0.68, 0.72),
        (0.72, 0.78),
        (0.14, 0.22),
    )
    scenario_seeds = (20260814, 20260821, 20260828, 20260904)
    rows: list[RealismCandidate] = []
    for index, design in enumerate(design_rows):
        (
            race_error_scale,
            ramp_error_multiplier,
            spread_scale,
            ramp_spread_multiplier,
            seed_scale_modulation,
        ) = design
        scenario_seed = scenario_seeds[index % len(scenario_seeds)]
        config = ScenarioGenerationConfig(
            revision="r4",
            race_error_scale=race_error_scale,
            ramp_error_multiplier=ramp_error_multiplier,
            race_lead_growth=0.90,
            stochastic_residual_weight=0.44,
            spread_scale=spread_scale,
            ramp_spread_multiplier=ramp_spread_multiplier,
            main_router_correct_bonus=1.00,
            router_noise_scale=0.16,
            race_site_modulation=0.10,
            race_lead_modulation=0.04,
            seed_scale_modulation=seed_scale_modulation,
        )
        rows.append(
            RealismCandidate(
                candidate_id=f"rv2-r4-{index:03d}-{scenario_seed}",
                config=config,
                config_sha256=canonical_config_sha256(config),
                scenario_seed=scenario_seed,
            )
        )
    return tuple(rows)


def _candidate_catalog_r5() -> tuple[RealismCandidate, ...]:
    """Return the predeclared 12-row whole-model revision-5 design."""

    designs = product((0.0745, 0.0755, 0.0765), (0.90, 0.94), (0.14, 0.22))
    seeds = (20260814, 20260821, 20260828, 20260904)
    rows: list[RealismCandidate] = []
    for index, (error_scale, ramp_multiplier, seed_modulation) in enumerate(designs):
        config = ScenarioGenerationConfig(
            revision="r5",
            race_error_scale=error_scale,
            ramp_error_multiplier=ramp_multiplier,
            race_lead_growth=0.90,
            stochastic_residual_weight=0.44,
            spread_scale=0.70,
            ramp_spread_multiplier=0.75,
            main_router_correct_bonus=1.00,
            router_noise_scale=0.16,
            race_site_modulation=0.10,
            race_lead_modulation=0.04,
            seed_scale_modulation=seed_modulation,
        )
        scenario_seed = seeds[index % 4]
        rows.append(
            RealismCandidate(
                candidate_id=f"rv2-r5-{index:03d}-{scenario_seed}",
                config=config,
                config_sha256=canonical_config_sha256(config),
                scenario_seed=scenario_seed,
            )
        )
    return tuple(rows)


def _candidate_catalog_r6() -> tuple[RealismCandidate, ...]:
    """Return the fixed six-row router-only revision-6 design."""

    designs = product((1.02, 1.04, 1.06), (0.14, 0.16))
    scenario_seed = 20260821
    rows: list[RealismCandidate] = []
    for index, (router_bonus, router_noise) in enumerate(designs):
        config = ScenarioGenerationConfig(
            revision="r6",
            race_error_scale=0.0765,
            ramp_error_multiplier=0.90,
            race_lead_growth=0.90,
            stochastic_residual_weight=0.44,
            spread_scale=0.70,
            ramp_spread_multiplier=0.75,
            main_router_correct_bonus=router_bonus,
            router_noise_scale=router_noise,
            race_site_modulation=0.10,
            race_lead_modulation=0.04,
            seed_scale_modulation=0.22,
        )
        rows.append(
            RealismCandidate(
                candidate_id=f"rv2-r6-{index:03d}-{scenario_seed}",
                config=config,
                config_sha256=canonical_config_sha256(config),
                scenario_seed=scenario_seed,
            )
        )
    return tuple(rows)


def candidate_catalog(revision: str = "r4") -> tuple[RealismCandidate, ...]:
    """Return one fixed deterministic whole-model candidate catalog revision."""

    if revision == "r1":
        return _candidate_catalog_r1()
    if revision == "r2":
        return _candidate_catalog_r2()
    if revision == "r3":
        return _candidate_catalog_r3()
    if revision == "r4":
        return _candidate_catalog_r4()
    if revision == "r5":
        return _candidate_catalog_r5()
    if revision == "r6":
        return _candidate_catalog_r6()
    raise ValueError("revision must be 'r1', 'r2', 'r3', 'r4', 'r5', or 'r6'")


__all__ = [
    "RealismCandidate",
    "ScenarioGenerationConfig",
    "candidate_catalog",
    "canonical_config_sha256",
]
