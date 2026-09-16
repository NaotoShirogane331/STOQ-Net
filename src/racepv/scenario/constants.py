"""Immutable protocol constants for the hypothetical Q2 ideal scenario."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
SEEDS = (13, 29, 47, 71, 101)
HORIZON = 24
STEP_MINUTES = 15
REPORT_LEADS = (1, 2, 4, 8, 16, 24)
SCENARIO_SCHEMA_VERSION = 1
EVIDENCE_METADATA = MappingProxyType(
    {
        "evidence_status": "hypothetical_oracle_conditioned",
        "oracle_conditioned": True,
        "registered_evidence_eligible": False,
        "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
    }
)


@dataclass(frozen=True)
class ModelProfile:
    """A deliberately conservative, auditable simulation profile.

    Realism-v2 generation controls are kept in the separate immutable
    ``ScenarioGenerationConfig`` boundary.
    """

    name: str
    family: str
    seed_behavior: str  # deterministic or stochastic
    applies_to: tuple[str, ...]
    error_scale: float
    bias: float = 0.0
    lead_growth: float = 1.0
    is_main: bool = False
    is_ablation: bool = False
    validation_only: bool = False


_BASELINES = (
    ("smart_persistence", "persistence", "deterministic", 0.115),
    ("clear_sky_persistence", "persistence", "deterministic", 0.100),
    ("quantile_lightgbm", "tree", "stochastic", 0.090),
    ("random_forest_mondrian", "tree", "stochastic", 0.102),
    ("lstm_qr", "neural", "stochastic", 0.088),
    ("tcn_qr", "neural", "stochastic", 0.084),
    ("n_hits", "neural", "stochastic", 0.086),
    ("temporal_fusion_transformer", "neural", "stochastic", 0.081),
    ("xlstm", "neural", "stochastic", 0.082),
    ("patchtst_qr", "neural", "stochastic", 0.079),
    ("s_mamba_qr", "neural", "stochastic", 0.077),
)

BASELINES = tuple(
    ModelProfile(name, family, behavior, ("validation", "test"), scale)
    for name, family, behavior, scale in _BASELINES
)
RACE_PV = ModelProfile(
    "race_pv",
    "mixture_of_experts",
    "stochastic",
    ("validation", "test"),
    0.062,
    lead_growth=0.70,
    is_main=True,
)

# A compact primary ablation set; CQR alternatives are retained as validation-only
# sensitivities because they do not retrain the forecaster.
_ABLATION_NAMES = (
    "a1_no_clear_sky",
    "a2_amplitude_router",
    "a3a_no_router_focal",
    "a3b_label_free_losses",
    "a4_equal_expert_weights",
    "a5_stable_only",
    "a6_transient_only",
    "a7_no_specialist_loss",
    "a9_tcn_backbone",
)
ABLATIONS = tuple(
    ModelProfile(
        name,
        "ablation",
        "stochastic",
        ("validation", "test"),
        0.070 + i * 0.003,
        lead_growth=0.78,
        is_ablation=True,
    )
    for i, name in enumerate(_ABLATION_NAMES)
)
_SENSITIVITIES = (
    "a8a_lead_band_cqr",
    "a8b_global_cqr",
    "ramp_q85",
    "ramp_q95",
    "ramp_fixed_01",
    "coverage_80",
    "coverage_90",
    "bootstrap_7day",
    "bootstrap_21day",
)
VALIDATION_SENSITIVITIES = tuple(
    ModelProfile(name, "sensitivity", "deterministic", ("validation",), 0.062, validation_only=True)
    for name in _SENSITIVITIES
)
MODEL_REGISTRY = MappingProxyType(
    {
        profile.name: profile
        for profile in (*BASELINES, RACE_PV, *ABLATIONS, *VALIDATION_SENSITIVITIES)
    }
)


def model_profile(model: str | ModelProfile) -> ModelProfile:
    if isinstance(model, ModelProfile):
        return model
    try:
        return MODEL_REGISTRY[model]
    except KeyError as exc:
        raise KeyError(f"Unknown scenario model {model!r}") from exc
