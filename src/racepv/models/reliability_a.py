"""Candidate A: explicit history-observation reliability in forecast fusion.

This is a distinct, untrained model revision, not an alias for an old checkpoint.
It reuses the three experts and detached state head unchanged. The new objective
omits specialist computation entirely; see ``make_objective``. No production
training CLI or existing configuration is changed by importing this module.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from racepv.reliability_a_loss import ReliabilityAObjective

from .config import EXPERT_NAMES, RaCEPVConfig
from .experts import ForecastFusionRouter
from .race_pv import RaCEPV

RELIABILITY_FEATURE_NAMES = (
    "power_missing_fraction_30min",
    "power_missing_fraction_60min",
    "power_missing_fraction_180min",
    "last_observed_interval_end_age_fraction",
    "power_observed_any",
)
MODEL_REVISION = "racepv.reliability-gate-a.v1"


def _validated_power_mask(history_mask: Tensor, power_index: int) -> Tensor:
    if not isinstance(history_mask, Tensor):
        raise TypeError("history_mask must be a Tensor")
    if history_mask.ndim != 3 or any(size == 0 for size in history_mask.shape):
        raise ValueError("history_mask must have nonempty shape [B, L, C]")
    if history_mask.shape[1] < 12:
        raise ValueError("candidate A needs at least 180 minutes of history (12 steps)")
    if type(power_index) is not int:
        raise TypeError("power_index must be an integer")
    if not 0 <= power_index < history_mask.shape[2]:
        raise ValueError("power_index is outside the history feature dimension")
    if history_mask.dtype != torch.bool:
        if history_mask.is_complex() or not torch.all((history_mask == 0) | (history_mask == 1)):
            raise ValueError("history_mask must be finite binary observations")
    return history_mask[..., power_index].bool()


def power_observation_reliability(
    history_mask: Tensor,
    *,
    power_index: int,
    sampling_minutes: int,
) -> Tensor:
    """Return five history-only features [B, 5] as float32, with no target input.

    The candidate fixes a regular 15-minute grid. Fractions use exactly the last
    2, 4, and 12 intervals, never a partial window. Age is the number of trailing
    unobserved intervals divided by history length: latest observed => 0; none
    observed => 1. It measures age from the end of an observed interval to issue
    time, not the start of that interval. ``power_observed_any`` disambiguates an
    unavailable observation. The caller must supply the pre-imputation observed
    mask intersected with finite values, not an interpolated-value availability
    mask. This API cannot establish that external provenance on its own.
    """
    if type(sampling_minutes) is not int or sampling_minutes != 15:
        raise ValueError("candidate A fixes sampling_minutes=15")
    observed = _validated_power_mask(history_mask, power_index)
    length = observed.shape[1]
    missing = (~observed).to(torch.float32)
    fractions = [missing[:, -steps:].mean(dim=1) for steps in (2, 4, 12)]
    positions = torch.arange(1, length + 1, device=observed.device)
    last_position = torch.where(observed, positions, 0).amax(dim=1)
    age = (length - last_position).to(torch.float32) / length
    available = observed.any(dim=1).to(torch.float32)
    return torch.stack((*fractions, age, available), dim=-1)


class ReliabilityAwareFusionRouter(nn.Module):
    """Existing horizon-aware logits plus a 15-parameter reliability correction.

    The extra correction is shared across leads; the original router retains all
    its lead-specific context. Zero initialization preserves the original logits
    initially. No new dropout, hidden network, state coupling, or expert is added.
    """

    def __init__(self, base_router: ForecastFusionRouter) -> None:
        super().__init__()
        if not isinstance(base_router, ForecastFusionRouter):
            raise TypeError("base_router must be ForecastFusionRouter")
        self.base_router = base_router
        self.config = base_router.config
        parameter = next(base_router.parameters())
        self.reliability_logits = nn.Linear(
            len(RELIABILITY_FEATURE_NAMES),
            3,
            bias=False,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        nn.init.zeros_(self.reliability_logits.weight)

    def forward(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor,
        lead_time_hours: Tensor,
        physical_time_hours: Tensor,
        stable_hidden: Tensor,
        up_hidden: Tensor,
        down_hidden: Tensor,
        expert_quantiles: Tensor,
        *,
        weather_context: Tensor | None = None,
        detach_expert_inputs: bool | None = None,
    ) -> Tensor:
        reliability = power_observation_reliability(
            history_mask,
            power_index=self.config.power_index,
            sampling_minutes=self.config.sampling_minutes,
        ).to(dtype=history_values.dtype)
        logits = self.base_router(
            history_values,
            history_mask,
            future_known,
            site_metadata,
            lead_time_hours,
            physical_time_hours,
            stable_hidden,
            up_hidden,
            down_hidden,
            expert_quantiles,
            weather_context=weather_context,
            detach_expert_inputs=detach_expert_inputs,
        )
        correction = self.reliability_logits(reliability) / self.config.router_temperature
        return logits + correction.unsqueeze(1)


class ReliabilityGateARaCEPV(RaCEPV):
    """Independent candidate using the original experts and flat detached state.

    ``gru`` is available for synthetic CPU checks; ``mamba`` must be explicitly
    requested for a future qualified Mamba experiment. No automatic fallback is
    allowed here. Construct the matching loss with ``model.make_objective()``.
    Old checkpoints are deliberately incompatible with strict loading because
    the fusion router has a different parameter namespace and extra weights.
    """

    def __init__(self, config: RaCEPVConfig) -> None:
        if (
            config.prediction_state_coupling != "decoupled"
            or config.fusion_mode != "learned"
            or config.router_mode != "contextual"
            or config.state_head_mode != "flat"
            or config.state_representation_mode != "expert_detached"
            or config.active_experts != EXPERT_NAMES
            or config.enable_auxiliary_gate_path
            or config.tail_residual_mode != "off"
            or config.future_weather_features != 0
            or config.forecast_weather_mode != "off"
            or not config.use_clear_sky_prior
            or config.backbone not in {"gru", "mamba"}
            or config.sampling_minutes != 15
            or config.history_length < 12
        ):
            raise ValueError("candidate A requires its isolated decoupled three-expert contract")
        super().__init__(config)
        self.fusion_router = ReliabilityAwareFusionRouter(self.fusion_router)

    def _validate_inputs(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor,
    ) -> None:
        super()._validate_inputs(history_values, history_mask, future_known, site_metadata)
        # Validate before legacy bool conversion; its imputer then intersects the
        # observed mask with isfinite(history), retaining this mask for the gate.
        _validated_power_mask(history_mask, self.config.power_index)

    def make_objective(self) -> ReliabilityAObjective:
        return ReliabilityAObjective(self.config.quantiles)

    @property
    def model_metadata(self) -> dict[str, object]:
        metadata = super().model_metadata
        metadata.update(
            {
                "base_architecture_version": metadata["architecture_version"],
                "architecture_version": MODEL_REVISION,
                "model_revision": MODEL_REVISION,
                "objective_version": ReliabilityAObjective.objective_id,
                "candidate_only": True,
                "reliability_features": list(RELIABILITY_FEATURE_NAMES),
                "reliability_feature_cadence_minutes": 15,
                "reliability_feature_mask": "observed_mask_intersect_finite_before_imputation",
                "reliability_age_reference": "observed_interval_end_to_issue_over_history_duration",
                "reliability_extra_parameters": 15,
                "reliability_correction_initialization": "zero",
                "specialist_auxiliary_computation": "absent_from_matching_objective",
            }
        )
        return metadata
