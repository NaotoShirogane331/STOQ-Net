"""Untrained research controls; importing this module grants no data admission."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn

from racepv.reliability_experiment import make_reliability_components

from .components import OrderedQuantileHead, cyclic_hour_to_hours
from .config import RaCEPVConfig
from .experts import last_observed_power
from .race_pv import RaCEPV
from .reliability_a import ReliabilityGateARaCEPV

M1_REVISION = "racepv.research-single-head.v1"
M3_REVISION = "racepv.research-equal-fusion.v1"


class M2MergedError(ValueError):
    """M2 has no removable up/down direction treatment in the current F."""


def prediction_parameter_count(model: nn.Module) -> int:
    """Count trainable prediction parameters, excluding detached state diagnostics."""
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("state_head.")
    )


def _variant_metadata(model: RaCEPV, revision: str) -> dict[str, object]:
    metadata = RaCEPV.model_metadata.fget(model)
    metadata.update(
        base_architecture_version=metadata["architecture_version"],
        architecture_version=revision,
        model_revision=revision,
        objective_version=model.make_objective().objective_id,
        candidate_only=True,
        reliability_extra_parameters=0,
        reliability_features=[],
        specialist_auxiliary_computation="absent_from_matching_objective",
        prediction_parameters=prediction_parameter_count(model),
        reference_prediction_parameters=model.reference_prediction_parameters,
        prediction_parameter_relative_difference=(
            prediction_parameter_count(model) / model.reference_prediction_parameters - 1.0
        ),
    )
    return metadata


class _EqualRouter(nn.Module):
    """Parameter-free logits: the existing forward computes exact uniform weights."""

    def forward(
        self, history_values: Tensor, history_mask: Tensor, future_known: Tensor, *args, **kwargs
    ) -> Tensor:
        return future_known.new_zeros((*future_known.shape[:2], 3))


class EqualFusionRaCEPV(ReliabilityGateARaCEPV):
    """Train all three F experts end-to-end under a fixed 1/3 mixture."""

    def __init__(self, config: RaCEPVConfig) -> None:
        # Accept this variant's public config on reconstruction, while retaining
        # every other F admission check and its original initialization stream.
        base_config = (
            replace(config, fusion_mode="learned") if config.fusion_mode == "equal" else config
        )
        super().__init__(base_config)
        self.reference_prediction_parameters = prediction_parameter_count(self)
        self.fusion_router = _EqualRouter()
        self.config = replace(config, fusion_mode="equal")

    @property
    def model_metadata(self) -> dict[str, object]:
        metadata = _variant_metadata(self, M3_REVISION)
        metadata.update(
            research_model_id="M3",
            router_mode="parameter_free_equal",
            fusion_gate_semantics="fixed_equal_one_third",
            fusion_weight_semantics="fixed_equal_one_third",
            requires_independent_training=True,
        )
        return metadata


class SingleHeadRaCEPV(ReliabilityGateARaCEPV):
    """Retain common encoders; use one capacity-matched ordered readout.

    Input is [stable horizon, transient shared horizon], each width d_model.
    Location is the stable persistence/clear-sky blend plus harmonic residual.
    The one head has no direction restriction. Width is chosen by parameter
    count alone; smaller width wins an exact tie. No tensors are cached.
    """

    def __init__(self, config: RaCEPVConfig) -> None:
        super().__init__(config)
        self.reference_prediction_parameters = prediction_parameter_count(self)
        del self.stable_expert.quantile_head
        del self.transient_expert.up_hidden
        del self.transient_expert.down_hidden
        del self.transient_expert.up_quantile_head
        del self.transient_expert.down_quantile_head
        del self.fusion_router
        common = prediction_parameter_count(self)
        quantiles = len(config.quantiles)
        # Linear(2d,w) + Linear(w,Q): w*(2d+1+Q)+Q parameters.
        step = 2 * config.d_model + 1 + quantiles
        budget = self.reference_prediction_parameters - common - quantiles
        lower = max(1, budget // step)
        self.readout_width = min(
            (lower, lower + 1),
            key=lambda width: (
                abs(common + width * step + quantiles - self.reference_prediction_parameters),
                width,
            ),
        )
        self.forecast_readout = nn.Sequential(
            nn.Linear(2 * config.d_model, self.readout_width),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.quantile_head = OrderedQuantileHead(
            self.readout_width,
            config.quantiles,
            config.quantile_gap_scale,
        )

    @property
    def model_metadata(self) -> dict[str, object]:
        metadata = _variant_metadata(self, M1_REVISION)
        metadata.update(
            research_model_id="M1",
            active_experts=["single_head"],
            router_mode="absent",
            fusion_mode="single_head",
            fusion_gate_semantics="no_expert_fusion",
            fusion_weight_semantics="single_prediction",
            readout_width=self.readout_width,
            width_selection="nearest_prediction_parameter_count_tie_smaller",
            forecast_head_count=1,
            forecast_anchor="stable_persistence_clear_sky_plus_harmonic_residual",
            forecast_readout_input="concat_stable_horizon_transient_shared_horizon",
            diagnostic_inputs_equal_to_F=False,
            diagnostic_representation="detached_stable_shared_shared_single_prediction_repeated",
        )
        return metadata

    def forward(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor,
        future_weather: Tensor | None = None,
        future_weather_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        self._validate_inputs(history_values, history_mask, future_known, site_metadata)
        self._weather_context(history_values, future_weather, future_weather_mask)
        last_power, available = last_observed_power(
            history_values[..., self.config.power_index],
            history_mask[..., self.config.power_index],
        )
        history, mask = self._impute_history(history_values, history_mask)
        # Validation above enforces finite known-future and static inputs.
        lead = torch.arange(
            1, self.config.horizon + 1, device=history.device, dtype=history.dtype
        ) * (self.config.sampling_minutes / 60)
        lead = lead.unsqueeze(0).expand(history.shape[0], -1)
        clock = cyclic_hour_to_hours(
            future_known[..., self.config.future_hour_sin_index],
            future_known[..., self.config.future_hour_cos_index],
        )
        stable, state_stable, location = self.stable_expert.encode(
            history,
            mask,
            future_known,
            site_metadata,
            clock,
            lead,
            last_power,
            available,
        )
        shared, state_shared = self.transient_expert.encode(
            history,
            mask,
            future_known,
            site_metadata,
            lead,
            clock,
        )
        hidden = self.forecast_readout(torch.cat((stable, shared), dim=-1))
        quantiles = self.quantile_head(hidden, location)
        clear_sky = future_known[..., self.config.future_clear_sky_index]
        quantiles = torch.where(available[:, None, None], quantiles, clear_sky[..., None])
        quantiles = quantiles.clamp(0.0, self.config.power_upper_bound)
        # The flat detached diagnostic retains its 3-slot interface. These are
        # views of ONE prediction, not three heads or a forecast ensemble.
        diagnostic_quantiles = quantiles.unsqueeze(2).expand(-1, -1, 3, -1)
        state_logits = self.state_head(
            history,
            mask,
            future_known,
            site_metadata,
            lead,
            clock,
            state_stable,
            state_shared,
            state_shared,
            diagnostic_quantiles,
            detach_expert_inputs=True,
        )
        night = clear_sky < self.config.daylight_threshold
        presence = torch.logsumexp(state_logits[..., 1:], dim=-1) - state_logits[..., 0]
        direction = state_logits[..., 1:]
        state_logits = torch.where(
            night[..., None],
            state_logits.new_tensor((0.0, -1.0e4, -1.0e4)),
            state_logits,
        )
        quantiles = quantiles.masked_fill(night[..., None], 0.0)
        return {
            "quantiles": quantiles,
            "base_quantiles": quantiles,
            "tail_trigger": torch.zeros_like(clear_sky),
            "tail_correction": torch.zeros_like(clear_sky),
            "state_probs": state_logits.softmax(-1),
            "state_logits": state_logits,
            "router_logits": state_logits,
            "ramp_presence_logits": presence.masked_fill(night, -1.0e4),
            "ramp_direction_logits": direction.masked_fill(night[..., None], 0.0),
            "expert_quantiles": quantiles.unsqueeze(2),
            "gate_weights": torch.ones_like(clear_sky[..., None]),
            "fusion_logits": torch.zeros_like(clear_sky[..., None]),
        }


def build_research_candidate(model_id: str, model_config: RaCEPVConfig):
    """Build new untrained components; no weights or data are read or migrated."""
    if model_id in {"F", "C"}:
        return make_reliability_components(model_config, "A" if model_id == "F" else "control")
    if model_id == "M2":
        raise M2MergedError(
            "M2 merged/unavailable: current F has no up/down direction-specific "
            "structure or specialist supervision to remove; heterogeneous experts "
            "are not equivalent to three homogeneous heads."
        )
    variants = {"M1": SingleHeadRaCEPV, "M3": EqualFusionRaCEPV}
    if model_id not in variants:
        raise ValueError(f"unknown research candidate: {model_id}")
    model = variants[model_id](model_config)
    return model, model.make_objective()
