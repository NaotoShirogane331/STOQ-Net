"""End-to-end Ramp-aware Calibrated Experts model."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .components import MaskedFutureWeatherEncoder, cyclic_hour_to_hours
from .config import (
    EXPERT_NAMES,
    FORECAST_SELECTION_METRIC,
    FUSION_GATE_SEMANTICS,
    OBJECTIVE_VERSION,
    STATE_HEAD_SEMANTICS,
    RaCEPVConfig,
)
from .experts import (
    ForecastFusionRouter,
    IndependentRampStateHead,
    RampStateHead,
    StablePeriodicExpert,
    TransientDirectionalExpert,
    last_observed_power,
)
from .tail_residual import SparseTailResidualHead

STATE_NAMES = EXPERT_NAMES


class RaCEPV(nn.Module):
    """Joint PV quantile and ramp-state forecaster.

    The forward pass consumes only history, an availability mask, known-future
    deterministic features, and static site metadata.  Ground-truth targets and
    future measured weather are intentionally absent from this interface.

    Expert order is always ``stable, ramp_up, ramp_down``.
    """

    def __init__(self, config: RaCEPVConfig) -> None:
        super().__init__()
        self.config = config
        self.stable_expert = StablePeriodicExpert(config)
        self.transient_expert = TransientDirectionalExpert(config)
        self.fusion_router = ForecastFusionRouter(config)
        self.weather_encoder = (
            MaskedFutureWeatherEncoder(
                config.future_weather_features,
                config.future_weather_hidden_size,
            )
            if config.future_weather_features
            else None
        )
        self.state_head = (
            IndependentRampStateHead(config)
            if config.state_representation_mode == "independent_raw"
            else RampStateHead(config)
        )
        self.tail_residual_head = (
            SparseTailResidualHead(config)
            if config.tail_residual_mode == "sparse_ramp_up"
            else None
        )
        self.register_buffer(
            "active_expert_mask",
            torch.tensor(
                [name in config.active_experts for name in EXPERT_NAMES],
                dtype=torch.bool,
            ),
            persistent=False,
        )

    @property
    def backbone_name(self) -> str:
        """Return the actual transient sequence implementation in use."""

        return self.transient_expert.backend_name

    @property
    def model_metadata(self) -> dict[str, object]:
        """Auditable metadata for checkpoints and experiment result tables.

        In particular, a native-Windows debugging run that requested ``auto``
        but cannot load official ``mamba_ssm`` is recorded as
        ``causal_conv_gru`` rather than being mislabeled as Mamba.
        """

        mode = self.config.prediction_state_coupling
        coupling_metadata = {
            "decoupled": {
                "state_gradient_updates_forecast": False,
                "forecast_gradient_updates_state": False,
                "fusion_weight_semantics": FUSION_GATE_SEMANTICS,
            },
            "joint_head": {
                "state_gradient_updates_forecast": True,
                "forecast_gradient_updates_state": False,
                "fusion_weight_semantics": FUSION_GATE_SEMANTICS,
            },
            "state_as_gate": {
                "state_gradient_updates_forecast": False,
                "forecast_gradient_updates_state": True,
                "fusion_weight_semantics": "physical_state_probabilities_as_expert_weights",
            },
        }[mode]
        return {
            "architecture_version": self.config.architecture_version,
            "objective_version": OBJECTIVE_VERSION,
            "forecast_selection_metric": FORECAST_SELECTION_METRIC,
            "state_head_semantics": STATE_HEAD_SEMANTICS,
            "fusion_gate_semantics": FUSION_GATE_SEMANTICS,
            "tail_residual_mode": self.config.tail_residual_mode,
            "tail_residual_cap": self.config.tail_residual_cap,
            "tail_residual_hidden_size": self.config.tail_residual_hidden_size,
            "tail_residual_semantics": (
                "detached_forecast_sparse_signed_quantile_shift"
            ),
            "all_missing_power_fallback": (
                "clear_sky_only" if self.config.use_clear_sky_prior else "zero_no_clear_sky_prior"
            ),
            "requested_backbone": self.config.backbone,
            "actual_backbone": self.backbone_name,
            "uses_official_mamba": self.backbone_name == "mamba_ssm",
            "use_clear_sky_prior": self.config.use_clear_sky_prior,
            "router_mode": self.config.router_mode,
            "fusion_mode": self.config.fusion_mode,
            "state_head_mode": self.config.state_head_mode,
            "state_representation_mode": self.config.state_representation_mode,
            "prediction_state_coupling": mode,
            **coupling_metadata,
            "enable_auxiliary_gate_path": self.config.enable_auxiliary_gate_path,
            "ramp_decision_threshold": self.config.ramp_decision_threshold,
            "forecast_weather_mode": self.config.forecast_weather_mode,
            "forecast_weather_contract_version": (
                self.config.forecast_weather_contract_version
            ),
            "future_weather_features": self.config.future_weather_features,
            "future_weather_hidden_size": self.config.future_weather_hidden_size,
            "weather_direct_expert_access": self.config.forecast_weather_mode
            in {"experts", "experts_fusion", "transient_isolated"},
            "weather_direct_stable_expert_access": self.config.forecast_weather_mode
            in {"experts", "experts_fusion"},
            "weather_direct_transient_expert_access": self.config.forecast_weather_mode
            in {"experts", "experts_fusion", "transient_isolated"},
            "weather_direct_fusion_access": self.config.forecast_weather_mode
            == "experts_fusion",
            "weather_indirect_fusion_access": self.config.forecast_weather_mode
            in {"experts", "experts_fusion"},
            "fusion_representation_semantics": (
                "weather_free_same_run_expert_representations"
                if self.config.forecast_weather_mode == "transient_isolated"
                else "forecast_expert_representations"
            ),
            "state_head_weather_access": False,
            "active_experts": list(self.config.active_experts),
            "clear_sky_history_indices": sorted(
                {
                    self.config.history_clear_sky_index,
                    *self.config.clear_sky_history_feature_indices,
                }
            ),
            "clear_sky_future_indices": sorted(
                {
                    self.config.future_clear_sky_index,
                    *self.config.clear_sky_future_feature_indices,
                }
            ),
        }

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
        weather_context = self._weather_context(
            history_values,
            future_weather,
            future_weather_mask,
        )
        stable_weather_context = weather_context
        transient_weather_context = weather_context
        fusion_weather_context = weather_context
        if weather_context is not None:
            if self.config.forecast_weather_mode == "off":
                stable_weather_context = torch.zeros_like(weather_context)
                transient_weather_context = torch.zeros_like(weather_context)
                fusion_weather_context = torch.zeros_like(weather_context)
            elif self.config.forecast_weather_mode == "experts":
                fusion_weather_context = torch.zeros_like(weather_context)
            elif self.config.forecast_weather_mode == "transient_isolated":
                stable_weather_context = torch.zeros_like(weather_context)
                fusion_weather_context = torch.zeros_like(weather_context)
        observed_last_power, history_power_available = last_observed_power(
            history_values[..., self.config.power_index],
            history_mask[..., self.config.power_index],
        )
        clean_history, clean_mask = self._impute_history(history_values, history_mask)
        clean_future = torch.nan_to_num(future_known, nan=0.0, posinf=0.0, neginf=0.0)
        clean_site = torch.nan_to_num(site_metadata, nan=0.0, posinf=0.0, neginf=0.0)
        clear_sky_for_night = clean_future[..., self.config.future_clear_sky_index]
        if not self.config.use_clear_sky_prior:
            clean_history, clean_mask, clean_future = self._remove_clear_sky_prior(
                clean_history,
                clean_mask,
                clean_future,
            )

        batch_size = history_values.shape[0]
        lead_time_hours = (
            (
                torch.arange(
                    1,
                    self.config.horizon + 1,
                    device=history_values.device,
                    dtype=history_values.dtype,
                )
                * (self.config.sampling_minutes / 60.0)
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        if (
            self.config.future_hour_sin_index is None or self.config.future_hour_cos_index is None
        ):  # pragma: no cover - rejected by configuration validation
            raise RuntimeError("v2 hour feature indices were not initialized")
        physical_time_hours = cyclic_hour_to_hours(
            clean_future[..., self.config.future_hour_sin_index],
            clean_future[..., self.config.future_hour_cos_index],
        )

        (
            stable_quantiles,
            stable_hidden,
            state_stable_quantiles,
            state_stable_hidden,
        ) = self.stable_expert(
            clean_history,
            clean_mask,
            clean_future,
            clean_site,
            physical_time_hours,
            lead_time_hours,
            observed_last_power,
            history_power_available,
            stable_weather_context,
        )
        (
            up_quantiles,
            down_quantiles,
            up_hidden,
            down_hidden,
            state_up_quantiles,
            state_down_quantiles,
            state_up_hidden,
            state_down_hidden,
        ) = self.transient_expert(
            clean_history,
            clean_mask,
            clean_future,
            clean_site,
            lead_time_hours,
            physical_time_hours,
            observed_last_power,
            history_power_available,
            transient_weather_context,
        )
        expert_quantiles = torch.stack(
            (stable_quantiles, up_quantiles, down_quantiles),
            dim=2,
        )
        state_expert_quantiles = torch.stack(
            (state_stable_quantiles, state_up_quantiles, state_down_quantiles),
            dim=2,
        )
        if self.config.use_clear_sky_prior:
            clear_sky_fallback = (
                clean_future[..., self.config.future_clear_sky_index].unsqueeze(-1).unsqueeze(-1)
            )
            expert_quantiles = torch.where(
                history_power_available.view(-1, 1, 1, 1),
                expert_quantiles,
                clear_sky_fallback.expand_as(expert_quantiles),
            )
            state_expert_quantiles = torch.where(
                history_power_available.view(-1, 1, 1, 1),
                state_expert_quantiles,
                clear_sky_fallback.expand_as(state_expert_quantiles),
            )
        expert_quantiles = expert_quantiles.clamp(0.0, self.config.power_upper_bound)
        state_expert_quantiles = state_expert_quantiles.clamp(
            0.0,
            self.config.power_upper_bound,
        )

        active = self.active_expert_mask.to(dtype=stable_hidden.dtype)
        isolated_fusion = self.config.forecast_weather_mode == "transient_isolated"
        fusion_stable_hidden = state_stable_hidden if isolated_fusion else stable_hidden
        fusion_up_hidden = state_up_hidden if isolated_fusion else up_hidden
        fusion_down_hidden = state_down_hidden if isolated_fusion else down_hidden
        fusion_expert_quantiles = (
            state_expert_quantiles if isolated_fusion else expert_quantiles
        )
        fusion_logits = self.fusion_router(
            clean_history,
            clean_mask,
            clean_future,
            clean_site,
            lead_time_hours,
            physical_time_hours,
            fusion_stable_hidden * active[0],
            fusion_up_hidden * active[1],
            fusion_down_hidden * active[2],
            fusion_expert_quantiles * active.view(1, 1, 3, 1),
            weather_context=fusion_weather_context,
        )

        if self.config.state_representation_mode == "independent_raw":
            raw_state_outputs = self.state_head(
                clean_history,
                clean_mask,
                clean_future,
                clean_site,
                lead_time_hours,
                physical_time_hours,
            )
        else:
            detach_state_inputs = self.config.prediction_state_coupling != "joint_head"
            raw_state_outputs = self.state_head(
                clean_history,
                clean_mask,
                clean_future,
                clean_site,
                lead_time_hours,
                physical_time_hours,
                state_stable_hidden,
                state_up_hidden,
                state_down_hidden,
                state_expert_quantiles,
                detach_expert_inputs=detach_state_inputs,
            )

        auxiliary_fusion_logits: Tensor | None = None
        if self.config.enable_auxiliary_gate_path:
            # The detached-input auxiliary needs its own graph so its gradient
            # reaches router parameters without reaching experts.  Forking the
            # RNG restores the main training trajectory after router dropout.
            cuda_devices = []
            if history_values.is_cuda:
                device_index = history_values.device.index
                cuda_devices = [
                    torch.cuda.current_device() if device_index is None else device_index
                ]
            with torch.random.fork_rng(devices=cuda_devices):
                auxiliary_fusion_logits = self.fusion_router(
                    clean_history,
                    clean_mask,
                    clean_future,
                    clean_site,
                    lead_time_hours,
                    physical_time_hours,
                    fusion_stable_hidden,
                    fusion_up_hidden,
                    fusion_down_hidden,
                    fusion_expert_quantiles,
                    weather_context=fusion_weather_context,
                    detach_expert_inputs=True,
                )

        night = clear_sky_for_night < self.config.daylight_threshold
        if self.config.state_head_mode == "hierarchical":
            ramp_presence_logits = raw_state_outputs[..., 0]
            ramp_direction_logits = raw_state_outputs[..., 1:]
            ramp_probability = torch.sigmoid(ramp_presence_logits)
            direction_probability = torch.softmax(ramp_direction_logits, dim=-1)
            state_probs = torch.cat(
                (
                    (1.0 - ramp_probability).unsqueeze(-1),
                    ramp_probability.unsqueeze(-1) * direction_probability,
                ),
                dim=-1,
            )
            state_logits = state_probs.clamp_min(torch.finfo(state_probs.dtype).tiny).log()
        else:
            state_logits = raw_state_outputs
            state_probs = torch.softmax(state_logits, dim=-1)
            ramp_presence_logits = (
                torch.logsumexp(state_logits[..., 1:], dim=-1) - state_logits[..., 0]
            )
            ramp_direction_logits = state_logits[..., 1:]

        night_logits = state_logits.new_tensor((0.0, -1.0e4, -1.0e4))
        state_logits = torch.where(night.unsqueeze(-1), night_logits, state_logits)
        state_probs = torch.softmax(state_logits, dim=-1)
        ramp_presence_logits = torch.where(
            night,
            ramp_presence_logits.new_full((), -1.0e4),
            ramp_presence_logits,
        )
        ramp_direction_logits = torch.where(
            night.unsqueeze(-1),
            torch.zeros_like(ramp_direction_logits),
            ramp_direction_logits,
        )

        active_gate = self.active_expert_mask.view(1, 1, 3)
        inactive_logit = torch.finfo(fusion_logits.dtype).min
        if self.config.prediction_state_coupling == "state_as_gate":
            masked_state = torch.where(active_gate, state_probs, torch.zeros_like(state_probs))
            gate_weights = masked_state / masked_state.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(torch.finfo(masked_state.dtype).tiny)
            fusion_logits = torch.where(
                active_gate,
                gate_weights.clamp_min(torch.finfo(gate_weights.dtype).tiny).log(),
                gate_weights.new_full((), inactive_logit),
            )
        else:
            if self.config.fusion_mode == "equal":
                fusion_logits = torch.zeros_like(fusion_logits)
            fusion_logits = torch.where(
                active_gate,
                fusion_logits,
                fusion_logits.new_full((), inactive_logit),
            )
            gate_weights = torch.softmax(fusion_logits, dim=-1)
        auxiliary_gate_weights: Tensor | None = None
        if auxiliary_fusion_logits is not None:
            if self.config.fusion_mode == "equal":
                auxiliary_fusion_logits = torch.zeros_like(auxiliary_fusion_logits)
            auxiliary_fusion_logits = torch.where(
                active_gate,
                auxiliary_fusion_logits,
                auxiliary_fusion_logits.new_full((), inactive_logit),
            )
            auxiliary_gate_weights = torch.softmax(auxiliary_fusion_logits, dim=-1)

        # Enforce the physical night boundary before fusion.  Every expert is
        # zero at night; gate masking still respects the configured active set.
        expert_quantiles = expert_quantiles.masked_fill(
            night.unsqueeze(-1).unsqueeze(-1),
            0.0,
        )
        base_quantiles = (gate_weights.unsqueeze(-1) * expert_quantiles).sum(dim=2)
        base_quantiles = base_quantiles.clamp(0.0, self.config.power_upper_bound)
        if self.tail_residual_head is None:
            tail_trigger = base_quantiles.new_zeros(base_quantiles.shape[:2])
            tail_correction = tail_trigger.clone()
        else:
            normalized_lead = lead_time_hours / (
                self.config.horizon * self.config.sampling_minutes / 60.0
            )
            tail_trigger, tail_correction = self.tail_residual_head(
                stable_hidden=fusion_stable_hidden,
                up_hidden=fusion_up_hidden,
                down_hidden=fusion_down_hidden,
                expert_quantiles=fusion_expert_quantiles,
                gate_weights=gate_weights,
                future_known=clean_future,
                site_metadata=clean_site,
                normalized_lead=normalized_lead,
                clock_sin=clean_future[..., self.config.future_hour_sin_index],
                clock_cos=clean_future[..., self.config.future_hour_cos_index],
                median_index=self.config.median_index,
            )
        tail_trigger = tail_trigger.masked_fill(night, 0.0)
        tail_correction = tail_correction.masked_fill(night, 0.0)
        quantiles = (base_quantiles + tail_correction.unsqueeze(-1)).clamp(
            0.0,
            self.config.power_upper_bound,
        )

        outputs = {
            "quantiles": quantiles,
            "base_quantiles": base_quantiles,
            "tail_trigger": tail_trigger,
            "tail_correction": tail_correction,
            "state_probs": state_probs,
            "expert_quantiles": expert_quantiles,
            "gate_weights": gate_weights,
            "fusion_logits": fusion_logits,
            "state_logits": state_logits,
            "router_logits": state_logits,
            "ramp_presence_logits": ramp_presence_logits,
            "ramp_direction_logits": ramp_direction_logits,
        }
        if auxiliary_gate_weights is not None:
            outputs["_auxiliary_gate_weights"] = auxiliary_gate_weights
        return outputs

    def _weather_context(
        self,
        history_values: Tensor,
        future_weather: Tensor | None,
        future_weather_mask: Tensor | None,
    ) -> Tensor | None:
        if self.weather_encoder is None:
            if future_weather is not None or future_weather_mask is not None:
                raise ValueError("legacy/off model has no registered future weather features")
            return None
        if future_weather is None or future_weather_mask is None:
            raise ValueError(
                "weather-configured model requires future_weather and future_weather_mask"
            )
        expected = (
            history_values.shape[0],
            self.config.horizon,
            self.config.future_weather_features,
        )
        if tuple(future_weather.shape) != expected:
            raise ValueError(
                f"future_weather has shape {tuple(future_weather.shape)}, expected {expected}"
            )
        if tuple(future_weather_mask.shape) != expected:
            raise ValueError(
                "future_weather_mask has shape "
                f"{tuple(future_weather_mask.shape)}, expected {expected}"
            )
        if (
            future_weather.device != history_values.device
            or future_weather_mask.device != history_values.device
        ):
            raise ValueError("future weather tensors must share the model input device")
        if not future_weather.is_floating_point() or not future_weather_mask.is_floating_point():
            raise TypeError("future weather tensors must be floating point")
        available = future_weather_mask.bool()
        if not torch.isfinite(future_weather[available]).all():
            raise ValueError("available future_weather values must be finite")
        return self.weather_encoder(future_weather, future_weather_mask)

    def _remove_clear_sky_prior(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Hide all declared clear-sky inputs while retaining night physics."""

        history_indices = sorted(
            {
                self.config.history_clear_sky_index,
                *self.config.clear_sky_history_feature_indices,
            }
        )
        future_indices = sorted(
            {
                self.config.future_clear_sky_index,
                *self.config.clear_sky_future_feature_indices,
            }
        )
        ablated_history = history_values.clone()
        ablated_mask = history_mask.clone()
        ablated_future = future_known.clone()
        ablated_history[..., history_indices] = 0.0
        ablated_mask[..., history_indices] = False
        ablated_future[..., future_indices] = 0.0
        return ablated_history, ablated_mask, ablated_future

    def _validate_inputs(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor,
    ) -> None:
        if history_values.ndim != 3:
            raise ValueError("history_values must have shape [B, L, C]")
        expected_history = (
            history_values.shape[0],
            self.config.history_length,
            self.config.history_features,
        )
        if tuple(history_values.shape) != expected_history:
            raise ValueError(
                f"history_values has shape {tuple(history_values.shape)}, "
                f"expected {expected_history}"
            )
        if tuple(history_mask.shape) != expected_history:
            raise ValueError(
                f"history_mask has shape {tuple(history_mask.shape)}, expected {expected_history}"
            )
        expected_future = (
            history_values.shape[0],
            self.config.horizon,
            self.config.future_features,
        )
        if tuple(future_known.shape) != expected_future:
            raise ValueError(
                f"future_known has shape {tuple(future_known.shape)}, expected {expected_future}"
            )
        expected_site = (history_values.shape[0], self.config.site_features)
        if tuple(site_metadata.shape) != expected_site:
            raise ValueError(
                f"site_metadata has shape {tuple(site_metadata.shape)}, expected {expected_site}"
            )
        if history_mask.device != history_values.device:
            raise ValueError("history_values and history_mask must be on the same device")
        if (
            future_known.device != history_values.device
            or site_metadata.device != history_values.device
        ):
            raise ValueError("all model inputs must be on the same device")
        if not history_values.is_floating_point():
            raise TypeError("history_values must be floating point")
        if not future_known.is_floating_point() or not site_metadata.is_floating_point():
            raise TypeError("future_known and site_metadata must be floating point")
        if not torch.isfinite(future_known).all():
            raise ValueError(
                "future_known must be finite; deterministic future features cannot be missing"
            )
        if not torch.isfinite(site_metadata).all():
            raise ValueError(
                "site_metadata must be finite; impute static metadata before the model"
            )

    @staticmethod
    def _impute_history(history_values: Tensor, history_mask: Tensor) -> tuple[Tensor, Tensor]:
        finite = torch.isfinite(history_values)
        valid = history_mask.bool() & finite
        finite_values = torch.nan_to_num(history_values, nan=0.0, posinf=0.0, neginf=0.0)
        weights = valid.to(history_values.dtype)
        channel_mean = (finite_values * weights).sum(dim=1, keepdim=True) / weights.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)
        imputed = torch.where(valid, finite_values, channel_mean)
        return imputed, valid
