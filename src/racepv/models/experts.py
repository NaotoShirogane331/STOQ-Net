"""Stable, transient, and routing experts for RaCE-PV."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .components import (
    CausalConvEncoder,
    CausalSequenceBackbone,
    OrderedQuantileHead,
    causal_rolling_std,
    make_harmonic_basis,
    normalized_spectral_entropy,
)
from .config import RaCEPVConfig


def _finite(values: Tensor) -> Tensor:
    return torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def last_observed_power(power: Tensor, observed_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Return each row's most recent finite, explicitly observed power value."""

    if power.ndim != 2 or observed_mask.shape != power.shape:
        raise ValueError("power and observed_mask must have matching shape [B, L]")
    valid = observed_mask.bool() & torch.isfinite(power)
    positions = torch.arange(power.shape[1], device=power.device).unsqueeze(0)
    last_positions = torch.where(valid, positions, positions.new_full((), -1)).amax(dim=1)
    available = last_positions >= 0
    safe_positions = last_positions.clamp_min(0)
    finite_power = _finite(power)
    selected = finite_power.gather(1, safe_positions.unsqueeze(1)).squeeze(1)
    return torch.where(available, selected, torch.zeros_like(selected)), available


def persistence_clear_sky_anchor(
    last_power: Tensor,
    history_available: Tensor,
    future_clear_sky: Tensor,
    lead_time_hours: Tensor,
    tau_hours: Tensor,
) -> Tensor:
    """Blend short-term persistence into clear-sky power with exponential decay."""

    if last_power.ndim != 1 or history_available.shape != last_power.shape:
        raise ValueError("last_power and history_available must have shape [B]")
    if future_clear_sky.shape != lead_time_hours.shape:
        raise ValueError("future_clear_sky and lead_time_hours must have shape [B, H]")
    if future_clear_sky.shape[0] != last_power.shape[0]:
        raise ValueError("anchor inputs must share their batch dimension")
    persistence_weight = torch.exp(-lead_time_hours / tau_hours.clamp_min(1e-6))
    blended = (
        persistence_weight * last_power.unsqueeze(1) + (1.0 - persistence_weight) * future_clear_sky
    )
    return torch.where(history_available.unsqueeze(1), blended, future_clear_sky)


class StablePeriodicExpert(nn.Module):
    """Clear-sky-anchored periodic expert with physical-time harmonics."""

    def __init__(self, config: RaCEPVConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = CausalConvEncoder(
            input_size=4,
            hidden_size=config.d_model,
            kernel_size=config.stable_conv_kernel,
            dropout=config.dropout,
        )
        self.num_basis = 2 + 2 * len(config.harmonic_periods_hours)
        self.coefficient_head = nn.Linear(config.d_model, self.num_basis)
        horizon_input_size = (
            config.d_model + self.num_basis + config.future_features + config.site_features
        )
        self.horizon_projection = nn.Sequential(
            nn.Linear(horizon_input_size, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.d_model),
        )
        self.quantile_head = OrderedQuantileHead(
            config.d_model,
            config.quantiles,
            config.quantile_gap_scale,
        )
        self.weather_projection = (
            nn.Sequential(
                nn.Linear(
                    config.future_weather_hidden_size,
                    config.d_model,
                    bias=False,
                ),
                nn.GELU(),
            )
            if config.future_weather_features
            else None
        )
        tau_fraction = (config.persistence_tau_initial_hours - config.persistence_tau_min_hours) / (
            config.persistence_tau_max_hours - config.persistence_tau_min_hours
        )
        initial_logit = math.log(tau_fraction / (1.0 - tau_fraction))
        self.persistence_tau_unconstrained = nn.Parameter(torch.tensor(initial_logit))

    @property
    def persistence_tau_hours(self) -> Tensor:
        """Learnable persistence decay constrained to the configured safe range."""

        width = self.config.persistence_tau_max_hours - self.config.persistence_tau_min_hours
        return self.config.persistence_tau_min_hours + width * torch.sigmoid(
            self.persistence_tau_unconstrained
        )

    def forward(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor,
        physical_time_hours: Tensor,
        lead_time_hours: Tensor,
        last_power: Tensor,
        history_power_available: Tensor,
        weather_context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        forecast_hidden, state_hidden, physical_location = self.encode(
            history_values,
            history_mask,
            future_known,
            site_metadata,
            physical_time_hours,
            lead_time_hours,
            last_power,
            history_power_available,
            weather_context,
        )
        forecast_quantiles = self.quantile_head(forecast_hidden, physical_location)
        state_quantiles = self.quantile_head(state_hidden, physical_location)
        return forecast_quantiles, forecast_hidden, state_quantiles, state_hidden

    def encode(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor,
        physical_time_hours: Tensor,
        lead_time_hours: Tensor,
        last_power: Tensor,
        history_power_available: Tensor,
        weather_context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return horizon representations and the original periodic physical location."""
        power = history_values[..., self.config.power_index : self.config.power_index + 1]
        if self.config.use_clear_sky_prior:
            clear_sky = history_values[
                ...,
                self.config.history_clear_sky_index : self.config.history_clear_sky_index + 1,
            ]
            valid = (
                history_mask[..., self.config.power_index].bool()
                & history_mask[..., self.config.history_clear_sky_index].bool()
                & (clear_sky.squeeze(-1) >= self.config.daylight_threshold)
            )
            stable_signal = torch.where(
                clear_sky >= self.config.daylight_threshold,
                power / clear_sky.clamp_min(1e-4),
                torch.zeros_like(power),
            ).clamp(0.0, 2.0)
            residual = power - clear_sky
        else:
            # The ablation remains a meaningful periodic power expert: it uses
            # no historical/future clear-sky value, and anchors its residual to
            # persistence instead of silently degenerating to an all-zero
            # context.
            valid = history_mask[..., self.config.power_index].bool()
            stable_signal = power
            residual = power - power.mean(dim=1, keepdim=True)
        first_difference = F.pad(
            stable_signal[:, 1:] - stable_signal[:, :-1],
            (0, 0, 1, 0),
        )
        encoder_inputs = torch.cat(
            (
                stable_signal,
                residual,
                first_difference,
                valid.to(power.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        _, context = self.encoder(_finite(encoder_inputs), valid)

        basis = make_harmonic_basis(
            physical_time_hours,
            lead_time_hours,
            self.config.harmonic_periods_hours,
        )
        coefficients = (
            torch.tanh(self.coefficient_head(context)) * self.config.stable_residual_scale
        )
        periodic_residual = torch.einsum("bhd,bd->bh", basis, coefficients)
        if self.config.use_clear_sky_prior:
            physical_anchor = persistence_clear_sky_anchor(
                last_power,
                history_power_available,
                future_known[..., self.config.future_clear_sky_index],
                lead_time_hours,
                self.persistence_tau_hours,
            )
        else:
            physical_anchor = torch.where(
                history_power_available.unsqueeze(1),
                last_power.unsqueeze(1).expand_as(periodic_residual),
                torch.zeros_like(periodic_residual),
            )
        physical_location = physical_anchor + periodic_residual

        repeated_context = context.unsqueeze(1).expand(-1, future_known.shape[1], -1)
        pieces = [repeated_context, basis, future_known]
        if self.config.site_features:
            pieces.append(site_metadata.unsqueeze(1).expand(-1, future_known.shape[1], -1))
        state_hidden = self.horizon_projection(torch.cat(pieces, dim=-1))
        forecast_hidden = state_hidden
        if self.weather_projection is not None:
            if weather_context is None:
                raise ValueError("weather-enabled stable expert requires weather_context")
            forecast_hidden = state_hidden + self.weather_projection(weather_context)
        return forecast_hidden, state_hidden, physical_location


class TransientDirectionalExpert(nn.Module):
    """Shared causal sequence backbone with ramp-up and ramp-down heads."""

    def __init__(self, config: RaCEPVConfig) -> None:
        super().__init__()
        self.config = config
        # Original values + availability flags + clear-sky residual, two
        # derivatives, and one-hour trailing volatility.
        input_size = 2 * config.history_features + 4
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.backbone = CausalSequenceBackbone(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            num_layers=config.num_backbone_layers,
            dropout=config.dropout,
            backend=config.backbone,
            expand=config.mamba_expand,
        )
        query_input_size = config.future_features + config.site_features + 3
        self.query_projection = nn.Sequential(
            nn.Linear(query_input_size, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.shared_horizon = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.d_model),
        )
        self.up_hidden = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.down_hidden = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.up_quantile_head = OrderedQuantileHead(
            config.d_model,
            config.quantiles,
            config.quantile_gap_scale,
        )
        self.down_quantile_head = OrderedQuantileHead(
            config.d_model,
            config.quantiles,
            config.quantile_gap_scale,
        )
        self.weather_projection = (
            nn.Sequential(
                nn.Linear(
                    config.future_weather_hidden_size,
                    config.d_model,
                    bias=False,
                ),
                nn.GELU(),
            )
            if config.future_weather_features
            else None
        )

    @property
    def backend_name(self) -> str:
        """Actual sequence implementation, never an aspirational model name."""

        return self.backbone.backend_name

    def forward(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor,
        lead_time_hours: Tensor,
        physical_time_hours: Tensor,
        last_power: Tensor,
        history_power_available: Tensor,
        weather_context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        forecast_shared, state_shared = self.encode(
            history_values,
            history_mask,
            future_known,
            site_metadata,
            lead_time_hours,
            physical_time_hours,
            weather_context,
        )
        state_up_hidden = self.up_hidden(state_shared)
        state_down_hidden = self.down_hidden(state_shared)
        forecast_up_hidden = self.up_hidden(forecast_shared)
        forecast_down_hidden = self.down_hidden(forecast_shared)

        # Direction was supplied by specialist supervision, never a hard sign rule.
        fallback = (
            future_known[..., self.config.future_clear_sky_index]
            if self.config.use_clear_sky_prior
            else torch.zeros_like(lead_time_hours)
        )
        physical_anchor = torch.where(
            history_power_available.unsqueeze(1),
            last_power.unsqueeze(1).expand(-1, future_known.shape[1]),
            fallback,
        )
        forecast_up_quantiles = self.up_quantile_head(forecast_up_hidden, physical_anchor)
        forecast_down_quantiles = self.down_quantile_head(forecast_down_hidden, physical_anchor)
        state_up_quantiles = self.up_quantile_head(state_up_hidden, physical_anchor)
        state_down_quantiles = self.down_quantile_head(state_down_hidden, physical_anchor)
        return (
            forecast_up_quantiles,
            forecast_down_quantiles,
            forecast_up_hidden,
            forecast_down_hidden,
            state_up_quantiles,
            state_down_quantiles,
            state_up_hidden,
            state_down_hidden,
        )

    def encode(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor,
        lead_time_hours: Tensor,
        physical_time_hours: Tensor,
        weather_context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return the shared horizon representation before the independent heads."""
        power = history_values[..., self.config.power_index : self.config.power_index + 1]
        if self.config.use_clear_sky_prior:
            clear_sky = history_values[
                ...,
                self.config.history_clear_sky_index : self.config.history_clear_sky_index + 1,
            ]
            residual = power - clear_sky
        else:
            residual = power - power.mean(dim=1, keepdim=True)
        first_difference = F.pad(power[:, 1:] - power[:, :-1], (0, 0, 1, 0))
        second_difference = F.pad(
            first_difference[:, 1:] - first_difference[:, :-1],
            (0, 0, 1, 0),
        )
        one_hour_steps = max(1, round(60 / self.config.sampling_minutes))
        volatility = causal_rolling_std(power, one_hour_steps)
        sequence_inputs = torch.cat(
            (
                history_values,
                history_mask.to(history_values.dtype),
                residual,
                first_difference,
                second_difference,
                volatility,
            ),
            dim=-1,
        )
        sequence_hidden = self.backbone(self.input_projection(_finite(sequence_inputs)))

        max_lead = lead_time_hours.amax(dim=1, keepdim=True).clamp_min(1e-6)
        lead_normalized = lead_time_hours / max_lead
        daily_phase = 2.0 * math.pi * physical_time_hours / 24.0
        query_pieces = [
            future_known,
            lead_normalized.unsqueeze(-1),
            torch.sin(daily_phase).unsqueeze(-1),
            torch.cos(daily_phase).unsqueeze(-1),
        ]
        if self.config.site_features:
            query_pieces.append(site_metadata.unsqueeze(1).expand(-1, future_known.shape[1], -1))
        queries = self.query_projection(torch.cat(query_pieces, dim=-1))

        key_padding_mask = ~history_mask.bool().any(dim=-1)
        # MultiheadAttention produces NaNs for a row in which every key is
        # masked.  Missing flags remain part of the encoded input, while this
        # single unmask supplies a finite neutral key for that degenerate row.
        all_missing = key_padding_mask.all(dim=1)
        if all_missing.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_missing, 0] = False
        attended, _ = self.cross_attention(
            queries,
            sequence_hidden,
            sequence_hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attended = self.attention_norm(queries + self.attention_dropout(attended))
        state_shared = self.shared_horizon(torch.cat((queries, attended), dim=-1))
        forecast_shared = state_shared
        if self.weather_projection is not None:
            if weather_context is None:
                raise ValueError("weather-enabled transient expert requires weather_context")
            forecast_shared = state_shared + self.weather_projection(weather_context)
        return forecast_shared, state_shared


def ramp_history_statistics(config: RaCEPVConfig, history_values: Tensor) -> Tensor:
    """Summarize recent raw history without depending on forecast experts."""

    power = history_values[..., config.power_index : config.power_index + 1]
    statistics: list[Tensor] = []
    for minutes in (30, 60, 180):
        steps = max(1, round(minutes / config.sampling_minutes))
        steps = min(steps, history_values.shape[1] - 1)
        change = power[:, -1] - power[:, -(steps + 1)]
        volatility = power[:, -(steps + 1) :].std(dim=1, unbiased=False)
        statistics.extend((change, volatility))
    six_hour_steps = max(2, round(360 / config.sampling_minutes))
    day_steps = max(2, round(1440 / config.sampling_minutes))
    statistics.extend(
        (
            normalized_spectral_entropy(power, six_hour_steps),
            normalized_spectral_entropy(power, day_steps),
        )
    )
    if config.use_clear_sky_prior:
        clear_sky = history_values[
            ...,
            config.history_clear_sky_index : config.history_clear_sky_index + 1,
        ]
        reference_deviation = power[:, -1] - clear_sky[:, -1]
    else:
        reference_deviation = power[:, -1] - power.mean(dim=1)
    statistics.append(reference_deviation)
    return torch.cat(statistics, dim=-1)


class _HorizonRouter(nn.Module):
    """Shared construction for independent fusion and physical-state heads."""

    def __init__(self, config: RaCEPVConfig, *, detach_expert_inputs: bool) -> None:
        super().__init__()
        self.config = config
        self.detach_expert_inputs = detach_expert_inputs
        # 3 change + 3 volatility + 2 entropy + 1 clear-sky deviation,
        # 3 expert states, 3 median disagreements, known future variables,
        # site metadata, and 3 physical lead encodings.
        input_size = 9 + 3 * config.d_model + 3 + config.future_features + config.site_features + 3
        self.contextual_network: nn.Sequential | None = None
        self.amplitude_network: nn.Sequential | None = None
        if config.router_mode == "contextual":
            self.contextual_network = nn.Sequential(
                nn.Linear(input_size, config.router_hidden_size),
                nn.LayerNorm(config.router_hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.router_hidden_size, config.router_hidden_size // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.router_hidden_size // 2, 3),
            )
        else:
            self.amplitude_network = nn.Sequential(
                nn.Linear(1, config.router_hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.router_hidden_size, 3),
            )

    def _history_statistics(self, history_values: Tensor) -> Tensor:
        return ramp_history_statistics(self.config, history_values)

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
        detach_expert_inputs: bool | None = None,
    ) -> Tensor:
        horizon = future_known.shape[1]
        if self.config.router_mode == "amplitude_only":
            # Faithful ablation of the original scalar-amplitude router.  This
            # branch deliberately does not access expert states, expert
            # forecasts, future-known features, or contextual statistics.
            amplitude, _ = last_observed_power(
                history_values[..., self.config.power_index],
                history_mask[..., self.config.power_index],
            )
            if self.amplitude_network is None:
                raise RuntimeError("amplitude router was not initialized")
            logits = (
                self.amplitude_network(amplitude.unsqueeze(-1)) / self.config.router_temperature
            )
            return logits.unsqueeze(1).expand(-1, horizon, -1)

        should_detach = (
            self.detach_expert_inputs if detach_expert_inputs is None else detach_expert_inputs
        )
        if should_detach:
            stable_hidden = stable_hidden.detach()
            up_hidden = up_hidden.detach()
            down_hidden = down_hidden.detach()
            expert_quantiles = expert_quantiles.detach()

        history_statistics = self._history_statistics(history_values)
        history_statistics = history_statistics.unsqueeze(1).expand(-1, horizon, -1)

        median = self.config.median_index
        expert_medians = expert_quantiles[..., median]
        disagreements = torch.stack(
            (
                expert_medians[..., 1] - expert_medians[..., 0],
                expert_medians[..., 2] - expert_medians[..., 0],
                expert_medians[..., 1] - expert_medians[..., 2],
            ),
            dim=-1,
        )
        max_lead = lead_time_hours.amax(dim=1, keepdim=True).clamp_min(1e-6)
        normalized_lead = lead_time_hours / max_lead
        clock_phase = 2.0 * math.pi * physical_time_hours / 24.0

        pieces = [
            history_statistics,
            stable_hidden,
            up_hidden,
            down_hidden,
            disagreements,
            future_known,
        ]
        if self.config.site_features:
            pieces.append(site_metadata.unsqueeze(1).expand(-1, horizon, -1))
        pieces.extend(
            (
                normalized_lead.unsqueeze(-1),
                torch.sin(clock_phase).unsqueeze(-1),
                torch.cos(clock_phase).unsqueeze(-1),
            )
        )
        if self.contextual_network is None:
            raise RuntimeError("contextual router was not initialized")
        return self.contextual_network(torch.cat(pieces, dim=-1)) / self.config.router_temperature


class ForecastFusionRouter(_HorizonRouter):
    """Forecast-risk router producing convex expert-fusion logits."""

    def __init__(self, config: RaCEPVConfig) -> None:
        super().__init__(config, detach_expert_inputs=False)
        self.weather_logits = (
            nn.Linear(config.future_weather_hidden_size, 3, bias=False)
            if config.future_weather_features
            else None
        )

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
        logits = super().forward(
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
            detach_expert_inputs=detach_expert_inputs,
        )
        if self.weather_logits is None:
            return logits
        if weather_context is None:
            raise ValueError("weather-enabled fusion router requires weather_context")
        return logits + self.weather_logits(weather_context)


class IndependentRampStateHead(nn.Module):
    """Raw-input ramp classifier with no forecast-expert representation inputs."""

    def __init__(self, config: RaCEPVConfig) -> None:
        super().__init__()
        self.config = config
        self.history_encoder = CausalConvEncoder(
            input_size=2 * config.history_features,
            hidden_size=config.d_model,
            kernel_size=config.stable_conv_kernel,
            dropout=0.0,
        )
        input_size = 9 + config.d_model + config.future_features + config.site_features + 3
        self.contextual_network = nn.Sequential(
            nn.Linear(input_size, config.router_hidden_size),
            nn.LayerNorm(config.router_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.router_hidden_size, config.router_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.router_hidden_size // 2, 3),
        )

    def forward(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor,
        lead_time_hours: Tensor,
        physical_time_hours: Tensor,
    ) -> Tensor:
        horizon = future_known.shape[1]
        encoder_input = torch.cat(
            (history_values, history_mask.to(history_values.dtype)),
            dim=-1,
        )
        valid_history = history_mask.bool().any(dim=-1)
        _, pooled = self.history_encoder(encoder_input, valid_history)
        history_statistics = (
            ramp_history_statistics(self.config, history_values)
            .unsqueeze(1)
            .expand(-1, horizon, -1)
        )
        repeated_context = pooled.unsqueeze(1).expand(-1, horizon, -1)
        max_lead = lead_time_hours.amax(dim=1, keepdim=True).clamp_min(1e-6)
        normalized_lead = lead_time_hours / max_lead
        clock_phase = 2.0 * math.pi * physical_time_hours / 24.0
        pieces = [history_statistics, repeated_context, future_known]
        if self.config.site_features:
            pieces.append(site_metadata.unsqueeze(1).expand(-1, horizon, -1))
        pieces.extend(
            (
                normalized_lead.unsqueeze(-1),
                torch.sin(clock_phase).unsqueeze(-1),
                torch.cos(clock_phase).unsqueeze(-1),
            )
        )
        return self.contextual_network(torch.cat(pieces, dim=-1)) / (self.config.router_temperature)


class RampStateHead(_HorizonRouter):
    """Physical ramp-state classifier isolated from forecast representations."""

    def __init__(self, config: RaCEPVConfig) -> None:
        super().__init__(config, detach_expert_inputs=True)
