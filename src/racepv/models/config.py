"""Configuration for the RaCE-PV neural model.

The experiment protocol fixes a 48-hour history and a six-hour horizon at a
15-minute cadence.  The lengths remain configurable here so that the model can
be unit-tested and reused at another cadence without introducing
length-dependent spectral parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

DEFAULT_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
EXPERT_NAMES = ("stable", "ramp_up", "ramp_down")
ARCHITECTURE_VERSION = 2
OBJECTIVE_VERSION = 2
FORECAST_SELECTION_METRIC = "daylight_qcrps"
STATE_HEAD_SEMANTICS = "physical_ramp_state_classification"
FUSION_GATE_SEMANTICS = "forecast_risk_convex_expert_fusion"
REGISTERED_SPECIALIST_LOSS_WEIGHT = 0.15
REGISTERED_RAMP_DECISION_THRESHOLD = 0.25
FORECAST_WEATHER_CONTRACT_VERSION = 1
REGISTERED_FORECAST_WEATHER_FEATURES = 7


@dataclass(frozen=True)
class RaCEPVConfig:
    """Shape and architecture configuration for :class:`RaCEPV`.

    Feature index conventions are explicit to make future-information audits
    possible.  In particular, ``future_clear_sky_index`` must point to a
    quantity that is computable at issue time, never to measured future power
    or weather.
    """

    history_features: int
    future_features: int
    site_features: int = 0
    history_length: int = 192
    horizon: int = 24
    sampling_minutes: int = 15
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES
    architecture_version: int = ARCHITECTURE_VERSION

    power_index: int = 0
    history_clear_sky_index: int = 1
    future_clear_sky_index: int = 0
    future_hour_sin_index: int | None = None
    future_hour_cos_index: int | None = None
    # Retained only so legacy serialized configurations remain parseable.  V2
    # never uses this field as a physical clock.
    future_time_hours_index: int | None = None
    future_weather_features: int = 0
    future_weather_hidden_size: int = 16
    forecast_weather_mode: Literal[
        "off",
        "experts",
        "experts_fusion",
        "transient_isolated",
    ] = "off"
    forecast_weather_contract_version: int = FORECAST_WEATHER_CONTRACT_VERSION
    # Extra prepared columns that directly encode clear-sky quantities, such
    # as clear-sky residual and clear-sky index.  They are removed together
    # with the primary clear-sky channel in the no-prior ablation.
    clear_sky_history_feature_indices: tuple[int, ...] = ()
    clear_sky_future_feature_indices: tuple[int, ...] = ()

    d_model: int = 64
    d_state: int = 16
    d_conv: int = 4
    num_backbone_layers: int = 2
    num_attention_heads: int = 4
    dropout: float = 0.1
    stable_conv_kernel: int = 5
    harmonic_periods_hours: tuple[float, ...] = (24.0, 12.0, 8.0, 6.0)

    # ``auto`` uses official mamba_ssm when importable and otherwise emits a
    # warning before selecting the explicitly named causal Conv-GRU fallback.
    backbone: Literal["auto", "mamba", "gru", "tcn"] = "auto"
    mamba_expand: int = 2

    use_clear_sky_prior: bool = True
    router_mode: Literal["contextual", "amplitude_only"] = "contextual"
    fusion_mode: Literal["learned", "equal"] = "learned"
    state_head_mode: Literal["flat", "hierarchical"] = "flat"
    state_representation_mode: Literal["expert_detached", "independent_raw"] = "expert_detached"
    prediction_state_coupling: Literal[
        "decoupled",
        "joint_head",
        "state_as_gate",
    ] = "decoupled"
    enable_auxiliary_gate_path: bool = False
    ramp_decision_threshold: float = REGISTERED_RAMP_DECISION_THRESHOLD
    active_experts: tuple[str, ...] = EXPERT_NAMES

    daylight_threshold: float = 0.01
    power_upper_bound: float = 1.05
    quantile_gap_scale: float = 0.05
    stable_residual_scale: float = 0.25
    router_hidden_size: int = 128
    router_temperature: float = 1.0
    persistence_tau_initial_hours: float = 1.0
    persistence_tau_min_hours: float = 0.25
    persistence_tau_max_hours: float = 6.0
    tail_residual_mode: Literal["off", "sparse_ramp_up"] = "off"
    tail_residual_cap: float = 0.08
    tail_residual_hidden_size: int = 64

    def __post_init__(self) -> None:
        if self.architecture_version != ARCHITECTURE_VERSION:
            raise ValueError(
                f"RaCEPV only constructs architecture_version={ARCHITECTURE_VERSION}; "
                "legacy checkpoints are evaluation-only"
            )
        positive_ints = {
            "history_features": self.history_features,
            "future_features": self.future_features,
            "history_length": self.history_length,
            "horizon": self.horizon,
            "sampling_minutes": self.sampling_minutes,
            "d_model": self.d_model,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "num_backbone_layers": self.num_backbone_layers,
            "num_attention_heads": self.num_attention_heads,
            "stable_conv_kernel": self.stable_conv_kernel,
            "router_hidden_size": self.router_hidden_size,
            "future_weather_hidden_size": self.future_weather_hidden_size,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.site_features < 0:
            raise ValueError("site_features cannot be negative")
        if self.future_weather_features not in {
            0,
            REGISTERED_FORECAST_WEATHER_FEATURES,
        }:
            raise ValueError(
                "future_weather_features must be 0 for legacy/off models or "
                f"{REGISTERED_FORECAST_WEATHER_FEATURES} for registered weather experiments"
            )
        if self.forecast_weather_mode not in {
            "off",
            "experts",
            "experts_fusion",
            "transient_isolated",
        }:
            raise ValueError(
                "forecast_weather_mode must be off, experts, experts_fusion, "
                "or transient_isolated"
            )
        if (
            self.forecast_weather_mode != "off"
            and self.future_weather_features != REGISTERED_FORECAST_WEATHER_FEATURES
        ):
            raise ValueError(
                "enabled forecast weather requires future_weather_features=7"
            )
        if self.forecast_weather_contract_version != FORECAST_WEATHER_CONTRACT_VERSION:
            raise ValueError(
                "unsupported forecast_weather_contract_version; legacy checkpoints "
                "remain evaluation-only"
            )
        if self.router_hidden_size < 2:
            raise ValueError("router_hidden_size must be at least 2")
        if self.d_model % self.num_attention_heads:
            raise ValueError("d_model must be divisible by num_attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.backbone not in {"auto", "mamba", "gru", "tcn"}:
            raise ValueError("backbone must be one of: auto, mamba, gru, tcn")
        if self.router_mode not in {"contextual", "amplitude_only"}:
            raise ValueError("router_mode must be contextual or amplitude_only")
        if self.fusion_mode not in {"learned", "equal"}:
            raise ValueError("fusion_mode must be learned or equal")
        if self.state_head_mode not in {"flat", "hierarchical"}:
            raise ValueError("state_head_mode must be flat or hierarchical")
        if self.state_representation_mode not in {
            "expert_detached",
            "independent_raw",
        }:
            raise ValueError("state_representation_mode must be expert_detached or independent_raw")
        if self.prediction_state_coupling not in {
            "decoupled",
            "joint_head",
            "state_as_gate",
        }:
            raise ValueError(
                "prediction_state_coupling must be decoupled, joint_head, or state_as_gate"
            )
        if (
            self.prediction_state_coupling == "joint_head"
            and self.state_representation_mode != "expert_detached"
        ):
            raise ValueError("joint_head requires state_representation_mode=expert_detached")
        if not 0.0 < self.ramp_decision_threshold <= 1.0:
            raise ValueError("ramp_decision_threshold must lie in (0, 1]")
        if not self.active_experts:
            raise ValueError("active_experts must contain at least one expert")
        if len(set(self.active_experts)) != len(self.active_experts):
            raise ValueError("active_experts cannot contain duplicates")
        unknown_experts = set(self.active_experts).difference(EXPERT_NAMES)
        if unknown_experts:
            raise ValueError(f"unknown active experts: {sorted(unknown_experts)}")
        if self.router_temperature <= 0:
            raise ValueError("router_temperature must be positive")
        if self.daylight_threshold < 0:
            raise ValueError("daylight_threshold cannot be negative")
        if self.power_upper_bound <= 0:
            raise ValueError("power_upper_bound must be positive")
        if self.quantile_gap_scale <= 0:
            raise ValueError("quantile_gap_scale must be positive")
        if self.stable_residual_scale <= 0:
            raise ValueError("stable_residual_scale must be positive")
        if not (
            0.0
            < self.persistence_tau_min_hours
            <= self.persistence_tau_initial_hours
            <= self.persistence_tau_max_hours
        ):
            raise ValueError("persistence tau bounds must satisfy 0 < min <= initial <= max")
        if self.persistence_tau_min_hours == self.persistence_tau_max_hours:
            raise ValueError("persistence tau bounds must have non-zero width")
        if self.tail_residual_mode not in {"off", "sparse_ramp_up"}:
            raise ValueError("tail_residual_mode must be off or sparse_ramp_up")
        if not math.isfinite(self.tail_residual_cap) or self.tail_residual_cap != 0.08:
            raise ValueError("tail_residual_cap is fixed at 0.08")
        if self.tail_residual_hidden_size != 64:
            raise ValueError("tail_residual_hidden_size is fixed at 64")

        if len(self.quantiles) < 3:
            raise ValueError("at least three quantiles are required")
        if tuple(sorted(set(self.quantiles))) != self.quantiles:
            raise ValueError("quantiles must be unique and strictly increasing")
        if not all(0.0 < q < 1.0 for q in self.quantiles):
            raise ValueError("quantiles must lie strictly between zero and one")
        if 0.5 not in self.quantiles:
            raise ValueError("quantiles must include 0.5 for the ordered head")
        if not self.harmonic_periods_hours or any(
            period <= 0 for period in self.harmonic_periods_hours
        ):
            raise ValueError("harmonic periods must be non-empty and positive")

        self._validate_index("power_index", self.power_index, self.history_features)
        self._validate_index(
            "history_clear_sky_index", self.history_clear_sky_index, self.history_features
        )
        self._validate_index(
            "future_clear_sky_index", self.future_clear_sky_index, self.future_features
        )
        if self.future_hour_sin_index is None or self.future_hour_cos_index is None:
            raise ValueError(
                "architecture_version=2 requires future_hour_sin_index and future_hour_cos_index"
            )
        self._validate_index(
            "future_hour_sin_index", self.future_hour_sin_index, self.future_features
        )
        self._validate_index(
            "future_hour_cos_index", self.future_hour_cos_index, self.future_features
        )
        if self.future_hour_sin_index == self.future_hour_cos_index:
            raise ValueError("future hour sine and cosine indices must be distinct")
        if self.future_time_hours_index is not None:
            self._validate_index(
                "future_time_hours_index", self.future_time_hours_index, self.future_features
            )
        self._validate_indices(
            "clear_sky_history_feature_indices",
            self.clear_sky_history_feature_indices,
            self.history_features,
        )
        self._validate_indices(
            "clear_sky_future_feature_indices",
            self.clear_sky_future_feature_indices,
            self.future_features,
        )

    @staticmethod
    def _validate_index(name: str, index: int, size: int) -> None:
        if not 0 <= index < size:
            raise ValueError(f"{name}={index} is outside a feature dimension of size {size}")

    @classmethod
    def _validate_indices(cls, name: str, indices: tuple[int, ...], size: int) -> None:
        if len(set(indices)) != len(indices):
            raise ValueError(f"{name} cannot contain duplicates")
        for index in indices:
            cls._validate_index(name, index, size)

    @property
    def median_index(self) -> int:
        """Index of the fixed median quantile."""

        return self.quantiles.index(0.5)
