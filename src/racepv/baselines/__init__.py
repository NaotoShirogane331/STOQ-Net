"""Reference baselines sharing the RaCE-PV information protocol."""

from .classical import (
    ClearSkyPersistence,
    QuantileLightGBM,
    RandomForestPointForecaster,
    SmartPersistence,
    ramp_state_probabilities_from_median,
)
from .external import build_neuralforecast_model
from .modern import (
    MODERN_BASELINES,
    REGISTERED_LEVELS,
    ModernBaselineLoss,
    ModernBaselineSpec,
    ModernCovariateTokenBuilder,
    ModernQuantilePostprocessor,
    RampClassWeight,
    build_modern_baseline,
    ramp_class_weight_from_training_labels,
)
from .neural import LSTMQuantile, PatchTSTQuantile, TCNQuantile
from .registry import build_baseline, list_baselines
from .s_mamba import SMambaQuantile

__all__ = [
    "ClearSkyPersistence",
    "LSTMQuantile",
    "MODERN_BASELINES",
    "ModernBaselineLoss",
    "ModernBaselineSpec",
    "ModernCovariateTokenBuilder",
    "ModernQuantilePostprocessor",
    "PatchTSTQuantile",
    "QuantileLightGBM",
    "RandomForestPointForecaster",
    "REGISTERED_LEVELS",
    "RampClassWeight",
    "SmartPersistence",
    "ramp_state_probabilities_from_median",
    "SMambaQuantile",
    "TCNQuantile",
    "build_baseline",
    "build_modern_baseline",
    "build_neuralforecast_model",
    "list_baselines",
    "ramp_class_weight_from_training_labels",
]
