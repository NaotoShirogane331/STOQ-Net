"""Training, prediction, and efficiency benchmarking utilities."""

from .checkpoint import load_racepv_checkpoint
from .engine import EarlyStoppingTrainer, TrainerProgress, TrainingResult
from .frozen_adapter import (
    FROZEN_WEATHER_ADAPTER_PREFIXES,
    FROZEN_WEATHER_ADAPTER_SCHEMA,
    initialize_frozen_weather_adapter,
)
from .loading import build_window_loader, loader_settings
from .search import GridTrial, run_bounded_grid_search
from .splits import ChronologicalDevelopmentSplit, chronological_development_split
from .timing import (
    EFFICIENCY_BENCHMARK_BATCH_SIZE,
    EFFICIENCY_BENCHMARK_REPETITIONS,
    EFFICIENCY_BENCHMARK_WARMUP,
    benchmark_callable,
    benchmark_inference,
    count_parameters,
    efficiency_audit_record,
)

__all__ = [
    "EarlyStoppingTrainer",
    "EFFICIENCY_BENCHMARK_BATCH_SIZE",
    "EFFICIENCY_BENCHMARK_REPETITIONS",
    "EFFICIENCY_BENCHMARK_WARMUP",
    "GridTrial",
    "FROZEN_WEATHER_ADAPTER_PREFIXES",
    "FROZEN_WEATHER_ADAPTER_SCHEMA",
    "ChronologicalDevelopmentSplit",
    "TrainingResult",
    "TrainerProgress",
    "benchmark_callable",
    "benchmark_inference",
    "build_window_loader",
    "count_parameters",
    "chronological_development_split",
    "efficiency_audit_record",
    "loader_settings",
    "initialize_frozen_weather_adapter",
    "load_racepv_checkpoint",
    "run_bounded_grid_search",
]
