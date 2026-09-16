"""Parameter, latency, and peak-memory measurements."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

EFFICIENCY_BENCHMARK_BATCH_SIZE = 128
EFFICIENCY_BENCHMARK_WARMUP = 5
EFFICIENCY_BENCHMARK_REPETITIONS = 20


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def efficiency_audit_record(
    *,
    trainable_parameters: int | None,
    trainable_parameters_unavailable_reason: str | None,
    model_artifact_bytes: int,
    model_artifact_definition: str,
    batch_inference: Mapping[str, object],
) -> dict[str, object]:
    """Wrap model size and batch timing in the registered efficiency schema."""

    if trainable_parameters is None and not trainable_parameters_unavailable_reason:
        raise ValueError("a null trainable-parameter count requires an explicit reason")
    if trainable_parameters is not None and trainable_parameters < 0:
        raise ValueError("trainable_parameters cannot be negative")
    if model_artifact_bytes < 0:
        raise ValueError("model_artifact_bytes cannot be negative")
    if not model_artifact_definition.strip():
        raise ValueError("model_artifact_definition cannot be empty")
    required_batch_fields = {
        "batch_size_windows",
        "warmup_repetitions",
        "timed_repetitions",
        "latency_ms_mean",
        "throughput_windows_per_second",
        "device",
        "device_synchronization",
        "peak_gpu_memory_bytes",
        "peak_gpu_memory_unavailable_reason",
    }
    missing = sorted(required_batch_fields.difference(batch_inference))
    if missing:
        raise ValueError(f"batch_inference is missing efficiency fields: {missing}")
    return {
        "schema_version": 1,
        "trainable_parameters": trainable_parameters,
        "trainable_parameters_unavailable_reason": (
            trainable_parameters_unavailable_reason
        ),
        "model_artifact_bytes": int(model_artifact_bytes),
        "model_artifact_definition": model_artifact_definition,
        "batch_inference": dict(batch_inference),
    }


def _synchronize_device(device: torch.device) -> str:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return "torch.cuda.synchronize"
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
        return "torch.mps.synchronize"
    return "not_applicable_cpu"


def benchmark_callable(
    predict: Callable[[], Any],
    *,
    batch_size: int,
    device: str | torch.device = "cpu",
    warmup: int = EFFICIENCY_BENCHMARK_WARMUP,
    repetitions: int = EFFICIENCY_BENCHMARK_REPETITIONS,
    scope: str,
    inference_mode: bool = True,
) -> dict[str, float | int | str | None]:
    """Benchmark one already-prepared batch with an auditable common schema.

    Feature/dataframe construction and artifact serialization should happen before
    this function. ``predict`` must execute exactly one batch inference call.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    resolved_device = torch.device(device)
    synchronization = "not_applicable_cpu"
    context = torch.inference_mode() if inference_mode else nullcontext()
    with context:
        for _ in range(warmup):
            predict()
        synchronization = _synchronize_device(resolved_device)
        if resolved_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resolved_device)
            gpu_memory_baseline = int(torch.cuda.memory_allocated(resolved_device))
        else:
            gpu_memory_baseline = None
        timings = []
        for _ in range(repetitions):
            started = time.perf_counter()
            predict()
            synchronization = _synchronize_device(resolved_device)
            timings.append((time.perf_counter() - started) * 1000.0)

    mean_latency = float(np.mean(timings))
    peak_memory = (
        int(torch.cuda.max_memory_allocated(resolved_device))
        if resolved_device.type == "cuda"
        else None
    )
    peak_increment = (
        max(0, peak_memory - gpu_memory_baseline)
        if peak_memory is not None and gpu_memory_baseline is not None
        else None
    )
    if resolved_device.type == "cuda":
        memory_reason = None
    elif resolved_device.type == "mps":
        memory_reason = "PyTorch exposes no resettable peak-allocation counter for MPS"
    else:
        memory_reason = "benchmark executed on CPU; GPU memory is not applicable"
    return {
        "schema_version": 1,
        "scope": scope,
        "latency_unit": "milliseconds_per_batch",
        "timing_boundary": (
            "inputs_preloaded; synchronize after warmup and after each timed call"
        ),
        "batch_size_windows": int(batch_size),
        "warmup_repetitions": int(warmup),
        "timed_repetitions": int(repetitions),
        "latency_ms_mean": mean_latency,
        "latency_ms_std": (
            float(np.std(timings, ddof=1)) if len(timings) > 1 else 0.0
        ),
        "throughput_windows_per_second": (
            float(batch_size * 1000.0 / mean_latency)
            if mean_latency > 0
            else float("inf")
        ),
        "device": str(resolved_device),
        "device_synchronization": synchronization,
        "peak_gpu_memory_bytes": peak_memory,
        "gpu_memory_baseline_bytes": gpu_memory_baseline,
        "peak_gpu_memory_increment_bytes": peak_increment,
        "peak_gpu_memory_unavailable_reason": memory_reason,
    }


@torch.inference_mode()
def benchmark_inference(
    model: nn.Module,
    inputs: Mapping[str, Tensor],
    *,
    warmup: int = EFFICIENCY_BENCHMARK_WARMUP,
    repetitions: int = EFFICIENCY_BENCHMARK_REPETITIONS,
) -> dict[str, float | int | str | None]:
    device = next(model.parameters()).device
    prepared = {key: value.to(device) for key, value in inputs.items()}
    model.eval()
    result = benchmark_callable(
        lambda: model(**prepared),
        batch_size=int(next(iter(prepared.values())).shape[0]),
        device=device,
        warmup=warmup,
        repetitions=repetitions,
        scope="torch_module_forward_on_preloaded_batch_excluding_artifact_io",
    )
    # Compatibility aliases remain for existing main-model run readers.
    result["parameters"] = count_parameters(model)
    result["peak_memory_bytes"] = result["peak_gpu_memory_bytes"]
    result["batch_size"] = result["batch_size_windows"]
    return result
