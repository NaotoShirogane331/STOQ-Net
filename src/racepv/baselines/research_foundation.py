"""Origin-window mapping to the existing sealed, prediction-only foundation API.

Construction validates externally pinned source/weight receipts, not live weight
availability. The real backend rechecks official code and every declared weight
byte before loading. Synthetic providers exercise this mapping only.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from racepv.baselines.foundation_extension import (
    build_moirai_1_1_r_small_adapter,
    build_sundial_base_adapter,
    build_timesfm3_adapter,
)
from racepv.baselines.foundation_runtime import build_runtime_backend_factory
from racepv.io import REGISTERED_HISTORY_FEATURES

_BUILDERS = {
    "timesfm3_zs": build_timesfm3_adapter,
    "sundial_base_zs": build_sundial_base_adapter,
    "moirai_1_1_r_small_zs": build_moirai_1_1_r_small_adapter,
}
MAPPING_VERSION = "schema3_origin_history_power_only_v1"


def _signature(value):
    geometry = value["geometry"]
    if (
        geometry["lookback"], geometry["horizon"], geometry["stride"],
        value["clock"]["sampling_interval_ns"],
    ) != (192, 24, 4, 900_000_000_000):
        raise ValueError("foundation research geometry differs")
    columns = value["history_columns"]
    if len(columns) != 14 or set(columns) != set(REGISTERED_HISTORY_FEATURES):
        raise ValueError("foundation historical feature identity differs")
    return columns.index("power_norm")


class ResearchFoundation:
    """Prediction-only facade; labels and future values never reach the backend."""

    def __init__(self, adapter, identity):
        self._adapter = adapter
        self._identity = deepcopy(identity)

    def resolved_identity(self):
        return deepcopy(self._identity)

    def predict(self, batch):
        if any(key.startswith("future") and key != "future_known" for key in batch):
            raise ValueError("future observed inputs are forbidden")
        history = batch["history_values"]
        mask = batch["history_mask"]
        daylight = batch["daylight_mask"]
        if not all(isinstance(value, torch.Tensor) for value in (history, mask, daylight)):
            raise ValueError("foundation window inputs require tensors")
        if history.ndim != 3 or history.shape[0] < 1 or history.shape[1:] != (192, 14):
            raise ValueError("foundation history must be nonempty Bx192x14")
        if mask.shape != history.shape or mask.dtype != torch.bool:
            raise ValueError("foundation observation mask must be boolean and match history")
        if daylight.shape != (history.shape[0], 24) or daylight.dtype != torch.bool:
            raise ValueError("foundation daylight mask must be boolean Bx24")
        if not history.is_floating_point() or history.is_complex():
            raise ValueError("foundation power history must be real floating point")
        power = history[..., self._identity["power_index"]]
        if not bool(torch.isfinite(power).all()):
            raise ValueError("foundation history must retain finite origin-causal values")
        context = power.detach().float().cpu().numpy().copy()
        # Validate every consumed batch field before the lazy backend can load.
        night = ~daylight.detach().cpu().numpy()
        result = np.array(self._adapter.predict(context), dtype=np.float32, copy=True)
        expected = (len(context), 24)
        if self._identity["prediction_kind"] == "quantile":
            expected += (7,)
        if result.shape != expected or not np.isfinite(result).all():
            raise ValueError("sealed foundation prediction shape/finite contract differs")
        if np.any(result < 0) or np.any(result > 1.05):
            raise ValueError("sealed foundation physical range differs")
        if result.ndim == 3 and np.any(np.diff(result, axis=-1) < 0):
            raise ValueError("sealed foundation quantile order differs")
        result[night] = 0.
        return result


def build_research_foundation(
    model_id,
    feature_signature,
    *,
    source_review_bytes=None,
    source_review_sha256=None,
    weight_receipt_bytes=None,
    weight_receipt_sha256=None,
    project_root=None,
    device="cpu",
    synthetic_test=False,
    synthetic_backend_factory=None,
):
    """Bind receipts lazily. Injected providers are confined to synthetic identity."""
    if model_id not in _BUILDERS:
        raise ValueError("unknown or excluded foundation model")
    if type(synthetic_test) is not bool:
        raise ValueError("synthetic_test must be boolean")
    power_index = _signature(feature_signature)
    if synthetic_backend_factory is not None and (
        not synthetic_test or feature_signature.get("simulated") is not True
    ):
        raise ValueError("injected backend requires a simulated synthetic identity")
    if synthetic_test and feature_signature.get("simulated") is not True:
        raise ValueError("synthetic execution requires a simulated signature")
    parsed_device = torch.device(device)
    if parsed_device.type not in ("cpu", "cuda"):
        raise ValueError("foundation device must explicitly be cpu or cuda")
    snapshot = f"model-cache/baseline-extension/{model_id}"
    if synthetic_backend_factory is None:
        if project_root is None:
            raise ValueError("real sealed foundation backend requires project_root")
        factory = build_runtime_backend_factory(
            Path(project_root), model_id, device=str(parsed_device), snapshot_relative=snapshot
        )
    else:
        factory = synthetic_backend_factory
    adapter = _BUILDERS[model_id](
        source_review_bytes=source_review_bytes,
        source_review_sha256=source_review_sha256,
        weight_receipt_bytes=weight_receipt_bytes,
        weight_receipt_sha256=weight_receipt_sha256,
        horizon=24,
        backend_factory=factory,
    )
    frozen = adapter.resolved_identity()
    identity = dict(
        model_id=model_id,
        mapping_version=MAPPING_VERSION,
        feature_signature=deepcopy(feature_signature),
        power_index=power_index,
        input_information="history_target_only",
        history_missing_policy="retain_finite_origin_causal_power_no_explicit_mask_to_backend",
        prediction_kind="point" if model_id == "timesfm3_zs" else "quantile",
        training_kind="frozen_zero_shot",
        synthetic_test=synthetic_test,
        device=str(parsed_device),
        snapshot_relative=snapshot,
        postprocessing="sealed_physical_order_checks_then_input_daylight_zero_v1",
        frozen_adapter_identity=frozen,
        formal_ready=False,
        registered_evidence_eligible=False,
        source_interpretation_verified=False,
        live_weights_verified=False,
    )
    return ResearchFoundation(adapter, identity)
