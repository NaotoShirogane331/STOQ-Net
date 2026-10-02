"""Official NeuralForecast backbones on independent schema3 origin windows.

The future-only encoding uses structural zero padding plus availability channels;
it does not assert that unprovided historical solar/time measurements were zero.
No panel assembly, fit, normalization, target access, or output correction occurs
here. The research runner owns masked loss and common forecast postprocessing.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from racepv.baselines.external import build_neuralforecast_model
from racepv.io import (
    REGISTERED_FUTURE_FEATURES,
    REGISTERED_HISTORY_FEATURES,
    REGISTERED_SITE_FEATURES,
)

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
MAPPING_VERSION = "schema3_origin_future_only_values_availability_v1"
STRUCTURE_KEYS = {
    "nhits": (
        "stack_types", "n_blocks", "mlp_units", "n_pool_kernel_size",
        "n_freq_downsample", "pooling_mode", "interpolation_mode",
        "dropout_prob_theta", "activation",
    ),
    "tft": (
        "hidden_size", "n_head", "attn_dropout", "grn_activation", "n_rnn_layers",
        "rnn_type", "one_rnn_initial_state", "dropout",
    ),
    "xlstm": (
        "encoder_n_blocks", "encoder_hidden_size", "encoder_bias", "encoder_dropout",
        "decoder_hidden_size", "decoder_layers", "decoder_dropout", "decoder_activation",
        "backbone",
    ),
}


def _feature_names(signature: Mapping[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """Validate exact registered information sets while allowing column permutation."""
    geometry = signature["geometry"]
    if (geometry["lookback"], geometry["horizon"], geometry["stride"],
            signature["clock"]["sampling_interval_ns"]) != (192, 24, 4, 900_000_000_000):
        raise ValueError("official origin window geometry requires 192/24/stride4 at 15 minutes")
    groups = []
    for key, registered in (
        ("history_columns", REGISTERED_HISTORY_FEATURES),
        ("future_known_columns", REGISTERED_FUTURE_FEATURES),
        ("site_metadata_names", REGISTERED_SITE_FEATURES),
    ):
        names = list(signature[key])
        if len(names) != len(registered) or set(names) != set(registered):
            raise ValueError(f"feature signature must contain exactly the registered {key}")
        groups.append(names)
    return tuple(groups)


class OfficialOriginWindow(nn.Module):
    """One official backbone with an explicit, lossless legal-input window mapping."""

    def __init__(self, backbone: nn.Module, identity: Mapping[str, Any]) -> None:
        super().__init__()
        self.backbone = backbone
        self.research_identity = deepcopy(dict(identity))
        self.power_index = identity["feature_signature"]["history_columns"].index("power_norm")
        if identity["model_id"] == "tft":
            # NeuralForecast constructs this fixed causal attention buffer from
            # input_size+h; forward only adds it and never mutates it. Preserve
            # the exact tensor/forward, reconstruct it on each audited factory
            # call, and keep non-finite constants outside trainable checkpoint
            # state. All existing finite checkpoint admission remains strict.
            from neuralforecast.models.tft import InterpretableMultiHeadAttention
            attention = backbone.temporal_fusion_decoder.attention
            if type(attention) is not InterpretableMultiHeadAttention:
                raise ValueError("TFT attention implementation differs")
            mask = attention._mask
            expected = torch.triu(torch.full((216, 216), float("-inf"),
                dtype=mask.dtype, device=mask.device), 1).unsqueeze(0)
            if mask.requires_grad or not torch.equal(mask, expected):
                raise ValueError("TFT deterministic causal attention mask differs")
            attention.register_buffer("_mask", mask, persistent=False)
            self.research_identity["checkpoint_buffer_policy"] = dict(
                version="tft_fixed_causal_mask_reconstruction_v1",
                path="backbone.temporal_fusion_decoder.attention._mask",
                shape=[1, 216, 216], persistent=False,
                reconstruction="triu(full((192+24,192+24),-inf),1).unsqueeze(0)",
                forward_mutation=False, original_forward_unchanged=True)

    def to_windows_batch(
        self, history_values: Tensor, history_mask: Tensor,
        future_known: Tensor, site_metadata: Tensor,
    ) -> dict[str, Tensor]:
        """Map one already prepared batch without deleting finite causal imputations."""
        values = (history_values, history_mask, future_known, site_metadata)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError("official window inputs must be tensors with registered shapes")
        if history_values.ndim != 3:
            raise ValueError("history_values shape must be Bx192x14")
        batch_size = history_values.shape[0]
        shapes = ((batch_size, 192, 14), (batch_size, 192, 14),
                  (batch_size, 24, 9), (batch_size, 6))
        if batch_size < 1 or any(tuple(value.shape) != shape
                                 for value, shape in zip(values, shapes, strict=True)):
            raise ValueError("official window input shapes differ from Bx192x14/Bx24x9/Bx6")
        if history_mask.dtype != torch.bool:
            raise ValueError("history_mask must be boolean")
        numeric = (history_values, future_known, site_metadata)
        if not all(value.is_floating_point() for value in numeric):
            raise ValueError("official window numerical inputs must be floating point")
        if any(value.device != history_values.device for value in values):
            raise ValueError("official window input devices must agree")
        if any(value.dtype != history_values.dtype for value in numeric):
            raise ValueError("official window numerical input dtypes must agree")
        if not all(bool(torch.isfinite(value).all()) for value in numeric):
            raise ValueError("official window inputs must be finite, including causal imputations")

        masks = history_mask.to(history_values.dtype)
        # These zeros denote unavailable structural slots, never inferred observations.
        historical_padding = future_known.new_zeros((batch_size, 192, 18))
        future_values_and_availability = torch.cat(
            (future_known, torch.ones_like(future_known)), dim=-1,
        )
        return {
            "insample_y": history_values[..., self.power_index:self.power_index + 1],
            "insample_mask": masks[..., self.power_index:self.power_index + 1],
            "hist_exog": torch.cat((history_values, masks), dim=-1),
            "futr_exog": torch.cat((historical_padding, future_values_and_availability), dim=1),
            "stat_exog": site_metadata,
        }

    def forward(
        self, history_values: Tensor, history_mask: Tensor,
        future_known: Tensor, site_metadata: Tensor,
    ) -> Tensor:
        """Return raw Bx24x7 quantiles directly from the official backbone."""
        window = self.to_windows_batch(history_values, history_mask, future_known, site_metadata)
        result = self.backbone(window)
        if not isinstance(result, Tensor) or result.shape != (history_values.shape[0], 24, 7):
            raise RuntimeError("official backbone output shape must be Bx24x7")
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("official backbone emitted non-finite raw quantiles")
        return result


def build_official_window_model(
    model_id: str, feature_signature: Mapping[str, Any], resolved_config: Mapping[str, Any],
) -> OfficialOriginWindow:
    """Construct NHITS/TFT/xLSTM; accept only architecture and explicit seed settings.

    Omitted structure settings retain installed official constructor defaults.
    ``random_seed`` defaults to 13, matching the existing audited external factory.
    Caller configuration cannot override geometry, inputs, loss or normalization.
    """
    if model_id not in STRUCTURE_KEYS:
        raise ValueError(f"unsupported official window model: {model_id!r}")
    history, future, site = _feature_names(feature_signature)
    supplied = deepcopy(dict(resolved_config))
    unexpected = set(supplied) - set(STRUCTURE_KEYS[model_id]) - {"random_seed"}
    if unexpected:
        raise ValueError(f"unsupported official window configuration keys: {sorted(unexpected)}")
    seed = supplied.pop("random_seed", 13)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("configuration random_seed must be a nonnegative integer")
    if model_id == "xlstm" and supplied.get("backbone", "mLSTM") not in {"mLSTM", "sLSTM"}:
        raise ValueError("configuration backbone must be mLSTM or sLSTM")
    historical_columns = [f"hist__{name}" for name in history]
    historical_columns += [f"hist_mask__{name}" for name in history]
    future_columns = [f"futr__{name}" for name in future]
    future_columns += [f"futr_available__{name}" for name in future]
    static_columns = [f"stat__{name}" for name in site]
    backbone = build_neuralforecast_model(
        model_id, horizon=24, input_size=192, quantiles=QUANTILES,
        historical_exogenous=historical_columns, future_exogenous=future_columns,
        static_exogenous=static_columns, scaler_type="identity", random_seed=seed, **supplied,
    )
    parameters = inspect.signature(type(backbone).__init__).parameters
    defaults = {name: deepcopy(parameters[name].default) for name in STRUCTURE_KEYS[model_id]}
    source = inspect.getsourcefile(type(backbone))
    if source is None:
        raise RuntimeError("official backbone source provenance is unavailable")
    module_parts = type(backbone).__module__.split(".")
    source_path = Path(source)
    relative_source = Path(*module_parts).with_suffix(".py")
    if module_parts[0] != "neuralforecast" or source_path.parts[-len(relative_source.parts):] != relative_source.parts:
        raise RuntimeError("official backbone source provenance is unavailable: module path differs")
    identity = {
        "model_id": model_id,
        "implementation_kind": "official_NeuralForecast_backbone_origin_window_adaptation",
        "mapping_version": MAPPING_VERSION,
        "official_package": "neuralforecast",
        "official_version": version("neuralforecast"),
        "official_class": f"{type(backbone).__module__}.{type(backbone).__qualname__}",
        "official_source_file": relative_source.as_posix(),
        "official_source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "adapter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "feature_signature": deepcopy(dict(feature_signature)),
        "historical_exogenous": historical_columns,
        "future_exogenous": future_columns,
        "static_exogenous": static_columns,
        "official_structure_defaults": defaults,
        "resolved_structure": dict(defaults, **deepcopy(supplied)),
        "resolved_config": deepcopy(dict(resolved_config)),
        "random_seed": seed,
        "horizon": 24,
        "history_length": 192,
        "quantiles": list(QUANTILES),
        "scaler_type": "identity",
        "normalization": "none; direct backbone forward bypasses NeuralForecast window scaling",
        "loss_contract": "official MQLoss defines seven outputs only; runner owns masked loss",
        "history_missingness": "retain all finite causal values and original observation masks",
        "future_encoding": {
            "history": "nine structural zero values plus nine zero availability channels",
            "forecast": "nine supplied deterministic values plus nine one availability channels",
            "historical_measurements_inferred": False,
        },
        "training_api": "independent origin windows; no continuous-panel fit/predict",
        "formal_ready": False,
        "registered_evidence_eligible": False,
        "source_interpretation_verified": False,
    }
    if model_id == "xlstm":
        identity["xlstm_version"] = version("xlstm")
    return OfficialOriginWindow(backbone, identity)
