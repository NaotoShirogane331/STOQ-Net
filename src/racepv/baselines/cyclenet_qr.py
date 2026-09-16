"""Fixed CycleNet backbone adaptation for validation-only quantile forecasting."""

from __future__ import annotations

import _imp
import copy
import importlib.util
import math
import os
import re
import stat
import sys
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from racepv.baselines.neural import OrderedQuantileProjection
from racepv.io import (
    REGISTERED_FUTURE_FEATURES,
    REGISTERED_HISTORY_FEATURES,
    REGISTERED_SITE_FEATURES,
)
from scripts.audit_cyclenet_source_capsule import (
    SOURCE_CAPSULE_RELATIVE,
    SOURCE_RECEIPT_RELATIVE,
)
from scripts.cyclenet_validation_contract import (
    EXECUTABLE_ALLOWLIST,
    FIXED_SELECTED_FILES,
    MODEL_REVISION,
    MODEL_SCHEMA,
    SOURCE_SCHEMA,
    canonical_json_bytes,
    load_source_capsule_bytes,
    validate_source_capsule,
)
from scripts.racepv_reference_io import SafeReader

_SHA256 = re.compile(r"[0-9a-f]{64}")
_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
_MODEL_FIELDS = frozenset(
    {
        "schema_version",
        "model_id",
        "model_revision",
        "source_capsule_pin",
        "history_feature_names",
        "future_feature_names",
        "site_feature_names",
        "history_length",
        "horizon",
        "quantiles",
        "input_mapping",
        "future_scaling",
        "phase_contract",
        "backbone",
        "readout",
        "postprocess",
        "implementation_identity",
    }
)
_BACKBONE = {
    "seq_len": 192,
    "pred_len": 24,
    "enc_in": 28,
    "cycle": 96,
    "model_type": "mlp",
    "d_model": 512,
    "use_revin": False,
}
_FUTURE_SCALING = {
    "solar_elevation_divisor": 90.0,
    "lead_time_hours_divisor": 6.0,
}
_PHASE_CONTRACT = {
    "source": "future_known_hour_sin_cos",
    "cycle_length": 96,
    "first_future_slot_is_output_start": True,
    "history_start_formula": "(first_future_slot-history_length)%cycle_length",
    "unit_circle_tolerance": 1e-4,
    "slot_tolerance": 1e-3,
    "lead_tolerance": 1e-6,
}
_READOUT = {
    "input_dim": 43,
    "output_dim": 7,
    "shared_across_leads": True,
    "weight_init": {"distribution": "normal", "mean": 0.0, "std": 0.02},
    "bias_init": [-3.0, -3.0, -3.0, 0.5, -3.0, -3.0, -3.0],
}
_POSTPROCESS = {
    "ordered_projection": "median_softplus_gaps",
    "minimum_power": 0.0,
    "maximum_power": 1.05,
    "night_rule": "clear_sky_and_solar_elevation_positive",
}
_IMPLEMENTATION_IDENTITY = {
    "official_code_used": True,
    "implementation_kind": "official_backbone_protocol_adaptation",
    "native_probabilistic": False,
    "adapted_output_kind": "seven_ordered_quantiles",
    "point_projection_retained": True,
}


def _exact_mapping(value: object, expected: frozenset[str], *, role: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{role} keys differ")
    return value


def _validate_model_config(resolved_config: Mapping[str, object]) -> dict[str, object]:
    v2 = resolved_config.get("schema_version") == "racepv.cyclenet-qr-model.v2"
    value = _exact_mapping(
        resolved_config,
        _MODEL_FIELDS | {"prepared_contract"} if v2 else _MODEL_FIELDS,
        role="model config",
    )
    if v2:
        from racepv.baselines.extension_registry import validate_prepared_identity

        validate_prepared_identity(value["prepared_contract"])
    fixed = {
        "schema_version": "racepv.cyclenet-qr-model.v2" if v2 else MODEL_SCHEMA,
        "model_id": "cyclenet_qr",
        "model_revision": MODEL_REVISION,
        "history_feature_names": list(REGISTERED_HISTORY_FEATURES),
        "future_feature_names": list(REGISTERED_FUTURE_FEATURES),
        "site_feature_names": list(REGISTERED_SITE_FEATURES),
        "history_length": 192,
        "horizon": 24,
        "quantiles": list(_QUANTILES),
        "input_mapping": "masked_history_and_mask_to_cycle_then_future_site_readout_v1",
        "future_scaling": _FUTURE_SCALING,
        "phase_contract": _PHASE_CONTRACT,
        "backbone": _BACKBONE,
        "readout": _READOUT,
        "postprocess": _POSTPROCESS,
        "implementation_identity": _IMPLEMENTATION_IDENTITY,
    }
    messages = {
        "history_feature_names": "history feature identity differs",
        "future_feature_names": "future feature identity differs",
        "site_feature_names": "site feature identity differs",
        "backbone": "backbone identity differs",
    }
    for key, expected in fixed.items():
        if value[key] != expected:
            raise ValueError(messages.get(key, f"model config {key} differs"))
    pin = _exact_mapping(
        value["source_capsule_pin"],
        frozenset({"path", "sha256", "schema_version"}),
        role="source capsule pin",
    )
    if (
        pin["path"] != SOURCE_RECEIPT_RELATIVE
        or pin["schema_version"] != SOURCE_SCHEMA
        or not isinstance(pin["sha256"], str)
        or _SHA256.fullmatch(pin["sha256"]) is None
    ):
        raise ValueError("source capsule pin differs")
    return copy.deepcopy(dict(value))


def cycle_phase_from_future_known(
    future_known: Tensor,
    *,
    hour_sin_index: int,
    hour_cos_index: int,
    lead_index: int,
    history_length: int,
    cycle_length: int,
) -> Tensor:
    """Validate the fixed 15-minute clock and return the history-start slot."""

    if future_known.ndim != 3 or future_known.shape[1] != 24:
        raise ValueError("future clock must contain exactly 24 forecast steps")
    if not future_known.is_floating_point() or not bool(torch.isfinite(future_known).all()):
        raise ValueError("future clock and lead values must be finite")
    if type(history_length) is not int or history_length <= 0:
        raise ValueError("history length differs")
    if type(cycle_length) is not int or cycle_length != 96:
        raise ValueError("clock cycle length differs")
    indices = (hour_sin_index, hour_cos_index, lead_index)
    if (
        any(type(index) is not int for index in indices)
        or len(set(indices)) != 3
        or any(index < 0 or index >= future_known.shape[-1] for index in indices)
    ):
        raise ValueError("clock feature indices differ")

    sine = future_known[..., hour_sin_index]
    cosine = future_known[..., hour_cos_index]
    radius = sine.square() + cosine.square()
    if not bool((torch.abs(radius - 1.0) <= 1e-4).all()):
        raise ValueError("clock unit-circle identity differs")
    raw_slots = torch.remainder(torch.atan2(sine, cosine), 2.0 * math.pi)
    raw_slots = raw_slots * (cycle_length / (2.0 * math.pi))
    nearest = torch.round(raw_slots)
    distance = torch.minimum(
        torch.abs(raw_slots - nearest), cycle_length - torch.abs(raw_slots - nearest)
    )
    if not bool((distance <= 1e-3).all()):
        raise ValueError("clock slot is off the fixed grid")
    slots = torch.remainder(nearest.to(torch.int64), cycle_length)
    increments = torch.remainder(slots[:, 1:] - slots[:, :-1], cycle_length)
    if not bool((increments == 1).all()):
        raise ValueError("clock slots are not contiguous")

    expected_lead = (
        torch.arange(
            1,
            future_known.shape[1] + 1,
            dtype=future_known.dtype,
            device=future_known.device,
        )
        / 4.0
    )
    if not bool(
        torch.isclose(
            future_known[..., lead_index],
            expected_lead[None, :],
            rtol=0.0,
            atol=1e-6,
        ).all()
    ):
        raise ValueError("lead vector differs from the fixed 15-minute horizon")
    return torch.remainder(slots[:, 0] - history_length, cycle_length)


def _safe_project_root(project_root: Path) -> Path:
    root = Path(project_root)
    if not root.is_absolute():
        raise ValueError("project root must be absolute")
    info = root.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("project root must be a real directory")
    return root.resolve(strict=True)


def _verify_capsule_tree(project_root: Path) -> tuple[Path, dict[str, object]]:
    with SafeReader(project_root) as reader:
        receipt_bytes, receipt_identity = reader.read(
            SOURCE_RECEIPT_RELATIVE, role="CycleNet source capsule receipt"
        )
        record = load_source_capsule_bytes(receipt_bytes)
        reader.revalidate(receipt_identity)
    capsule = project_root.joinpath(*SOURCE_CAPSULE_RELATIVE.split("/"))
    info = os.lstat(capsule)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("source capsule directory differs")
    observed: set[str] = set()
    for base, dirnames, filenames in os.walk(capsule, followlinks=False):
        base_path = Path(base)
        for name in dirnames:
            node_info = os.lstat(base_path / name)
            if stat.S_ISLNK(node_info.st_mode) or not stat.S_ISDIR(node_info.st_mode):
                raise ValueError("source capsule tree differs")
        for name in filenames:
            path = base_path / name
            node_info = os.lstat(path)
            relative = path.relative_to(capsule).as_posix()
            if (
                stat.S_ISLNK(node_info.st_mode)
                or not stat.S_ISREG(node_info.st_mode)
                or node_info.st_nlink != 1
            ):
                raise ValueError("source capsule tree differs")
            observed.add(relative)
    expected = {str(item["path"]): str(item["sha256"]) for item in FIXED_SELECTED_FILES}
    if observed != set(expected):
        raise ValueError("source capsule tree differs")
    with SafeReader(capsule) as reader:
        for relative, digest in expected.items():
            identity = reader.digest_source(relative)
            if identity.sha256 != digest:
                raise ValueError("source capsule file hash differs")
    return capsule, record


def load_cyclenet_model_class(
    project_root: Path, source_record: Mapping[str, object]
) -> type[nn.Module]:
    """Load only the audited upstream model text from a private snapshot."""

    root = _safe_project_root(project_root)
    expected_record = validate_source_capsule(source_record)
    capsule, observed_record = _verify_capsule_tree(root)
    if observed_record != expected_record or canonical_json_bytes(
        observed_record
    ) != canonical_json_bytes(expected_record):
        raise ValueError("source capsule record differs")
    if tuple(EXECUTABLE_ALLOWLIST) != ("models/CycleNet.py",):
        raise ValueError("source executable allowlist differs")

    with SafeReader(capsule) as reader:
        model_bytes, model_identity = reader.read(
            "models/CycleNet.py", role="CycleNet audited model source"
        )
        module_name = f"_racepv_cyclenet_{uuid.uuid4().hex}"
        with tempfile.TemporaryDirectory(prefix="racepv-cyclenet-source-") as temporary:
            snapshot = Path(temporary) / "CycleNet.py"
            descriptor = os.open(
                snapshot,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            try:
                view = memoryview(model_bytes)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("private source snapshot write made no progress")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

            _imp.acquire_lock()
            try:
                spec = importlib.util.spec_from_file_location(module_name, snapshot)
                if spec is None or spec.loader is None:
                    raise ImportError("cannot create CycleNet source module")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                model_class = getattr(module, "Model", None)
                if not isinstance(model_class, type) or not issubclass(model_class, nn.Module):
                    raise TypeError("audited CycleNet source does not expose nn.Module Model")
            finally:
                sys.modules.pop(module_name, None)
                _imp.release_lock()
        reader.revalidate(model_identity)
    _, final_record = _verify_capsule_tree(root)
    if final_record != observed_record:
        raise ValueError("source capsule changed during model load")
    return model_class


def _validated_mask(mask: Tensor) -> Tensor:
    if mask.dtype == torch.bool:
        return mask
    if not mask.is_floating_point() or not bool(torch.isfinite(mask).all()):
        raise ValueError("history_mask must be bool or finite binary floating point")
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("history_mask must be binary")
    return mask.bool()


class CycleNetQuantileAdapter(nn.Module):
    """Official CycleNet point backbone plus one fixed ordered-quantile readout."""

    def __init__(
        self,
        resolved_config: Mapping[str, object],
        upstream_model_class: type[nn.Module],
    ) -> None:
        super().__init__()
        self._resolved_config = _validate_model_config(resolved_config)
        if not isinstance(upstream_model_class, type) or not issubclass(
            upstream_model_class, nn.Module
        ):
            raise TypeError("upstream CycleNet Model must be an nn.Module class")
        self.history_length = 192
        self.horizon = 24
        self.cycle_length = 96
        self.quantiles = _QUANTILES
        self.hour_sin_index = REGISTERED_FUTURE_FEATURES.index("hour_sin")
        self.hour_cos_index = REGISTERED_FUTURE_FEATURES.index("hour_cos")
        self.lead_index = REGISTERED_FUTURE_FEATURES.index("lead_time_hours")
        self.solar_elevation_index = REGISTERED_FUTURE_FEATURES.index("solar_elevation")
        self.clear_sky_index = REGISTERED_FUTURE_FEATURES.index("clear_sky_power_norm")

        self.backbone = upstream_model_class(SimpleNamespace(**_BACKBONE))
        self.quantile_head = nn.Linear(43, 7)
        nn.init.normal_(self.quantile_head.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.quantile_head.bias.copy_(
                torch.tensor(_READOUT["bias_init"], dtype=self.quantile_head.bias.dtype)
            )
        self.ordered = OrderedQuantileProjection(self.quantiles)

    def resolved_identity(self) -> dict[str, object]:
        return {
            "model_id": "cyclenet_qr",
            "model_revision": MODEL_REVISION,
            **_IMPLEMENTATION_IDENTITY,
            "input_mapping": "masked_history_and_mask_to_cycle_then_future_site_readout_v1",
        }

    def _validate_inputs(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor,
    ) -> Tensor:
        batch = history_values.shape[0] if history_values.ndim == 3 else -1
        if history_values.shape != (batch, 192, len(REGISTERED_HISTORY_FEATURES)):
            raise ValueError("history_values shape differs")
        if history_mask.shape != history_values.shape:
            raise ValueError("history_mask shape differs")
        if future_known.shape != (batch, 24, len(REGISTERED_FUTURE_FEATURES)):
            raise ValueError("future_known shape differs")
        if site_metadata.shape != (batch, len(REGISTERED_SITE_FEATURES)):
            raise ValueError("site_metadata shape differs")
        if history_values.dtype != torch.float32:
            raise ValueError("history_values must be float32")
        if future_known.dtype != torch.float32 or site_metadata.dtype != torch.float32:
            raise ValueError("future_known and site_metadata must be float32")
        if not (
            history_values.device
            == history_mask.device
            == future_known.device
            == site_metadata.device
        ):
            raise ValueError("all CycleNet adapter inputs must share one device")
        valid_mask = _validated_mask(history_mask)
        if not bool(torch.isfinite(history_values[valid_mask]).all()):
            raise ValueError("observed history values must be finite")
        if not bool(torch.isfinite(future_known).all()) or not bool(
            torch.isfinite(site_metadata).all()
        ):
            raise ValueError("future and site values must be finite")
        return valid_mask

    def backbone_output(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch = history_values.shape[0] if history_values.ndim == 3 else -1
        dummy_site = torch.zeros(
            (batch, len(REGISTERED_SITE_FEATURES)),
            dtype=history_values.dtype,
            device=history_values.device,
        )
        valid_mask = self._validate_inputs(history_values, history_mask, future_known, dummy_site)
        masked = torch.where(valid_mask, history_values, torch.zeros_like(history_values))
        cycle_inputs = torch.cat((masked, valid_mask.to(history_values.dtype)), dim=-1)
        cycle_index = cycle_phase_from_future_known(
            future_known,
            hour_sin_index=self.hour_sin_index,
            hour_cos_index=self.hour_cos_index,
            lead_index=self.lead_index,
            history_length=self.history_length,
            cycle_length=self.cycle_length,
        )
        output = self.backbone(cycle_inputs, cycle_index)
        if output.shape != (batch, self.horizon, 28):
            raise ValueError("CycleNet backbone output shape differs")
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("CycleNet backbone output contains non-finite values")
        return output, cycle_index

    def raw_quantiles(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor,
    ) -> Tensor:
        valid_mask = self._validate_inputs(
            history_values, history_mask, future_known, site_metadata
        )
        del valid_mask
        backbone_output, _ = self.backbone_output(history_values, history_mask, future_known)
        future_scaled = future_known.clone()
        future_scaled[..., self.solar_elevation_index] /= 90.0
        future_scaled[..., self.lead_index] /= 6.0
        readout_input = torch.cat(
            (
                backbone_output,
                future_scaled,
                site_metadata[:, None, :].expand(-1, self.horizon, -1),
            ),
            dim=-1,
        )
        raw = self.quantile_head(readout_input)
        if raw.shape != (history_values.shape[0], self.horizon, len(self.quantiles)):
            raise ValueError("raw quantile shape differs")
        if not bool(torch.isfinite(raw).all()):
            raise FloatingPointError("raw quantiles contain non-finite values")
        return raw

    def forward(
        self,
        history_values: Tensor,
        history_mask: Tensor,
        future_known: Tensor,
        site_metadata: Tensor,
    ) -> Tensor:
        raw = self.raw_quantiles(history_values, history_mask, future_known, site_metadata)
        ordered = self.ordered(raw)
        bounded = ordered.clamp(0.0, 1.05)
        daylight = (future_known[..., self.clear_sky_index] > 0) & (
            future_known[..., self.solar_elevation_index] > 0
        )
        return torch.where(daylight[..., None], bounded, torch.zeros_like(bounded))


def build_cyclenet_qr(
    project_root: Path, resolved_config: Mapping[str, object]
) -> CycleNetQuantileAdapter:
    root = _safe_project_root(project_root)
    config = _validate_model_config(resolved_config)
    with SafeReader(root) as reader:
        receipt_bytes, receipt = reader.read(
            SOURCE_RECEIPT_RELATIVE, role="CycleNet source capsule receipt"
        )
        if receipt.sha256 != config["source_capsule_pin"]["sha256"]:
            raise ValueError("source capsule receipt hash differs")
        source_record = load_source_capsule_bytes(receipt_bytes)
        reader.revalidate(receipt)
    model_class = load_cyclenet_model_class(root, source_record)
    return CycleNetQuantileAdapter(config, model_class)
