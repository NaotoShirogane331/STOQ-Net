"""Frozen zero-shot adapters with a sealed, past-target-only boundary."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import PurePosixPath
from typing import Protocol

import numpy as np

from scripts.baseline_extension_sources import load_source_review_bytes

QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
SUNDIAL_SAMPLE_COUNT = 100
SUNDIAL_RNG_SEED = 20260907
MOIRAI_SAMPLE_COUNT = 100
MOIRAI_RNG_SEED = 20260907
MAXIMUM_POWER = 1.05

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_WEIGHT_RECEIPT_KEYS = {
    "schema_version",
    "purpose",
    "status",
    "model_id",
    "source_review_sha256",
    "weight_id",
    "weight_revision",
    "weight_license_id",
    "weight_license_sha256",
    "model_card_sha256",
    "config_sha256",
    "checkpoint_format",
    "deserialization",
    "files",
    "runtime_code_files",
}
_WEIGHT_LICENSES = {
    "timesfm3_zs": "timesfm-non-commercial-license-v1.0",
    "sundial_base_zs": "Apache-2.0",
    "moirai_1_1_r_small_zs": "CC-BY-NC-4.0",
}
_DESERIALIZATION = {
    "timesfm3_zs": {"allow_pickle": False, "loader": "safetensors_offline"},
    "sundial_base_zs": {
        "allow_pickle": False,
        "loader": "transformers_trust_remote_code_offline_sealed",
    },
    "moirai_1_1_r_small_zs": {"allow_pickle": False, "loader": "safetensors_offline"},
}
_SUNDIAL_RUNTIME_CODE = (
    "configuration_sundial.py",
    "flow_loss.py",
    "modeling_sundial.py",
    "ts_generation_mixin.py",
)
_PROBABILISTIC_METRICS = [
    "overall_qcrps",
    "daylight_nmae",
    "daylight_nrmse",
    "ramp_nmae",
    "ramp_nrmse",
    "picp80",
    "picp90",
    "interval_score",
]
_POINT_METRICS = [
    "overall_nmae",
    "overall_nrmse",
    "daylight_nmae",
    "daylight_nrmse",
    "ramp_nmae",
    "ramp_nrmse",
]
_PROBABILISTIC_ONLY_METRICS = ["overall_qcrps", "picp80", "picp90", "interval_score"]
_MODEL_IDENTITIES = {
    "timesfm3_zs": {
        "repository_url": "https://github.com/google-research/timesfm.git",
        "git_commit": "0df95ae62085a6ac0d0afd1ad40dee2e6c1356ab",
        "source_tree_sha256": "a314c525867c5534ed7294fdb44263e3b3d373749dbea33acefc71b349bd786a",
        "weight_id": "google/timesfm-3.0-pytorch",
    },
    "sundial_base_zs": {
        "repository_url": "https://github.com/thuml/Sundial.git",
        "git_commit": "3ef03b8c3804f64a506e57101a173470040aaece",
        "source_tree_sha256": "add737ced5e14f5405610eb36c9d0ef9795291c0ccfbd5e199097ea54ca2560c",
        "weight_id": "thuml/sundial-base-128m",
    },
    "moirai_1_1_r_small_zs": {
        "repository_url": "https://github.com/SalesforceAIResearch/uni2ts.git",
        "git_commit": "cfd46d4510ed8896f263116f32928eede05b0a75",
        "source_tree_sha256": "575784cbd4418e2884324c030e15e6109312357eddfa2330bc13ecf3720a3c6b",
        "weight_id": "Salesforce/moirai-1.1-R-small",
    },
}
_CAPABILITY_IDENTITIES = {
    "timesfm3_zs": {
        "native_probabilistic": False,
        "output_kind": "point_median",
        "accepted_context": "past_target_only",
        "accepted_horizon": "project_horizon",
        "accepted_covariates": [],
        "exact_quantile_levels": [],
        "requires_sampling": False,
        "fixed_sample_count": None,
        "fixed_rng_seed": None,
        "fixed_patch_size": None,
        "comparable_metrics": _POINT_METRICS,
        "excluded_metrics": _PROBABILISTIC_ONLY_METRICS,
        "exclusion_reasons": [
            {
                "code": "not_comparable_point_forecast_only",
                "metrics": _PROBABILISTIC_ONLY_METRICS,
            }
        ],
    },
    "sundial_base_zs": {
        "native_probabilistic": True,
        "output_kind": "samples",
        "accepted_context": "past_target_only",
        "accepted_horizon": "project_horizon",
        "accepted_covariates": [],
        "exact_quantile_levels": list(QUANTILES),
        "requires_sampling": True,
        "fixed_sample_count": SUNDIAL_SAMPLE_COUNT,
        "fixed_rng_seed": SUNDIAL_RNG_SEED,
        "fixed_patch_size": None,
        "comparable_metrics": _PROBABILISTIC_METRICS,
        "excluded_metrics": [],
        "exclusion_reasons": [],
    },
    "moirai_1_1_r_small_zs": {
        "native_probabilistic": True,
        "output_kind": "samples",
        "accepted_context": "past_target_only",
        "accepted_horizon": "project_horizon",
        "accepted_covariates": [],
        "exact_quantile_levels": list(QUANTILES),
        "requires_sampling": True,
        "fixed_sample_count": MOIRAI_SAMPLE_COUNT,
        "fixed_rng_seed": MOIRAI_RNG_SEED,
        "fixed_patch_size": 32,
        "comparable_metrics": _PROBABILISTIC_METRICS,
        "excluded_metrics": [],
        "exclusion_reasons": [],
    },
}


class DirectQuantileBackend(Protocol):
    def forecast_quantiles(
        self,
        past_target: np.ndarray,
        *,
        horizon: int,
        quantiles: tuple[float, ...],
    ) -> Mapping[str, object]: ...


class SamplingBackend(Protocol):
    def sample(
        self,
        past_target: np.ndarray,
        *,
        horizon: int,
        num_samples: int,
        rng_seed: int,
        patch_size: int | None = None,
    ) -> np.ndarray: ...


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _parse_unique_json(content: bytes, *, label: str) -> dict[str, object]:
    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant in {label}: {value}")

    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=unique_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if type(payload) is not dict:
        raise ValueError(f"{label} must be a JSON mapping")
    return payload


def _validate_relative_checkpoint_path(value: object) -> str:
    if not isinstance(value, str) or "\\" in value or "\x00" in value:
        raise ValueError("unsafe checkpoint path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("unsafe checkpoint path")
    if path.suffix != ".safetensors":
        raise ValueError("unsafe checkpoint deserialization format")
    return value


def _validate_relative_runtime_code_path(value: object) -> str:
    if not isinstance(value, str) or "\\" in value or "\x00" in value:
        raise ValueError("unsafe runtime code path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("unsafe runtime code path")
    if path.suffix != ".py":
        raise ValueError("unsafe runtime code path")
    return value


def _validate_weight_receipt(
    model_id: str,
    content: bytes,
    expected_sha256: str,
    source_record: Mapping[str, object],
    source_review_sha256: str,
) -> dict[str, object]:
    if not _is_sha256(expected_sha256) or hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("weight receipt SHA-256 differs")
    receipt = _parse_unique_json(content, label="weight receipt")
    if set(receipt) != _WEIGHT_RECEIPT_KEYS:
        raise ValueError("weight receipt schema differs")
    expected = _MODEL_IDENTITIES[model_id]
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or receipt["purpose"] != "racepv_frozen_weight_receipt"
        or receipt["status"] != "passed"
        or receipt["model_id"] != model_id
        or receipt["source_review_sha256"] != source_review_sha256
        or receipt["weight_id"] != expected["weight_id"]
        or receipt["weight_id"] != source_record["weight_id"]
        or receipt["weight_revision"] != source_record["weight_revision"]
        or receipt["weight_license_id"] != source_record["weight_license_id"]
        or receipt["weight_license_sha256"] != source_record["weight_license_sha256"]
    ):
        raise ValueError("weight receipt identity differs")
    if (
        not isinstance(receipt["weight_revision"], str)
        or _COMMIT.fullmatch(receipt["weight_revision"]) is None
    ):
        raise ValueError("weight revision must be an exact commit")
    for field in ("weight_license_sha256", "model_card_sha256", "config_sha256"):
        if not _is_sha256(receipt[field]):
            raise ValueError(f"{field} must be a lower-case SHA-256")
    if receipt["weight_license_id"] != _WEIGHT_LICENSES[model_id]:
        raise ValueError("weight license identity differs")
    if (
        receipt["checkpoint_format"] != "safetensors"
        or receipt["deserialization"] != _DESERIALIZATION[model_id]
    ):
        raise ValueError("unsafe checkpoint deserialization is forbidden")
    files = receipt["files"]
    if type(files) is not list or not files:
        raise ValueError("weight receipt files differ")
    paths: list[str] = []
    for item in files:
        if type(item) is not dict or set(item) != {"path", "sha256"}:
            raise ValueError("weight receipt file schema differs")
        paths.append(_validate_relative_checkpoint_path(item["path"]))
        if not _is_sha256(item["sha256"]):
            raise ValueError("checkpoint SHA-256 differs")
    if paths != sorted(set(paths)) or len({path.casefold() for path in paths}) != len(paths):
        raise ValueError("weight receipt file paths differ")
    runtime_files = receipt["runtime_code_files"]
    if type(runtime_files) is not list:
        raise ValueError("runtime code inventory differs")
    runtime_paths: list[str] = []
    for item in runtime_files:
        if type(item) is not dict or set(item) != {"path", "sha256"}:
            raise ValueError("runtime code file schema differs")
        runtime_paths.append(_validate_relative_runtime_code_path(item["path"]))
        if not _is_sha256(item["sha256"]):
            raise ValueError("runtime code SHA-256 differs")
    expected_runtime = list(_SUNDIAL_RUNTIME_CODE if model_id == "sundial_base_zs" else ())
    if runtime_paths != expected_runtime or len({path.casefold() for path in runtime_paths}) != len(
        runtime_paths
    ):
        raise ValueError("runtime code inventory differs")
    return copy.deepcopy(receipt)


def validate_weight_receipt_bytes(
    model_id: str,
    content: bytes,
    *,
    expected_sha256: str,
    source_record: Mapping[str, object],
    source_review_sha256: str,
) -> dict[str, object]:
    """Validate a frozen-weight receipt without constructing or loading a model."""

    return _validate_weight_receipt(
        model_id,
        content,
        expected_sha256,
        source_record,
        source_review_sha256,
    )


def _authorize(
    model_id: str,
    *,
    source_review_bytes: bytes,
    source_review_sha256: str,
    weight_receipt_bytes: bytes,
    weight_receipt_sha256: str,
    horizon: int,
) -> dict[str, object]:
    if model_id not in _MODEL_IDENTITIES:
        raise ValueError(f"unknown frozen foundation model: {model_id}")
    if not _is_sha256(source_review_sha256) or (
        hashlib.sha256(source_review_bytes).hexdigest() != source_review_sha256
    ):
        raise ValueError("source review SHA-256 differs")
    review = load_source_review_bytes(source_review_bytes)
    if review["status"] != "passed":
        raise ValueError("source review is not accepted")
    source_record = review["models"][model_id]
    expected = _MODEL_IDENTITIES[model_id]
    for field in ("repository_url", "git_commit", "source_tree_sha256", "weight_id"):
        if source_record[field] != expected[field]:
            raise ValueError(f"sealed source identity differs for {model_id}: {field}")
    if source_record["status"] != "passed" or source_record["blockers"] != []:
        raise ValueError("foundation source record is not accepted")
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    weight_receipt = _validate_weight_receipt(
        model_id,
        weight_receipt_bytes,
        weight_receipt_sha256,
        source_record,
        source_review_sha256,
    )
    capability = source_record["capability_contract"]
    expected_sampling = model_id in {"sundial_base_zs", "moirai_1_1_r_small_zs"}
    if capability != _CAPABILITY_IDENTITIES[model_id]:
        raise ValueError("sealed capability contract differs")
    return {
        "schema_version": "racepv.foundation-extension-adapter.v2",
        "model_id": model_id,
        "repository_url": expected["repository_url"],
        "git_commit": expected["git_commit"],
        "source_tree_sha256": expected["source_tree_sha256"],
        "source_review_sha256": source_review_sha256,
        "weight_id": expected["weight_id"],
        "weight_revision": weight_receipt["weight_revision"],
        "weight_receipt_sha256": weight_receipt_sha256,
        "weight_license_id": weight_receipt["weight_license_id"],
        "weight_license_sha256": weight_receipt["weight_license_sha256"],
        "model_card_sha256": weight_receipt["model_card_sha256"],
        "config_sha256": weight_receipt["config_sha256"],
        "checkpoint_format": "safetensors",
        "deserialization": copy.deepcopy(weight_receipt["deserialization"]),
        "checkpoint_files": copy.deepcopy(weight_receipt["files"]),
        "runtime_code_files": copy.deepcopy(weight_receipt["runtime_code_files"]),
        "training_regime": "frozen_zero_shot",
        "input_information": "past_target_only",
        "horizon": horizon,
        "prediction_kind": capability["output_kind"],
        "quantiles": copy.deepcopy(capability["exact_quantile_levels"]),
        "comparable_metrics": copy.deepcopy(capability["comparable_metrics"]),
        "excluded_metrics": copy.deepcopy(capability["excluded_metrics"]),
        "exclusion_reasons": copy.deepcopy(capability["exclusion_reasons"]),
        "sampling_contract": (
            {"num_samples": 100, "rng_seed": 20260907} if expected_sampling else None
        ),
    }


def _real_numeric_array(value: object, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype == object
        or np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.complexfloating)
        or not np.issubdtype(array.dtype, np.number)
    ):
        raise ValueError(f"{label} must be a real numeric array")
    return array


def _past_target_array(value: np.ndarray) -> np.ndarray:
    array = _real_numeric_array(value, label="past_target")
    if not isinstance(value, np.ndarray) or array.ndim != 2 or min(array.shape) <= 0:
        raise ValueError("past_target must have shape [batch, history]")
    if not np.isfinite(array).all():
        raise ValueError("past_target must be finite")
    if np.any(np.abs(array) > np.finfo(np.float32).max):
        raise ValueError("past_target must remain finite after float32 conversion")
    result = np.array(array, dtype=np.float32, copy=True, order="C")
    if not np.isfinite(result).all():
        raise ValueError("past_target must remain finite after float32 conversion")
    result.setflags(write=False)
    return result


def _validate_predictions(value: object, batch: int, horizon: int) -> np.ndarray:
    array = _real_numeric_array(value, label="predictions")
    expected = (batch, horizon, len(QUANTILES))
    if array.shape != expected:
        raise ValueError(f"prediction shape must be {expected}")
    if not np.isfinite(array).all():
        raise ValueError("predictions must be finite")
    if np.any(array < 0.0) or np.any(array > MAXIMUM_POWER):
        raise ValueError("predictions must remain within the physical range [0, 1.05]")
    if np.any(np.diff(array, axis=-1) < 0.0):
        raise ValueError("prediction quantile crossing is forbidden")
    result = np.array(array, dtype=np.float32, copy=True, order="C")
    result.setflags(write=False)
    return result


def _validate_point_predictions(value: object, batch: int, horizon: int) -> np.ndarray:
    array = _real_numeric_array(value, label="point predictions")
    expected = (batch, horizon)
    if array.shape != expected:
        raise ValueError(f"point prediction shape must be {expected}")
    if not np.isfinite(array).all():
        raise ValueError("point predictions must be finite")
    if np.any(array < 0.0) or np.any(array > MAXIMUM_POWER):
        raise ValueError("point predictions must remain within the physical range [0, 1.05]")
    result = np.array(array, dtype=np.float32, copy=True, order="C")
    result.setflags(write=False)
    return result


class FoundationForecastAdapter(Protocol):
    """Public prediction-only interface; concrete construction stays sealed."""

    def resolved_identity(self) -> dict[str, object]: ...

    def predict(self, past_target: np.ndarray) -> np.ndarray: ...


class _SealedFoundationForecastAdapter:
    """Concrete adapter constructible only by this module's audited builders."""

    def __init__(
        self,
        identity: Mapping[str, object],
        backend_factory: Callable[[dict[str, object]], object],
    ) -> None:
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable")
        self.model_id = str(identity["model_id"])
        self._identity = copy.deepcopy(identity)
        self._backend_factory = backend_factory
        self._backend: object | None = None
        self._repeat_digests: dict[str, str] = {}

    def _backend_instance(self) -> object:
        if self._backend is None:
            self._backend = self._backend_factory(copy.deepcopy(self._identity))
        return self._backend

    def resolved_identity(self) -> dict[str, object]:
        return copy.deepcopy(self._identity)

    def _direct_predictions(self, past_target: np.ndarray) -> np.ndarray:
        method = getattr(self._backend_instance(), "forecast_quantiles", None)
        if not callable(method):
            raise TypeError("direct foundation backend lacks forecast_quantiles")
        payload = method(
            past_target,
            horizon=int(self._identity["horizon"]),
            quantiles=QUANTILES,
        )
        if type(payload) is not dict or set(payload) != {"quantile_levels", "predictions"}:
            raise ValueError("direct quantile backend result schema differs")
        if payload["quantile_levels"] != list(QUANTILES):
            raise ValueError("backend must return the exact registered quantile levels")
        return _validate_predictions(
            payload["predictions"], past_target.shape[0], int(self._identity["horizon"])
        )

    def _point_predictions(self, past_target: np.ndarray) -> np.ndarray:
        method = getattr(self._backend_instance(), "forecast_point", None)
        if not callable(method):
            raise TypeError("point foundation backend lacks forecast_point")
        return _validate_point_predictions(
            method(past_target, horizon=int(self._identity["horizon"])),
            past_target.shape[0],
            int(self._identity["horizon"]),
        )

    def _sampling_predictions(self, past_target: np.ndarray) -> np.ndarray:
        method = getattr(self._backend_instance(), "sample", None)
        if not callable(method):
            raise TypeError(f"{self.model_id} backend lacks sample")
        sample_kwargs = {
            "horizon": int(self._identity["horizon"]),
            "num_samples": SUNDIAL_SAMPLE_COUNT,
            "rng_seed": SUNDIAL_RNG_SEED,
        }
        if self.model_id == "moirai_1_1_r_small_zs":
            sample_kwargs["patch_size"] = 32
        raw_samples = _real_numeric_array(
            method(past_target, **sample_kwargs),
            label=f"{self.model_id} samples",
        )
        expected = (past_target.shape[0], SUNDIAL_SAMPLE_COUNT, int(self._identity["horizon"]))
        if raw_samples.shape != expected:
            raise ValueError(
                f"{self.model_id} backend must return exactly 100 samples with shape {expected}"
            )
        if not np.isfinite(raw_samples).all():
            raise ValueError(f"{self.model_id} samples must be finite")
        if np.any(raw_samples < 0.0) or np.any(raw_samples > MAXIMUM_POWER):
            raise ValueError(
                f"{self.model_id} samples must remain within the physical range [0, 1.05]"
            )

        input_digest = hashlib.sha256()
        input_digest.update(str(past_target.shape).encode("ascii"))
        input_digest.update(past_target.tobytes())
        input_key = input_digest.hexdigest()
        sample_digest = hashlib.sha256()
        sample_digest.update(raw_samples.dtype.str.encode("ascii"))
        sample_digest.update(str(raw_samples.shape).encode("ascii"))
        sample_digest.update(raw_samples.tobytes())
        observed_digest = sample_digest.hexdigest()
        previous = self._repeat_digests.setdefault(input_key, observed_digest)
        if previous != observed_digest:
            raise ValueError(f"{self.model_id} stochastic repeat mismatch for the fixed RNG seed")

        quantiles = np.quantile(raw_samples, QUANTILES, axis=1).transpose(1, 2, 0)
        return _validate_predictions(
            quantiles, past_target.shape[0], int(self._identity["horizon"])
        )

    def predict(self, past_target: np.ndarray) -> np.ndarray:
        """Forecast without fit, fine-tuning, calibration, or covariates."""

        history = _past_target_array(past_target)
        if self.model_id == "timesfm3_zs":
            return self._point_predictions(history)
        if self.model_id in {"sundial_base_zs", "moirai_1_1_r_small_zs"}:
            return self._sampling_predictions(history)
        return self._direct_predictions(history)


def _build_foundation_adapter(
    model_id: str,
    *,
    source_review_bytes: bytes,
    source_review_sha256: str,
    weight_receipt_bytes: bytes,
    weight_receipt_sha256: str,
    horizon: int,
    backend_factory: Callable[[dict[str, object]], object],
) -> FoundationForecastAdapter:
    identity = _authorize(
        model_id,
        source_review_bytes=source_review_bytes,
        source_review_sha256=source_review_sha256,
        weight_receipt_bytes=weight_receipt_bytes,
        weight_receipt_sha256=weight_receipt_sha256,
        horizon=horizon,
    )
    return _SealedFoundationForecastAdapter(identity, backend_factory)


def build_timesfm3_adapter(**kwargs: object) -> FoundationForecastAdapter:
    """Build the source- and weight-bound TimesFM-3 adapter."""

    return _build_foundation_adapter("timesfm3_zs", **kwargs)


def build_sundial_base_adapter(**kwargs: object) -> FoundationForecastAdapter:
    """Build the fixed 100-sample Sundial Base adapter."""

    return _build_foundation_adapter("sundial_base_zs", **kwargs)


def build_moirai_1_1_r_small_adapter(**kwargs: object) -> FoundationForecastAdapter:
    """Build the exact frozen Moirai 1.1-R Small adapter."""

    return _build_foundation_adapter("moirai_1_1_r_small_zs", **kwargs)
