"""Schema3 research bridges. No legacy admission changes, CLI, or implicit fitting.

The caller owns execution authorization. All fit methods require train identity;
real numerical execution and lifecycle/recovery remain a separate task.
"""

import inspect
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from racepv.baselines.research_registry import QUANTILES, get_research_spec
from racepv.data import scoped_consumer
from racepv.data.prepared_contract import canonical_json_sha256
from racepv.data.scoped_dataset import ScopedCandidateDataset
from racepv.data.scoped_scientific_view import (
    require_train_validation_compatibility,
    scientific_view_from_metadata,
)

NONFORMAL = dict(formal_ready=False, registered_evidence_eligible=False, source_interpretation_verified=False)
OBJECTIVE_VERSION = "qcrps_trapezoid_zero_endpoints_v1"
OFFICIAL_WINDOW_MODELS = frozenset(("nhits", "tft", "xlstm"))
OFFICIAL_QUANTILE_MAPPING = "official_raw_quantile_monotone_rearrangement_v1"


def _signature(signature):
    geometry = signature["geometry"]
    if (geometry["lookback"], geometry["horizon"], geometry["stride"],
            signature["clock"]["sampling_interval_ns"]) != (192, 24, 4, 900_000_000_000):
        raise ValueError("research geometry requires 192/24/stride4 at 15 minutes")
    names = [signature[key] for key in ("history_columns", "future_known_columns", "site_metadata_names")]
    if tuple(map(len, names)) != (14, 9, 6) or any(len(set(x)) != len(x) for x in names):
        raise ValueError("research feature signature requires unique 14/9/6 channels")
    if any(x in names[1] for x in ("power_norm", "targets", "future_observed", "future_target")):
        raise ValueError("future observed values are forbidden")
    for group, required in ((names[0], ("power_norm", "clear_sky_power_norm")),
                            (names[1], ("clear_sky_power_norm", "solar_elevation"))):
        if not all(x in group for x in required):
            raise ValueError("required named deterministic features are missing")
    return names


def inspect_research_inputs(args, *, synthetic_test=False):
    """Metadata only; exact scoped pair and same explicit evidence assumptions."""
    report = scoped_consumer.inspect_scoped_consumer_inputs(args)
    evidence, views = None, {}
    for split, entry in report["inputs"].items():
        metadata = entry["metadata"]
        for leaf in metadata["leaves"]:
            current = leaf["schema"]["source_time_contract"]["evidence"]
            if current["kind"] != ("synthetic_fixture" if synthetic_test else "research_assumption"):
                raise ValueError("research source evidence domain differs")
            if evidence is not None and current != evidence:
                raise ValueError("research assumptions differ across leaves")
            evidence = current
        views[split] = scientific_view_from_metadata(metadata, selected_split=split,
            expected_metadata_sha256=canonical_json_sha256(metadata))
        _signature(views[split].feature_signature)
    require_train_validation_compatibility(views["train"], views["validation"])
    return dict(report, evidence=deepcopy(evidence), feature_signature=views["train"].feature_signature, **NONFORMAL)


def load_research_inputs(args, expected_report, *, synthetic_test=False):
    actual = inspect_research_inputs(args, synthetic_test=synthetic_test)
    if canonical_json_sha256(actual) != canonical_json_sha256(expected_report):
        raise ValueError("research preflight identity drift")
    # Reconstruct the exact inspected scoped report, not a freshly accepted identity.
    preflight = {k: v for k, v in actual.items() if k not in ("evidence", "feature_signature")}
    loaded = scoped_consumer.load_scoped_engineering_inputs(args,
        expected_preflight_sha256=canonical_json_sha256(preflight))
    views, datasets = {}, {}
    for split, store in loaded["stores"].items():
        metadata = store["metadata"]
        views[split] = scientific_view_from_metadata(metadata, selected_split=split,
            expected_metadata_sha256=canonical_json_sha256(metadata))
        datasets[split] = ScopedCandidateDataset(store, views[split])
    require_train_validation_compatibility(views["train"], views["validation"])
    metadata = dict(evidence=actual["evidence"], feature_signature=actual["feature_signature"],
                    preflight_sha256=loaded["preflight_sha256"], **NONFORMAL)
    return dict(loaded, views=views, datasets=datasets, metadata=metadata, **NONFORMAL)


def _finite_raw(module, args, kwargs):
    if not bool(torch.isfinite(args[0]).all()):
        raise FloatingPointError("non-finite raw neural quantiles")


def _json_identity(value):
    """Detach identity and normalize tuples/Paths without losing numeric types."""
    def path_only(item):
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"unsupported research identity value: {type(item).__name__}")
    return json.loads(json.dumps(value, default=path_only, allow_nan=False))


def _constructor_config(model, kwargs):
    """Resolve defaults from the constructor that actually produced this model."""
    bound = inspect.signature(type(model)).bind(**kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def build_research_neural(model_id, feature_signature, resolved_config):
    """Reuse original computation/source auditing; never substitute a backbone."""
    spec = get_research_spec(model_id)
    if spec["bridge_status"] == "NOT_CONNECTED" or spec["training_kind"] != "supervised":
        raise RuntimeError(f"NOT_READY: {model_id} schema3 adapter is not connected")
    history, future, site = _signature(feature_signature)
    supplied = deepcopy(resolved_config)
    for key, required in (("history_length", 192), ("horizon", 24)):
        if key in supplied and supplied[key] != required:
            raise ValueError(f"model config geometry differs: {key}")
    common = dict(input_dim=len(history), future_dim=len(future), site_dim=len(site), quantiles=QUANTILES)
    from racepv.baselines import neural
    from racepv.baselines.modern import (
        MODERN_BASELINES,
        ModernQuantilePostprocessor,
        build_modern_baseline,
    )
    factories = dict(lstm_qr=neural.LSTMQuantile, patchtst_qr=neural.PatchTSTQuantile)
    config = dict(spec["default_config"], **supplied)
    factory_spec, upstream_identity, official_identity = None, None, None
    try:
        if model_id in OFFICIAL_WINDOW_MODELS:
            from racepv.baselines.research_official import build_official_window_model
            try:
                model = build_official_window_model(model_id, feature_signature, supplied)
            except (ImportError, OSError, RuntimeError) as error:
                raise RuntimeError(f"NOT_READY: {model_id}: {error}") from error
            official_identity = deepcopy(model.research_identity)
            # The independent adapter's resolved_config means requested overrides.
            # Public resolved_config instead records its complete effective contract.
            effective = {key: deepcopy(value) for key, value in official_identity.items()
                         if key not in ("resolved_config", "model_id", *NONFORMAL)}
            effective.update(common, output_dim=7, geometry=deepcopy(feature_signature["geometry"]),
                             clock=deepcopy(feature_signature["clock"]))
        elif model_id == "tcn_qr":
            from racepv.baselines.research_tcn import ResearchTCNQuantile

            constructor_kwargs = dict(common, **config)
            model = ResearchTCNQuantile(**constructor_kwargs)
            effective = _constructor_config(model, constructor_kwargs)
            effective.update(model.resolved_identity())
        elif model_id in factories:
            constructor_kwargs = dict(common, **config)
            model = factories[model_id](**common, **config)
            effective = _constructor_config(model, constructor_kwargs)
        elif model_id == "s_mamba_qr":
            from racepv.baselines.s_mamba import SMambaQuantile
            constructor_kwargs = dict(common, **config)
            model = SMambaQuantile(**common, **config)
            effective = _constructor_config(model, constructor_kwargs)
        elif model_id in MODERN_BASELINES:
            source = config.pop("source_identity", None)
            root = config.pop("openltm_root", None)
            if model_id != "timer_xl_qr":
                config.setdefault("history_length", 192)
                config.setdefault("horizon", 24)
            model, factory_spec = build_modern_baseline(model_id, **common, hyperparameters=config,
                openltm_root=Path(root) if root is not None else None, source_identity=source)
            constructor_kwargs = dict(common, **config)
            constructor_kwargs.pop("implementation_kind", None)
            initialization = constructor_kwargs.pop("initialization", None)
            if model_id == "timer_xl_qr":
                constructor_kwargs.update(openltm_root=Path(root), source_identity=source)
            effective = _constructor_config(model, constructor_kwargs)
            effective["implementation_kind"] = factory_spec.implementation_kind
            if model_id == "timer_xl_qr":
                from racepv.baselines.modern import build_timer_xl_config
                effective["initialization"] = initialization or "task_specific_from_scratch"
                effective["backbone_config"] = vars(build_timer_xl_config(
                    input_dim=model.token_builder.token_dim,
                    **{key: effective[key] for key in ("patch_len", "d_model", "n_heads", "e_layers", "dropout")}))
        elif model_id in ("fact_qr", "itransformer_qr"):
            from racepv.baselines.extension_registry import build_extension_model
            for key, expected in dict(input_dim=len(history), future_dim=len(future), site_dim=len(site),
                    history_feature_names=history, future_feature_names=future, site_feature_names=site,
                    history_length=192, horizon=24, quantiles=list(QUANTILES)).items():
                if supplied.get(key) != expected:
                    raise ValueError(f"extension config differs from feature signature: {key}")
            model = build_extension_model(model_id, supplied)
            upstream_identity = model.resolved_identity()
            effective = deepcopy(supplied)
        else:
            raise ValueError(f"{model_id} is not a neural baseline")
    except (ImportError, FileNotFoundError) as error:
        raise RuntimeError(f"NOT_READY: {model_id}: {error}") from error
    except ValueError as error:
        if model_id in ("timer_xl_qr", "fact_qr", "itransformer_qr"):
            raise RuntimeError(f"NOT_READY: source/config validation for {model_id}: {error}") from error
        raise
    for module in model.modules():
        if isinstance(module, ModernQuantilePostprocessor):
            module.clear_sky_feature_index = future.index("clear_sky_power_norm")
            module.daylight_feature_index = future.index("solar_elevation")
            module.register_forward_pre_hook(_finite_raw, with_kwargs=True)
    effective.update(history_length=192, horizon=24, postprocessing=dict(
        clear_sky_feature_index=future.index("clear_sky_power_norm"),
        daylight_feature_index=future.index("solar_elevation"), maximum_power=1.05,
        minimum_power=0., night_zero=True, night_mask="batch.daylight_mask",
        require_finite=True, require_ordered=True))
    effective["history_missingness"] = (
        "retain_all_finite_causal_values" if official_identity else "zero_maskfalse_values")
    effective["postprocessing"]["quantile_mapping"] = (
        OFFICIAL_QUANTILE_MAPPING if official_identity else "require_ordered_input")
    implementation = model.implementation_metadata() if callable(getattr(model, "implementation_metadata", None)) else {}
    backbone = getattr(model, "backbone", getattr(model, "forward_mamba", model))
    model_type, backbone_type = type(model), type(backbone)
    actual_backbone = factory_spec.actual_backbone if factory_spec else f"{backbone_type.__module__}.{backbone_type.__qualname__}"
    official = factory_spec.official_code_used if factory_spec else bool(
        official_identity or (upstream_identity or {}).get("official_code_used", model_id == "s_mamba_qr"))
    model.research_identity = _json_identity(dict(spec,
        feature_signature=feature_signature, requested_config=resolved_config,
        resolved_config=effective, model_class=model_type.__qualname__, model_module=model_type.__module__,
        actual_backbone=actual_backbone, official_code_used=official,
        backbone_class=backbone_type.__qualname__, backbone_module=backbone_type.__module__,
        factory_spec=asdict(factory_spec) if factory_spec else None,
        implementation_metadata=implementation, upstream_resolved_identity=upstream_identity,
        official_resolved_identity=official_identity,
        source_identity=effective.get("source_identity"), **NONFORMAL))
    return model


def _inputs(batch, *, history_missingness="zero_maskfalse_values"):
    if history_missingness not in ("zero_maskfalse_values", "retain_all_finite_causal_values"):
        raise ValueError("unknown history missingness policy")
    if any(key.startswith("future") and key != "future_known" for key in batch):
        raise ValueError("future observed inputs are forbidden")
    result = {key: batch[key] for key in ("history_values", "history_mask", "future_known", "site_metadata")}
    h, m, f, s = result.values()
    n = h.shape[0]
    if h.shape != (n, 192, 14) or m.shape != h.shape or f.shape != (n, 24, 9) or s.shape != (n, 6):
        raise ValueError("research input shapes differ from 192x14/24x9/6")
    if m.dtype != torch.bool:
        raise ValueError("history_mask must be boolean")
    preserve = history_missingness == "retain_all_finite_causal_values"
    if not all(bool(torch.isfinite(x).all()) for x in (h if preserve else h[m], f, s)):
        raise ValueError("non-finite observed model inputs")
    result["history_values"] = h if preserve else torch.where(m, h, torch.zeros_like(h))
    return result


def _daylight(batch, n):
    mask = batch["daylight_mask"]
    if mask.shape != (n, 24) or mask.dtype != torch.bool:
        raise ValueError("daylight_mask must be boolean Bx24")
    return mask


def predict_research_neural(model, batch, model_id):
    get_research_spec(model_id)
    if getattr(model, "research_identity", {}).get("model_id") != model_id:
        raise ValueError("model identity differs")
    official = model_id in OFFICIAL_WINDOW_MODELS
    result = model(**_inputs(batch, history_missingness=(
        "retain_all_finite_causal_values" if official else "zero_maskfalse_values")))
    result = dict(result) if isinstance(result, dict) else {"quantiles": result}
    q = result["quantiles"]
    n = batch["history_values"].shape[0]
    if q.shape != (n, 24, 7) or not bool(torch.isfinite(q).all()):
        raise ValueError("quantiles must be finite Bx24x7")
    if official:
        q = torch.sort(q, dim=-1).values
    elif bool((q[..., 1:] < q[..., :-1]).any()):
        raise ValueError("neural quantiles are not ordered")
    result["quantiles"] = torch.where(_daylight(batch, n)[..., None], q.clamp(0., 1.05), torch.zeros_like(q))
    if "ramp_logits" in result:
        if model_id != "rqr_tft" or result["ramp_logits"].shape != (n, 24) or not bool(torch.isfinite(result["ramp_logits"]).all()):
            raise ValueError("invalid diagnostic ramp logits; never a mixture gate")
    return result


def _target_mask(batch):
    y, mask = batch["targets"], batch["target_mask"]
    if y.ndim != 2 or y.shape[1] != 24 or mask.shape != y.shape or mask.dtype != torch.bool:
        raise ValueError("targets and boolean target_mask must be Bx24")
    legal = mask & _daylight(batch, y.shape[0])
    if not bool(torch.isfinite(y[legal]).all()):
        raise ValueError("non-finite legal targets")
    return legal


@dataclass(frozen=True)
class ResearchRampWeight:
    stable_count: int
    ramp_count: int
    pos_weight: float
    split: str = "train"


def fit_research_ramp_weight(batch, *, split):
    if split != "train":
        raise ValueError("ramp class weight requires train only")
    labels = batch["state_labels"]
    legal = _target_mask(batch)
    if labels.shape != legal.shape:
        raise ValueError("state labels shape differs")
    stable = int((legal & (labels == 0)).sum())
    ramp = int((legal & ((labels == 1) | (labels == 2))).sum())
    return ResearchRampWeight(stable, ramp, (stable / ramp) ** .5 if ramp else 1.)


def masked_research_loss(predictions, batch, *, model_id, ramp_training_weight=None,
                         ramp_weighted_quantile_multiplier=6., ramp_loss_weight=1.):
    get_research_spec(model_id)
    outputs = predictions if isinstance(predictions, dict) else {"quantiles": predictions}
    q, y = outputs["quantiles"], batch["targets"]
    legal = _target_mask(batch)
    if q.shape != (*y.shape, 7) or not bool(torch.isfinite(q[legal]).all()):
        raise ValueError("loss requires finite seven quantiles on legal cells")
    count = int(legal.sum())
    if not count:
        return dict(loss=None, skip=True, valid_count=0, objective_version=OBJECTIVE_VERSION)
    error = y[legal, None] - q[legal]
    levels = q.new_tensor(QUANTILES)
    cells = (torch.maximum(levels * error, (levels - 1) * error) * q.new_tensor((.1, .2, .4, .5, .4, .2, .1))).sum(-1)
    result = dict(loss=cells.mean(), daylight_qcrps=cells.mean(), valid_count=count, skip=False, objective_version=OBJECTIVE_VERSION)
    if model_id != "rqr_tft":
        if "ramp_logits" in outputs:
            raise ValueError("unrelated state task is forbidden")
        return result
    if not isinstance(ramp_training_weight, ResearchRampWeight) or ramp_training_weight.split != "train":
        raise ValueError("RQR requires fixed train ramp class weights")
    if not np.isfinite(ramp_training_weight.pos_weight) or ramp_training_weight.pos_weight < 0:
        raise ValueError("invalid train ramp class weight")
    if not np.isfinite(ramp_weighted_quantile_multiplier) or ramp_weighted_quantile_multiplier <= 0 or not np.isfinite(ramp_loss_weight) or ramp_loss_weight < 0:
        raise ValueError("invalid RQR loss coefficients")
    labels, logits = batch["state_labels"], outputs["ramp_logits"]
    if labels.shape != y.shape or logits.shape != y.shape:
        raise ValueError("RQR diagnostic shapes differ")
    labelled = legal & ((labels == 0) | (labels == 1) | (labels == 2))
    ramp = (labels == 1) | (labels == 2)
    forecast = (cells * torch.where(ramp[legal], cells.new_tensor(ramp_weighted_quantile_multiplier), cells.new_tensor(1.))).mean()
    if not bool(torch.isfinite(logits[labelled]).all()):
        raise ValueError("non-finite labelled ramp logits")
    bce = nn.functional.binary_cross_entropy_with_logits(logits[labelled], ramp[labelled].to(logits.dtype),
        pos_weight=logits.new_tensor(ramp_training_weight.pos_weight)) if bool(labelled.any()) else logits[labelled].sum()
    return dict(result, loss=forecast + ramp_loss_weight * bce, forecast_loss=forecast, ramp_bce=bce)


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def predict_research_persistence(model_id, batch, feature_signature):
    get_research_spec(model_id)
    from racepv.baselines.classical import ClearSkyPersistence, SmartPersistence
    history, future, _ = _signature(feature_signature)
    inputs = _inputs(batch)
    h, mask, f = (_numpy(inputs[key]) for key in ("history_values", "history_mask", "future_known"))
    clear = f[..., future.index("clear_sky_power_norm")]
    if model_id == "smart_persistence":
        p, c = history.index("power_norm"), history.index("clear_sky_power_norm")
        point = SmartPersistence().predict(h[..., p], h[..., c], clear, mask[..., p] & mask[..., c])
    elif model_id == "clear_sky_persistence":
        point = ClearSkyPersistence().predict(clear)
    else:
        raise ValueError("persistence requires a point reference identity")
    return np.where(_numpy(_daylight(batch, len(h))), point, 0.)


def research_tree_features(batch):
    """Full history+mask and each lead's deterministic future and static site."""
    x = _inputs(batch)
    h, mask, f, s = (_numpy(x[key]).astype(np.float32) for key in x)
    flat = np.concatenate((h.reshape(len(h), -1), mask.reshape(len(h), -1), s), axis=1)
    return np.concatenate((np.broadcast_to(flat[:, None, :], (len(h), 24, flat.shape[1])), f), axis=-1)


class ResearchTreeForecaster:
    """24x7 quantile estimators or 24 masked RF point estimators; no calibration."""

    def __init__(self, model_id, *, estimator_factory=None, **config):
        spec = get_research_spec(model_id)
        if model_id not in ("quantile_lightgbm", "random_forest_mondrian"):
            raise ValueError("unknown research tree identity")
        self.model_id = model_id
        self.config = dict(spec["default_config"], **config)
        if "objective" in self.config or "alpha" in self.config:
            raise ValueError("quantile objective/alpha cannot be overridden")
        self.estimator_factory, self.models = estimator_factory, []

    def fit(self, batch, *, split):
        if split != "train":
            raise ValueError("tree fit requires train only")
        x, legal = research_tree_features(batch), _numpy(_target_mask(batch))
        y = _numpy(batch["targets"])
        if not legal.any(axis=0).all():
            raise ValueError("every lead requires legal daylight train support")
        factory = self.estimator_factory
        if factory is None:
            try:
                if self.model_id == "quantile_lightgbm":
                    from lightgbm import LGBMRegressor
                    factory = LGBMRegressor
                else:
                    from sklearn.ensemble import RandomForestRegressor
                    factory = RandomForestRegressor
            except ImportError as error:
                raise RuntimeError(f"NOT_READY: {self.model_id}: {error}") from error
        models = []
        for lead in range(24):
            estimators = []
            for quantile in QUANTILES if self.model_id == "quantile_lightgbm" else (None,):
                options = dict(self.config)
                if quantile is not None:
                    options.update(objective="quantile", alpha=quantile)
                estimator = factory(**options)
                estimator.fit(x[legal[:, lead], lead], y[legal[:, lead], lead])
                estimators.append(estimator)
            models.append(estimators)
        self.models = models
        return self

    def predict(self, batch):
        if len(self.models) != 24:
            raise ValueError("tree is not fitted")
        x = research_tree_features(batch)
        raw = np.stack([np.stack([m.predict(x[:, lead]) for m in models], -1)
                        for lead, models in enumerate(self.models)], 1)
        if not np.isfinite(raw).all():
            raise ValueError("non-finite tree predictions")
        result = np.where(_numpy(_daylight(batch, len(x)))[..., None], np.clip(np.sort(raw, axis=-1), 0., 1.05), 0.)
        return result if self.model_id == "quantile_lightgbm" else result[..., 0]


def _empirical_keys(batch):
    import pandas as pd
    times = batch["timestamps"]
    sites = batch["site_ids"]
    if times.ndim != 2 or times.shape[1] != 24 or times.dtype != torch.int64 or sites.shape != (times.shape[0],) or sites.dtype != torch.int64:
        raise ValueError("empirical keys require int64 timestamps Bx24 and site_ids B")
    ns = _numpy(times)
    local = pd.to_datetime(ns.reshape(-1), utc=True).tz_convert("Australia/Darwin")
    return ns, _numpy(sites), np.asarray(local.month).reshape(ns.shape), np.asarray(local.hour).reshape(ns.shape)


class ConditionalEmpiricalQuantiles:
    """Unique physical train observations, site/month/hour -> site/hour -> site.

    Fit uses target_mask (including observed nights); deterministic night-zero
    projection is applied at prediction. Support is unique observations, not
    repeated overlapping forecast windows. No validation/calibration fitting.
    """

    def __init__(self):
        self.support_counts = {}
        self._groups = {}
        self.last_fallbacks = None

    def fit(self, batches, *, split):
        if split != "train":
            raise ValueError("empirical distribution fit requires train only")
        records = {}
        for batch in (batches,) if isinstance(batches, dict) else batches:
            times, sites, months, hours = _empirical_keys(batch)
            y, observed = _numpy(batch["targets"]), _numpy(batch["target_mask"])
            if y.shape != times.shape or observed.shape != times.shape or observed.dtype != bool:
                raise ValueError("empirical targets and boolean masks must match keys")
            if not np.isfinite(y[observed]).all():
                raise ValueError("non-finite observed train target")
            for row, site in enumerate(sites):
                for lead, stamp in enumerate(times[row]):
                    key = int(site), int(stamp)
                    value = float(y[row, lead])
                    record = bool(observed[row, lead]), value, int(months[row, lead]), int(hours[row, lead])
                    if key in records:
                        old = records[key]
                        equal_value = old[1] == value or (np.isnan(old[1]) and np.isnan(value))
                        if old[0] != record[0] or not equal_value:
                            raise ValueError("conflicting target values/masks at one physical site/time")
                    else:
                        records[key] = record
        groups, support = {}, {}
        for (site, _), (observed, value, month, hour) in records.items():
            support.setdefault(site, 0)
            if observed:
                support[site] += 1
                for key in ((site, month, hour), (site, hour), (site,)):
                    groups.setdefault(key, []).append(value)
        if not support or any(count == 0 for count in support.values()):
            raise ValueError("each site requires nonempty train target support")
        self.support_counts = support
        self._groups = {key: (len(values), np.quantile(values, QUANTILES, method="linear")) for key, values in groups.items()}
        return self

    def predict(self, batch):
        times, sites, months, hours = _empirical_keys(batch)
        day = _numpy(_daylight(batch, len(sites)))
        output = np.empty((*times.shape, 7), dtype=np.float64)
        fallbacks = np.empty(times.shape, dtype=object)
        for row, site in enumerate(sites):
            for lead in range(24):
                for key, label in (((int(site), int(months[row, lead]), int(hours[row, lead])), "site_month_hour"),
                                   ((int(site), int(hours[row, lead])), "site_hour"), ((int(site),), "site_global")):
                    group = self._groups.get(key)
                    if group is not None and (group[0] >= 200 or label == "site_global"):
                        output[row, lead], fallbacks[row, lead] = group[1], label
                        break
                else:
                    raise ValueError("no empirical train support for requested site")
        self.last_fallbacks = fallbacks
        return np.where(day[..., None], np.clip(output, 0., 1.05), 0.)
