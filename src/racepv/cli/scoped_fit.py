"""In-memory train-fit production/restoration, not preparation qualification.

Only the caller's selected training frame is fitted. The resulting bytes need
independent preparation acceptance before a formal consumer may trust their pin.
No raw files, arrays on disk, checkpoints or validation values are opened here.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from racepv.cli.prepare import (
    WEATHER_COLUMNS,
    _bytes_identity,
    _fit_evidence_from_train,
    _json_bytes,
    _validate_zero_parameter_transfer_entry,
)
from racepv.data import prepared_contract as contract
from racepv.data.preprocessing import ResampledPVFrame, TrainOnlyNormalizer
from racepv.data.ramp import RampLabeler
from racepv.data.scoped_input import _selected_labels, _validated_source_and_scope
from racepv.physics import ClearSkyFeatureBuilder, SiteMetadata


@dataclass
class FrozenTrainFit:
    roles: dict
    normalizer: TrainOnlyNormalizer
    physics: ClearSkyFeatureBuilder
    labeler: RampLabeler

    def build_physical(self, index):
        """Apply the frozen physical mapping and reject engine/model drift."""
        physical = self.physics.build(index)
        state = self.roles["clear_sky_power_mapping"]["state"]
        for name in ("solar_engine", "clear_sky_model", "power_model"):
            if physical.attrs.get(name) != state[name]:
                raise ValueError("frozen fit solar engine changed")
        return physical


def _context(entry, plan, source, raw_sha, site_index, ramp_method, approximate):
    source, scope = _validated_source_and_scope(
        entry,
        frozen_split_contract=plan,
        selected_split="train",
        source_time_contract=source,
    )
    contract._sha(raw_sha, "raw_input_sha256")
    contract._integer(site_index, "site_index")
    if type(ramp_method) is not str or ramp_method not in {"q85", "q90", "q95", "fixed0.1"}:
        raise ValueError("unregistered ramp method")
    if type(approximate) is not bool or (approximate and entry.dataset != "synthetic"):
        raise ValueError("approximate solar is restricted to synthetic inputs")
    _validate_zero_parameter_transfer_entry(entry)
    return {
        "kind": "racepv.scoped-train-fit",
        "schema_version": 1,
        "formal_ready": False,
        "manifest_entry": contract._strict_json_value(_json_bytes(asdict(entry))),
        "source_time_contract": source,
        "split_plan": plan,
        "row_scope": scope,
        "raw_input_sha256": raw_sha,
        "site_index": site_index,
    }


def _validate_train(sampled, scope):
    if not isinstance(sampled, ResampledPVFrame):
        raise TypeError("sampled must be a ResampledPVFrame")
    values = sampled.values
    if (
        not isinstance(values, pd.DataFrame)
        or not values.columns.is_unique
        or "power" not in values
    ):
        raise ValueError("training values need unique columns including power")
    if not isinstance(values.index, pd.DatetimeIndex) or values.index.tz is None:
        raise ValueError("training timestamps must be timezone-aware")
    if not np.array_equal(values.index.as_unit("ns").asi8, _selected_labels(scope)):
        raise ValueError("training values must cover exactly the frozen train grid")
    for mask in (sampled.observed_mask, sampled.imputed_mask):
        if (
            not isinstance(mask, pd.DataFrame)
            or not mask.index.equals(values.index)
            or not mask.columns.equals(values.columns)
        ):
            raise ValueError("training masks must match values index and columns")
        if any(dtype != np.dtype(bool) for dtype in mask.dtypes):
            raise TypeError("training masks must be boolean")
    if any(
        not np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.complexfloating)
        for dtype in values.dtypes
    ):
        raise TypeError("training values must be real numeric columns")
    raw = values.to_numpy()
    observed, filled = sampled.observed_mask.to_numpy(), sampled.imputed_mask.to_numpy()
    if (
        np.isinf(raw).any()
        or np.any(observed & filled)
        or not np.array_equal(np.isfinite(raw), observed | filled)
    ):
        raise ValueError("training values and observed/imputed provenance disagree")


def _physics(entry, capacity, approximate):
    return ClearSkyFeatureBuilder(
        SiteMetadata(
            latitude=entry.latitude,
            longitude=entry.longitude,
            timezone=entry.timezone,
            capacity_kw=capacity,
            surface_tilt=entry.surface_tilt,
            surface_azimuth=entry.surface_azimuth,
            name=entry.site_id,
        ),
        allow_approximate_fallback=approximate,
    )


def build_scoped_train_fit(
    entry,
    sampled,
    *,
    frozen_split_contract,
    source_time_contract,
    raw_input_sha256,
    site_index,
    ramp_method,
    allow_approximate_solar=False,
):
    """Fit complete selected train once; return evidence bytes, never an acceptance."""
    context = _context(
        entry,
        frozen_split_contract,
        source_time_contract,
        raw_input_sha256,
        site_index,
        ramp_method,
        allow_approximate_solar,
    )
    _validate_train(sampled, context["row_scope"])
    train = sampled.values
    available = tuple(
        name for name in WEATHER_COLUMNS if name in train and train[name].notna().any()
    )
    normalizer = TrainOnlyNormalizer().fit(
        train,
        feature_columns=available,
        rated_capacity=entry.capacity_kw,
        partition="train",
    )
    physics = _physics(entry, float(normalizer.capacity), allow_approximate_solar)
    if entry.surface_tilt is None or entry.surface_azimuth is None:
        physics.fit_empirical_mapping(train.index, train["power"], partition="train")
    physical = physics.build(train.index)
    fit_power = normalizer.normalize_power(train["power"]).mask(physical["is_night"], 0.0)
    labeler = RampLabeler(method=ramp_method).fit(
        fit_power,
        daylight_mask=physical["is_daylight"],
        observed_mask=sampled.observed_mask["power"],
        partition="train",
    )
    evidence = {}
    roles = _fit_evidence_from_train(
        entry,
        train,
        sampled.observed_mask,
        sampled.imputed_mask,
        frozen_split_contract,
        normalizer,
        physics,
        physical,
        labeler,
        site_index=site_index,
        evidence=evidence,
        raw_sha=raw_input_sha256,
    )
    blob = _json_bytes(
        {
            **context,
            "fit_states": roles,
            "evidence": {name: value.decode("utf-8") for name, value in evidence.items()},
        }
    )
    # Validate the public byte contract without repeating any statistical fit.
    _decode(blob, hashlib.sha256(blob).hexdigest(), context, entry, ramp_method)
    return blob


def _decode(blob, expected_sha, context, entry, ramp_method):
    contract._sha(expected_sha, "expected_sha256")
    if type(blob) is not bytes or hashlib.sha256(blob).hexdigest() != expected_sha:
        raise ValueError("frozen train-fit SHA-256 mismatch")
    document = contract._strict_json_value(blob)
    contract._keys(document, (*context, "fit_states", "evidence"), "scoped train-fit")
    for key, value in context.items():
        contract._exact(document[key], value, "train-fit." + key)
    roles = document["fit_states"]
    schema = {
        "fit_states": roles,
        "split_plan": context["split_plan"],
        "dataset_kind": entry.dataset,
        "site": {
            "id": context["site_index"],
            "name": entry.site_id,
            "spatial_role": entry.spatial_role,
        },
        "sources": {
            "raw_input": {"sha256": context["raw_input_sha256"]},
            "site_metadata": {"path": "site-metadata"},
        },
        "source_time_contract": context["source_time_contract"],
    }
    contract._validate_fit_states(schema)
    contract._exact(roles["ramp_label_threshold"]["method"], ramp_method, "ramp method")
    identities = {}
    input_names = {
        "capacity_normalization": "capacity",
        "weather_standardization": "weather",
        "clear_sky_power_mapping": "clear-sky",
        "ramp_label_threshold": "ramp",
    }
    for role, record in roles.items():
        contract._exact(record["file"]["path"], f"fit/{role}.json", "fit state path")
        identities[record["file"]["path"]] = record["file"]
        for field in ("fit", "training_audit"):
            if record[field] is not None:
                identity = record[field]["input_file"]
                contract._exact(
                    identity["path"], f"fit-input/{input_names[role]}.json", "fit input path"
                )
                identities[identity["path"]] = identity
    contract._keys(document["evidence"], identities, "fit evidence")
    documents = {}
    for name, identity in identities.items():
        text = document["evidence"][name]
        if type(text) is not str:
            raise TypeError("fit evidence must contain JSON strings")
        contract._exact(_bytes_identity(name, text.encode("utf-8")), identity, "fit evidence bytes")
        documents[name] = contract._strict_json_value(text)
    documents["site-metadata"] = {
        "latitude": float(entry.latitude),
        "longitude": float(entry.longitude),
        "timezone": entry.timezone,
        "rated_capacity_kw": entry.capacity_kw,
        "surface_tilt": entry.surface_tilt,
        "surface_azimuth": entry.surface_azimuth,
    }
    contract._validate_state_documents(schema, documents)
    return roles


def restore_scoped_train_fit(
    blob,
    entry,
    *,
    expected_sha256,
    frozen_split_contract,
    source_time_contract,
    raw_input_sha256,
    site_index,
    ramp_method,
    allow_approximate_solar=False,
):
    """Restore pinned parameters; caller must separately establish acceptance.

    No validation frame is accepted, hence neither values nor missing-weather
    availability in validation can select or refit these training parameters.
    """
    context = _context(
        entry,
        frozen_split_contract,
        source_time_contract,
        raw_input_sha256,
        site_index,
        ramp_method,
        allow_approximate_solar,
    )
    roles = _decode(blob, expected_sha256, context, entry, ramp_method)
    capacity = roles["capacity_normalization"]["state"]
    weather = roles["weather_standardization"]["state"]
    normalizer = TrainOnlyNormalizer(
        capacity_quantile=capacity["capacity_quantile"],
        clip_power=tuple(capacity["clip_power"]),
        eps=capacity["eps"],
    )
    normalizer.capacity = capacity["capacity_kw"]
    normalizer.capacity_source = capacity["capacity_source"]
    normalizer.power_column = "power"
    normalizer.feature_columns = tuple(weather["feature_columns"])
    normalizer.feature_mean = weather["feature_mean"].copy()
    normalizer.feature_scale = weather["feature_scale"].copy()
    bounds = context["split_plan"]["splits"]["train"]
    normalizer.training_start = pd.Timestamp(bounds["start_ns"], tz="UTC").isoformat()
    normalizer.training_end = pd.Timestamp(
        bounds["end_exclusive_ns"] - 900_000_000_000, tz="UTC"
    ).isoformat()
    normalizer.fitted = True
    ramp = roles["ramp_label_threshold"]["state"]
    labeler = RampLabeler(method=ramp_method, fixed_threshold=ramp["fixed_threshold"])
    labeler.threshold_ = ramp["threshold"]
    labeler.n_training_ramps_ = ramp["n_training_ramps"]
    physics = _physics(entry, float(normalizer.capacity), allow_approximate_solar)
    clear = roles["clear_sky_power_mapping"]
    if clear["state"]["solar_engine"] == "noaa_approximation" and not allow_approximate_solar:
        raise ValueError("frozen approximate solar requires explicit synthetic permission")
    if clear["method"] == "train_empirical_q95":
        physics.mapper.mapping_quantile = clear["state"]["mapping_quantile"]
        physics.mapper.scale_kw_per_wm2 = clear["state"]["scale_kw_per_wm2"]
        physics.mapper.capacity_kw = float(normalizer.capacity)
        physics.mapper.n_training_points = clear["state"]["n_training_points"]
    return FrozenTrainFit(roles, normalizer, physics, labeler)
