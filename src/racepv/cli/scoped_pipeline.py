"""Pinned split-local CSV preparation; not original-source extraction."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
from pathlib import Path

import pandas as pd

from racepv import io as window_io
from racepv.cli.prepare import FUTURE_COLUMNS, SITE_METADATA_NAMES, _bytes_identity, _json_bytes
from racepv.cli.scoped_acceptance import accept_published_preparation
from racepv.cli.scoped_fit import build_scoped_train_fit, restore_scoped_train_fit
from racepv.cli.scoped_prepare import prepare_scoped_split, scoped_records_sha256
from racepv.data import prepared_contract as contract
from racepv.data.adapters import adapter_for
from racepv.data.causal_history import _POWER_DEPENDENCIES, _WEATHER_FEATURES
from racepv.data.scoped_input import (
    _validated_source_and_scope,
    resample_scoped_source_frame,
    scoped_source_positions,
)
from racepv.data.scoped_prepared import scoped_forecast_population_sha256, validate_v3_schema
from racepv.data.scoped_publication import publish_resolved_scoped_leaf
from racepv.data.scoped_store import _reject_link_components


def _pinned_bytes(blob, expected_sha256, name):
    contract._sha(expected_sha256, name)
    if type(blob) is not bytes or hashlib.sha256(blob).hexdigest() != expected_sha256:
        raise ValueError(name + " bytes do not match external pin")


def _csv_options(entry):
    if dict(entry.read_options) not in ({}, {"float_precision": "round_trip"}):
        raise ValueError("scoped CSV supports only fixed UTF-8/default CSV with round_trip")


def decode_scoped_csv(
    blob,
    entry,
    *,
    expected_sha256,
    frozen_split_contract,
    source_time_contract,
    selected_split,
):
    """Check pinned bytes, CSV structure and complete row scope before numbers.

    All supplied bytes are consumed. Supply an independently extracted split
    payload, never an entire multi-split provider export.
    """
    _pinned_bytes(blob, expected_sha256, "scoped CSV")
    _csv_options(entry)
    if selected_split not in {"train", "validation"}:
        raise ValueError("scoped CSV accepts train/validation only")
    _validated_source_and_scope(
        entry,
        frozen_split_contract=frozen_split_contract,
        selected_split=selected_split,
        source_time_contract=source_time_contract,
    )
    try:
        reader = csv.reader(io.StringIO(blob.decode("utf-8"), newline=""), strict=True)
        header = next(reader)
        if not header or len(set(header)) != len(header) or any(not name for name in header):
            raise ValueError("CSV requires unique nonempty headers")
        required = {entry.timestamp_column, entry.power_column, *entry.columns.values()}
        if not required.issubset(header):
            raise ValueError("CSV missing declared provider columns")
        timestamp_column = header.index(entry.timestamp_column)
        timestamps = []
        for row in reader:
            if len(row) != len(header):
                raise ValueError("CSV row width differs from header")
            timestamps.append(row[timestamp_column])
    except (csv.Error, UnicodeError, StopIteration) as error:
        raise ValueError("invalid UTF-8 scoped CSV") from error
    if not timestamps:
        raise ValueError("empty scoped CSV")
    positions = scoped_source_positions(
        entry,
        timestamps,
        frozen_split_contract=frozen_split_contract,
        selected_split=selected_split,
        source_time_contract=source_time_contract,
    )
    if len(positions) != len(timestamps):
        raise ValueError("CSV includes records outside the selected split")
    frame = pd.read_csv(io.BytesIO(blob), float_precision="round_trip", skip_blank_lines=False)
    if list(frame.columns) != header or len(frame) != len(timestamps):
        raise ValueError("numeric parser changed CSV columns or record count")
    return frame


def _producer_sources():
    root = Path(__file__).parents[1]
    names = (
        "cli/scoped_pipeline.py",
        "cli/prepare.py",
        "cli/scoped_fit.py",
        "cli/scoped_prepare.py",
        "cli/scoped_acceptance.py",
        "data/adapters.py",
        "data/scoped_input.py",
        "data/scoped_prepared.py",
        "data/scoped_store.py",
        "data/scoped_publication.py",
        "data/prepared_contract.py",
        "data/causal_history.py",
        "data/origin_clock.py",
        "data/preprocessing.py",
        "data/dataset.py",
        "data/ramp.py",
    )
    return [_bytes_identity(name, (root / name).read_bytes()) for name in names]


def _schema(entry, candidate, fit_blob, evidence, options):
    identity = candidate.identity
    source = copy.deepcopy(options["source_time_contract"])
    source["evidence"]["reference"] = "evidence/time.txt"
    schema = {
        "schema_kind": contract.SCHEMA_KIND,
        "schema_version": 3,
        "causality_profile": contract.PROFILE,
        "dataset_kind": entry.dataset,
        "simulated": entry.dataset in {"synthetic", "nrel"},
        "history_columns": list(window_io.REGISTERED_HISTORY_FEATURES),
        "future_known_columns": list(FUTURE_COLUMNS),
        "site_metadata_names": list(SITE_METADATA_NAMES),
        "target": "power_norm",
        "state_labels": dict(window_io.REGISTERED_STATE_LABELS),
        "site": {
            "id": options["site_index"],
            "name": entry.site_id,
            "spatial_role": entry.spatial_role,
        },
        "geometry": {"lookback": 192, "horizon": 24, "stride": options["stride"]},
        "clock": {
            "sampling_interval_ns": 900_000_000_000,
            "stored_timezone": "UTC",
            "timestamps": "target_interval_start_ns",
            "issue": "last_history_interval_end",
            "lead": "target_interval_end_minus_issue",
            "day_ids": "utc_days_since_epoch",
            "deterministic_sampling_points": contract.deterministic_sampling_points(),
        },
        "source_time_contract": source,
        "split_plan": identity["split_plan"],
        "split": options["selected_split"],
        "row_scope": identity["row_scope"],
        "causal_rules": {
            "max_gap_steps": 2,
            "gap_right": "strictly_before_first_target",
            "endpoint_scope": "same_split_real_observations",
            "unclosed_daylight": "reject",
            "night": "clear_sky_norm_lt_0.01_source_precision",
            "missing_values": "zero_value_zero_mask",
            "imputed_values": "never_exposed",
            "provenance_flag_mask": 1,
            "power_dependencies": {
                name: list(value) for name, value in _POWER_DEPENDENCIES.items()
            },
            "weather_sources": dict(_WEATHER_FEATURES),
            "deterministic_history": ["clear_sky_power_norm"],
            "provenance_flags": ["power_observed", "power_imputed"],
        },
        "sources": {
            role: _bytes_identity(path, evidence[path])
            for role, path in {
                "time_evidence": "evidence/time.txt",
                "raw_input": "evidence/selected.csv",
                "producer": "evidence/producer.json",
                "site_metadata": "evidence/site.json",
            }.items()
        },
        "fit_states": contract._strict_json_value(fit_blob)["fit_states"],
        "arrays": identity["arrays"],
        "forecast_population_sha256": "",
    }
    if options["selected_split"] == "validation":
        schema["sources"]["frozen_fit_raw_input"] = _bytes_identity(
            "evidence/train.csv", evidence["evidence/train.csv"]
        )
    schema["forecast_population_sha256"] = scoped_forecast_population_sha256(schema)
    return validate_v3_schema(schema)


def prepare_scoped_csv_artifacts(
    output_directory,
    entry,
    *,
    train_csv,
    selected_csv,
    expected_train_csv_sha256,
    expected_selected_csv_sha256,
    time_evidence,
    frozen_split_contract,
    source_time_contract,
    selected_split,
    site_index,
    ramp_method,
    stride,
    allowed_splits,
    frozen_fit_blob=None,
    expected_fit_sha256=None,
    allow_approximate_solar=False,
):
    """Publish one split from explicit byte inputs; never opens original raw files.

    Train builds fit; validation must supply its frozen train fit. Independent
    acceptance then recomputes fit as verification, not parameter selection.
    Original extraction/source qualification remain false in the final receipt.
    """
    output = _reject_link_components(output_directory)
    window_io._v2_check_destination(output)
    if selected_split not in {"train", "validation"}:
        raise ValueError("pipeline accepts train/validation only")
    scope = ("train",) if selected_split == "train" else ("train", "validation")
    contract._exact(allowed_splits, scope, "allowed_splits")
    contract._integer(stride, "stride", minimum=1)
    _csv_options(entry)
    source, _ = _validated_source_and_scope(
        entry,
        frozen_split_contract=frozen_split_contract,
        source_time_contract=source_time_contract,
        selected_split=selected_split,
    )
    _pinned_bytes(time_evidence, source["evidence"]["sha256"], "time evidence")
    _pinned_bytes(train_csv, expected_train_csv_sha256, "train CSV")
    _pinned_bytes(selected_csv, expected_selected_csv_sha256, "selected CSV")
    fit_options = dict(
        frozen_split_contract=frozen_split_contract,
        source_time_contract=source,
        raw_input_sha256=expected_train_csv_sha256,
        site_index=site_index,
        ramp_method=ramp_method,
        allow_approximate_solar=allow_approximate_solar,
    )
    if selected_split == "train":
        if frozen_fit_blob is not None or expected_fit_sha256 is not None:
            raise ValueError("train pipeline builds one fit; do not supply another")
        if train_csv != selected_csv:
            raise ValueError("train selected payload must equal train fit payload")
    else:
        restore_scoped_train_fit(
            frozen_fit_blob, entry, expected_sha256=expected_fit_sha256, **fit_options
        )
    decode_options = dict(frozen_split_contract=frozen_split_contract, source_time_contract=source)
    train = decode_scoped_csv(
        train_csv,
        entry,
        expected_sha256=expected_train_csv_sha256,
        selected_split="train",
        **decode_options,
    )
    selected = (
        train
        if selected_split == "train"
        else decode_scoped_csv(
            selected_csv,
            entry,
            expected_sha256=expected_selected_csv_sha256,
            selected_split=selected_split,
            **decode_options,
        )
    )
    producer = _producer_sources()
    attempt = window_io._v2_attempt(output, directory=True)
    try:
        if selected_split == "train":
            sampled = resample_scoped_source_frame(
                adapter_for(entry, "."),
                entry,
                train,
                selected_split="train",
                **decode_options,
            )
            frozen_fit_blob = build_scoped_train_fit(entry, sampled, **fit_options)
            expected_fit_sha256 = hashlib.sha256(frozen_fit_blob).hexdigest()
        window_io._v2_write_bytes(output / "frozen-fit.json", frozen_fit_blob)
        options = dict(
            **decode_options,
            selected_split=selected_split,
            frozen_fit_blob=frozen_fit_blob,
            expected_fit_sha256=expected_fit_sha256,
            train_raw_input_sha256=expected_train_csv_sha256,
            site_index=site_index,
            ramp_method=ramp_method,
            stride=stride,
            expected_records_sha256=scoped_records_sha256(selected),
            allowed_splits=allowed_splits,
            allow_approximate_solar=allow_approximate_solar,
        )
        candidate = prepare_scoped_split(entry, selected, **options)
        fit = contract._strict_json_value(frozen_fit_blob)
        evidence = {name: text.encode("utf-8") for name, text in fit["evidence"].items()}
        evidence.update(
            {
                "evidence/time.txt": time_evidence,
                "evidence/selected.csv": selected_csv,
                "evidence/producer.json": _json_bytes({"sources": producer}),
                "evidence/site.json": _json_bytes(
                    {
                        "latitude": float(entry.latitude),
                        "longitude": float(entry.longitude),
                        "timezone": entry.timezone,
                        "rated_capacity_kw": entry.capacity_kw,
                        "surface_tilt": entry.surface_tilt,
                        "surface_azimuth": entry.surface_azimuth,
                    }
                ),
            }
        )
        if selected_split == "validation":
            evidence["evidence/train.csv"] = train_csv
        schema = _schema(entry, candidate, frozen_fit_blob, evidence, options)
        schema_sha = contract.canonical_json_sha256(schema)
        publication = publish_resolved_scoped_leaf(
            output / "leaf",
            staging_directory=output / "stage",
            schema=schema,
            expected_schema_sha256=schema_sha,
            causal_arrays=candidate.arrays,
            evidence=evidence,
            allowed_splits=allowed_splits,
        )
        acceptance = accept_published_preparation(
            output / "prepare-acceptance.json",
            candidate,
            entry,
            selected,
            train_source_frame=train,
            expected_train_records_sha256=scoped_records_sha256(train),
            descriptor_path=publication["descriptor_path"],
            expected_descriptor_sha256=publication["descriptor_sha256"],
            expected_read_scope_sha256=publication["read_scope_sha256"],
            expected_schema_sha256=schema_sha,
            expected_preparation_sha256=contract.canonical_json_sha256(candidate.identity),
            **options,
        )
        contract._exact(_producer_sources(), producer, "producer source drift")
        receipt = {
            "kind": "racepv.scoped-csv-preparation",
            "version": 1,
            "selected_split": selected_split,
            "allowed_splits": list(allowed_splits),
            "csv_codec": "utf8-rfc4180-pandas-round-trip-v1",
            "inputs": {
                "train": _bytes_identity(
                    "leaf/evidence/selected.csv"
                    if selected_split == "train"
                    else "leaf/evidence/train.csv",
                    train_csv,
                ),
                "selected": _bytes_identity("leaf/evidence/selected.csv", selected_csv),
            },
            "records": {
                "train": scoped_records_sha256(train),
                "selected": scoped_records_sha256(selected),
            },
            "fit_sha256": expected_fit_sha256,
            "schema_sha256": schema_sha,
            "producer_sources": producer,
            "publication": publication,
            "preparation_acceptance": acceptance,
            "split_payload_binding_verified": True,
            "original_extraction_verified": False,
            "source_interpretation_verified": False,
            "formal_ready": False,
        }
        encoded = window_io._v2_json_bytes(receipt)
        path = output / "pipeline.json"
        window_io._v2_write_bytes(path, encoded)
        return {
            "receipt_path": str(path),
            "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
            "formal_ready": False,
        }
    except BaseException as error:
        window_io._v2_failure(attempt, error)
        raise
