"""Pinned, nonformal exclusion of explicitly listed invalid timestamp records.

Original bytes remain the source identity. This module reads text fields only;
callers retain the originals and keep this receipt with downstream products.
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import asdict

import numpy as np
import pandas as pd

from racepv.cli.exploratory_source import (
    DOMAIN,
    ExploratoryExtraction,
    _encoded,
    _header,
    _text_rows,
    _timestamp_label,
    source_contract_from_assumptions,
)
from racepv.data import prepared_contract as c
from racepv.data.adapters import BasePVAdapter, ManifestEntry
from racepv.data.scoped_input import _validated_source_and_scope

KIND = "racepv.exploratory-timestamp-quarantine"


def _declaration(blob, expected_sha256, original_file, pins):
    c._sha(expected_sha256, "quarantine SHA256")
    if type(blob) is not bytes or hashlib.sha256(blob).hexdigest() != expected_sha256:
        raise ValueError("quarantine declaration differs from external pin")
    doc = c._strict_json_value(blob)
    c._keys(doc, ("kind", "version", "authorization", "policy", "quoting_mode",
                  "split_assignment", "formal_ready", "sources", *pins), "quarantine")
    for key, value in {"kind": KIND, "version": 1, "policy": "exclude-listed-invalid-only",
                       "quoting_mode": "none", "split_assignment": "unknown",
                       "formal_ready": False}.items():
        c._exact(doc[key], value, "quarantine " + key)
    if not isinstance(doc["authorization"], str) or not doc["authorization"].strip():
        raise ValueError("explicit quarantine authorization reference required")
    for key, value in pins.items():
        c._sha(value, key)
        c._exact(doc[key], value, "quarantine " + key)
    if type(doc["sources"]) is not list or not doc["sources"]:
        raise ValueError("quarantine sources must be a nonempty list")
    paths, match = set(), None
    for item in doc["sources"]:
        c._keys(item, ("original_file", "records"), "quarantine source")
        identity = item["original_file"]
        c._file_identity(identity)
        if identity["path"] in paths:
            raise ValueError("duplicate quarantine source path")
        paths.add(identity["path"])
        if type(item["records"]) is not list or not item["records"]:
            raise ValueError("quarantine records must be a nonempty list")
        coordinates, lines = set(), set()
        for record in item["records"]:
            c._keys(record, ("data_record_1based", "physical_line", "timestamp_text"),
                    "quarantine record")
            position = c._integer(record["data_record_1based"], "record coordinate", minimum=1)
            line = c._integer(record["physical_line"], "physical line", minimum=2)
            if position in coordinates or line in lines:
                raise ValueError("duplicate quarantine record coordinate")
            coordinates.add(position)
            lines.add(line)
            if not isinstance(record["timestamp_text"], str):
                raise TypeError("quarantine timestamp_text must be a string")
            # QUOTE_NONE forbids multiline quoted records: one header plus one line per record.
            if line != position + 1:
                raise ValueError("quarantine physical coordinate differs from literal CSV record")
        if identity["path"] == original_file["path"]:
            c._exact(identity, original_file, "quarantine original identity")
            match = item
    if match is None:
        raise ValueError("original source is absent from quarantine declaration")
    return doc, match["records"]


def extract_quarantined_exploratory_csv_bytes(
    original_csv,
    entry,
    *,
    original_file,
    assumptions_blob,
    expected_assumptions_sha256,
    frozen_split_contract,
    expected_split_plan_sha256,
    selected_split,
    quarantine_blob,
    expected_quarantine_sha256,
    manifest_sha256,
    anomaly_record_sha256,
    acknowledge_full_source_bytes=False,
    quoting_mode="none",
    raw_label_window_inclusive=None,
):
    """Exclude only pinned malformed labels, preserving zero-based original positions."""
    if selected_split not in ("train", "validation"):
        raise ValueError("exploration only permits train/validation")
    if acknowledge_full_source_bytes is not True:
        raise ValueError("full-source-byte acknowledgement required")
    c._exact(quoting_mode, "none", "quarantine quoting mode")
    if type(original_csv) is not bytes:
        raise TypeError("original source must be opaque bytes")
    c._file_identity(original_file)
    c._verify_bytes(original_file, original_csv)
    source = source_contract_from_assumptions(
        assumptions_blob, expected_sha256=expected_assumptions_sha256)
    c._sha(expected_split_plan_sha256, "split plan SHA256")
    c._exact(c.canonical_json_sha256(frozen_split_contract), expected_split_plan_sha256,
             "externally frozen split plan")
    pins = {"assumptions_sha256": expected_assumptions_sha256,
            "split_plan_sha256": expected_split_plan_sha256,
            "manifest_sha256": manifest_sha256, "anomaly_record_sha256": anomaly_record_sha256}
    declaration, records = _declaration(
        quarantine_blob, expected_quarantine_sha256, original_file, pins)
    if not isinstance(entry, ManifestEntry) or entry.dataset != "dkasc":
        raise ValueError("exploratory quarantine requires an explicit DKASC entry")
    c._exact(entry.source_time_contract, source, "entry assumption contract")
    c._exact(list(entry.files), [original_file["path"]], "single original file")
    c._exact(entry.source_file_identities, [original_file], "entry raw identity")
    if entry.analysis_start is not None or entry.analysis_end is not None:
        raise ValueError("no hidden analysis cropping is allowed")
    if dict(entry.read_options) not in ({}, {"float_precision": "round_trip"}):
        raise ValueError("no parser skips or alternate row interpretation allowed")
    raw_bounds = None
    if raw_label_window_inclusive is not None:
        if type(raw_label_window_inclusive) is not list or len(raw_label_window_inclusive) != 2:
            raise ValueError("raw label window requires two timezone-aware strings")
        if any(not isinstance(value, str) for value in raw_label_window_inclusive):
            raise TypeError("raw label window endpoints must be strings")
        raw_bounds = [pd.Timestamp(value) for value in raw_label_window_inclusive]
        if any(pd.isna(value) or value.tzinfo is None for value in raw_bounds):
            raise ValueError("raw label window endpoints must be valid and timezone aware")
        if raw_bounds[0] >= raw_bounds[1]:
            raise ValueError("raw label window must increase")
    required = [entry.timestamp_column, entry.power_column, *entry.columns.values()]
    reader = _text_rows(original_csv, "none")
    header = _header(reader, required)
    column = header.index(entry.timestamp_column)
    excluded = {record["data_record_1based"]: record for record in records}
    timestamps, original_positions, seen = [], [], set()
    source_records = 0
    for source_records, row in enumerate(reader, 1):
        if len(row) <= column:
            raise ValueError("source record has no timestamp")
        label = _timestamp_label(row[column], "none")
        if source_records in excluded:
            record = excluded[source_records]
            c._exact(reader.line_num, record["physical_line"], "quarantined physical line")
            c._exact(row[column], record["timestamp_text"], "quarantined literal timestamp")
            # Match the existing scalar parsing rule; DST/interval errors cannot justify removal.
            try:
                parsed = pd.Timestamp(pd.to_datetime(
                    label, format=entry.timestamp_format or "mixed",
                    dayfirst=entry.dayfirst, errors="raise"))
                invalid = bool(pd.isna(parsed))
            except (ValueError, TypeError, OverflowError):
                invalid = True
            if not invalid:
                raise ValueError("listed quarantine timestamp is valid")
            seen.add(source_records)
            continue
        timestamps.append(label)
        original_positions.append(source_records - 1)
    if seen != set(excluded):
        raise ValueError("listed quarantine record missing from original source")
    if not timestamps:
        raise ValueError("no retained source records")
    source, scope = _validated_source_and_scope(
        entry, frozen_split_contract=frozen_split_contract,
        selected_split=selected_split, source_time_contract=source)
    index = BasePVAdapter._strict_source_index(timestamps, entry, source)
    if not index.is_monotonic_increasing:
        raise ValueError("remaining source timestamps must increase")
    selected = (index.asi8 >= scope["stored_start_ns"]) & (
        index.asi8 < scope["stored_end_exclusive_ns"])
    if raw_bounds is not None:
        # The strict index represents interval starts; bounds are on original raw labels.
        labels = index
        if source["source_timestamp_semantics"] == "interval_end":
            labels = labels + pd.Timedelta(source["interval_duration_ns"], unit="ns")
        selected &= (labels >= raw_bounds[0]) & (labels <= raw_bounds[1])
    positions = np.asarray(original_positions, dtype="<i8")[selected]
    if not len(positions):
        raise ValueError("no selected source records")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    wanted_positions = iter(map(int, positions))
    wanted = next(wanted_positions, None)
    reader = _text_rows(original_csv, "none")
    next(reader)
    for position, row in enumerate(reader):
        if position == wanted:
            if len(row) != len(header):
                raise ValueError("ragged selected source record")
            row[column] = _timestamp_label(row[column], "none")
            writer.writerow(row)
            wanted = next(wanted_positions, None)
    result = output.getvalue().encode("utf-8")
    receipt = {
        "kind": "racepv.exploratory-quarantined-csv-extraction", "version": 1,
        "domain": DOMAIN, "original_file": original_file, "manifest_entry": asdict(entry),
        "quoting_mode": "none", "timestamp_quote_policy": "remove_one_balanced_pair_only",
        **pins, "quarantine_sha256": expected_quarantine_sha256,
        "quarantine_authorization": declaration["authorization"],
        "quarantine_policy": declaration["policy"], "excluded_records": records,
        "excluded_split_assignment": "unknown",
        "warnings": ["Excluded invalid timestamps have unknown split membership."],
        "split_plan": frozen_split_contract, "selected_split": selected_split,
        "raw_label_window_inclusive": raw_label_window_inclusive, "source_time_contract": source,
        "source_records": source_records, "retained_records": len(timestamps),
        "selected_records": len(positions),
        "selected_position_basis": "zero-based-original-data-record",
        "selected_position_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
        "output_sha256": hashlib.sha256(result).hexdigest(), "output_bytes": len(result),
        "full_source_bytes_consumed": True, "all_timestamps_scanned": True,
        "numeric_values_parsed": False, "source_interpretation_verified": False,
        "source_eligible": False, "evidence_eligible": False, "formal_ready": False,
    }
    return ExploratoryExtraction(result, _encoded(receipt))
