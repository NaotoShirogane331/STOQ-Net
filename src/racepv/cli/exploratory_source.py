"""Nonformal research assumptions and timestamp-first train/validation extraction.

This module never grants source qualification and never parses power/weather
numbers. Numeric preparation remains in the existing scoped pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from racepv.data import prepared_contract as c
from racepv.data.adapters import ManifestEntry
from racepv.data.origin_clock import validate_source_time_contract
from racepv.data.scoped_input import scoped_source_positions
from racepv.data.scoped_store import _reject_link_components

DOMAIN = "research_assumption"
KIND = "racepv.exploratory-source-assumptions"


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")


def source_contract_from_assumptions(blob, *, expected_sha256):
    """Validate a pinned declaration, without claiming its assumptions are facts."""
    c._sha(expected_sha256, "assumptions SHA256")
    if type(blob) is not bytes or hashlib.sha256(blob).hexdigest() != expected_sha256:
        raise ValueError("assumptions differ from external pin")
    doc = c._strict_json_value(blob)
    c._keys(
        doc,
        (
            "kind",
            "version",
            "authorization",
            "source_parameters",
            "assumptions",
            "allowed_numeric_splits",
            "formal_ready",
            "source_interpretation_verified",
        ),
        "exploratory assumptions",
    )
    c._exact(doc["kind"], KIND, "assumption kind")
    c._exact(doc["version"], 1, "assumption version")
    c._exact(doc["formal_ready"], False, "nonformal assumptions")
    c._exact(doc["source_interpretation_verified"], False, "unverified interpretation")
    c._exact(doc["allowed_numeric_splits"], ["train", "validation"], "numeric scope")
    if not isinstance(doc["authorization"], str) or not doc["authorization"].strip():
        raise ValueError("explicit user authorization reference required")
    c._keys(
        doc["assumptions"],
        ("timezone", "interval", "power_mapping", "weather_alignment"),
        "assumption statements",
    )
    if any(not isinstance(v, str) or not v.strip() for v in doc["assumptions"].values()):
        raise ValueError("all unresolved assumption statements must be explicit")
    parameters = doc["source_parameters"]
    if type(parameters) is not dict or "evidence" in parameters:
        raise ValueError("assumption parameters cannot supply evidence identity")
    return validate_source_time_contract(
        {
            **parameters,
            "evidence": {
                "kind": DOMAIN,
                "reference": "assumptions.json",
                "sha256": expected_sha256,
            },
        }
    )


def _text_rows(blob, quoting_mode="minimal"):
    if type(blob) is not bytes:
        raise TypeError("source must be opaque bytes")
    if quoting_mode not in ("minimal", "none"):
        raise ValueError("explicit quoting mode must be minimal or none")
    quoting = csv.QUOTE_NONE if quoting_mode == "none" else csv.QUOTE_MINIMAL
    return csv.reader(
        io.StringIO(blob.decode("utf-8-sig"), newline=""), strict=True, quoting=quoting
    )


def _timestamp_label(value, quoting_mode):
    # QUOTE_NONE preserves the export's surrounding timestamp quotes. Remove
    # one complete pair only; damaged/unbalanced/internal quotes stay invalid.
    if quoting_mode == "none" and len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


def _header(reader, required):
    try:
        header = next(reader)
    except StopIteration as error:
        raise ValueError("empty source") from error
    if not header or any(not value for value in header) or len(header) != len(set(header)):
        raise ValueError("source requires unique nonempty headers")
    if not set(required).issubset(header):
        raise ValueError("source missing declared columns")
    return header


def audit_source_bytes(
    blob,
    *,
    original_file,
    timestamp_column,
    timestamp_format,
    acknowledge_full_source_bytes=False,
    quoting_mode="minimal",
):
    """Audit literal timestamp labels only: no timezone/interval inference or numbers.

    Ragged records are counted, never silently removed from a prepared dataset.
    This output is diagnostic, not an extraction or source-qualification receipt.
    """
    if acknowledge_full_source_bytes is not True:
        raise ValueError("full-source-byte acknowledgement required")
    c._file_identity(original_file)
    c._verify_bytes(original_file, blob)
    if not isinstance(timestamp_format, str) or not timestamp_format:
        raise ValueError("explicit timestamp format required")
    reader = _text_rows(blob, quoting_mode)
    header = _header(reader, [timestamp_column])
    column = header.index(timestamp_column)
    timestamps, malformed = [], []
    records = 0
    for records, row in enumerate(reader, 1):
        if len(row) != len(header):
            malformed.append(
                {
                    "data_record_1based": records,
                    "physical_line": reader.line_num,
                    "field_count": len(row),
                }
            )
        timestamps.append(_timestamp_label(row[column], quoting_mode) if len(row) > column else "")
    parsed = pd.to_datetime(timestamps, format=timestamp_format, errors="coerce")
    valid = parsed[~parsed.isna()]
    return {
        "kind": "racepv.exploratory-timestamp-audit",
        "version": 1,
        "original_file": original_file,
        "data_records": records,
        "timestamp_column": timestamp_column,
        "timestamp_format": timestamp_format,
        "quoting_mode": quoting_mode,
        "timestamp_quote_policy": "remove_one_balanced_pair_only"
        if quoting_mode == "none"
        else "csv_dialect",
        "header": header,
        "malformed_records": malformed,
        "invalid_timestamp_records": int(parsed.isna().sum()),
        "duplicate_valid_timestamps": int(valid.duplicated().sum()),
        "valid_timestamps_monotonic": bool(valid.is_monotonic_increasing),
        "minimum_literal_timestamp": None if not len(valid) else str(valid.min()),
        "maximum_literal_timestamp": None if not len(valid) else str(valid.max()),
        "timestamp_timezone_inferred": False,
        "numeric_values_parsed": False,
        "full_source_bytes_consumed": True,
        "formal_ready": False,
    }


@dataclass(frozen=True)
class ExploratoryExtraction:
    csv_bytes: bytes
    receipt_json: bytes


def extract_exploratory_csv_bytes(
    original_csv,
    entry,
    *,
    original_file,
    assumptions_blob,
    expected_assumptions_sha256,
    frozen_split_contract,
    expected_split_plan_sha256,
    selected_split,
    acknowledge_full_source_bytes=False,
    quoting_mode="minimal",
    raw_label_window_inclusive=None,
):
    """Select full split rows before any numeric parser can see their values.

    Unlike the formal extractor this binds an honest research assumption document,
    not fictitious provider registration/review hashes. Selected bad rows fail
    closed; outside-split fields are never interpreted or repaired.
    """
    if selected_split not in ("train", "validation"):
        raise ValueError("exploration only permits train/validation")
    raw_bounds = None
    if raw_label_window_inclusive is not None:
        if type(raw_label_window_inclusive) is not list or len(raw_label_window_inclusive) != 2:
            raise ValueError("raw label window requires two explicit timezone-aware strings")
        if any(not isinstance(value, str) for value in raw_label_window_inclusive):
            raise TypeError("raw label window endpoints must be strings")
        raw_bounds = [pd.Timestamp(value) for value in raw_label_window_inclusive]
        if any(pd.isna(value) or value.tzinfo is None for value in raw_bounds):
            raise ValueError("raw label window endpoints must be valid and timezone aware")
        if raw_bounds[0] >= raw_bounds[1]:
            raise ValueError("raw label window must increase")
    if acknowledge_full_source_bytes is not True:
        raise ValueError("full-source-byte acknowledgement required")
    source = source_contract_from_assumptions(
        assumptions_blob, expected_sha256=expected_assumptions_sha256
    )
    c._sha(expected_split_plan_sha256, "split plan SHA256")
    c._exact(
        c.canonical_json_sha256(frozen_split_contract),
        expected_split_plan_sha256,
        "externally frozen split plan",
    )
    if not isinstance(entry, ManifestEntry) or entry.dataset != "dkasc":
        raise ValueError("exploratory source currently supports explicit DKASC entries only")
    c._exact(entry.source_time_contract, source, "entry assumption contract")
    if entry.analysis_start is not None or entry.analysis_end is not None:
        raise ValueError("no hidden analysis cropping is allowed")
    if dict(entry.read_options) not in ({}, {"float_precision": "round_trip"}):
        raise ValueError("no parser skips or alternate row interpretation allowed")
    c._file_identity(original_file)
    c._verify_bytes(original_file, original_csv)
    c._exact(list(entry.files), [original_file["path"]], "single original file")
    c._exact(entry.source_file_identities, [original_file], "entry raw identity")
    required = [entry.timestamp_column, entry.power_column, *entry.columns.values()]
    reader = _text_rows(original_csv, quoting_mode)
    header = _header(reader, required)
    column = header.index(entry.timestamp_column)
    timestamps = []
    for row in reader:
        if len(row) <= column:
            raise ValueError("source record has no timestamp")
        timestamps.append(_timestamp_label(row[column], quoting_mode))
    if not timestamps:
        raise ValueError("empty source records")
    positions = scoped_source_positions(
        entry,
        timestamps,
        frozen_split_contract=frozen_split_contract,
        selected_split=selected_split,
        source_time_contract=source,
    )
    if raw_bounds is not None and len(positions):
        selected_labels = pd.DatetimeIndex(
            pd.to_datetime(
                [timestamps[int(position)] for position in positions],
                format=entry.timestamp_format or "mixed",
                errors="raise",
            )
        )
        if selected_labels.tz is None:
            selected_labels = selected_labels.tz_localize(
                source["timezone"], ambiguous="raise", nonexistent="raise"
            )
        positions = positions[
            (selected_labels >= raw_bounds[0]) & (selected_labels <= raw_bounds[1])
        ]
    if not len(positions):
        raise ValueError("no selected source records")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    selected_positions = iter(map(int, positions))
    wanted = next(selected_positions, None)
    reader = _text_rows(original_csv, quoting_mode)
    next(reader)
    for position, row in enumerate(reader):
        if wanted == position:
            if len(row) != len(header):
                raise ValueError(
                    "ragged selected source record: explicit quarantine decision required"
                )
            row[column] = _timestamp_label(row[column], quoting_mode)
            writer.writerow(row)
            wanted = next(selected_positions, None)
    result = output.getvalue().encode("utf-8")
    receipt = {
        "kind": "racepv.exploratory-csv-extraction",
        "version": 1,
        "domain": DOMAIN,
        "original_file": original_file,
        "manifest_entry": asdict(entry),
        "quoting_mode": quoting_mode,
        "timestamp_quote_policy": "remove_one_balanced_pair_only"
        if quoting_mode == "none"
        else "csv_dialect",
        "assumptions_sha256": expected_assumptions_sha256,
        "split_plan_sha256": expected_split_plan_sha256,
        "split_plan": frozen_split_contract,
        "selected_split": selected_split,
        "raw_label_window_inclusive": raw_label_window_inclusive,
        "source_time_contract": source,
        "source_records": len(timestamps),
        "selected_records": len(positions),
        "selected_position_sha256": hashlib.sha256(positions.astype("<i8").tobytes()).hexdigest(),
        "output_sha256": hashlib.sha256(result).hexdigest(),
        "output_bytes": len(result),
        "full_source_bytes_consumed": True,
        "all_timestamps_scanned": True,
        "numeric_values_parsed": False,
        "source_interpretation_verified": False,
        "formal_ready": False,
    }
    return ExploratoryExtraction(result, _encoded(receipt))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-bytes", required=True, type=int)
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--timestamp-format", default="%Y-%m-%d %H:%M:%S")
    parser.add_argument("--quoting-mode", choices=("minimal", "none"), default="minimal")
    parser.add_argument("--acknowledge-full-source-bytes", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.acknowledge_full_source_bytes:
        parser.error("explicit full-source-byte acknowledgement required")
    identity = {"path": args.raw.name, "sha256": args.expected_sha256, "bytes": args.expected_bytes}
    c._file_identity(identity)
    output = _reject_link_components(args.output)
    if output.exists():
        raise FileExistsError(output)
    raw = _reject_link_components(args.raw)
    receipt = audit_source_bytes(
        raw.read_bytes(),
        original_file=identity,
        timestamp_column=args.timestamp_column,
        timestamp_format=args.timestamp_format,
        acknowledge_full_source_bytes=True,
        quoting_mode=args.quoting_mode,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(_encoded(receipt))
    print(
        json.dumps(
            {
                "output": str(output),
                "data_records": receipt["data_records"],
                "numeric_values_parsed": False,
                "formal_ready": False,
            }
        )
    )


if __name__ == "__main__":
    main()
