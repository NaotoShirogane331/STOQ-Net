"""Explicit original-byte extraction and nonformal preparation-stage binding.

Supplying bytes consumes the entire source, even when only train/validation rows
are exported. This API is not permission to read a real source. Registration pins
are bound declarations here; reviewed source qualification remains separate.
"""

from __future__ import annotations

import csv
import hashlib
import io
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from racepv import io as window_io
from racepv.cli import scoped_pipeline
from racepv.cli.prepare import _bytes_identity, _json_bytes
from racepv.data import prepared_contract as contract
from racepv.data.adapters import BasePVAdapter, ManifestEntry
from racepv.data.scoped_input import _validated_source_and_scope, scoped_source_positions
from racepv.data.scoped_store import _reject_link_components


@dataclass(frozen=True)
class ScopedExtraction:
    csv_bytes: bytes
    receipt_json: bytes


def _producer_sources():
    root = Path(__file__).parents[1]
    names = (
        "cli/scoped_extraction.py",
        "cli/scoped_pipeline.py",
        "data/adapters.py",
        "data/scoped_input.py",
        "data/origin_clock.py",
        "data/scoped_prepared.py",
        "data/prepared_contract.py",
    )
    return [_bytes_identity(name, (root / name).read_bytes()) for name in names]


def extract_scoped_csv_bytes(
    original_csv,
    entry,
    *,
    original_file,
    expected_original_sha256,
    registration_sha256,
    review_sha256,
    frozen_split_contract,
    source_time_contract,
    selected_split,
    site_index,
    acknowledge_full_source_bytes=False,
):
    """Select every matching source record using existing origin-time semantics.

    This single-file path reads all supplied UTF-8 bytes and all timestamp strings,
    but never numerically parses power/weather columns. No source paths are opened.
    """
    if acknowledge_full_source_bytes is not True:
        raise ValueError("explicit acknowledgement of full original byte consumption required")
    for value, name in (
        (expected_original_sha256, "original"),
        (registration_sha256, "registration"),
        (review_sha256, "review"),
    ):
        contract._sha(value, name + " SHA256")
    contract._file_identity(original_file)
    contract._exact(original_file["sha256"], expected_original_sha256, "original external pin")
    if type(original_csv) is not bytes:
        raise TypeError("original CSV must be explicit bytes")
    contract._verify_bytes(original_file, original_csv)
    if not isinstance(entry, ManifestEntry):
        raise TypeError("expected ManifestEntry")
    contract._exact(list(entry.files), [original_file["path"]], "single source file")
    contract._exact(entry.source_file_identities, [original_file], "manifest source identity")
    if entry.analysis_start is not None or entry.analysis_end is not None:
        raise ValueError("additional analysis cropping is not supported by scoped extraction")
    scoped_pipeline._csv_options(entry)
    if type(selected_split) is not str or selected_split not in ("train", "validation"):
        raise ValueError("extraction supports train/validation only")
    contract._integer(site_index, "site_index")
    source, scope = _validated_source_and_scope(
        entry,
        frozen_split_contract=frozen_split_contract,
        selected_split=selected_split,
        source_time_contract=source_time_contract,
    )
    producer = _producer_sources()
    try:
        reader = csv.reader(io.StringIO(original_csv.decode("utf-8"), newline=""), strict=True)
        header = next(reader)
        if not header or any(not name for name in header) or len(set(header)) != len(header):
            raise ValueError("CSV requires unique nonempty headers")
        if not {entry.timestamp_column, entry.power_column, *entry.columns.values()}.issubset(
            header
        ):
            raise ValueError("CSV missing declared columns")
        timestamp_column = header.index(entry.timestamp_column)
        rows, timestamps = [], []
        for row in reader:
            if len(row) != len(header):
                raise ValueError("CSV record width differs from header")
            rows.append(row)
            timestamps.append(row[timestamp_column])
    except (csv.Error, UnicodeError, StopIteration) as error:
        raise ValueError("invalid UTF-8 CSV extraction input") from error
    if not rows:
        raise ValueError("empty original CSV")
    positions = scoped_source_positions(
        entry,
        timestamps,
        frozen_split_contract=frozen_split_contract,
        selected_split=selected_split,
        source_time_contract=source,
    )
    if len(positions) == 0:
        raise ValueError("source has no records for selected split")
    selected = [rows[int(position)] for position in positions]
    selected_times = BasePVAdapter._strict_source_index(
        [row[timestamp_column] for row in selected], entry, source
    ).asi8
    end_exclusive = int(selected_times.max()) + (source["interval_duration_ns"] or 1)
    if end_exclusive > 2**63 - 1:
        raise ValueError("source interval end overflows signed int64")
    record_ranges = []
    for position in positions:
        position = int(position)
        if record_ranges and record_ranges[-1][1] == position:
            record_ranges[-1][1] += 1
        else:
            record_ranges.append([position, position + 1])
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows([header, *selected])
    csv_bytes = output.getvalue().encode("utf-8")
    contract._exact(_producer_sources(), producer, "extraction producer drift")
    receipt = {
        "kind": "racepv.scoped-csv-extraction",
        "version": 1,
        "domain": source["evidence"]["kind"],
        "original_file": original_file,
        "registration_sha256": registration_sha256,
        "review_sha256": review_sha256,
        "manifest_entry": asdict(entry),
        "site_index": site_index,
        "source_time_contract": source,
        "split_plan": frozen_split_contract,
        "selected_split": selected_split,
        "row_scope": scope,
        "codec": "utf8-rfc4180-field-preserving-lf-v1",
        "source_records": len(rows),
        "selected_records": len(selected),
        "record_ranges": record_ranges,
        "record_coordinates": "zero_based_data_record_half_open_excluding_header",
        "source_interval": {
            "minimum_start_ns": int(selected_times.min()),
            "maximum_start_ns": int(selected_times.max()),
            "maximum_end_exclusive_ns": end_exclusive,
            "instantaneous_point_width_ns": 1 if source["interval_duration_ns"] is None else None,
        },
        "output": _bytes_identity("selected.csv", csv_bytes),
        "producer_sources": producer,
        "runtime": {"python": sys.version, "pandas": pd.__version__, "numpy": np.__version__},
        "read_scope": {
            "full_original_bytes_consumed": True,
            "all_timestamp_fields_scanned": True,
            "numeric_values_parsed": False,
        },
        "source_registration_verified": False,
        "formal_ready": False,
    }
    return ScopedExtraction(csv_bytes, _json_bytes(receipt))


def verify_scoped_extraction_bytes(
    extraction,
    original_csv,
    entry,
    *,
    expected_extraction_sha256,
    **options,
):
    """Verify external receipt pin, strict JSON and actual deterministic replay."""
    contract._sha(expected_extraction_sha256, "expected extraction SHA256")
    if type(extraction) is not ScopedExtraction:
        raise TypeError("expected ScopedExtraction")
    scoped_pipeline._pinned_bytes(
        extraction.receipt_json, expected_extraction_sha256, "extraction receipt"
    )
    contract._strict_json_value(extraction.receipt_json)
    actual = extract_scoped_csv_bytes(original_csv, entry, **options)
    if actual != extraction:
        raise ValueError("extraction receipt/output differs from original-byte replay")
    return actual


def _read_result(root, path, expected_sha256):
    path = _reject_link_components(path)
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("stage result must be contained beneath binding root") from error
    contract._relative_name(relative)
    raw = path.read_bytes()
    scoped_pipeline._pinned_bytes(raw, expected_sha256, "stage result")
    return contract._strict_json_value(raw), _bytes_identity(relative, raw)


def prepare_extracted_csv_artifacts(
    output_directory,
    entry,
    *,
    original_csv,
    train_extraction,
    selected_extraction,
    expected_train_extraction_sha256,
    expected_selected_extraction_sha256,
    extraction_options,
    **pipeline_options,
):
    """Replay both extractions, run existing preparation, publish binding last.

    The new outer stage records computed extraction evidence without rewriting prior
    preparation receipts or granting source/formal qualification. Failed directories
    and extraction receipts are retained, never overwritten or silently resumed.
    """
    output = _reject_link_components(output_directory)
    window_io._v2_check_destination(output)
    for pin in (expected_train_extraction_sha256, expected_selected_extraction_sha256):
        contract._sha(pin, "expected extraction SHA256")
    for field in ("selected_split", "site_index", "frozen_split_contract", "source_time_contract"):
        if field not in extraction_options or field not in pipeline_options:
            raise ValueError("missing extraction/preparation context")
        contract._exact(
            extraction_options[field], pipeline_options[field], "extraction/preparation " + field
        )
    if any(
        field in pipeline_options
        for field in (
            "train_csv",
            "selected_csv",
            "expected_train_csv_sha256",
            "expected_selected_csv_sha256",
        )
    ):
        raise ValueError("prepared input bytes must come exclusively from replayed extraction")
    split = pipeline_options["selected_split"]
    contract._exact(
        pipeline_options["allowed_splits"],
        ("train",) if split == "train" else ("train", "validation"),
        "allowed_splits",
    )
    train = verify_scoped_extraction_bytes(
        train_extraction,
        original_csv,
        entry,
        expected_extraction_sha256=expected_train_extraction_sha256,
        **{**extraction_options, "selected_split": "train"},
    )
    selected = verify_scoped_extraction_bytes(
        selected_extraction,
        original_csv,
        entry,
        expected_extraction_sha256=expected_selected_extraction_sha256,
        **extraction_options,
    )
    attempt = window_io._v2_attempt(output, directory=True)
    try:
        extractions = {}
        for name, item in {"train": train, split: selected}.items():
            relative = f"extraction/{name}.json"
            window_io._v2_write_bytes(output / relative, item.receipt_json)
            extractions[name] = _bytes_identity(relative, item.receipt_json)
        result = scoped_pipeline.prepare_scoped_csv_artifacts(
            output / "prepared",
            entry,
            train_csv=train.csv_bytes,
            selected_csv=selected.csv_bytes,
            expected_train_csv_sha256=hashlib.sha256(train.csv_bytes).hexdigest(),
            expected_selected_csv_sha256=hashlib.sha256(selected.csv_bytes).hexdigest(),
            **pipeline_options,
        )
        pipeline, pipeline_identity = _read_result(
            output, result["receipt_path"], result["receipt_sha256"]
        )
        acceptance_result = pipeline["preparation_acceptance"]
        acceptance, acceptance_identity = _read_result(
            output,
            acceptance_result["acceptance_path"],
            acceptance_result["acceptance_sha256"],
        )
        for name, item in (("train", train), ("selected", selected)):
            actual = pipeline["inputs"][name]
            for field, want in (
                ("sha256", hashlib.sha256(item.csv_bytes).hexdigest()),
                ("bytes", len(item.csv_bytes)),
            ):
                contract._exact(actual[field], want, "extraction versus prepared input " + name)
        for field, want in (
            ("selected_split", split),
            ("site_index", extraction_options["site_index"]),
            ("source_time_contract", extraction_options["source_time_contract"]),
            ("split_plan", extraction_options["frozen_split_contract"]),
        ):
            contract._exact(
                acceptance["preparation"][field], want, "accepted extraction context " + field
            )
        contract._exact(acceptance["schema_sha256"], pipeline["schema_sha256"], "accepted schema")
        contract._exact(
            acceptance["expected_train_records_sha256"],
            pipeline["records"]["train"],
            "accepted train records",
        )
        contract._exact(
            acceptance["preparation"]["records_sha256"],
            pipeline["records"]["selected"],
            "accepted selected records",
        )
        binding = {
            "kind": "racepv.extraction-preparation-binding",
            "version": 1,
            "selected_split": split,
            "site_index": extraction_options["site_index"],
            "registration_sha256": extraction_options["registration_sha256"],
            "review_sha256": extraction_options["review_sha256"],
            "extractions": extractions,
            "pipeline_receipt": pipeline_identity,
            "preparation_acceptance": acceptance_identity,
            "schema_sha256": pipeline["schema_sha256"],
            "extraction_replayed": True,
            "source_registration_verified": False,
            "formal_ready": False,
        }
        blob = _json_bytes(binding)
        path = output / "binding.json"
        window_io._v2_write_bytes(path, blob)
        return {
            "binding_path": str(path),
            "binding_sha256": hashlib.sha256(blob).hexdigest(),
            "formal_ready": False,
        }
    except BaseException as error:
        window_io._v2_failure(attempt, error)
        raise
