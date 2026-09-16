"""Strict cross-artifact verification for the Q2 oracle-conditioned bundle."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import shutil
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .compact_storage import (
    ACTUAL_LIMIT_BYTES,
    COMPACT_SCHEMA,
    compact_prediction_paths,
    load_prediction_arrays,
    prediction_entries,
)
from .constants import (
    EVIDENCE_METADATA,
    QUANTILES,
    RACE_PV,
    REPORT_LEADS,
    SEEDS,
    VALIDATION_SENSITIVITIES,
)
from .scenario_artifacts import (
    DISCLOSURE,
    MAX_BYTES,
    artifact_root_digest,
    atomic_bytes,
    canonical_bytes,
    load_artifact_index,
    record_verification,
    sha256_file,
    write_json,
)
from .scenario_evaluate import sensitivity_settings
from .scenario_export import (
    FIGURE_KINDS,
    _geometry,
    _render_pdf,
    _render_png,
    _rendered_text,
)
from .scenario_manifest import (
    ACCEPTANCE_RANGES,
    NUMBERED_LAYOUT,
    load_manifest,
    require_stage,
    run_id,
)

REQUIRED_TABLES = (
    "main-results",
    "site-lead",
    "ablations",
    "calibration",
    "significance",
    "transfer",
    "claim-evidence-map",
)
REQUIRED_PAPER = (
    "results.md",
    "captions.md",
    "supplement.md",
    "explanation-zh.md",
    "replacement-guide.md",
)

AUDIT_SAMPLE_FIELDS = (
    "run_id",
    "protocol",
    "split",
    "model",
    "seed",
    "block_id",
    "site_id",
    "origin",
    "lead",
    "quantile",
    "prediction",
    "target",
)
AUDIT_SAMPLE_SAMPLING = "one_stable_hash_origin_per_run_and_selected_block"
AUDIT_SAMPLE_SIDECAR_KEYS = {
    "storage_schema",
    "path",
    "sha256",
    "sampling",
    "row_count",
    "prediction_run_ids",
    "aliases",
}


def validate_acceptance(
    policy: dict[str, Any], report: dict[str, Any] | None
) -> list[str]:
    """Validate a non-smoke acceptance report against immutable declared ranges."""
    errors: list[str] = []
    ranges = policy.get("ranges", {})
    if ranges != ACCEPTANCE_RANGES:
        errors.append("acceptance ranges mismatch")
    if policy.get("mode") == "not_enforced_smoke":
        return errors
    if policy.get("mode") != "enforce_non_smoke_ranges":
        return [*errors, "non-smoke acceptance mode is not enforceable"]
    if not isinstance(report, dict):
        return [*errors, "non-smoke acceptance report missing"]
    checks = report.get("checks", {})
    for name, bounds in ranges.items():
        check = checks.get(name)
        if not isinstance(check, dict):
            errors.append(f"acceptance check missing: {name}")
            continue
        if check.get("minimum") != bounds["minimum"] or check.get("maximum") != bounds[
            "maximum"
        ]:
            errors.append(f"acceptance bounds mismatch: {name}")
            continue
        try:
            value = float(check["value"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"acceptance value invalid: {name}")
            continue
        in_range = np.isfinite(value) and bounds["minimum"] <= value <= bounds["maximum"]
        if not in_range or check.get("passed") is not True:
            errors.append(f"acceptance check out of range: {name}")
    return errors


def _valid_png(path: Path) -> bool:
    content = path.read_bytes()
    if not content.startswith(b"\x89PNG\r\n\x1a\n"):
        return False
    position, width, height, compressed, ended = 8, None, None, bytearray(), False
    while position + 12 <= len(content):
        length = struct.unpack(">I", content[position : position + 4])[0]
        kind = content[position + 4 : position + 8]
        end = position + 12 + length
        if end > len(content):
            return False
        value = content[position + 8 : position + 8 + length]
        expected_crc = struct.unpack(">I", content[position + 8 + length : end])[0]
        if zlib.crc32(kind + value) & 0xFFFFFFFF != expected_crc:
            return False
        position = end
        if kind == b"IHDR":
            if len(value) != 13:
                return False
            width, height = struct.unpack(">II", value[:8])
        elif kind == b"IDAT":
            compressed.extend(value)
        elif kind == b"IEND":
            ended = True
            break
    if not ended or width is None or height is None or position != len(content):
        return False
    try:
        raw = zlib.decompress(compressed)
    except zlib.error:
        return False
    return len(raw) == height * (1 + width * 3)


def _unescaped(value: str, index: int) -> bool:
    slashes = 0
    position = index - 1
    while position >= 0 and value[position] == "\\":
        slashes += 1
        position -= 1
    return slashes % 2 == 0


def _split_tex_cells(line: str) -> list[str]:
    result, start = [], 0
    for index, character in enumerate(line):
        if character == "&" and _unescaped(line, index):
            result.append(line[start:index].strip())
            start = index + 1
    result.append(line[start:].strip())
    return result


def valid_tex_text(text: str) -> bool:
    if text.count(r"\begin{tabular}") != 1 or text.count(r"\end{tabular}") != 1:
        return False
    match = re.match(r"\A\\begin\{tabular\}\{([lcr]+)\}\n", text)
    if not match:
        return False
    column_count = len(match.group(1))
    end_at = text.find(r"\end{tabular}")
    trailing = text[end_at + len(r"\end{tabular}") :]
    if trailing.strip() != "% " + DISCLOSURE:
        return False
    depth = 0
    for index, character in enumerate(text[: end_at + len(r"\end{tabular}")]):
        if character == "{" and _unescaped(text, index):
            depth += 1
        elif character == "}" and _unescaped(text, index):
            depth -= 1
            if depth < 0:
                return False
    if depth != 0:
        return False
    start_at = match.end()
    body = text[start_at:end_at]
    rows = [line for line in body.splitlines() if line and line != r"\hline"]
    if not rows:
        return False
    for row in rows:
        if not row.endswith(r"\\") or row.endswith("\\\\\\"):
            return False
        cells = _split_tex_cells(row[:-2])
        if len(cells) != column_count:
            return False
        for cell in cells:
            for index, character in enumerate(cell):
                if character in "_#%$^~" and _unescaped(cell, index):
                    return False
    return True


def _valid_tex(path: Path) -> bool:
    return valid_tex_text(path.read_text(encoding="utf-8"))


def _compile_tex_tables(paths: list[Path]) -> bool:
    engine = shutil.which("pdflatex")
    if not engine:
        return True
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        inputs = []
        for index, path in enumerate(paths):
            name = f"table-{index}.tex"
            shutil.copy2(path, temporary / name)
            inputs.append(f"\\input{{{name}}}")
        document = temporary / "document.tex"
        document.write_text(
            "\\documentclass{article}\n\\begin{document}\n"
            + "\n\\clearpage\n".join(inputs)
            + "\n\\end{document}\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [engine, "-interaction=nonstopmode", "-halt-on-error", document.name],
            cwd=temporary,
            capture_output=True,
            check=False,
        )
        return completed.returncode == 0 and (temporary / "document.pdf").is_file()


def _valid_pdf(path: Path) -> bool:
    if not path.read_bytes().startswith(b"%PDF-"):
        return False
    pdfinfo = shutil.which("pdfinfo")
    if not pdfinfo:
        return True
    return subprocess.run(
        [pdfinfo, str(path)], capture_output=True, check=False
    ).returncode == 0


def _verify_dag(
    root: Path,
    errors: list[str],
    *,
    required_stages: tuple[str, ...] = (
        "prepare",
        "generate",
        "evaluate",
        "aggregate",
        "export",
    ),
) -> tuple[dict[str, Any], str]:
    path = root / "artifact-index.json"
    if not path.is_file():
        errors.append("missing artifact DAG index")
        return {}, ""
    try:
        index = load_artifact_index(root)
        if index.get("schema") != "append_only_artifact_dag_v2":
            errors.append("artifact DAG schema mismatch")
        artifacts = index.get("artifacts", {})
        for identifier, entry in artifacts.items():
            if entry.get("artifact_id") != identifier:
                errors.append(f"artifact DAG identity mismatch: {identifier}")
            candidate = root / entry.get("path", "")
            if not candidate.is_file() or sha256_file(candidate) != entry.get("sha256"):
                errors.append(f"artifact DAG hash mismatch: {entry.get('path', identifier)}")
            if entry.get("evidence_metadata") != dict(EVIDENCE_METADATA):
                errors.append(f"artifact DAG evidence metadata mismatch: {identifier}")
            for source in entry.get("source_artifact_ids", []):
                if source not in artifacts:
                    errors.append(f"artifact DAG dangling edge: {identifier} -> {source}")
        stage_order = [item.get("stage") for item in index.get("stage_runs", [])]
        positions = [
            stage_order.index(stage) if stage in stage_order else -1
            for stage in required_stages
        ]
        if any(position < 0 for position in positions) or positions != sorted(positions):
            errors.append("artifact DAG stage ordering failure")
        for run in index.get("stage_runs", []):
            output_ids = run.get("output_artifact_ids", [])
            if any(identifier not in artifacts for identifier in output_ids):
                errors.append(f"artifact DAG stage output missing: {run.get('stage')}")
                continue
            digest = hashlib.sha256(
                canonical_bytes({identifier: artifacts[identifier] for identifier in output_ids})
            ).hexdigest()
            if digest != run.get("stage_digest"):
                errors.append(f"artifact DAG stage digest mismatch: {run.get('stage')}")
        return index, artifact_root_digest(index)
    except Exception as error:
        errors.append(f"unreadable artifact DAG: {error}")
        return {}, ""


def _verify_manifest(
    root: Path,
    manifest: dict[str, Any],
    errors: list[str],
    *,
    verify_acceptance_report: bool = True,
) -> None:
    for key, expected in EVIDENCE_METADATA.items():
        if manifest.get(key) != expected:
            errors.append(f"manifest {key} mismatch")
    cohort_path = root / manifest.get("cohort_index_path", "")
    if not cohort_path.is_file() or sha256_file(cohort_path) != manifest.get("cohort_index_sha256"):
        errors.append("cohort index hash mismatch")
        return
    if not manifest.get("smoke"):
        prepared = Path(manifest.get("prepared_cohort_manifest", ""))
        if not prepared.is_file() or sha256_file(prepared) != manifest.get(
            "prepared_cohort_manifest_sha256"
        ):
            errors.append("prepared cohort manifest hash mismatch")
    index = json.loads(cohort_path.read_text(encoding="utf-8"))
    roles = manifest.get("source_role_map", {})
    seen: list[tuple[str, dict[str, Any]]] = []
    for split, cohort in index.get("cohorts", {}).items():
        if cohort.get("cohort_sha256") != manifest.get("cohorts", {}).get(split, {}).get(
            "cohort_sha256"
        ):
            errors.append(f"cohort prepared hash mismatch: {split}")
        expected_role = roles.get(split)
        for block in cohort.get("blocks", []):
            if block.get("split") != split or block.get("spatial_role") != expected_role:
                errors.append(f"cohort role/leakage mismatch: {split}/{block.get('block_id')}")
            for other_split, other in seen:
                same_raw = bool(set(block.get("raw_hashes", [])) & set(other.get("raw_hashes", [])))
                overlap = block.get("start") <= other.get("end") and other.get("start") <= block.get(
                    "end"
                )
                if split != other_split and same_raw and overlap:
                    errors.append("cohort raw interval leakage across splits")
            seen.append((split, block))
    policy = manifest.get("acceptance_policy", {})
    expected_mode = "not_enforced_smoke" if manifest.get("smoke") else "enforce_non_smoke_ranges"
    if policy.get("mode") != expected_mode or not policy.get("reason"):
        errors.append("smoke/non-smoke acceptance policy mismatch")
    if verify_acceptance_report:
        acceptance_path = root / "acceptance.json"
        acceptance = (
            json.loads(acceptance_path.read_text(encoding="utf-8"))
            if acceptance_path.is_file()
            else None
        )
        errors.extend(validate_acceptance(policy, acceptance))
    estimate = manifest.get("storage_estimate", {})
    if estimate.get("total_bytes") != manifest.get("storage_estimate_bytes"):
        errors.append("storage estimate detail mismatch")
    estimate_limit = (
        manifest.get("prediction_storage", {}).get("preflight_limit_bytes", MAX_BYTES)
    )
    if manifest.get("storage_estimate_bytes", estimate_limit + 1) > estimate_limit:
        errors.append("storage estimate exceeds size cap")
    if manifest.get("numbered_layout") != NUMBERED_LAYOUT or not (root / "README.md").is_file():
        errors.append("numbered placement contract mismatch")
    for directory in NUMBERED_LAYOUT.values():
        if not (root / directory / "README.md").is_file():
            errors.append(f"numbered placement directory missing: {directory}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(centered)
    return values / values.sum(axis=-1, keepdims=True)


def _strict_artifact(
    root: Path, path: Path, label: str, errors: list[str]
) -> bool:
    """Require a regular, non-symlinked artifact contained by ``root``."""

    try:
        path.relative_to(root)
    except ValueError:
        errors.append(f"{label} artifact containment invariant failure")
        return False
    cursor = path
    while cursor != root:
        if cursor.is_symlink():
            errors.append(f"{label} artifact symlink invariant failure")
            return False
        cursor = cursor.parent
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        errors.append(f"{label} artifact missing invariant failure")
        return False
    if not resolved.is_relative_to(root.resolve()) or not path.is_file():
        errors.append(f"{label} artifact containment invariant failure")
        return False
    return True


def _verify_shared_cohorts(
    root: Path, manifest: dict[str, Any], errors: list[str]
) -> None:
    expected_arrays = {
        "block_codes",
        "block_dictionary",
        "day_ids",
        "daylight",
        "previous_targets",
        "site_codes",
        "site_dictionary",
        "source_indices",
        "targets",
        "timestamps_ns",
    }
    cohort_index = json.loads((root / manifest["cohort_index_path"]).read_text())
    shared = manifest.get("shared_cohorts", {})
    if set(shared) != {"validation", "calibration", "test"}:
        errors.append("shared cohort split invariant failure")
        return
    for split, declared in shared.items():
        try:
            path = root / declared["path"]
            sidecar_path = path.with_suffix(".json")
            if not _strict_artifact(root, path, f"shared cohort {split}", errors):
                continue
            if not _strict_artifact(
                root,
                sidecar_path,
                f"shared cohort sidecar {split}",
                errors,
            ):
                continue
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if sidecar != declared or sha256_file(path) != declared.get("sha256"):
                errors.append(f"shared cohort hash mismatch: {split}")
                continue
            with np.load(path, allow_pickle=False) as source:
                arrays = {name: source[name] for name in source.files}
            if set(arrays) != expected_arrays or set(declared.get("arrays", [])) != expected_arrays:
                errors.append(f"shared cohort array invariant failure: {split}")
                continue
            targets = arrays["targets"]
            previous = arrays["previous_targets"]
            daylight = arrays["daylight"]
            shape = tuple(declared.get("shape", ()))
            if (
                targets.dtype != np.float32
                or previous.dtype != np.float32
                or daylight.dtype.kind != "b"
                or targets.shape != shape
                or previous.shape != shape
                or daylight.shape != shape
                or arrays["site_codes"].shape != shape
                or arrays["timestamps_ns"].shape != shape
                or arrays["source_indices"].shape != shape
                or arrays["day_ids"].shape != shape
                or arrays["block_codes"].shape != (shape[0],)
            ):
                errors.append(f"shared cohort dtype/shape invariant failure: {split}")
            if (
                not np.all(np.isfinite(targets))
                or not np.all(np.isfinite(previous))
                or np.any((targets < 0) | (targets > 1.05))
                or np.any(targets[~daylight] != 0)
            ):
                errors.append(f"shared cohort physics invariant failure: {split}")
            site_codes = arrays["site_codes"]
            block_codes = arrays["block_codes"]
            site_dictionary = arrays["site_dictionary"]
            block_dictionary = arrays["block_dictionary"]
            if (
                site_codes.dtype.kind not in "iu"
                or block_codes.dtype.kind not in "iu"
                or site_dictionary.dtype.kind not in "US"
                or block_dictionary.dtype.kind not in "US"
                or (site_codes.size and int(site_codes.max()) >= len(site_dictionary))
                or (block_codes.size and int(block_codes.max()) >= len(block_dictionary))
            ):
                errors.append(f"shared cohort dictionary invariant failure: {split}")
                continue
            expected = cohort_index.get("cohorts", {}).get(split, {})
            actual_sites = set(map(str, site_dictionary[site_codes].reshape(-1)))
            actual_blocks = set(map(str, block_dictionary[block_codes].reshape(-1)))
            if actual_sites != set(map(str, expected.get("site_ids", []))) or actual_blocks != {
                str(block.get("block_id")) for block in expected.get("blocks", [])
            }:
                errors.append(f"shared cohort identity invariant failure: {split}")
        except Exception as error:
            errors.append(f"shared cohort hash/invariant unreadable: {split}: {error}")


def _verify_prediction(
    root: Path,
    archive_path: Path,
    expected_ids: set[str],
    seen: set[str],
    errors: list[str],
) -> None:
    try:
        sidecar_path = archive_path.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("storage_schema") == COMPACT_SCHEMA:
            identity = sidecar.get("run_id")
            if identity not in expected_ids or identity in seen:
                errors.append(f"invalid or duplicate run identity: {archive_path.name}")
            seen.add(identity)
            if any(sidecar.get(key) != value for key, value in EVIDENCE_METADATA.items()):
                errors.append(f"prediction evidence metadata mismatch: {archive_path.name}")
            arrays = load_prediction_arrays(root, sidecar)
            persisted = {
                name
                for name in arrays
                if name in {"quantiles", "state_probabilities", "expert_quantiles", "router_logits"}
            }
            expected_persisted = {"quantiles", "state_probabilities"}
            if sidecar.get("model") == RACE_PV.name:
                expected_persisted |= {"expert_quantiles", "router_logits"}
            if persisted != expected_persisted or set(sidecar.get("arrays", [])) != expected_persisted:
                errors.append(f"compact array/key contract failure: {archive_path.name}")
                return
            quantiles = arrays["quantiles"]
            probabilities = arrays["state_probabilities"]
            targets = arrays["targets"]
            previous_targets = arrays["previous_targets"]
            daylight = arrays["daylight"]
            labels = arrays["state_labels"]
            ramps = arrays["ramp_scores"]
            shape = targets.shape
            if (
                quantiles.dtype != np.float32
                or probabilities.dtype != np.float32
                or targets.dtype != np.float64
                or previous_targets.dtype != np.float64
                or daylight.dtype.kind != "b"
                or quantiles.shape != (*shape, len(QUANTILES))
                or probabilities.shape != (*shape, 3)
                or previous_targets.shape != shape
                or labels.shape != shape
                or ramps.shape != shape
            ):
                errors.append(f"compact dtype/shape failure: {archive_path.name}")
            if (
                not np.all(np.isfinite(quantiles))
                or not np.all(np.isfinite(probabilities))
                or not np.all(np.isfinite(targets))
                or not np.all(np.isfinite(previous_targets))
                or np.any(np.diff(quantiles, axis=-1) < 0)
                or np.any((quantiles < 0) | (quantiles > 1.05))
                or np.any((targets < 0) | (targets > 1.05))
                or np.any(probabilities < 0)
                or np.any(probabilities > 1)
                or not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-6)
                or not np.allclose(ramps, probabilities[..., 1:].sum(axis=-1))
            ):
                errors.append(f"compact prediction invariant failure: {archive_path.name}")
            if (
                np.any(targets[~daylight] != 0)
                or np.any(quantiles[~daylight] != 0)
                or np.any(labels[~daylight] != -100)
                or np.any(labels[daylight] == -100)
                or not np.allclose(
                    probabilities[~daylight],
                    np.array([1.0, 0.0, 0.0]),
                    atol=1e-7,
                )
            ):
                errors.append(f"compact night invariant failure: {archive_path.name}")
            if sidecar.get("model") == RACE_PV.name:
                experts = arrays["expert_quantiles"]
                logits = arrays["router_logits"]
                if (
                    experts.dtype != np.float32
                    or logits.dtype != np.float32
                    or experts.shape != (*shape, 3, len(QUANTILES))
                    or logits.shape != (*shape, 3)
                ):
                    errors.append(f"compact RaCE-PV dtype failure: {archive_path.name}")
                if (
                    not np.all(np.isfinite(experts))
                    or not np.all(np.isfinite(logits))
                    or np.any((experts < 0) | (experts > 1.05))
                    or np.any(np.diff(experts, axis=-1) < 0)
                    or np.any(experts[~daylight] != 0)
                ):
                    errors.append(
                        f"compact RaCE-PV invariant failure: {archive_path.name}"
                    )
                if not np.allclose(_softmax(logits), probabilities, atol=2e-7):
                    errors.append(f"compact RaCE-PV softmax failure: {archive_path.name}")
                if not np.allclose(
                    quantiles,
                    np.einsum("nhe,nheq->nhq", probabilities, experts),
                    atol=2e-7,
                ):
                    errors.append(f"compact RaCE-PV fusion failure: {archive_path.name}")
            audit_path = root / sidecar.get("audit_sample_path", "")
            if not audit_path.is_file() or sha256_file(audit_path) != sidecar.get(
                "audit_sample_sha256"
            ):
                errors.append(f"compact audit reference mismatch: {archive_path.name}")
            return
        csv_path = archive_path.with_suffix(".csv.gz")
        identity = sidecar.get("run_id")
        if identity not in expected_ids or identity in seen:
            errors.append(f"invalid or duplicate run identity: {archive_path.name}")
        seen.add(identity)
        if any(sidecar.get(key) != value for key, value in EVIDENCE_METADATA.items()):
            errors.append(f"prediction evidence metadata mismatch: {archive_path.name}")
        if sidecar.get("content_sha256") != sha256_file(archive_path):
            errors.append(f"hash mismatch: {archive_path.name}")
        if not csv_path.is_file() or sidecar.get("long_form_sha256") != sha256_file(csv_path):
            errors.append(f"CSV hash mismatch: {archive_path.name}")
        with np.load(archive_path, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        base_keys = {
            "quantiles",
            "targets",
            "daylight",
            "site_ids",
            "state_probabilities",
            "state_labels",
            "ramp_scores",
            "shared_weather",
        }
        moe_keys = {"expert_quantiles", "router_logits", "gates"}
        expected_keys = base_keys | (moe_keys if sidecar.get("model") == RACE_PV.name else set())
        if set(arrays) != expected_keys or set(sidecar.get("arrays", [])) != expected_keys:
            errors.append(f"required arrays/key contract failure: {archive_path.name}")
            return
        quantiles = arrays["quantiles"]
        targets = arrays["targets"]
        daylight = arrays["daylight"]
        sites = arrays["site_ids"]
        probabilities = arrays["state_probabilities"]
        labels = arrays["state_labels"]
        ramps = arrays["ramp_scores"]
        shared_weather = arrays["shared_weather"]
        shape = targets.shape
        if (
            quantiles.shape != (*shape, len(QUANTILES))
            or targets.ndim != 2
            or daylight.shape != shape
            or sites.shape != shape
            or probabilities.shape != (*shape, 3)
            or labels.shape != shape
            or ramps.shape != shape
            or shared_weather.shape != shape
        ):
            errors.append(f"array shape failure: {archive_path.name}")
        if daylight.dtype.kind != "b" or labels.dtype.kind not in "iu" or sites.dtype.kind not in "US":
            errors.append(f"array dtype failure: {archive_path.name}")
        numeric = (quantiles, targets, probabilities, ramps, shared_weather)
        if any(array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)) for array in numeric):
            errors.append(f"array finiteness/dtype failure: {archive_path.name}")
        if np.any((targets < 0) | (targets > 1.05)) or np.any((quantiles < 0) | (quantiles > 1.05)):
            errors.append(f"physical bound failure: {archive_path.name}")
        if np.any(np.diff(quantiles, axis=-1) < 0):
            errors.append(f"quantile ordering failure: {archive_path.name}")
        if set(np.unique(labels).tolist()) - {-100, 0, 1, 2}:
            errors.append(f"state label set failure: {archive_path.name}")
        if np.any(labels[~daylight] != -100) or np.any(labels[daylight] == -100):
            errors.append(f"night label invariant failure: {archive_path.name}")
        if np.any(quantiles[~daylight] != 0) or np.any(ramps[~daylight] != 0):
            errors.append(f"night forecast invariant failure: {archive_path.name}")
        if (
            np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-12)
            or np.any((ramps < 0) | (ramps > 1))
            or not np.allclose(ramps, probabilities[..., 1:].sum(axis=-1), atol=1e-12)
        ):
            errors.append(f"probability/ramp simplex failure: {archive_path.name}")
        if not np.all(probabilities[~daylight] == np.array([1.0, 0.0, 0.0])):
            errors.append(f"night probability invariant failure: {archive_path.name}")
        if sidecar.get("model") == RACE_PV.name:
            experts, logits, gates = (
                arrays["expert_quantiles"],
                arrays["router_logits"],
                arrays["gates"],
            )
            if experts.shape != (*shape, 3, len(QUANTILES)) or logits.shape != (*shape, 3) or gates.shape != (*shape, 3):
                errors.append(f"RaCE-PV MoE shape failure: {archive_path.name}")
            if any(not np.all(np.isfinite(array)) for array in (experts, logits, gates)):
                errors.append(f"RaCE-PV MoE finiteness failure: {archive_path.name}")
            if np.any((experts < 0) | (experts > 1.05)) or np.any(np.diff(experts, axis=-1) < 0):
                errors.append(f"RaCE-PV expert bound/order failure: {archive_path.name}")
            if np.any(gates < 0) or np.any(gates > 1) or not np.allclose(gates.sum(axis=-1), 1, atol=1e-12):
                errors.append(f"RaCE-PV gate simplex failure: {archive_path.name}")
            if not np.allclose(_softmax(logits), probabilities, atol=1e-12) or not np.array_equal(gates, probabilities):
                errors.append(f"RaCE-PV softmax/gate identity failure: {archive_path.name}")
            if not np.allclose(quantiles, np.einsum("nhe,nheq->nhq", gates, experts), atol=1e-12):
                errors.append(f"RaCE-PV fusion identity failure: {archive_path.name}")
            if np.any(experts[~daylight] != 0) or not np.all(gates[~daylight] == np.array([1.0, 0.0, 0.0])):
                errors.append(f"RaCE-PV night MoE invariant failure: {archive_path.name}")
        with gzip.open(csv_path, "rt", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected_coordinates = {
            (origin, lead + 1, format(level, ".12g"))
            for origin in range(shape[0])
            for lead in range(shape[1])
            for level in QUANTILES
        }
        actual_coordinates: set[tuple[int, int, str]] = set()
        for row in rows:
            coordinate = (int(row["origin"]), int(row["lead"]), row["quantile"])
            if coordinate in actual_coordinates:
                errors.append(f"CSV coordinates duplicate: {archive_path.name}")
                break
            actual_coordinates.add(coordinate)
            origin, one_lead, level = coordinate
            lead = one_lead - 1
            if coordinate not in expected_coordinates:
                errors.append(f"CSV coordinates invalid: {archive_path.name}")
                break
            quantile_index = [format(value, ".12g") for value in QUANTILES].index(level)
            if (
                row["run_id"] != identity
                or row["split"] != sidecar.get("split")
                or row["site_id"] != str(sites[origin, lead])
                or row["prediction"] != format(float(quantiles[origin, lead, quantile_index]), ".12g")
                or row["target"] != format(float(targets[origin, lead]), ".12g")
            ):
                errors.append(f"CSV content/split/site mismatch: {archive_path.name}")
                break
        if actual_coordinates != expected_coordinates or len(rows) != len(expected_coordinates):
            errors.append(f"CSV coordinates incomplete: {archive_path.name}")
    except Exception as error:
        errors.append(f"unreadable prediction {archive_path.name}: {error}")


def _verify_aliases(
    root: Path, expected_ids: set[str], seen: set[str], errors: list[str]
) -> None:
    runs_root = root / "predictions" / "runs"
    paths = (
        sorted(runs_root.rglob("*.alias.json"))
        if runs_root.is_dir()
        else sorted((root / "predictions").glob("*.alias.json"))
    )
    for path in paths:
        try:
            alias = json.loads(path.read_text(encoding="utf-8"))
            identity = alias["run_id"]
            if identity not in expected_ids or identity in seen:
                errors.append(f"invalid alias identity: {path.name}")
            seen.add(identity)
            base = root / alias["alias_of_path"]
            if (
                not base.is_file()
                or alias.get("alias_of_run_id") not in expected_ids
                or alias.get("model") not in {item.name for item in VALIDATION_SENSITIVITIES}
                or alias.get("split") != "validation"
                or any(alias.get(key) != value for key, value in EVIDENCE_METADATA.items())
            ):
                errors.append(f"invalid alias source/evidence: {path.name}")
        except Exception as error:
            errors.append(f"unreadable alias {path.name}: {error}")


def _verify_compact_run_bindings(
    root: Path,
    manifest: dict[str, Any],
    archives: list[Path],
    sidecars: list[Path],
    aliases: list[Path],
    errors: list[str],
) -> None:
    """Bind every compact path and reference to its manifest identity tuple."""

    protocol = str(manifest.get("protocols", [""])[0])
    sensitivity_names = {profile.name for profile in VALIDATION_SENSITIVITIES}
    identities = manifest.get("run_identities", [])
    if not isinstance(identities, list) or any(
        not isinstance(identity, dict) for identity in identities
    ):
        errors.append("manifest run identity structure invariant failure")
        return
    expected_physical: dict[str, Path] = {}
    expected_aliases: dict[str, Path] = {}
    rows: dict[str, dict[str, Any]] = {}
    for identity in identities:
        try:
            identity_protocol = str(identity["protocol"])
            model = str(identity["model"])
            seed = int(identity["seed"])
            split = str(identity["split"])
            identifier = str(identity["run_id"])
        except (KeyError, TypeError, ValueError):
            errors.append("manifest run identity tuple invariant failure")
            continue
        if (
            identity_protocol != protocol
            or identifier != run_id(protocol, model, seed, split)
            or identifier in rows
        ):
            errors.append(f"manifest run identity tuple invariant failure: {identifier}")
            continue
        rows[identifier] = identity
        archive, _sidecar = compact_prediction_paths(root, protocol, model, seed, split)
        if model in sensitivity_names:
            expected_aliases[identifier] = archive.with_suffix(".alias.json")
        else:
            expected_physical[identifier] = archive
    if set(archives) != set(expected_physical.values()) or set(sidecars) != {
        path.with_suffix(".json") for path in expected_physical.values()
    }:
        errors.append("compact archive/sidecar manifest path invariant failure")
    if set(aliases) != set(expected_aliases.values()):
        errors.append("compact alias manifest path invariant failure")

    root_resolved = root.resolve()

    def exact_reference(value: Any, expected: Path, label: str) -> bool:
        if not isinstance(value, str):
            errors.append(f"{label} path invariant failure")
            return False
        relative = Path(value)
        expected_relative = expected.relative_to(root)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != expected_relative.as_posix()
        ):
            errors.append(f"{label} path traversal/identity invariant failure")
            return False
        try:
            resolved = (root / relative).resolve(strict=True)
        except OSError:
            errors.append(f"{label} path missing invariant failure")
            return False
        if (
            not resolved.is_relative_to(root_resolved)
            or resolved != expected.resolve()
            or not _strict_artifact(root, expected, label, errors)
        ):
            errors.append(f"{label} path symlink escape invariant failure")
            return False
        return True

    for identifier, archive in expected_physical.items():
        sidecar_path = archive.with_suffix(".json")
        try:
            _strict_artifact(root, archive, f"archive {identifier}", errors)
            _strict_artifact(root, sidecar_path, f"sidecar {identifier}", errors)
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            row = rows[identifier]
            split = str(row["split"])
            cohort = manifest.get("shared_cohorts", {}).get(split, {})
            if (
                sidecar.get("run_id") != identifier
                or sidecar.get("protocol") != protocol
                or sidecar.get("model") != row["model"]
                or sidecar.get("seed") != row["seed"]
                or sidecar.get("split") != split
                or sidecar.get("storage_schema") != COMPACT_SCHEMA
                or sidecar.get("content_sha256") != sha256_file(archive)
            ):
                errors.append(f"compact sidecar identity/hash invariant failure: {identifier}")
            exact_reference(sidecar.get("path"), archive, f"sidecar {identifier}")
            cohort_path = root / str(cohort.get("path", ""))
            if (
                sidecar.get("cohort_path") != cohort.get("path")
                or sidecar.get("cohort_sha256") != cohort.get("sha256")
                or not exact_reference(
                    sidecar.get("cohort_path"),
                    cohort_path,
                    f"sidecar cohort {identifier}",
                )
            ):
                errors.append(f"compact sidecar cohort hash invariant failure: {identifier}")
            audit_path = root / "predictions/audit-sample.csv.gz"
            if not exact_reference(
                sidecar.get("audit_sample_path"),
                audit_path,
                f"sidecar audit {identifier}",
            ):
                errors.append(f"compact sidecar audit path invariant failure: {identifier}")
        except Exception as error:
            errors.append(f"compact sidecar binding unreadable: {identifier}: {error}")

    for identifier, alias_path in expected_aliases.items():
        try:
            _strict_artifact(root, alias_path, f"alias {identifier}", errors)
            alias = json.loads(alias_path.read_text(encoding="utf-8"))
            row = rows[identifier]
            base_id = run_id(protocol, RACE_PV.name, int(row["seed"]), "validation")
            base_archive = expected_physical[base_id]
            base_sidecar = json.loads(
                base_archive.with_suffix(".json").read_text(encoding="utf-8")
            )
            if (
                alias.get("run_id") != identifier
                or alias.get("protocol") != protocol
                or alias.get("model") != row["model"]
                or alias.get("seed") != row["seed"]
                or alias.get("split") != "validation"
                or alias.get("alias_of_run_id") != base_id
                or base_sidecar.get("run_id") != base_id
                or base_sidecar.get("content_sha256") != sha256_file(base_archive)
            ):
                errors.append(f"compact alias identity/hash invariant failure: {identifier}")
            if alias.get("storage_schema") != "compact_shared_v2_alias":
                errors.append(f"compact alias storage schema invariant failure: {identifier}")
            if alias.get("declared_intervention") != sensitivity_settings(
                str(row["model"])
            ):
                errors.append(f"compact alias intervention invariant failure: {identifier}")
            exact_reference(
                alias.get("alias_of_path"), base_archive, f"alias {identifier}"
            )
        except Exception as error:
            errors.append(f"compact alias binding unreadable: {identifier}: {error}")


def _verify_compact_audit(root: Path, errors: list[str]) -> None:
    audit_path = root / "predictions" / "audit-sample.csv.gz"
    sidecar_path = root / "predictions" / "audit-sample.json"
    if not _strict_artifact(root, audit_path, "compact audit CSV", errors):
        return
    if not _strict_artifact(root, sidecar_path, "compact audit JSON", errors):
        return
    try:
        audit = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(audit, dict) or set(audit) != AUDIT_SAMPLE_SIDECAR_KEYS:
            errors.append("compact audit sidecar contract mismatch")
            return
        if audit.get("storage_schema") != COMPACT_SCHEMA:
            errors.append("compact audit storage schema mismatch")
        if audit.get("path") != "predictions/audit-sample.csv.gz":
            errors.append("compact audit path mismatch")
        if audit.get("sampling") != AUDIT_SAMPLE_SAMPLING:
            errors.append("compact audit sampling contract mismatch")
        if (
            not isinstance(audit.get("row_count"), int)
            or isinstance(audit.get("row_count"), bool)
            or audit["row_count"] < 0
        ):
            errors.append("compact audit row count contract mismatch")
        if audit.get("sha256") != sha256_file(audit_path):
            errors.append("compact audit sample hash mismatch")
            return
        entries = prediction_entries(root)
        base = {
            logical["run_id"]: source
            for _path, logical, source in entries
            if "alias_of_run_id" not in logical
        }
        aliases = {
            logical["run_id"]: logical["alias_of_run_id"]
            for _path, logical, _source in entries
            if "alias_of_run_id" in logical
        }
        if audit.get("aliases") != dict(sorted(aliases.items())):
            errors.append("compact audit alias map mismatch")
        if audit.get("prediction_run_ids") != sorted(base):
            errors.append("compact audit run set mismatch")
        row_count = 0
        current_id = ""
        arrays: dict[str, np.ndarray] = {}
        chosen: dict[str, int] = {}
        coordinates: set[tuple[str, int, str]] = set()
        with gzip.open(audit_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(AUDIT_SAMPLE_FIELDS):
                errors.append("compact audit field contract mismatch")
                return
            for row in reader:
                identity = row["run_id"]
                if identity != current_id:
                    if current_id and len(coordinates) != len(chosen) * 24 * len(QUANTILES):
                        errors.append(f"compact audit coordinates incomplete: {current_id}")
                    if identity not in base:
                        errors.append(f"compact audit unknown run: {identity}")
                        return
                    current_id = identity
                    arrays = load_prediction_arrays(root, base[identity])
                    chosen = {}
                    blocks = np.asarray(arrays["block_ids"]).reshape(-1)
                    for block in sorted(set(map(str, blocks))):
                        candidates = np.flatnonzero(blocks == block)
                        digest = hashlib.sha256(f"{identity}|{block}".encode()).digest()
                        chosen[block] = int(
                            candidates[int.from_bytes(digest[:8], "big") % len(candidates)]
                        )
                    coordinates = set()
                block = row["block_id"]
                origin = int(row["origin"])
                lead = int(row["lead"]) - 1
                level = row["quantile"]
                coordinate = (block, lead, level)
                if coordinate in coordinates:
                    errors.append(f"compact audit coordinate duplicate: {identity}")
                    return
                coordinates.add(coordinate)
                if block not in chosen or origin != chosen[block] or not 0 <= lead < 24:
                    errors.append(f"compact audit coordinate invalid: {identity}")
                    return
                q_index = [format(value, ".12g") for value in QUANTILES].index(level)
                if (
                    row["prediction"]
                    != format(float(arrays["quantiles"][origin, lead, q_index]), ".12g")
                    or row["target"]
                    != format(float(arrays["targets"][origin, lead]), ".12g")
                    or row["site_id"] != str(arrays["site_ids"][origin, lead])
                ):
                    errors.append(f"compact audit content mismatch: {identity}")
                    return
                row_count += 1
        if current_id and len(coordinates) != len(chosen) * 24 * len(QUANTILES):
            errors.append(f"compact audit coordinates incomplete: {current_id}")
        if row_count != audit.get("row_count"):
            errors.append("compact audit row count mismatch")
    except Exception as error:
        errors.append(f"compact audit unreadable: {error}")


def _verify_evaluation_chain(root: Path, manifest: dict[str, Any], errors: list[str]) -> None:
    try:
        evaluations = json.loads((root / "evaluations.json").read_text(encoding="utf-8"))["runs"]
        expected = {item["run_id"] for item in manifest["run_identities"]}
        if {item["run_id"] for item in evaluations} != expected or len(evaluations) != len(expected):
            errors.append("evaluation identity/matrix mismatch")
        individual = {}
        for path in sorted((root / "evaluations").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if not str(record.get("run_id", "")).startswith(
                ("source_calibration/", "calibration_output/")
            ):
                individual[record["run_id"]] = record
        if individual != {item["run_id"]: item for item in evaluations}:
            errors.append("individual/evaluations JSON consistency mismatch")
        sensitivity_names = {item.name for item in VALIDATION_SENSITIVITIES}
        sensitivities = [item for item in evaluations if item["model"] in sensitivity_names]
        if len(sensitivities) != 45:
            errors.append("evaluation sensitivity matrix mismatch")
        for record in evaluations:
            input_artifact = record.get("calibration_input_artifact", {})
            output_artifact = record.get("calibration_output_artifact", {})
            input_path, output_path = root / input_artifact.get("path", ""), root / output_artifact.get("path", "")
            if not input_path.is_file() or sha256_file(input_path) != input_artifact.get("sha256"):
                errors.append(f"calibration input artifact mismatch: {record['run_id']}")
                continue
            expected_model = RACE_PV.name if record["model"] in sensitivity_names else record["model"]
            if input_artifact.get("run_id") != f"source_calibration/{expected_model}/{record['seed']}":
                errors.append(f"calibration identity mismatch: {record['run_id']}")
            input_payload = json.loads(input_path.read_text(encoding="utf-8"))
            if (
                input_payload.get("persistence")
                != "parameters_only_no_calibration_matrices"
                or set(input_payload.get("array_contract", {}))
                != {"quantiles", "targets", "daylight", "state_probabilities", "ramp_scores"}
            ):
                errors.append(f"calibration input contract mismatch: {record['run_id']}")
            if not output_path.is_file() or sha256_file(output_path) != output_artifact.get("sha256"):
                errors.append(f"calibration output artifact mismatch: {record['run_id']}")
            else:
                output_payload = json.loads(output_path.read_text(encoding="utf-8"))
                if (
                    output_payload.get("storage_schema") != "compact_cqr_parameters_v2"
                    or set(output_payload.get("control_parameters", {}))
                    != {"global", "lead_only", "state_lead"}
                ):
                    errors.append(f"calibration output contract mismatch: {record['run_id']}")
            settings = record.get("sensitivity_settings", {})
            if record["model"] == "a8a_lead_band_cqr" and settings.get("selected_control") != "lead_only":
                errors.append("lead-only CQR sensitivity mismatch")
            if record["model"] == "a8b_global_cqr" and settings.get("selected_control") != "global":
                errors.append("global CQR sensitivity mismatch")
            if record["model"] in {"ramp_q85", "ramp_q95"} and not str(settings.get("ramp_threshold_source", "")).startswith("calibration_q"):
                errors.append(f"validation-derived ramp threshold mismatch: {record['model']}")
            if record["model"] == "ramp_fixed_01" and settings.get("ramp_threshold") != 0.1:
                errors.append("fixed ramp threshold mismatch")
        aggregate = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
        if len(aggregate.get("sensitivities", [])) != 9 or {item["model"] for item in aggregate["sensitivities"]} != sensitivity_names:
            errors.append("aggregate exact-nine sensitivity mismatch")
        rows = {(item["model"], item["split"]): item for item in aggregate.get("rows", [])}
        for key, row in rows.items():
            matching = [item for item in evaluations if (item["model"], item["split"]) == key]
            expected_crps = float(np.mean([item["metrics"]["quantile_crps"] for item in matching]))
            if not np.isclose(row["mean_quantile_crps"], expected_crps, atol=1e-15):
                errors.append(f"aggregate/evaluation consistency mismatch: {key}")
        selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
        candidates = selection.get("candidates", [])
        selected = min(candidates, key=lambda item: (item["mean_quantile_crps"], item["model"]))
        if selection.get("selected_model") != selected.get("model") or selection.get("split") != "validation":
            errors.append("selection validation-only consistency mismatch")
        protocol = str(manifest["protocols"][0])
        for split in ("validation", "test"):
            for seed in SEEDS:
                identity = run_id(protocol, selection["selected_model"], seed, split)
                sidecar = next(
                    source
                    for _path, logical, source in prediction_entries(root)
                    if logical["run_id"] == identity
                )
                if selection["selected_artifact_hashes"][split][str(seed)] != sidecar["content_sha256"]:
                    errors.append(f"selection artifact hash mismatch: {identity}")
    except Exception as error:
        errors.append(f"evaluation/selection/aggregate chain unreadable: {error}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _claim_coordinate_matches(rows: list[dict[str, str]], coordinate: str, field: str, value: str) -> bool:
    clauses = dict(part.split("=", 1) for part in coordinate.split(";") if "=" in part)
    clauses.pop("field", None)
    for row in rows:
        if all(row.get(key) == expected for key, expected in clauses.items()):
            try:
                if np.isclose(float(row[field]), float(value), atol=1e-12, rtol=1e-12):
                    return True
            except (KeyError, ValueError):
                if row.get(field) == value:
                    return True
    return False


def claim_figure_coordinate_matches(
    spec: dict[str, Any], coordinate: str, field: str, value: str | float
) -> bool:
    clauses = dict(part.split("=", 1) for part in coordinate.split(";") if "=" in part)
    clauses.pop("field", None)
    for row in spec.get("coordinates", []):
        if not all(str(row.get(key)) == expected for key, expected in clauses.items()):
            continue
        try:
            return bool(
                np.isclose(float(row[field]), float(value), atol=1e-12, rtol=1e-12)
            )
        except (KeyError, TypeError, ValueError):
            return str(row.get(field)) == str(value)
    return False


def _verify_exports(root: Path, errors: list[str]) -> None:
    tables = root / "exports" / "tables"
    figures = root / "exports" / "figures"
    compilable_tables = []
    for name in REQUIRED_TABLES:
        csv_path, tex_path = tables / f"{name}.csv", tables / f"{name}.tex"
        if not csv_path.is_file():
            errors.append(f"missing table: {name}.csv")
        elif DISCLOSURE not in csv_path.read_text(encoding="utf-8"):
            errors.append(f"table disclosure missing: {name}.csv")
        if not tex_path.is_file():
            errors.append(f"missing table: {name}.tex")
        elif not _valid_tex(tex_path):
            errors.append(f"invalid tex: {name}.tex")
        else:
            compilable_tables.append(tex_path)
    if compilable_tables and not _compile_tex_tables(compilable_tables):
        errors.append("LaTeX compilation failed")
    try:
        aggregate = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
        expected_main = {
            (row["model"], row["split"]): row
            for row in aggregate["rows"]
            if row["split"] == "test"
        }
        actual_main = {
            (row["model"], row["split"]): row
            for row in _read_rows(tables / "main-results.csv")
        }
        if set(actual_main) != set(expected_main):
            errors.append("main-results/aggregate row consistency mismatch")
        else:
            for key, expected in expected_main.items():
                actual = actual_main[key]
                for field in (
                    "mean_quantile_crps",
                    "std_quantile_crps",
                    "daylight_nrmse",
                    "selected_picp",
                    "selected_mean_width",
                ):
                    if not np.isclose(float(actual[field]), float(expected[field]), atol=1e-15):
                        errors.append(f"main-results/aggregate value mismatch: {key}/{field}")
        significance = _read_rows(tables / "significance.csv")
        if len(significance) != len(aggregate["dm_bh"]):
            errors.append("significance/aggregate row consistency mismatch")
        else:
            by_model = {row["model"]: row for row in aggregate["dm_bh"]}
            for row in significance:
                expected = by_model.get(row["model"])
                if expected is None or not np.isclose(
                    float(row["statistic"]), float(expected["statistic"]), equal_nan=True
                ):
                    errors.append(f"significance/aggregate value mismatch: {row['model']}")
        transfer = _read_rows(tables / "transfer.csv")
        if len(transfer) != len(aggregate["forecast_skill"]):
            errors.append("transfer/aggregate row consistency mismatch")
        evaluations = json.loads(
            (root / "evaluations.json").read_text(encoding="utf-8")
        )["runs"]
        calibration_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for record in evaluations:
            for control, values in record["interval_metrics"].items():
                calibration_groups.setdefault(
                    (record["model"], record["split"], control), []
                ).append(values)
        expected_calibration = {
            key: {
                "picp": float(np.mean([item["picp"] for item in values])),
                "mean_width": float(np.mean([item["mean_width"] for item in values])),
                "coverage_target": float(
                    np.mean([item["coverage_target"] for item in values])
                ),
                "calibration_role": values[0]["calibration_role"],
                "n_seeds": len(values),
            }
            for key, values in calibration_groups.items()
        }
        actual_calibration = {
            (row["model"], row["split"], row["control"]): row
            for row in _read_rows(tables / "calibration.csv")
        }
        if set(actual_calibration) != set(expected_calibration):
            errors.append("calibration/evaluation row consistency mismatch")
        else:
            for key, expected in expected_calibration.items():
                actual = actual_calibration[key]
                numeric_match = all(
                    np.isclose(float(actual[field]), expected[field], atol=1e-15)
                    for field in ("picp", "mean_width", "coverage_target")
                )
                if (
                    not numeric_match
                    or actual["calibration_role"] != expected["calibration_role"]
                    or int(actual["n_seeds"]) != expected["n_seeds"]
                ):
                    errors.append(f"calibration/evaluation value mismatch: {key}")
        site_groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
        for record in evaluations:
            for item in record["site_lead"]:
                if int(item["lead"]) in REPORT_LEADS:
                    site_groups.setdefault(
                        (
                            record["model"],
                            record["split"],
                            item["site_id"],
                            int(item["lead"]),
                        ),
                        [],
                    ).append(item)
        expected_site_lead = {
            key: {
                "mean_quantile_crps": float(
                    np.mean([item["quantile_crps"] for item in values])
                ),
                "mean_daylight_nrmse": float(
                    np.mean([item["point"]["daylight"]["nrmse"] for item in values])
                ),
                "n_seeds": len(values),
            }
            for key, values in site_groups.items()
        }
        actual_site_lead = {
            (row["model"], row["split"], row["site_id"], int(row["lead"])): row
            for row in _read_rows(tables / "site-lead.csv")
        }
        if set(actual_site_lead) != set(expected_site_lead):
            errors.append("site-lead/evaluation row consistency mismatch")
        else:
            for key, expected in expected_site_lead.items():
                actual = actual_site_lead[key]
                if (
                    not all(
                        np.isclose(float(actual[field]), expected[field], atol=1e-15)
                        for field in ("mean_quantile_crps", "mean_daylight_nrmse")
                    )
                    or int(actual["n_seeds"]) != expected["n_seeds"]
                ):
                    errors.append(f"site-lead/evaluation value mismatch: {key}")
        expected_ablations = {
            (row["model"], row["split"]): row
            for row in aggregate["rows"]
            if row["model"].startswith("a")
        }
        actual_ablations = {
            (row["model"], row["split"]): row
            for row in _read_rows(tables / "ablations.csv")
        }
        if set(actual_ablations) != set(expected_ablations):
            errors.append("ablations/aggregate row consistency mismatch")
        else:
            for key, expected in expected_ablations.items():
                actual = actual_ablations[key]
                if not all(
                    np.isclose(float(actual[field]), float(expected[field]), atol=1e-15)
                    for field in (
                        "mean_quantile_crps",
                        "std_quantile_crps",
                        "daylight_nrmse",
                        "selected_picp",
                        "selected_mean_width",
                    )
                ):
                    errors.append(f"ablations/aggregate value mismatch: {key}")
    except Exception as error:
        errors.append(f"table/aggregate consistency unreadable: {error}")
    coordinate_keys = {
        "performance-heatmap": {"site_id", "lead", "value"},
        "seed-distribution": {"model", "seed", "value"},
        "coverage-width": {"coverage", "width", "control"},
        "significance-forest": {"model", "statistic", "adjusted_pvalue"},
        "expert-specialization": {"state", "expert", "value"},
        "gate-utilization": {"expert", "value"},
        "site-skill": {"site_id", "value", "reference_model"},
    }
    for name, kind in FIGURE_KINDS.items():
        spec_path = figures / f"{name}.json"
        png_path = figures / f"{name}.png"
        pdf_path = figures / f"{name}.pdf"
        if not png_path.is_file():
            errors.append(f"missing figure: {name}.png")
        elif not _valid_png(png_path):
            errors.append(f"invalid png: {name}.png")
        if not pdf_path.is_file():
            errors.append(f"missing figure: {name}.pdf")
        elif not _valid_pdf(pdf_path):
            errors.append(f"invalid pdf: {name}.pdf")
        if not spec_path.is_file():
            errors.append(f"missing figure spec: {name}.json")
            continue
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            required = {"coordinates", "series", "categories", "values", "intervals", "markers", "source_hashes", "caption", "renderer_contract"}
            if spec.get("kind") != kind or not required <= set(spec) or not spec["coordinates"]:
                errors.append(f"invalid chart spec: {name}.json")
            elif not coordinate_keys[name] <= set(spec["coordinates"][0]):
                errors.append(f"chart semantic coordinate mismatch: {name}.json")
            if DISCLOSURE not in spec.get("caption", "") or spec.get("disclosure") != DISCLOSURE:
                errors.append(f"figure disclosure mismatch: {name}.json")
            geometry_hash = hashlib.sha256(canonical_bytes(_geometry(spec))).hexdigest()
            contract = spec.get("renderer_contract", {})
            if contract.get("png_geometry_sha256") != geometry_hash or contract.get("pdf_geometry_sha256") != geometry_hash:
                errors.append(f"chart renderer geometry mismatch: {name}.json")
            text_hash = hashlib.sha256(canonical_bytes(_rendered_text(spec))).hexdigest()
            if contract.get("rendered_text_sha256") != text_hash:
                errors.append(f"chart rendered text mismatch: {name}.json")
            if png_path.is_file() and pdf_path.is_file():
                with tempfile.TemporaryDirectory() as directory:
                    expected_png = Path(directory) / "expected.png"
                    expected_pdf = Path(directory) / "expected.pdf"
                    geometry = _geometry(spec)
                    _render_png(expected_png, geometry, spec)
                    _render_pdf(expected_pdf, geometry, spec)
                    if expected_png.read_bytes() != png_path.read_bytes():
                        errors.append(f"PNG/spec render mismatch: {name}.png")
                    if expected_pdf.read_bytes() != pdf_path.read_bytes():
                        errors.append(f"PDF/spec render mismatch: {name}.pdf")
            for source, digest in spec.get("source_hashes", {}).items():
                if not (root / source).is_file() or sha256_file(root / source) != digest:
                    errors.append(f"chart source hash mismatch: {name}/{source}")
        except Exception as error:
            errors.append(f"unreadable chart spec {name}: {error}")
    for name in REQUIRED_PAPER:
        path = root / "exports" / "paper" / name
        if not path.is_file():
            errors.append(f"missing paper artifact: {name}")
        elif DISCLOSURE not in path.read_text(encoding="utf-8"):
            errors.append(f"paper disclosure missing: {name}")
    claim_path = tables / "claim-evidence-map.csv"
    if claim_path.is_file():
        for claim in _read_rows(claim_path):
            csv_path = root / claim.get("csv_path", "")
            spec_path = root / claim.get("figure_spec_path", "")
            png_path = root / claim.get("png_path", "")
            pdf_path = root / claim.get("pdf_path", "")
            if not csv_path.is_file() or sha256_file(csv_path) != claim.get("csv_sha256"):
                errors.append(f"claim CSV hash mismatch: {claim.get('claim_id')}")
                continue
            if not png_path.is_file() or sha256_file(png_path) != claim.get("png_sha256"):
                errors.append(f"claim PNG hash mismatch: {claim.get('claim_id')}")
            if not pdf_path.is_file() or sha256_file(pdf_path) != claim.get("pdf_sha256"):
                errors.append(f"claim PDF hash mismatch: {claim.get('claim_id')}")
            if not spec_path.is_file() or sha256_file(spec_path) != claim.get(
                "figure_spec_sha256"
            ):
                errors.append(f"claim figure spec hash mismatch: {claim.get('claim_id')}")
            else:
                spec = json.loads(spec_path.read_text(encoding="utf-8"))
                if not claim_figure_coordinate_matches(
                    spec,
                    claim.get("figure_coordinate", ""),
                    claim.get("figure_statistic_field", ""),
                    claim.get("value", ""),
                ):
                    errors.append(f"claim figure/statistic mismatch: {claim.get('claim_id')}")
            if not claim.get("run_source_identities") or not _claim_coordinate_matches(
                _read_rows(csv_path),
                claim.get("csv_coordinate", ""),
                claim.get("statistic_field", ""),
                claim.get("value", ""),
            ):
                errors.append(f"claim statistic/coordinate mismatch: {claim.get('claim_id')}")


def verify(root: Path) -> None:
    require_stage(root, "export", ("exports/tables", "exports/figures", "exports/paper"))
    manifest = load_manifest(root)
    errors: list[str] = []
    _index, root_digest = _verify_dag(root, errors)
    _verify_manifest(root, manifest, errors)
    expected_ids = {item["run_id"] for item in manifest.get("run_identities", [])}
    runs_root = root / "predictions" / "runs"
    archives = (
        sorted(runs_root.rglob("*.npz"))
        if runs_root.is_dir()
        else sorted((root / "predictions").glob("*.npz"))
    )
    aliases = (
        sorted(runs_root.rglob("*.alias.json"))
        if runs_root.is_dir()
        else sorted((root / "predictions").glob("*.alias.json"))
    )
    if len(archives) + len(aliases) != len(expected_ids):
        errors.append("prediction matrix incomplete")
    seen: set[str] = set()
    for archive in archives:
        _verify_prediction(root, archive, expected_ids, seen, errors)
    _verify_aliases(root, expected_ids, seen, errors)
    if runs_root.is_dir():
        _verify_compact_audit(root, errors)
    if seen != expected_ids:
        errors.append("run identity set does not match manifest")
    _verify_evaluation_chain(root, manifest, errors)
    _verify_exports(root, errors)
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    size_limit = ACTUAL_LIMIT_BYTES if runs_root.is_dir() else MAX_BYTES
    if size > size_limit:
        errors.append("size cap exceeded")
    isolation_flags = {
        "evidence_status": manifest.get("evidence_status"),
        "oracle_conditioned": manifest.get("oracle_conditioned"),
        "registered_evidence_eligible": manifest.get("registered_evidence_eligible"),
    }
    report = {
        **dict(EVIDENCE_METADATA),
        "disclosure": DISCLOSURE,
        "ok": not errors,
        "errors": errors,
        "artifact_bytes": size,
        "matrix_runs": len(seen),
        "isolation_flags": isolation_flags,
        "verified_root_digest": root_digest,
        "verification_outputs_excluded_from_digest": ["verification.json", "verification.txt"],
    }
    write_json(root / "verification.json", report)
    atomic_bytes(
        root / "verification.txt",
        (("PASS\n" if not errors else "FAIL\n") + DISCLOSURE + "\n" + "\n".join(errors)).encode(
            "utf-8"
        ),
    )
    if errors:
        raise ValueError("verification failed: " + "; ".join(errors))
    record_verification(root, root_digest)


def verify_protocol_core(root: Path) -> dict[str, Any]:
    """Verify one compact 4B--4C protocol bundle without requiring 4E exports."""

    root = Path(root).expanduser().resolve()
    manifest = load_manifest(root)
    errors: list[str] = []
    _index, root_digest = _verify_dag(
        root,
        errors,
        required_stages=("prepare", "generate", "evaluate", "aggregate"),
    )
    _verify_manifest(root, manifest, errors, verify_acceptance_report=False)
    _verify_shared_cohorts(root, manifest, errors)
    expected_ids = {item["run_id"] for item in manifest.get("run_identities", [])}
    runs_root = root / "predictions" / "runs"
    archives = sorted(runs_root.rglob("*.npz"))
    sidecars = sorted(
        path
        for path in runs_root.rglob("*.json")
        if not path.name.endswith(".alias.json")
    )
    aliases = sorted(runs_root.rglob("*.alias.json"))
    estimate = manifest.get("storage_estimate", {}).get("per_protocol", {}).get(
        str(manifest.get("protocols", [""])[0]),
        {},
    )
    expected_archives = int(estimate.get("prediction_archives", -1))
    expected_aliases = int(estimate.get("sensitivity_aliases", -1))
    if (
        len(archives) != expected_archives
        or len(sidecars) != expected_archives
        or len(aliases) != expected_aliases
    ):
        errors.append("compact physical archive/sidecar/alias count invariant failure")
    if len(expected_ids) != expected_archives + expected_aliases:
        errors.append("manifest logical run count invariant failure")
    _verify_compact_run_bindings(
        root, manifest, archives, sidecars, aliases, errors
    )
    seen: set[str] = set()
    for archive in archives:
        _verify_prediction(root, archive, expected_ids, seen, errors)
        try:
            sidecar = json.loads(
                archive.with_suffix(".json").read_text(encoding="utf-8")
            )
            if (
                sidecar.get("generation_config_sha256")
                != manifest.get("generation_config_sha256")
                or sidecar.get("cohort_index_sha256")
                != manifest.get("cohort_index_sha256")
            ):
                errors.append(
                    f"prediction provenance hash invariant failure: {archive.name}"
                )
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"prediction sidecar hash/invariant unreadable: {error}")
    _verify_aliases(root, expected_ids, seen, errors)
    _verify_compact_audit(root, errors)
    if seen != expected_ids:
        errors.append("run identity set does not match manifest")
    _verify_evaluation_chain(root, manifest, errors)
    if errors:
        raise ValueError("protocol verification failed: " + "; ".join(errors))
    return {
        "protocol": str(manifest["protocols"][0]),
        "prediction_archives": len(archives),
        "prediction_sidecars": len(sidecars),
        "sensitivity_aliases": len(aliases),
        "logical_runs": len(seen),
        "artifact_root_sha256": root_digest,
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "cohort_index_sha256": sha256_file(root / "cohort-index.json"),
        "audit_sample_sha256": sha256_file(root / "predictions/audit-sample.csv.gz"),
    }


__all__ = ["valid_tex_text", "verify", "verify_protocol_core"]
