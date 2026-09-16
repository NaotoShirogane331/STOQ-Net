"""Read-only R6 confirmation preregistration and metadata-only selection.

This module deliberately has no scenario writer or cohort materializer.  It
only authenticates the retained R5 failure evidence and ranks in-memory block
metadata supplied by a caller.
"""

from __future__ import annotations

import csv
import hashlib
import json
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .cohort import BLOCK_STEPS, BlockRecord, NormalizedSiteSeries, _valid_blocks, stable_hash
from .constants import STEP_MINUTES
from .loader import role_allows_split
from .nested_validation import EXPECTED_AUXILIARY_OK_SITE_IDS

_FIXED_PLAN = {
    "catalog_sha256": "975777378945a7cd8b0dbaad32d3642fe79785e09f091b5a1af7bbde0946435e",
    "dkasc_blocks_per_site": 1,
    "embargo_days": 14,
    "held_out_allowed": False,
    "oklahoma_blocks_per_site": 3,
    "oklahoma_pairwise_embargo": True,
    "predecessor_revision": "r5",
    "preregistration_sha256": "2c17c2cc527528c024e7944559bc811c9757277c587d24d744312b0ebe5350d1",
    "prior_block_counts": {"dkasc": 60, "oklahoma": 15},
    "prior_coordinate_count": 75,
    "prior_site_counts": {"dkasc": 12, "oklahoma": 5},
    "r5_confirmation_decision_sha256": "aed8c8b9759ee1f7a04a208201fcf80cbe3852a93df9141a5ae99c9ace48993d",
    "r5_confirmation_protocol_sha256": "8f0299e041510502b0b65173b86e8c08ea9679e2c0abfb97920a9f9f4e86d85f",
    "r5_confirmation_state_sha256": "603cbb6e6ee0f70e90789fdc21c1c13ad0ed847bc6b57fe63c2a98d6931e3062",
    "revision": "r6",
    "schema": "realism_r6_confirmation_plan_v1",
    "selected_blocks_authority_sha256": "d1496e86d93f3073b05b053234e4078f5c85f9903cc7e7fd3325fbfa1f380dd6",
    "selection_seed": 20260818,
}
_PLAN_SHA256 = "483b452aa74d15a073137c596dde1102bc25c201a66958bf45edc88887f69f81"
_SELECTED_BLOCKS_SIZE = 53090
_NAMESPACE = "nested-confirmation-r6"
_SELECTION_SEED = 20260818
_EMBARGO_DAYS = 14
_EMBARGO = pd.Timedelta(days=_EMBARGO_DAYS)


@dataclass(frozen=True)
class R6ConfirmationSelection:
    """A sealed, metadata-only set of proposed R6 block records."""

    blocks: tuple[BlockRecord, ...]
    prior_coordinate_sha256: str
    selected_coordinate_sha256: str
    plan_sha256: str
    selection_seed: int
    embargo_days: int


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _coordinate_hash(coordinates: Iterable[tuple[str, str, str]]) -> str:
    return _sha256(_canonical_bytes([list(item) for item in sorted(coordinates)]))


def r6_confirmation_plan_payload() -> dict[str, object]:
    """Return a detached copy of the exact, canonical R6 plan payload."""
    return json.loads(json.dumps(_FIXED_PLAN))


def r6_confirmation_plan_sha256() -> str:
    """Recompute the immutable plan hash and reject a source-level mutation."""
    actual = _sha256(_canonical_bytes(r6_confirmation_plan_payload()))
    if actual != _PLAN_SHA256:
        raise ValueError("r6 confirmation plan does not match its fixed hash")
    return actual


def _pinned_plan_payload() -> dict[str, object]:
    """Return plan fields only after independently validating the fixed plan seal."""
    expected = r6_confirmation_plan_sha256()
    payload = r6_confirmation_plan_payload()
    if _sha256(_canonical_bytes(payload)) != expected:
        raise ValueError("r6 confirmation plan changed during validation")
    return payload


def _reject_symlink_components(path: Path) -> None:
    absolute = path.expanduser().absolute()
    for component in (absolute, *absolute.parents):
        try:
            if stat.S_ISLNK(component.lstat().st_mode):
                raise ValueError(f"symlink path component is forbidden: {component}")
        except FileNotFoundError:
            continue


def _secure_root(root: Path) -> Path:
    raw = Path(root).expanduser()
    if not raw.is_absolute() or ".." in raw.parts:
        raise ValueError("r6 scenario root path is noncanonical")
    _reject_symlink_components(raw)
    try:
        mode = raw.lstat().st_mode
    except OSError as error:
        raise ValueError("r6 scenario root is missing") from error
    if not stat.S_ISDIR(mode):
        raise ValueError("r6 scenario root is not a directory")
    resolved = raw.resolve(strict=True)
    if raw != resolved:
        raise ValueError("r6 scenario root path is noncanonical")
    return resolved


def _secure_file(root: Path, path: Path, *, label: str, digest: str, size: int | None = None) -> bytes:
    raw = Path(path).expanduser()
    _reject_symlink_components(raw)
    try:
        resolved = raw.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError(f"{label} path escapes the scenario root")
        mode = raw.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise ValueError(f"{label} is not a regular file")
        content = raw.read_bytes()
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{label} is missing or unreadable") from error
    if raw != resolved or not raw.is_relative_to(root):
        raise ValueError(f"{label} path is noncanonical")
    if size is not None and len(content) != size:
        raise ValueError(f"{label} size does not match fixed authority")
    if _sha256(content) != digest:
        raise ValueError(f"{label} hash does not match fixed authority")
    return content


def _json_no_duplicates(content: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"{label} contains a duplicate JSON key")
            output[key] = value
        return output

    try:
        result = json.loads(content.decode("utf-8"), object_pairs_hook=reject_duplicate)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical JSON") from error
    if not isinstance(result, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return result


def _read_csv(content: bytes, *, label: str, columns: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    try:
        text = content.decode("utf-8")
        rows = tuple(csv.DictReader(text.splitlines(), strict=True))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"{label} is not canonical CSV") from error
    if not rows or tuple(rows[0]) != columns or any(tuple(row) != columns for row in rows):
        raise ValueError(f"{label} columns are noncanonical")
    if any(any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{label} has incomplete rows")
    return rows


def _coordinate_from_row(row: dict[str, Any]) -> tuple[str, str, str]:
    site_id, start, end = row.get("site_id"), row.get("start"), row.get("end")
    if not all(isinstance(value, str) and value for value in (site_id, start, end)):
        raise ValueError("evidence coordinate is incomplete")
    try:
        parsed_start, parsed_end = pd.Timestamp(start), pd.Timestamp(end)
    except (TypeError, ValueError) as error:
        raise ValueError("evidence coordinate is not ISO8601") from error
    if parsed_start.tz is None or parsed_end.tz is None or parsed_end < parsed_start:
        raise ValueError("evidence coordinate is not a timezone-aware interval")
    if parsed_start.isoformat() != start or parsed_end.isoformat() != end:
        raise ValueError("evidence coordinate is noncanonical")
    return site_id, start, end


def _validate_r5_terminal(decision: dict[str, Any], state: dict[str, Any]) -> None:
    if (
        decision.get("schema") != "r5_confirmation_decision_v1"
        or decision.get("status") != "failed_confirmation"
        or decision.get("candidate_id") != "rv2-r5-009-20260821"
        or decision.get("reasons") != ["ramp_aucpr_absolute_gain:below_minimum"]
        or decision.get("held_out_used") is not False
    ):
        raise ValueError("R5 decision is not the retained failed terminal")
    if (
        state.get("schema") != "r5_confirmation_state_v1"
        or state.get("state") != "failed_confirmation"
        or not isinstance(state.get("history"), list)
        or not state["history"]
        or state["history"][-1].get("state") != "failed_confirmation"
    ):
        raise ValueError("R5 state is not the retained failed terminal")


def _validate_r5_block(
    row: dict[str, Any], *, cohort: str, dataset: str, role: str
) -> tuple[str, str, str]:
    coordinate = _coordinate_from_row(row)
    if (
        row.get("dataset") != dataset
        or row.get("split") != "validation"
        or row.get("spatial_role") != role
        or row.get("spatial_role") in {"test", "zero_parameter_test"}
        or row.get("n_steps") != BLOCK_STEPS
        or row.get("quarter") != f"{pd.Timestamp(coordinate[1]).year}Q{pd.Timestamp(coordinate[1]).quarter}"
    ):
        raise ValueError("R5 protocol block is outside the validation-only contract")
    expected_end = pd.Timestamp(coordinate[1]) + pd.Timedelta(minutes=STEP_MINUTES * (BLOCK_STEPS - 1))
    if pd.Timestamp(coordinate[2]) != expected_end:
        raise ValueError("R5 protocol block boundary is invalid")
    raw_hashes = row.get("raw_hashes")
    if not isinstance(raw_hashes, list) or not raw_hashes or not all(
        isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")
        for value in raw_hashes
    ):
        raise ValueError("R5 protocol raw hashes are invalid")
    block_id, selection_hash = row.get("block_id"), row.get("selection_hash")
    if not isinstance(block_id, str) or not isinstance(selection_hash, str) or len(block_id) != 20 or len(selection_hash) != 64:
        raise ValueError("R5 protocol block identity is invalid")
    source_digest = stable_hash(dataset, coordinate[0], "validation", coordinate[1], coordinate[2])
    expected_selection = (
        stable_hash("nested-confirmation-v1", 20260816, coordinate[0], coordinate[1], coordinate[2])
        if cohort == "dkasc"
        else source_digest
    )
    if block_id != source_digest[:20] or selection_hash != expected_selection:
        raise ValueError("R5 protocol block identity is invalid")
    return coordinate


def load_r6_prior_coordinates(scenario_root: Path, selected_blocks_csv: Path) -> tuple[tuple[str, str, str], ...]:
    """Authenticate and load exactly the 48+27 retained validation coordinates."""
    plan = _pinned_plan_payload()
    root = _secure_root(scenario_root)
    expected_selected = root / "01-cohorts" / "selected-blocks.csv"
    if Path(selected_blocks_csv).expanduser() != expected_selected:
        raise ValueError("selected-blocks path is noncanonical or escapes the scenario root")
    selected = _secure_file(root, expected_selected, label="selected blocks", digest=str(plan["selected_blocks_authority_sha256"]), size=_SELECTED_BLOCKS_SIZE)
    protocol = _secure_file(root, root / "00-manifest" / "r5-confirmation-protocol.json", label="R5 protocol", digest=str(plan["r5_confirmation_protocol_sha256"]))
    decision = _secure_file(root, root / "09-logs" / "r5-confirmation" / "decision.json", label="R5 decision", digest=str(plan["r5_confirmation_decision_sha256"]))
    state = _secure_file(root, root / "09-logs" / "r5-confirmation" / "state.json", label="R5 state", digest=str(plan["r5_confirmation_state_sha256"]))
    _validate_r5_terminal(_json_no_duplicates(decision, label="R5 decision"), _json_no_duplicates(state, label="R5 state"))
    selected_rows = _read_csv(selected, label="selected blocks", columns=("block_id", "site_id", "group", "protocol", "split", "start", "end", "quarter", "n_steps", "selection_hash"))
    old = tuple(row for row in selected_rows if row["group"] == "DKASC" and row["protocol"] == "dkasc-yulara" and row["split"] == "validation")
    if len(old) != 48:
        raise ValueError("selected blocks must contain exactly forty-eight DKASC validation rows")
    old_coordinates = tuple(_coordinate_from_row(row) for row in old)
    for row, coordinate in zip(old, old_coordinates, strict=True):
        source_digest = stable_hash("dkasc", coordinate[0], "validation", coordinate[1], coordinate[2])
        if (
            row.get("n_steps") != str(BLOCK_STEPS)
            or row.get("quarter") != f"{pd.Timestamp(coordinate[1]).year}Q{pd.Timestamp(coordinate[1]).quarter}"
            or row.get("block_id") != source_digest[:20]
            or row.get("selection_hash") != source_digest
        ):
            raise ValueError("selected blocks contain a noncanonical historical block")
    old_sites = tuple(sorted({coordinate[0] for coordinate in old_coordinates}))
    if len(set(old_coordinates)) != 48 or len(old_sites) != 12 or any(sum(row["site_id"] == site for row in old) != 4 for site in old_sites):
        raise ValueError("selected blocks have an invalid DKASC coordinate distribution")
    payload = _json_no_duplicates(protocol, label="R5 protocol")
    if payload.get("schema") != "nested_confirmation_protocol_v1" or payload.get("auxiliary_confirmation_only") is not True:
        raise ValueError("R5 protocol is not the validation-only confirmation protocol")
    blocks = payload.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 27:
        raise ValueError("R5 protocol must contain exactly twenty-seven blocks")
    dkasc = tuple(row for row in blocks if isinstance(row, dict) and row.get("cohort") == "dkasc")
    oklahoma = tuple(row for row in blocks if isinstance(row, dict) and row.get("cohort") == "oklahoma")
    if len(dkasc) != 12 or len(oklahoma) != 15:
        raise ValueError("R5 protocol cohort distribution is invalid")
    protocol_dkasc = tuple(_validate_r5_block(row, cohort="dkasc", dataset="dkasc", role="source_train") for row in dkasc)
    protocol_oklahoma = tuple(_validate_r5_block(row, cohort="oklahoma", dataset="nrel", role="validation_calibration") for row in oklahoma)
    if tuple(sorted({coordinate[0] for coordinate in protocol_dkasc})) != old_sites or any(sum(coordinate[0] == site for coordinate in protocol_dkasc) != 1 for site in old_sites):
        raise ValueError("R5 protocol DKASC site distribution is invalid")
    if tuple(sorted({coordinate[0] for coordinate in protocol_oklahoma})) != EXPECTED_AUXILIARY_OK_SITE_IDS or any(sum(coordinate[0] == site for coordinate in protocol_oklahoma) != 3 for site in EXPECTED_AUXILIARY_OK_SITE_IDS):
        raise ValueError("R5 protocol Oklahoma site distribution is invalid")
    coordinates = tuple(sorted((*old_coordinates, *protocol_dkasc, *protocol_oklahoma)))
    if len(coordinates) != 75 or len(set(coordinates)) != 75:
        raise ValueError("R5 prior coordinates must be exactly seventy-five unique rows")
    return coordinates


def _interval_gap_to_coordinate(block: BlockRecord, coordinate: tuple[str, str, str]) -> pd.Timedelta:
    _, start, end = coordinate
    prior_start, prior_end = pd.Timestamp(start), pd.Timestamp(end)
    if block.end < prior_start:
        return prior_start - block.end
    if prior_end < block.start:
        return block.start - prior_end
    return pd.Timedelta(0)


def _interval_gap(left: BlockRecord, right: BlockRecord) -> pd.Timedelta:
    if left.end < right.start:
        return right.start - left.end
    if right.end < left.start:
        return left.start - right.end
    return pd.Timedelta(0)


def _rank(block: BlockRecord) -> str:
    return stable_hash(_NAMESPACE, _SELECTION_SEED, block.site_id, block.start.isoformat(), block.end.isoformat())


def _valid_raw_hashes(value: object) -> bool:
    return isinstance(value, tuple) and bool(value) and tuple(sorted(set(value))) == value and all(
        isinstance(item, str)
        and len(item) == 64
        and all(character in "0123456789abcdef" for character in item)
        for item in value
    )


def _validate_site(site: NormalizedSiteSeries, *, dataset: str, role: str, state: str | None) -> None:
    if not _valid_raw_hashes(site.raw_hashes):
        raise ValueError("R6 selector source raw hashes are invalid")
    if (
        site.dataset != dataset
        or site.split != "validation"
        or site.spatial_role != role
        or site.spatial_role in {"test", "zero_parameter_test"}
        or (state is not None and site.state != state)
        or not role_allows_split(site.dataset, site.spatial_role or "", site.state, "validation", site_id=site.site_id)
    ):
        raise ValueError("R6 selector source role/split is invalid or held-out")


def _validate_block(block: BlockRecord, site: NormalizedSiteSeries) -> None:
    expected_digest = stable_hash(block.dataset, block.site_id, block.split, block.start.isoformat(), block.end.isoformat())
    expected_end = block.start + pd.Timedelta(minutes=STEP_MINUTES * (BLOCK_STEPS - 1))
    if (
        block.dataset != site.dataset
        or block.site_id != site.site_id
        or block.split != site.split
        or block.spatial_role != site.spatial_role
        or block.state != site.state
        or block.block_id != expected_digest[:20]
        or block.selection_hash != expected_digest
        or block.raw_hashes != site.raw_hashes
        or not _valid_raw_hashes(block.raw_hashes)
        or block.n_steps != BLOCK_STEPS
        or block.end != expected_end
        or block.quarter != f"{block.start.year}Q{block.start.quarter}"
    ):
        raise ValueError("R6 selector received a noncanonical block record")


def _require_prior_contract(
    prior: tuple[tuple[str, str, str], ...],
    dkasc_site_ids: frozenset[str],
) -> None:
    if len(prior) != 75:
        raise ValueError("R6 prior coordinates must contain exactly seventy-five rows")
    counts: dict[str, int] = {}
    for site_id, _start, _end in prior:
        counts[site_id] = counts.get(site_id, 0) + 1
    oklahoma_site_ids = frozenset(EXPECTED_AUXILIARY_OK_SITE_IDS)
    if (
        set(counts) != dkasc_site_ids | oklahoma_site_ids
        or any(counts.get(site_id) != 5 for site_id in dkasc_site_ids)
        or any(counts.get(site_id) != 3 for site_id in oklahoma_site_ids)
    ):
        raise ValueError("R6 prior coordinate site distribution is invalid")


def _post_validate_selection(
    blocks: tuple[BlockRecord, ...], source_by_id: dict[str, NormalizedSiteSeries]
) -> None:
    """Recheck the sealed metadata result after greedy selection has finished."""
    coordinates = tuple((block.site_id, block.start.isoformat(), block.end.isoformat()) for block in blocks)
    if (
        len(blocks) != 27
        or len(source_by_id) != 17
        or len({block.site_id for block in blocks}) != 17
        or len(set(coordinates)) != 27
        or len({block.block_id for block in blocks}) != 27
    ):
        raise ValueError("R6 selection cardinality or identities are invalid")
    for block in blocks:
        source = source_by_id.get(block.site_id)
        if source is None:
            raise ValueError("R6 selection source identity is invalid")
        _validate_block(block, source)
    for site_id in EXPECTED_AUXILIARY_OK_SITE_IDS:
        site_blocks = tuple(block for block in blocks if block.site_id == site_id)
        if len(site_blocks) != 3:
            raise ValueError("R6 Oklahoma selection cardinality is invalid")
        if any(
            _interval_gap(left, right) < _EMBARGO
            for index, left in enumerate(site_blocks)
            for right in site_blocks[index + 1 :]
        ):
            raise ValueError("R6 Oklahoma selection violates the pairwise embargo")


def select_r6_confirmation_blocks(
    dkasc_sites: Iterable[NormalizedSiteSeries],
    oklahoma_sites: Iterable[NormalizedSiteSeries],
    prior_coordinates: Iterable[tuple[str, str, str]],
) -> R6ConfirmationSelection:
    """Rank fresh validation metadata; intentionally never build cohort targets or write files."""
    r6_confirmation_plan_sha256()
    dkasc = tuple(dkasc_sites)
    oklahoma = tuple(oklahoma_sites)
    if len(dkasc) != 12 or len({site.site_id for site in dkasc}) != 12:
        raise ValueError("R6 selection requires exactly twelve DKASC sites")
    if tuple(sorted(site.site_id for site in oklahoma)) != EXPECTED_AUXILIARY_OK_SITE_IDS:
        raise ValueError("R6 selection requires the exact five Oklahoma validation sites")
    if len(oklahoma) != 5 or len({site.site_id for site in oklahoma}) != 5:
        raise ValueError("R6 selection requires exactly five distinct Oklahoma sites")
    unsorted_prior = tuple(prior_coordinates)
    if any(
        not isinstance(item, tuple)
        or len(item) != 3
        or any(not isinstance(value, str) for value in item)
        for item in unsorted_prior
    ):
        raise ValueError("R6 prior coordinates must be unique triples")
    if len(set(unsorted_prior)) != len(unsorted_prior):
        raise ValueError("R6 prior coordinates must be unique triples")
    prior = tuple(sorted(unsorted_prior))
    for coordinate in prior:
        _coordinate_from_row({"site_id": coordinate[0], "start": coordinate[1], "end": coordinate[2]})
    _require_prior_contract(prior, frozenset(site.site_id for site in dkasc))
    selected: list[BlockRecord] = []
    for site in sorted(dkasc, key=lambda item: item.site_id):
        _validate_site(site, dataset="dkasc", role="source_train", state=None)
        eligible = []
        for _, block in _valid_blocks(site):
            _validate_block(block, site)
            coordinate = (block.site_id, block.start.isoformat(), block.end.isoformat())
            if coordinate not in prior and all(block.site_id != old[0] or _interval_gap_to_coordinate(block, old) >= _EMBARGO for old in prior):
                eligible.append(block)
        if not eligible:
            raise ValueError(f"DKASC site {site.site_id!r} lacks an eligible R6 block")
        selected.append(min(eligible, key=_rank))
    for site in sorted(oklahoma, key=lambda item: item.site_id):
        _validate_site(site, dataset="nrel", role="validation_calibration", state="OK")
        eligible = []
        for _, block in _valid_blocks(site):
            _validate_block(block, site)
            coordinate = (block.site_id, block.start.isoformat(), block.end.isoformat())
            if coordinate not in prior and all(block.site_id != old[0] or _interval_gap_to_coordinate(block, old) >= _EMBARGO for old in prior):
                eligible.append(block)
        chosen: list[BlockRecord] = []
        for block in sorted(eligible, key=_rank):
            if all(_interval_gap(block, prior_block) >= _EMBARGO for prior_block in chosen):
                chosen.append(block)
            if len(chosen) == 3:
                break
        if len(chosen) != 3:
            raise ValueError(f"Oklahoma site {site.site_id!r} lacks three eligible R6 blocks")
        selected.extend(chosen)
    blocks = tuple(sorted(selected, key=lambda block: (block.dataset, block.site_id, block.start, block.end, block.block_id)))
    source_by_id = {site.site_id: site for site in (*dkasc, *oklahoma)}
    _post_validate_selection(blocks, source_by_id)
    if any(block.site_id == old[0] and _interval_gap_to_coordinate(block, old) < _EMBARGO for block in blocks for old in prior):
        raise ValueError("R6 selection violates the prior embargo")
    coordinates = tuple((block.site_id, block.start.isoformat(), block.end.isoformat()) for block in blocks)
    return R6ConfirmationSelection(blocks, _coordinate_hash(prior), _coordinate_hash(coordinates), r6_confirmation_plan_sha256(), _SELECTION_SEED, _EMBARGO_DAYS)
