"""Versioned, metadata-only R6b 12+15 confirmation contract."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .cohort import BLOCK_STEPS, BlockRecord, NormalizedSiteSeries, _valid_blocks, stable_hash
from .constants import STEP_MINUTES
from .loader import role_allows_split
from .nested_validation import EXPECTED_AUXILIARY_OK_SITE_IDS
from .scenario_revision import r6b_preregistration_payload, r6b_preregistration_sha256

EXPECTED_R6B_OKLAHOMA_SITE_IDS = (
    "nrel_ok_p34d65_m98d45_29mw_dpv",
    "nrel_ok_p35d15_m97d35_31mw_dpv",
    "nrel_ok_p35d25_m97d35_31mw_dpv",
    "nrel_ok_p35d35_m97d35_31mw_dpv",
    "nrel_ok_p35d45_m97d35_37mw_dpv",
    "nrel_ok_p35d45_m97d45_37mw_dpv",
    "nrel_ok_p35d45_m97d55_37mw_dpv",
    "nrel_ok_p35d55_m97d35_37mw_dpv",
    "nrel_ok_p35d55_m97d95_28mw_dpv",
    "nrel_ok_p35d65_m97d35_37mw_dpv",
    "nrel_ok_p35d65_m97d45_37mw_dpv",
    "nrel_ok_p35d65_m97d55_37mw_dpv",
    "nrel_ok_p35d65_m97d95_28mw_dpv",
    "nrel_ok_p35d75_m97d45_37mw_dpv",
    "nrel_ok_p35d95_m95d55_6mw_dpv",
    "nrel_ok_p36d05_m96d95_21mw_dpv",
    "nrel_ok_p36d15_m95d95_28mw_dpv",
    "nrel_ok_p36d15_m96d95_21mw_dpv",
    "nrel_ok_p36d25_m95d95_28mw_dpv",
    "nrel_ok_p36d75_m95d95_5mw_dpv",
)

_CANDIDATE_SHA256 = "0330e0ffc2e7a0445fb72d75d045034668cff7113170cfd6e5493275c1a69fa4"
_NREL_MANIFEST_SHA256 = "a0d6934419b10c08f648646d3a7acf86435e53af97dc18ad0b1d23d8724fc450"
_NREL_SELECTION_AUDIT_SHA256 = "35120739dd1000e148040836c04bb310b1315499eba736413a800ec694a0ab82"
_PRIOR_SHA256 = "58073a22e4d92c77be0008c3de707d9fe06c7c1cadfafdde9bbeb46121361e98"
_NAMESPACE = "nested-confirmation-r6b"
_OK_SITE_NAMESPACE = "nested-confirmation-r6b-oklahoma-site"
_SELECTION_SEED = 20260818
_EMBARGO_DAYS = 14
_EMBARGO = pd.Timedelta(days=_EMBARGO_DAYS)
_FIXED_PLAN = {
    "candidate_universe_sha256": _CANDIDATE_SHA256,
    "candidate_universe_site_count": 20,
    "catalog_sha256": "975777378945a7cd8b0dbaad32d3642fe79785e09f091b5a1af7bbde0946435e",
    "dkasc_blocks_per_site": 1,
    "dkasc_site_count": 12,
    "embargo_days": 14,
    "held_out_allowed": False,
    "oklahoma_blocks_per_site": 1,
    "oklahoma_pairwise_embargo": True,
    "oklahoma_site_count": 15,
    "predecessor_revision": "r6",
    "preregistration_sha256": "d06310a7f6a788da487ec8a44fbcb8f378ddaacb0538d50bc93216fefb53a28e",
    "prior_block_counts": {"dkasc": 60, "oklahoma": 15},
    "prior_coordinate_count": 75,
    "prior_coordinate_sha256": _PRIOR_SHA256,
    "prior_site_counts": {"dkasc": 12, "oklahoma": 5},
    "range_definition_sha256": "8aa7adfe3b001f6d7dbc34fc09f9e4fd423591d91e867605373d9d13f6503d4c",
    "revision": "r6b",
    "schema": "realism_r6b_confirmation_plan_v1",
    "selected_block_count": 27,
    "selected_site_count": 27,
    "selection_namespace": _NAMESPACE,
    "selection_seed": _SELECTION_SEED,
    "validation_only": True,
}
_PLAN_SHA256 = "80e02db14d1ef9815fcf756cd0d4fe51e6ef8dd60abb8bab7f392d79f333d9cb"


@dataclass(frozen=True)
class R6bConfirmationSelection:
    """A sealed, metadata-only R6b confirmation selection."""

    blocks: tuple[BlockRecord, ...]
    prior_coordinate_sha256: str
    selected_coordinate_sha256: str
    selected_oklahoma_site_sha256: str
    candidate_universe_sha256: str
    plan_sha256: str
    preregistration_sha256: str
    selection_seed: int
    embargo_days: int


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity_hash(site_ids: Iterable[str]) -> str:
    return _sha256(_canonical_bytes(sorted(site_ids)))


def _coordinate_hash(coordinates: Iterable[tuple[str, str, str]]) -> str:
    return _sha256(_canonical_bytes([list(item) for item in sorted(coordinates)]))


def r6b_oklahoma_candidate_sha256() -> str:
    """Recompute the fixed audited-OK20 identity hash."""

    actual = _identity_hash(EXPECTED_R6B_OKLAHOMA_SITE_IDS)
    if actual != _CANDIDATE_SHA256:
        raise ValueError("r6b Oklahoma candidate universe does not match its fixed hash")
    return actual


def r6b_nrel_manifest_sha256() -> str:
    """Return the frozen canonical bytes seal for the R6b NREL manifest."""

    return _NREL_MANIFEST_SHA256


def r6b_nrel_selection_audit_sha256() -> str:
    """Return the frozen canonical bytes seal for the R6b NREL selection audit."""

    return _NREL_SELECTION_AUDIT_SHA256


def validate_r6b_oklahoma_candidate_ids(site_ids: Iterable[str]) -> tuple[str, ...]:
    """Return the canonical OK20 IDs only when the caller supplied exactly that set."""

    supplied = tuple(site_ids)
    if (
        len(supplied) != 20
        or len(set(supplied)) != 20
        or tuple(sorted(supplied)) != EXPECTED_R6B_OKLAHOMA_SITE_IDS
        or _identity_hash(supplied) != r6b_oklahoma_candidate_sha256()
    ):
        raise ValueError("r6b requires the exact audited twenty-site Oklahoma candidate universe")
    return EXPECTED_R6B_OKLAHOMA_SITE_IDS


def r6b_confirmation_plan_payload() -> dict[str, object]:
    """Return a detached copy of the immutable R6b selection plan."""

    return json.loads(json.dumps(_FIXED_PLAN))


def r6b_confirmation_plan_sha256() -> str:
    """Recompute the fixed canonical R6b plan seal."""

    actual = _sha256(_canonical_bytes(r6b_confirmation_plan_payload()))
    if actual != _PLAN_SHA256:
        raise ValueError("r6b confirmation plan does not match its fixed hash")
    return actual


def _pinned_plan() -> dict[str, object]:
    actual = r6b_confirmation_plan_sha256()
    payload = r6b_confirmation_plan_payload()
    if _sha256(_canonical_bytes(payload)) != actual:
        raise ValueError("r6b confirmation plan changed during validation")
    preregistration_sha256 = r6b_preregistration_sha256()
    preregistration = r6b_preregistration_payload()
    if _sha256(_canonical_bytes(preregistration)) != preregistration_sha256:
        raise ValueError("r6b preregistration changed during cross-binding")
    overlapping_fields = (
        "revision",
        "predecessor_revision",
        "catalog_sha256",
        "candidate_universe_sha256",
        "candidate_universe_site_count",
        "prior_coordinate_sha256",
        "prior_coordinate_count",
        "selection_seed",
        "embargo_days",
        "dkasc_site_count",
        "dkasc_blocks_per_site",
        "oklahoma_site_count",
        "oklahoma_blocks_per_site",
        "selected_block_count",
        "selected_site_count",
        "held_out_allowed",
        "range_definition_sha256",
    )
    if any(
        type(payload.get(field)) is not type(preregistration.get(field))
        or payload.get(field) != preregistration.get(field)
        for field in overlapping_fields
    ):
        raise ValueError("r6b plan and preregistration cross-binding is invalid")
    expected_embargo = pd.Timedelta(days=int(preregistration["embargo_days"]))
    if (
        type(_SELECTION_SEED) is not int
        or _SELECTION_SEED != preregistration.get("selection_seed")
        or type(_EMBARGO_DAYS) is not int
        or _EMBARGO_DAYS != preregistration.get("embargo_days")
        or type(_EMBARGO) is not pd.Timedelta
        or _EMBARGO != expected_embargo
        or type(payload.get("selection_namespace")) is not str
        or payload.get("selection_namespace") != _NAMESPACE
        or type(payload.get("validation_only")) is not bool
        or payload.get("validation_only") is not True
        or type(payload.get("oklahoma_pairwise_embargo")) is not bool
        or payload.get("oklahoma_pairwise_embargo") is not True
        or payload.get("candidate_universe_sha256") != r6b_oklahoma_candidate_sha256()
        or type(payload.get("preregistration_sha256")) is not str
        or payload.get("preregistration_sha256") != preregistration_sha256
    ):
        raise ValueError("r6b confirmation plan runtime/preregistration cross-binding is invalid")
    return payload


def _coordinate(value: object) -> tuple[str, str, str]:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError("r6b prior coordinates must be nonempty string triples")
    site_id, start, end = value
    try:
        parsed_start = pd.Timestamp(start)
        parsed_end = pd.Timestamp(end)
    except (TypeError, ValueError) as error:
        raise ValueError("r6b prior coordinates are not ISO8601") from error
    if (
        parsed_start.tz is None
        or parsed_end.tz is None
        or parsed_end < parsed_start
        or parsed_start.isoformat() != start
        or parsed_end.isoformat() != end
    ):
        raise ValueError("r6b prior coordinates are noncanonical intervals")
    return site_id, start, end


def _validate_prior(
    values: Iterable[tuple[str, str, str]], dkasc_site_ids: frozenset[str]
) -> tuple[tuple[str, str, str], ...]:
    unsorted = tuple(_coordinate(value) for value in values)
    if len(unsorted) != 75 or len(set(unsorted)) != 75:
        raise ValueError("r6b prior coordinates must be exactly seventy-five unique rows")
    prior = tuple(sorted(unsorted))
    if _coordinate_hash(prior) != _PRIOR_SHA256:
        raise ValueError("r6b prior coordinates do not match the retained fixed hash")
    counts: dict[str, int] = {}
    for site_id, _start, _end in prior:
        counts[site_id] = counts.get(site_id, 0) + 1
    r5_ok = frozenset(EXPECTED_AUXILIARY_OK_SITE_IDS)
    if (
        set(counts) != dkasc_site_ids | r5_ok
        or any(counts.get(site_id) != 5 for site_id in dkasc_site_ids)
        or any(counts.get(site_id) != 3 for site_id in r5_ok)
        or r5_ok & frozenset(EXPECTED_R6B_OKLAHOMA_SITE_IDS)
    ):
        raise ValueError("r6b prior coordinate site distribution is invalid")
    return prior


def _valid_raw_hashes(value: object) -> bool:
    return isinstance(value, tuple) and bool(value) and tuple(sorted(set(value))) == value and all(
        isinstance(item, str)
        and len(item) == 64
        and all(character in "0123456789abcdef" for character in item)
        for item in value
    )


def _validate_site(site: NormalizedSiteSeries, *, dataset: str, role: str, state: str | None) -> None:
    if not _valid_raw_hashes(site.raw_hashes):
        raise ValueError("r6b selector source raw hashes are invalid")
    if (
        site.dataset != dataset
        or site.split != "validation"
        or site.spatial_role != role
        or site.spatial_role in {"test", "zero_parameter_test"}
        or (state is not None and site.state != state)
        or not role_allows_split(
            site.dataset,
            site.spatial_role or "",
            site.state,
            "validation",
            site_id=site.site_id,
        )
    ):
        raise ValueError("r6b selector source role/split is invalid or held-out")


def _validate_block(block: BlockRecord, site: NormalizedSiteSeries) -> None:
    digest = stable_hash(
        block.dataset,
        block.site_id,
        block.split,
        block.start.isoformat(),
        block.end.isoformat(),
    )
    expected_end = block.start + pd.Timedelta(minutes=STEP_MINUTES * (BLOCK_STEPS - 1))
    if (
        block.dataset != site.dataset
        or block.site_id != site.site_id
        or block.split != "validation"
        or block.spatial_role != site.spatial_role
        or block.state != site.state
        or block.block_id != digest[:20]
        or block.selection_hash != digest
        or block.raw_hashes != site.raw_hashes
        or not _valid_raw_hashes(block.raw_hashes)
        or block.n_steps != BLOCK_STEPS
        or block.end != expected_end
        or block.quarter != f"{block.start.year}Q{block.start.quarter}"
    ):
        raise ValueError("r6b selector received a noncanonical block record")


def _metadata_blocks(site: NormalizedSiteSeries) -> tuple[BlockRecord, ...]:
    metadata_site = replace(site, power=np.zeros(len(site.timestamps), dtype=float))
    blocks = tuple(block for _, block in _valid_blocks(metadata_site))
    for block in blocks:
        _validate_block(block, site)
    return blocks


def _gap_to_coordinate(block: BlockRecord, coordinate: tuple[str, str, str]) -> pd.Timedelta:
    _site_id, start, end = coordinate
    prior_start, prior_end = pd.Timestamp(start), pd.Timestamp(end)
    if block.end < prior_start:
        return prior_start - block.end
    if prior_end < block.start:
        return block.start - prior_end
    return pd.Timedelta(0)


def _eligible_blocks(
    site: NormalizedSiteSeries, prior: tuple[tuple[str, str, str], ...]
) -> tuple[BlockRecord, ...]:
    output = []
    for block in _metadata_blocks(site):
        coordinate = (block.site_id, block.start.isoformat(), block.end.isoformat())
        if coordinate in prior:
            continue
        if any(
            block.site_id == old[0] and _gap_to_coordinate(block, old) < _EMBARGO
            for old in prior
        ):
            continue
        output.append(block)
    return tuple(output)


def _block_rank(block: BlockRecord) -> str:
    return stable_hash(
        _NAMESPACE,
        _SELECTION_SEED,
        block.site_id,
        block.start.isoformat(),
        block.end.isoformat(),
    )


def _oklahoma_site_rank(site_id: str) -> str:
    return stable_hash(_OK_SITE_NAMESPACE, _SELECTION_SEED, site_id)


def _post_validate(
    blocks: tuple[BlockRecord, ...],
    sources: dict[str, NormalizedSiteSeries],
    prior: tuple[tuple[str, str, str], ...],
) -> None:
    coordinates = tuple(
        (block.site_id, block.start.isoformat(), block.end.isoformat()) for block in blocks
    )
    dkasc = tuple(block for block in blocks if block.dataset == "dkasc")
    oklahoma = tuple(
        block for block in blocks if block.dataset == "nrel" and block.state == "OK"
    )
    if (
        len(blocks) != 27
        or len(dkasc) != 12
        or len(oklahoma) != 15
        or len({block.site_id for block in blocks}) != 27
        or len(set(coordinates)) != 27
        or len({block.block_id for block in blocks}) != 27
        or any(sum(other.site_id == block.site_id for other in blocks) != 1 for block in blocks)
    ):
        raise ValueError("r6b selection cardinality or identities are invalid")
    for block in blocks:
        source = sources.get(block.site_id)
        if source is None:
            raise ValueError("r6b selection source identity is invalid")
        _validate_block(block, source)
    if set(coordinates) & set(prior):
        raise ValueError("r6b selection overlaps a retained prior coordinate")
    if any(
        block.site_id == old[0] and _gap_to_coordinate(block, old) < _EMBARGO
        for block in blocks
        for old in prior
    ):
        raise ValueError("r6b selection violates the prior embargo")


def select_r6b_confirmation_blocks(
    dkasc_sites: Iterable[NormalizedSiteSeries],
    oklahoma_sites: Iterable[NormalizedSiteSeries],
    prior_coordinates: Iterable[tuple[str, str, str]],
) -> R6bConfirmationSelection:
    """Select one metadata-only block at 12 DKASC and 15 audited Oklahoma sites."""

    plan = _pinned_plan()
    dkasc = tuple(dkasc_sites)
    oklahoma = tuple(oklahoma_sites)
    if len(dkasc) != 12 or len({site.site_id for site in dkasc}) != 12:
        raise ValueError("r6b selection requires exactly twelve distinct DKASC sites")
    validate_r6b_oklahoma_candidate_ids(site.site_id for site in oklahoma)
    prior = _validate_prior(prior_coordinates, frozenset(site.site_id for site in dkasc))

    selected: list[BlockRecord] = []
    for site in sorted(dkasc, key=lambda item: item.site_id):
        _validate_site(site, dataset="dkasc", role="source_train", state=None)
        eligible = _eligible_blocks(site, prior)
        if not eligible:
            raise ValueError(f"DKASC site {site.site_id!r} lacks an eligible R6b block")
        selected.append(min(eligible, key=_block_rank))

    feasible_oklahoma: dict[str, BlockRecord] = {}
    for site in sorted(oklahoma, key=lambda item: item.site_id):
        _validate_site(site, dataset="nrel", role="validation_calibration", state="OK")
        eligible = _eligible_blocks(site, prior)
        if eligible:
            feasible_oklahoma[site.site_id] = min(eligible, key=_block_rank)
    if len(feasible_oklahoma) < 15:
        raise ValueError("fewer than fifteen audited Oklahoma sites have an eligible R6b block")
    chosen_oklahoma = tuple(
        sorted(feasible_oklahoma, key=_oklahoma_site_rank)[:15]
    )
    selected.extend(feasible_oklahoma[site_id] for site_id in chosen_oklahoma)

    blocks = tuple(
        sorted(
            selected,
            key=lambda block: (
                block.dataset,
                block.site_id,
                block.start,
                block.end,
                block.block_id,
            ),
        )
    )
    sources = {site.site_id: site for site in (*dkasc, *oklahoma)}
    if len(sources) != 32:
        raise ValueError("r6b source identities overlap across candidate cohorts")
    _post_validate(blocks, sources, prior)
    coordinates = tuple(
        (block.site_id, block.start.isoformat(), block.end.isoformat()) for block in blocks
    )
    return R6bConfirmationSelection(
        blocks=blocks,
        prior_coordinate_sha256=_coordinate_hash(prior),
        selected_coordinate_sha256=_coordinate_hash(coordinates),
        selected_oklahoma_site_sha256=_identity_hash(chosen_oklahoma),
        candidate_universe_sha256=r6b_oklahoma_candidate_sha256(),
        plan_sha256=r6b_confirmation_plan_sha256(),
        preregistration_sha256=str(plan["preregistration_sha256"]),
        selection_seed=_SELECTION_SEED,
        embargo_days=_EMBARGO_DAYS,
    )


__all__ = [
    "EXPECTED_R6B_OKLAHOMA_SITE_IDS",
    "R6bConfirmationSelection",
    "r6b_confirmation_plan_payload",
    "r6b_confirmation_plan_sha256",
    "r6b_oklahoma_candidate_sha256",
    "select_r6b_confirmation_blocks",
    "validate_r6b_oklahoma_candidate_ids",
]
