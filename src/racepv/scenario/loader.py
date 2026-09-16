"""Manifest-backed raw-label loading without redistributing source data."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from racepv.data import adapter_for, chronological_split_plan, load_manifest
from racepv.physics import ClearSkyFeatureBuilder, SiteMetadata

from .cohort import NormalizedSiteSeries


def role_allows_split(
    dataset: str,
    spatial_role: str,
    state: str | None,
    split: str,
    *,
    site_id: str | None = None,
) -> bool:
    """Enforce registered source/test geography before data are loaded."""
    dataset = dataset.lower()
    role = spatial_role.lower()
    normalized_state = None if state is None else state.upper()
    normalized_site = "" if site_id is None else site_id.lower()
    if dataset == "dkasc":
        source_site = normalized_site.startswith("dkasc_") or not normalized_site
        yulara_site = normalized_site.startswith("yulara_") or not normalized_site
        if split == "test":
            return role == "zero_parameter_test" and yulara_site
        if split in {"validation", "calibration"}:
            return role == "source_train" and source_site
        if split == "train":
            return role == "source_train" and source_site
        raise ValueError("split must be train, validation, calibration, or test")
    if dataset != "nrel":
        return False
    if split == "test":
        return role == "zero_parameter_test" and normalized_state == "AZ"
    if split in {"validation", "calibration"}:
        return role == "validation_calibration" and normalized_state == "OK"
    if split == "train":
        return role == "source_train" and normalized_state in {"AL", "CA", "NY"}
    raise ValueError("split must be train, validation, calibration, or test")


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_normalized_sites(
    manifest_path: str | Path, *, split: str, site_ids: set[str] | None = None
) -> tuple[NormalizedSiteSeries, ...]:
    """Load 15-minute labels from existing adapters and preserve input hashes."""
    path = Path(manifest_path).expanduser().resolve()
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "source_manifests" in payload:
            return _load_protocol_wrapper(path, payload, split=split, site_ids=site_ids)
    return _load_one_manifest(path, split=split, site_ids=site_ids, skip_disallowed=False)


def _load_protocol_wrapper(
    wrapper_path: Path,
    payload: dict[str, Any],
    *,
    split: str,
    site_ids: set[str] | None,
) -> tuple[NormalizedSiteSeries, ...]:
    sources = payload.get("source_manifests")
    if payload.get("schema") != "protocol_manifest_v1" or not isinstance(sources, list):
        raise ValueError("invalid protocol manifest wrapper")
    output: list[NormalizedSiteSeries] = []
    for source in sources:
        if not isinstance(source, dict) or not source.get("path") or not source.get("sha256"):
            raise ValueError("protocol source manifest requires path and sha256")
        source_path = Path(source["path"]).expanduser()
        if not source_path.is_absolute():
            source_path = wrapper_path.parent / source_path
        source_path = source_path.resolve()
        if not source_path.is_file() or _file_hash(source_path) != source["sha256"]:
            raise ValueError(f"source manifest hash mismatch: {source_path}")
        declared_site_ids = source.get("site_ids")
        source_site_ids: set[str] | None = None
        if declared_site_ids is not None:
            if (
                not isinstance(declared_site_ids, list)
                or not declared_site_ids
                or any(not isinstance(value, str) or not value for value in declared_site_ids)
                or len(set(declared_site_ids)) != len(declared_site_ids)
            ):
                raise ValueError("protocol source site_ids must be unique non-empty strings")
            source_site_ids = set(declared_site_ids)
            known_site_ids = {entry.site_id for entry in load_manifest(source_path).entries}
            unknown = sorted(source_site_ids - known_site_ids)
            if unknown:
                raise ValueError(f"protocol source has unknown site_ids: {', '.join(unknown)}")
        requested_site_ids = source_site_ids
        if site_ids is not None:
            requested_site_ids = (
                set(site_ids)
                if requested_site_ids is None
                else requested_site_ids & set(site_ids)
            )
        output.extend(
            _load_one_manifest(
                source_path,
                split=split,
                site_ids=requested_site_ids,
                skip_disallowed=True,
            )
        )
    if not output:
        raise ValueError("protocol manifest contains no role-eligible requested sites")
    return tuple(output)


def _load_one_manifest(
    manifest_path: Path,
    *,
    split: str,
    site_ids: set[str] | None,
    skip_disallowed: bool,
) -> tuple[NormalizedSiteSeries, ...]:
    manifest = load_manifest(manifest_path)
    output = []
    for entry in manifest.entries:
        if site_ids is not None and entry.site_id not in site_ids:
            continue
        if not role_allows_split(
            entry.dataset, entry.spatial_role, entry.state, split, site_id=entry.site_id
        ):
            if skip_disallowed:
                continue
            raise ValueError(
                f"site {entry.site_id!r} role/state is not allowed for {split}: "
                f"{entry.spatial_role}/{entry.state}"
            )
        adapter = adapter_for(entry, manifest.root)
        frame = adapter.load_resampled(entry)
        # The exact half-open chronological interval becomes part of the output
        # contract. It prevents a caller label from selecting all observations.
        partition = chronological_split_plan(len(frame.values))[split]
        values = frame.values.iloc[partition]
        capacity = entry.capacity_kw
        if capacity is None or not np.isfinite(capacity) or capacity <= 0:
            raise ValueError(f"site {entry.site_id!r} lacks a registered rated capacity")
        power = np.asarray(values["power"], dtype=float) / float(capacity)
        power = np.clip(power, 0.0, 1.05)
        # Solar geometry determines night, never the measured target: a cloudy
        # zero during daylight remains an evaluated zero rather than being
        # silently discarded as night.
        solar = ClearSkyFeatureBuilder(
            SiteMetadata(
                latitude=entry.latitude,
                longitude=entry.longitude,
                timezone=entry.timezone,
                capacity_kw=float(capacity),
                name=entry.site_id,
            ),
            allow_approximate_fallback=True,
        ).build_solar_irradiance(values.index)
        daylight = np.asarray(solar["is_solar_daylight"], dtype=bool)
        power[~daylight & np.isfinite(power)] = 0.0
        hashes = tuple(_file_hash(path) for path in adapter.resolve_files(entry))
        output.append(
            NormalizedSiteSeries(
                entry.site_id,
                entry.dataset,
                values.index,
                power,
                daylight,
                split,
                hashes,
                entry.spatial_role,
                entry.state,
            )
        )
    if not output and not skip_disallowed:
        raise ValueError("manifest selection contains no requested sites")
    return tuple(output)
