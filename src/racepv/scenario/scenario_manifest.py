"""Scenario preparation, cohort indexing, layout, and storage planning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cohort import NormalizedSiteSeries, select_cohort, validate_split_isolation
from .compact_storage import (
    PREFLIGHT_LIMIT_BYTES,
    compact_storage_estimate,
    prediction_storage_contract,
    write_shared_cohort,
)
from .constants import (
    ABLATIONS,
    BASELINES,
    EVIDENCE_METADATA,
    RACE_PV,
    SEEDS,
    VALIDATION_SENSITIVITIES,
)
from .loader import load_normalized_sites
from .scenario_artifacts import (
    DISCLOSURE,
    atomic_bytes,
    canonical_bytes,
    register_stage_outputs,
    sha256_file,
    write_json,
)

NUMBERED_LAYOUT = {
    "01": "01-prepare",
    "02": "02-predictions",
    "03": "03-evaluations",
    "04": "04-aggregate",
    "05": "05-tables",
    "06": "06-figures",
    "07": "07-paper",
    "08": "08-scripts",
}

ACCEPTANCE_RANGES = {
    "daylight_nrmse_improvement_pct": {"minimum": 4.0, "maximum": 8.0},
    "mean_quantile_crps_improvement_pct": {"minimum": 5.0, "maximum": 10.0},
    "ramp_nrmse_improvement_pct": {"minimum": 8.0, "maximum": 14.0},
    "ramp_qcrps_improvement_pct": {"minimum": 9.0, "maximum": 16.0},
    "ramp_aucpr_absolute_gain": {"minimum": 0.04, "maximum": 0.08},
    "stable_picp90": {"minimum": 0.89, "maximum": 0.92},
    "ramp_picp90": {"minimum": 0.87, "maximum": 0.91},
    "state_lead_interval_score_improvement_pct": {"minimum": 4.0, "maximum": 8.0},
    "yulara_positive_skill_sites": {"minimum": 4.0, "maximum": 5.0},
    "arizona_positive_skill_sites": {"minimum": 15.0, "maximum": 18.0},
    "near_flat_report_leads": {"minimum": 0.0, "maximum": 1.0},
    "bh_significant_site_lead_fraction": {"minimum": 0.55, "maximum": 0.70},
    "neural_seed_relative_sd": {"minimum": 0.015, "maximum": 0.04},
    "correct_expert_mean_gate": {"minimum": 0.55, "maximum": 0.70},
    "normalized_gate_entropy": {"minimum": 0.60, "maximum": 0.82},
    "maximum_expert_global_share": {"minimum": 0.0, "maximum": 0.85},
}


def _smoke_sites(split: str) -> tuple[NormalizedSiteSeries, ...]:
    base = {"validation": "2024-01-01", "calibration": "2024-03-01", "test": "2024-05-01"}[
        split
    ]
    role = "zero_parameter_test" if split == "test" else "source_train"
    index = pd.date_range(base, periods=14 * 96, freq="15min", tz="UTC")
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60
    daylight = (hour >= 6) & (hour < 18)
    power = np.where(daylight, np.sin((hour - 6) * np.pi / 12) * 0.78, 0.0)
    return (
        NormalizedSiteSeries(
            "yulara_00" if split == "test" else "dkasc_00",
            "dkasc",
            index,
            power,
            daylight,
            split,
            (f"smoke-{split}-0",),
            role,
        ),
    )


def _cohort(split: str):
    """Return the deterministic tiny real-shaped cohort used by smoke execution."""
    cohort = select_cohort(_smoke_sites(split), split=split)
    # One full-daylight origin keeps every lead band calibratable; one sunset
    # origin makes the same tiny fixture exercise night invariants.
    indices = np.asarray([22, 60])
    return replace(
        cohort,
        targets=cohort.targets[indices],
        daylight=cohort.daylight[indices],
        timestamps=cohort.timestamps[indices],
        day_ids=cohort.day_ids[indices],
        site_ids=cohort.site_ids[indices],
        source_indices=cohort.source_indices[indices],
        previous_targets=(
            cohort.previous_targets[indices]
            if cohort.previous_targets is not None
            else None
        ),
        block_ids=cohort.block_ids[indices] if cohort.block_ids is not None else None,
    )


def _prepared_cohort(manifest_path: Path, split: str):
    return select_cohort(load_normalized_sites(manifest_path, split=split), split=split)


def stage_cohort(manifest: dict[str, Any], split: str):
    if manifest["smoke"]:
        return _cohort(split)
    source = manifest.get("prepared_cohort_manifest")
    expected = manifest.get("prepared_cohort_manifest_sha256")
    if not source or not expected:
        raise ValueError("scenario has no prepared cohort manifest")
    source_path = Path(source)
    if not source_path.is_file() or sha256_file(source_path) != expected:
        raise ValueError("prepared cohort manifest is missing or its hash changed")
    return _prepared_cohort(source_path, split)


def all_profiles(*, include_sensitivities: bool) -> tuple[Any, ...]:
    base = (*BASELINES, RACE_PV, *ABLATIONS)
    return (*base, *VALIDATION_SENSITIVITIES) if include_sensitivities else base


def matrix() -> tuple[tuple[str, Any], ...]:
    models = all_profiles(include_sensitivities=False)
    return (
        tuple(("validation", profile) for profile in models)
        + tuple(("test", profile) for profile in models)
        + tuple(("validation", profile) for profile in VALIDATION_SENSITIVITIES)
    )


def run_id(protocol: str, model: str, seed: int, split: str) -> str:
    return f"{protocol}/{model}/{seed}/{split}"


def cohort_sha256(cohort: Any) -> str:
    """Hash a cohort's labels, identities, and selected-block provenance."""

    digest = hashlib.sha256()
    for array in (
        cohort.targets,
        cohort.daylight,
        np.asarray(
            [[pd.Timestamp(value).value for value in row] for row in cohort.timestamps],
            dtype=np.int64,
        ),
        cohort.site_ids.astype("U"),
        cohort.source_indices,
        cohort.previous_targets,
    ):
        value = np.ascontiguousarray(array)
        digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    digest.update(canonical_bytes([record.block_id for record in cohort.blocks]))
    return digest.hexdigest()


_cohort_hash = cohort_sha256


def _storage_estimate(cohorts: dict[str, Any], protocol: str) -> dict[str, Any]:
    return compact_storage_estimate(
        {
            protocol: {
                split: int(cohorts[split].targets.shape[0])
                for split in ("validation", "test")
            }
        },
        calibration_origin_counts={
            protocol: int(cohorts["calibration"].targets.shape[0])
        },
        protocol_block_counts={
            protocol: {
                split: len(cohorts[split].blocks) for split in ("validation", "test")
            }
        },
    )


def _write_layout(root: Path) -> list[Path]:
    outputs = []
    destinations = {
        "01": "manifest.json and cohort-index.json",
        "02": "shared cohorts, compact run NPZs, one audit CSV.gz, JSON, and aliases",
        "03": "evaluations/ and evaluations.json",
        "04": "aggregate.json and selection.json",
        "05": "exports/tables/",
        "06": "exports/figures/",
        "07": "exports/paper/",
        "08": "verification.json, verification.txt, and artifact-index.json",
    }
    for number, directory in NUMBERED_LAYOUT.items():
        path = root / directory / "README.md"
        atomic_bytes(
            path,
            (
                f"# {directory}\n\nPlacement pointer: `{destinations[number]}`. "
                f"This directory is part of the numbered placement contract.\n\n{DISCLOSURE}\n"
            ).encode(),
        )
        outputs.append(path)
    root_readme = root / "README.md"
    atomic_bytes(
        root_readme,
        (
            "# Q2 ideal scenario bundle\n\n"
            "## Placement contract\n\n"
            + "\n".join(
                f"- `{number}` → `{directory}/`; see its README for the canonical artifact path."
                for number, directory in NUMBERED_LAYOUT.items()
            )
            + f"\n\n{DISCLOSURE}\n"
        ).encode("utf-8"),
    )
    outputs.append(root_readme)
    return outputs


def _require_compact_generation_gate(root: Path) -> None:
    gate_root = next(
        (
            candidate
            for candidate in (root.resolve(), *root.resolve().parents)
            if (candidate / "00-manifest" / "preflight-4a.json").is_file()
            and (
                candidate / "00-manifest" / "compact-shared-v2-validation.json"
            ).is_file()
        ),
        None,
    )
    if gate_root is None:
        raise ValueError("4B is locked until compact_shared_v2 validation and 4A pass")
    preflight_path = gate_root / "00-manifest" / "preflight-4a.json"
    validation_path = gate_root / "00-manifest" / "compact-shared-v2-validation.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    storage = preflight.get("storage", {})
    valid = (
        preflight.get("generation_ready") is True
        and storage.get("schema") == "compact_shared_v2"
        and storage.get("total_bytes", PREFLIGHT_LIMIT_BYTES + 1)
        <= PREFLIGHT_LIMIT_BYTES
        and validation.get("generation_gate_unlocked") is True
        and validation.get("preflight_sha256") == sha256_file(preflight_path)
        and validation.get("smoke_matrix_missing_runs") == 0
        and validation.get("full_test_suite_passed") is True
    )
    if not valid:
        raise ValueError("4B is locked: compact_shared_v2 validation gate is stale or failed")


def prepare(
    root: Path,
    *,
    smoke: bool,
    manifest_path: Path | None,
    candidate_seeds: list[int],
    protocol: str = "dkasc-yulara",
    generation_config: dict[str, Any] | None = None,
    generation_config_sha256: str | None = None,
) -> dict[str, Any]:
    if not smoke and manifest_path is None:
        raise ValueError("non-smoke prepare requires a supplied prepared cohort manifest")
    if manifest_path is not None and not manifest_path.is_file():
        raise ValueError("prepared cohort manifest does not exist")
    if not smoke:
        _require_compact_generation_gate(root)
    if not candidate_seeds or any(seed < 0 for seed in candidate_seeds):
        raise ValueError("candidate seeds must be non-negative integers")
    if (generation_config is None) != (generation_config_sha256 is None):
        raise ValueError("generation config and SHA-256 must be supplied together")
    if generation_config is not None:
        from .realism import ScenarioGenerationConfig, canonical_config_sha256

        resolved_config = ScenarioGenerationConfig(**generation_config)
        if canonical_config_sha256(resolved_config) != generation_config_sha256:
            raise ValueError("generation config SHA-256 mismatch")
    cohorts = (
        {split: _cohort(split) for split in ("validation", "calibration", "test")}
        if smoke
        else {
            split: _prepared_cohort(manifest_path, split)
            for split in ("validation", "calibration", "test")
        }
    )
    validate_split_isolation(*cohorts.values())
    identities = [
        {
            "run_id": run_id(protocol, profile.name, seed, split),
            "protocol": protocol,
            "model": profile.name,
            "seed": seed,
            "split": split,
        }
        for split, profile in matrix()
        for seed in SEEDS
    ]
    estimate = _storage_estimate(cohorts, protocol)
    if estimate["total_bytes"] > PREFLIGHT_LIMIT_BYTES:
        raise ValueError("scenario storage estimate exceeds the 40 GiB preflight limit")
    cohort_index = {
        **dict(EVIDENCE_METADATA),
        "disclosure": DISCLOSURE,
        "cohorts": {
            split: {
                "origin_count": int(cohort.targets.shape[0]),
                "horizon": int(cohort.targets.shape[1]),
                "cohort_sha256": _cohort_hash(cohort),
                "site_ids": sorted(set(cohort.site_ids[:, 0].tolist())),
                "origin_timestamps": [str(value) for value in cohort.timestamps[:, 0]],
                "blocks": [
                    {
                        "block_id": block.block_id,
                        "site_id": block.site_id,
                        "dataset": block.dataset,
                        "split": block.split,
                        "start": block.start.isoformat(),
                        "end": block.end.isoformat(),
                        "raw_hashes": list(block.raw_hashes),
                        "spatial_role": block.spatial_role,
                        "state": block.state,
                    }
                    for block in cohort.blocks
                ],
            }
            for split, cohort in cohorts.items()
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    shared_cohorts = {
        split: write_shared_cohort(root, protocol, cohort)
        for split, cohort in cohorts.items()
    }
    cohort_path = root / "cohort-index.json"
    write_json(cohort_path, cohort_index)
    manifest = {
        **dict(EVIDENCE_METADATA),
        "disclosure": DISCLOSURE,
        "smoke": smoke,
        "protocols": [protocol],
        "prediction_storage": prediction_storage_contract(),
        "candidate_scenario_seeds": sorted(set(candidate_seeds)),
        "registered_replicate_seeds": list(SEEDS),
        "model_matrix": [
            {"split": split, "model": profile.name} for split, profile in matrix()
        ],
        "run_identities": identities,
        "storage_estimate": estimate,
        "storage_estimate_bytes": estimate["total_bytes"],
        "source_role_map": {
            "validation": "source_train",
            "calibration": "source_train",
            "test": "zero_parameter_test",
        },
        "cohorts": {
            split: {
                "blocks": [record.block_id for record in cohort.blocks],
                "origins": int(cohort.targets.shape[0]),
                "horizon": int(cohort.targets.shape[1]),
                "cohort_sha256": _cohort_hash(cohort),
            }
            for split, cohort in cohorts.items()
        },
        "cohort_index_path": cohort_path.name,
        "cohort_index_sha256": sha256_file(cohort_path),
        "shared_cohorts": shared_cohorts,
        "numbered_layout": dict(NUMBERED_LAYOUT),
        "acceptance_policy": {
            "mode": "not_enforced_smoke" if smoke else "enforce_non_smoke_ranges",
            "reason": (
                "tiny smoke cohort; ranges are carried but not evaluated"
                if smoke
                else "every declared numerical range must be recomputed and pass verification"
            ),
            "ranges": ACCEPTANCE_RANGES,
        },
    }
    if generation_config is not None:
        manifest["generation_config"] = generation_config
        manifest["generation_config_sha256"] = generation_config_sha256
    if manifest_path is not None:
        manifest["prepared_cohort_manifest"] = str(manifest_path.resolve())
        manifest["prepared_cohort_manifest_sha256"] = sha256_file(manifest_path)
    manifest_file = root / "manifest.json"
    write_json(manifest_file, manifest)
    layout_outputs = _write_layout(root)
    shared_outputs = [
        root / value[key]
        for value in shared_cohorts.values()
        for key in ("path",)
    ] + [
        (root / value["path"]).with_suffix(".json") for value in shared_cohorts.values()
    ]
    register_stage_outputs(
        root,
        "prepare",
        [manifest_file, cohort_path, *shared_outputs, *layout_outputs],
        source_paths=(),
    )
    return manifest


def load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise ValueError("scenario manifest is missing; run prepare first")
    return json.loads(path.read_text(encoding="utf-8"))


def write_candidate_screening(root: Path, report: Any) -> Path:
    """Persist one deterministic validation-only screening summary."""

    path = root / "00-manifest" / "realism-candidate-screening.json"
    write_json(path, report.to_payload())
    return path


def require_stage(root: Path, stage: str, paths: tuple[str, ...]) -> None:
    missing = [path for path in paths if not (root / path).exists()]
    if missing:
        raise ValueError(f"{stage} stage required before this stage: missing {', '.join(missing)}")


__all__ = [
    "NUMBERED_LAYOUT",
    "_cohort",
    "all_profiles",
    "cohort_sha256",
    "load_manifest",
    "matrix",
    "prepare",
    "require_stage",
    "run_id",
    "stage_cohort",
    "write_candidate_screening",
]
