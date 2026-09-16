"""Prediction bundles, deterministic serialization, resume, and artifact DAG support."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .constants import EVIDENCE_METADATA, QUANTILES, RACE_PV, SEEDS
from .synthesis import synthesize_predictions

DISCLOSURE = (
    "evidence_status=hypothetical_oracle_conditioned; oracle_conditioned=true; "
    "registered_evidence_eligible=false"
)
MAX_BYTES = 60 * 1024**3


def canonical_bytes(payload: Any) -> bytes:
    def default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"cannot serialize {type(value).__name__}")

    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=default
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_bytes(path, canonical_bytes(payload) + b"\n")


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def artifact_id(stage: str, relative: str) -> str:
    return f"artifact:{stage}:{relative}"


def _new_index() -> dict[str, Any]:
    return {
        **dict(EVIDENCE_METADATA),
        "disclosure": DISCLOSURE,
        "schema": "append_only_artifact_dag_v2",
        "acceptance_schema": {
            "mode": "inherited_from_manifest",
            "reason": "resolved from the immutable prepared manifest",
        },
        "artifacts": {},
        "stage_runs": [],
        "verification_runs": [],
        "stages": {},
    }


def load_artifact_index(root: Path) -> dict[str, Any]:
    path = root / "artifact-index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else _new_index()


def artifact_root_digest(index: dict[str, Any]) -> str:
    evidence = {
        key: {
            "path": value["path"],
            "sha256": value["sha256"],
            "source_artifact_ids": value["source_artifact_ids"],
            "evidence_metadata": value["evidence_metadata"],
        }
        for key, value in sorted(index.get("artifacts", {}).items())
    }
    return hashlib.sha256(canonical_bytes(evidence)).hexdigest()


def register_stage_outputs(
    root: Path,
    stage: str,
    output_paths: Iterable[Path | str],
    *,
    source_paths: Iterable[Path | str] = (),
    source_paths_by_output: dict[str, Iterable[Path | str]] | None = None,
) -> None:
    """Append one immutable stage run using stable path-based artifact identities."""
    root = root.expanduser().resolve()
    index = load_artifact_index(root)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index["acceptance_schema"] = manifest.get("acceptance_policy", {})
    path_to_id = {entry["path"]: key for key, entry in index["artifacts"].items()}

    def relative(value: Path | str) -> str:
        path = Path(value)
        resolved = path.expanduser().resolve()
        if resolved.is_relative_to(root):
            return str(resolved.relative_to(root))
        return str(path)

    common_sources = [relative(value) for value in source_paths]
    output_relatives = sorted({relative(path) for path in output_paths})
    # Resolve same-stage edges independently of lexical output order.
    for value in output_relatives:
        path_to_id.setdefault(value, artifact_id(stage, value))
    output_ids: list[str] = []
    files: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    for value in output_relatives:
        candidate = root / value
        if not candidate.is_file():
            raise ValueError(f"stage {stage} output is missing: {value}")
        identifier = artifact_id(stage, value)
        sources = list(common_sources)
        if source_paths_by_output and value in source_paths_by_output:
            sources.extend(relative(item) for item in source_paths_by_output[value])
        missing_sources = sorted({item for item in sources if item not in path_to_id})
        if missing_sources:
            raise ValueError(
                f"unindexed source artifact for {value}: {', '.join(missing_sources)}"
            )
        source_ids = sorted({path_to_id[item] for item in sources})
        digest = sha256_file(candidate)
        entry = {
            "artifact_id": identifier,
            "path": value,
            "sha256": digest,
            "stage": stage,
            "type": candidate.suffix.lstrip(".") or "file",
            "evidence_metadata": dict(EVIDENCE_METADATA),
            "source_artifact_ids": source_ids,
        }
        prior = index["artifacts"].get(identifier)
        if prior is not None and prior != entry:
            raise ValueError(f"immutable artifact identity changed: {identifier}")
        index["artifacts"].setdefault(identifier, entry)
        path_to_id[value] = identifier
        output_ids.append(identifier)
        files[value] = digest
        entries.append(entry)
    stage_digest = hashlib.sha256(
        canonical_bytes({identifier: index["artifacts"][identifier] for identifier in output_ids})
    ).hexdigest()
    run = {
        "run_sequence": len(index["stage_runs"]) + 1,
        "stage": stage,
        "output_artifact_ids": output_ids,
        "source_artifact_ids": sorted(
            {
                source
                for identifier in output_ids
                for source in index["artifacts"][identifier]["source_artifact_ids"]
            }
        ),
        "stage_digest": stage_digest,
    }
    index["stage_runs"].append(run)
    index["stages"].setdefault(
        stage,
        {
            "sha256": stage_digest,
            "files": files,
            "artifact_entries": entries,
            "predecessor_stage_hashes": [
                item["stage_digest"] for item in index["stage_runs"][:-1]
            ],
        },
    )
    write_json(root / "artifact-index.json", index)


def supersede_final_stage_registration(root: Path, stage: str, *, reason: str) -> None:
    """Auditably remove an invalid final-stage registration before recomputation.

    The original artifact entries and hashes remain in ``superseded_stage_runs``.
    A stage with downstream consumers cannot use this repair path.
    """
    if not reason.strip():
        raise ValueError("supersession reason must be non-empty")
    root = root.expanduser().resolve()
    index = load_artifact_index(root)
    stage_runs = index.get("stage_runs", [])
    positions = [position for position, run in enumerate(stage_runs) if run.get("stage") == stage]
    if not positions:
        raise ValueError(f"stage is not registered: {stage}")
    if positions != list(range(positions[0], len(stage_runs))):
        raise ValueError(f"only the final registered stage can be superseded: {stage}")
    selected_runs = stage_runs[positions[0] :]
    artifact_ids = {
        identifier
        for run in selected_runs
        for identifier in run.get("output_artifact_ids", [])
    }
    artifacts = index.get("artifacts", {})
    downstream = [
        identifier
        for identifier, entry in artifacts.items()
        if identifier not in artifact_ids
        and artifact_ids.intersection(entry.get("source_artifact_ids", []))
    ]
    if downstream:
        raise ValueError(
            "cannot supersede a stage with downstream artifacts: " + ", ".join(sorted(downstream))
        )
    index.setdefault("superseded_stage_runs", []).append(
        {
            "stage": stage,
            "reason": reason.strip(),
            "artifacts": {
                identifier: artifacts[identifier] for identifier in sorted(artifact_ids)
            },
            "stage_runs": selected_runs,
            "stage_summary": index.get("stages", {}).get(stage),
        }
    )
    for identifier in artifact_ids:
        artifacts.pop(identifier, None)
    del stage_runs[positions[0] :]
    index.get("stages", {}).pop(stage, None)
    write_json(root / "artifact-index.json", index)


def record_verification(root: Path, verified_root_digest: str) -> None:
    index = load_artifact_index(root)
    if artifact_root_digest(index) != verified_root_digest:
        raise ValueError("artifact DAG changed during verification")
    index["verification_runs"].append(
        {
            "sequence": len(index["verification_runs"]) + 1,
            "verified_root_digest": verified_root_digest,
            "verification_outputs_excluded": ["verification.json", "verification.txt"],
        }
    )
    write_json(root / "artifact-index.json", index)


def _metrics(cohort: Any, quantiles: np.ndarray) -> float:
    from racepv.metrics import quantile_crps

    return float(quantile_crps(cohort.targets, quantiles, QUANTILES, mask=cohort.daylight))


def _prediction_paths(root: Path, identity: str) -> tuple[Path, Path, Path]:
    archive = root / "predictions" / f"{identity.replace('/', '__')}.npz"
    return archive, archive.with_suffix(".csv.gz"), archive.with_suffix(".json")


def _write_prediction(
    root: Path,
    cohort: Any,
    prediction: Any,
    model: str,
    seed: int,
    *,
    split: str,
    resume: bool,
    manifest: dict[str, Any],
) -> list[Path]:
    from .compact_storage import COMPACT_SCHEMA, compact_prediction_paths, ensure_bundle_space
    from .scenario_manifest import run_id

    protocol = str(manifest["protocols"][0])
    identity = run_id(protocol, model, seed, split)
    archive, sidecar_path = compact_prediction_paths(
        root, protocol, model, seed, split
    )
    existing = [archive.exists(), sidecar_path.exists()]
    if resume and any(existing) and not all(existing):
        raise ValueError(f"partial prediction state for {archive.name}")
    if resume and all(existing):
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if (
            sidecar.get("run_id") == identity
            and sidecar.get("content_sha256") == sha256_file(archive)
            and sidecar.get("storage_schema") == COMPACT_SCHEMA
        ):
            generation_config_id = getattr(prediction, "generation_config_id", None)
            generation_config_sha256 = getattr(prediction, "generation_config_sha256", None)
            if generation_config_id is not None or generation_config_sha256 is not None:
                if (
                    sidecar.get("generation_config_id") != generation_config_id
                    or sidecar.get("generation_config_sha256") != generation_config_sha256
                ):
                    raise ValueError(f"resume generation config mismatch for {archive.name}")
            return [archive, sidecar_path]
        raise ValueError(f"resume identity/hash mismatch for {archive.name}")
    arrays = {
        "quantiles": np.asarray(prediction.quantiles, dtype=np.float32),
        "state_probabilities": np.asarray(
            prediction.state_probabilities, dtype=np.float32
        ),
    }
    if prediction.expert_quantiles is not None:
        arrays.update(
            {
                "expert_quantiles": np.asarray(
                    prediction.expert_quantiles, dtype=np.float32
                ),
                "router_logits": np.asarray(prediction.router_logits, dtype=np.float32),
            }
        )
    ensure_bundle_space(root, sum(value.nbytes for value in arrays.values()) + 65_536)
    write_npz(archive, arrays)
    shared = manifest["shared_cohorts"][split]
    sidecar = {
        **dict(EVIDENCE_METADATA),
        "disclosure": DISCLOSURE,
        "storage_schema": COMPACT_SCHEMA,
        "run_id": identity,
        "protocol": protocol,
        "model": model,
        "seed": seed,
        "split": split,
        "path": str(archive.relative_to(root)),
        "content_sha256": sha256_file(archive),
        "cohort_path": shared["path"],
        "cohort_sha256": shared["sha256"],
        "audit_sample_path": "predictions/audit-sample.csv.gz",
        "audit_sample_sha256": "",
        "cohort_index_sha256": manifest["cohort_index_sha256"],
        "arrays": sorted(arrays),
    }
    generation_config_id = getattr(prediction, "generation_config_id", None)
    if generation_config_id is not None:
        sidecar["generation_config_id"] = generation_config_id
        sidecar["generation_config_sha256"] = getattr(
            prediction, "generation_config_sha256", None
        )
    write_json(sidecar_path, sidecar)
    return [archive, sidecar_path]


def generate(root: Path, *, resume: bool) -> None:
    from .compact_storage import compact_prediction_paths, write_audit_sample
    from .scenario_evaluate import sensitivity_settings
    from .scenario_manifest import all_profiles, load_manifest, matrix, run_id, stage_cohort

    manifest = load_manifest(root)
    generation_config = None
    if "generation_config" in manifest or "generation_config_sha256" in manifest:
        from .realism import ScenarioGenerationConfig, canonical_config_sha256

        payload = manifest.get("generation_config")
        expected = manifest.get("generation_config_sha256")
        if not isinstance(payload, dict) or not isinstance(expected, str):
            raise ValueError("scenario manifest has an incomplete generation config")
        generation_config = ScenarioGenerationConfig(**payload)
        if canonical_config_sha256(generation_config) != expected:
            raise ValueError("scenario manifest generation config hash mismatch")
    protocol = str(manifest["protocols"][0])
    cohorts = {split: stage_cohort(manifest, split) for split in ("validation", "test")}
    candidates = list(manifest["candidate_scenario_seeds"])
    base_scores: dict[str, list[dict[str, Any]]] = {}
    for profile in all_profiles(include_sensitivities=False):
        scores = []
        for scenario_seed in candidates:
            value = float(
                np.mean(
                    [
                        _metrics(
                            cohorts["validation"],
                            synthesize_predictions(
                                cohorts["validation"],
                                profile,
                                seed=seed,
                                scenario_seed=scenario_seed,
                                generation_config=generation_config,
                            ).quantiles,
                        )
                        for seed in SEEDS
                    ]
                )
            )
            scores.append({"seed": scenario_seed, "value": value})
        base_scores[profile.name] = scores
    profile_rows = []
    for profile in all_profiles(include_sensitivities=True):
        source_profile = RACE_PV.name if profile.validation_only else profile.name
        profile_rows.append(
            {
                "profile": profile.name,
                "source_profile": source_profile,
                "candidate_scores": [
                    {
                        "seed": item["seed"],
                        "metric": "validation_quantile_crps",
                        "value": item["value"],
                    }
                    for item in base_scores[source_profile]
                ],
            }
        )
    candidate_rows = []
    for scenario_seed in candidates:
        values = [
            next(item["value"] for item in row["candidate_scores"] if item["seed"] == scenario_seed)
            for row in profile_rows
        ]
        candidate_rows.append(
            {
                "seed": scenario_seed,
                "metric": "whole_30_profile_validation_quantile_crps",
                "value": float(np.mean(values)),
            }
        )
    accepted = min(candidate_rows, key=lambda row: (row["value"], row["seed"]))["seed"]
    for row in profile_rows:
        for score in row["candidate_scores"]:
            selected = score["seed"] == accepted
            score.update(
                {
                    "selection_status": "accepted" if selected else "rejected",
                    "reason": (
                        "lowest whole-scenario score with numeric seed tie-break"
                        if selected
                        else "higher whole-scenario score or numeric seed tie-break loss"
                    ),
                }
            )
        row["accepted_seed"] = accepted
        row["rejected"] = [score for score in row["candidate_scores"] if score["seed"] != accepted]
    audit_path = root / "candidate-audit.json"
    write_json(
        audit_path,
        {
            **dict(EVIDENCE_METADATA),
            "disclosure": DISCLOSURE,
            "candidate_scenario_seeds": candidates,
            "selection_metric": "whole_30_profile_validation_quantile_crps",
            "candidate_scores": candidate_rows,
            "accepted_seed": accepted,
            "tie_break": "lowest_numeric_scenario_seed",
            "profiles": profile_rows,
        },
    )

    outputs: list[Path] = [audit_path]
    sources_by_output: dict[str, list[Path | str]] = {}
    prediction_records: list[tuple[dict[str, Any], Any]] = []
    prediction_sidecars: list[Path] = []
    aliases: dict[str, str] = {}
    for split, profile in matrix():
        for seed in SEEDS:
            if profile.validation_only:
                identity = run_id(protocol, profile.name, seed, split)
                base_identity = run_id(protocol, RACE_PV.name, seed, split)
                base_archive, _ = compact_prediction_paths(
                    root, protocol, RACE_PV.name, seed, split
                )
                alias = (
                    root
                    / "predictions"
                    / "runs"
                    / protocol
                    / profile.name
                    / str(seed)
                    / f"{split}.alias.json"
                )
                write_json(
                    alias,
                    {
                        **dict(EVIDENCE_METADATA),
                        "disclosure": DISCLOSURE,
                        "storage_schema": "compact_shared_v2_alias",
                        "run_id": identity,
                        "protocol": protocol,
                        "model": profile.name,
                        "seed": seed,
                        "split": split,
                        "alias_of_run_id": base_identity,
                        "alias_of_path": str(base_archive.relative_to(root)),
                        "declared_intervention": sensitivity_settings(profile.name),
                    },
                )
                aliases[identity] = base_identity
                outputs.append(alias)
                sources_by_output[str(alias.relative_to(root))] = [
                    str(base_archive.relative_to(root)),
                    "manifest.json",
                ]
                continue
            produced = _write_prediction(
                root,
                cohorts[split],
                synthesize_predictions(
                    cohorts[split],
                    profile,
                    seed=seed,
                    scenario_seed=int(accepted),
                    generation_config=generation_config,
                ),
                profile.name,
                seed,
                split=split,
                resume=resume,
                manifest=manifest,
            )
            outputs.extend(produced)
            archive = produced[0]
            sidecar_path = produced[1]
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            prediction_records.append((sidecar, cohorts[split]))
            prediction_sidecars.append(sidecar_path)
            sources_by_output[str(archive.relative_to(root))] = [
                sidecar["cohort_path"]
            ]
    sample_path, sample_sidecar, sample_hash = write_audit_sample(
        root, prediction_records, aliases
    )
    outputs.extend((sample_path, sample_sidecar))
    sources_by_output[str(sample_path.relative_to(root))] = [
        sidecar["path"] for sidecar, _ in prediction_records
    ]
    sources_by_output[str(sample_sidecar.relative_to(root))] = [sample_path]
    for sidecar_path in prediction_sidecars:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["audit_sample_sha256"] = sample_hash
        write_json(sidecar_path, sidecar)
        sources_by_output[str(sidecar_path.relative_to(root))] = [
            sidecar["path"],
            sidecar["cohort_path"],
            sample_path,
        ]
    for alias_path in [path for path in outputs if path.name.endswith(".alias.json")]:
        sources_by_output[str(alias_path.relative_to(root))].append(sample_sidecar)
    register_stage_outputs(
        root,
        "generate",
        outputs,
        source_paths=("manifest.json", "cohort-index.json"),
        source_paths_by_output=sources_by_output,
    )


__all__ = [
    "DISCLOSURE",
    "MAX_BYTES",
    "artifact_root_digest",
    "atomic_bytes",
    "canonical_bytes",
    "generate",
    "load_artifact_index",
    "record_verification",
    "register_stage_outputs",
    "sha256_file",
    "supersede_final_stage_registration",
    "write_json",
    "write_npz",
]
