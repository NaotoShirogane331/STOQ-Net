"""Reproducibility and future-information leakage audits."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import atomic_json_dump

FORBIDDEN_FUTURE_TOKENS = (
    "observed",
    "measurement",
    "actual",
    "future_ghi",
    "future_dni",
    "future_dhi",
    "future_temperature",
    "future_cloud",
    "future_humidity",
    "future_wind",
)


@dataclass
class LeakageAuditReport:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    def require_pass(self) -> None:
        if not self.passed:
            raise RuntimeError("leakage audit failed:\n- " + "\n- ".join(self.errors))

    def save(self, path: str | Path) -> None:
        atomic_json_dump(asdict(self), path)


def audit_future_feature_names(
    feature_names: Sequence[str], allowed_names: Iterable[str]
) -> LeakageAuditReport:
    allowed = set(allowed_names)
    errors = []
    for name in feature_names:
        normalized = name.lower()
        if name not in allowed:
            errors.append(f"future feature {name!r} is not on the protocol allowlist")
        if any(token in normalized for token in FORBIDDEN_FUTURE_TOKENS):
            errors.append(f"future feature {name!r} appears to contain an observed future value")
    return LeakageAuditReport(
        passed=not errors,
        errors=errors,
        checks={"future_feature_allowlist": not errors},
    )


def audit_temporal_splits(
    split_timestamps: Mapping[str, Sequence[pd.Timestamp | np.datetime64 | str]],
    *,
    expected_order: Sequence[str] = ("train", "validation", "calibration", "test"),
) -> LeakageAuditReport:
    errors: list[str] = []
    previous_end: pd.Timestamp | None = None
    for name in expected_order:
        if name not in split_timestamps:
            errors.append(f"missing split {name!r}")
            continue
        values = pd.DatetimeIndex(split_timestamps[name])
        if values.empty:
            errors.append(f"split {name!r} is empty")
            continue
        if not values.is_monotonic_increasing:
            errors.append(f"split {name!r} is not chronological")
        if values.has_duplicates:
            errors.append(f"split {name!r} contains duplicate timestamps")
        if previous_end is not None and values.min() <= previous_end:
            errors.append(f"split {name!r} overlaps or precedes the prior split")
        previous_end = values.max()
    return LeakageAuditReport(
        passed=not errors,
        errors=errors,
        checks={"strict_chronology": not errors},
    )


def audit_train_only_statistics(
    fit_indices: Sequence[int], train_indices: Sequence[int], statistic_name: str
) -> LeakageAuditReport:
    fit_set = set(int(index) for index in fit_indices)
    train_set = set(int(index) for index in train_indices)
    leaked = sorted(fit_set - train_set)
    errors = (
        [f"{statistic_name} used {len(leaked)} observations outside the training split"]
        if leaked
        else []
    )
    return LeakageAuditReport(
        passed=not errors,
        errors=errors,
        checks={f"train_only:{statistic_name}": not errors},
    )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact_manifest(paths: Iterable[str | Path], destination: str | Path) -> None:
    entries = []
    for path in sorted((Path(item) for item in paths), key=str):
        entries.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    atomic_json_dump({"artifacts": entries}, destination)

