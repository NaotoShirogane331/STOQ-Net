"""Leakage-resistant internal development splits for chronological windows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset, Subset


@dataclass(frozen=True)
class ChronologicalDevelopmentSplit:
    """Training/development subsets plus an auditable target embargo."""

    train: Subset[Any]
    development: Subset[Any]
    audit: dict[str, Any]


def chronological_development_split(
    dataset: Dataset[Any],
    *,
    fraction: float,
    horizon_steps: int,
    origin_stride_steps: int,
) -> ChronologicalDevelopmentSplit:
    """Reserve each shard's chronological tail with no overlapping targets."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("development fraction must lie in (0, 1)")
    if horizon_steps <= 0 or origin_stride_steps <= 0:
        raise ValueError("horizon_steps and origin_stride_steps must be positive")
    raw_shards = getattr(dataset, "shards", None)
    shards = tuple(raw_shards) if raw_shards is not None else (dataset,)
    if not shards:
        raise ValueError("development split requires at least one shard")

    embargo_origins = max(0, math.ceil(horizon_steps / origin_stride_steps) - 1)
    train_indices: list[int] = []
    development_indices: list[int] = []
    shard_audit: list[dict[str, int]] = []
    offset = 0
    for shard_index, shard in enumerate(shards):
        length = len(shard)
        development_count = math.ceil(length * fraction)
        development_start = length - development_count
        train_count = development_start - embargo_origins
        if train_count < 1 or development_count < 1:
            raise ValueError(
                f"shard {shard_index} is too short for the requested development "
                "fraction and target embargo"
            )
        train_indices.extend(range(offset, offset + train_count))
        development_indices.extend(
            range(offset + development_start, offset + length)
        )
        shard_audit.append(
            {
                "shard": shard_index,
                "total": length,
                "train_count": train_count,
                "development_count": development_count,
                "embargo_count": embargo_origins,
                "development_start": development_start,
            }
        )
        offset += length
    if offset != len(dataset):
        raise ValueError("shard lengths do not sum to the dataset length")

    audit: dict[str, Any] = {
        "method": "per_shard_chronological_tail",
        "fraction": float(fraction),
        "horizon_steps": int(horizon_steps),
        "origin_stride_steps": int(origin_stride_steps),
        "embargo_origins": embargo_origins,
        "train_count": len(train_indices),
        "development_count": len(development_indices),
        "embargo_count": embargo_origins * len(shards),
        "shards": shard_audit,
    }
    return ChronologicalDevelopmentSplit(
        train=Subset(dataset, train_indices),
        development=Subset(dataset, development_indices),
        audit=audit,
    )


__all__ = ["ChronologicalDevelopmentSplit", "chronological_development_split"]
