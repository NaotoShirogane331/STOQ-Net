"""Auditable DataLoader construction for prepared window datasets."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


def loader_settings(workers: int) -> dict[str, Any]:
    """Return throughput settings that do not change batch or sample semantics."""

    if workers < 0:
        raise ValueError("num_workers must be non-negative")
    settings: dict[str, Any] = {
        "num_workers": int(workers),
        "pin_memory": torch.cuda.is_available(),
    }
    if workers > 0:
        # Keep mmap-backed workers alive across epochs and prepare the next two
        # batches while CUDA is processing the current batch. DataLoader keeps
        # its default in-order delivery, so sampling order is unchanged.
        settings.update(
            persistent_workers=True,
            prefetch_factor=2,
        )
    return settings


def build_window_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
) -> DataLoader:
    """Build a loader with parallel prefetch but unchanged optimization inputs."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        **loader_settings(workers),
    )
