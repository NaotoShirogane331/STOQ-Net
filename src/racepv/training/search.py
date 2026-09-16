"""Bounded validation search with an explicit maximum trial budget."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GridTrial:
    parameters: dict[str, Any]
    validation_score: float


def run_bounded_grid_search(
    parameter_grid: dict[str, Iterable[Any]],
    evaluate: Callable[[dict[str, Any]], float],
    *,
    max_trials: int = 15,
) -> list[GridTrial]:
    if max_trials < 1:
        raise ValueError("max_trials must be positive")
    names = tuple(sorted(parameter_grid))
    combinations = itertools.product(*(parameter_grid[name] for name in names))
    trials: list[GridTrial] = []
    for values in itertools.islice(combinations, max_trials):
        parameters = dict(zip(names, values, strict=True))
        trials.append(GridTrial(parameters, float(evaluate(parameters))))
    return sorted(trials, key=lambda trial: trial.validation_score)

