"""Training-only PV ramp thresholds and three-state labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

import numpy as np
import pandas as pd


class RampState(IntEnum):
    STABLE = 0
    RAMP_UP = 1
    RAMP_DOWN = 2


IGNORE_STATE_LABEL = -100
RampMethod = Literal["q85", "q90", "q95", "fixed0.1"]


def _quantile_from_method(method: str) -> float | None:
    normalized = method.lower().replace("_", "").replace("-", "")
    aliases = {
        "q85": 0.85,
        "85": 0.85,
        "q90": 0.90,
        "90": 0.90,
        "q95": 0.95,
        "95": 0.95,
        "fixed0.1": None,
        "fixed01": None,
        "0.1": None,
    }
    if normalized not in aliases:
        raise ValueError("method must be one of q85, q90, q95, or fixed0.1")
    return aliases[normalized]


def ramp_change_30min(power_normalized: pd.Series) -> pd.Series:
    """Thirty-minute change for data sampled every 15 minutes."""

    power = pd.to_numeric(power_normalized, errors="coerce")
    return power - power.shift(2)


@dataclass
class RampLabeler:
    """Fit a site-specific ramp threshold using daylight training observations."""

    method: RampMethod | str = "q90"
    fixed_threshold: float = 0.1
    threshold_: float | None = field(default=None, init=False)
    training_start_: str | None = field(default=None, init=False)
    training_end_: str | None = field(default=None, init=False)
    n_training_ramps_: int = field(default=0, init=False)

    def fit(
        self,
        training_power_normalized: pd.Series,
        *,
        daylight_mask: pd.Series | np.ndarray | None = None,
        observed_mask: pd.Series | np.ndarray | None = None,
        partition: str = "train",
    ) -> RampLabeler:
        if partition != "train":
            raise ValueError("Ramp thresholds may only be fitted with partition='train'")
        quantile = _quantile_from_method(str(self.method))
        changes = ramp_change_30min(training_power_normalized)
        valid = changes.notna() & np.isfinite(changes)
        if daylight_mask is not None:
            daylight = pd.Series(
                np.asarray(daylight_mask, dtype=bool), index=changes.index, dtype=bool
            )
            # Both ends of the 30-minute difference must be in daylight.
            valid &= daylight & daylight.shift(2, fill_value=False)
        if observed_mask is not None:
            observed = pd.Series(
                np.asarray(observed_mask, dtype=bool), index=changes.index, dtype=bool
            )
            valid &= observed & observed.shift(2, fill_value=False)
        absolute = changes.loc[valid].abs()
        if absolute.empty:
            raise ValueError("No valid daylight 30-minute ramps are available in training data")

        if quantile is None:
            threshold = float(self.fixed_threshold)
        else:
            threshold = float(absolute.quantile(quantile))
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("Ramp threshold must be finite and positive")

        self.threshold_ = threshold
        self.n_training_ramps_ = int(len(absolute))
        if isinstance(training_power_normalized.index, pd.DatetimeIndex):
            self.training_start_ = training_power_normalized.index.min().isoformat()
            self.training_end_ = training_power_normalized.index.max().isoformat()
        return self

    def transform(
        self,
        power_normalized: pd.Series,
        *,
        daylight_mask: pd.Series | np.ndarray | None = None,
        observed_mask: pd.Series | np.ndarray | None = None,
        ignore_label: int = IGNORE_STATE_LABEL,
    ) -> pd.Series:
        if self.threshold_ is None:
            raise RuntimeError("RampLabeler must be fitted before transform")
        changes = ramp_change_30min(power_normalized)
        labels = pd.Series(
            int(RampState.STABLE), index=changes.index, dtype="int64", name="state_label"
        )
        labels.loc[changes >= self.threshold_] = int(RampState.RAMP_UP)
        labels.loc[changes <= -self.threshold_] = int(RampState.RAMP_DOWN)
        invalid = changes.isna() | ~np.isfinite(changes)
        if daylight_mask is not None:
            daylight = pd.Series(
                np.asarray(daylight_mask, dtype=bool), index=changes.index, dtype=bool
            )
            invalid |= ~daylight | ~daylight.shift(2, fill_value=False)
        if observed_mask is not None:
            observed = pd.Series(
                np.asarray(observed_mask, dtype=bool), index=changes.index, dtype=bool
            )
            invalid |= ~observed | ~observed.shift(2, fill_value=False)
        labels.loc[invalid] = int(ignore_label)
        return labels

    def fit_transform(
        self,
        training_power_normalized: pd.Series,
        *,
        daylight_mask: pd.Series | np.ndarray | None = None,
        observed_mask: pd.Series | np.ndarray | None = None,
        partition: str = "train",
        ignore_label: int = IGNORE_STATE_LABEL,
    ) -> pd.Series:
        return self.fit(
            training_power_normalized,
            daylight_mask=daylight_mask,
            observed_mask=observed_mask,
            partition=partition,
        ).transform(
            training_power_normalized,
            daylight_mask=daylight_mask,
            observed_mask=observed_mask,
            ignore_label=ignore_label,
        )

    def state_dict(self) -> dict[str, float | int | str | None]:
        if self.threshold_ is None:
            raise RuntimeError("RampLabeler has not been fitted")
        return {
            "method": str(self.method),
            "threshold": self.threshold_,
            "fixed_threshold": self.fixed_threshold,
            "n_training_ramps": self.n_training_ramps_,
            "training_start": self.training_start_,
            "training_end": self.training_end_,
        }
