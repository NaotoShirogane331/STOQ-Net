"""Independent forecast/state objective for the reliability-gate A candidate.

This candidate objective has no expert-specific auxiliary loss. Its availability
is an engineering implementation and does not authorize formal training.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from .losses import state_brier_score, state_cross_entropy, trapezoidal_qcrps
from .models.config import DEFAULT_QUANTILES


class ReliabilityAObjective(nn.Module):
    """Daylight trapezoidal qCRPS plus flat state CE, each with weight one.

    Only ``quantiles`` and ``state_logits`` are consumed from model outputs.
    State diagnostics do not contribute an additional optimization term.
    """

    objective_id = "racepv.forecast-state-objective-without-specialist.v1"

    def __init__(self, quantiles: Sequence[float] = DEFAULT_QUANTILES) -> None:
        super().__init__()
        levels = torch.as_tensor(tuple(quantiles), dtype=torch.float32)
        if (
            levels.ndim != 1
            or levels.numel() == 0
            or not bool(((levels > 0.0) & (levels < 1.0)).all())
            or not bool((levels[1:] > levels[:-1]).all())
        ):
            raise ValueError(
                "quantile levels must be non-empty and strictly increasing inside (0, 1)"
            )
        self.register_buffer("quantiles", levels)

    def forward(
        self,
        outputs: Mapping[str, object],
        targets: Tensor,
        state_labels: Tensor,
        *,
        daylight_mask: Tensor | None = None,
        target_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if not isinstance(outputs, Mapping):
            raise TypeError("outputs must be a mapping")
        missing = {"quantiles", "state_logits"}.difference(outputs)
        if missing:
            raise KeyError(f"model outputs are missing: {sorted(missing)}")

        predictions = outputs["quantiles"]
        state_logits = outputs["state_logits"]
        for name, value in (
            ("quantiles", predictions),
            ("state_logits", state_logits),
            ("targets", targets),
            ("state_labels", state_labels),
        ):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a tensor")
        if state_labels.shape != targets.shape:
            raise ValueError("state_labels must have the same shape as targets")
        if tuple(state_logits.shape) != tuple(targets.shape) + (3,):
            raise ValueError("state_logits must have shape targets.shape + [3]")
        for name, mask in (("daylight_mask", daylight_mask), ("target_mask", target_mask)):
            if mask is not None:
                if not isinstance(mask, Tensor):
                    raise TypeError(f"{name} must be a tensor")
                if mask.shape != targets.shape:
                    raise ValueError(f"{name} must have the same shape as targets")

        valid_daylight = torch.isfinite(targets)
        if target_mask is not None:
            valid_daylight = valid_daylight & target_mask.bool()
        if daylight_mask is not None:
            valid_daylight = valid_daylight & daylight_mask.bool()

        forecast_loss = trapezoidal_qcrps(predictions, targets, self.quantiles, mask=valid_daylight)
        state_loss = state_cross_entropy(state_logits, state_labels, mask=valid_daylight)
        state_brier = state_brier_score(state_logits, state_labels, mask=valid_daylight)
        return {
            "loss": forecast_loss + state_loss,
            "forecast_loss": forecast_loss,
            "state_loss": state_loss,
            "daylight_qcrps": forecast_loss,
            "state_ce": state_loss,
            "state_brier": state_brier,
            "state_nll": state_loss,
        }


__all__ = ["ReliabilityAObjective"]
