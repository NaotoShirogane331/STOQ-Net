"""Losses for joint probabilistic power and ramp-state forecasting."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models.config import DEFAULT_QUANTILES, EXPERT_NAMES


def _masked_reduce(loss: Tensor, mask: Tensor | None, reduction: str) -> Tensor:
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: none, mean, sum")
    if mask is not None:
        expanded_mask = mask.to(dtype=loss.dtype, device=loss.device)
        while expanded_mask.ndim < loss.ndim:
            expanded_mask = expanded_mask.unsqueeze(-1)
        expanded_mask = expanded_mask.expand_as(loss)
        loss = torch.where(expanded_mask.bool(), loss, torch.zeros_like(loss))
    else:
        expanded_mask = None
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if expanded_mask is None:
        return loss.mean()
    denominator = expanded_mask.sum().clamp_min(1.0)
    return loss.sum() / denominator


def pinball_loss(
    predictions: Tensor,
    targets: Tensor,
    quantiles: Sequence[float] | Tensor = DEFAULT_QUANTILES,
    *,
    mask: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Quantile (pinball) loss for ``[..., Q]`` predictions."""

    if predictions.ndim != targets.ndim + 1:
        raise ValueError("predictions must have exactly one quantile dimension beyond targets")
    if tuple(predictions.shape[:-1]) != tuple(targets.shape):
        raise ValueError("prediction and target leading dimensions must match")
    levels = torch.as_tensor(quantiles, dtype=predictions.dtype, device=predictions.device)
    if levels.ndim != 1 or levels.numel() != predictions.shape[-1]:
        raise ValueError("number of quantile levels must match predictions.shape[-1]")
    if not bool(((levels > 0) & (levels < 1)).all()):
        raise ValueError("quantile levels must lie strictly between zero and one")

    finite = torch.isfinite(targets)
    effective_mask = finite if mask is None else (finite & mask.bool())
    safe_targets = torch.where(finite, targets, torch.zeros_like(targets))
    error = safe_targets.unsqueeze(-1) - predictions
    loss = torch.maximum(levels * error, (levels - 1.0) * error)
    return _masked_reduce(loss, effective_mask, reduction)


def trapezoidal_qcrps(
    predictions: Tensor,
    targets: Tensor,
    quantiles: Sequence[float] | Tensor = DEFAULT_QUANTILES,
    *,
    mask: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Differentiable qCRPS matching the reported trapezoidal metric."""

    if predictions.ndim != targets.ndim + 1:
        raise ValueError("predictions must have exactly one quantile dimension beyond targets")
    if tuple(predictions.shape[:-1]) != tuple(targets.shape):
        raise ValueError("prediction and target leading dimensions must match")
    levels = torch.as_tensor(quantiles, dtype=predictions.dtype, device=predictions.device)
    if levels.ndim != 1 or levels.numel() != predictions.shape[-1]:
        raise ValueError("number of quantile levels must match predictions.shape[-1]")
    if not bool(((levels > 0) & (levels < 1)).all()) or not bool((levels[1:] > levels[:-1]).all()):
        raise ValueError("quantile levels must be strictly increasing inside (0, 1)")

    finite = torch.isfinite(targets)
    effective_mask = finite if mask is None else (finite & mask.bool())
    safe_targets = torch.where(finite, targets, torch.zeros_like(targets))
    error = safe_targets.unsqueeze(-1) - predictions
    pinball = torch.maximum(levels * error, (levels - 1.0) * error)
    endpoint = torch.zeros_like(pinball[..., :1])
    integration_levels = torch.cat((levels.new_zeros(1), levels, levels.new_ones(1)), dim=0)
    per_cell = 2.0 * torch.trapezoid(
        torch.cat((endpoint, pinball, endpoint), dim=-1),
        integration_levels,
        dim=-1,
    )
    return _masked_reduce(per_cell, effective_mask, reduction)


def weighted_focal_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    class_weights: Sequence[float] | Tensor | None = None,
    gamma: float = 2.0,
    mask: Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> Tensor:
    """Class-weighted multiclass focal loss on unnormalized router logits."""

    if gamma < 0:
        raise ValueError("gamma cannot be negative")
    if logits.ndim != targets.ndim + 1 or tuple(logits.shape[:-1]) != tuple(targets.shape):
        raise ValueError("logits must have shape targets.shape + [num_classes]")
    num_classes = logits.shape[-1]
    valid = (targets != ignore_index) & (targets >= 0) & (targets < num_classes)
    if mask is not None:
        valid = valid & mask.bool()
    safe_targets = targets.clamp(0, num_classes - 1).long()

    log_probabilities = torch.log_softmax(logits, dim=-1)
    selected_log_probability = log_probabilities.gather(
        dim=-1,
        index=safe_targets.unsqueeze(-1),
    ).squeeze(-1)
    probability = selected_log_probability.exp()
    loss = -((1.0 - probability).clamp_min(0.0) ** gamma) * selected_log_probability

    if class_weights is not None:
        weights = torch.as_tensor(class_weights, dtype=logits.dtype, device=logits.device)
        if weights.ndim != 1 or weights.numel() != num_classes:
            raise ValueError("class_weights must contain one value per class")
        if bool((weights < 0).any()):
            raise ValueError("class_weights cannot be negative")
        loss = loss * weights.gather(0, safe_targets.reshape(-1)).reshape_as(safe_targets)
    return _masked_reduce(loss, valid, reduction)


def state_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    *,
    mask: Tensor | None = None,
    ignore_index: int = -100,
) -> Tensor:
    """Unweighted daylight cross-entropy for the physical state head."""

    if logits.ndim != targets.ndim + 1 or tuple(logits.shape[:-1]) != tuple(targets.shape):
        raise ValueError("logits must have shape targets.shape + [num_classes]")
    num_classes = logits.shape[-1]
    valid = (targets != ignore_index) & (targets >= 0) & (targets < num_classes)
    if mask is not None:
        valid = valid & mask.bool()
    safe_targets = targets.clamp(0, num_classes - 1).long()
    loss = F.cross_entropy(
        logits.reshape(-1, num_classes),
        safe_targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    return _masked_reduce(loss, valid, "mean")


def transition_weighted_state_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    *,
    boundary_weight: float = 1.0,
    mask: Tensor | None = None,
    ignore_index: int = -100,
) -> Tensor:
    """Flat state CE with extra weight on valid within-horizon transitions.

    The first forecast lead has no archived preceding target and is therefore
    never classified as a boundary. Ignored or masked cells also break the
    transition chain so that night and missing labels cannot create artificial
    onset/offset events.
    """

    weight = torch.as_tensor(boundary_weight, dtype=logits.dtype, device=logits.device)
    if weight.ndim != 0 or not bool(torch.isfinite(weight)) or float(weight) < 1.0:
        raise ValueError("boundary_weight must be a finite scalar at least one")
    if float(weight) == 1.0:
        return state_cross_entropy(logits, targets, mask=mask, ignore_index=ignore_index)
    if logits.ndim != targets.ndim + 1 or tuple(logits.shape[:-1]) != tuple(targets.shape):
        raise ValueError("logits must have shape targets.shape + [num_classes]")

    num_classes = logits.shape[-1]
    valid = (targets != ignore_index) & (targets >= 0) & (targets < num_classes)
    if mask is not None:
        valid = valid & mask.bool()
    safe_targets = targets.clamp(0, num_classes - 1).long()
    per_cell = F.cross_entropy(
        logits.reshape(-1, num_classes),
        safe_targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)

    boundary = torch.zeros_like(valid)
    if targets.shape[-1] > 1:
        boundary[..., 1:] = (
            valid[..., 1:]
            & valid[..., :-1]
            & (safe_targets[..., 1:] != safe_targets[..., :-1])
        )
    cell_weights = torch.ones_like(per_cell)
    cell_weights = torch.where(boundary, weight, cell_weights)
    selected_weights = torch.where(valid, cell_weights, torch.zeros_like(cell_weights))
    return (per_cell * selected_weights).sum() / selected_weights.sum().clamp_min(1.0)


def state_brier_score(
    logits: Tensor,
    targets: Tensor,
    *,
    mask: Tensor | None = None,
    ignore_index: int = -100,
) -> Tensor:
    """Multiclass Brier score on valid daylight state labels."""

    if logits.ndim != targets.ndim + 1 or tuple(logits.shape[:-1]) != tuple(targets.shape):
        raise ValueError("logits must have shape targets.shape + [num_classes]")
    num_classes = logits.shape[-1]
    valid = (targets != ignore_index) & (targets >= 0) & (targets < num_classes)
    if mask is not None:
        valid = valid & mask.bool()
    safe_targets = targets.clamp(0, num_classes - 1).long()
    expected = F.one_hot(safe_targets, num_classes=num_classes).to(logits.dtype)
    per_cell = (torch.softmax(logits, dim=-1) - expected).square().sum(dim=-1)
    return _masked_reduce(per_cell, valid, "mean")


def hierarchical_state_loss(
    ramp_presence_logits: Tensor,
    ramp_direction_logits: Tensor,
    targets: Tensor,
    *,
    positive_weight: float | Tensor = 1.0,
    mask: Tensor | None = None,
    ignore_index: int = -100,
) -> Tensor:
    """Binary ramp-presence loss plus conditional up/down direction loss."""

    if tuple(ramp_presence_logits.shape) != tuple(targets.shape):
        raise ValueError("ramp_presence_logits must have the same shape as targets")
    if tuple(ramp_direction_logits.shape) != tuple(targets.shape) + (2,):
        raise ValueError("ramp_direction_logits must have shape targets.shape + [2]")
    weight = torch.as_tensor(
        positive_weight,
        dtype=ramp_presence_logits.dtype,
        device=ramp_presence_logits.device,
    )
    if weight.ndim != 0 or not bool(torch.isfinite(weight)) or float(weight) < 1.0:
        raise ValueError("positive_weight must be a finite scalar at least one")

    valid = (targets != ignore_index) & (targets >= 0) & (targets <= 2)
    if mask is not None:
        valid = valid & mask.bool()
    safe_targets = targets.clamp(0, 2).long()
    ramp_targets = (safe_targets != 0).to(ramp_presence_logits.dtype)
    presence = F.binary_cross_entropy_with_logits(
        ramp_presence_logits,
        ramp_targets,
        pos_weight=weight,
        reduction="none",
    )
    presence_loss = _masked_reduce(presence, valid, "mean")

    ramp_mask = valid & (safe_targets != 0)
    direction_targets = (safe_targets - 1).clamp(0, 1)
    direction = F.cross_entropy(
        ramp_direction_logits.reshape(-1, 2),
        direction_targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    direction_loss = _masked_reduce(direction, ramp_mask, "mean")
    return presence_loss + direction_loss


def ramp_median_smooth_l1_loss(
    predictions: Tensor,
    targets: Tensor,
    state_labels: Tensor,
    *,
    median_index: int,
    daylight_mask: Tensor | None = None,
    beta: float = 0.1,
) -> Tensor:
    """Smooth-L1 median error restricted to valid daylight ramp cells."""

    if predictions.ndim != targets.ndim + 1 or predictions.shape[:-1] != targets.shape:
        raise ValueError("predictions must have shape targets.shape + [Q]")
    if state_labels.shape != targets.shape:
        raise ValueError("state_labels must match targets")
    if not 0 <= median_index < predictions.shape[-1]:
        raise ValueError("median_index is outside the quantile dimension")
    if beta <= 0:
        raise ValueError("beta must be positive")
    finite = torch.isfinite(targets)
    selected = finite & ((state_labels == 1) | (state_labels == 2))
    if daylight_mask is not None:
        selected = selected & daylight_mask.bool()
    safe_targets = torch.where(finite, targets, torch.zeros_like(targets))
    per_cell = F.smooth_l1_loss(
        predictions[..., median_index],
        safe_targets,
        beta=beta,
        reduction="none",
    )
    return _masked_reduce(per_cell, selected, "mean")


def ramp_expert_median_smooth_l1_loss(
    expert_quantiles: Tensor,
    targets: Tensor,
    state_labels: Tensor,
    *,
    median_index: int,
    daylight_mask: Tensor | None = None,
    beta: float = 0.1,
) -> Tensor:
    """Smooth-L1 loss on the true direction expert for daylight ramp cells."""

    if expert_quantiles.ndim != 4 or expert_quantiles.shape[2] != 3:
        raise ValueError("expert_quantiles must have shape [B, H, 3, Q]")
    if tuple(expert_quantiles.shape[:2]) != tuple(targets.shape):
        raise ValueError("expert and target batch/horizon dimensions must match")
    if tuple(state_labels.shape) != tuple(targets.shape):
        raise ValueError("state_labels must match targets")
    if not 0 <= median_index < expert_quantiles.shape[-1]:
        raise ValueError("median_index is outside the quantile dimension")
    if beta <= 0:
        raise ValueError("beta must be positive")
    finite = torch.isfinite(targets)
    selected = finite & ((state_labels == 1) | (state_labels == 2))
    if daylight_mask is not None:
        selected = selected & daylight_mask.bool()
    safe_targets = torch.where(finite, targets, torch.zeros_like(targets))
    expert_index = state_labels.clamp(0, 2).long().unsqueeze(-1)
    selected_median = (
        expert_quantiles[..., median_index]
        .gather(
            dim=2,
            index=expert_index,
        )
        .squeeze(-1)
    )
    per_cell = F.smooth_l1_loss(
        selected_median,
        safe_targets,
        beta=beta,
        reduction="none",
    )
    return _masked_reduce(per_cell, selected, "mean")


def ramp_oracle_gate_regret_loss(
    expert_quantiles: Tensor,
    gate_weights: Tensor,
    targets: Tensor,
    state_labels: Tensor,
    *,
    median_index: int,
    daylight_mask: Tensor | None = None,
) -> Tensor:
    """Train only the fusion gate against detached per-expert median regret."""

    if expert_quantiles.ndim != 4 or expert_quantiles.shape[2] != 3:
        raise ValueError("expert_quantiles must have shape [B, H, 3, Q]")
    if tuple(expert_quantiles.shape[:2]) != tuple(targets.shape):
        raise ValueError("expert and target batch/horizon dimensions must match")
    if tuple(gate_weights.shape) != tuple(targets.shape) + (3,):
        raise ValueError("gate_weights must have shape targets.shape + [3]")
    if tuple(state_labels.shape) != tuple(targets.shape):
        raise ValueError("state_labels must match targets")
    if not 0 <= median_index < expert_quantiles.shape[-1]:
        raise ValueError("median_index is outside the quantile dimension")
    finite = torch.isfinite(targets)
    selected = finite & ((state_labels == 1) | (state_labels == 2))
    if daylight_mask is not None:
        selected = selected & daylight_mask.bool()
    safe_targets = torch.where(finite, targets, torch.zeros_like(targets))
    expert_medians = expert_quantiles[..., median_index]
    squared_error = (expert_medians - safe_targets.unsqueeze(-1)).square().detach()
    regret = squared_error - squared_error.amin(dim=-1, keepdim=True)
    per_cell = (gate_weights * regret).sum(dim=-1)
    return _masked_reduce(per_cell, selected, "mean")


def ramp_direction_expert_tail_mse_loss(
    expert_quantiles: Tensor,
    targets: Tensor,
    state_labels: Tensor,
    *,
    median_index: int,
    daylight_mask: Tensor | None = None,
) -> Tensor:
    """Tail-sensitive squared median error on the true ramp-direction expert."""

    if expert_quantiles.ndim != 4 or expert_quantiles.shape[2] != 3:
        raise ValueError("expert_quantiles must have shape [B, H, 3, Q]")
    if tuple(expert_quantiles.shape[:2]) != tuple(targets.shape):
        raise ValueError("expert and target batch/horizon dimensions must match")
    if tuple(state_labels.shape) != tuple(targets.shape):
        raise ValueError("state_labels must match targets")
    if not 0 <= median_index < expert_quantiles.shape[-1]:
        raise ValueError("median_index is outside the quantile dimension")
    finite = torch.isfinite(targets)
    selected = finite & ((state_labels == 1) | (state_labels == 2))
    if daylight_mask is not None:
        selected = selected & daylight_mask.bool()
    safe_targets = torch.where(finite, targets, torch.zeros_like(targets))
    expert_index = state_labels.clamp(0, 2).long().unsqueeze(-1)
    matching_median = (
        expert_quantiles[..., median_index].gather(dim=2, index=expert_index).squeeze(-1)
    )
    per_cell = (matching_median - safe_targets).square()
    return _masked_reduce(per_cell, selected, "mean")


def ramp_up_tail_mse_loss(
    base_quantiles: Tensor,
    tail_correction: Tensor,
    targets: Tensor,
    state_labels: Tensor,
    *,
    median_index: int,
    daylight_mask: Tensor | None = None,
) -> Tensor:
    """Squared corrected-median error on finite daylight ramp-up cells only."""

    if base_quantiles.ndim != targets.ndim + 1:
        raise ValueError("base_quantiles must have shape targets.shape + [Q]")
    if tuple(base_quantiles.shape[:-1]) != tuple(targets.shape):
        raise ValueError("base_quantiles and targets must share leading dimensions")
    if tuple(tail_correction.shape) != tuple(targets.shape):
        raise ValueError("tail_correction must have the same shape as targets")
    if tuple(state_labels.shape) != tuple(targets.shape):
        raise ValueError("state_labels must have the same shape as targets")
    if not 0 <= median_index < base_quantiles.shape[-1]:
        raise ValueError("median_index is outside the quantile dimension")

    finite = torch.isfinite(targets)
    selected = finite & (state_labels == 1)
    if daylight_mask is not None:
        selected = selected & daylight_mask.bool()
    safe_targets = torch.where(finite, targets, torch.zeros_like(targets))
    corrected = base_quantiles[..., median_index].detach() + tail_correction
    return _masked_reduce((corrected - safe_targets).square(), selected, "mean")


def tail_trigger_sparsity_loss(
    tail_trigger: Tensor,
    targets: Tensor,
    *,
    daylight_mask: Tensor | None = None,
) -> Tensor:
    """Mean trigger activation on finite daylight cells."""

    if tuple(tail_trigger.shape) != tuple(targets.shape):
        raise ValueError("tail_trigger must have the same shape as targets")
    selected = torch.isfinite(targets)
    if daylight_mask is not None:
        selected = selected & daylight_mask.bool()
    return _masked_reduce(tail_trigger, selected, "mean")


def specialist_masked_loss(
    expert_quantiles: Tensor,
    targets: Tensor,
    state_labels: Tensor,
    quantiles: Sequence[float] | Tensor = DEFAULT_QUANTILES,
    *,
    daylight_mask: Tensor | None = None,
    active_experts: Sequence[bool] | Tensor | None = None,
) -> Tensor:
    """Fit each expert only on its assigned state and balance active states.

    Expert order is stable, ramp-up, ramp-down, matching label values 0, 1, 2.
    Taking a mean of the per-state losses prevents the majority stable state
    from erasing the two transient specialists.
    """

    if expert_quantiles.ndim != 4 or expert_quantiles.shape[2] != 3:
        raise ValueError("expert_quantiles must have shape [B, H, 3, Q]")
    if tuple(expert_quantiles.shape[:2]) != tuple(targets.shape):
        raise ValueError("expert and target batch/horizon dimensions must match")
    if tuple(state_labels.shape) != tuple(targets.shape):
        raise ValueError("state_labels must match targets")
    if active_experts is None:
        active = torch.ones(3, dtype=torch.bool, device=expert_quantiles.device)
    else:
        active = torch.as_tensor(
            active_experts,
            dtype=torch.bool,
            device=expert_quantiles.device,
        )
        if active.ndim != 1 or active.numel() != 3 or not bool(active.any()):
            raise ValueError("active_experts must enable a non-empty subset of three experts")

    finite = torch.isfinite(targets)
    valid_daylight = finite
    if daylight_mask is not None:
        valid_daylight = valid_daylight & daylight_mask.bool()

    per_state: list[Tensor] = []
    for state in range(3):
        if not bool(active[state]):
            continue
        state_mask = valid_daylight & (state_labels == state)
        if bool(state_mask.any()):
            per_state.append(
                pinball_loss(
                    expert_quantiles[:, :, state, :],
                    targets,
                    quantiles,
                    mask=state_mask,
                )
            )
    if not per_state:
        return expert_quantiles.sum() * 0.0
    return torch.stack(per_state).mean()


def specialist_qcrps_loss(
    expert_quantiles: Tensor,
    targets: Tensor,
    state_labels: Tensor,
    quantiles: Sequence[float] | Tensor = DEFAULT_QUANTILES,
    *,
    daylight_mask: Tensor | None = None,
    active_experts: Sequence[bool] | Tensor | None = None,
) -> Tensor:
    """Balance per-state daylight qCRPS across the active experts."""

    if expert_quantiles.ndim != 4 or expert_quantiles.shape[2] != 3:
        raise ValueError("expert_quantiles must have shape [B, H, 3, Q]")
    if tuple(expert_quantiles.shape[:2]) != tuple(targets.shape):
        raise ValueError("expert and target batch/horizon dimensions must match")
    if tuple(state_labels.shape) != tuple(targets.shape):
        raise ValueError("state_labels must match targets")
    if active_experts is None:
        active = torch.ones(3, dtype=torch.bool, device=expert_quantiles.device)
    else:
        active = torch.as_tensor(
            active_experts,
            dtype=torch.bool,
            device=expert_quantiles.device,
        )
        if active.ndim != 1 or active.numel() != 3 or not bool(active.any()):
            raise ValueError("active_experts must enable a non-empty subset of three experts")

    valid_daylight = torch.isfinite(targets)
    if daylight_mask is not None:
        valid_daylight = valid_daylight & daylight_mask.bool()
    per_state: list[Tensor] = []
    for state in range(3):
        if not bool(active[state]):
            continue
        state_mask = valid_daylight & (state_labels == state)
        if bool(state_mask.any()):
            per_state.append(
                trapezoidal_qcrps(
                    expert_quantiles[:, :, state, :],
                    targets,
                    quantiles,
                    mask=state_mask,
                )
            )
    if not per_state:
        return expert_quantiles.sum() * 0.0
    return torch.stack(per_state).mean()


class RaCEPVLoss(nn.Module):
    """Separated forecast and physical-state objectives for RaCE-PV."""

    def __init__(
        self,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        *,
        specialist_weight: float = 0.3,
        active_experts: Sequence[str] = EXPERT_NAMES,
        state_head_mode: str = "flat",
        ramp_positive_weight: float = 1.0,
        ramp_auxiliary_weight: float = 0.0,
        ramp_auxiliary_beta: float = 0.1,
        ramp_auxiliary_scope: str = "fusion_median",
        state_boundary_weight: float = 1.0,
        tail_auxiliary_weight: float = 0.0,
        tail_sparsity_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if specialist_weight < 0:
            raise ValueError("specialist_weight cannot be negative")
        if state_head_mode not in {"flat", "hierarchical"}:
            raise ValueError("state_head_mode must be flat or hierarchical")
        if not torch.isfinite(torch.tensor(ramp_positive_weight)) or ramp_positive_weight < 1:
            raise ValueError("ramp_positive_weight must be finite and at least one")
        if ramp_auxiliary_weight < 0:
            raise ValueError("ramp_auxiliary_weight cannot be negative")
        if ramp_auxiliary_beta <= 0:
            raise ValueError("ramp_auxiliary_beta must be positive")
        if not torch.isfinite(torch.tensor(state_boundary_weight)) or state_boundary_weight < 1:
            raise ValueError("state_boundary_weight must be finite and at least one")
        if not torch.isfinite(torch.tensor(tail_auxiliary_weight)) or tail_auxiliary_weight < 0:
            raise ValueError("tail_auxiliary_weight must be finite and non-negative")
        if not torch.isfinite(torch.tensor(tail_sparsity_weight)) or tail_sparsity_weight < 0:
            raise ValueError("tail_sparsity_weight must be finite and non-negative")
        ramp_auxiliary_scopes = {
            "fusion_median",
            "direction_experts",
            "oracle_gate_regret",
            "isolated_oracle_gate_regret",
            "direction_expert_tail_mse",
        }
        if ramp_auxiliary_scope not in ramp_auxiliary_scopes:
            raise ValueError(
                "ramp_auxiliary_scope must be one of: " + ", ".join(sorted(ramp_auxiliary_scopes))
            )
        self.register_buffer("quantiles", torch.as_tensor(tuple(quantiles), dtype=torch.float32))
        if 0.5 not in tuple(float(value) for value in quantiles):
            raise ValueError("quantiles must include the median for the ramp auxiliary")
        active_names = tuple(active_experts)
        if not active_names or len(set(active_names)) != len(active_names):
            raise ValueError("active_experts must be a non-empty unique expert subset")
        unknown = set(active_names).difference(EXPERT_NAMES)
        if unknown:
            raise ValueError(f"unknown active experts: {sorted(unknown)}")
        self.register_buffer(
            "active_expert_mask",
            torch.tensor([name in active_names for name in EXPERT_NAMES], dtype=torch.bool),
        )
        self.specialist_weight = specialist_weight
        self.state_head_mode = state_head_mode
        self.ramp_positive_weight = float(ramp_positive_weight)
        self.ramp_auxiliary_weight = float(ramp_auxiliary_weight)
        self.ramp_auxiliary_beta = float(ramp_auxiliary_beta)
        self.ramp_auxiliary_scope = ramp_auxiliary_scope
        self.state_boundary_weight = float(state_boundary_weight)
        self.tail_auxiliary_weight = float(tail_auxiliary_weight)
        self.tail_sparsity_weight = float(tail_sparsity_weight)
        self.median_index = tuple(float(value) for value in quantiles).index(0.5)

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: Tensor,
        state_labels: Tensor,
        *,
        daylight_mask: Tensor | None = None,
        target_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        required = {"quantiles", "state_logits", "expert_quantiles"}
        if self.ramp_auxiliary_scope == "oracle_gate_regret":
            required.add("gate_weights")
        elif (
            self.ramp_auxiliary_scope == "isolated_oracle_gate_regret"
            and self.ramp_auxiliary_weight > 0.0
        ):
            required.add("_auxiliary_gate_weights")
        if self.tail_auxiliary_weight > 0.0 or self.tail_sparsity_weight > 0.0:
            required.update({"base_quantiles", "tail_correction", "tail_trigger"})
        missing = required.difference(outputs)
        if missing:
            raise KeyError(f"model outputs are missing: {sorted(missing)}")

        finite = torch.isfinite(targets)
        valid_targets = finite if target_mask is None else (finite & target_mask.bool())
        valid_daylight = valid_targets
        if daylight_mask is not None:
            valid_daylight = valid_daylight & daylight_mask.bool()

        final_qcrps = trapezoidal_qcrps(
            outputs["quantiles"],
            targets,
            self.quantiles,
            mask=valid_daylight,
        )
        state_ce = state_cross_entropy(
            outputs["state_logits"],
            state_labels,
            mask=valid_daylight,
        )
        if self.state_head_mode == "hierarchical":
            hierarchical_required = {
                "ramp_presence_logits",
                "ramp_direction_logits",
            }
            hierarchical_missing = hierarchical_required.difference(outputs)
            if hierarchical_missing:
                raise KeyError(
                    f"hierarchical state outputs are missing: {sorted(hierarchical_missing)}"
                )
            state_loss = hierarchical_state_loss(
                outputs["ramp_presence_logits"],
                outputs["ramp_direction_logits"],
                state_labels,
                positive_weight=self.ramp_positive_weight,
                mask=valid_daylight,
            )
        else:
            state_loss = transition_weighted_state_cross_entropy(
                outputs["state_logits"],
                state_labels,
                boundary_weight=self.state_boundary_weight,
                mask=valid_daylight,
            )
        state_brier = state_brier_score(
            outputs["state_logits"],
            state_labels,
            mask=valid_daylight,
        )
        specialist = specialist_qcrps_loss(
            outputs["expert_quantiles"],
            targets,
            state_labels,
            self.quantiles,
            daylight_mask=valid_daylight,
            active_experts=self.active_expert_mask,
        )
        if self.ramp_auxiliary_weight == 0.0:
            ramp_auxiliary = final_qcrps.new_zeros(())
        elif self.ramp_auxiliary_scope == "direction_experts":
            ramp_auxiliary = ramp_expert_median_smooth_l1_loss(
                outputs["expert_quantiles"],
                targets,
                state_labels,
                median_index=self.median_index,
                daylight_mask=valid_daylight,
                beta=self.ramp_auxiliary_beta,
            )
        elif self.ramp_auxiliary_scope == "oracle_gate_regret":
            ramp_auxiliary = ramp_oracle_gate_regret_loss(
                outputs["expert_quantiles"],
                outputs["gate_weights"],
                targets,
                state_labels,
                median_index=self.median_index,
                daylight_mask=valid_daylight,
            )
        elif self.ramp_auxiliary_scope == "isolated_oracle_gate_regret":
            ramp_auxiliary = ramp_oracle_gate_regret_loss(
                outputs["expert_quantiles"],
                outputs["_auxiliary_gate_weights"],
                targets,
                state_labels,
                median_index=self.median_index,
                daylight_mask=valid_daylight,
            )
        elif self.ramp_auxiliary_scope == "direction_expert_tail_mse":
            ramp_auxiliary = ramp_direction_expert_tail_mse_loss(
                outputs["expert_quantiles"],
                targets,
                state_labels,
                median_index=self.median_index,
                daylight_mask=valid_daylight,
            )
        else:
            ramp_auxiliary = ramp_median_smooth_l1_loss(
                outputs["quantiles"],
                targets,
                state_labels,
                median_index=self.median_index,
                daylight_mask=valid_daylight,
                beta=self.ramp_auxiliary_beta,
            )
        if self.tail_auxiliary_weight > 0.0 or self.tail_sparsity_weight > 0.0:
            tail_ramp_up_mse = ramp_up_tail_mse_loss(
                outputs["base_quantiles"],
                outputs["tail_correction"],
                targets,
                state_labels,
                median_index=self.median_index,
                daylight_mask=valid_daylight,
            )
            tail_trigger_sparsity = tail_trigger_sparsity_loss(
                outputs["tail_trigger"],
                targets,
                daylight_mask=valid_daylight,
            )
        else:
            tail_ramp_up_mse = final_qcrps.new_zeros(())
            tail_trigger_sparsity = final_qcrps.new_zeros(())
        forecast_loss = (
            final_qcrps
            + self.specialist_weight * specialist
            + self.ramp_auxiliary_weight * ramp_auxiliary
            + self.tail_auxiliary_weight * tail_ramp_up_mse
            + self.tail_sparsity_weight * tail_trigger_sparsity
        )
        total = forecast_loss + state_loss
        return {
            "loss": total,
            "forecast_loss": forecast_loss,
            "state_loss": state_loss,
            "daylight_qcrps": final_qcrps,
            "specialist_qcrps": specialist,
            "ramp_auxiliary": ramp_auxiliary,
            "tail_ramp_up_mse": tail_ramp_up_mse,
            "tail_trigger_sparsity": tail_trigger_sparsity,
            "state_ce": state_ce,
            "state_brier": state_brier,
            "state_nll": state_ce,
        }


def racepv_loss(
    outputs: dict[str, Tensor],
    targets: Tensor,
    state_labels: Tensor,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    specialist_weight: float = 0.3,
    active_experts: Sequence[str] = EXPERT_NAMES,
    state_head_mode: str = "flat",
    ramp_positive_weight: float = 1.0,
    ramp_auxiliary_weight: float = 0.0,
    ramp_auxiliary_beta: float = 0.1,
    ramp_auxiliary_scope: str = "fusion_median",
    state_boundary_weight: float = 1.0,
    tail_auxiliary_weight: float = 0.0,
    tail_sparsity_weight: float = 0.0,
    daylight_mask: Tensor | None = None,
    target_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Functional convenience wrapper around :class:`RaCEPVLoss`."""

    objective = RaCEPVLoss(
        quantiles,
        specialist_weight=specialist_weight,
        active_experts=active_experts,
        state_head_mode=state_head_mode,
        ramp_positive_weight=ramp_positive_weight,
        ramp_auxiliary_weight=ramp_auxiliary_weight,
        ramp_auxiliary_beta=ramp_auxiliary_beta,
        ramp_auxiliary_scope=ramp_auxiliary_scope,
        state_boundary_weight=state_boundary_weight,
        tail_auxiliary_weight=tail_auxiliary_weight,
        tail_sparsity_weight=tail_sparsity_weight,
    ).to(device=targets.device)
    return objective(
        outputs,
        targets,
        state_labels,
        daylight_mask=daylight_mask,
        target_mask=target_mask,
    )


__all__ = [
    "RaCEPVLoss",
    "hierarchical_state_loss",
    "pinball_loss",
    "ramp_direction_expert_tail_mse_loss",
    "ramp_expert_median_smooth_l1_loss",
    "ramp_median_smooth_l1_loss",
    "ramp_oracle_gate_regret_loss",
    "ramp_up_tail_mse_loss",
    "racepv_loss",
    "specialist_masked_loss",
    "specialist_qcrps_loss",
    "state_brier_score",
    "state_cross_entropy",
    "tail_trigger_sparsity_loss",
    "transition_weighted_state_cross_entropy",
    "trapezoidal_qcrps",
    "weighted_focal_loss",
]
