from __future__ import annotations

import pytest
import torch

from racepv.losses import (
    RaCEPVLoss,
    ramp_direction_expert_tail_mse_loss,
    ramp_oracle_gate_regret_loss,
)


def test_oracle_gate_regret_updates_only_gate_and_uses_detached_expert_costs() -> None:
    experts = torch.tensor(
        [[[[0.0], [0.2], [0.6]], [[0.2], [0.4], [0.8]]]],
        requires_grad=True,
    )
    gates = torch.tensor([[[0.5, 0.25, 0.25], [0.2, 0.7, 0.1]]], requires_grad=True)
    targets = torch.tensor([[0.2, 0.4]])
    labels = torch.tensor([[1, 1]])

    loss = ramp_oracle_gate_regret_loss(
        experts,
        gates,
        targets,
        labels,
        median_index=0,
        daylight_mask=torch.ones_like(labels, dtype=torch.bool),
    )
    loss.backward()

    assert loss.item() == pytest.approx((0.5 * 0.04 + 0.25 * 0.16 + 0.2 * 0.04 + 0.1 * 0.16) / 2)
    assert experts.grad is None
    assert gates.grad is not None
    assert torch.count_nonzero(gates.grad) > 0


def test_oracle_gate_regret_ignores_stable_night_and_missing_targets() -> None:
    experts = torch.zeros(1, 3, 3, 1)
    gates = torch.full((1, 3, 3), 1 / 3)
    targets = torch.tensor([[0.5, 0.5, float("nan")]])
    labels = torch.tensor([[1, 0, 2]])
    daylight = torch.tensor([[True, True, True]])
    experts[0, 0, :, 0] = torch.tensor([0.0, 0.5, 1.0])

    observed = ramp_oracle_gate_regret_loss(
        experts,
        gates,
        targets,
        labels,
        median_index=0,
        daylight_mask=daylight,
    )

    assert observed.item() == pytest.approx((0.25 + 0.0 + 0.25) / 3)


def test_direction_expert_tail_mse_targets_matching_expert_and_weights_outliers() -> None:
    experts = torch.zeros(1, 3, 3, 1, requires_grad=True)
    targets = torch.tensor([[0.2, 0.8, 0.4]])
    labels = torch.tensor([[1, 2, 0]])
    with torch.no_grad():
        experts[0, 0, 1, 0] = 0.3  # squared error 0.01
        experts[0, 1, 2, 0] = 0.0  # squared error 0.64 dominates the tail objective
        experts[0, 2, 0, 0] = 1.0  # stable is excluded

    loss = ramp_direction_expert_tail_mse_loss(
        experts,
        targets,
        labels,
        median_index=0,
        daylight_mask=torch.ones_like(labels, dtype=torch.bool),
    )
    loss.backward()

    assert loss.item() == pytest.approx((0.01 + 0.64) / 2)
    assert experts.grad is not None
    assert experts.grad[0, 0, 1, 0] != 0
    assert experts.grad[0, 2, 0, 0] == 0


def _objective_outputs() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    expert_quantiles = torch.tensor(
        [
            [
                [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.4, 0.5, 0.6]],
                [[0.3, 0.4, 0.5], [0.4, 0.5, 0.6], [0.5, 0.6, 0.7]],
            ]
        ],
        requires_grad=True,
    )
    gate_logits = torch.zeros(1, 2, 3, requires_grad=True)
    outputs = {
        "quantiles": torch.tensor([[[0.2, 0.3, 0.4], [0.4, 0.5, 0.6]]]),
        "expert_quantiles": expert_quantiles,
        "gate_weights": gate_logits.softmax(dim=-1),
        "state_logits": torch.zeros(1, 2, 3, requires_grad=True),
    }
    return outputs, gate_logits


def test_objective_oracle_gate_scope_routes_gradient_to_gate_not_experts() -> None:
    outputs, gate_logits = _objective_outputs()
    objective = RaCEPVLoss(
        (0.1, 0.5, 0.9),
        specialist_weight=0.0,
        ramp_auxiliary_weight=1.0,
        ramp_auxiliary_scope="oracle_gate_regret",
    )

    metrics = objective(
        outputs,
        torch.tensor([[0.3, 0.5]]),
        torch.tensor([[1, 2]]),
        daylight_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    metrics["forecast_loss"].backward()

    assert gate_logits.grad is not None and torch.count_nonzero(gate_logits.grad) > 0
    expert_grad = outputs["expert_quantiles"].grad
    assert expert_grad is None or torch.count_nonzero(expert_grad) == 0


def test_objective_tail_scope_requires_no_gate_and_updates_matching_experts() -> None:
    outputs, _ = _objective_outputs()
    outputs.pop("gate_weights")
    objective = RaCEPVLoss(
        (0.1, 0.5, 0.9),
        specialist_weight=0.0,
        ramp_auxiliary_weight=1.0,
        ramp_auxiliary_scope="direction_expert_tail_mse",
    )

    metrics = objective(
        outputs,
        torch.tensor([[0.3, 0.5]]),
        torch.tensor([[1, 2]]),
        daylight_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    metrics["forecast_loss"].backward()

    assert torch.count_nonzero(outputs["expert_quantiles"].grad) > 0


def test_objective_oracle_gate_scope_fails_when_gate_weights_are_missing() -> None:
    outputs, _ = _objective_outputs()
    outputs.pop("gate_weights")
    objective = RaCEPVLoss(
        (0.1, 0.5, 0.9),
        ramp_auxiliary_scope="oracle_gate_regret",
    )

    with pytest.raises(KeyError, match="gate_weights"):
        objective(outputs, torch.tensor([[0.3, 0.5]]), torch.tensor([[1, 2]]))


def test_objective_isolated_oracle_scope_uses_only_auxiliary_gate_graph() -> None:
    outputs, main_gate_logits = _objective_outputs()
    auxiliary_gate_logits = torch.zeros(1, 2, 3, requires_grad=True)
    outputs["_auxiliary_gate_weights"] = auxiliary_gate_logits.softmax(dim=-1)
    objective = RaCEPVLoss(
        (0.1, 0.5, 0.9),
        specialist_weight=0.0,
        ramp_auxiliary_weight=1.0,
        ramp_auxiliary_scope="isolated_oracle_gate_regret",
    )

    metrics = objective(
        outputs,
        torch.tensor([[0.3, 0.5]]),
        torch.tensor([[1, 2]]),
        daylight_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    metrics["forecast_loss"].backward()

    assert main_gate_logits.grad is None
    assert auxiliary_gate_logits.grad is not None
    assert torch.count_nonzero(auxiliary_gate_logits.grad) > 0
    expert_grad = outputs["expert_quantiles"].grad
    assert expert_grad is None or torch.count_nonzero(expert_grad) == 0


def test_objective_isolated_oracle_scope_fails_without_auxiliary_gate() -> None:
    outputs, _ = _objective_outputs()
    objective = RaCEPVLoss(
        (0.1, 0.5, 0.9),
        ramp_auxiliary_weight=1.0,
        ramp_auxiliary_scope="isolated_oracle_gate_regret",
    )

    with pytest.raises(KeyError, match="_auxiliary_gate_weights"):
        objective(outputs, torch.tensor([[0.3, 0.5]]), torch.tensor([[1, 2]]))


def test_zero_weight_isolated_scope_does_not_require_auxiliary_gate() -> None:
    outputs, _ = _objective_outputs()
    objective = RaCEPVLoss(
        (0.1, 0.5, 0.9),
        ramp_auxiliary_weight=0.0,
        ramp_auxiliary_scope="isolated_oracle_gate_regret",
    )

    metrics = objective(
        outputs,
        torch.tensor([[0.3, 0.5]]),
        torch.tensor([[1, 2]]),
    )

    assert metrics["ramp_auxiliary"].item() == pytest.approx(0.0)
