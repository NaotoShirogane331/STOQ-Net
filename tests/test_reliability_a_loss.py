"""Synthetic forward/backward contract for the candidate's independent objective."""

from __future__ import annotations

import math

import pytest
import torch

from racepv import losses as legacy_losses
from racepv.reliability_a_loss import ReliabilityAObjective

QUANTILES = (0.1, 0.5, 0.9)


def _batch() -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
    generator = torch.Generator().manual_seed(915)
    predictions = torch.rand(2, 4, 3, generator=generator, dtype=torch.float64)
    state_logits = torch.randn(2, 4, 3, generator=generator, dtype=torch.float64)
    outputs = {
        "quantiles": predictions.requires_grad_(),
        "state_logits": state_logits.requires_grad_(),
    }
    targets = torch.tensor(
        [[0.25, 0.75, float("nan"), 0.6], [0.4, 0.15, 0.9, 0.3]],
        dtype=torch.float64,
    )
    labels = torch.tensor([[0, 1, 2, -100], [2, 0, 1, 2]])
    masks = {
        "daylight_mask": torch.tensor([[1, 1, 1, 1], [1, 0, 1, 1]], dtype=torch.bool),
        "target_mask": torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]], dtype=torch.bool),
    }
    return outputs, targets, labels, masks


@pytest.mark.parametrize("use_masks", [False, True])
def test_matches_legacy_forecast_state_values_and_gradients(use_masks: bool) -> None:
    outputs, targets, labels, masks = _batch()
    if not use_masks:
        masks = {}
    legacy_outputs = {
        name: value.detach().clone().requires_grad_() for name, value in outputs.items()
    }
    legacy_outputs["expert_quantiles"] = torch.zeros(2, 4, 3, 3, dtype=torch.float64)
    expected = legacy_losses.RaCEPVLoss(
        QUANTILES,
        specialist_weight=0.0,
        state_head_mode="flat",
        state_boundary_weight=1.0,
        ramp_auxiliary_weight=0.0,
        tail_auxiliary_weight=0.0,
        tail_sparsity_weight=0.0,
    )(legacy_outputs, targets, labels, **masks)
    actual = ReliabilityAObjective(QUANTILES)(outputs, targets, labels, **masks)

    for name in (
        "loss",
        "forecast_loss",
        "state_loss",
        "daylight_qcrps",
        "state_ce",
        "state_brier",
        "state_nll",
    ):
        torch.testing.assert_close(actual[name], expected[name], rtol=0.0, atol=1e-12)
    for name in ("loss", "forecast_loss", "state_loss"):
        actual_gradients = torch.autograd.grad(
            actual[name], tuple(outputs.values()), retain_graph=True, allow_unused=True
        )
        expected_gradients = torch.autograd.grad(
            expected[name],
            (legacy_outputs["quantiles"], legacy_outputs["state_logits"]),
            retain_graph=True,
            allow_unused=True,
        )
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients, strict=True
        ):
            if expected_gradient is None:
                assert actual_gradient is None
            else:
                torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0.0, atol=1e-12)


class _ExpertReadForbidden(dict):
    def __getitem__(self, key):
        if key == "expert_quantiles":
            raise AssertionError("the objective read the expert prediction input")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "expert_quantiles":
            raise AssertionError("the objective read the expert prediction input")
        return super().get(key, default)


def test_ignores_poisoned_expert_input_and_never_calls_legacy_specialist(monkeypatch) -> None:
    def forbidden_specialist(*args, **kwargs):
        raise AssertionError("the specialist computation still runs")

    monkeypatch.setattr(legacy_losses, "specialist_qcrps_loss", forbidden_specialist)
    outputs, targets, labels, masks = _batch()
    guarded_outputs = _ExpertReadForbidden(outputs, expert_quantiles=object())
    actual = ReliabilityAObjective(QUANTILES)(guarded_outputs, targets, labels, **masks)
    assert torch.isfinite(actual["loss"])
    actual["loss"].backward()
    assert all(torch.isfinite(value.grad).all() for value in outputs.values())


def test_has_independent_identity_and_no_specialist_configuration_or_logs() -> None:
    objective = ReliabilityAObjective(QUANTILES)
    assert objective.objective_id == "racepv.forecast-state-objective-without-specialist.v1"
    assert not hasattr(objective, "specialist_weight")
    with pytest.raises(TypeError, match="specialist_weight"):
        ReliabilityAObjective(QUANTILES, specialist_weight=0.0)
    outputs, targets, labels, masks = _batch()
    result = objective(outputs, targets, labels, **masks)
    assert set(result) == {
        "loss",
        "forecast_loss",
        "state_loss",
        "daylight_qcrps",
        "state_ce",
        "state_brier",
        "state_nll",
    }
    torch.testing.assert_close(result["loss"], result["forecast_loss"] + result["state_loss"])
    torch.testing.assert_close(result["forecast_loss"], result["daylight_qcrps"])
    torch.testing.assert_close(result["state_loss"], result["state_ce"])


def test_daylight_target_and_finite_masks_select_only_one_cell() -> None:
    outputs = {
        "quantiles": torch.zeros(1, 5, 3, dtype=torch.float64, requires_grad=True),
        "state_logits": torch.zeros(1, 5, 3, dtype=torch.float64, requires_grad=True),
    }
    targets = torch.tensor([[1.0, 2.0, float("nan"), float("inf"), 4.0]])
    labels = torch.tensor([[0, 1, 2, 0, 2]])
    result = ReliabilityAObjective(QUANTILES)(
        outputs,
        targets,
        labels,
        daylight_mask=torch.tensor([[1, 0, 1, 1, 1]], dtype=torch.bool),
        target_mask=torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool),
    )
    assert result["forecast_loss"].item() == pytest.approx(0.9)
    assert result["state_loss"].item() == pytest.approx(math.log(3.0))
    assert result["loss"].item() == pytest.approx(0.9 + math.log(3.0))
    assert result["state_brier"].item() == pytest.approx(2.0 / 3.0)
    result["loss"].backward()
    for value in outputs.values():
        assert torch.count_nonzero(value.grad[:, :1]) > 0
        assert torch.count_nonzero(value.grad[:, 1:]) == 0
        assert torch.isfinite(value.grad).all()


@pytest.mark.parametrize("mask_name", ["daylight_mask", "target_mask"])
def test_empty_valid_set_has_finite_zero_loss_and_gradients(mask_name: str) -> None:
    outputs, targets, labels, _ = _batch()
    result = ReliabilityAObjective(QUANTILES)(
        outputs, targets, labels, **{mask_name: torch.zeros_like(labels, dtype=torch.bool)}
    )
    for value in result.values():
        assert torch.isfinite(value)
        assert value.item() == 0.0
    result["loss"].backward()
    for value in outputs.values():
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) == 0


def test_ignored_state_label_does_not_remove_forecast_supervision() -> None:
    outputs = {
        "quantiles": torch.zeros(1, 1, 3, requires_grad=True),
        "state_logits": torch.zeros(1, 1, 3, requires_grad=True),
    }
    result = ReliabilityAObjective(QUANTILES)(outputs, torch.ones(1, 1), torch.full((1, 1), -100))
    assert result["forecast_loss"].item() == pytest.approx(0.9)
    assert result["state_loss"].item() == 0.0
    result["loss"].backward()
    assert torch.count_nonzero(outputs["quantiles"].grad) > 0
    assert torch.count_nonzero(outputs["state_logits"].grad) == 0


@pytest.mark.parametrize("missing", ["quantiles", "state_logits"])
def test_rejects_missing_required_output(missing: str) -> None:
    outputs, targets, labels, _ = _batch()
    del outputs[missing]
    with pytest.raises(KeyError, match=missing):
        ReliabilityAObjective(QUANTILES)(outputs, targets, labels)


@pytest.mark.parametrize("name", ["quantiles", "state_logits"])
def test_rejects_required_output_that_is_not_a_tensor(name: str) -> None:
    outputs, targets, labels, _ = _batch()
    outputs[name] = object()
    with pytest.raises(TypeError, match=name):
        ReliabilityAObjective(QUANTILES)(outputs, targets, labels)


@pytest.mark.parametrize("name", ["targets", "state_labels", "daylight_mask", "target_mask"])
def test_rejects_non_tensor_supervision(name: str) -> None:
    outputs, targets, labels, masks = _batch()
    arguments = {"targets": targets, "state_labels": labels, **masks}
    arguments[name] = object()
    with pytest.raises(TypeError, match=name):
        ReliabilityAObjective(QUANTILES)(outputs, **arguments)


@pytest.mark.parametrize("name", ["state_labels", "daylight_mask", "target_mask"])
def test_rejects_mismatched_supervision_shapes(name: str) -> None:
    outputs, targets, labels, masks = _batch()
    arguments = {"targets": targets, "state_labels": labels, **masks}
    arguments[name] = arguments[name][:, :1]
    with pytest.raises(ValueError, match=name):
        ReliabilityAObjective(QUANTILES)(outputs, **arguments)


@pytest.mark.parametrize("name", ["quantiles", "state_logits"])
def test_rejects_mismatched_prediction_shapes(name: str) -> None:
    outputs, targets, labels, _ = _batch()
    outputs[name] = outputs[name][:, :1]
    with pytest.raises(ValueError):
        ReliabilityAObjective(QUANTILES)(outputs, targets, labels)


def test_rejects_state_logits_with_wrong_class_count() -> None:
    outputs, targets, labels, _ = _batch()
    outputs["state_logits"] = outputs["state_logits"][..., :2]
    with pytest.raises(ValueError, match="state_logits"):
        ReliabilityAObjective(QUANTILES)(outputs, targets, labels)


@pytest.mark.parametrize(
    "levels", [(), (0.0, 0.5), (0.5, 1.0), (0.5, 0.5), (0.8, 0.2), (float("nan"),)]
)
def test_rejects_invalid_quantile_levels(levels: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="quantile"):
        ReliabilityAObjective(levels)


def test_accepts_quantiles_without_median_and_rejects_wrong_output_count() -> None:
    objective = ReliabilityAObjective((0.1, 0.9))
    outputs, targets, labels, _ = _batch()
    with pytest.raises(ValueError, match="quantile"):
        objective(outputs, targets, labels)
    outputs["quantiles"] = outputs["quantiles"][..., :2]
    result = objective(outputs, targets, labels)
    assert torch.isfinite(result["loss"])
