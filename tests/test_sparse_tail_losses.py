from __future__ import annotations

import pytest
import torch

from racepv import losses as loss_module
from racepv.losses import RaCEPVLoss, trapezoidal_qcrps
from racepv.models import RaCEPV, RaCEPVConfig
from racepv.training import EarlyStoppingTrainer


def _config() -> RaCEPVConfig:
    return RaCEPVConfig(
        history_features=5,
        future_features=4,
        site_features=2,
        history_length=32,
        horizon=8,
        d_model=16,
        d_state=8,
        d_conv=4,
        num_attention_heads=4,
        router_hidden_size=32,
        dropout=0.0,
        backbone="gru",
        future_hour_sin_index=1,
        future_hour_cos_index=2,
        tail_residual_mode="sparse_ramp_up",
    )


def _batch(config: RaCEPVConfig) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20260830)
    history = torch.rand(
        2,
        config.history_length,
        config.history_features,
        generator=generator,
    )
    history[..., config.history_clear_sky_index] = 0.8
    history[..., config.power_index] = 0.5
    mask = torch.ones_like(history, dtype=torch.bool)
    future = torch.rand(
        2,
        config.horizon,
        config.future_features,
        generator=generator,
    )
    clear_sky = torch.tensor([0.0, 0.0, 0.1, 0.4, 0.8, 0.6, 0.2, 0.0])
    future[..., config.future_clear_sky_index] = clear_sky
    hours = torch.arange(config.horizon, dtype=future.dtype) + 6.0
    phase = 2.0 * torch.pi * hours / 24.0
    future[..., config.future_hour_sin_index] = torch.sin(phase)
    future[..., config.future_hour_cos_index] = torch.cos(phase)
    site = torch.rand(2, config.site_features, generator=generator)
    return history, mask, future, site


def _targets_and_labels(config: RaCEPVConfig) -> tuple[torch.Tensor, ...]:
    targets = torch.tensor(
        [
            [0.0, 0.0, 0.25, 0.65, 0.95, 0.40, 0.10, 0.0],
            [0.0, 0.0, 0.20, 0.55, 0.85, 0.35, 0.15, 0.0],
        ]
    )
    labels = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 0, 0],
            [0, 0, 1, 2, 1, 2, 0, 0],
        ]
    )
    daylight = torch.zeros(2, config.horizon, dtype=torch.bool)
    daylight[:, 2:7] = True
    return targets, labels, daylight


def _has_nonzero_grad(parameters: object) -> bool:
    return any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in parameters  # type: ignore[union-attr]
    )


def _has_no_nonzero_grad(parameters: object) -> bool:
    return all(
        parameter.grad is None or not bool(torch.count_nonzero(parameter.grad))
        for parameter in parameters  # type: ignore[union-attr]
    )


def test_ramp_up_tail_mse_selects_only_finite_daylight_label_one() -> None:
    """Catch stable, ramp-down, missing, or night cells entering tail MSE."""

    base = torch.tensor([[[0.2], [0.3], [0.4], [0.5], [0.6]]], requires_grad=True)
    correction = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], requires_grad=True)
    targets = torch.tensor([[0.5, 0.8, 0.9, float("nan"), 1.0]])
    labels = torch.tensor([[1, 2, 0, 1, -100]])
    daylight = torch.tensor([[True, True, True, True, True]])

    loss_function = getattr(loss_module, "ramp_up_tail_mse_loss", None)
    assert loss_function is not None
    loss = loss_function(
        base,
        correction,
        targets,
        labels,
        median_index=0,
        daylight_mask=daylight,
    )

    assert loss.item() == pytest.approx(0.04)
    loss.backward()
    assert base.grad is None
    assert correction.grad is not None
    assert torch.count_nonzero(correction.grad) == 1


def test_trigger_sparsity_uses_only_finite_daylight_cells() -> None:
    """Catch missing targets or night cells diluting the sparsity penalty."""

    trigger = torch.tensor([[0.1, 0.2, 0.3, 0.4]], requires_grad=True)
    targets = torch.tensor([[0.5, float("nan"), 0.8, 0.9]])
    daylight = torch.tensor([[True, True, False, True]])

    loss_function = getattr(loss_module, "tail_trigger_sparsity_loss", None)
    assert loss_function is not None
    loss = loss_function(trigger, targets, daylight_mask=daylight)

    assert loss.item() == pytest.approx(0.25)
    loss.backward()
    assert trigger.grad is not None
    assert torch.count_nonzero(trigger.grad) == 2


def test_tail_losses_and_state_loss_respect_gradient_boundaries() -> None:
    """Catch auxiliary or state gradients leaking across optimizer ownership."""

    config = _config()
    model = RaCEPV(config).train()
    targets, labels, daylight = _targets_and_labels(config)
    objective = RaCEPVLoss(
        config.quantiles,
        specialist_weight=0.15,
        tail_auxiliary_weight=0.02,
        tail_sparsity_weight=0.001,
    )

    outputs = model(*_batch(config))
    losses = objective(outputs, targets, labels, daylight_mask=daylight)
    losses["state_loss"].backward()
    assert model.tail_residual_head is not None
    assert _has_no_nonzero_grad(model.tail_residual_head.parameters())

    model.zero_grad(set_to_none=True)
    outputs = model(*_batch(config))
    losses = objective(outputs, targets, labels, daylight_mask=daylight)
    tail_only = (
        0.02 * losses["tail_ramp_up_mse"]
        + 0.001 * losses["tail_trigger_sparsity"]
    )
    tail_only.backward()
    assert _has_nonzero_grad(model.tail_residual_head.parameters())
    assert _has_no_nonzero_grad(model.stable_expert.parameters())
    assert _has_no_nonzero_grad(model.transient_expert.parameters())
    assert _has_no_nonzero_grad(model.fusion_router.parameters())
    assert _has_no_nonzero_grad(model.state_head.parameters())


def test_main_qcrps_updates_tail_head_and_forecast_branch() -> None:
    """Catch detachment of the residual head from the registered main score."""

    config = _config()
    model = RaCEPV(config).train()
    targets, _labels, daylight = _targets_and_labels(config)
    outputs = model(*_batch(config))

    loss = trapezoidal_qcrps(
        outputs["quantiles"],
        targets,
        config.quantiles,
        mask=daylight,
    )
    loss.backward()

    assert model.tail_residual_head is not None
    assert _has_nonzero_grad(model.tail_residual_head.parameters())
    assert _has_nonzero_grad(model.stable_expert.parameters())
    assert _has_nonzero_grad(model.fusion_router.parameters())
    assert _has_no_nonzero_grad(model.state_head.parameters())


def test_separated_optimizer_assigns_tail_head_to_forecast_only() -> None:
    """Catch residual parameters entering the state optimizer or neither group."""

    config = _config()
    model = RaCEPV(config)
    trainer = EarlyStoppingTrainer(
        model,
        RaCEPVLoss(config.quantiles),
        mixed_precision=False,
        separate_state_optimizer=True,
    )
    assert model.tail_residual_head is not None
    tail_ids = {id(parameter) for parameter in model.tail_residual_head.parameters()}
    forecast_ids = {id(parameter) for parameter in trainer.forecast_parameters}
    state_ids = {id(parameter) for parameter in trainer.state_parameters}

    assert tail_ids
    assert tail_ids <= forecast_ids
    assert tail_ids.isdisjoint(state_ids)
