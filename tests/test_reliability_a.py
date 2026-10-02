"""CPU synthetic checks of candidate A, not a prediction-performance experiment."""

from __future__ import annotations

import inspect

import pytest
import torch

from racepv.models import RaCEPV, RaCEPVConfig
from racepv.models.experts import ForecastFusionRouter
from racepv.models.reliability_a import (
    RELIABILITY_FEATURE_NAMES,
    ReliabilityAwareFusionRouter,
    ReliabilityGateARaCEPV,
    power_observation_reliability,
)


def _config(**changes):
    values = dict(
        history_features=3,
        future_features=3,
        site_features=1,
        history_length=32,
        horizon=8,
        d_model=8,
        d_state=4,
        num_attention_heads=2,
        router_hidden_size=16,
        dropout=0.0,
        backbone="gru",
        future_hour_sin_index=1,
        future_hour_cos_index=2,
    )
    values.update(changes)
    return RaCEPVConfig(**values)


def _batch(config):
    rng = torch.Generator().manual_seed(15)
    history = 0.1 + 0.6 * torch.rand(2, config.history_length, 3, generator=rng)
    mask = torch.ones_like(history, dtype=torch.bool)
    mask[0, -4:, 0] = False
    future = torch.zeros(2, config.horizon, 3)
    future[..., 0] = 0.7
    future[:, 0, 0] = 0.0
    future[..., 2] = 1.0
    site = torch.zeros(2, 1)
    return history, mask, future, site


def _features(mask, **changes):
    kwargs = dict(power_index=0, sampling_minutes=15)
    kwargs.update(changes)
    return power_observation_reliability(mask, **kwargs)


def test_literal_missing_rates_and_interval_end_age():
    mask = torch.ones(3, 12, 2, dtype=torch.bool)
    mask[0, -4:, 0] = False
    mask[1, :, 0] = False
    mask[2, :-1, 0] = False
    torch.testing.assert_close(
        _features(mask),
        torch.tensor(
            [
                [1.0, 1.0, 1.0 / 3.0, 4.0 / 12.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 0.0],
                [0.5, 0.75, 11.0 / 12.0, 0.0, 1.0],
            ]
        ),
    )
    assert RELIABILITY_FEATURE_NAMES == (
        "power_missing_fraction_30min",
        "power_missing_fraction_60min",
        "power_missing_fraction_180min",
        "last_observed_interval_end_age_fraction",
        "power_observed_any",
    )


def test_other_channels_do_not_change_power_features():
    mask = torch.ones(2, 16, 3, dtype=torch.bool)
    expected = _features(mask)
    mask[..., 1:] = False
    torch.testing.assert_close(_features(mask), expected, rtol=0, atol=0)
    torch.testing.assert_close(_features(mask.float()), expected, rtol=0, atol=0)


def test_age_tracks_oldest_observation_and_history_denominator():
    mask = torch.zeros(1, 16, 1, dtype=torch.bool)
    mask[:, 0] = True
    assert _features(mask)[0, 3].item() == 15.0 / 16.0
    mask[:, -1] = True
    assert _features(mask)[0, 3].item() == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0.2, 2.0])
def test_nonbinary_or_nonfinite_mask_rejected(bad):
    mask = torch.ones(1, 12, 1)
    mask[:, -1, 0] = bad
    with pytest.raises(ValueError, match="binary"):
        _features(mask)


@pytest.mark.parametrize("shape", [(12, 2), (0, 12, 1), (1, 11, 1), (1, 12, 0)])
def test_invalid_mask_shape_rejected(shape):
    with pytest.raises(ValueError):
        _features(torch.ones(shape, dtype=torch.bool))


@pytest.mark.parametrize("power_index", [-1, 2, True, 0.5])
def test_invalid_power_index_rejected(power_index):
    with pytest.raises((ValueError, TypeError)):
        _features(torch.ones(1, 12, 2), power_index=power_index)


@pytest.mark.parametrize("sampling_minutes", [0, 5, 30, float("nan"), True])
def test_candidate_fixes_fifteen_minute_cadence(sampling_minutes):
    with pytest.raises(ValueError, match="15"):
        _features(torch.ones(1, 12, 1), sampling_minutes=sampling_minutes)


def test_feature_api_accepts_only_history_mask_not_future_or_targets():
    assert tuple(inspect.signature(power_observation_reliability).parameters) == (
        "history_mask",
        "power_index",
        "sampling_minutes",
    )


def test_router_has_exact_zero_initialized_fifteen_parameter_correction():
    config = _config()
    router = ReliabilityAwareFusionRouter(ForecastFusionRouter(config))
    assert router.reliability_logits.weight.shape == (3, 5)
    assert router.reliability_logits.bias is None
    assert torch.count_nonzero(router.reliability_logits.weight) == 0
    assert sum(p.numel() for p in router.parameters()) == (
        sum(p.numel() for p in router.base_router.parameters()) + 15
    )


def test_router_actual_logits_add_known_correction_at_existing_temperature():
    config = _config(router_temperature=2.0)
    base = ForecastFusionRouter(config).eval()
    router = ReliabilityAwareFusionRouter(base).eval()
    history, mask, future, site = _batch(config)
    lead = torch.arange(1, config.horizon + 1).float().expand(2, -1) / 4.0
    hidden = torch.zeros(2, config.horizon, config.d_model)
    expert_quantiles = torch.zeros(2, config.horizon, 3, len(config.quantiles))
    args = (history, mask, future, site, lead, lead, hidden, hidden, hidden, expert_quantiles)
    with torch.no_grad():
        router.reliability_logits.weight[0, 0] = 2.0
        actual = router(*args)
        expected = base(*args)
    difference = actual - expected
    torch.testing.assert_close(difference[0, :, 0], torch.ones(config.horizon))
    torch.testing.assert_close(difference[0, :, 1:], torch.zeros(config.horizon, 2))
    torch.testing.assert_close(difference[1], torch.zeros(config.horizon, 3))


def test_same_seed_zero_correction_preserves_full_original_model_outputs():
    config = _config()
    torch.manual_seed(37)
    baseline = RaCEPV(config).eval()
    torch.manual_seed(37)
    candidate = ReliabilityGateARaCEPV(config).eval()
    with torch.no_grad():
        expected = baseline(*_batch(config))
        actual = candidate(*_batch(config))
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert actual["expert_quantiles"].shape[2] == 3


def test_reliability_correction_changes_fusion_but_not_state_or_expert_outputs():
    model = ReliabilityGateARaCEPV(_config()).eval()
    inputs = _batch(model.config)
    with torch.no_grad():
        before = model(*inputs)
        model.fusion_router.reliability_logits.weight[0, 0] = 2.0
        after = model(*inputs)
    assert not torch.equal(after["gate_weights"], before["gate_weights"])
    for name in ("state_logits", "state_probs", "expert_quantiles"):
        torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)
    assert torch.isfinite(after["quantiles"]).all()
    assert torch.all(torch.diff(after["quantiles"], dim=-1) >= 0)
    assert torch.all((after["quantiles"] >= 0) & (after["quantiles"] <= 1.05))
    assert torch.count_nonzero(after["quantiles"][:, 0]) == 0


def test_model_loss_backward_learns_gate_without_specialist_term():
    model = ReliabilityGateARaCEPV(_config())
    outputs = model(*_batch(model.config))
    result = model.make_objective()(
        outputs,
        torch.full((2, 8), 0.4),
        torch.zeros(2, 8, dtype=torch.long),
        daylight_mask=torch.tensor([[False] + [True] * 7] * 2),
    )
    assert not any("specialist" in key for key in result)
    result["loss"].backward()
    gradient = model.fusion_router.reliability_logits.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_state_only_backward_still_does_not_update_forecast_gate_or_experts():
    model = ReliabilityGateARaCEPV(_config())
    model(*_batch(model.config))["state_logits"].sum().backward()
    for module in (model.fusion_router, model.stable_expert, model.transient_expert):
        assert all(p.grad is None for p in module.parameters())
    assert any(p.grad is not None for p in model.state_head.parameters())


def test_nonfinite_power_is_missing_even_if_incoming_mask_claims_observed():
    model = ReliabilityGateARaCEPV(_config()).eval()
    history, mask, future, site = _batch(model.config)
    history[:, -1, 0] = float("nan")
    seen = []
    handle = model.fusion_router.reliability_logits.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().clone())
    )
    try:
        model(history, mask, future, site)
    finally:
        handle.remove()
    assert seen[0][1, 0].item() == 0.5
    assert seen[0][1, 3].item() == 1.0 / 32.0


def test_model_rejects_nonbinary_mask_before_legacy_bool_conversion():
    model = ReliabilityGateARaCEPV(_config())
    history, mask, future, site = _batch(model.config)
    mask = mask.float()
    mask[:, 0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        model(history, mask, future, site)


def test_candidate_metadata_and_old_checkpoint_are_distinct():
    config = _config()
    model = ReliabilityGateARaCEPV(config)
    metadata = model.model_metadata
    assert metadata["model_revision"] == "racepv.reliability-gate-a.v1"
    assert metadata["architecture_version"] != 2
    assert metadata["base_architecture_version"] == 2
    assert metadata["objective_version"] == model.make_objective().objective_id
    assert metadata["candidate_only"] is True
    assert metadata["actual_backbone"] == "causal_conv_gru"
    assert metadata["reliability_features"] == list(RELIABILITY_FEATURE_NAMES)
    with pytest.raises(RuntimeError):
        model.load_state_dict(RaCEPV(config).state_dict(), strict=True)


@pytest.mark.parametrize(
    "changes",
    [
        {"prediction_state_coupling": "joint_head"},
        {"prediction_state_coupling": "state_as_gate"},
        {"fusion_mode": "equal"},
        {"router_mode": "amplitude_only"},
        {"active_experts": ("stable",)},
        {"enable_auxiliary_gate_path": True},
        {"state_representation_mode": "independent_raw"},
        {"sampling_minutes": 30},
        {"history_length": 8},
    ],
)
def test_candidate_rejects_unapproved_additional_model_changes(changes):
    with pytest.raises(ValueError):
        ReliabilityGateARaCEPV(_config(**changes))
