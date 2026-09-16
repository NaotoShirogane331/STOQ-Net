from __future__ import annotations

import inspect

import pytest
import torch

from racepv import losses as loss_module
from racepv.losses import (
    RaCEPVLoss,
    hierarchical_state_loss,
    pinball_loss,
    ramp_expert_median_smooth_l1_loss,
    ramp_median_smooth_l1_loss,
    specialist_masked_loss,
    weighted_focal_loss,
)
from racepv.models import RaCEPV, RaCEPVConfig
from racepv.models import components as model_components
from racepv.models import experts as model_experts


def _config(**overrides: object) -> RaCEPVConfig:
    values: dict[str, object] = {
        "history_features": 5,
        "future_features": 4,
        "site_features": 2,
        "history_length": 32,
        "horizon": 8,
        "d_model": 16,
        "d_state": 8,
        "d_conv": 4,
        "num_attention_heads": 4,
        "router_hidden_size": 32,
        "dropout": 0.0,
        "backbone": "gru",
        "future_hour_sin_index": 1,
        "future_hour_cos_index": 2,
    }
    values.update(overrides)
    return RaCEPVConfig(**values)  # type: ignore[arg-type]


def _batch(config: RaCEPVConfig, batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20260804)
    history = torch.rand(
        batch_size,
        config.history_length,
        config.history_features,
        generator=generator,
    )
    # Power and historical clear-sky power have a physically meaningful scale.
    clear_sky_history = torch.linspace(0.05, 0.95, config.history_length)
    history[..., config.history_clear_sky_index] = clear_sky_history
    history[..., config.power_index] = (
        0.75 * clear_sky_history
        + 0.08 * torch.rand(batch_size, config.history_length, generator=generator)
    ).clamp(0.0, 1.05)
    mask = torch.ones_like(history, dtype=torch.bool)
    mask[0, 4:6, 3] = False
    history[0, 4:6, 3] = float("nan")

    future = torch.rand(
        batch_size,
        config.horizon,
        config.future_features,
        generator=generator,
    )
    clear_sky_future = torch.tensor([0.0, 0.0, 0.12, 0.35, 0.62, 0.80, 0.40, 0.0])
    future[..., config.future_clear_sky_index] = clear_sky_future
    local_hour = (
        torch.arange(config.horizon, dtype=future.dtype) * (config.sampling_minutes / 60.0) + 6.0
    )
    phase = 2.0 * torch.pi * local_hour / 24.0
    future[..., config.future_hour_sin_index] = torch.sin(phase)
    future[..., config.future_hour_cos_index] = torch.cos(phase)
    site = torch.rand(batch_size, config.site_features, generator=generator)
    return history, mask, future, site


def test_public_forward_interface_and_output_shapes() -> None:
    config = _config()
    model = RaCEPV(config).eval()
    history, mask, future, site = _batch(config)

    parameters = tuple(inspect.signature(model.forward).parameters)
    assert parameters == (
        "history_values",
        "history_mask",
        "future_known",
        "site_metadata",
        "future_weather",
        "future_weather_mask",
    )
    with torch.no_grad():
        outputs = model(history, mask, future, site)

    assert set(outputs) == {
        "quantiles",
        "base_quantiles",
        "tail_trigger",
        "tail_correction",
        "state_probs",
        "expert_quantiles",
        "gate_weights",
        "fusion_logits",
        "state_logits",
        "router_logits",
        "ramp_presence_logits",
        "ramp_direction_logits",
    }
    assert outputs["quantiles"].shape == (2, config.horizon, len(config.quantiles))
    assert outputs["state_probs"].shape == (2, config.horizon, 3)
    assert outputs["expert_quantiles"].shape == (
        2,
        config.horizon,
        3,
        len(config.quantiles),
    )
    assert outputs["gate_weights"].shape == (2, config.horizon, 3)
    assert outputs["fusion_logits"].shape == (2, config.horizon, 3)
    assert outputs["state_logits"].shape == (2, config.horizon, 3)
    assert outputs["router_logits"].shape == (2, config.horizon, 3)
    assert outputs["ramp_presence_logits"].shape == (2, config.horizon)
    assert outputs["ramp_direction_logits"].shape == (2, config.horizon, 2)
    torch.testing.assert_close(
        outputs["state_probs"].sum(dim=-1),
        torch.ones(2, config.horizon),
    )
    torch.testing.assert_close(
        outputs["state_probs"],
        torch.softmax(outputs["state_logits"], dim=-1),
    )
    torch.testing.assert_close(
        outputs["gate_weights"],
        torch.softmax(outputs["fusion_logits"], dim=-1),
    )
    torch.testing.assert_close(outputs["router_logits"], outputs["state_logits"])


def test_auxiliary_gate_path_updates_router_without_contaminating_experts() -> None:
    """Catch reuse of the main gate graph by a supposedly router-only loss."""

    config = _config(enable_auxiliary_gate_path=True)
    model = RaCEPV(config).train()
    history, mask, future, site = _batch(config)
    outputs = model(history, mask, future, site)
    targets = torch.full((2, config.horizon), 0.45)
    labels = torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2], [2, 1, 2, 1, 2, 1, 2, 1]])
    daylight = torch.ones_like(labels, dtype=torch.bool)

    loss = loss_module.ramp_oracle_gate_regret_loss(
        outputs["expert_quantiles"],
        outputs["_auxiliary_gate_weights"],
        targets,
        labels,
        median_index=config.median_index,
        daylight_mask=daylight,
    )
    loss.backward()

    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.fusion_router.parameters()
    )
    for module in (model.stable_expert, model.transient_expert, model.state_head):
        assert all(
            parameter.grad is None or not bool(torch.count_nonzero(parameter.grad))
            for parameter in module.parameters()
        )


def test_default_forward_omits_internal_auxiliary_gate() -> None:
    config = _config()
    outputs = RaCEPV(config).eval()(*_batch(config))

    assert "_auxiliary_gate_weights" not in outputs


def test_auxiliary_gate_forward_preserves_global_cpu_rng_trajectory() -> None:
    control_config = _config(dropout=0.2, enable_auxiliary_gate_path=False)
    candidate_config = _config(dropout=0.2, enable_auxiliary_gate_path=True)
    control = RaCEPV(control_config).train()
    candidate = RaCEPV(candidate_config).train()
    candidate.load_state_dict(control.state_dict())
    batch = _batch(control_config)

    torch.manual_seed(20260826)
    control(*batch)
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(20260826)
    candidate_outputs = candidate(*batch)
    observed_rng = torch.random.get_rng_state()

    assert "_auxiliary_gate_weights" in candidate_outputs
    torch.testing.assert_close(observed_rng, expected_rng, rtol=0.0, atol=0.0)


def test_state_and_fusion_branches_are_semantically_and_gradient_isolated() -> None:
    config = _config()
    model = RaCEPV(config).train()
    history, mask, future, site = _batch(config)

    outputs = model(history, mask, future, site)
    daylight = future[..., config.future_clear_sky_index] >= config.daylight_threshold
    assert not torch.allclose(outputs["state_probs"][daylight], outputs["gate_weights"][daylight])

    outputs["state_logits"][daylight].sum().backward()
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.state_head.parameters()
    )

    for module in (model.stable_expert, model.transient_expert, model.fusion_router):
        assert all(
            parameter.grad is None or not bool(torch.count_nonzero(parameter.grad))
            for parameter in module.parameters()
        )

    model.zero_grad(set_to_none=True)
    outputs = model(history, mask, future, site)
    outputs["quantiles"][daylight].sum().backward()
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.fusion_router.parameters()
    )
    assert all(parameter.grad is None for parameter in model.state_head.parameters())


def test_independent_state_head_signature_excludes_expert_outputs() -> None:
    config = _config(state_representation_mode="independent_raw")
    model = RaCEPV(config)

    assert isinstance(model.state_head, model_experts.IndependentRampStateHead)
    assert tuple(inspect.signature(model.state_head.forward).parameters) == (
        "history_values",
        "history_mask",
        "future_known",
        "site_metadata",
        "lead_time_hours",
        "physical_time_hours",
    )


def test_independent_state_logits_do_not_depend_on_expert_parameters() -> None:
    config = _config(state_representation_mode="independent_raw")
    model = RaCEPV(config).eval()
    batch = _batch(config)

    with torch.no_grad():
        before = model(*batch)["state_logits"].clone()
        for module in (
            model.stable_expert,
            model.transient_expert,
            model.fusion_router,
        ):
            for parameter in module.parameters():
                parameter.add_(torch.randn_like(parameter))
        after = model(*batch)["state_logits"]

    torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)


def test_state_representation_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="state_representation_mode"):
        _config(state_representation_mode="shared_again")


def test_ordered_quantiles_and_shared_gate_give_convex_fusion() -> None:
    config = _config()
    model = RaCEPV(config).eval()
    history, mask, future, site = _batch(config)
    with torch.no_grad():
        outputs = model(history, mask, future, site)

    final_differences = outputs["quantiles"].diff(dim=-1)
    expert_differences = outputs["expert_quantiles"].diff(dim=-1)
    assert bool(torch.isfinite(outputs["quantiles"]).all())
    assert bool(torch.isfinite(outputs["expert_quantiles"]).all())
    assert bool((outputs["quantiles"] >= 0.0).all())
    assert bool((outputs["quantiles"] <= config.power_upper_bound).all())
    assert bool((outputs["expert_quantiles"] >= 0.0).all())
    assert bool((outputs["expert_quantiles"] <= config.power_upper_bound).all())
    assert bool((final_differences >= -1e-7).all())
    assert bool((expert_differences >= -1e-7).all())

    expected = (outputs["gate_weights"].unsqueeze(-1) * outputs["expert_quantiles"]).sum(dim=2)
    torch.testing.assert_close(outputs["quantiles"], expected)
    assert bool((outputs["gate_weights"] >= 0).all())
    assert bool((outputs["gate_weights"] <= 1).all())


def test_night_is_zero_and_routes_to_stable() -> None:
    config = _config()
    model = RaCEPV(config).eval()
    history, mask, future, site = _batch(config)
    night = future[..., config.future_clear_sky_index] < config.daylight_threshold
    with torch.no_grad():
        outputs = model(history, mask, future, site)

    assert torch.count_nonzero(outputs["quantiles"][night]) == 0
    assert torch.count_nonzero(outputs["expert_quantiles"][night]) == 0
    expected_state = torch.tensor([1.0, 0.0, 0.0]).expand(int(night.sum()), -1)
    torch.testing.assert_close(outputs["state_probs"][night], expected_state)
    assert bool(torch.isfinite(outputs["gate_weights"][night]).all())
    torch.testing.assert_close(
        outputs["gate_weights"][night].sum(dim=-1),
        torch.ones(int(night.sum())),
    )


def test_actual_backbone_is_exposed_without_mislabeling() -> None:
    model = RaCEPV(_config())
    assert model.backbone_name == "causal_conv_gru"
    assert model.model_metadata == {
        "architecture_version": 2,
        "objective_version": 2,
        "forecast_selection_metric": "daylight_qcrps",
        "state_head_semantics": "physical_ramp_state_classification",
        "fusion_gate_semantics": "forecast_risk_convex_expert_fusion",
        "tail_residual_mode": "off",
        "tail_residual_cap": 0.08,
        "tail_residual_hidden_size": 64,
        "tail_residual_semantics": "detached_forecast_sparse_signed_quantile_shift",
        "all_missing_power_fallback": "clear_sky_only",
        "requested_backbone": "gru",
        "actual_backbone": "causal_conv_gru",
        "uses_official_mamba": False,
        "use_clear_sky_prior": True,
        "router_mode": "contextual",
        "fusion_mode": "learned",
            "state_head_mode": "flat",
            "state_representation_mode": "expert_detached",
            "prediction_state_coupling": "decoupled",
            "state_gradient_updates_forecast": False,
            "forecast_gradient_updates_state": False,
            "fusion_weight_semantics": "forecast_risk_convex_expert_fusion",
            "enable_auxiliary_gate_path": False,
        "ramp_decision_threshold": 0.25,
        "forecast_weather_mode": "off",
        "forecast_weather_contract_version": 1,
        "future_weather_features": 0,
        "future_weather_hidden_size": 16,
        "weather_direct_expert_access": False,
        "weather_direct_stable_expert_access": False,
        "weather_direct_transient_expert_access": False,
        "weather_direct_fusion_access": False,
        "weather_indirect_fusion_access": False,
        "fusion_representation_semantics": "forecast_expert_representations",
        "state_head_weather_access": False,
        "active_experts": ["stable", "ramp_up", "ramp_down"],
        "clear_sky_history_indices": [1],
        "clear_sky_future_indices": [0],
    }


def test_real_clock_features_and_v2_configuration_fail_closed() -> None:
    hours = torch.tensor([[0.0, 6.0, 12.0, 18.0, 23.5]])
    phase = 2.0 * torch.pi * hours / 24.0
    recovered = model_components.cyclic_hour_to_hours(torch.sin(phase), torch.cos(phase))
    torch.testing.assert_close(recovered, hours, atol=1e-5, rtol=0.0)

    lead = torch.full_like(hours, 0.25)
    basis = model_components.make_harmonic_basis(
        recovered,
        lead,
        (24.0, 12.0, 8.0, 6.0),
    )
    assert not torch.allclose(basis[:, 1], basis[:, 3])

    with pytest.raises(ValueError, match="future_hour_sin_index.*future_hour_cos_index"):
        RaCEPVConfig(history_features=2, future_features=3)


def test_last_observed_power_and_persistence_anchor_are_mask_aware() -> None:
    power = torch.tensor(
        [
            [0.10, 0.35, 0.90, 0.20],
            [float("nan"), 0.40, 0.80, 0.60],
            [0.70, 0.60, 0.50, 0.40],
        ]
    )
    observed = torch.tensor(
        [
            [True, True, False, False],
            [False, True, False, False],
            [False, False, False, False],
        ]
    )
    last, available = model_experts.last_observed_power(power, observed)
    torch.testing.assert_close(last, torch.tensor([0.35, 0.40, 0.0]))
    assert torch.equal(available, torch.tensor([True, True, False]))

    clear_sky = torch.tensor([[0.20, 0.60], [0.30, 0.70], [0.40, 0.80]])
    lead = torch.tensor([[0.25, 6.0]]).expand(3, -1)
    tau = torch.tensor(1.0)
    anchor = model_experts.persistence_clear_sky_anchor(
        last,
        available,
        clear_sky,
        lead,
        tau,
    )
    expected_weight = torch.exp(-lead / tau)
    expected_first = expected_weight[0] * 0.35 + (1.0 - expected_weight[0]) * clear_sky[0]
    torch.testing.assert_close(anchor[0], expected_first)
    torch.testing.assert_close(anchor[2], clear_sky[2])


def test_learned_persistence_tau_is_bounded() -> None:
    model = RaCEPV(_config())
    tau = model.stable_expert.persistence_tau_hours.detach()
    assert 0.25 <= float(tau) <= 6.0
    assert float(tau) == pytest.approx(1.0, abs=1e-5)

    with torch.no_grad():
        model.stable_expert.persistence_tau_unconstrained.fill_(1_000.0)
    assert float(model.stable_expert.persistence_tau_hours.detach()) == pytest.approx(6.0)
    with torch.no_grad():
        model.stable_expert.persistence_tau_unconstrained.fill_(-1_000.0)
    assert float(model.stable_expert.persistence_tau_hours.detach()) == pytest.approx(0.25)


def test_all_missing_power_history_falls_back_to_clear_sky_only() -> None:
    config = _config()
    model = RaCEPV(config).eval()
    history, mask, future, site = _batch(config)
    history[..., config.power_index] = float("nan")
    mask[..., config.power_index] = False
    with torch.no_grad():
        outputs = model(history, mask, future, site)

    clear_sky = future[..., config.future_clear_sky_index]
    expected_experts = clear_sky[..., None, None].expand_as(outputs["expert_quantiles"])
    expected_fused = clear_sky[..., None].expand_as(outputs["quantiles"])
    torch.testing.assert_close(outputs["expert_quantiles"], expected_experts)
    torch.testing.assert_close(outputs["quantiles"], expected_fused)


def test_tcn_ablation_has_distinct_auditable_backend() -> None:
    config = _config(backbone="tcn")
    model = RaCEPV(config).eval()
    history, mask, future, site = _batch(config)
    with torch.no_grad():
        outputs = model(history, mask, future, site)
    assert outputs["quantiles"].shape == (2, config.horizon, len(config.quantiles))
    assert model.backbone_name == "causal_tcn"
    assert model.model_metadata["actual_backbone"] == "causal_tcn"
    assert model.model_metadata["uses_official_mamba"] is False


@pytest.mark.parametrize("fusion_mode", ["learned", "equal"])
def test_inactive_experts_have_zero_renormalized_gate(fusion_mode: str) -> None:
    config = _config(
        fusion_mode=fusion_mode,
        active_experts=("stable", "ramp_down"),
    )
    model = RaCEPV(config).eval()
    history, mask, future, site = _batch(config)
    with torch.no_grad():
        outputs = model(history, mask, future, site)

    gates = outputs["gate_weights"]
    assert torch.count_nonzero(gates[..., 1]) == 0
    torch.testing.assert_close(gates.sum(dim=-1), torch.ones_like(gates[..., 0]))
    if fusion_mode == "equal":
        expected = torch.tensor([0.5, 0.0, 0.5]).expand_as(gates)
        torch.testing.assert_close(gates, expected)


def test_single_active_expert_keeps_fixed_three_expert_shapes() -> None:
    config = _config(active_experts=("ramp_up",))
    model = RaCEPV(config).eval()
    history, mask, future, site = _batch(config)
    with torch.no_grad():
        outputs = model(history, mask, future, site)
    expected_gate = torch.tensor([0.0, 1.0, 0.0]).expand_as(outputs["gate_weights"])
    torch.testing.assert_close(outputs["gate_weights"], expected_gate)
    assert outputs["expert_quantiles"].shape[2] == 3
    assert outputs["state_probs"].shape[-1] == 3


def test_amplitude_router_ignores_context_future_and_expert_predictions() -> None:
    config = _config(router_mode="amplitude_only")
    model = RaCEPV(config).eval()
    history, mask, future, site = _batch(config)
    altered_history = history.clone()
    altered_history[:, :-1] = torch.flip(altered_history[:, :-1], dims=(1,))
    altered_future = future.clone()
    altered_future[..., 1:] = 10.0 * altered_future[..., 1:] - 3.0
    altered_site = site + 100.0
    with torch.no_grad():
        original = model(history, mask, future, site)["router_logits"]
        altered = model(altered_history, mask, altered_future, altered_site)["router_logits"]
    torch.testing.assert_close(original, altered)
    # A scalar current-amplitude router is horizon-invariant before the common
    # night override.
    daylight = future[..., config.future_clear_sky_index] >= config.daylight_threshold
    for batch_index in range(original.shape[0]):
        daylight_logits = original[batch_index, daylight[batch_index]]
        torch.testing.assert_close(
            daylight_logits,
            daylight_logits[:1].expand_as(daylight_logits),
        )


def test_no_clear_sky_prior_uses_clear_sky_only_for_night_boundary() -> None:
    config = _config(
        use_clear_sky_prior=False,
        clear_sky_history_feature_indices=(2,),
    )
    model = RaCEPV(config).eval()
    assert model.model_metadata["all_missing_power_fallback"] == "zero_no_clear_sky_prior"
    history, mask, future, site = _batch(config)
    changed_history = history.clone()
    changed_history[..., 1] = 0.99 - torch.nan_to_num(changed_history[..., 1])
    changed_history[..., 2] = 50.0
    changed_future = future.clone()
    original_night = future[..., config.future_clear_sky_index] < config.daylight_threshold
    changed_future[..., config.future_clear_sky_index] = torch.where(
        original_night,
        torch.zeros_like(changed_future[..., config.future_clear_sky_index]),
        torch.full_like(changed_future[..., config.future_clear_sky_index], 0.99),
    )
    with torch.no_grad():
        original = model(history, mask, future, site)
        changed = model(changed_history, mask, changed_future, site)
    for key in original:
        torch.testing.assert_close(original[key], changed[key])


def test_input_shape_validation() -> None:
    config = _config()
    model = RaCEPV(config)
    history, mask, future, site = _batch(config)
    with pytest.raises(ValueError, match="history_values has shape"):
        model(history[:, :-1], mask[:, :-1], future, site)
    with pytest.raises(ValueError, match="future_known must be finite"):
        broken_future = future.clone()
        broken_future[0, 0, 1] = float("nan")
        model(history, mask, broken_future, site)

    with pytest.raises(ValueError, match="router_hidden_size must be at least 2"):
        RaCEPVConfig(
            history_features=2,
            future_features=3,
            future_hour_sin_index=1,
            future_hour_cos_index=2,
            router_hidden_size=1,
        )
    with pytest.raises(ValueError, match="unknown active experts"):
        _config(active_experts=("stable", "not_an_expert"))
    with pytest.raises(ValueError, match="active_experts must contain"):
        _config(active_experts=())


def test_individual_losses_have_expected_behavior() -> None:
    prediction = torch.tensor([[[0.0, 1.0, 2.0]]])
    target = torch.tensor([[1.0]])
    result = pinball_loss(prediction, target, (0.25, 0.5, 0.75))
    torch.testing.assert_close(result, torch.tensor(1.0 / 6.0))

    uniform = weighted_focal_loss(
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, dtype=torch.long),
        gamma=2.0,
    )
    confident = weighted_focal_loss(
        torch.tensor([[[8.0, -4.0, -4.0]]]),
        torch.zeros(1, 1, dtype=torch.long),
        gamma=2.0,
    )
    assert confident < uniform

    experts = target[..., None, None].expand(1, 1, 3, 3).clone()
    specialist = specialist_masked_loss(
        experts,
        target,
        torch.zeros_like(target, dtype=torch.long),
        (0.25, 0.5, 0.75),
    )
    torch.testing.assert_close(specialist, torch.tensor(0.0))

    # A disabled specialist must not affect the ablation objective.
    two_targets = torch.tensor([[0.4, 0.7]])
    two_labels = torch.tensor([[0, 1]])
    bad_experts = torch.zeros(1, 2, 3, 3)
    bad_experts[:, 0, 0, :] = 0.4
    bad_experts[:, 1, 1, :] = 100.0
    active_only = specialist_masked_loss(
        bad_experts,
        two_targets,
        two_labels,
        (0.25, 0.5, 0.75),
        active_experts=(True, False, False),
    )
    torch.testing.assert_close(active_only, torch.tensor(0.0))


def test_trapezoidal_qcrps_matches_hand_integrated_pinball() -> None:
    prediction = torch.tensor([[[0.0, 1.0, 2.0]]])
    target = torch.tensor([[1.0]])
    score = loss_module.trapezoidal_qcrps(
        prediction,
        target,
        (0.25, 0.5, 0.75),
    )
    torch.testing.assert_close(score, torch.tensor(0.25))


def test_v2_objective_is_invariant_to_appended_night_zero_samples() -> None:
    quantiles = (0.25, 0.5, 0.75)
    objective = RaCEPVLoss(quantiles, specialist_weight=0.3)
    daylight_outputs = {
        "quantiles": torch.tensor([[[0.2, 0.4, 0.6], [0.4, 0.6, 0.8]]]),
        "expert_quantiles": torch.tensor(
            [
                [
                    [[0.2, 0.4, 0.6], [0.1, 0.3, 0.5], [0.3, 0.5, 0.7]],
                    [[0.4, 0.6, 0.8], [0.5, 0.7, 0.9], [0.2, 0.4, 0.6]],
                ]
            ]
        ),
        "state_logits": torch.tensor([[[2.0, 0.0, -1.0], [-1.0, 2.0, 0.0]]]),
    }
    targets = torch.tensor([[0.5, 0.7]])
    labels = torch.tensor([[0, 1]])
    daylight = torch.ones_like(targets, dtype=torch.bool)
    original = objective(daylight_outputs, targets, labels, daylight_mask=daylight)

    night_quantiles = torch.zeros(1, 2, 3)
    extended_outputs = {
        "quantiles": torch.cat((daylight_outputs["quantiles"], night_quantiles), dim=1),
        "expert_quantiles": torch.cat(
            (
                daylight_outputs["expert_quantiles"],
                torch.zeros(1, 2, 3, 3),
            ),
            dim=1,
        ),
        "state_logits": torch.cat((daylight_outputs["state_logits"], torch.zeros(1, 2, 3)), dim=1),
    }
    extended = objective(
        extended_outputs,
        torch.cat((targets, torch.zeros(1, 2)), dim=1),
        torch.cat((labels, torch.full((1, 2), -100)), dim=1),
        daylight_mask=torch.cat((daylight, torch.zeros(1, 2, dtype=torch.bool)), dim=1),
    )
    for name in (
        "loss",
        "forecast_loss",
        "state_loss",
        "daylight_qcrps",
        "specialist_qcrps",
        "ramp_auxiliary",
        "state_ce",
        "state_brier",
        "state_nll",
    ):
        torch.testing.assert_close(extended[name], original[name])


def test_combined_loss_is_finite_and_backpropagates() -> None:
    config = _config()
    model = RaCEPV(config).train()
    history, mask, future, site = _batch(config)
    outputs = model(history, mask, future, site)
    generator = torch.Generator().manual_seed(17)
    targets = torch.rand(2, config.horizon, generator=generator)
    state_labels = torch.randint(0, 3, (2, config.horizon), generator=generator)
    daylight = future[..., config.future_clear_sky_index] >= config.daylight_threshold

    objective = RaCEPVLoss(
        config.quantiles,
        specialist_weight=0.3,
    )
    losses = objective(outputs, targets, state_labels, daylight_mask=daylight)
    assert set(losses) == {
        "loss",
        "forecast_loss",
        "state_loss",
        "daylight_qcrps",
        "state_ce",
        "state_brier",
        "state_nll",
        "specialist_qcrps",
        "ramp_auxiliary",
        "tail_ramp_up_mse",
        "tail_trigger_sparsity",
    }
    torch.testing.assert_close(losses["tail_ramp_up_mse"], torch.tensor(0.0))
    torch.testing.assert_close(losses["tail_trigger_sparsity"], torch.tensor(0.0))
    assert all(bool(torch.isfinite(value)) for value in losses.values())
    losses["loss"].backward()
    trainable_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert trainable_gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in trainable_gradients)


def test_state_ce_and_forecast_qcrps_have_bidirectional_gradient_isolation() -> None:
    config = _config()
    model = RaCEPV(config).train()
    history, mask, future, site = _batch(config)
    targets = torch.full((2, config.horizon), 0.5)
    labels = torch.arange(config.horizon).remainder(3).expand(2, -1)
    daylight = future[..., config.future_clear_sky_index] >= config.daylight_threshold
    objective = RaCEPVLoss(config.quantiles)

    losses = objective(
        model(history, mask, future, site),
        targets,
        labels,
        daylight_mask=daylight,
    )
    losses["state_ce"].backward()
    assert any(parameter.grad is not None for parameter in model.state_head.parameters())
    for module in (model.stable_expert, model.transient_expert, model.fusion_router):
        assert all(parameter.grad is None for parameter in module.parameters())

    model.zero_grad(set_to_none=True)
    losses = objective(
        model(history, mask, future, site),
        targets,
        labels,
        daylight_mask=daylight,
    )
    losses["daylight_qcrps"].backward()
    assert any(parameter.grad is not None for parameter in model.fusion_router.parameters())
    assert all(parameter.grad is None for parameter in model.state_head.parameters())


def test_hierarchical_state_head_composes_auditable_three_state_probabilities() -> None:
    config = _config(state_head_mode="hierarchical")
    model = RaCEPV(config).eval()
    history, mask, future, site = _batch(config)

    with torch.no_grad():
        outputs = model(history, mask, future, site)

    assert outputs["ramp_presence_logits"].shape == (2, config.horizon)
    assert outputs["ramp_direction_logits"].shape == (2, config.horizon, 2)
    daylight = future[..., config.future_clear_sky_index] >= config.daylight_threshold
    ramp_probability = torch.sigmoid(outputs["ramp_presence_logits"][daylight])
    direction_probability = torch.softmax(outputs["ramp_direction_logits"][daylight], dim=-1)
    torch.testing.assert_close(
        outputs["state_probs"][..., 1:][daylight].sum(dim=-1),
        ramp_probability,
    )
    torch.testing.assert_close(
        outputs["state_probs"][..., 1:][daylight],
        ramp_probability.unsqueeze(-1) * direction_probability,
    )
    torch.testing.assert_close(
        outputs["state_probs"],
        torch.softmax(outputs["state_logits"], dim=-1),
    )
    torch.testing.assert_close(outputs["router_logits"], outputs["state_logits"])


def test_hierarchical_state_loss_is_finite_without_ramp_or_direction_labels() -> None:
    presence = torch.tensor([[0.0, -1.0]], requires_grad=True)
    direction = torch.tensor([[[1.0, -1.0], [-2.0, 2.0]]], requires_grad=True)
    labels = torch.zeros((1, 2), dtype=torch.long)

    result = hierarchical_state_loss(
        presence,
        direction,
        labels,
        positive_weight=3.0,
        mask=torch.ones_like(labels, dtype=torch.bool),
    )

    assert bool(torch.isfinite(result))
    result.backward()
    assert presence.grad is not None
    assert direction.grad is not None
    assert torch.count_nonzero(direction.grad) == 0


@pytest.mark.parametrize(
    "labels",
    (
        torch.tensor([[1, 1]]),
        torch.tensor([[2, 2]]),
        torch.tensor([[1, 2]]),
        torch.tensor([[-100, -100]]),
    ),
)
def test_hierarchical_state_loss_handles_ramp_direction_and_missing_edge_batches(
    labels: torch.Tensor,
) -> None:
    presence = torch.tensor([[0.0, 0.5]], requires_grad=True)
    direction = torch.tensor([[[1.0, -1.0], [-2.0, 2.0]]], requires_grad=True)

    result = hierarchical_state_loss(
        presence,
        direction,
        labels,
        positive_weight=2.0,
        mask=torch.ones_like(labels, dtype=torch.bool),
    )

    assert bool(torch.isfinite(result))
    result.backward()
    assert presence.grad is not None
    assert direction.grad is not None
    if bool((labels == -100).all()):
        torch.testing.assert_close(result, torch.tensor(0.0))
        assert torch.count_nonzero(presence.grad) == 0
        assert torch.count_nonzero(direction.grad) == 0


def test_ramp_auxiliary_uses_only_daylight_true_ramp_medians() -> None:
    predictions = torch.tensor([[[0.8, 1.0, 1.1], [0.5, 0.7, 0.9], [0.7, 0.9, 1.0]]])
    targets = torch.tensor([[0.5, 0.5, 0.5]])
    labels = torch.tensor([[0, 1, 2]])
    daylight = torch.tensor([[True, True, False]])

    loss = ramp_median_smooth_l1_loss(
        predictions,
        targets,
        labels,
        median_index=1,
        daylight_mask=daylight,
        beta=0.1,
    )
    changed_outside_scope = predictions.clone()
    changed_outside_scope[:, 0, 1] = 100.0
    changed_outside_scope[:, 2, 1] = -100.0
    unchanged = ramp_median_smooth_l1_loss(
        changed_outside_scope,
        targets,
        labels,
        median_index=1,
        daylight_mask=daylight,
        beta=0.1,
    )

    torch.testing.assert_close(loss, torch.tensor(0.15))
    torch.testing.assert_close(unchanged, loss)


def test_direction_expert_ramp_auxiliary_only_updates_matching_specialists() -> None:
    experts = torch.tensor(
        [
            [
                [[0.3, 0.5, 0.7], [0.2, 0.4, 0.6], [0.4, 0.6, 0.8]],
                [[0.3, 0.5, 0.7], [0.5, 0.7, 0.9], [0.1, 0.2, 0.3]],
                [[0.3, 0.5, 0.7], [0.8, 0.9, 1.0], [0.1, 0.2, 0.4]],
            ]
        ],
        requires_grad=True,
    )
    targets = torch.tensor([[0.5, 0.5, 0.5]])
    labels = torch.tensor([[0, 1, 2]])

    loss = ramp_expert_median_smooth_l1_loss(
        experts,
        targets,
        labels,
        median_index=1,
        daylight_mask=torch.tensor([[True, True, True]]),
        beta=0.1,
    )
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(0.20))
    assert experts.grad is not None
    expected_nonzero = torch.zeros_like(experts, dtype=torch.bool)
    expected_nonzero[0, 1, 1, 1] = True
    expected_nonzero[0, 2, 2, 1] = True
    assert torch.equal(experts.grad != 0, expected_nonzero)


def test_direction_expert_objective_does_not_read_fused_median() -> None:
    quantiles = (0.25, 0.5, 0.75)
    experts = torch.tensor([[[[0.2, 0.4, 0.6], [0.4, 0.7, 0.9], [0.1, 0.2, 0.3]]]])
    outputs = {
        "quantiles": torch.tensor([[[0.1, 0.9, 1.0]]]),
        "expert_quantiles": experts,
        "state_logits": torch.log(torch.tensor([[[0.1, 0.8, 0.1]]])),
    }
    objective = RaCEPVLoss(
        quantiles,
        specialist_weight=0.15,
        ramp_auxiliary_weight=0.05,
        ramp_auxiliary_scope="direction_experts",
    )
    baseline = objective(
        outputs,
        torch.tensor([[0.5]]),
        torch.tensor([[1]]),
        daylight_mask=torch.tensor([[True]]),
    )
    changed = objective(
        {**outputs, "quantiles": torch.tensor([[[0.0, 0.0, 0.0]]])},
        torch.tensor([[0.5]]),
        torch.tensor([[1]]),
        daylight_mask=torch.tensor([[True]]),
    )

    torch.testing.assert_close(baseline["ramp_auxiliary"], torch.tensor(0.15))
    torch.testing.assert_close(changed["ramp_auxiliary"], baseline["ramp_auxiliary"])


def test_hierarchical_objective_separates_forecast_and_state_losses() -> None:
    quantiles = (0.25, 0.5, 0.75)
    outputs = {
        "quantiles": torch.tensor([[[0.3, 0.5, 0.7], [0.4, 0.7, 0.9]]]),
        "expert_quantiles": torch.tensor(
            [
                [
                    [[0.3, 0.5, 0.7], [0.2, 0.4, 0.6], [0.4, 0.6, 0.8]],
                    [[0.3, 0.5, 0.7], [0.4, 0.7, 0.9], [0.5, 0.8, 1.0]],
                ]
            ]
        ),
        "state_logits": torch.log(torch.tensor([[[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]])),
        "ramp_presence_logits": torch.tensor([[-1.0, 1.0]]),
        "ramp_direction_logits": torch.tensor([[[0.0, 0.0], [2.0, -2.0]]]),
    }
    targets = torch.tensor([[0.5, 0.5]])
    labels = torch.tensor([[0, 1]])
    daylight = torch.ones_like(labels, dtype=torch.bool)
    objective = RaCEPVLoss(
        quantiles,
        specialist_weight=0.15,
        state_head_mode="hierarchical",
        ramp_positive_weight=2.0,
        ramp_auxiliary_weight=0.05,
        ramp_auxiliary_beta=0.1,
    )

    result = objective(outputs, targets, labels, daylight_mask=daylight)

    torch.testing.assert_close(
        result["forecast_loss"],
        result["daylight_qcrps"]
        + 0.15 * result["specialist_qcrps"]
        + 0.05 * result["ramp_auxiliary"],
    )
    torch.testing.assert_close(result["loss"], result["forecast_loss"] + result["state_loss"])
    torch.testing.assert_close(
        result["state_loss"],
        hierarchical_state_loss(
            outputs["ramp_presence_logits"],
            outputs["ramp_direction_logits"],
            labels,
            positive_weight=2.0,
            mask=daylight,
        ),
    )


def test_zero_ramp_auxiliary_weight_exactly_recovers_forecast_objective() -> None:
    quantiles = (0.25, 0.5, 0.75)
    outputs = {
        "quantiles": torch.tensor([[[0.3, 0.5, 0.7], [0.4, 0.7, 0.9]]]),
        "expert_quantiles": torch.tensor(
            [
                [
                    [[0.3, 0.5, 0.7], [0.2, 0.4, 0.6], [0.4, 0.6, 0.8]],
                    [[0.3, 0.5, 0.7], [0.4, 0.7, 0.9], [0.5, 0.8, 1.0]],
                ]
            ]
        ),
        "state_logits": torch.log(torch.tensor([[[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]]])),
    }
    targets = torch.tensor([[0.5, 0.5]])
    labels = torch.tensor([[0, 1]])
    objective = RaCEPVLoss(
        quantiles,
        specialist_weight=0.15,
        ramp_auxiliary_weight=0.0,
    )

    result = objective(
        outputs,
        targets,
        labels,
        daylight_mask=torch.ones_like(labels, dtype=torch.bool),
    )

    torch.testing.assert_close(
        result["forecast_loss"],
        result["daylight_qcrps"] + 0.15 * result["specialist_qcrps"],
        rtol=0.0,
        atol=0.0,
    )


def _weather_batch(config: RaCEPVConfig, batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20260826)
    values = torch.randn(
        batch_size,
        config.horizon,
        config.future_weather_features,
        generator=generator,
    )
    mask = torch.ones_like(values)
    mask[:, 0, :] = 0.0
    return values, mask


@pytest.mark.parametrize("mode", ["experts", "experts_fusion"])
def test_weather_modes_require_registered_feature_count(mode: str) -> None:
    with pytest.raises(ValueError, match="future_weather_features.*7"):
        _config(forecast_weather_mode=mode, future_weather_features=6)


def test_masked_future_weather_encoder_returns_exact_zero_for_all_missing() -> None:
    encoder = model_components.MaskedFutureWeatherEncoder(7, 16).eval()
    values = torch.randn(2, 8, 7)
    mask = torch.zeros_like(values)

    encoded = encoder(values, mask)

    assert encoded.shape == (2, 8, 16)
    assert torch.count_nonzero(encoded) == 0


def test_masked_future_weather_encoder_ignores_masked_numeric_values() -> None:
    encoder = model_components.MaskedFutureWeatherEncoder(7, 16).eval()
    values = torch.randn(2, 8, 7)
    mask = torch.ones_like(values)
    mask[:, :, 3] = 0.0
    changed = values.clone()
    changed[:, :, 3] = 1.0e6

    original = encoder(values, mask)
    altered = encoder(changed, mask)

    torch.testing.assert_close(original, altered, rtol=0.0, atol=0.0)


def test_masked_future_weather_encoder_rejects_non_binary_mask() -> None:
    encoder = model_components.MaskedFutureWeatherEncoder(7, 16)
    values = torch.randn(2, 8, 7)
    mask = torch.ones_like(values)
    mask[0, 0, 0] = 0.5

    with pytest.raises(ValueError, match="binary"):
        encoder(values, mask)


def test_weather_cells_have_identical_parameter_count() -> None:
    counts = []
    for mode in ("off", "experts", "experts_fusion"):
        model = RaCEPV(
            _config(forecast_weather_mode=mode, future_weather_features=7)
        )
        counts.append(sum(parameter.numel() for parameter in model.parameters()))

    assert len(set(counts)) == 1


def test_w00_outputs_are_invariant_to_weather() -> None:
    config = _config(forecast_weather_mode="off", future_weather_features=7)
    model = RaCEPV(config).eval()
    history, history_mask, future, site = _batch(config)
    weather, weather_mask = _weather_batch(config)

    with torch.no_grad():
        original = model(history, history_mask, future, site, weather, weather_mask)
        altered = model(history, history_mask, future, site, weather + 20.0, weather_mask)

    assert original.keys() == altered.keys()
    for name in original:
        torch.testing.assert_close(original[name], altered[name], rtol=0.0, atol=0.0)


def test_transient_isolated_weather_changes_only_transient_forecast_path() -> None:
    config = _config(
        forecast_weather_mode="transient_isolated",
        future_weather_features=7,
    )
    model = RaCEPV(config).eval()
    history, history_mask, future, site = _batch(config)
    weather, weather_mask = _weather_batch(config)

    with torch.no_grad():
        original = model(history, history_mask, future, site, weather, weather_mask)
        altered = model(history, history_mask, future, site, weather + 20.0, weather_mask)

    torch.testing.assert_close(
        original["expert_quantiles"][:, :, 0],
        altered["expert_quantiles"][:, :, 0],
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.allclose(
        original["expert_quantiles"][:, :, 1:],
        altered["expert_quantiles"][:, :, 1:],
    )
    torch.testing.assert_close(
        original["fusion_logits"], altered["fusion_logits"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        original["gate_weights"], altered["gate_weights"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        original["state_logits"], altered["state_logits"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        original["state_probs"], altered["state_probs"], rtol=0.0, atol=0.0
    )
    assert not torch.allclose(original["quantiles"], altered["quantiles"])


def test_transient_isolated_weather_access_metadata_is_explicit() -> None:
    metadata = RaCEPV(
        _config(
            forecast_weather_mode="transient_isolated",
            future_weather_features=7,
        )
    ).model_metadata

    assert metadata["weather_direct_expert_access"] is True
    assert metadata["weather_direct_stable_expert_access"] is False
    assert metadata["weather_direct_transient_expert_access"] is True
    assert metadata["weather_direct_fusion_access"] is False
    assert metadata["weather_indirect_fusion_access"] is False
    assert (
        metadata["fusion_representation_semantics"]
        == "weather_free_same_run_expert_representations"
    )
    assert metadata["state_head_weather_access"] is False


def test_transient_isolated_auxiliary_gate_is_also_weather_invariant() -> None:
    config = _config(
        forecast_weather_mode="transient_isolated",
        future_weather_features=7,
        enable_auxiliary_gate_path=True,
    )
    model = RaCEPV(config).eval()
    history, history_mask, future, site = _batch(config)
    weather, weather_mask = _weather_batch(config)

    with torch.no_grad():
        original = model(history, history_mask, future, site, weather, weather_mask)
        altered = model(history, history_mask, future, site, weather + 20.0, weather_mask)

    torch.testing.assert_close(
        original["_auxiliary_gate_weights"],
        altered["_auxiliary_gate_weights"],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("mode", ["experts", "experts_fusion"])
@pytest.mark.parametrize("state_mode", ["expert_detached", "independent_raw"])
def test_weather_changes_forecasts_but_never_state_outputs(
    mode: str,
    state_mode: str,
) -> None:
    config = _config(
        forecast_weather_mode=mode,
        future_weather_features=7,
        state_representation_mode=state_mode,
    )
    model = RaCEPV(config).eval()
    history, history_mask, future, site = _batch(config)
    weather, weather_mask = _weather_batch(config)

    with torch.no_grad():
        original = model(history, history_mask, future, site, weather, weather_mask)
        altered = model(history, history_mask, future, site, weather + 20.0, weather_mask)

    assert not torch.allclose(original["expert_quantiles"], altered["expert_quantiles"])
    torch.testing.assert_close(
        original["state_logits"], altered["state_logits"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        original["state_probs"], altered["state_probs"], rtol=0.0, atol=0.0
    )


def test_state_loss_never_updates_weather_encoder() -> None:
    config = _config(forecast_weather_mode="experts_fusion", future_weather_features=7)
    model = RaCEPV(config).train()
    history, history_mask, future, site = _batch(config)
    weather, weather_mask = _weather_batch(config)

    outputs = model(history, history_mask, future, site, weather, weather_mask)
    labels = torch.zeros(outputs["state_logits"].shape[:2], dtype=torch.long)
    torch.nn.functional.cross_entropy(outputs["state_logits"].reshape(-1, 3), labels.reshape(-1)).backward()

    assert all(parameter.grad is None for parameter in model.weather_encoder.parameters())


@pytest.mark.parametrize("mode", ["experts", "experts_fusion"])
def test_forecast_loss_updates_weather_encoder(mode: str) -> None:
    config = _config(forecast_weather_mode=mode, future_weather_features=7)
    model = RaCEPV(config).train()
    history, history_mask, future, site = _batch(config)
    weather, weather_mask = _weather_batch(config)

    outputs = model(history, history_mask, future, site, weather, weather_mask)
    outputs["quantiles"].sum().backward()

    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.weather_encoder.parameters()
    )


@pytest.mark.parametrize(
    ("mode", "expects_direct_gradient"),
    [("experts", False), ("experts_fusion", True)],
)
def test_only_w11_has_direct_weather_to_fusion_gradient(
    mode: str,
    expects_direct_gradient: bool,
) -> None:
    config = _config(forecast_weather_mode=mode, future_weather_features=7)
    model = RaCEPV(config).train()
    history, history_mask, future, site = _batch(config)
    weather, weather_mask = _weather_batch(config)

    outputs = model(history, history_mask, future, site, weather, weather_mask)
    outputs["fusion_logits"].sum().backward()

    assert model.fusion_router.weather_logits is not None
    gradient = model.fusion_router.weather_logits.weight.grad
    has_direct_gradient = gradient is not None and bool(torch.count_nonzero(gradient))
    assert has_direct_gradient is expects_direct_gradient
    state_parameters = tuple(inspect.signature(model.state_head.forward).parameters)
    assert "future_weather" not in state_parameters
    assert "weather_context" not in state_parameters
