"""Synthetic CPU engineering checks; never open real data or checkpoints."""

import importlib.util
import inspect
import json
import os
from dataclasses import asdict

import pytest
import torch
from test_reliability_a import _batch, _config

from racepv.models.components import OrderedQuantileHead
from racepv.models.config import RaCEPVConfig


def build(name, config=None):
    from racepv.models.research_variants import build_research_candidate

    return build_research_candidate(name, config or _config())


def test_research_variant_factory_is_available():
    assert importlib.util.find_spec("racepv.models.research_variants") is not None


@pytest.mark.parametrize("name", ["M1", "M3"])
def test_forward_physics_fallback_and_strict_checkpoint_identity(name):
    torch.manual_seed(4)
    model, objective = build(name)
    args = _batch(model.config)
    output = model(*args)
    assert output["quantiles"].shape == (2, 8, 7)
    assert torch.isfinite(output["quantiles"]).all()
    assert (output["quantiles"].diff(dim=-1) >= 0).all()
    assert torch.count_nonzero(output["quantiles"][:, 0]) == 0
    missing = tuple(v.clone() for v in args)
    missing[1].fill_(False)
    missing[0].fill_(float("nan"))
    q = model(*missing)["quantiles"]
    torch.testing.assert_close(q, args[2][..., :1].expand_as(q))
    assert objective.objective_id.endswith("without-specialist.v1")
    assert model.model_metadata["reliability_extra_parameters"] == 0
    reference, _ = build("F")
    with pytest.raises(RuntimeError):
        model.load_state_dict(reference.state_dict(), strict=True)


@pytest.mark.parametrize("name", ["M1", "M3"])
def test_forecast_gradient_reaches_every_counted_parameter_and_ce_is_detached(name):
    torch.manual_seed(7)
    model, objective = build(name)
    args = _batch(model.config)
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, OrderedQuantileHead):
                layer.projection.weight.mul_(0.03)
                layer.projection.bias[0] = 0
    output = model(*args)
    target = torch.full((2, 8), 0.3)
    labels = torch.arange(16).reshape(2, 8) % 3
    losses = objective(output, target, labels, daylight_mask=args[2][..., 0] > 0)
    losses["forecast_loss"].backward()
    for key, parameter in model.named_parameters():
        if key.startswith("state_head."):
            assert parameter.grad is None, key
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), key
            assert torch.count_nonzero(parameter.grad), key
    model.zero_grad(set_to_none=True)
    output = model(*args)
    objective(output, target, labels)["state_loss"].backward()
    for key, parameter in model.named_parameters():
        assert (parameter.grad is not None) == key.startswith("state_head."), key


def test_equal_fusion_has_no_trainable_router_and_matches_own_experts():
    model, _ = build("M3")
    output = model(*_batch(model.config))
    assert not list(model.fusion_router.parameters())
    assert torch.equal(output["gate_weights"], torch.full((2, 8, 3), 1 / 3))
    torch.testing.assert_close(output["quantiles"], output["expert_quantiles"].mean(2))


@pytest.mark.parametrize("serialization", ["asdict", "json"])
def test_m3_public_config_roundtrip_strict_load_and_equal_eval(serialization):
    torch.manual_seed(917)
    original, _ = build("M3", _config(dropout=0.25))
    saved = asdict(original.config)
    assert saved["fusion_mode"] == "equal"
    if serialization == "json":
        saved = json.loads(json.dumps(saved))
        # JSON represents tuples as arrays. Restore the declared tuple schema
        # before constructing RaCEPVConfig; its original validation is unchanged.
        for key, value in asdict(original.config).items():
            if isinstance(value, tuple):
                saved[key] = tuple(saved[key])
    rebuilt, _ = build("M3", RaCEPVConfig(**saved))
    rebuilt.load_state_dict(original.state_dict(), strict=True)
    assert rebuilt.config == original.config
    assert rebuilt.model_metadata == original.model_metadata
    original.eval()
    rebuilt.eval()
    with torch.no_grad():
        expected = original(*_batch(original.config))
        actual = rebuilt(*_batch(rebuilt.config))
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)
    reference, _ = build("F", _config(dropout=0.25))
    with pytest.raises(RuntimeError):
        rebuilt.load_state_dict(reference.state_dict(), strict=True)


@pytest.mark.parametrize("name", ["F", "C", "M1"])
def test_equal_config_admission_is_specific_to_m3(name):
    with pytest.raises(ValueError, match="isolated decoupled three-expert contract"):
        build(name, _config(fusion_mode="equal"))


@pytest.mark.parametrize(
    "changes",
    [
        {"prediction_state_coupling": "joint_head"},
        {"active_experts": ("stable", "ramp_up")},
        {"router_mode": "amplitude_only"},
        {"state_head_mode": "hierarchical"},
        {"state_representation_mode": "independent_raw"},
        {"enable_auxiliary_gate_path": True},
        {"tail_residual_mode": "sparse_ramp_up"},
        {"future_weather_features": 7, "forecast_weather_mode": "experts"},
        {"use_clear_sky_prior": False},
        {"backbone": "tcn"},
        {"sampling_minutes": 30},
        {"history_length": 8},
    ],
)
def test_m3_equal_config_retains_other_f_contract_rejections(changes):
    with pytest.raises(ValueError, match="isolated decoupled three-expert contract"):
        build("M3", _config(fusion_mode="equal", **changes))


def test_single_head_capacity_determinism_and_no_stale_batch_state():
    torch.manual_seed(7)
    model, _ = build("M1")
    reference, _ = build("F")

    def count(m):
        return sum(
            p.numel()
            for n, p in m.named_parameters()
            if p.requires_grad and not n.startswith("state_head.")
        )

    assert sum(isinstance(m, OrderedQuantileHead) for m in model.modules()) == 1
    assert not any(
        "fusion_router" in n or "up_hidden" in n or "down_hidden" in n
        for n, _ in model.named_parameters()
    )
    assert abs(count(model) - count(reference)) / count(reference) <= 0.05
    torch.manual_seed(71)
    other, _ = build("M1")
    assert model.readout_width == other.readout_width
    model.eval()
    args = _batch(model.config)
    first = model(*args)["quantiles"]
    model(*(v[:1] for v in args))
    torch.testing.assert_close(model(*args)["quantiles"], first, rtol=0, atol=0)
    assert not any(m._forward_hooks for m in model.modules())
    assert model.model_metadata["diagnostic_inputs_equal_to_F"] is False


@pytest.mark.parametrize("dimension,router_width", [(8, 16), (16, 32), (64, 128)])
def test_capacity_width_is_nearest_integer_by_independent_count(dimension, router_width):
    config = _config(d_model=dimension, router_hidden_size=router_width)
    model, _ = build("M1", config)
    reference, _ = build("F", config)

    def count(m):
        return sum(
            p.numel()
            for n, p in m.named_parameters()
            if not n.startswith("state_head.") and p.requires_grad
        )

    difference = abs(count(model) - count(reference))
    step = (2 * dimension + 1) + len(config.quantiles)
    assert difference <= abs(count(model) + step - count(reference))
    if model.readout_width > 1:
        assert difference <= abs(count(model) - step - count(reference))
    metadata = model.model_metadata
    assert metadata["prediction_parameters"] == count(model)
    assert metadata["reference_prediction_parameters"] == count(reference)
    assert abs(metadata["prediction_parameter_relative_difference"]) <= 0.05


def test_factory_preserves_f_c_revisions_and_merges_m2():
    from racepv.models.research_variants import M2MergedError

    assert build("F")[0].model_metadata["model_revision"] == "racepv.reliability-gate-a.v1"
    assert build("C")[0].model_metadata["model_revision"] == "racepv.reliability-gate-a-control.v1"
    with pytest.raises(M2MergedError, match="merged"):
        build("M2")
    with pytest.raises(ValueError):
        build("arbitrary")


@pytest.mark.parametrize("name", ["F", "C"])
@pytest.mark.parametrize("training", [False, True])
def test_original_source_extraction_preserves_outputs_namespace_and_rng(
    monkeypatch, name, training
):
    from racepv.models import race_pv
    from racepv.models.experts import StablePeriodicExpert, TransientDirectionalExpert

    original_path = os.environ.get("RACEPV_ORIGINAL_EXPERTS")
    if original_path is None:
        pytest.skip("set RACEPV_ORIGINAL_EXPERTS to the read-only production experts.py")
    spec = importlib.util.spec_from_file_location("racepv.models._original_experts", original_path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    for old, new in (
        (original.StablePeriodicExpert, StablePeriodicExpert),
        (original.TransientDirectionalExpert, TransientDirectionalExpert),
    ):
        assert inspect.signature(old.forward) == inspect.signature(new.forward)
    config = _config(dropout=0.25)
    torch.manual_seed(193)
    with monkeypatch.context() as context:
        context.setattr(race_pv, "StablePeriodicExpert", original.StablePeriodicExpert)
        context.setattr(race_pv, "TransientDirectionalExpert", original.TransientDirectionalExpert)
        old, _ = build(name, config)
    old_init_rng = torch.get_rng_state().clone()
    torch.manual_seed(193)
    new, _ = build(name, config)
    assert torch.equal(old_init_rng, torch.get_rng_state())
    assert old.state_dict().keys() == new.state_dict().keys()
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    args = _batch(config)
    old.train(training)
    new.train(training)
    torch.manual_seed(311)
    expected = old(*args)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(311)
    actual = new(*args)
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)
