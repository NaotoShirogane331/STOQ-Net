"""Synthetic interface checks: no observations, checkpoints or scoring."""

import importlib.util
from dataclasses import asdict, fields
from pathlib import Path

import pytest
import torch

from racepv.models.config import RaCEPVConfig

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "stoq_minimal.py"


def example():
    # A missing demo is the intended first RED, rather than an import error.
    assert EXAMPLE.is_file(), "the synthetic STOQ-Net example is not implemented"
    spec = importlib.util.spec_from_file_location("stoq_minimal", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_profile_constructs_complete_historical_mamba_config():
    config = example().load_model_config(ROOT / "configs" / "stoq-paper-model.json")
    assert set(asdict(config)) == {field.name for field in fields(RaCEPVConfig)}
    assert (config.history_length, config.horizon) == (192, 24)
    assert (config.history_features, config.future_features, config.site_features) == (14, 9, 6)
    assert config.backbone == "mamba"
    assert config.quantiles == (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    assert config.harmonic_periods_hours == (24, 12, 8, 6)
    assert config.active_experts == ("stable", "ramp_up", "ramp_down")
    assert config.clear_sky_history_feature_indices == (2, 7)
    assert config.clear_sky_future_feature_indices == ()
    assert (config.fusion_mode, config.router_mode) == ("learned", "contextual")


def test_default_demo_is_explicit_gru_and_checks_quantile_contract():
    cuda_was_initialized = torch.cuda.is_initialized()
    output, summary = example().run_demo()
    quantiles = output["quantiles"]
    assert quantiles.device.type == "cpu"
    assert quantiles.shape == (2, 24, 7)
    assert torch.isfinite(quantiles).all()
    assert (quantiles.diff(dim=-1) >= 0).all()
    assert ((quantiles >= 0) & (quantiles <= 1.05)).all()
    assert torch.count_nonzero(quantiles[:, 0]) == 0
    assert torch.count_nonzero(quantiles[:, 1:]) > 0
    assert summary["requested_backbone"] == "gru"
    assert summary["actual_backbone"] == "causal_conv_gru"
    assert summary["model_revision"] == "racepv.research-single-head.v1"
    assert summary["model_class"] == "SingleHeadRaCEPV"
    assert summary["mode"] == "random_initialization_synthetic_interface_demo"
    assert torch.cuda.is_initialized() == cuda_was_initialized


def test_mamba_dependency_failure_never_selects_gru(monkeypatch):
    from racepv.models import components

    module = example()
    config = module.load_model_config(ROOT / "configs" / "stoq-paper-model.json")
    monkeypatch.setattr(components, "_OfficialMamba", None)
    with pytest.raises(ImportError, match="mamba"):
        module.build_demo_model(config, "mamba")


def test_demo_rejects_implicit_auto_backend():
    module = example()
    config = module.load_model_config(ROOT / "configs" / "stoq-paper-model.json")
    with pytest.raises(ValueError, match="gru.*mamba"):
        module.build_demo_model(config, "auto")


def test_unobserved_power_uses_original_clear_sky_fallback():
    module = example()
    config = module.load_model_config(ROOT / "configs" / "stoq-paper-model.json")
    model = module.build_demo_model(config, "gru")
    model.eval()
    history, mask, future, static = module.make_synthetic_inputs(config)
    history[..., 0] = float("nan")
    mask[..., 0] = False
    with torch.inference_mode():
        q = model(history, mask, future, static)["quantiles"]
    torch.testing.assert_close(q, future[..., :1].expand(2, 24, 7))
