"""Synthetic identity/entry checks; no optimizer step or real dataset loading."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from test_reliability_a import _batch, _config

from racepv.cli import train
from racepv.models import RaCEPV
from racepv.reliability_a_loss import ReliabilityAObjective
from racepv.reliability_experiment import (
    make_reliability_components,
    require_reliability_recovery_binding,
    resolve_reliability_request,
)
from racepv.training import load_racepv_checkpoint


def _args(cell="A", *extra):
    return train.build_parser().parse_args(
        [
            "--train",
            "never-read-train.json",
            "--validation",
            "never-read-validation.json",
            "--validation-only",
            "--reliability-cell",
            cell,
            "--expected-prepared-schema-version",
            "3",
            "--expected-causality-profile",
            "as-of-origin-v1",
            "--seed",
            "13",
            *extra,
        ]
    )


def test_pair_same_initial_weights_outputs_and_post_construction_rng():
    config = _config(dropout=0.15)
    torch.manual_seed(13)
    control, control_loss = make_reliability_components(config, "control")
    control_rng = torch.get_rng_state().clone()
    torch.manual_seed(13)
    candidate, candidate_loss = make_reliability_components(config, "A")
    candidate_rng = torch.get_rng_state().clone()
    assert torch.equal(control_rng, candidate_rng)
    assert type(control_loss) is type(candidate_loss) is ReliabilityAObjective
    control_state = control.state_dict()
    for name, value in candidate.state_dict().items():
        if name == "fusion_router.reliability_logits.weight":
            assert torch.count_nonzero(value) == 0
        else:
            name = name.replace("fusion_router.base_router.", "fusion_router.")
            torch.testing.assert_close(value, control_state[name], rtol=0, atol=0)
    assert (
        sum(p.numel() for p in candidate.parameters())
        - sum(p.numel() for p in control.parameters())
        == 15
    )
    inputs = _batch(config)
    torch.set_rng_state(control_rng)
    expected = control(*inputs)
    torch.set_rng_state(candidate_rng)
    actual = candidate(*inputs)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert control.model_metadata["model_revision"] != candidate.model_metadata["model_revision"]
    assert (
        control.model_metadata["objective_version"] == candidate.model_metadata["objective_version"]
    )
    assert control.model_metadata["reliability_extra_parameters"] == 0
    assert not any("reliability_logits" in key for key in control_state)


@pytest.mark.parametrize("cell", ["control", "A"])
def test_selected_cli_objective_never_constructs_legacy_loss(cell, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("legacy specialist objective constructed")

    monkeypatch.setattr(train, "RaCEPVLoss", forbidden)
    model, objective = train._build_reliability_components(_config(), cell)
    result = objective(
        model(*_batch(model.config)),
        torch.full((2, 8), 0.4),
        torch.zeros(2, 8, dtype=torch.long),
    )
    assert torch.isfinite(result["loss"])
    assert not any("specialist" in key for key in result)


@pytest.mark.parametrize("cell", ["control", "A"])
def test_request_pins_candidate_protocol(cell):
    value = resolve_reliability_request(_args(cell))
    assert value["reliability_cell"] == cell
    assert value["objective_version"] == ReliabilityAObjective.objective_id
    assert value["prepared_schema_version"] == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("validation_only", False),
        ("calibration", "forbidden.npz"),
        ("test", "forbidden.npz"),
        ("expected_prepared_schema_version", 2),
        ("expected_causality_profile", "legacy"),
        ("allow_development_data", True),
        ("specialist_loss_weight", 0.0),
        ("state_loss_weight", 0.5),
        ("backbone", "auto"),
        ("backbone", "tcn"),
        ("prediction_state_coupling", "state_as_gate"),
        ("diagnostic_ablation", "specialist_zero"),
        ("adjustment_cell", "F00"),
        ("specialist_ramp_cell", "S00"),
        ("ramp_refinement_cell", "G00"),
        ("independent_state_cell", "D00"),
        ("state_boundary_cell", "E00"),
        ("tail_residual_cell", "R00"),
        ("weather_store", "forbidden.npz"),
        ("forecast_weather_mode", "experts"),
        ("frozen_weather_adapter_from", "old.ckpt"),
        ("internal_development_fraction", 0.2),
        ("active_experts", ["stable"]),
        ("no_clear_sky_prior", True),
        ("router_mode", "amplitude_only"),
        ("fusion_mode", "equal"),
        ("seed", None),
    ],
)
def test_confounds_rejected_before_preflight_or_any_data_read(field, value, monkeypatch):
    from racepv.cli import modern_baseline

    args = _args()
    setattr(args, field, value)

    def forbidden(*args, **kwargs):
        raise AssertionError("input preflight reached before rejecting candidate request")

    monkeypatch.setattr(modern_baseline, "_preflight_prepared_inputs", forbidden)
    with pytest.raises(ValueError):
        train.run_training(args)


def test_valid_request_reaches_existing_qualification_gate_without_bypass(monkeypatch):
    from racepv.cli import modern_baseline

    class QualificationPending(ValueError):
        pass

    def pending(args):
        assert args.expected_prepared_schema_version == 3
        raise QualificationPending("source qualification pending")

    monkeypatch.setattr(modern_baseline, "_preflight_prepared_inputs", pending)
    with pytest.raises(QualificationPending):
        train.run_training(_args())


def test_legacy_request_remains_unselected():
    args = train.build_parser().parse_args(["--train", "x", "--validation", "y"])
    assert resolve_reliability_request(args) is None


@pytest.mark.parametrize("cell", ["control", "A"])
@pytest.mark.parametrize("location", ["metadata", "model_metadata"])
def test_checkpoint_and_export_round_trip(cell, location, tmp_path):
    model, _ = make_reliability_components(_config(), cell)
    path = tmp_path / "synthetic.pt"
    torch.save({"model_state_dict": model.state_dict(), location: model.model_metadata}, path)
    restored, _ = make_reliability_components(_config(), cell)
    load_racepv_checkpoint(restored, path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "source,target",
    [
        ("control", "A"),
        ("A", "control"),
        ("legacy", "control"),
        ("legacy", "A"),
        ("A", "legacy"),
        ("control", "legacy"),
    ],
)
def test_checkpoint_cross_identity_rejected_before_weights(source, target, tmp_path, monkeypatch):
    def build(cell):
        return (
            RaCEPV(_config())
            if cell == "legacy"
            else make_reliability_components(_config(), cell)[0]
        )

    old, new = build(source), build(target)
    path = tmp_path / "synthetic.pt"
    torch.save({"model_state_dict": old.state_dict(), "metadata": old.model_metadata}, path)

    def forbidden(*args, **kwargs):
        raise AssertionError("weights touched before rejecting identity")

    monkeypatch.setattr(new, "load_state_dict", forbidden)
    with pytest.raises(RuntimeError):
        load_racepv_checkpoint(new, path)


@pytest.mark.parametrize("field", ["model_revision", "objective_version", "architecture_version"])
def test_candidate_checkpoint_identity_is_required_and_exact(field, tmp_path, monkeypatch):
    model, _ = make_reliability_components(_config(), "A")
    metadata = copy.deepcopy(model.model_metadata)
    metadata.pop(field)
    path = tmp_path / "synthetic.pt"
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, path)
    with pytest.raises(RuntimeError):
        load_racepv_checkpoint(model, path)


def test_conflicting_checkpoint_metadata_rejected(tmp_path):
    model, _ = make_reliability_components(_config(), "A")
    other = dict(model.model_metadata, objective_version="wrong")
    path = tmp_path / "synthetic.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": model.model_metadata,
            "model_metadata": other,
        },
        path,
    )
    with pytest.raises(RuntimeError):
        load_racepv_checkpoint(model, path)


@pytest.mark.parametrize("mutation", [None, "legacy", "cell", "objective", "seed"])
def test_recovery_uses_verified_frozen_experiment_identity(mutation, monkeypatch):
    from racepv.training import recovery

    protocol = resolve_reliability_request(_args())
    recorded = copy.deepcopy(protocol)
    if mutation == "cell":
        recorded["reliability_cell"] = "control"
    if mutation == "objective":
        recorded["objective_version"] = "legacy-loss"
    contract = {} if mutation == "legacy" else {"reliability_experiment": recorded}
    binding = SimpleNamespace(
        seed=99 if mutation == "seed" else 13,
        raw={"resolved_training_contract": {"path": "inputs/training.json"}},
    )
    controller = SimpleNamespace(request=SimpleNamespace(binding=binding))
    calls = []

    def pinned(actual):
        assert actual is binding
        calls.append(True)
        return {"inputs/training.json": SimpleNamespace(data=json.dumps(contract).encode())}

    monkeypatch.setattr(recovery, "read_pinned_binding_inputs", pinned)
    if mutation is None:
        require_reliability_recovery_binding(controller, protocol, seed=13)
        assert calls == [True]
    else:
        with pytest.raises(ValueError):
            require_reliability_recovery_binding(controller, protocol, seed=13)


def test_explicit_legacy_coupling_treatment_is_not_a_candidate_alias():
    args = _args()
    args.prediction_state_coupling = "decoupled"
    with pytest.raises(ValueError):
        resolve_reliability_request(args)
