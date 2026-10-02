"""Matched candidate A/control identities for the corrected-data experiment.

This selects components and validates CLI intent. Source qualification remains
owned by the existing prepared-input gate; this module grants no data admission.
"""

from racepv.models import RaCEPV, RaCEPVConfig
from racepv.models.reliability_a import MODEL_REVISION, ReliabilityGateARaCEPV
from racepv.reliability_a_loss import ReliabilityAObjective

CONTROL_REVISION = "racepv.reliability-gate-a-control.v1"
PAIR_PROTOCOL = "racepv.reliability-gate-a-matched-control.v1"


class ReliabilityControlRaCEPV(ReliabilityGateARaCEPV):
    """Same base model and initialization stream, with no reliability correction.

    Construct the same candidate before unwrapping its base router. This preserves
    both shared parameter initialization and the post-construction RNG state used
    by dropout and shuffled loaders. The returned control has no extra weights,
    reliability feature computation, or specialist objective computation.
    """

    def __init__(self, config: RaCEPVConfig) -> None:
        super().__init__(config)
        self.fusion_router = self.fusion_router.base_router

    @property
    def model_metadata(self) -> dict[str, object]:
        metadata = RaCEPV.model_metadata.fget(self)
        metadata.update(
            base_architecture_version=metadata["architecture_version"],
            architecture_version=CONTROL_REVISION,
            model_revision=CONTROL_REVISION,
            objective_version=ReliabilityAObjective.objective_id,
            candidate_only=True,
            reliability_extra_parameters=0,
            reliability_features=[],
            specialist_auxiliary_computation="absent_from_matching_objective",
        )
        return metadata


def make_reliability_components(config: RaCEPVConfig, cell: str):
    if cell not in ("control", "A"):
        raise ValueError("reliability cell must be control or A")
    model_type = ReliabilityControlRaCEPV if cell == "control" else ReliabilityGateARaCEPV
    model = model_type(config)
    return model, model.make_objective()


def resolve_reliability_request(args):
    """Reject confounds before existing preflight can open any input metadata."""
    cell = getattr(args, "reliability_cell", None)
    if cell is None:
        return None
    if cell not in ("control", "A"):
        raise ValueError("reliability cell must be control or A")
    if (
        args.validation_only is not True
        or args.calibration is not None
        or args.test is not None
        or type(args.expected_prepared_schema_version) is not int
        or args.expected_prepared_schema_version != 3
        or args.expected_causality_profile != "as-of-origin-v1"
        or args.allow_development_data
    ):
        raise ValueError("reliability pair requires explicit corrected schema3 train/validation")
    conflicting = (
        "specialist_loss_weight",
        "diagnostic_ablation",
        "adjustment_cell",
        "specialist_ramp_cell",
        "ramp_refinement_cell",
        "independent_state_cell",
        "state_boundary_cell",
        "tail_residual_cell",
        "weather_store",
        "frozen_weather_adapter_from",
        "internal_development_fraction",
    )
    if any(getattr(args, key, None) is not None for key in conflicting):
        raise ValueError(
            "reliability pair forbids other treatments and specialist weight overrides"
        )
    if (
        args.state_loss_weight not in (None, 1.0)
        or args.prediction_state_coupling is not None
        or args.backbone not in ("mamba", "gru")
        or args.forecast_weather_mode != "off"
        or args.no_clear_sky_prior
        or args.router_mode != "contextual"
        or args.fusion_mode != "learned"
        or tuple(args.active_experts) != ("stable", "ramp_up", "ramp_down")
        or type(args.seed) is not int
        or args.seed < 0
    ):
        raise ValueError(
            "reliability pair requires its fixed three-expert contract and explicit seed"
        )
    return {
        "reliability_cell": cell,
        "experiment_protocol": PAIR_PROTOCOL,
        "model_revision": CONTROL_REVISION if cell == "control" else MODEL_REVISION,
        "objective_version": ReliabilityAObjective.objective_id,
        "prepared_schema_version": 3,
        "causality_profile": "as-of-origin-v1",
        "specialist_auxiliary_computation": "absent_from_matching_objective",
        "selection_boundary": "validation_only",
    }


def require_reliability_recovery_binding(controller, protocol, *, seed):
    """Bind the new experiment to the existing pinned recovery input contract.

    An old reference binding cannot admit the new control even though its base
    weights have the legacy shapes. This reads only pinned JSON contracts.
    """
    if controller is None or protocol is None:
        return
    from racepv.training.recovery import parse_json_bytes, read_pinned_binding_inputs

    binding = controller.request.binding
    if binding.seed != seed:
        raise ValueError("candidate recovery seed differs from request")
    inputs = read_pinned_binding_inputs(binding)
    path = binding.raw["resolved_training_contract"]["path"]
    contract = parse_json_bytes(inputs[path].data, "candidate resolved training contract")
    if contract.get("reliability_experiment") != protocol:
        raise ValueError("candidate recovery requires its exact frozen experiment identity")
