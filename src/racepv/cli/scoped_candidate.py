"""Shape and output bridge for the qualified no-weather A/control candidate.

Execution remains in cli.train and its existing epoch recovery controller.
"""

from dataclasses import asdict
from pathlib import Path

from racepv.data import prepared_contract as c
from racepv.data.scoped_dataset import ScopedCandidateDataset
from racepv.data.scoped_qualification import validate_qualified_output_identity
from racepv.models import RaCEPVConfig
from racepv.reliability_a_loss import ReliabilityAObjective

AUDIT_METADATA = dict(
    day_ids_semantics="utc_days_since_epoch",
    timestamps_semantics="target_interval_start_ns",
    target_local_day_ids_semantics="source_timezone_target_interval_start_YYYYMMDD",
    issue_ns_semantics="last_history_interval_end",
)


def add_candidate_qualification_arguments(parser):
    parser.add_argument("--qualification-root", type=Path)
    parser.add_argument("--qualification-name")
    parser.add_argument("--qualification-sha256")
    parser.add_argument("--producer-root", type=Path)
    parser.add_argument("--producer-review-sha256")


def candidate_model_config(dataset, experiment, *, backbone):
    if not isinstance(dataset, ScopedCandidateDataset):
        raise TypeError("candidate requires the verified schema3 dataset adapter")
    if backbone not in ("gru", "mamba"):
        raise ValueError("candidate supports only its current GRU/Mamba backbone")
    signature = dataset.feature_signature  # Derived consistently from every leaf.
    return candidate_model_config_from_signature(signature, experiment, backbone=backbone)


def candidate_model_config_from_signature(signature, experiment, *, backbone):
    """Derive the identical model shape during metadata-only recovery admission."""
    if backbone not in ("gru", "mamba"):
        raise ValueError("candidate supports only its current GRU/Mamba backbone")
    geometry = signature["geometry"]
    if (
        experiment.data.history_length,
        experiment.data.horizon,
        experiment.data.stride,
        experiment.data.frequency_minutes * 60_000_000_000,
    ) != (
        geometry["lookback"],
        geometry["horizon"],
        geometry["stride"],
        signature["clock"]["sampling_interval_ns"],
    ):
        raise ValueError("candidate configured geometry differs from verified inputs")
    history, future, site = (
        signature[name]
        for name in ("history_columns", "future_known_columns", "site_metadata_names")
    )
    return RaCEPVConfig(
        history_features=len(history),
        future_features=len(future),
        site_features=len(site),
        history_length=geometry["lookback"],
        horizon=geometry["horizon"],
        sampling_minutes=signature["clock"]["sampling_interval_ns"] // 60_000_000_000,
        quantiles=experiment.quantiles,
        backbone=backbone,
        power_index=history.index("power_norm"),
        history_clear_sky_index=history.index("clear_sky_power_norm"),
        clear_sky_history_feature_indices=tuple(
            history.index(name) for name in ("clear_sky_residual", "clear_sky_index")
        ),
        future_clear_sky_index=future.index("clear_sky_power_norm"),
        future_hour_sin_index=future.index("hour_sin"),
        future_hour_cos_index=future.index("hour_cos"),
    )


def candidate_dataset_metadata(dataset, identities):
    checked = validate_qualified_output_identity(identities)
    return {
        **dataset.export_metadata,
        **AUDIT_METADATA,
        "source_timezone": dataset.feature_signature["source_clock"]["timezone"],
        "registered_evidence_eligible": checked["qualification"]["formal_ready"],
    }


def candidate_output_metadata(identities, *, model_config, model_metadata):
    checked = validate_qualified_output_identity(identities)
    c._exact(
        model_metadata["objective_version"],
        ReliabilityAObjective.objective_id,
        "candidate objective",
    )
    return dict(
        qualified_prepared_inputs=checked,
        model_config=asdict(model_config),
        model_metadata=dict(model_metadata),
        objective_version=ReliabilityAObjective.objective_id,
        **AUDIT_METADATA,
    )
