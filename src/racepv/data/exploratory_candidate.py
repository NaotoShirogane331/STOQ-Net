"""Strict nonformal train/validation admission; no formal qualification claims."""

from dataclasses import replace

from racepv.config import ExperimentConfig
from racepv.data import scoped_consumer
from racepv.data.prepared_contract import canonical_json_sha256
from racepv.data.scoped_dataset import ScopedCandidateDataset
from racepv.data.scoped_scientific_view import (
    require_train_validation_compatibility,
    scientific_view_from_metadata,
)

NONFORMAL = dict(
    formal_ready=False, source_interpretation_verified=False, registered_evidence_eligible=False
)


def default_experiment():
    config = ExperimentConfig()
    return replace(
        config,
        training=replace(config.training, max_epochs=8, patience=2, specialist_loss_weight=0.0),
    )


def validate_budget(experiment, *, synthetic_test=False):
    experiment.validate()
    training = experiment.training
    if (
        type(training.max_epochs) is not int
        or not 1 <= training.max_epochs <= 8
        or type(training.patience) is not int
        or training.patience != 2
        or type(training.num_workers) is not int
        or training.num_workers != 0
        or type(training.seed) is not int
        or training.seed not in (13, 29)
        or type(training.batch_size) is not int
        or training.batch_size < 1
        or training.state_loss_weight != 1.0
    ):
        raise ValueError("exploratory fixed training budget differs")
    if not synthetic_test and (training.batch_size != 64 or training.mixed_precision is not True):
        raise ValueError("research execution requires batch64 and mixed precision")


def inspect_inputs(args, *, experiment=None, synthetic_test=False):
    """Metadata only: every leaf must share the explicitly admitted evidence domain."""
    from racepv.cli.scoped_candidate import candidate_model_config_from_signature

    experiment = default_experiment() if experiment is None else experiment
    validate_budget(experiment, synthetic_test=synthetic_test)
    report = scoped_consumer.inspect_scoped_consumer_inputs(args)
    evidence = None
    views = {}
    for split, entry in report["inputs"].items():
        metadata = entry["metadata"]
        for leaf in metadata["leaves"]:
            current = leaf["schema"]["source_time_contract"]["evidence"]
            domain = "synthetic_fixture" if synthetic_test else "research_assumption"
            if current["kind"] != domain:
                raise ValueError("exploratory source evidence domain differs")
            if evidence is not None and current != evidence:
                raise ValueError("exploratory assumptions differ across leaves")
            evidence = current
        views[split] = scientific_view_from_metadata(
            metadata, selected_split=split, expected_metadata_sha256=canonical_json_sha256(metadata)
        )
        candidate_model_config_from_signature(
            views[split].feature_signature,
            experiment,
            backbone="gru" if synthetic_test else "mamba",
        )
    require_train_validation_compatibility(views["train"], views["validation"])
    return dict(
        report, evidence=evidence, feature_signature=views["train"].feature_signature, **NONFORMAL
    )


def load_inputs(args, expected_report, *, experiment=None, synthetic_test=False):
    """Verify the full scoped population, retaining observed/target masks lazily."""
    actual = inspect_inputs(args, experiment=experiment, synthetic_test=synthetic_test)
    if canonical_json_sha256(actual) != canonical_json_sha256(expected_report):
        raise ValueError("exploratory preflight identity drift")
    preflight = scoped_consumer.inspect_scoped_consumer_inputs(args)
    loaded = scoped_consumer.load_scoped_engineering_inputs(
        args, expected_preflight_sha256=canonical_json_sha256(preflight)
    )
    views, datasets = {}, {}
    for split, store in loaded["stores"].items():
        metadata = store["metadata"]
        views[split] = scientific_view_from_metadata(
            metadata, selected_split=split, expected_metadata_sha256=canonical_json_sha256(metadata)
        )
        datasets[split] = ScopedCandidateDataset(store, views[split])
    require_train_validation_compatibility(views["train"], views["validation"])
    return dict(loaded, views=views, datasets=datasets, **NONFORMAL)
