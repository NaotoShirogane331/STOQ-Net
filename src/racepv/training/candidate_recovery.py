"""Versioned candidate admission into the existing epoch recovery executor.

The explicit synthetic domain is an engineering API only. The command runner
and train CLI always require provider_document and the sealed CUDA runtime.
"""

import json
from dataclasses import asdict
from pathlib import Path

from .recovery import (
    RecoveryIntegrityError,
    parse_json_bytes,
    read_pinned_binding_inputs,
    read_regular_file_nofollow,
    validate_relative_path,
)

_ARGUMENTS = frozenset(
    {
        "train",
        "validation",
        "qualification_name",
        "qualification_sha256",
        "producer_review_sha256",
        "prepared_train_descriptor_sha256",
        "prepared_train_read_scope_sha256",
        "prepared_validation_descriptor_sha256",
        "prepared_validation_read_scope_sha256",
        "backbone",
        "device",
        "seed",
        "reliability_cell",
    }
)
_CONTRACT = frozenset(
    {
        "schema_version",
        "purpose",
        "arguments",
        "reliability_experiment",
        "experiment",
        "model_config",
        "qualified_prepared_inputs",
        "resolved_run",
    }
)


def is_candidate(binding):
    return binding.raw.get("schema_version") == 2


def _equal(actual, expected, role):
    # JSON normalization preserves scalar types and ignores tuple/list transport differences.
    def encoded(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    if encoded(actual) != encoded(expected):
        raise RecoveryIntegrityError("candidate " + role + " differs")


def candidate_contract(binding):
    if (
        not is_candidate(binding)
        or binding.raw.get("purpose") != "racepv_candidate_epoch_recovery_binding"
    ):
        raise RecoveryIntegrityError("candidate requires its schema2 binding")
    inputs = read_pinned_binding_inputs(binding)

    def read(role):
        return parse_json_bytes(inputs[binding.raw[role]["path"]].data, role)

    contract = read("resolved_training_contract")
    if (
        frozenset(contract) != _CONTRACT
        or type(contract["schema_version"]) is not int
        or contract["schema_version"] != 2
        or contract["purpose"] != "racepv_candidate_training_contract"
    ):
        raise RecoveryIntegrityError("candidate training contract schema differs")
    return contract, read("reference_policy"), read("source_inventory"), read("feature_contract")


def candidate_arguments(binding, *, data_root, output):
    """Create only fixed parser defaults and an allowlisted set of pinned values."""
    from racepv.cli.train import build_parser
    from racepv.data import scoped_store
    from racepv.reliability_experiment import resolve_reliability_request

    contract, policy, _, _ = candidate_contract(binding)
    selected = contract["arguments"]
    if not isinstance(selected, dict) or frozenset(selected) != _ARGUMENTS:
        raise RecoveryIntegrityError("candidate arguments keys differ")
    if data_root is None:
        raise RecoveryIntegrityError("candidate data root must be explicit")
    root = scoped_store._reject_link_components(data_root)
    for field in ("train", "validation", "qualification_name"):
        validate_relative_path(selected[field])
    if (
        len({selected[field].casefold() for field in ("train", "validation", "qualification_name")})
        != 3
    ):
        raise RecoveryIntegrityError("candidate input roles alias")
    args = build_parser().parse_args(
        [
            "--train",
            str(root / selected["train"]),
            "--validation",
            str(root / selected["validation"]),
            "--validation-only",
        ]
    )
    for field, value in selected.items():
        if field not in ("train", "validation"):
            setattr(args, field, value)
    args.expected_prepared_schema_version = 3
    args.expected_causality_profile = "as-of-origin-v1"
    args.qualification_root = root
    args.producer_root = Path(__file__).resolve().parents[1]
    args.config = binding.binding_root / policy["config"]["path"]
    args.output = output
    if type(args.seed) is not int or args.seed != binding.seed or args.seed not in (13, 29):
        raise RecoveryIntegrityError("candidate seed differs or is outside pilot")
    _equal(
        resolve_reliability_request(args), contract["reliability_experiment"], "experiment protocol"
    )
    return args


def _validate_executing_sources(source):
    project = Path(__file__).resolve().parents[3]
    if (
        frozenset(source) != {"schema_version", "purpose", "files"}
        or type(source["schema_version"]) is not int
        or source["schema_version"] != 2
        or source["purpose"] != "racepv_candidate_training_sources"
    ):
        raise RecoveryIntegrityError("candidate source schema differs")
    paths = sorted(
        [p.relative_to(project).as_posix() for p in (project / "src/racepv").rglob("*.py")]
        + ["scripts/run_racepv_reference_seed.py"]
    )
    inventory = source["files"]
    if not isinstance(inventory, list) or not all(
        isinstance(item, dict) and set(item) == {"path", "sha256"} for item in inventory
    ):
        raise RecoveryIntegrityError("candidate source inventory differs")
    if [item["path"] for item in inventory] != paths:
        raise RecoveryIntegrityError(
            "candidate source inventory must cover entire executing package and runner"
        )
    from scripts.run_racepv_reference_seed import _source_digest

    for item in inventory:
        if _source_digest(project, item["path"]) != item["sha256"]:
            raise RecoveryIntegrityError("candidate executing source changed")
    # The source root is derived from this imported module, never supplied by a caller.
    import sys

    for name, module in tuple(sys.modules.items()):
        if name == "racepv" or name.startswith("racepv."):
            filename = getattr(module, "__file__", None)
            if filename is not None and not Path(filename).resolve().is_relative_to(
                project / "src/racepv"
            ):
                raise RecoveryIntegrityError("candidate imported source root differs")


def inspect_candidate_recovery(binding, args, *, expected_domain="provider_document"):
    """Metadata/source/config validation only: no model, CUDA or numerical arrays."""
    import yaml

    from racepv.cli.scoped_candidate import candidate_model_config_from_signature
    from racepv.config import _construct_config
    from racepv.data import scoped_consumer
    from racepv.data import scoped_qualification as qualification
    from racepv.data.prepared_contract import canonical_json_sha256
    from racepv.data.scoped_scientific_view import scientific_view_from_metadata

    contract, policy, source, features = candidate_contract(binding)
    if (
        expected_domain not in ("provider_document", "synthetic_fixture")
        or policy["domain"] != expected_domain
    ):
        raise RecoveryIntegrityError("candidate qualification policy domain differs")
    expected_args = candidate_arguments(
        binding, data_root=args.qualification_root, output=args.output
    )
    _equal(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(expected_args).items()},
        "requested arguments",
    )
    _validate_executing_sources(source)
    # Qualification verifies E1 and producer correspondence before any arrays/config use.
    inspection = qualification.inspect_qualified_candidate(
        args, expected_domain=expected_domain
    ).report
    identity = qualification.validate_qualified_output_identity(
        contract["qualified_prepared_inputs"]
    )
    predicted = dict(
        inspection,
        content_verified=True,
        array_semantics_verified=True,
        formal_ready=expected_domain == "provider_document",
    )
    _equal(
        qualification._qualification_identity(predicted),
        identity["qualification"],
        "qualified metadata",
    )
    preflight = scoped_consumer.inspect_scoped_consumer_inputs(args)
    for split, entry in preflight["inputs"].items():
        metadata = entry["metadata"]
        view = scientific_view_from_metadata(
            metadata, selected_split=split, expected_metadata_sha256=canonical_json_sha256(metadata)
        )
        _equal(view.feature_signature, features, "feature contract")
        # Compare the complete current descriptor projection. Local paths and the
        # two content-verification result flags are not metadata identity claims.
        # Neither the current engineering flags nor the pinned identity is changed.
        portable = {
            key: value
            for key, value in metadata.items()
            if key not in ("descriptor_path", "content_verified", "array_semantics_verified")
        }
        portable["leaves"] = [
            {key: value for key, value in leaf.items() if key not in ("descriptor_path", "arrays")}
            for leaf in metadata["leaves"]
        ]
        portable["files"] = [
            {key: value for key, value in item.items() if key != "resolved_path"}
            for item in metadata["files"]
        ]
        saved = {
            key: value
            for key, value in identity["prepared_contracts"][split]["scoped_metadata"].items()
            if key not in ("content_verified", "array_semantics_verified")
        }
        _equal(portable, saved, "complete portable " + split + " metadata")
        for field, identity_field in (
            ("descriptor_sha256", "descriptor_hashes"),
            ("aggregate_content_sha256", "content_hashes"),
        ):
            _equal(metadata[field], identity[identity_field][split], split + " input")
        for suffix, actual in (
            ("descriptor_sha256", identity["descriptor_hashes"][split]),
            ("content_sha256", identity["content_hashes"][split]),
        ):
            _equal(actual, binding.protocol_identity[split + "_" + suffix], "protocol")
    config_raw = read_regular_file_nofollow(binding.binding_root, Path(policy["config"]["path"]))
    if config_raw.sha256 != policy["config"]["sha256"]:
        raise RecoveryIntegrityError("candidate config changed")

    # Reject duplicate YAML mappings instead of silently accepting the last key.
    class StrictLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        pairs = loader.construct_pairs(node, deep=True)
        value = {}
        for key, item in pairs:
            if key in value:
                raise RecoveryIntegrityError("duplicate candidate config key")
            value[key] = item
        return value

    StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    raw_config = yaml.load(config_raw.data, Loader=StrictLoader)
    if not isinstance(raw_config, dict) or set(raw_config) - {
        "data",
        "training",
        "calibration",
        "quantiles",
        "output_dir",
        "run_name",
    }:
        raise RecoveryIntegrityError("candidate config keys differ")
    experiment = _construct_config(raw_config)
    _equal(experiment.to_dict(), contract["experiment"], "actual configuration")
    model_config = candidate_model_config_from_signature(
        features, experiment, backbone=args.backbone
    )
    _equal(asdict(model_config), contract["model_config"], "model configuration")
    training = experiment.training
    if (
        type(training.max_epochs) is not int
        or not 1 <= training.max_epochs <= 8
        or training.patience != 2
        or training.num_workers != 0
        or training.seed != binding.seed
    ):
        raise RecoveryIntegrityError("candidate pilot training budget differs")
    if expected_domain == "provider_document" and (
        args.device != "cuda"
        or args.backbone != "mamba"
        or training.mixed_precision is not True
        or training.batch_size != 64
    ):
        raise RecoveryIntegrityError("candidate formal execution contract differs")
    return contract


def validate_candidate_resolved(binding, resolved):
    contract, _, _, _ = candidate_contract(binding)
    _equal(resolved, contract["resolved_run"], "resolved training contract")


def validate_candidate_metadata(binding, metadata):
    contract, _, _, _ = candidate_contract(binding)
    if not isinstance(metadata, dict):
        raise RecoveryIntegrityError("candidate output metadata missing")
    _equal(metadata.get("seed"), binding.seed, "output seed")
    _equal(
        metadata.get("qualified_prepared_inputs"),
        contract["qualified_prepared_inputs"],
        "output qualification",
    )
    _equal(
        metadata.get("model_config", metadata.get("model")),
        contract["model_config"],
        "output model config",
    )
    for field, value in contract["reliability_experiment"].items():
        _equal(metadata.get(field), value, "output " + field)
    model_metadata = metadata.get("model_metadata", metadata)
    for field in ("model_revision", "objective_version"):
        _equal(
            model_metadata.get(field),
            contract["reliability_experiment"][field],
            "output model " + field,
        )


def validate_candidate_trainer(binding, trainer, train_loader, validation_loader):
    """Validate actual trainer identity before restore or run-directory creation."""
    from torch.utils.data import RandomSampler, SequentialSampler

    from racepv.data.scoped_dataset import ScopedCandidateDataset
    from racepv.reliability_a_loss import ReliabilityAObjective

    contract, policy, _, features = candidate_contract(binding)
    validate_candidate_metadata(binding, trainer.checkpoint_metadata)
    _equal(asdict(trainer.model.config), contract["model_config"], "trainer model config")
    for field in ("model_revision", "objective_version"):
        _equal(
            trainer.model.model_metadata[field],
            contract["reliability_experiment"][field],
            "trainer " + field,
        )
    if type(trainer.criterion) is not ReliabilityAObjective or not trainer.separate_state_optimizer:
        raise RecoveryIntegrityError("candidate trainer objective or optimizer differs")
    training = contract["experiment"]["training"]
    for name in ("max_epochs", "patience", "gradient_clip_norm"):
        _equal(getattr(trainer, name), training[name], "trainer " + name)
    if trainer.selection_metric != "daylight_qcrps":
        raise RecoveryIntegrityError("candidate checkpoint selection metric differs")
    for optimizer in (trainer.forecast_optimizer, trainer.state_optimizer):
        for group in optimizer.param_groups:
            _equal(group["lr"], training["learning_rate"], "optimizer learning rate")
            _equal(group["weight_decay"], training["weight_decay"], "optimizer weight decay")
    for split, loader, sampler_type in (
        ("train", train_loader, RandomSampler),
        ("validation", validation_loader, SequentialSampler),
    ):
        if (
            type(loader.sampler) is not sampler_type
            or loader.num_workers != 0
            or loader.drop_last is not False
        ):
            raise RecoveryIntegrityError("candidate loader contract differs")
        _equal(loader.batch_size, training["batch_size"], "loader batch size")
        dataset = loader.dataset
        if policy["domain"] == "provider_document":
            if type(dataset) is not ScopedCandidateDataset:
                raise RecoveryIntegrityError(
                    "formal candidate loader must cover the qualified full population"
                )
            _equal(dataset.feature_signature, features, "loader features")
            _equal(
                dataset.content_identity,
                contract["qualified_prepared_inputs"]["content_hashes"][split],
                "loader contents",
            )
            if trainer.device.type != "cuda" or trainer.use_amp is not True:
                raise RecoveryIntegrityError("formal candidate trainer runtime differs")


def _tensor_same(actual, expected, role):
    import torch

    if isinstance(expected, torch.Tensor):
        if (
            not isinstance(actual, torch.Tensor)
            or actual.dtype != expected.dtype
            or actual.shape != expected.shape
            or not torch.equal(actual.cpu(), expected.cpu())
        ):
            raise RecoveryIntegrityError("candidate " + role + " differs from selected checkpoint")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise RecoveryIntegrityError("candidate " + role + " mapping differs")
        for key in expected:
            _tensor_same(actual[key], expected[key], role)
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise RecoveryIntegrityError("candidate " + role + " sequence differs")
        for left, right in zip(actual, expected, strict=True):
            _tensor_same(left, right, role)
    else:
        _equal(actual, expected, role)


def validate_candidate_export_stage(binding, run_root, relative, stage, terminal_commit_sha256):
    """Validate committed candidate semantics without reopening external arrays."""
    import io

    import torch

    from .recoverable_run import read_generation_chain

    generations = read_generation_chain(run_root, binding_sha256=binding.raw_sha256)
    terminal = generations[-1]
    if (
        terminal.commit_sha256 != terminal_commit_sha256
        or terminal.state["terminal_reason"] is None
    ):
        raise RecoveryIntegrityError("candidate export terminal generation differs")
    selected = terminal.state["best_checkpoint"]
    validate_candidate_metadata(binding, selected["metadata"])

    def read(name):
        return read_regular_file_nofollow(run_root, relative / name).data

    if stage == "model":
        checkpoint = torch.load(
            io.BytesIO(read("best.ckpt")), map_location="cpu", weights_only=True
        )
        model = torch.load(io.BytesIO(read("model.pt")), map_location="cpu", weights_only=True)
        _tensor_same(checkpoint, selected, "best checkpoint")
        validate_candidate_metadata(binding, model)
        _tensor_same(model.get("model_state_dict"), selected["model_state_dict"], "model weights")
        validate_candidate_resolved(
            binding, parse_json_bytes(read("resolved_run_config.json"), "resolved run")
        )
        import yaml

        contract, _, _, _ = candidate_contract(binding)
        _equal(
            yaml.safe_load(read("resolved_config.yaml")),
            contract["experiment"],
            "export configuration",
        )
    elif stage in ("validation", "summary"):
        name = "validation_predictions.json" if stage == "validation" else "run.json"
        validate_candidate_metadata(binding, parse_json_bytes(read(name), "candidate export"))
    else:
        raise RecoveryIntegrityError("unknown candidate export stage")
