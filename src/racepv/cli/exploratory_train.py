"""Public, explicitly nonformal A/control campaign build/preflight/recovery CLI."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from racepv.data.exploratory_candidate import default_experiment, inspect_inputs
from racepv.training.exploratory_contract import create_campaign, equal, open_binding, preflight
from racepv.training.exploratory_recovery import check_run, execute


def build_parser():
    parser = argparse.ArgumentParser(
        description="Research-assumption A/control candidate; always nonformal. "
        "Build freezes exactly control/A x seeds13/29. No synthetic CLI training mode."
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    build = modes.add_parser("build", help="Metadata-only freeze of one four-job campaign.")
    build.add_argument(
        "--campaign-root",
        type=Path,
        required=True,
        help="New directory in an existing parent; cannot overwrite a campaign.",
    )
    build.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root containing both descriptors; must not overlap campaign.",
    )
    for split in ("train", "validation"):
        build.add_argument(
            "--" + split,
            type=Path,
            required=True,
            help=f"Absolute {split} schema3 descriptor inside data-root.",
        )
        for pin in ("descriptor", "read-scope"):
            build.add_argument(
                f"--prepared-{split}-{pin}-sha256",
                required=True,
                help="Exact externally supplied SHA256; no automatic pin discovery.",
            )
    build.add_argument(
        "--assumptions-sha256",
        required=True,
        help="Expected shared research-assumption evidence SHA256.",
    )
    build.add_argument("--max-epochs", type=int, default=8, help="1..8; patience remains 2.")
    build.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Frozen prepared stride (default 4); history192/horizon24 remain fixed.",
    )
    for mode, help_text in (
        ("preflight", "Metadata-only validation of current inputs, sources and frozen identity."),
        ("start", "Start the fixed cell/seed job bound by the supplied binding."),
        ("continue", "Resume the same job; verify and skip a completed job."),
        ("check-only", "Read internal committed integrity only; no external arrays or CUDA."),
    ):
        command = modes.add_parser(mode, help=help_text)
        command.add_argument(
            "--binding",
            type=Path,
            required=True,
            help="One campaign bindings/{control|A}-s{13|29}.json.",
        )
        command.add_argument(
            "--binding-sha256",
            required=True,
            help="Externally pinned exact binding file SHA256; always required.",
        )
        if mode != "check-only":
            command.add_argument(
                "--data-root",
                type=Path,
                help="Explicit root migration; portable data/evidence pins revalidated.",
            )
        if mode in ("start", "continue"):
            command.add_argument(
                "--acknowledge-research-assumptions",
                action="store_true",
                required=True,
                help="Accept unverified research-assumption identity.",
            )
            command.add_argument(
                "--acknowledge-train-validation-numerics",
                action="store_true",
                required=True,
                help="Authorize declared train/validation numerical bytes.",
            )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "build":
        experiment = default_experiment()
        experiment = replace(
            experiment,
            data=replace(experiment.data, stride=args.stride),
            training=replace(experiment.training, max_epochs=args.max_epochs),
        )
        args.expected_prepared_schema_version = 3
        args.expected_causality_profile = "as-of-origin-v1"
        args.validation_only = True
        report = inspect_inputs(args, experiment=experiment)
        equal(report["evidence"]["sha256"], args.assumptions_sha256, "external assumptions pin")
        result = create_campaign(
            args, args.campaign_root, data_root=args.data_root, experiment=experiment
        )
    else:
        binding = open_binding(args.binding, args.binding_sha256)
        if args.mode == "preflight":
            _, report = preflight(binding, data_root=args.data_root)
            result = dict(
                status="preflight_passed",
                metadata_only=True,
                content_verified=False,
                array_semantics_verified=False,
                binding_sha256=binding.sha256,
                evidence=report["evidence"],
                formal_ready=False,
                source_interpretation_verified=False,
                registered_evidence_eligible=False,
            )
        elif args.mode == "check-only":
            result = check_run(binding, synthetic_test=False)
        else:
            result = execute(
                binding,
                mode=args.mode,
                data_root=args.data_root,
                acknowledge_assumptions=args.acknowledge_research_assumptions,
                acknowledge_scope=args.acknowledge_train_validation_numerics,
            )
    print(json.dumps(result, indent=2, sort_keys=True, default=str, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
