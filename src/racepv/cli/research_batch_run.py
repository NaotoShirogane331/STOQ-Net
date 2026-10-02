"""Explicit metadata preparation and validation-only non-epoch research execution."""

import argparse
import json
from pathlib import Path

from racepv.training.exploratory_contract import safe_read, strict_json
from racepv.training.research_batch_campaign import (
    DEFAULT_RESOURCES,
    create_campaign,
    open_binding,
    preflight,
)
from racepv.training.research_batch_runtime import check_run, execute


def build_parser():
    parser = argparse.ArgumentParser(description="Non-epoch research jobs; no qualified evidence or heldout access.")
    modes = parser.add_subparsers(dest="mode")
    prepare = modes.add_parser("prepare", help="Metadata/receipt text only; never arrays, weights, models or fit.")
    prepare.add_argument("--campaign-root", type=Path, required=True)
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--jobs-json", type=Path, required=True)
    prepare.add_argument("--assumptions-sha256", required=True)
    for key, default in DEFAULT_RESOURCES.items():
        prepare.add_argument("--" + key.replace("_", "-"), type=int, default=default)
    for split in ("train", "validation"):
        prepare.add_argument("--" + split, type=Path, required=True)
        for pin in ("descriptor", "read-scope"):
            prepare.add_argument(f"--prepared-{split}-{pin}-sha256", required=True)
    for mode in ("preflight", "start", "continue", "check-run"):
        command = modes.add_parser(mode)
        command.add_argument("--binding", type=Path, required=True)
        command.add_argument("--binding-sha256", required=True)
        command.add_argument("--campaign-sha256", required=True)
        if mode != "check-run":
            command.add_argument("--data-root", type=Path, required=True)
        if mode in ("start", "continue"):
            command.add_argument("--acknowledge-research-assumptions", action="store_true", required=True)
            command.add_argument("--acknowledge-train-validation-numerics", action="store_true", required=True)
            command.add_argument("--foundation-project-root", type=Path)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode is None:
        parser.print_help()
        return dict(metadata_only=True, status="no_action")
    if args.mode == "prepare":
        requested = strict_json(safe_read(args.jobs_json), "batch jobs")
        if set(requested) != {"jobs"} or not isinstance(requested["jobs"], list) or not requested["jobs"]:
            raise ValueError("jobs JSON requires one nonempty jobs list")
        args.expected_prepared_schema_version = 3
        args.expected_causality_profile = "as-of-origin-v1"
        args.validation_only = True
        result = create_campaign(args, args.campaign_root, data_root=args.data_root, jobs=requested["jobs"],
            assumptions_sha256=args.assumptions_sha256,
            **{key: getattr(args, key) for key in DEFAULT_RESOURCES})
    else:
        binding = open_binding(args.binding, args.binding_sha256, campaign_sha256=args.campaign_sha256)
        if args.mode == "preflight":
            _, report = preflight(binding, data_root=args.data_root)
            result = dict(metadata_only=True, status="preflight_passed", evidence=report["evidence"], formal_ready=False)
        elif args.mode == "check-run":
            result = check_run(binding)
        else:
            result = execute(binding, mode=args.mode, data_root=args.data_root,
                acknowledge_assumptions=args.acknowledge_research_assumptions,
                acknowledge_scope=args.acknowledge_train_validation_numerics,
                foundation_project_root=args.foundation_project_root)
    print(json.dumps(result, sort_keys=True, indent=2, default=str, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
