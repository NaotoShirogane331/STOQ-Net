"""Analyze the frozen RaCE-PV adjustment study using validation data only."""

from __future__ import annotations

import argparse
from pathlib import Path

from racepv.adjustment import analyze_adjustment_manifest
from racepv.utils import atomic_json_dump


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite adjustment analysis: {args.output}")
    report = analyze_adjustment_manifest(args.manifest, args.config)
    atomic_json_dump(report, args.output)
    print(f"Wrote validation-only adjustment analysis to {args.output}")


if __name__ == "__main__":
    main()
