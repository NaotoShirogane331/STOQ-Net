"""Analyze the reviewed RaCE-PV day-one manifest using validation data only."""

from __future__ import annotations

import argparse
from pathlib import Path

from racepv.utils import atomic_json_dump
from racepv.validation_screening import analyze_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite validation analysis: {args.output}")
    report = analyze_manifest(args.manifest, args.config)
    atomic_json_dump(report, args.output)
    print(f"Wrote validation-only screening analysis to {args.output}")


if __name__ == "__main__":
    main()
