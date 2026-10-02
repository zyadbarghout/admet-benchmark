"""Command-line interface for evaluating prepared prediction files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .evaluation import run_evaluation, validate_evaluation_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="admet-evaluate",
        description="Validate and evaluate prepared molecular prediction records.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Check inputs without writing results")
    validate.add_argument("config", type=Path, help="Evaluation YAML configuration")
    run = subparsers.add_parser("run", help="Write a complete evaluation bundle")
    run.add_argument("config", type=Path, help="Evaluation YAML configuration")
    run.add_argument("--output-dir", type=Path, help="Override output_dir from YAML")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    try:
        config = load_config(arguments.config)
        if arguments.command == "validate":
            result = validate_evaluation_config(config)
            print(json.dumps(result, indent=2))
            if result["errors"]:
                raise SystemExit(2)
            return
        output = arguments.output_dir.expanduser().resolve() if arguments.output_dir else None
        result = run_evaluation(config, output)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"admet-evaluate: error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "validation_status": result.validation["status"],
                "prediction_rows": len(result.predictions),
                "metric_rows": len(result.fold_metrics),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
