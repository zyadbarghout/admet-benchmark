"""Command-line entry point for configured molecular benchmark experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .pipeline import run_experiment


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select molecular models by HPO/CV and evaluate one held-out test split."
    )
    parser.add_argument("config", type=Path, help="YAML experiment configuration")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory declared in the configuration",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    config = load_config(arguments.config)
    output_dir = arguments.output_dir or config.get("output_dir")
    if not output_dir:
        raise ValueError("Set output_dir in YAML or pass --output-dir")
    if not Path(output_dir).is_absolute():
        output_dir = Path(config["_config_dir"]) / output_dir
    result = run_experiment(config, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "selected_model": result.selected_model,
                "validation_score": result.validation_score,
                "held_out_metrics": result.metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
