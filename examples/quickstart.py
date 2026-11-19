"""Build both bundled evaluations and an offline chemical diagnostics report.

Run from a checkout with the visual extra installed:
    python examples/quickstart.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from admet_benchmark.chemical_report import generate_chemical_report
from admet_benchmark.config import load_config
from admet_benchmark.evaluation import run_evaluation, validate_evaluation_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/quickstart"),
        help="Destination for generated files (default: outputs/quickstart)",
    )
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    output = args.output_dir.expanduser().resolve()
    bundles = []
    for task in ("regression", "classification"):
        config = load_config(repository / "configs" / f"evaluation_{task}_example.yaml")
        validation = validate_evaluation_config(config)
        if validation["errors"]:
            raise ValueError(f"{task} example failed validation: {validation['errors']}")
        result = run_evaluation(config, output / task)
        bundles.append(result.output_dir)
        print(
            f"{task}: {len(result.predictions)} predictions, "
            f"{len(result.fold_metrics)} metric rows -> {result.output_dir}",
            flush=True,
        )
    report = generate_chemical_report(
        bundles, output_path=output / "chemical_diagnostics.html", embedding="pca",
    )
    print(f"Open in your browser: {report.as_uri()}")


if __name__ == "__main__":
    main()
