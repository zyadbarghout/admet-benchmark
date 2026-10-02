"""Run several configured model-arena benchmarks and collect their summaries."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from .benchmark import run_model_benchmark
from .config import load_config, resolve_config_path


def run_benchmark_suite(config_path: str | Path) -> pd.DataFrame:
    suite = load_config(config_path)
    output_dir = resolve_config_path(suite["output_dir"], suite).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    model_rows: list[pd.DataFrame] = []
    for experiment_path, split_override in _expand_experiments(suite):
        experiment = load_config(resolve_config_path(experiment_path, suite))
        if split_override:
            experiment = deepcopy(experiment)
            experiment["dataset"]["split"] = {
                **dict(experiment["dataset"].get("split") or {}),
                **split_override,
            }
        run_name = str(experiment["run_name"])
        split_strategy = str(
            experiment["dataset"].get("split", {}).get(
                "strategy",
                "predefined" if experiment["dataset"]["kind"] == "openadmet" else "random",
            )
        )
        if split_override:
            run_name = f"{run_name}__{split_strategy}"
        result = run_model_benchmark(experiment, output_dir / run_name)
        selected = result.comparison.loc[
            result.comparison["model"] == result.selected_model
        ].iloc[0]
        rows.append(
            {
                "run_name": run_name,
                "task": experiment["task"],
                "target": experiment["dataset"]["target"],
                "split_strategy": split_strategy,
                "selected_model": result.selected_model,
                "validation_selection_score": selected["validation_selection_score"],
                "output_dir": str(result.output_dir),
            }
        )
        model_rows.append(
            result.comparison.assign(
                run_name=run_name,
                task=experiment["task"],
                target=experiment["dataset"]["target"],
                split_strategy=split_strategy,
            )
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "suite_summary.csv", index=False)
    if model_rows:
        comparison = pd.concat(model_rows, ignore_index=True)
        comparison.to_csv(output_dir / "cross_split_model_comparison.csv", index=False)
    return summary


def _expand_experiments(suite: dict[str, Any]) -> list[tuple[str, dict[str, Any] | None]]:
    """Expand optional split matrices while keeping the original YAML form valid.

    A string entry runs one experiment unchanged. A mapping can set ``config``
    and a ``splits`` list; each split is merged into ``dataset.split`` for a
    separate, independently tuned benchmark run.
    """

    expanded: list[tuple[str, dict[str, Any] | None]] = []
    for entry in suite["experiments"]:
        if isinstance(entry, str):
            expanded.append((entry, None))
            continue
        if not isinstance(entry, dict) or "config" not in entry:
            raise ValueError("Each suite experiment must be a config path or a mapping with 'config'")
        splits = entry.get("splits", [None])
        if not isinstance(splits, list) or not splits:
            raise ValueError("Suite experiment 'splits' must be a non-empty list")
        for split in splits:
            if split is not None and not isinstance(split, dict):
                raise ValueError("Each split override must be a mapping")
            expanded.append((str(entry["config"]), split))
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured molecular model benchmark suite.")
    parser.add_argument("config", type=Path, help="Suite YAML file")
    arguments = parser.parse_args()
    print(run_benchmark_suite(arguments.config).to_string(index=False))


if __name__ == "__main__":
    main()
