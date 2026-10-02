from pathlib import Path

import pandas as pd

from admet_benchmark.pipeline import run_experiment


def _molecules() -> list[str]:
    return ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "Cc1ccccc1"] * 8


def test_regression_pipeline_selects_by_validation_then_writes_test_outputs(tmp_path: Path):
    data = pd.DataFrame({"smiles": _molecules(), "value": list(range(48))})
    csv_path = tmp_path / "toy_regression.csv"
    data.to_csv(csv_path, index=False)
    config = {
        "_config_dir": str(tmp_path),
        "task": "regression",
        "seed": 7,
        "dataset": {
            "kind": "csv",
            "path": str(csv_path),
            "target": "value",
            "smiles_column": "smiles",
            "test_size": 0.25,
            "target_transform": "none",
        },
        "features": {"n_bits": 64},
        "hpo": {"cv_folds": 2, "n_iter": 1, "n_jobs": 1},
        "models": ["ridge"],
    }

    result = run_experiment(config, tmp_path / "regression_output")

    assert result.selected_model == "ridge"
    assert (result.output_dir / "validation_leaderboard.csv").exists()
    assert (result.output_dir / "held_out_predictions.csv").exists()
    assert (result.output_dir / "report.html").exists()


def test_classification_pipeline_exports_binary_metrics(tmp_path: Path):
    data = pd.DataFrame(
        {"smiles": _molecules(), "label": [index % 2 for index in range(48)]}
    )
    csv_path = tmp_path / "toy_classification.csv"
    data.to_csv(csv_path, index=False)
    config = {
        "_config_dir": str(tmp_path),
        "task": "classification",
        "seed": 7,
        "dataset": {
            "kind": "csv",
            "path": str(csv_path),
            "target": "label",
            "smiles_column": "smiles",
            "test_size": 0.25,
        },
        "features": {"n_bits": 64},
        "hpo": {"cv_folds": 2, "n_iter": 1, "n_jobs": 1},
        "models": ["logistic_regression"],
    }

    result = run_experiment(config, tmp_path / "classification_output")

    assert "mcc" in result.metrics
    assert (result.output_dir / "plots/roc_pr_curves.png").exists()
