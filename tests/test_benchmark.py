from pathlib import Path

import numpy as np
import pandas as pd

from admet_benchmark.benchmark import _prediction_column, _selection_score, run_model_benchmark
from admet_benchmark.features import build_features


def _molecules() -> list[str]:
    return ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "Cc1ccccc1"] * 8


def test_selection_score_flips_error_metrics_only():
    assert _selection_score({"mae": 0.4}, "mae") == -0.4
    assert _selection_score({"pr_auc": 0.4}, "average_precision") == 0.4


def test_prediction_column_supports_chemprop_output_variants():
    assert _prediction_column(pd.DataFrame({"smiles": ["CCO"], "pred_target": [1.0]})) == "pred_target"
    assert _prediction_column(pd.DataFrame({"smiles": ["CCO"], "target": [1.0]})) == "target"


def test_full_rdkit_descriptors_are_finite_or_explicitly_missing():
    features = build_features(
        ["CCO", "CCCCCCCCCCCCCCCCCCCC"],
        representation="rdkit_descriptors",
        descriptor_set="full",
    )

    assert features.values.dtype == np.float64
    assert len(features.names) > 200
    assert not np.isinf(features.values).any()


def test_model_arena_reports_validation_and_held_out_test(tmp_path: Path):
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
        "features": {"n_bits": 64, "include_rdkit_descriptors": False},
        "benchmark": {
            "validation_size": 0.2,
            "selection_metric": "mae",
            "classical_hpo": {"cv_folds": 2, "n_iter": 1, "n_jobs": 1},
            "models": {"classical": ["ridge"], "neural": []},
        },
    }

    result = run_model_benchmark(config, tmp_path / "arena_output")

    assert result.selected_model == "ridge"
    comparison = pd.read_csv(result.output_dir / "model_comparison.csv")
    assert {"validation_mae", "test_mae"}.issubset(comparison.columns)
    assert (result.output_dir / "validation_predictions.csv").exists()
    assert (result.output_dir / "held_out_test_predictions.csv").exists()
    assert (result.output_dir / "data/split_manifest.json").exists()
    assert (result.output_dir / "data/validation_split_manifest.json").exists()
    assert (result.output_dir / "evaluation/report.html").exists()
    predictions = pd.read_csv(result.output_dir / "evaluation_predictions.csv")
    validation_ids = set(predictions.loc[predictions["partition"] == "validation", "sample_id"])
    test_ids = set(predictions.loc[predictions["partition"] == "test", "sample_id"])
    assert validation_ids.isdisjoint(test_ids)
