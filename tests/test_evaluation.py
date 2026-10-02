from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from admet_benchmark.evaluation import (
    canonicalize_predictions,
    run_evaluation,
    validate_evaluation_config,
)


def _regression_records() -> pd.DataFrame:
    rows = []
    for repeat in range(2):
        for fold in range(2):
            for model, offset in [("model_a", 0.1), ("model_b", 0.25)]:
                for index, observed in enumerate([1.0, 10.0, 100.0]):
                    rows.append(
                        {
                            "id": f"mol_{index}",
                            "smiles": ["CCO", "CCN", "CCC"][index],
                            "endpoint": "KSOL",
                            "model": model,
                            "partition": "validation",
                            "repeat": repeat,
                            "fold": fold,
                            "split": "random",
                            "profile": "raw",
                            "observed": observed,
                            "prediction": observed * (10**offset),
                        }
                    )
    return pd.DataFrame(rows)


def _base_config(path: Path, output: Path) -> dict:
    return {
        "_config_dir": str(path.parent),
        "schema_version": 1,
        "project_name": "toy evaluation",
        "task": "regression",
        "output_dir": str(output),
        "predictions": {
            "path": str(path),
            "columns": {
                "molecule_id": "id",
                "sample_id": "id",
                "smiles": "smiles",
                "endpoint": "endpoint",
                "model": "model",
                "partition": "partition",
                "repeat": "repeat",
                "fold": "fold",
                "split_strategy": "split",
                "preprocessing": "profile",
                "observed": "observed",
                "prediction": "prediction",
            },
        },
        "target_transform": {"default": "log10"},
        "statistics": {"bootstrap_iterations": 25, "seed": 7},
    }


def test_regression_evaluation_transforms_truth_and_prediction_once(tmp_path: Path):
    path = tmp_path / "predictions.csv"
    _regression_records().to_csv(path, index=False)
    result = run_evaluation(_base_config(path, tmp_path / "output"))

    model_a = result.fold_metrics.query("model == 'model_a' and metric == 'mae'")
    assert np.allclose(model_a["value"], 0.1)
    assert result.validation["status"] == "passed"
    assert (result.output_dir / "evaluation_manifest.json").exists()
    assert (result.output_dir / "statistical_comparisons.csv").exists()
    assert "model_a" in pd.read_csv(
        result.output_dir / "statistical_comparisons.csv"
    ).columns
    assert (result.output_dir / "report.html").exists()


def test_multi_endpoint_regression_adds_macro_relative_error(tmp_path: Path):
    records = _regression_records()
    second_endpoint = records.assign(endpoint="LogD")
    path = tmp_path / "multi_endpoint.csv"
    pd.concat([records, second_endpoint], ignore_index=True).to_csv(path, index=False)
    config = _base_config(path, tmp_path / "output")
    config["target_transform"]["endpoints"] = {"LogD": "none"}

    result = run_evaluation(config)
    macro = result.fold_metrics.query("endpoint == '__macro__' and metric == 'ma_rae'")

    assert len(macro) == 8
    assert macro["n_endpoints"].eq(2).all()


def test_wide_classification_input_requires_and_uses_declared_threshold(tmp_path: Path):
    source = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "smiles": ["CCO", "CCN", "CCC", "CCCl"],
            "label": [0, 0, 1, 1],
            "rf": [0.1, 0.45, 0.55, 0.8],
            "cp": [0.2, 0.3, 0.6, 0.7],
        }
    )
    path = tmp_path / "classification.csv"
    source.to_csv(path, index=False)
    config = {
        "_config_dir": str(tmp_path),
        "schema_version": 1,
        "task": "classification",
        "output_dir": str(tmp_path / "output"),
        "predictions": {
            "path": str(path),
            "columns": {
                "molecule_id": "id",
                "sample_id": "id",
                "smiles": "smiles",
                "observed": "label",
            },
            "prediction_columns": {"random_forest": "rf", "chemprop": "cp"},
            "defaults": {
                "endpoint": "BBBP",
                "partition": "test",
                "preprocessing": "raw",
            },
        },
        "statistics": {"bootstrap_iterations": 25, "seed": 7},
    }

    failed = validate_evaluation_config(config)
    assert "threshold" in failed["errors"][0].lower()

    config["thresholds"] = {"endpoints": {"BBBP": 0.4}}
    result = run_evaluation(config)
    assert set(result.fold_metrics["decision_threshold"]) == {0.4}
    assert set(result.predictions["model"]) == {"random_forest", "chemprop"}


def test_duplicate_prediction_key_fails_validation(tmp_path: Path):
    records = _regression_records()
    records = pd.concat([records, records.iloc[[0]]], ignore_index=True)
    path = tmp_path / "duplicates.csv"
    records.to_csv(path, index=False)
    result = validate_evaluation_config(_base_config(path, tmp_path / "output"))

    assert result["status"] == "failed"
    assert any("duplicate" in error for error in result["errors"])


def test_incomplete_model_coverage_is_excluded_from_paired_comparison(tmp_path: Path):
    records = _regression_records()
    remove = (
        records["model"].eq("model_b")
        & records["repeat"].eq(0)
        & records["fold"].eq(0)
        & records["id"].eq("mol_0")
    )
    path = tmp_path / "incomplete.csv"
    records.loc[~remove].to_csv(path, index=False)

    result = run_evaluation(_base_config(path, tmp_path / "output"))
    comparisons = pd.read_csv(result.output_dir / "statistical_comparisons.csv")

    assert result.validation["status"] == "passed_with_warnings"
    assert comparisons["paired_results"].eq(3).all()


def test_three_models_use_conover_friedman_with_holm_adjustment(tmp_path: Path):
    records = _regression_records()
    third = records.loc[records["model"] == "model_a"].copy()
    third["model"] = "model_c"
    third["prediction"] = third["observed"] * (10**0.18)
    path = tmp_path / "three_models.csv"
    pd.concat([records, third], ignore_index=True).to_csv(path, index=False)

    result = run_evaluation(_base_config(path, tmp_path / "output"))
    comparisons = pd.read_csv(result.output_dir / "statistical_comparisons.csv")

    assert set(comparisons.query("metric == 'mae'")["pairwise_test"]) == {
        "conover_friedman"
    }
    assert comparisons["paired_results"].eq(4).all()
    assert comparisons["inference_status"].eq("exploratory_unverified_dependence").all()
    assert comparisons["inference_caveat"].str.contains("share molecules").all()
    assert comparisons["holm_adjusted_p_value"].notna().all()


def test_canonicalize_reports_missing_mapped_columns(tmp_path: Path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"id": ["a"]}).to_csv(path, index=False)
    config = _base_config(path, tmp_path / "output")
    with pytest.raises(KeyError, match="smiles"):
        canonicalize_predictions(config)


def test_same_sample_cannot_change_chemical_identity_between_models(tmp_path: Path):
    records = _regression_records()
    conflict = (
        records["model"].eq("model_b")
        & records["repeat"].eq(0)
        & records["fold"].eq(0)
        & records["id"].eq("mol_0")
    )
    records.loc[conflict, "smiles"] = "CCOC"
    path = tmp_path / "identity_conflict.csv"
    records.to_csv(path, index=False)

    result = validate_evaluation_config(_base_config(path, tmp_path / "output"))

    assert result["status"] == "failed"
    assert any("canonical_smiles disagrees" in error for error in result["errors"])


def test_uncertainty_and_intervals_are_evaluated_per_fold(tmp_path: Path):
    records = _regression_records()
    rank = records["id"].str.extract(r"(\d+)$")[0].astype(int)
    offsets = rank.map({0: 0.05, 1: 0.15, 2: 0.30})
    records["prediction"] = records["observed"] * np.power(10.0, offsets)
    records["uncertainty"] = rank.map({0: 0.1, 1: 0.5, 2: 0.9})
    records["lower"] = records["prediction"] * (10**-0.1)
    records["upper"] = records["prediction"] * (10**0.1)
    path = tmp_path / "uncertainty.csv"
    records.to_csv(path, index=False)
    config = _base_config(path, tmp_path / "output")
    config["predictions"]["columns"].update(
        {
            "uncertainty": "uncertainty",
            "prediction_lower": "lower",
            "prediction_upper": "upper",
        }
    )
    config["uncertainty"] = {
        "higher_is_more_uncertain": True,
        "scale": "evaluation",
        "interval_confidence": 0.8,
    }

    result = run_evaluation(config)
    fold = result.fold_metrics.query("model == 'model_a' and repeat == 0 and fold == 0")

    assert result.validation["uncertainty_models"] == ["model_a", "model_b"]
    assert fold.set_index("metric").loc["uncertainty_error_spearman", "value"] == pytest.approx(1)
    assert fold.set_index("metric").loc["interval_coverage", "value"] == pytest.approx(1 / 3)
    assert fold["n_samples"].eq(3).all()
    assert fold["n_molecules"].eq(3).all()
    assert result.predictions["uncertainty_percentile"].between(0, 1).all()


def test_cli_output_override_is_relative_to_working_directory(tmp_path):
    import json
    import subprocess
    import sys
    import yaml
    nested = tmp_path / 'configs'
    nested.mkdir()
    predictions = tmp_path / 'input.csv'
    _regression_records().to_csv(predictions, index=False)
    config = _base_config(predictions, tmp_path / 'unused')
    config.pop('_config_dir')
    path = nested / 'evaluation.yaml'
    path.write_text(yaml.safe_dump(config))
    completed = subprocess.run(
        [sys.executable, '-m', 'admet_benchmark.evaluate_cli', 'run', str(path), '--output-dir', 'chosen'],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert Path(json.loads(completed.stdout)['output_dir']) == tmp_path / 'chosen'
    assert (tmp_path / 'chosen' / 'predictions.csv').exists()
    assert not (nested / 'chosen').exists()
