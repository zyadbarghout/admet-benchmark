from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytest.importorskip("plotly")

from admet_benchmark.chemical_report import generate_chemical_report, _sample_records


def _write_run(path: Path, *, task: str, target: str, label: str) -> None:
    path.mkdir()
    (path / "data").mkdir()
    (path / "evaluation").mkdir()
    metadata = {
        "task": task,
        "target": target,
        "selected_model": "random_forest",
        "selection_metric": "mae" if task == "regression" else "pr_auc",
        "split": {"strategy": "scaffold"},
        "validation_split": {"strategy": "scaffold"},
    }
    (path / "benchmark_metadata.json").write_text(json.dumps(metadata))
    config = {
        "project_name": f"{label} model arena",
        "run_name": label.lower(),
        "dataset": {
            "smiles_column": "smiles",
            "target_transform": "none",
        },
        "benchmark": {"evaluation": {"decision_threshold": 0.5}},
    }
    (path / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    pd.DataFrame(
        {
            "model": ["random_forest", "extra_trees"],
            "family": ["classical", "classical"],
            "validation_mae" if task == "regression" else "validation_pr_auc": [
                0.2,
                0.3,
            ],
        }
    ).to_csv(path / "model_comparison.csv", index=False)
    base = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "molecule_id": ["mol-a", "mol-b"],
            "smiles": ["CCO", "c1ccccc1"],
            "target_raw": [0.0, 1.0],
            "target_model": [0.0, 1.0],
        }
    )
    base.to_csv(path / "data" / "development.csv", index=False)
    predictions = base.assign(
        prediction__random_forest=[0.1, 0.8],
        prediction__extra_trees=[0.2, 0.7],
    )
    predictions.to_csv(path / "validation_predictions.csv", index=False)
    predictions.to_csv(path / "held_out_test_predictions.csv", index=False)
    metric = "mae" if task == "regression" else "pr_auc"
    pd.DataFrame(
        {
            "partition": ["validation", "test"] * 2,
            "model": ["random_forest", "random_forest", "extra_trees", "extra_trees"],
            "metric": [metric] * 4,
            "value": [0.2, 0.25, 0.3, 0.35],
            "ci_lower": [0.1, 0.15, 0.2, 0.25],
            "ci_upper": [0.3, 0.35, 0.4, 0.45],
            "bootstrap_iterations": [100] * 4,
            "n_molecules": [2] * 4,
        }
    ).to_csv(path / "evaluation" / "fold_metrics.csv", index=False)


def test_multi_endpoint_report_contains_chemical_and_task_specific_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regression = tmp_path / "regression"
    classification = tmp_path / "classification"
    _write_run(regression, task="regression", target="KSOL", label="KSOL")
    _write_run(classification, task="classification", target="label", label="BBBP")

    def fake_embedding(records, **_):
        for index, record in enumerate(records):
            record["x"] = float(index)
            record["y"] = float(-index)

    def fake_structures(records):
        images = {}
        for record in records:
            record["structure"] = record["chemical_id"]
            images[record["chemical_id"]] = "data:image/webp;base64,UklGRg=="
        return images

    monkeypatch.setattr("admet_benchmark.chemical_report._attach_embedding", fake_embedding)
    monkeypatch.setattr(
        "admet_benchmark.chemical_report._attach_structure_images", fake_structures
    )
    output = generate_chemical_report(
        [regression, classification], output_path=tmp_path / "report.html"
    )

    html = output.read_text(encoding="utf-8")
    assert "Chemical Model Diagnostics" in html
    assert "Chemical space" in html
    assert "Predicted probability vs true class" in html
    assert "Observed vs predicted" in html
    assert "Molecules to investigate" in html
    assert "data:image/webp;base64" in html
    assert "KSOL" in html
    assert "BBBP" in html
    assert "How to use this report" in html
    assert "Complete benchmark results" in html
    assert str(tmp_path) not in html


def test_report_rejects_incomplete_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing benchmark artifacts"):
        generate_chemical_report([tmp_path], output_path=tmp_path / "report.html")


def test_canonical_fold_bundle_exposes_uncertainty_and_raw_fold_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "canonical"
    run.mkdir()
    rows = []
    metric_rows = []
    for fold in (0, 1):
        for model, offset in (("model_a", 0.1), ("model_b", 0.2)):
            for index, smiles in enumerate(("CCO", "CCN", "CCC")):
                observed = float(index)
                prediction = observed + offset * (index + 1)
                rows.append(
                    {
                        "task": "regression",
                        "endpoint": "KSOL",
                        "model": model,
                        "partition": "validation",
                        "repeat": 0,
                        "fold": fold,
                        "sample_id": f"sample-{index}",
                        "molecule_id": f"mol-{index}",
                        "smiles": smiles,
                        "observed_submitted": observed,
                        "observed": observed,
                        "prediction": prediction,
                        "residual": prediction - observed,
                        "absolute_error": abs(prediction - observed),
                        "uncertainty": float(index + 1),
                        "uncertainty_percentile": (index + 1) / 3,
                        "prediction_lower": prediction - 0.25,
                        "prediction_upper": prediction + 0.25,
                        "preprocessing": "raw",
                        "split_strategy": "scaffold",
                        "target_transform": "none",
                    }
                )
            metric_rows.append(
                {
                    "task": "regression",
                    "endpoint": "KSOL",
                    "model": model,
                    "partition": "validation",
                    "repeat": 0,
                    "fold": fold,
                    "metric": "mae",
                    "value": offset * (fold + 1),
                    "ci_lower": offset / 2,
                    "ci_upper": offset * 2,
                    "n_molecules": 3,
                    "preprocessing": "raw",
                    "split_strategy": "scaffold",
                }
            )
    pd.DataFrame(rows).to_csv(run / "predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(run / "fold_metrics.csv", index=False)
    (run / "resolved_config.yaml").write_text(
        yaml.safe_dump({"project_name": "Canonical UQ", "selected_model": "model_a"})
    )

    def fake_embedding(records, **_):
        for index, record in enumerate(records):
            record["x"] = float(index)
            record["y"] = float(-index)

    def fake_structures(records):
        images = {}
        for record in records:
            record["structure"] = record["chemical_id"]
            images[record["chemical_id"]] = "data:image/webp;base64,UklGRg=="
        return images

    monkeypatch.setattr("admet_benchmark.chemical_report._attach_embedding", fake_embedding)
    monkeypatch.setattr(
        "admet_benchmark.chemical_report._attach_structure_images", fake_structures
    )
    output = generate_chemical_report([run], output_path=tmp_path / "canonical.html")
    html = output.read_text(encoding="utf-8")

    assert "Uncertainty diagnostics" in html
    assert "Error-retention curve" in html
    assert "All folds" in html
    assert "type:'violin'" in html
    assert "meanline:{visible:false}" in html
    assert '"fold":1' in html
    assert '"uncertainty":3.0' in html


def test_report_sampling_keeps_complete_chemical_histories() -> None:
    records = [
        {
            "id": f"{chemical}-{fold}",
            "chemical_id": chemical,
            "split": "validation",
            "fold": fold,
        }
        for chemical in ("a", "b", "c")
        for fold in range(4)
    ]

    sampled = _sample_records(records, max_points=6, seed=5)
    selected = {record["chemical_id"] for record in sampled}

    assert len(sampled) <= 6
    assert selected
    assert all(
        sum(record["chemical_id"] == chemical for record in sampled) == 4
        for chemical in selected
    )


@pytest.mark.parametrize('smiles', [['CCO'], ['CCO', 'CCN']])
def test_tiny_reports_have_finite_coordinates(smiles):
    import math
    from admet_benchmark.chemical_report import _attach_embedding
    records=[{'canonical_smiles':value} for value in smiles]
    _attach_embedding(records,perplexity=35,seed=42)
    assert all(math.isfinite(row['x']) and math.isfinite(row['y']) for row in records)


def test_pca_coordinates_are_deterministic():
    from admet_benchmark.chemical_report import _attach_embedding
    smiles=['CCO','CCN','CCC','c1ccccc1']
    first=[{'canonical_smiles':value} for value in smiles]
    second=[{'canonical_smiles':value} for value in smiles]
    _attach_embedding(first,perplexity=35,seed=42,method='pca')
    _attach_embedding(second,perplexity=35,seed=42,method='pca')
    assert first==second
