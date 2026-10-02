from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from admet_benchmark.evaluation import canonicalize_predictions, validate_evaluation_config, run_evaluation
from admet_benchmark.provenance import runtime_signature, feature_schema, implementation_signature, verify_compatibility


def example(tmp_path, frame):
    path = tmp_path / "predictions.csv"
    frame.to_csv(path, index=False)
    return {"schema_version": 1, "task": "regression", "_config_dir": str(tmp_path),
        "output_dir": str(tmp_path / "evaluated"), "target_transform": {"default": "none"},
        "statistics": {"bootstrap_iterations": 20},
        "predictions": {"path": str(path), "columns": {c: c for c in frame.columns},
            "defaults": {"endpoint": "assay", "model": "model", "partition": "validation"}}}


def records():
    return pd.DataFrame({"sample_id": ["001", "002", "NA"], "molecule_id": ["001", "002", "NA"],
        "smiles": ["CCO", "CCN", "CCC"], "observed": [1., 2., 3.], "prediction": [1.1, 1.9, 2.8]})


def test_identifiers_survive_evaluation_and_report(tmp_path):
    config = example(tmp_path, records())
    result = run_evaluation(config)
    assert result.predictions.sample_id.tolist() == ["001", "002", "NA"]
    from admet_benchmark.chemical_report import _load_evaluation_bundle
    bundle = _load_evaluation_bundle(result.output_dir)[0]
    assert {r["sample_id"] for r in bundle.records} == {"001", "002", "NA"}
    assert result.validation["training_leakage_check"] == "unverified_no_training_assignments"


@pytest.mark.parametrize("kind", ["training", "cross_fold", "alias"])
def test_leakage_is_rejected(tmp_path, kind):
    rows = records().assign(partition="validation", fold=0)
    if kind == "training":
        other = rows.assign(partition="train")
    else:
        other = rows.assign(partition="test", fold=1)
        if kind == "alias":
            other = other.assign(sample_id=["a", "b", "c"], molecule_id=["a", "b", "c"])
    result = validate_evaluation_config(example(tmp_path, pd.concat([rows, other])))
    assert result["status"] == "failed"
    assert any("leakage" in error.lower() for error in result["errors"])


def test_legitimate_cv_training_rotation_is_allowed(tmp_path):
    rows = records().assign(fold=0, partition=["validation", "train", "train"])
    other = records().assign(fold=1, partition=["train", "validation", "validation"])
    result = validate_evaluation_config(example(tmp_path, pd.concat([rows, other])))
    assert result["status"] == "passed"


def test_chemical_aliases_are_one_bootstrap_unit(tmp_path, monkeypatch):
    import admet_benchmark.evaluation as module
    original = module._molecule_bootstrap_indices
    groups = []
    def capture(ids, rng):
        groups.append(ids.copy())
        return original(ids, rng)
    monkeypatch.setattr(module, "_molecule_bootstrap_indices", capture)
    result = run_evaluation(example(tmp_path, records().assign(smiles=["CCO", "OCC", "CCC"])))
    assert result.predictions.molecule_id.nunique() == 3
    assert result.fold_metrics.n_molecules.eq(2).all()
    assert groups and all(ids[0] == ids[1] and ids[0] != ids[2] for ids in groups)


def test_runtime_and_feature_order_are_checked():
    data = {"runtime": runtime_signature(), "implementation_hash": implementation_signature(),
            "feature_schema": feature_schema("rdkit_descriptors")}
    verify_compatibility(data, representation="rdkit_descriptors")
    data["feature_schema"]["names"].reverse()
    with pytest.raises(ValueError, match="Feature schema"):
        verify_compatibility(data, representation="rdkit_descriptors")
    data["runtime"]["rdkit"] = "different"
    with pytest.raises(ValueError, match="Runtime compatibility"):
        verify_compatibility(data)


def test_fractional_fold_fails_cleanly(tmp_path):
    result = validate_evaluation_config(example(tmp_path, records().assign(fold=.5)))
    assert result["status"] == "failed"
    assert any("integers" in error for error in result["errors"])
