from pathlib import Path

import pandas as pd
import pytest

from admet_benchmark.data import load_data_split_bundle
from admet_benchmark.splits import split_dataset
from admet_benchmark.suite import _expand_experiments


def _molecular_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "smiles": [
                "c1ccccc1",
                "Cc1ccccc1",
                "c1ccncc1",
                "Cc1ccncc1",
                "C1CCCCC1",
                "CC1CCCCC1",
                "c1ccoc1",
                "Cc1ccoc1",
            ],
            "value": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "label": [0, 0, 1, 1, 0, 0, 1, 1],
            "measured_at": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
            ],
        }
    )


@pytest.mark.parametrize("strategy", ["scaffold", "butina"])
def test_chemical_splits_keep_groups_entirely_in_one_partition(strategy: str):
    result = split_dataset(
        _molecular_frame(),
        target="value",
        smiles_column="smiles",
        task="regression",
        strategy=strategy,
        test_size=0.3,
        seed=7,
        split_config={"butina_distance_threshold": 0.4},
    )

    assert result.metadata["overlapping_group_count"] == 0
    assert set(result.train["benchmark_split_group"]).isdisjoint(
        result.test["benchmark_split_group"]
    )
    assert len(result.train) + len(result.test) == len(_molecular_frame())


def test_temporal_split_keeps_all_later_observations_for_test():
    result = split_dataset(
        _molecular_frame(),
        target="label",
        smiles_column="smiles",
        task="classification",
        strategy="temporal",
        test_size=0.5,
        seed=7,
        split_config={"time_column": "measured_at"},
    )

    assert result.metadata["time_column"] == "measured_at"
    assert result.metadata["train_time_max"] < result.metadata["test_time_min"]
    assert set(result.train["benchmark_split"]) == {"train"}
    assert set(result.test["benchmark_split"]) == {"test"}


def test_temporal_split_requires_an_explicit_time_column():
    with pytest.raises(ValueError, match="time_column"):
        split_dataset(
            _molecular_frame(),
            target="value",
            smiles_column="smiles",
            task="regression",
            strategy="temporal",
            test_size=0.25,
            seed=7,
            split_config={},
        )


def test_csv_random_split_records_reproducible_metadata(tmp_path: Path):
    path = tmp_path / "data.csv"
    _molecular_frame().to_csv(path, index=False)
    config = {
        "_config_dir": str(tmp_path),
        "task": "classification",
        "seed": 7,
        "dataset": {
            "kind": "csv",
            "path": str(path),
            "target": "label",
            "smiles_column": "smiles",
            "split": {"strategy": "random", "test_size": 0.25},
        },
    }

    first = load_data_split_bundle(config)
    second = load_data_split_bundle(config)

    assert first.metadata["strategy"] == "random"
    assert first.metadata["train_class_counts"] == second.metadata["train_class_counts"]
    assert first.train.index.tolist() == second.train.index.tolist()


def test_suite_split_matrix_expands_each_config_without_breaking_legacy_entries():
    expanded = _expand_experiments(
        {
            "experiments": [
                "benchmarks/legacy.yaml",
                {
                    "config": "benchmarks/matrix.yaml",
                    "splits": [{"strategy": "random"}, {"strategy": "scaffold"}],
                },
            ]
        }
    )

    assert expanded == [
        ("benchmarks/legacy.yaml", None),
        ("benchmarks/matrix.yaml", {"strategy": "random"}),
        ("benchmarks/matrix.yaml", {"strategy": "scaffold"}),
    ]
