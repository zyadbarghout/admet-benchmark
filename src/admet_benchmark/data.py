"""Dataset loading and configurable, reproducible train/test splitting."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset
from .config import resolve_config_path
from .splits import split_dataset
from .tables import read_csv


OPENADMET_DATASET_ID = "openadmet/openadmet-expansionrx-challenge-data"
_INTERNAL_SAMPLE_ID = "_benchmark_sample_id"


@dataclass(frozen=True)
class LoadedDataSplits:
    """Raw train/test frames plus the split provenance written to reports."""

    train: pd.DataFrame
    test: pd.DataFrame
    metadata: dict[str, Any]


def _attach_sample_ids(
    frame: pd.DataFrame,
    *,
    source_name: str,
    sample_id_column: str | None,
) -> pd.DataFrame:
    """Attach a stable observation ID before any split or index reset."""

    prepared = frame.copy()
    if sample_id_column:
        if sample_id_column not in prepared:
            raise KeyError(
                f"Configured sample ID column {sample_id_column!r} is missing from {source_name}"
            )
        identifiers = prepared[sample_id_column].astype("string")
        if identifiers.isna().any() or identifiers.str.strip().eq("").any():
            raise ValueError(f"Sample IDs in {source_name} cannot be missing or empty")
        if identifiers.duplicated().any():
            raise ValueError(f"Sample IDs in {source_name} must be unique")
        prepared[_INTERNAL_SAMPLE_ID] = identifiers
    else:
        prepared[_INTERNAL_SAMPLE_ID] = [
            f"{source_name}:{index}" for index in range(len(prepared))
        ]
    return prepared


def _cached_openadmet_splits(cache_root: str | None) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Read locally cached Arrow files when they are available.

    This avoids a network dependency for reproducible local analysis, while the
    normal Hugging Face loader remains the portable fallback for other users.
    """

    roots: list[Path] = []
    if cache_root:
        roots.append(Path(cache_root))
    roots.append(
        Path.home()
        / ".cache/huggingface/datasets/openadmet___openadmet-expansionrx-challenge-data"
    )
    for root in roots:
        if not root.exists():
            continue
        arrows = list(root.rglob("*.arrow"))
        train_path = next((path for path in arrows if path.name.endswith("-train.arrow")), None)
        test_path = next((path for path in arrows if path.name.endswith("-test.arrow")), None)
        if train_path and test_path:
            return (
                Dataset.from_file(str(train_path)).to_pandas(),
                Dataset.from_file(str(test_path)).to_pandas(),
            )
    return None


def load_data_splits(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility wrapper returning only train and test frames."""

    loaded = load_data_split_bundle(config)
    return loaded.train, loaded.test


def load_data_split_bundle(config: dict[str, Any]) -> LoadedDataSplits:
    """Load source data and apply the configured split policy.

    OpenADMET defaults to its official predefined split. Any other strategy
    combines its published train/test source rows before making a fresh,
    target-specific split. CSV datasets default to a seeded random split.
    """

    dataset_config = config["dataset"]
    kind = dataset_config["kind"]
    split_config = dict(dataset_config.get("split") or {})
    default_strategy = "predefined" if kind == "openadmet" else "random"
    strategy = str(split_config.get("strategy", default_strategy)).lower()
    seed = int(config.get("seed", 42))

    if kind == "openadmet":
        cached = _cached_openadmet_splits(dataset_config.get("cache_root"))
        if cached is not None:
            source_train, source_test = cached
        else:
            dataset_id = dataset_config.get("dataset_id", OPENADMET_DATASET_ID)
            source_train, source_test = (
                load_dataset(dataset_id, split="train").to_pandas(),
                load_dataset(dataset_id, split="test").to_pandas(),
            )
        sample_id_column = dataset_config.get("sample_id_column")
        source_train = _attach_sample_ids(
            source_train,
            source_name="openadmet_train",
            sample_id_column=sample_id_column,
        )
        source_test = _attach_sample_ids(
            source_test,
            source_name="openadmet_test",
            sample_id_column=sample_id_column,
        )
        if strategy == "predefined":
            return _predefined_bundle(source_train, source_test, dataset_config, seed)
        source = pd.concat([source_train, source_test], ignore_index=True)
    elif kind == "csv":
        if strategy == "predefined":
            raise ValueError(
                "dataset.split.strategy 'predefined' is only available for OpenADMET datasets"
            )
        csv_path = resolve_config_path(dataset_config["path"], config)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {csv_path}")
        source = _attach_sample_ids(
            read_csv(csv_path, text_columns=[dataset_config.get("sample_id_column", "sample_id"), dataset_config["smiles_column"]]),
            source_name="csv",
            sample_id_column=dataset_config.get("sample_id_column"),
        )
    else:
        raise ValueError("dataset.kind must be 'openadmet' or 'csv'")

    result = split_dataset(
        source,
        target=str(dataset_config["target"]),
        smiles_column=str(dataset_config["smiles_column"]),
        task=str(config["task"]),
        strategy=strategy,
        test_size=float(split_config.get("test_size", dataset_config.get("test_size", 0.2))),
        seed=seed,
        split_config=split_config,
    )
    metadata = {
        **result.metadata,
        "source_kind": kind,
        "source_split": "fresh_target_specific_resplit",
    }
    return LoadedDataSplits(result.train, result.test, metadata)


def _predefined_bundle(
    train: pd.DataFrame,
    test: pd.DataFrame,
    dataset_config: dict[str, Any],
    seed: int,
) -> LoadedDataSplits:
    """Record the challenge split without changing its source membership."""

    train = train.copy()
    test = test.copy()
    train["benchmark_split"] = "train"
    test["benchmark_split"] = "test"
    return LoadedDataSplits(
        train,
        test,
        {
            "strategy": "predefined",
            "seed": seed,
            "source_kind": "openadmet",
            "source_split": "official_predefined",
            "target": str(dataset_config["target"]),
            "smiles_column": str(dataset_config["smiles_column"]),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "actual_test_size": float(len(test) / (len(train) + len(test))),
        },
    )
