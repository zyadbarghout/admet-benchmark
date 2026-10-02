"""Deterministic molecular train/test split strategies.

Chemical split strategies operate on labelled, valid molecules only. This is
intentional: allocating rows that cannot be fitted or evaluated would distort
the requested test fraction and can make a small endpoint unexpectedly empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass(frozen=True)
class SplitResult:
    """Frames and provenance produced by a non-predefined split."""

    train: pd.DataFrame
    test: pd.DataFrame
    metadata: dict[str, Any]


def split_dataset(
    frame: pd.DataFrame,
    *,
    target: str,
    smiles_column: str,
    task: str,
    strategy: str,
    test_size: float,
    seed: int,
    split_config: dict[str, Any],
) -> SplitResult:
    """Create a reproducible split without leaking chemical groups.

    ``random`` uses stratification for binary tasks. ``scaffold`` and
    ``butina`` select whole chemical groups for test. ``temporal`` keeps all
    earlier observations in train and requires a user-provided time column.
    """

    if strategy not in {"random", "scaffold", "butina", "temporal"}:
        raise ValueError(
            "dataset.split.strategy must be one of random, scaffold, butina, or temporal"
        )
    if not 0 < test_size < 1:
        raise ValueError("dataset.split.test_size must be between zero and one")

    eligible, exclusion_counts = _eligible_rows(frame, target, smiles_column)
    if len(eligible) < 4:
        raise ValueError("Too few labelled, valid molecules remain to make a train/test split")
    y = eligible[target].to_numpy()
    if task == "classification":
        _require_binary_labels(y)

    groups: np.ndarray | None = None
    time_metadata: dict[str, Any] = {}
    if strategy == "random":
        identities = eligible[smiles_column].map(lambda value: Chem.MolToSmiles(Chem.MolFromSmiles(value), isomericSmiles=True)).to_numpy()
        if len(set(identities)) < len(identities):
            groups = identities
            train_index, test_index = _group_indices(groups, y, task, test_size, seed, split_config)
        else:
            train_index, test_index = _random_indices(y, task, test_size, seed)
    elif strategy == "scaffold":
        groups = _scaffold_groups(eligible[smiles_column])
        train_index, test_index = _group_indices(
            groups, y, task, test_size, seed, split_config
        )
    elif strategy == "butina":
        groups = _butina_groups(eligible[smiles_column], split_config)
        train_index, test_index = _group_indices(
            groups, y, task, test_size, seed, split_config
        )
    else:
        train_index, test_index, time_metadata = _temporal_indices(
            eligible, y, task, test_size, split_config
        )

    train = eligible.iloc[train_index].copy()
    test = eligible.iloc[test_index].copy()
    train["benchmark_split"] = "train"
    test["benchmark_split"] = "test"
    if groups is not None:
        train["benchmark_split_group"] = groups[train_index]
        test["benchmark_split_group"] = groups[test_index]

    metadata: dict[str, Any] = {
        "strategy": strategy,
        "seed": seed,
        "requested_test_size": test_size,
        "source_rows": int(len(frame)),
        "eligible_rows": int(len(eligible)),
        "excluded_before_split": exclusion_counts,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "actual_test_size": float(len(test) / len(eligible)),
        "target": target,
        "smiles_column": smiles_column,
    }
    if task == "classification":
        metadata["train_class_counts"] = _class_counts(train[target])
        metadata["test_class_counts"] = _class_counts(test[target])
    if groups is not None:
        metadata.update(_group_metadata(groups, train_index, test_index))
    metadata.update(time_metadata)
    return SplitResult(train=train, test=test, metadata=metadata)


def _eligible_rows(
    frame: pd.DataFrame, target: str, smiles_column: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    missing = {target, smiles_column}.difference(frame.columns)
    if missing:
        raise KeyError(f"Dataset is missing required column(s): {', '.join(sorted(missing))}")

    prepared = frame.copy()
    prepared[target] = pd.to_numeric(prepared[target], errors="coerce")
    non_empty_smiles = (
        prepared[smiles_column].notna()
        & prepared[smiles_column].astype(str).str.strip().ne("")
    )
    labelled = prepared[target].notna()
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    valid_smiles = prepared[smiles_column].map(
        lambda value: Chem.MolFromSmiles(str(value)) is not None
    )
    keep = labelled & non_empty_smiles & valid_smiles
    return prepared.loc[keep].copy(), {
        "missing_or_non_numeric_target": int((~labelled).sum()),
        "missing_or_empty_smiles": int((~non_empty_smiles).sum()),
        "invalid_smiles": int((non_empty_smiles & ~valid_smiles).sum()),
    }


def _require_binary_labels(values: np.ndarray) -> None:
    labels = set(np.unique(values))
    if not labels.issubset({0.0, 1.0}) or len(labels) != 2:
        raise ValueError(
            "Classification split strategies require both binary classes encoded as 0 and 1"
        )


def _random_indices(
    y: np.ndarray, task: str, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    train_index, test_index = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y if task == "classification" else None,
    )
    return np.asarray(train_index), np.asarray(test_index)


def _scaffold_groups(smiles: pd.Series) -> np.ndarray:
    groups: list[str] = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(str(value))
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        groups.append(scaffold or "[no_bemis_murcko_scaffold]")
    return np.asarray(groups, dtype=object)


def _butina_groups(smiles: pd.Series, split_config: dict[str, Any]) -> np.ndarray:
    """Cluster circular fingerprints without allocating a Python float per pair."""

    radius = int(split_config.get("butina_radius", 2))
    n_bits = int(split_config.get("butina_n_bits", 2048))
    distance_threshold = float(split_config.get("butina_distance_threshold", 0.4))
    if not 0 < distance_threshold <= 1:
        raise ValueError("dataset.split.butina_distance_threshold must be in (0, 1]")

    molecules = [Chem.MolFromSmiles(str(value)) for value in smiles]
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=n_bits) for molecule in molecules]
    count = len(fingerprints)
    distances = np.empty(count * (count - 1) // 2, dtype=np.float32)
    position = 0
    for index in range(1, count):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[index], fingerprints[:index]
        )
        next_position = position + index
        distances[position:next_position] = 1.0 - np.asarray(similarities, dtype=np.float32)
        position = next_position

    clusters = Butina.ClusterData(
        distances,
        count,
        distance_threshold,
        isDistData=True,
        reordering=True,
    )
    groups = np.empty(count, dtype=object)
    for cluster_index, cluster in enumerate(clusters):
        groups[list(cluster)] = f"butina_{cluster_index:05d}"
    return groups


def _group_indices(
    groups: np.ndarray,
    y: np.ndarray,
    task: str,
    test_size: float,
    seed: int,
    split_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Chemical split requires at least two distinct chemical groups")
    attempts = int(split_config.get("group_split_attempts", 128))
    if attempts < 1:
        raise ValueError("dataset.split.group_split_attempts must be positive")

    best: tuple[float, np.ndarray, np.ndarray] | None = None
    splitter = GroupShuffleSplit(
        n_splits=attempts, test_size=test_size, random_state=seed
    )
    for train_index, test_index in splitter.split(np.zeros(len(y)), y, groups):
        if task == "classification" and (
            len(np.unique(y[train_index])) != 2 or len(np.unique(y[test_index])) != 2
        ):
            continue
        fraction_penalty = abs(len(test_index) / len(y) - test_size)
        balance_penalty = (
            abs(float(np.mean(y[test_index])) - float(np.mean(y)))
            if task == "classification"
            else 0.0
        )
        score = fraction_penalty + 0.25 * balance_penalty
        if best is None or score < best[0]:
            best = (score, np.asarray(train_index), np.asarray(test_index))
    if best is None:
        raise ValueError(
            "Could not create a grouped split containing both classes in train and test"
        )
    return best[1], best[2]


def _temporal_indices(
    frame: pd.DataFrame,
    y: np.ndarray,
    task: str,
    test_size: float,
    split_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    time_column = split_config.get("time_column")
    if not time_column:
        raise ValueError(
            "dataset.split.time_column is required when dataset.split.strategy is temporal"
        )
    if time_column not in frame:
        raise KeyError(f"Temporal split column {time_column!r} is not present in the dataset")

    values, value_type = _time_values(frame[time_column])
    if values.isna().any():
        raise ValueError(f"Temporal split column {time_column!r} contains missing or invalid values")
    unique_values = np.sort(values.unique())
    if len(unique_values) < 2:
        raise ValueError("Temporal split requires at least two distinct time values")

    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for cutoff in unique_values[:-1]:
        train_index = np.flatnonzero(values.to_numpy() <= cutoff)
        test_index = np.flatnonzero(values.to_numpy() > cutoff)
        if task == "classification" and (
            len(np.unique(y[train_index])) != 2 or len(np.unique(y[test_index])) != 2
        ):
            continue
        fraction_penalty = abs(len(test_index) / len(y) - test_size)
        balance_penalty = (
            abs(float(np.mean(y[test_index])) - float(np.mean(y)))
            if task == "classification"
            else 0.0
        )
        score = fraction_penalty + 0.25 * balance_penalty
        if best is None or score < best[0]:
            best = (score, train_index, test_index)
    if best is None:
        raise ValueError(
            "Could not create a temporal split containing both classes in train and test"
        )

    train_index, test_index = best[1], best[2]
    return train_index, test_index, {
        "time_column": str(time_column),
        "time_value_type": value_type,
        "train_time_max": str(values.iloc[train_index].max()),
        "test_time_min": str(values.iloc[test_index].min()),
    }


def _time_values(series: pd.Series) -> tuple[pd.Series, str]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric, "numeric"
    timestamps = pd.to_datetime(series, errors="coerce", utc=True)
    return timestamps, "datetime"


def _class_counts(series: pd.Series) -> dict[str, int]:
    return {str(int(label)): int(count) for label, count in series.value_counts().items()}


def _group_metadata(
    groups: np.ndarray, train_index: np.ndarray, test_index: np.ndarray
) -> dict[str, Any]:
    train_groups = set(groups[train_index])
    test_groups = set(groups[test_index])
    group_sizes = pd.Series(groups).value_counts()
    return {
        "train_group_count": len(train_groups),
        "test_group_count": len(test_groups),
        "overlapping_group_count": len(train_groups.intersection(test_groups)),
        "smallest_group_size": int(group_sizes.min()),
        "largest_group_size": int(group_sizes.max()),
    }
