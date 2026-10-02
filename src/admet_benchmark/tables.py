"""CSV boundaries preserve identifiers before pandas performs type inference."""
from pathlib import Path

import pandas as pd


IDENTIFIER_COLUMNS = (
    "sample_id", "molecule_id", "source_id", "original_sample_id", "smiles",
    "canonical_smiles", "chemical_id", "endpoint", "model", "task", "partition",
    "preprocessing", "split_strategy", "nearest_train_id", "metric",
)


def read_csv(path: str | Path, *, text_columns=IDENTIFIER_COLUMNS, **kwargs):
    # Converters preserve both leading zeros and literal NA-like identifiers;
    # numeric columns retain normal missing-value parsing.
    converters = {name: str for name in text_columns}
    converters.update(kwargs.pop("converters", {}))
    return pd.read_csv(path, converters=converters, **kwargs)
