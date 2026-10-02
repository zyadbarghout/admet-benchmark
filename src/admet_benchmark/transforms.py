"""Target transformations matching the OpenADMET challenge evaluation policy."""

from __future__ import annotations

import numpy as np
import pandas as pd


ALREADY_LOG_SCALED_ENDPOINTS = frozenset({"LogD"})


def transform_for_challenge(values: pd.Series, endpoint: str) -> pd.Series:
    """Apply log10 unless an endpoint is already defined on a log scale.

    The transform is fitted nowhere: it is a fixed challenge rule and is
    applied to train, validation, and test targets alike.
    """

    numeric = pd.to_numeric(values, errors="coerce")
    if endpoint in ALREADY_LOG_SCALED_ENDPOINTS:
        return numeric
    invalid = numeric.notna() & (numeric <= 0)
    if invalid.any():
        raise ValueError(
            f"{endpoint} has {int(invalid.sum())} non-positive target values; "
            "log10 evaluation is undefined for these rows."
        )
    return pd.Series(np.log10(numeric), index=values.index, name=values.name)
