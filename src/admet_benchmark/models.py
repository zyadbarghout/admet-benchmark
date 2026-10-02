"""Model registry and small, explicit HPO search spaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _pipeline(model, *, scale: bool = False) -> Pipeline:
    """Impute descriptors inside each CV fold, then fit the estimator."""

    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Pipeline
    search_space: dict[str, list[object]]


def model_specs(task: str, names: list[str], seed: int) -> list[ModelSpec]:
    """Build supported model/search-space pairs without nested parallelism."""

    registry: dict[str, ModelSpec]
    if task == "regression":
        registry = {
            "extra_trees": ModelSpec(
                "extra_trees",
                _pipeline(ExtraTreesRegressor(random_state=seed, n_jobs=1)),
                {
                    "model__n_estimators": [200, 350],
                    "model__max_depth": [None, 12, 24],
                    "model__min_samples_leaf": [1, 2, 4],
                    "model__max_features": ["sqrt", 0.5, 1.0],
                },
            ),
            "random_forest": ModelSpec(
                "random_forest",
                _pipeline(RandomForestRegressor(random_state=seed, n_jobs=1)),
                {
                    "model__n_estimators": [200, 350],
                    "model__max_depth": [None, 12, 24],
                    "model__min_samples_leaf": [1, 2, 4],
                    "model__max_features": ["sqrt", 0.5, 1.0],
                },
            ),
            "ridge": ModelSpec(
                "ridge",
                _pipeline(Ridge(solver="lsqr"), scale=True),
                {"model__alpha": list(np.logspace(-3, 3, 13))},
            ),
        }
    elif task == "classification":
        registry = {
            "extra_trees": ModelSpec(
                "extra_trees",
                _pipeline(ExtraTreesClassifier(random_state=seed, n_jobs=1)),
                {
                    "model__n_estimators": [200, 350],
                    "model__max_depth": [None, 12, 24],
                    "model__min_samples_leaf": [1, 2, 4],
                    "model__max_features": ["sqrt", 0.5, 1.0],
                    "model__class_weight": [None, "balanced"],
                },
            ),
            "random_forest": ModelSpec(
                "random_forest",
                _pipeline(RandomForestClassifier(random_state=seed, n_jobs=1)),
                {
                    "model__n_estimators": [200, 350],
                    "model__max_depth": [None, 12, 24],
                    "model__min_samples_leaf": [1, 2, 4],
                    "model__max_features": ["sqrt", 0.5, 1.0],
                    "model__class_weight": [None, "balanced"],
                },
            ),
            "logistic_regression": ModelSpec(
                "logistic_regression",
                _pipeline(
                    LogisticRegression(
                        solver="liblinear", max_iter=2_000, random_state=seed
                    ),
                    scale=True,
                ),
                {
                    "model__C": list(np.logspace(-3, 3, 13)),
                    "model__class_weight": [None, "balanced"],
                },
            ),
        }
    else:
        raise ValueError("task must be 'regression' or 'classification'")

    if "xgboost" in names:
        registry["xgboost"] = _xgboost_spec(task, seed)
    if "lightgbm" in names:
        registry["lightgbm"] = _lightgbm_spec(task, seed)

    unknown = sorted(set(names).difference(registry))
    if unknown:
        raise ValueError(f"Unsupported {task} model(s): {', '.join(unknown)}")
    return [registry[name] for name in names]


def _xgboost_spec(task: str, seed: int) -> ModelSpec:
    """Construct the XGBoost baseline only when it is selected."""

    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "xgboost is required for the xgboost model. Install admet-benchmark[models]."
        ) from error

    common = {
        "n_estimators": 300,
        "tree_method": "hist",
        "n_jobs": 1,
        "random_state": seed,
        "verbosity": 0,
    }
    estimator = (
        XGBRegressor(objective="reg:absoluteerror", **common)
        if task == "regression"
        else XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    )
    return ModelSpec(
        "xgboost",
        _pipeline(estimator),
        {
            "model__n_estimators": [200, 350, 500],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.03, 0.06, 0.1],
            "model__min_child_weight": [1, 3, 6],
            "model__subsample": [0.7, 0.9, 1.0],
            "model__colsample_bytree": [0.5, 0.8, 1.0],
        },
    )


def _lightgbm_spec(task: str, seed: int) -> ModelSpec:
    """Construct the LightGBM baseline only when it is selected."""

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "lightgbm is required for the lightgbm model. Install admet-benchmark[models]."
        ) from error

    common = {"n_estimators": 300, "n_jobs": 1, "random_state": seed, "verbosity": -1}
    estimator = LGBMRegressor(**common) if task == "regression" else LGBMClassifier(**common)
    return ModelSpec(
        "lightgbm",
        _pipeline(estimator),
        {
            "model__n_estimators": [200, 350, 500],
            "model__num_leaves": [15, 31, 63],
            "model__learning_rate": [0.03, 0.06, 0.1],
            "model__min_child_samples": [10, 20, 40],
            "model__subsample": [0.7, 0.9, 1.0],
            "model__colsample_bytree": [0.5, 0.8, 1.0],
        },
    )
