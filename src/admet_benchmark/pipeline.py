"""End-to-end molecular model selection followed by one held-out evaluation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, RDLogger
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold

from .chemistry import ChemicalFilter, apply_chemical_filter
from .data import _INTERNAL_SAMPLE_ID, load_data_split_bundle
from .features import FeatureSet, build_features
from .metrics import classification_metrics, regression_metrics
from .models import model_specs
from .reporting import (
    plot_classification_diagnostics,
    plot_regression_diagnostics,
    write_json,
    write_report,
)
from .transforms import transform_for_challenge


@dataclass(frozen=True)
class ExperimentResult:
    output_dir: Path
    selected_model: str
    validation_score: float
    metrics: dict[str, float]


def _chemical_filter_from_config(config: dict[str, Any]) -> ChemicalFilter | None:
    raw_filter = config.get("chemical_filter")
    if not raw_filter or not raw_filter.get("smarts"):
        return None
    return ChemicalFilter(
        smarts=str(raw_filter["smarts"]),
        substitution=str(raw_filter.get("substitution", "any")),
    )


def _valid_smiles_mask(smiles: pd.Series) -> pd.Series:
    # RDKit otherwise prints each malformed value while the input is audited below.
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    return smiles.map(lambda value: Chem.MolFromSmiles(str(value)) is not None)


def _prepare_split(
    frame: pd.DataFrame,
    *,
    task: str,
    target: str,
    smiles_column: str,
    name_column: str | None,
    target_transform: str | None,
    chemical_filter: ChemicalFilter | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Drop unusable labels, apply the shared target policy, then filter chemistry."""

    required = {target, smiles_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Dataset is missing required column(s): {', '.join(sorted(missing))}")

    prepared = frame.copy().reset_index(names="source_row")
    prepared["sample_id"] = (
        prepared[_INTERNAL_SAMPLE_ID].astype(str)
        if _INTERNAL_SAMPLE_ID in prepared
        else prepared["source_row"].map(lambda value: f"row:{value}")
    )
    if name_column and name_column in prepared:
        names = prepared[name_column].astype("string").str.strip()
        prepared["molecule_id"] = names.where(
            names.notna() & names.ne(""), prepared["sample_id"]
        )
    else:
        prepared["molecule_id"] = prepared["sample_id"]
    prepared["target_raw"] = pd.to_numeric(prepared[target], errors="coerce")
    non_empty_smiles = prepared[smiles_column].notna() & prepared[smiles_column].astype(str).str.strip().ne("")
    labeled = prepared.loc[prepared["target_raw"].notna() & non_empty_smiles].copy()
    valid_smiles = _valid_smiles_mask(labeled[smiles_column])
    labeled = labeled.loc[valid_smiles].copy()

    if task == "classification":
        labels = set(labeled["target_raw"].unique())
        if not labels.issubset({0.0, 1.0}):
            raise ValueError(
                f"Classification target {target!r} must contain only 0/1 labels; found {sorted(labels)}"
            )
        labeled["target_model"] = labeled["target_raw"].astype(int)
    elif task == "regression":
        if target_transform in {None, "none"}:
            labeled["target_model"] = labeled["target_raw"]
        elif target_transform == "challenge":
            labeled["target_model"] = transform_for_challenge(labeled["target_raw"], target)
        else:
            raise ValueError("target_transform must be 'challenge' or 'none'")
    else:
        raise ValueError("task must be 'regression' or 'classification'")

    retained, audit = apply_chemical_filter(labeled, smiles_column, chemical_filter)
    summary = {
        "input_rows": int(len(frame)),
        "labeled_rows": int(len(labeled)),
        "invalid_or_missing_rows": int(len(frame) - len(labeled)),
        "retained_rows": int(len(retained)),
        "removed_by_chemical_filter": int(len(labeled) - len(retained)),
    }
    return retained, audit, summary


def _cross_validation(task: str, y: np.ndarray, folds: int, seed: int):
    if folds < 2:
        raise ValueError("hpo.cv_folds must be at least two")
    if task == "classification":
        counts = pd.Series(y).value_counts()
        if counts.min() < folds:
            raise ValueError(
                "Not enough examples in every class for the requested cross-validation folds"
            )
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def _inverse_transform(values: np.ndarray, target: str, target_transform: str | None) -> np.ndarray:
    if target_transform == "challenge" and target != "LogD":
        return np.power(10.0, values)
    return values


def _save_audit(audit: pd.DataFrame, output_path: Path, smiles_column: str) -> None:
    useful_columns = [
        column
        for column in [
            "source_row",
            "sample_id",
            "molecule_id",
            smiles_column,
            "target_raw",
            "target_model",
            "chemistry_valid_smiles",
            "chemistry_match_count",
            "chemistry_external_heavy_atoms",
            "chemistry_substitution_state",
            "chemistry_passes_filter",
        ]
        if column in audit
    ]
    audit.loc[:, useful_columns].to_csv(output_path, index=False)


def _search_workers(value: int | str | None) -> int:
    """Use one fewer available core by default without encoding a host size."""

    if value in {None, "auto"}:
        return max(1, (os.cpu_count() or 2) - 1)
    workers = int(value)
    if workers == 0 or workers < -1:
        raise ValueError("hpo.n_jobs must be 'auto', -1, or a positive integer")
    return workers


def run_experiment(config: dict[str, Any], output_dir: str | Path) -> ExperimentResult:
    """Run HPO/model selection on train and evaluate the winning model once on test."""

    task = config["task"]
    seed = int(config.get("seed", 42))
    dataset_config = config["dataset"]
    target = str(dataset_config["target"])
    smiles_column = str(dataset_config["smiles_column"])
    name_column = dataset_config.get("name_column")
    target_transform = dataset_config.get("target_transform", "none")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    plots = output / "plots"
    plots.mkdir(exist_ok=True)

    chemical_filter = _chemical_filter_from_config(config)
    loaded_splits = load_data_split_bundle(config)
    raw_train, raw_test = loaded_splits.train, loaded_splits.test
    train, train_audit, train_summary = _prepare_split(
        raw_train,
        task=task,
        target=target,
        smiles_column=smiles_column,
        name_column=name_column,
        target_transform=target_transform,
        chemical_filter=chemical_filter,
    )
    test, test_audit, test_summary = _prepare_split(
        raw_test,
        task=task,
        target=target,
        smiles_column=smiles_column,
        name_column=name_column,
        target_transform=target_transform,
        chemical_filter=chemical_filter,
    )
    if len(train) < 10 or len(test) < 2:
        raise ValueError("Chemical filter left too few rows for a train/test experiment")

    _save_audit(train_audit, output / "train_filter_audit.csv", smiles_column)
    _save_audit(test_audit, output / "test_filter_audit.csv", smiles_column)
    write_json(output / "split_manifest.json", loaded_splits.metadata)
    features_config = config.get("features", {})
    feature_options = {
        "radius": int(features_config.get("radius", 2)),
        "n_bits": int(features_config.get("n_bits", 1024)),
        "include_rdkit_descriptors": bool(features_config.get("include_rdkit_descriptors", True)),
    }
    train_features: FeatureSet = build_features(train[smiles_column].tolist(), **feature_options)
    test_features: FeatureSet = build_features(test[smiles_column].tolist(), **feature_options)
    write_json(
        output / "feature_schema.json",
        {"feature_count": len(train_features.names), "feature_names": train_features.names},
    )

    y_train = train["target_model"].to_numpy()
    y_test = test["target_model"].to_numpy()
    hpo_config = config.get("hpo", {})
    folds = int(hpo_config.get("cv_folds", 5))
    candidates = int(hpo_config.get("n_iter", 12))
    cv = _cross_validation(task, y_train, folds, seed)
    scoring = hpo_config.get(
        "scoring", "average_precision" if task == "classification" else "neg_mean_absolute_error"
    )
    configured_models = config.get("models")
    if not configured_models:
        raise ValueError("At least one model must be listed in models")

    candidate_tables: list[pd.DataFrame] = []
    validation_rows: list[dict[str, Any]] = []
    fitted_models: list[tuple[str, Any, float, str]] = []
    for spec in model_specs(task, list(configured_models), seed):
        search = RandomizedSearchCV(
            estimator=spec.estimator,
            param_distributions=spec.search_space,
            n_iter=candidates,
            scoring=scoring,
            cv=cv,
            random_state=seed,
            n_jobs=_search_workers(hpo_config.get("n_jobs", "auto")),
            refit=True,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(train_features.values, y_train)
        results = pd.DataFrame(search.cv_results_)
        results = results.loc[
            :, ["rank_test_score", "mean_test_score", "std_test_score", "params"]
        ].copy()
        results.insert(0, "model", spec.name)
        results.rename(
            columns={
                "rank_test_score": "validation_rank_within_model",
                "mean_test_score": "validation_score",
                "std_test_score": "validation_score_std",
            },
            inplace=True,
        )
        results["params"] = results["params"].map(
            lambda value: json.dumps(value, sort_keys=True, default=str)
        )
        candidate_tables.append(results)
        best_candidate = results.loc[
            results["validation_rank_within_model"] == 1
        ].iloc[0]
        validation_rows.append(
            {
                "model": spec.name,
                "validation_score": float(search.best_score_),
                "validation_score_std": float(best_candidate["validation_score_std"]),
                "best_parameters": json.dumps(search.best_params_, sort_keys=True, default=str),
            }
        )
        fitted_models.append((spec.name, search.best_estimator_, float(search.best_score_), json.dumps(search.best_params_, sort_keys=True, default=str)))

    hpo_candidates = pd.concat(candidate_tables, ignore_index=True).sort_values(
        ["validation_score", "model"], ascending=[False, True]
    )
    hpo_candidates.to_csv(output / "hpo_candidates.csv", index=False)
    validation_leaderboard = pd.DataFrame(validation_rows).sort_values(
        ["validation_score", "model"], ascending=[False, True]
    )
    validation_leaderboard.to_csv(output / "validation_leaderboard.csv", index=False)
    selected_name, selected_estimator, validation_score, best_parameters = max(
        fitted_models, key=lambda item: item[2]
    )

    if task == "regression":
        prediction = selected_estimator.predict(test_features.values)
        metrics = regression_metrics(y_test, prediction)
        prediction_table = test.loc[:, ["source_row", "molecule_id", smiles_column, "target_raw", "target_model"]].copy()
        prediction_table["prediction_model_scale"] = prediction
        prediction_table["prediction_original_scale"] = _inverse_transform(
            prediction, target, target_transform
        )
        prediction_table["absolute_error_model_scale"] = np.abs(y_test - prediction)
        prediction_table.to_csv(output / "held_out_predictions.csv", index=False)
        plot_regression_diagnostics(y_test, prediction, plots / "true_vs_predicted.png", target)
    else:
        probability = selected_estimator.predict_proba(test_features.values)[:, 1]
        metrics = classification_metrics(y_test, probability)
        prediction_table = test.loc[:, ["source_row", "molecule_id", smiles_column, "target_raw", "target_model"]].copy()
        prediction_table["probability"] = probability
        prediction_table["prediction"] = (probability >= 0.5).astype(int)
        prediction_table.to_csv(output / "held_out_predictions.csv", index=False)
        plot_classification_diagnostics(y_test, probability, plots / "roc_pr_curves.png", target)

    data_summary: dict[str, Any] = {
        "task": task,
        "target": target,
        "target_transform": target_transform,
        "split": loaded_splits.metadata,
        "train": train_summary,
        "test": test_summary,
        "chemical_filter": (
            {"smarts": chemical_filter.smarts, "substitution": chemical_filter.substitution}
            if chemical_filter
            else None
        ),
    }
    if task == "classification":
        data_summary["train_positive_rate"] = float(np.mean(y_train))
        data_summary["test_positive_rate"] = float(np.mean(y_test))
    write_json(output / "data_summary.json", data_summary)
    write_json(
        output / "selection.json",
        {
            "selected_model": selected_name,
            "validation_score": validation_score,
            "scoring": scoring,
            "best_parameters": json.loads(best_parameters),
            "test_metrics": metrics,
        },
    )
    resolved_config = {key: value for key, value in config.items() if key != "_config_dir"}
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(resolved_config, sort_keys=False))
    write_report(
        output,
        task=task,
        target=target,
        selected_model=selected_name,
        validation_score=validation_score,
        metrics=metrics,
        data_summary={
            "train_rows": train_summary["retained_rows"],
            "test_rows": test_summary["retained_rows"],
            "target_transform": target_transform,
            "chemical_filter": data_summary["chemical_filter"],
        },
    )
    return ExperimentResult(
        output_dir=output,
        selected_model=selected_name,
        validation_score=validation_score,
        metrics=metrics,
    )
