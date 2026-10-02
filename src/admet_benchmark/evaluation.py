"""Evaluate prepared molecular predictions without training a model.

The public input is deliberately smaller than the training configuration. A
caller maps their CSV columns to a stable prediction record, declares the
evaluation scale, and receives canonical predictions, fold metrics, pairing
audits, statistical comparisons, and a manifest.
"""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, RDLogger
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from .config import resolve_config_path
from .metrics import classification_metrics, regression_metrics
from .reporting import write_json
from .tables import read_csv


SCHEMA_VERSION = 1
IDENTITY_COLUMNS = [
    "task",
    "endpoint",
    "preprocessing",
    "split_strategy",
    "partition",
    "repeat",
    "fold",
]
PREDICTION_KEY = [*IDENTITY_COLUMNS, "model", "sample_id"]
EVALUATED_PARTITIONS = {"validation", "test"}
ALLOWED_PARTITIONS = {*EVALUATED_PARTITIONS, "train"}
OPTIONAL_PREDICTION_FIELDS = {
    "uncertainty": "uncertainty",
    "prediction_lower": "prediction_lower",
    "prediction_upper": "prediction_upper",
}
COMPARISON_COLUMNS = [
    "task",
    "endpoint",
    "preprocessing",
    "split_strategy",
    "partition",
    "metric",
    "model_a",
    "model_b",
    "paired_results",
    "median_delta_a_minus_b",
    "mean_delta_a_minus_b",
    "friedman_p_value",
    "pairwise_test",
    "unadjusted_p_value",
    "holm_adjusted_p_value",
    "inference_status",
    "inference_caveat",
]


@dataclass(frozen=True)
class EvaluationResult:
    output_dir: Path
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame
    validation: dict[str, Any]


def validate_config_schema(config: dict[str, Any]) -> None:
    """Validate the public YAML contract before reading prediction data."""

    from jsonschema import Draft202012Validator

    schema_path = files("admet_benchmark").joinpath(
        "schemas/evaluation-config.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    public_config = {key: value for key, value in config.items() if key != "_config_dir"}
    errors = sorted(
        Draft202012Validator(schema).iter_errors(public_config),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        messages = []
        for error in errors:
            location = ".".join(str(part) for part in error.absolute_path) or "config"
            messages.append(f"{location}: {error.message}")
        raise ValueError("Invalid evaluation configuration:\n- " + "\n- ".join(messages))


def _paths(value: Any) -> list[str]:
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return value
    raise ValueError("predictions.path must be a path or a non-empty list of paths")


def _read_prediction_files(config: dict[str, Any]) -> tuple[pd.DataFrame, list[Path]]:
    prediction_config = config.get("predictions")
    if not isinstance(prediction_config, dict):
        raise ValueError("Set predictions to a mapping containing path and columns")

    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for value in _paths(prediction_config.get("path")):
        path = resolve_config_path(value, config).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Prediction file does not exist: {path}")
        suffix = path.suffix.lower()
        text_fields = {"sample_id", "molecule_id", "smiles", "endpoint", "model",
                       "partition", "split_strategy", "preprocessing"}
        text_columns = [source for name, source in prediction_config.get("columns", {}).items()
                        if name in text_fields]
        if suffix == ".csv":
            frame = read_csv(path, text_columns=text_columns)
        elif suffix in {".tsv", ".txt"}:
            frame = read_csv(path, text_columns=text_columns, sep="\t")
        elif suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported prediction format {suffix!r}: {path}")
        frame["source_file"] = path.name
        frames.append(frame)
        paths.append(path)
    return pd.concat(frames, ignore_index=True), paths


def _column_or_default(
    frame: pd.DataFrame,
    columns: dict[str, str],
    defaults: dict[str, Any],
    name: str,
    *,
    required: bool = True,
) -> pd.Series:
    source = columns.get(name)
    if source:
        if source not in frame:
            raise KeyError(f"Configured {name!r} column {source!r} is missing")
        return frame[source]
    if name in defaults:
        return pd.Series(defaults[name], index=frame.index)
    if required:
        raise ValueError(
            f"Map predictions.columns.{name} or set predictions.defaults.{name}"
        )
    return pd.Series(pd.NA, index=frame.index)


def canonicalize_predictions(config: dict[str, Any]) -> tuple[pd.DataFrame, list[Path]]:
    """Map user columns to one versioned, long-form prediction table."""

    source, paths = _read_prediction_files(config)
    settings = config["predictions"]
    columns = dict(settings.get("columns") or {})
    defaults = dict(settings.get("defaults") or {})
    task = str(config.get("task", "")).lower()
    if task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression' or 'classification'")

    base = pd.DataFrame(index=source.index)
    base["molecule_id"] = _column_or_default(source, columns, defaults, "molecule_id")
    if columns.get("sample_id") or "sample_id" in defaults:
        base["sample_id"] = _column_or_default(
            source, columns, defaults, "sample_id"
        )
    else:
        # Simple datasets often use one ID for both the chemical and the
        # measurement. Replicate-containing data should map sample_id.
        base["sample_id"] = base["molecule_id"]
    base["smiles"] = _column_or_default(source, columns, defaults, "smiles")
    base["endpoint"] = _column_or_default(source, columns, defaults, "endpoint")
    base["partition"] = _column_or_default(source, columns, defaults, "partition")
    base["observed_submitted"] = _column_or_default(source, columns, defaults, "observed")
    for name, default in (
        ("repeat", 0),
        ("fold", 0),
        ("split_strategy", "provided"),
        ("preprocessing", "provided"),
    ):
        local_defaults = {name: defaults.get(name, default)}
        base[name] = _column_or_default(
            source, columns, local_defaults, name, required=False
        )
    base["source_file"] = source["source_file"]
    base["task"] = task

    prediction_columns = settings.get("prediction_columns")
    if prediction_columns:
        if not isinstance(prediction_columns, dict):
            raise ValueError("predictions.prediction_columns must map model names to columns")
        pieces: list[pd.DataFrame] = []
        for model, prediction_column in prediction_columns.items():
            if prediction_column not in source:
                raise KeyError(
                    f"Prediction column {prediction_column!r} for model {model!r} is missing"
                )
            piece = base.copy()
            piece["model"] = str(model)
            piece["prediction_submitted"] = source[prediction_column]
            for semantic_name, canonical_name in OPTIONAL_PREDICTION_FIELDS.items():
                model_columns = settings.get(f"{semantic_name}_columns") or {}
                common_column = columns.get(semantic_name)
                selected_column = model_columns.get(model, common_column)
                piece[f"{canonical_name}_submitted"] = (
                    source[selected_column]
                    if selected_column
                    else pd.Series(pd.NA, index=source.index)
                )
            pieces.append(piece)
        canonical = pd.concat(pieces, ignore_index=True)
    else:
        canonical = base.copy()
        canonical["model"] = _column_or_default(source, columns, defaults, "model")
        canonical["prediction_submitted"] = _column_or_default(
            source, columns, defaults, "prediction"
        )
        for semantic_name, canonical_name in OPTIONAL_PREDICTION_FIELDS.items():
            canonical[f"{canonical_name}_submitted"] = _column_or_default(
                source, columns, defaults, semantic_name, required=False
            )

    for name in [
        "molecule_id",
        "sample_id",
        "smiles",
        "endpoint",
        "model",
        "partition",
        "split_strategy",
        "preprocessing",
    ]:
        canonical[name] = canonical[name].astype("string").str.strip()
    canonical["partition"] = canonical["partition"].str.lower()
    canonical["repeat"] = pd.to_numeric(canonical["repeat"], errors="coerce")
    canonical["fold"] = pd.to_numeric(canonical["fold"], errors="coerce")
    canonical["observed_submitted"] = pd.to_numeric(
        canonical["observed_submitted"], errors="coerce"
    )
    canonical["prediction_submitted"] = pd.to_numeric(
        canonical["prediction_submitted"], errors="coerce"
    )
    for canonical_name in OPTIONAL_PREDICTION_FIELDS.values():
        column = f"{canonical_name}_submitted"
        canonical[column] = pd.to_numeric(canonical[column], errors="coerce")
    return canonical.reset_index(drop=True), paths


def _endpoint_setting(
    settings: dict[str, Any], endpoint: str, *, required: bool = False
) -> Any:
    endpoints = settings.get("endpoints") or {}
    if endpoint in endpoints:
        return endpoints[endpoint]
    if "default" in settings:
        return settings["default"]
    if required:
        raise ValueError(f"No setting was declared for endpoint {endpoint!r}")
    return None


def _apply_target_transforms(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    settings = dict(config.get("target_transform") or {"default": "none"})
    transformed = frame.copy()
    transformed["observed"] = transformed["observed_submitted"]
    transformed["prediction"] = transformed["prediction_submitted"]
    transformed["uncertainty"] = transformed["uncertainty_submitted"]
    transformed["prediction_lower"] = transformed["prediction_lower_submitted"]
    transformed["prediction_upper"] = transformed["prediction_upper_submitted"]
    transformed["target_transform"] = "none"

    for endpoint in transformed["endpoint"].dropna().unique():
        policy = str(_endpoint_setting(settings, str(endpoint)) or "none").lower()
        if policy == "challenge":
            policy = "none" if str(endpoint).strip().lower() == "logd" else "log10"
        if policy not in {"none", "log10"}:
            raise ValueError(
                f"Unsupported target transform {policy!r} for endpoint {endpoint!r}"
            )
        mask = transformed["endpoint"] == endpoint
        transformed.loc[mask, "target_transform"] = policy
        if policy == "log10":
            interval_columns = [
                column
                for column in ("prediction_lower", "prediction_upper")
                if transformed.loc[mask, column].notna().any()
            ]
            values = transformed.loc[mask, ["observed", "prediction", *interval_columns]]
            invalid = values.notna() & values.le(0)
            if invalid.any().any():
                counts = invalid.sum().to_dict()
                raise ValueError(
                    f"log10 for endpoint {endpoint!r} requires positive observed and "
                    f"predicted values; invalid counts: {counts}"
                )
            transformed.loc[mask, ["observed", "prediction"]] = np.log10(
                values[["observed", "prediction"]]
            )
            if interval_columns:
                transformed.loc[mask, interval_columns] = np.log10(
                    values[interval_columns]
                )
    return transformed


def _thresholds(config: dict[str, Any], endpoints: Iterable[str]) -> dict[str, float]:
    if str(config["task"]).lower() != "classification":
        return {}
    settings = config.get("thresholds")
    if not isinstance(settings, dict):
        raise ValueError(
            "Classification evaluation requires thresholds.default or thresholds.endpoints"
        )
    resolved: dict[str, float] = {}
    for endpoint in endpoints:
        value = float(_endpoint_setting(settings, endpoint, required=True))
        if not 0 <= value <= 1:
            raise ValueError(f"Threshold for endpoint {endpoint!r} must be between 0 and 1")
        resolved[endpoint] = value
    return resolved


def validate_predictions(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run semantic checks and return transformed records plus an audit."""

    frame = frame.copy()
    for canonical_name in OPTIONAL_PREDICTION_FIELDS.values():
        column = f"{canonical_name}_submitted"
        if column not in frame:
            frame[column] = pd.NA
    errors: list[str] = []
    warnings: list[str] = []
    prediction_config = dict(config.get("predictions") or {})
    mapped_columns = dict(prediction_config.get("columns") or {})
    defaults = dict(prediction_config.get("defaults") or {})
    if not mapped_columns.get("sample_id") and "sample_id" not in defaults:
        warnings.append(
            "sample_id was not mapped; molecule_id is assumed to identify each observation"
        )
    for column in [
        "molecule_id",
        "sample_id",
        "smiles",
        "endpoint",
        "model",
        "partition",
        "repeat",
        "fold",
        "observed_submitted",
        "prediction_submitted",
    ]:
        missing = int(frame[column].isna().sum())
        if missing:
            errors.append(f"{column} contains {missing} missing value(s)")
    for column in [
        "molecule_id",
        "sample_id",
        "smiles",
        "endpoint",
        "model",
        "partition",
        "split_strategy",
        "preprocessing",
    ]:
        empty = int(frame[column].fillna("").astype(str).str.strip().eq("").sum())
        if empty:
            errors.append(f"{column} contains {empty} empty value(s)")

    invalid_partitions = sorted(set(frame["partition"].dropna()) - ALLOWED_PARTITIONS)
    if invalid_partitions:
        errors.append(
            "partition contains unsupported values: " + ", ".join(invalid_partitions)
        )
    if not set(frame["partition"].dropna()).intersection(EVALUATED_PARTITIONS):
        errors.append("At least one validation or test prediction is required")

    for column in ("repeat", "fold"):
        invalid = frame[column].notna() & (~np.isfinite(frame[column]) | frame[column].mod(1).ne(0) | frame[column].lt(0))
        if invalid.any():
            errors.append("repeat and fold must be finite nonnegative integers")
            frame.loc[invalid, column] = pd.NA
        frame[column] = frame[column].astype("Int64")

    duplicates = frame.duplicated(PREDICTION_KEY, keep=False)
    if duplicates.any():
        errors.append(
            f"{int(duplicates.sum())} rows duplicate the canonical prediction key"
        )

    context = [*IDENTITY_COLUMNS, "sample_id"]
    truth_counts = frame.groupby(context, dropna=False)["observed_submitted"].nunique()
    inconsistent_truth = int(truth_counts.gt(1).sum())
    if inconsistent_truth:
        errors.append(
            f"Observed values disagree across models for {inconsistent_truth} molecule context(s)"
        )

    # A sample may occur in different CV folds, but not in validation and final
    # test for the same experiment/repeat/fold context.
    leakage_context = [
        "task",
        "endpoint",
        "preprocessing",
        "split_strategy",
        "repeat",
        "fold",
        "sample_id",
    ]
    partition_counts = frame.groupby(leakage_context, dropna=False)["partition"].agg(
        lambda values: len(set(values).intersection(EVALUATED_PARTITIONS))
    )
    overlapping_partitions = int(partition_counts.gt(1).sum())
    if overlapping_partitions:
        errors.append(
            f"{overlapping_partitions} sample context(s) occur in both validation and test"
        )

    task = str(config["task"]).lower()
    if task == "regression" and not isinstance(config.get("target_transform"), dict):
        errors.append(
            "Regression evaluation requires an explicit target_transform mapping"
        )
    if task == "classification":
        labels = set(frame["observed_submitted"].dropna().unique())
        if not labels.issubset({0.0, 1.0}):
            errors.append(f"Classification observed values must be 0/1; found {sorted(labels)}")
        probabilities = frame["prediction_submitted"].dropna()
        if probabilities.lt(0).any() or probabilities.gt(1).any():
            errors.append("Classification predictions must be probabilities in [0, 1]")

    finite_columns = ["observed_submitted", "prediction_submitted"]
    for column in finite_columns:
        values = frame[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            errors.append(f"{column} contains non-finite values")

    optional_numeric = [
        "uncertainty_submitted",
        "prediction_lower_submitted",
        "prediction_upper_submitted",
    ]
    for column in optional_numeric:
        values = frame[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            errors.append(f"{column} contains non-finite values")
    for semantic_name, canonical_name in OPTIONAL_PREDICTION_FIELDS.items():
        common_mapping = mapped_columns.get(semantic_name)
        model_mappings = prediction_config.get(f"{semantic_name}_columns") or {}
        if common_mapping:
            missing = int(frame[f"{canonical_name}_submitted"].isna().sum())
            if missing:
                errors.append(
                    f"Mapped {semantic_name} contains {missing} missing or non-numeric value(s)"
                )
        for model in model_mappings:
            mask = frame["model"].astype(str).eq(str(model))
            missing = int(frame.loc[mask, f"{canonical_name}_submitted"].isna().sum())
            if missing:
                errors.append(
                    f"Mapped {semantic_name} for model {model!r} contains "
                    f"{missing} missing or non-numeric value(s)"
                )
    uncertainty = frame["uncertainty_submitted"].dropna()
    if uncertainty.lt(0).any():
        errors.append("uncertainty values must be non-negative")

    lower_present = frame["prediction_lower_submitted"].notna()
    upper_present = frame["prediction_upper_submitted"].notna()
    incomplete_intervals = lower_present ^ upper_present
    if incomplete_intervals.any():
        errors.append(
            f"{int(incomplete_intervals.sum())} row(s) provide only one prediction interval bound"
        )
    complete_intervals = lower_present & upper_present
    if uncertainty.notna().any() or complete_intervals.any():
        if not isinstance(config.get("uncertainty"), dict):
            errors.append(
                "UQ values require an uncertainty mapping that declares score direction and scale"
            )
    if complete_intervals.any() and "interval_confidence" not in dict(
        config.get("uncertainty") or {}
    ):
        errors.append(
            "Prediction intervals require uncertainty.interval_confidence"
        )
    if (
        frame.loc[complete_intervals, "prediction_lower_submitted"]
        > frame.loc[complete_intervals, "prediction_upper_submitted"]
    ).any():
        errors.append("prediction_lower cannot exceed prediction_upper")

    uncertainty_group_columns = [*IDENTITY_COLUMNS, "model"]
    for keys, group in frame.groupby(uncertainty_group_columns, dropna=False):
        supplied = group["uncertainty_submitted"].notna()
        if supplied.any() and not supplied.all():
            errors.append(
                "Uncertainty coverage must be complete within each model/fold group; "
                f"invalid group: {dict(zip(uncertainty_group_columns, keys, strict=True))}"
            )
            break

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    unique_smiles = frame["smiles"].dropna().astype(str).unique()
    canonical_lookup: dict[str, str | None] = {}
    for smiles in unique_smiles:
        molecule = Chem.MolFromSmiles(smiles)
        canonical_lookup[smiles] = (
            Chem.MolToSmiles(molecule, isomericSmiles=True) if molecule else None
        )
    frame["canonical_smiles"] = frame["smiles"].astype(str).map(canonical_lookup)
    frame["chemical_id"] = frame["canonical_smiles"].map(
        lambda value: hashlib.sha256(value.encode()).hexdigest() if value else None
    )
    invalid_smiles = int(frame["canonical_smiles"].isna().sum())
    if invalid_smiles:
        errors.append(f"smiles contains {invalid_smiles} invalid structure(s)")

    identity_context = [*IDENTITY_COLUMNS, "sample_id"]
    for column in ("molecule_id", "canonical_smiles"):
        conflicts = frame.groupby(identity_context, dropna=False)[column].nunique().gt(1)
        if conflicts.any():
            errors.append(
                f"{column} disagrees across models for {int(conflicts.sum())} sample context(s)"
            )

    # A bundle represents one experiment per task/endpoint. Training and
    # validation may exchange chemistry across CV folds, never within a fold.
    # A locked final test must remain isolated across all development contexts.
    for identity in ("sample_id", "canonical_smiles"):
        for _, group in frame.groupby(["task", "endpoint", "preprocessing", "split_strategy", "repeat", "fold"], dropna=False):
            fitting = set(group.loc[group.partition.eq("train"), identity].dropna())
            evaluated = set(group.loc[group.partition.isin(EVALUATED_PARTITIONS), identity].dropna())
            if fitting & evaluated:
                errors.append(f"Training/evaluation leakage: {identity} overlaps within a fold")
                break
        for _, group in frame.groupby(["task", "endpoint"], dropna=False):
            test = set(group.loc[group.partition.eq("test"), identity].dropna())
            development = set(group.loc[~group.partition.eq("test"), identity].dropna())
            if test & development:
                errors.append(f"Locked-test leakage: {identity} overlaps development across folds or contexts")
                break
    identity_counts = frame.groupby(["task", "endpoint", "molecule_id"], dropna=False).canonical_smiles.nunique()
    if identity_counts.gt(1).any():
        errors.append("A molecule_id identifies multiple canonical structures within an endpoint")
    aliases = frame.groupby(["task", "endpoint", "canonical_smiles"], dropna=False).molecule_id.nunique()
    if aliases.gt(1).any():
        warnings.append("Multiple molecule IDs identify the same chemistry; bootstrap and molecule counts use canonical chemical identity")

    if errors:
        return frame, {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "errors": errors,
            "warnings": warnings,
            "rows": int(len(frame)),
        }

    chemical_context = [
        "task",
        "endpoint",
        "preprocessing",
        "split_strategy",
        "repeat",
        "fold",
    ]
    chemical_overlap = 0
    for _, group in frame.groupby(chemical_context, dropna=False):
        validation_structures = set(
            group.loc[group["partition"] == "validation", "canonical_smiles"]
        )
        test_structures = set(group.loc[group["partition"] == "test", "canonical_smiles"])
        chemical_overlap += len(validation_structures.intersection(test_structures))
    if chemical_overlap:
        warnings.append(
            f"{chemical_overlap} exact canonical structure overlap(s) occur between validation "
            "and test; inspect whether these are expected replicates or leakage"
        )

    endpoint_context = [
        "task",
        "preprocessing",
        "split_strategy",
        "partition",
        "repeat",
        "fold",
    ]
    incomplete_endpoint_contexts = 0
    for _, group in frame.groupby(endpoint_context, dropna=False):
        endpoint_sets = [
            set(values["endpoint"].astype(str))
            for _, values in group.groupby("model", sort=True)
        ]
        if endpoint_sets and any(values != endpoint_sets[0] for values in endpoint_sets[1:]):
            incomplete_endpoint_contexts += 1
    if incomplete_endpoint_contexts:
        warnings.append(
            f"{incomplete_endpoint_contexts} fold context(s) have different endpoint coverage "
            "between models; incomplete contexts do not receive MA-RAE"
        )

    try:
        transformed = _apply_target_transforms(frame, config)
        thresholds = _thresholds(config, transformed["endpoint"].astype(str).unique())
    except (KeyError, TypeError, ValueError) as error:
        errors.append(str(error))
        return frame, {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "errors": errors,
            "warnings": warnings,
            "rows": int(len(frame)),
        }
    if task == "classification":
        transformed["threshold"] = transformed["endpoint"].map(thresholds)
        transformed["predicted_class"] = (
            transformed["prediction"] >= transformed["threshold"]
        ).astype(int)
        transformed["is_error"] = (
            transformed["predicted_class"] != transformed["observed"].astype(int)
        )
    else:
        transformed["residual"] = transformed["prediction"] - transformed["observed"]
        transformed["absolute_error"] = transformed["residual"].abs()
    if task == "classification":
        transformed["residual"] = transformed["prediction"] - transformed["observed"]
        transformed["absolute_error"] = transformed["residual"].abs()

    transformed["uncertainty_percentile"] = np.nan
    uncertainty_context = [*IDENTITY_COLUMNS, "model"]
    for _, group in transformed.groupby(uncertainty_context, dropna=False):
        values = group["uncertainty"]
        if values.notna().all():
            transformed.loc[group.index, "uncertainty_percentile"] = values.rank(
                method="average", pct=True
            )

    coverage = _coverage_audit(transformed)
    incomplete = coverage["missing_from_union"].gt(0)
    if incomplete.any():
        warnings.append(
            f"{int(incomplete.sum())} model/fold group(s) do not cover the full molecule union; "
            "incomplete fold contexts are omitted from inferential comparisons"
        )
    return transformed, {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_with_warnings" if warnings else "passed",
        "errors": [],
        "warnings": warnings,
        "rows": int(len(transformed)),
        "training_leakage_check": "checked_supplied_train_roles" if frame.partition.eq("train").any() else "unverified_no_training_assignments",
        "chemical_identity_policy": "canonical_isomeric_smiles; submitted IDs preserved",
        "models": sorted(transformed["model"].astype(str).unique()),
        "endpoints": sorted(transformed["endpoint"].astype(str).unique()),
        "partitions": sorted(transformed["partition"].astype(str).unique()),
        "uncertainty_models": sorted(
            transformed.loc[transformed["uncertainty"].notna(), "model"]
            .astype(str)
            .unique()
        ),
    }


def _coverage_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(IDENTITY_COLUMNS, dropna=False, sort=True):
        model_sets = {
            str(model): set(values["sample_id"].astype(str))
            for model, values in group.groupby("model", sort=True)
        }
        union = set().union(*model_sets.values()) if model_sets else set()
        intersection = set.intersection(*model_sets.values()) if model_sets else set()
        context = dict(zip(IDENTITY_COLUMNS, keys, strict=True))
        for model, molecules in model_sets.items():
            rows.append(
                {
                    **context,
                    "model": model,
                    "prediction_count": len(molecules),
                    "molecule_union_count": len(union),
                    "shared_molecule_count": len(intersection),
                    "missing_from_union": len(union - molecules),
                }
            )
    return pd.DataFrame(rows)


def compute_fold_metrics(
    frame: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    task = str(config["task"]).lower()
    evaluated = frame[frame["partition"].isin(EVALUATED_PARTITIONS)]
    group_columns = [*IDENTITY_COLUMNS, "model"]
    statistics = dict(config.get("statistics") or {})
    bootstrap_iterations = int(statistics.get("bootstrap_iterations", 1_000))
    confidence_level = float(statistics.get("confidence_level", 0.95))
    interval_confidence = dict(config.get("uncertainty") or {}).get(
        "interval_confidence"
    )
    seed = int(statistics.get("seed", config.get("seed", 42)))
    default_bootstrap_metrics = (
        ["roc_auc", "pr_auc", "mcc"]
        if task == "classification"
        else ["mae", "relative_absolute_error"]
    )
    bootstrap_metrics = set(
        statistics.get("bootstrap_metrics", default_bootstrap_metrics)
    )
    if bootstrap_iterations < 0:
        raise ValueError("statistics.bootstrap_iterations cannot be negative")
    if not 0 < confidence_level < 1:
        raise ValueError("statistics.confidence_level must be between 0 and 1")
    for keys, group in evaluated.groupby(group_columns, dropna=False, sort=True):
        y_true = group["observed"].to_numpy(dtype=float)
        y_pred = group["prediction"].to_numpy(dtype=float)
        molecule_ids = group["chemical_id"].astype(str).to_numpy()
        if task == "classification" and len(np.unique(y_true)) < 2:
            raise ValueError(
                "Every classification evaluation group must contain both classes; "
                f"invalid group: {dict(zip(group_columns, keys, strict=True))}"
            )
        threshold = float(group["threshold"].iloc[0]) if task == "classification" else None
        metrics = (
            classification_metrics(y_true, y_pred, threshold=threshold)
            if task == "classification"
            else regression_metrics(y_true, y_pred)
        )
        uncertainty_values: np.ndarray | None = None
        lower_values: np.ndarray | None = None
        upper_values: np.ndarray | None = None
        if group["uncertainty"].notna().all():
            uncertainty_values = group["uncertainty"].to_numpy(dtype=float)
            metrics.update(
                _uncertainty_metrics(y_true, y_pred, uncertainty_values)
            )
        if group["prediction_lower"].notna().all() and group[
            "prediction_upper"
        ].notna().all():
            lower_values = group["prediction_lower"].to_numpy(dtype=float)
            upper_values = group["prediction_upper"].to_numpy(dtype=float)
            metrics.update(_interval_metrics(y_true, lower_values, upper_values))
        selected_bootstrap_metrics = set(bootstrap_metrics)
        if uncertainty_values is not None:
            selected_bootstrap_metrics.update(
                {"uncertainty_error_spearman", "uncertainty_error_kendall"}
            )
        if lower_values is not None and upper_values is not None:
            selected_bootstrap_metrics.update(
                {"interval_coverage", "median_interval_width"}
            )
        group_seed = int.from_bytes(
            hashlib.sha256(repr((seed, keys)).encode()).digest()[:8], "little"
        )
        intervals = _bootstrap_metric_intervals(
            task=task,
            y_true=y_true,
            y_pred=y_pred,
            molecule_ids=molecule_ids,
            threshold=threshold,
            uncertainty=uncertainty_values,
            prediction_lower=lower_values,
            prediction_upper=upper_values,
            iterations=bootstrap_iterations,
            confidence_level=confidence_level,
            seed=group_seed,
            selected_metrics=selected_bootstrap_metrics,
        )
        context = dict(zip(group_columns, keys, strict=True))
        for metric, value in metrics.items():
            lower, upper = intervals.get(metric, (np.nan, np.nan))
            rows.append(
                {
                    **context,
                    "metric": metric,
                    "value": value,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "confidence_level": confidence_level,
                    "bootstrap_iterations": bootstrap_iterations,
                    "n_samples": len(group),
                    "n_molecules": int(group["chemical_id"].nunique()),
                    "prediction_interval_confidence": interval_confidence,
                    "decision_threshold": threshold,
                }
            )
    result = pd.DataFrame(rows)
    if task == "regression":
        result = _append_macro_relative_error(result)
    return result


def _uncertainty_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, uncertainty: np.ndarray
) -> dict[str, float]:
    """Measure whether larger uncertainty identifies larger prediction errors."""

    from scipy.stats import kendalltau, spearmanr

    absolute_error = np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float))
    uncertainty = np.asarray(uncertainty, dtype=float)
    if len(uncertainty) < 2 or len(np.unique(uncertainty)) < 2:
        return {
            "uncertainty_error_spearman": float("nan"),
            "uncertainty_error_kendall": float("nan"),
        }
    return {
        "uncertainty_error_spearman": float(
            spearmanr(absolute_error, uncertainty).statistic
        ),
        "uncertainty_error_kendall": float(
            kendalltau(absolute_error, uncertainty).statistic
        ),
    }


def _interval_metrics(
    y_true: np.ndarray, prediction_lower: np.ndarray, prediction_upper: np.ndarray
) -> dict[str, float]:
    """Describe empirical prediction-interval coverage without assuming a method."""

    truth = np.asarray(y_true, dtype=float)
    lower = np.asarray(prediction_lower, dtype=float)
    upper = np.asarray(prediction_upper, dtype=float)
    covered = (truth >= lower) & (truth <= upper)
    return {
        "interval_coverage": float(np.mean(covered)),
        "median_interval_width": float(np.median(upper - lower)),
    }


def _append_macro_relative_error(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add challenge-style macro RAE for complete multi-endpoint contexts."""

    relative = metrics[metrics["metric"] == "relative_absolute_error"]
    if relative["endpoint"].nunique() < 2:
        return metrics
    group_columns = [
        "task",
        "preprocessing",
        "split_strategy",
        "partition",
        "repeat",
        "fold",
        "model",
    ]
    expected_endpoints = set(relative["endpoint"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for keys, group in relative.groupby(group_columns, dropna=False, sort=True):
        present = set(group["endpoint"].astype(str))
        if present != expected_endpoints:
            continue
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "endpoint": "__macro__",
                "metric": "ma_rae",
                "value": float(group["value"].mean()),
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "confidence_level": float(group["confidence_level"].iloc[0]),
                "bootstrap_iterations": int(group["bootstrap_iterations"].iloc[0]),
                "n_samples": int(group["n_samples"].sum()),
                "n_molecules": int(group["n_molecules"].sum()),
                "prediction_interval_confidence": np.nan,
                "n_endpoints": len(expected_endpoints),
                "decision_threshold": np.nan,
            }
        )
    return pd.concat([metrics, pd.DataFrame(rows)], ignore_index=True, sort=False)


def _bootstrap_metric_intervals(
    *,
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    molecule_ids: np.ndarray,
    threshold: float | None,
    uncertainty: np.ndarray | None,
    prediction_lower: np.ndarray | None,
    prediction_upper: np.ndarray | None,
    iterations: int,
    confidence_level: float,
    seed: int,
    selected_metrics: set[str],
) -> dict[str, tuple[float, float]]:
    if iterations == 0 or len(y_true) < 2:
        return {}
    rng = np.random.default_rng(seed)
    sampled: dict[str, list[float]] = {}
    for _ in range(iterations):
        indexes = _molecule_bootstrap_indices(molecule_ids, rng)
        truth = y_true[indexes]
        prediction = y_pred[indexes]
        if task == "classification" and len(np.unique(truth)) < 2:
            continue
        if task == "classification":
            if selected_metrics.issubset({"roc_auc", "pr_auc", "mcc"}):
                labels = (prediction >= float(threshold)).astype(int)
                all_values = {
                    "roc_auc": float(roc_auc_score(truth, prediction)),
                    "pr_auc": float(average_precision_score(truth, prediction)),
                    "mcc": float(matthews_corrcoef(truth, labels)),
                }
            else:
                all_values = classification_metrics(
                    truth, prediction, threshold=float(threshold)
                )
        elif selected_metrics.intersection({"r2", "spearman_r", "kendall_tau"}):
            all_values = regression_metrics(truth, prediction)
        else:
            mae = float(np.mean(np.abs(truth - prediction)))
            denominator = float(np.mean(np.abs(truth - np.mean(truth))))
            all_values = {
                "mae": mae,
                "relative_absolute_error": (
                    mae / denominator if denominator else float("nan")
                ),
            }
        if uncertainty is not None:
            all_values.update(
                _uncertainty_metrics(truth, prediction, uncertainty[indexes])
            )
        if prediction_lower is not None and prediction_upper is not None:
            all_values.update(
                _interval_metrics(
                    truth, prediction_lower[indexes], prediction_upper[indexes]
                )
            )
        for metric, value in all_values.items():
            if metric not in selected_metrics:
                continue
            if np.isfinite(value):
                sampled.setdefault(metric, []).append(float(value))
    alpha = (1 - confidence_level) / 2
    return {
        metric: (
            float(np.quantile(values, alpha)),
            float(np.quantile(values, 1 - alpha)),
        )
        for metric, values in sampled.items()
        if values
    }


def _molecule_bootstrap_indices(
    molecule_ids: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Resample chemical entities while retaining all observations per molecule."""

    identifiers = np.asarray(molecule_ids, dtype=str)
    unique_ids = np.unique(identifiers)
    sampled_ids = rng.choice(unique_ids, size=len(unique_ids), replace=True)
    return np.concatenate(
        [np.flatnonzero(identifiers == molecule_id) for molecule_id in sampled_ids]
    )


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "task",
        "endpoint",
        "preprocessing",
        "split_strategy",
        "partition",
        "model",
        "metric",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(group_columns, dropna=False, sort=True):
        values = group["value"].dropna().to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "n_results": len(values),
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "median": float(np.median(values)) if len(values) else np.nan,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "q1": float(np.quantile(values, 0.25)) if len(values) else np.nan,
                "q3": float(np.quantile(values, 0.75)) if len(values) else np.nan,
                "minimum": float(np.min(values)) if len(values) else np.nan,
                "maximum": float(np.max(values)) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def paired_comparisons(
    metrics: pd.DataFrame, predictions: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Compare models only on matching repeat/fold records.

    With at least three complete models, the table uses a Friedman omnibus test
    and Conover-Friedman pairwise tests with Holm correction. Two-model cases
    use a paired Wilcoxon test. These are nominal, exploratory tests: folds
    may share molecules/training data and are not independent datasets.
    Raw fold results remain the primary evidence.
    """

    family_columns = [
        "task",
        "endpoint",
        "preprocessing",
        "split_strategy",
        "partition",
        "metric",
    ]
    rows: list[dict[str, Any]] = []
    complete_contexts: set[tuple[Any, ...]] | None = None
    if predictions is not None:
        complete_contexts = set()
        for keys, group in predictions.groupby(IDENTITY_COLUMNS, dropna=False):
            model_sets = [
                set(values["sample_id"].astype(str))
                for _, values in group.groupby("model", sort=True)
            ]
            if model_sets and all(values == model_sets[0] for values in model_sets[1:]):
                complete_contexts.add(tuple(keys))
    for keys, family in metrics.groupby(family_columns, dropna=False, sort=True):
        if complete_contexts is not None:
            family = family[
                family.apply(
                    lambda row: tuple(row[column] for column in IDENTITY_COLUMNS)
                    in complete_contexts,
                    axis=1,
                )
            ]
        wide = family.pivot_table(
            index=["repeat", "fold"], columns="model", values="value", aggfunc="first"
        ).dropna(axis=0, how="any")
        if wide.shape[0] < 2 or wide.shape[1] < 2:
            continue
        all_tied = bool(np.all(np.ptp(wide.to_numpy(dtype=float), axis=1) == 0))
        omnibus_p = np.nan
        if all_tied:
            omnibus_p = 1.0
        elif wide.shape[1] >= 3:
            try:
                omnibus_p = float(friedmanchisquare(*[wide[column] for column in wide]).pvalue)
            except ValueError:
                omnibus_p = np.nan
        conover_raw: pd.DataFrame | None = None
        conover_holm: pd.DataFrame | None = None
        if wide.shape[1] >= 3 and not all_tied:
            import scikit_posthocs as posthocs

            # Perfectly consistent ranks can produce a zero denominator inside
            # the test. The implementation returns the limiting p-values, but
            # NumPy also emits a divide-by-zero warning that is not actionable.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                # Passing named pandas axes triggers a melt bug in some
                # scikit-posthocs/pandas combinations. The numerical result is
                # identical; restore model labels immediately after the call.
                model_names = list(wide.columns.astype(str))
                conover_raw = posthocs.posthoc_conover_friedman(
                    wide.to_numpy(dtype=float), p_adjust=None
                )
                conover_holm = posthocs.posthoc_conover_friedman(
                    wide.to_numpy(dtype=float), p_adjust="holm"
                )
                conover_raw.index = conover_raw.columns = model_names
                conover_holm.index = conover_holm.columns = model_names
        for left, right in combinations(wide.columns.astype(str), 2):
            delta = wide[left].to_numpy() - wide[right].to_numpy()
            if all_tied:
                pairwise_test = "all_tied"
                p_value = 1.0
                adjusted_p = 1.0
            elif conover_raw is not None and conover_holm is not None:
                pairwise_test = "conover_friedman"
                p_value = float(conover_raw.loc[left, right])
                adjusted_p = float(conover_holm.loc[left, right])
            else:
                pairwise_test = "wilcoxon_signed_rank"
                try:
                    p_value = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
                except ValueError:
                    p_value = np.nan
                adjusted_p = p_value
            rows.append(
                {
                    **dict(zip(family_columns, keys, strict=True)),
                    "model_a": left,
                    "model_b": right,
                    "paired_results": len(delta),
                    "median_delta_a_minus_b": float(np.median(delta)),
                    "mean_delta_a_minus_b": float(np.mean(delta)),
                    "friedman_p_value": omnibus_p,
                    "pairwise_test": pairwise_test,
                    "unadjusted_p_value": p_value,
                    "holm_adjusted_p_value": adjusted_p,
                    "inference_status": "exploratory_unverified_dependence",
                    "inference_caveat": (
                        "Fold/repeat blocks may share molecules and training data. "
                        "These nominal p-values do not establish confirmatory significance; "
                        "Holm correction does not correct dependence in the underlying test."
                    ),
                }
            )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _software_versions() -> dict[str, str]:
    packages = ["admet-benchmark", "numpy", "pandas", "scipy", "scikit-learn", "rdkit"]
    resolved: dict[str, str] = {"python": platform.python_version()}
    for package in packages:
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = "not-installed"
    return resolved


def run_evaluation(
    config: dict[str, Any], output_dir: str | Path | None = None
) -> EvaluationResult:
    """Validate prepared predictions and write a reproducible evaluation bundle."""

    validate_config_schema(config)
    if int(config.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
        raise ValueError(f"Only evaluation schema_version {SCHEMA_VERSION} is supported")
    canonical, source_paths = canonicalize_predictions(config)
    predictions, validation = validate_predictions(canonical, config)
    if validation["errors"]:
        message = "Prediction validation failed:\n- " + "\n- ".join(validation["errors"])
        raise ValueError(message)

    destination_value = output_dir or config.get("output_dir")
    if not destination_value:
        raise ValueError("Set output_dir in YAML or pass --output-dir")
    destination = Path(destination_value).expanduser()
    if not destination.is_absolute():
        destination = Path(config["_config_dir"]) / destination
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    metrics = compute_fold_metrics(predictions, config)
    summary = summarize_metrics(metrics)
    coverage = _coverage_audit(predictions)
    comparisons = paired_comparisons(metrics, predictions)
    predictions.to_csv(destination / "predictions.csv", index=False)
    metrics.to_csv(destination / "fold_metrics.csv", index=False)
    summary.to_csv(destination / "metric_distributions.csv", index=False)
    coverage.to_csv(destination / "coverage_audit.csv", index=False)
    comparisons.to_csv(destination / "statistical_comparisons.csv", index=False)
    write_json(destination / "validation_report.json", validation)
    resolved_config = {key: value for key, value in config.items() if key != "_config_dir"}
    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "project_name": config.get("project_name", destination.name),
        "task": config["task"],
        "source_files": [
            {"path": str(path), "sha256": _sha256(path)} for path in source_paths
        ],
        "outputs": {
            "predictions": "predictions.csv",
            "fold_metrics": "fold_metrics.csv",
            "metric_distributions": "metric_distributions.csv",
            "coverage_audit": "coverage_audit.csv",
            "statistical_comparisons": "statistical_comparisons.csv",
        },
        "software": _software_versions(),
    }
    write_json(destination / "evaluation_manifest.json", manifest)

    from .evaluation_report import write_evaluation_report

    write_evaluation_report(destination, config, predictions, metrics, summary, validation)
    return EvaluationResult(destination, predictions, metrics, summary, validation)


def validate_evaluation_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate an input configuration without writing evaluation artifacts."""

    validate_config_schema(config)
    canonical, paths = canonicalize_predictions(config)
    _, validation = validate_predictions(canonical, config)
    validation["source_files"] = [str(path) for path in paths]
    return validation
