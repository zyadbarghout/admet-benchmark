"""A model arena with a shared validation split and untouched final test set."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from html import escape
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    TimeSeriesSplit,
)

from .features import build_features
from .evaluation import run_evaluation
from .metrics import classification_metrics, regression_metrics
from .models import model_specs
from .splits import split_dataset
from .pipeline import (
    _chemical_filter_from_config,
    _cross_validation,
    _inverse_transform,
    _prepare_split,
    _search_workers,
)
from .config import load_config, resolve_config_path
from .data import load_data_split_bundle
from .reporting import write_json


@dataclass(frozen=True)
class ModelOutcome:
    name: str
    family: str
    parameters: dict[str, Any]
    hpo_score: float | None
    validation_metrics: dict[str, float]
    validation_selection_score: float
    test_metrics: dict[str, float]
    validation_prediction: np.ndarray
    test_prediction: np.ndarray


@dataclass(frozen=True)
class BenchmarkResult:
    output_dir: Path
    selected_model: str
    comparison: pd.DataFrame


def _metrics(task: str, y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if task == "regression":
        return regression_metrics(y_true, prediction)
    return classification_metrics(y_true, prediction)


def _selection_score(metrics: dict[str, float], metric_name: str) -> float:
    """Convert all selection metrics to a higher-is-better convention."""

    aliases = {
        "average_precision": "pr_auc",
        "prc": "pr_auc",
        "roc": "roc_auc",
        "neg_mean_absolute_error": "mae",
    }
    resolved = aliases.get(metric_name, metric_name)
    if resolved not in metrics:
        raise KeyError(f"Selection metric {metric_name!r} is not available")
    value = float(metrics[resolved])
    return -value if resolved in {"mae", "relative_absolute_error"} else value


def _split_development_data(
    frame: pd.DataFrame,
    task: str,
    validation_size: float,
    seed: int,
    smiles_column: str,
    split_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create the shared validation partition using the declared chemistry policy."""

    if not 0 < validation_size < 0.5:
        raise ValueError("benchmark.validation_size must be between 0 and 0.5")
    strategy = str(split_config.get("strategy", "random")).lower()
    if strategy == "predefined":
        raise ValueError("benchmark.validation.strategy cannot be predefined")
    result = split_dataset(
        frame,
        target="target_model",
        smiles_column=smiles_column,
        task=task,
        strategy=strategy,
        test_size=validation_size,
        seed=seed,
        split_config=split_config,
    )
    development = result.train.copy()
    validation = result.test.copy()
    if strategy == "temporal":
        time_column = str(split_config["time_column"])
        development = development.sort_values(time_column, kind="stable")
        validation = validation.sort_values(time_column, kind="stable")
    development["arena_partition"] = "development"
    validation["arena_partition"] = "validation"
    return (
        development.reset_index(drop=True),
        validation.reset_index(drop=True),
        {**result.metadata, "role": "shared_model_selection_validation"},
    )


def _inner_cross_validation(
    task: str,
    y: np.ndarray,
    folds: int,
    seed: int,
    strategy: str,
    groups: np.ndarray | None,
):
    """Match HPO folds to the validation split family whenever possible."""

    if strategy in {"scaffold", "butina"} or (strategy == "random" and groups is not None):
        if groups is None:
            raise ValueError(f"{strategy} HPO requires chemical group labels")
        if len(np.unique(groups)) < folds:
            raise ValueError(f"{strategy} HPO requires at least {folds} chemical groups")
        if task == "classification":
            return StratifiedGroupKFold(
                n_splits=folds, shuffle=True, random_state=seed
            )
        return GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    if strategy == "temporal":
        return TimeSeriesSplit(n_splits=folds)
    return _cross_validation(task, y, folds, seed)


def _prediction_column(frame: pd.DataFrame) -> str:
    candidates = [column for column in frame if column.startswith("pred_")]
    if len(candidates) == 1:
        return candidates[0]
    # Chemprop 2.3 uses the original target column name, while earlier
    # versions use ``pred_*``. Prediction inputs contain only SMILES, so the
    # remaining column is safe only when there is exactly one target.
    remaining = [column for column in frame if column != "smiles"]
    if len(remaining) == 1:
        return remaining[0]
    raise RuntimeError(
        "Chemprop prediction output must contain exactly one target column; "
        f"found {list(frame.columns)}"
    )


def _run_command(command: list[str], output_dir: Path, name: str) -> None:
    """Persist commands and combined logs so neural trials are reproducible."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}_command.json").write_text(json.dumps(command, indent=2) + "\n")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    (output_dir / f"{name}.log").write_text(
        "# stdout\n" + completed.stdout + "\n# stderr\n" + completed.stderr
    )
    if completed.returncode != 0:
        tail = (completed.stdout + "\n" + completed.stderr)[-4_000:]
        raise RuntimeError(f"{' '.join(command[:2])} failed:\n{tail}")


def _chemprop_accelerator(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "Chemprop requires its optional dependencies. Install admet-benchmark[models]."
        ) from error


def _write_chemprop_data(frame: pd.DataFrame, smiles_column: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [smiles_column, "target_model"]
    if "split" in frame:
        columns.append("split")
    frame.loc[:, columns].rename(
        columns={smiles_column: "smiles", "target_model": "target"}
    ).to_csv(path, index=False)


def _chemprop_train_command(
    *,
    task: str,
    train_path: Path,
    use_splits_column: bool,
    output_dir: Path,
    candidate: dict[str, Any],
    model_name: str,
    seed: int,
    accelerator: str,
    final_refit: bool = False,
) -> list[str]:
    chemprop_executable = Path(sys.executable).with_name("chemprop")
    if not chemprop_executable.exists():
        raise FileNotFoundError(
            f"Chemprop is not installed beside the active Python interpreter: {sys.executable}"
        )
    command = [
        str(chemprop_executable),
        "train",
        "--data-path",
        str(train_path),
        "--smiles-columns",
        "smiles",
        "--target-columns",
        "target",
        "--task-type",
        task,
        "--epochs",
        str(candidate["epochs"]),
        "--warmup-epochs",
        "0",
        "--patience",
        str(0 if final_refit else candidate.get("patience", 5)),
        "--batch-size",
        str(candidate.get("batch_size", 128)),
        "--num-workers",
        "0",
        "--pytorch-seed",
        str(seed),
        "--max-lr",
        str(candidate.get("max_lr", 1e-3)),
        "--init-lr",
        str(candidate.get("init_lr", 1e-4)),
        "--final-lr",
        str(candidate.get("final_lr", 1e-4)),
        "--ffn-hidden-dim",
        str(candidate.get("ffn_hidden_dim", 300)),
        "--ffn-num-layers",
        str(candidate.get("ffn_num_layers", 1)),
        "--dropout",
        str(candidate.get("dropout", 0.0)),
        "--output-dir",
        str(output_dir),
        "-q",
    ]
    if use_splits_column:
        # Do not let Chemprop make an implicit split. The caller owns every
        # split so that validation and final-test roles stay unambiguous.
        command.extend(["--splits-column", "split"])
    if accelerator != "cpu":
        command.extend(["--accelerator", accelerator, "--devices", "1"])
    if model_name == "chemeleon":
        command.extend(["--from-foundation", "CheMeleon"])
    else:
        command.extend(
            [
                "--depth",
                str(candidate.get("depth", 3)),
                "--message-hidden-dim",
                str(candidate.get("message_hidden_dim", 300)),
            ]
        )
    return command


def _chemprop_predict(
    checkpoint: Path,
    frame: pd.DataFrame,
    smiles_column: str,
    output_dir: Path,
    accelerator: str,
) -> np.ndarray:
    output_dir.mkdir(parents=True, exist_ok=True)
    test_path = output_dir / "prediction_input.csv"
    prediction_path = output_dir / "predictions.csv"
    frame.loc[:, [smiles_column]].rename(columns={smiles_column: "smiles"}).to_csv(
        test_path, index=False
    )
    command = [
        str(Path(sys.executable).with_name("chemprop")),
        "predict",
        "--test-path",
        str(test_path),
        "--smiles-columns",
        "smiles",
        "--model-paths",
        str(checkpoint),
        "--preds-path",
        str(prediction_path),
        "--num-workers",
        "0",
        "-q",
    ]
    if accelerator != "cpu":
        command.extend(["--accelerator", accelerator, "--devices", "1"])
    _run_command(command, output_dir, "predict")
    predictions = pd.read_csv(prediction_path)
    if len(predictions) != len(frame):
        raise RuntimeError("Chemprop returned a different number of predictions than inputs")
    return predictions[_prediction_column(predictions)].to_numpy(dtype=float)


def _best_epoch(training_dir: Path, fallback: int) -> int:
    checkpoints = list((training_dir / "model_0" / "checkpoints").glob("best-epoch=*.ckpt"))
    if not checkpoints:
        return fallback
    matched = re.search(r"best-epoch=(\d+)", checkpoints[0].name)
    return int(matched.group(1)) + 1 if matched else fallback


def _classical_outcome(
    *,
    spec,
    task: str,
    x_development: np.ndarray,
    y_development: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    hpo_config: dict[str, Any],
    inner_strategy: str,
    cv_groups: np.ndarray | None,
    selection_metric: str,
    seed: int,
    output_dir: Path,
) -> ModelOutcome:
    workers = _search_workers(hpo_config.get("n_jobs", "auto"))
    cv = _inner_cross_validation(
        task,
        y_development,
        int(hpo_config.get("cv_folds", 3)),
        seed,
        inner_strategy,
        cv_groups,
    )
    scoring = hpo_config.get(
        "scoring", "average_precision" if task == "classification" else "neg_mean_absolute_error"
    )
    search = RandomizedSearchCV(
        spec.estimator,
        spec.search_space,
        n_iter=int(hpo_config.get("n_iter", 3)),
        scoring=scoring,
        cv=cv,
        random_state=seed,
        n_jobs=workers,
        error_score="raise",
        refit=False,
    )
    fit_options = {"groups": cv_groups} if cv_groups is not None else {}
    search.fit(x_development, y_development, **fit_options)
    candidates = pd.DataFrame(search.cv_results_).loc[
        :, ["rank_test_score", "mean_test_score", "std_test_score", "params"]
    ].rename(
        columns={
            "rank_test_score": "inner_cv_rank",
            "mean_test_score": "inner_cv_score",
            "std_test_score": "inner_cv_score_std",
        }
    )
    candidates["params"] = candidates["params"].map(
        lambda value: json.dumps(value, sort_keys=True, default=str)
    )
    candidates.to_csv(output_dir / "hpo_candidates.csv", index=False)

    development_estimator = clone(spec.estimator).set_params(**search.best_params_)
    if "model__n_jobs" in development_estimator.get_params(deep=True):
        development_estimator.set_params(model__n_jobs=workers)
    development_estimator.fit(x_development, y_development)
    validation_prediction = (
        development_estimator.predict_proba(x_validation)[:, 1]
        if task == "classification"
        else development_estimator.predict(x_validation)
    )
    validation_metrics = _metrics(task, y_validation, validation_prediction)
    final_estimator = clone(spec.estimator).set_params(**search.best_params_)
    if "model__n_jobs" in final_estimator.get_params(deep=True):
        final_estimator.set_params(model__n_jobs=workers)
    final_estimator.fit(x_train, y_train)
    test_prediction = (
        final_estimator.predict_proba(x_test)[:, 1]
        if task == "classification"
        else final_estimator.predict(x_test)
    )
    test_metrics = _metrics(task, y_test, test_prediction)
    return ModelOutcome(
        name=spec.name,
        family="classical",
        parameters=search.best_params_,
        hpo_score=float(search.best_score_),
        validation_metrics=validation_metrics,
        validation_selection_score=_selection_score(validation_metrics, selection_metric),
        test_metrics=test_metrics,
        validation_prediction=np.asarray(validation_prediction),
        test_prediction=np.asarray(test_prediction),
    )


def _neural_outcome(
    *,
    model_name: str,
    candidates: list[dict[str, Any]],
    task: str,
    development: pd.DataFrame,
    validation: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    smiles_column: str,
    selection_metric: str,
    seed: int,
    inner_validation_size: float,
    inner_split_config: dict[str, Any],
    accelerator: str,
    output_dir: Path,
) -> ModelOutcome:
    """Tune a Chemprop-family model on development data and refit it on train."""

    inner_train, inner_validation, inner_split_metadata = _split_development_data(
        development,
        task,
        inner_validation_size,
        seed,
        smiles_column,
        inner_split_config,
    )
    candidate_rows: list[dict[str, Any]] = []
    trial_records: list[tuple[dict[str, Any], dict[str, float], np.ndarray, int]] = []

    for candidate_index, candidate in enumerate(candidates):
        trial_dir = output_dir / "hpo" / f"candidate_{candidate_index:02d}"
        input_dir = trial_dir / "input"
        training_path = input_dir / "development_splits.csv"
        # Chemprop's documented split-column mode consumes both development
        # partitions from one file. The common validation split is deliberately
        # not written here, so no candidate can train against its labels.
        inner_train_data = inner_train.copy()
        inner_train_data["split"] = "train"
        inner_validation_data = inner_validation.copy()
        inner_validation_data["split"] = "val"
        training_data = pd.concat(
            [inner_train_data, inner_validation_data], ignore_index=True
        )
        _write_chemprop_data(training_data, smiles_column, training_path)
        training_dir = trial_dir / "training"
        command = _chemprop_train_command(
            task=task,
            train_path=training_path,
            use_splits_column=True,
            output_dir=training_dir,
            candidate=candidate,
            model_name=model_name,
            seed=seed,
            accelerator=accelerator,
        )
        _run_command(command, trial_dir, "train")
        checkpoint = training_dir / "model_0" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Chemprop did not create a checkpoint at {checkpoint}")
        inner_prediction = _chemprop_predict(
            checkpoint,
            inner_validation,
            smiles_column,
            trial_dir / "inner_validation",
            accelerator,
        )
        inner_metrics = _metrics(
            task, inner_validation["target_model"].to_numpy(), inner_prediction
        )
        trained_epochs = _best_epoch(training_dir, int(candidate["epochs"]))
        row = {
            "candidate": candidate_index,
            "inner_validation_selection_score": _selection_score(
                inner_metrics, selection_metric
            ),
            "trained_epochs": trained_epochs,
            "parameters": json.dumps(candidate, sort_keys=True),
            **{f"inner_validation_{key}": value for key, value in inner_metrics.items()},
        }
        candidate_rows.append(row)
        trial_records.append((candidate, inner_metrics, inner_prediction, trained_epochs))

    candidates_frame = pd.DataFrame(candidate_rows).sort_values(
        "inner_validation_selection_score", ascending=False
    )
    candidates_frame.to_csv(output_dir / "hpo_candidates.csv", index=False)
    best_position = int(candidates_frame.iloc[0]["candidate"])
    best_candidate, inner_metrics, _, trained_epochs = trial_records[
        best_position
    ]

    validation_refit_dir = output_dir / "validation_refit"
    validation_input = validation_refit_dir / "input" / "development.csv"
    development_data = development.copy()
    development_data["split"] = "train"
    _write_chemprop_data(development_data, smiles_column, validation_input)
    selected_candidate = {**best_candidate, "epochs": trained_epochs}
    validation_command = _chemprop_train_command(
        task=task,
        train_path=validation_input,
        use_splits_column=True,
        output_dir=validation_refit_dir / "training",
        candidate=selected_candidate,
        model_name=model_name,
        seed=seed,
        accelerator=accelerator,
        final_refit=True,
    )
    _run_command(validation_command, validation_refit_dir, "train")
    validation_checkpoint = validation_refit_dir / "training" / "model_0" / "best.pt"
    validation_prediction = _chemprop_predict(
        validation_checkpoint,
        validation,
        smiles_column,
        validation_refit_dir / "validation",
        accelerator,
    )
    validation_metrics = _metrics(
        task, validation["target_model"].to_numpy(), validation_prediction
    )

    final_dir = output_dir / "final_refit"
    final_input = final_dir / "input" / "all_train.csv"
    full_train = train.copy()
    full_train["split"] = "train"
    _write_chemprop_data(full_train, smiles_column, final_input)
    # The selected epoch count comes only from development validation. The
    # final refit therefore uses every labelled training molecule with no test
    # labels or early-stopping feedback available to it.
    final_candidate = selected_candidate
    train_command = _chemprop_train_command(
        task=task,
        train_path=final_input,
        use_splits_column=True,
        output_dir=final_dir / "training",
        candidate=final_candidate,
        model_name=model_name,
        seed=seed,
        accelerator=accelerator,
        final_refit=True,
    )
    _run_command(train_command, final_dir, "train")
    final_checkpoint = final_dir / "training" / "model_0" / "best.pt"
    test_prediction = _chemprop_predict(
        final_checkpoint, test, smiles_column, final_dir / "test", accelerator
    )
    test_metrics = _metrics(task, test["target_model"].to_numpy(), test_prediction)
    write_json(
        output_dir / "selected_hyperparameters.json",
        {
            "parameters": best_candidate,
            "epochs_from_development_validation": trained_epochs,
            "inner_split": inner_split_metadata,
            "inner_validation_metrics": inner_metrics,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        },
    )
    return ModelOutcome(
        name=model_name,
        family="neural",
        parameters=final_candidate,
        hpo_score=_selection_score(inner_metrics, selection_metric),
        validation_metrics=validation_metrics,
        validation_selection_score=_selection_score(validation_metrics, selection_metric),
        test_metrics=test_metrics,
        validation_prediction=validation_prediction,
        test_prediction=test_prediction,
    )


def _write_prediction_table(
    frame: pd.DataFrame,
    smiles_column: str,
    outcomes: list[ModelOutcome],
    prediction_attribute: str,
    path: Path,
) -> None:
    table = frame.loc[
        :, ["source_row", "sample_id", "molecule_id", smiles_column, "target_raw", "target_model"]
    ].copy()
    for outcome in outcomes:
        table[f"prediction__{outcome.name}"] = getattr(outcome, prediction_attribute)
    table.to_csv(path, index=False)


def _write_evaluation_bundle(
    *,
    output_dir: Path,
    project_name: str,
    task: str,
    target: str,
    smiles_column: str,
    target_transform: str,
    protocol_name: str,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    outcomes: list[ModelOutcome],
    evaluation_config: dict[str, Any],
    seed: int,
) -> None:
    """Convert arena predictions to the public evaluator's stable input contract."""

    records: list[pd.DataFrame] = []
    for partition, frame, attribute in (
        ("validation", validation, "validation_prediction"),
        ("test", test, "test_prediction"),
    ):
        for outcome in outcomes:
            prediction = np.asarray(getattr(outcome, attribute), dtype=float)
            if task == "regression":
                prediction = _inverse_transform(prediction, target, target_transform)
            records.append(
                pd.DataFrame(
                    {
                        "sample_id": frame["sample_id"].astype(str).to_numpy(),
                        "molecule_id": frame["molecule_id"].astype(str).to_numpy(),
                        "smiles": frame[smiles_column].astype(str).to_numpy(),
                        "endpoint": target,
                        "model": outcome.name,
                        "partition": partition,
                        "observed": frame["target_raw"].to_numpy(dtype=float),
                        "prediction": prediction,
                        "repeat": 0,
                        "fold": 0,
                        "split_strategy": protocol_name,
                        "preprocessing": "challenge_target_policy",
                    }
                )
            )
    prediction_path = output_dir / "evaluation_predictions.csv"
    pd.concat(records, ignore_index=True).to_csv(prediction_path, index=False)

    config: dict[str, Any] = {
        "schema_version": 1,
        "project_name": project_name,
        "task": task,
        "output_dir": "evaluation",
        "predictions": {
            "path": prediction_path.name,
            "columns": {
                name: name
                for name in (
                    "sample_id",
                    "molecule_id",
                    "smiles",
                    "endpoint",
                    "model",
                    "partition",
                    "observed",
                    "prediction",
                    "repeat",
                    "fold",
                    "split_strategy",
                    "preprocessing",
                )
            },
        },
        "statistics": {
            "bootstrap_iterations": int(
                evaluation_config.get("bootstrap_iterations", 1_000)
            ),
            "confidence_level": float(evaluation_config.get("confidence_level", 0.95)),
            "seed": int(evaluation_config.get("seed", seed)),
        },
    }
    if task == "regression":
        config["target_transform"] = {"default": target_transform}
        config["statistics"]["bootstrap_metrics"] = evaluation_config.get(
            "bootstrap_metrics", ["mae", "relative_absolute_error"]
        )
    else:
        config["thresholds"] = {
            "endpoints": {
                target: float(evaluation_config.get("decision_threshold", 0.5))
            }
        }
        config["statistics"]["bootstrap_metrics"] = evaluation_config.get(
            "bootstrap_metrics", ["roc_auc", "pr_auc", "mcc"]
        )
    (output_dir / "evaluation.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    run_evaluation({**config, "_config_dir": str(output_dir)}, output_dir / "evaluation")


def _write_benchmark_report(
    output_dir: Path,
    comparison: pd.DataFrame,
    selected_model: str,
    task: str,
    target: str,
    selection_metric: str,
    split_metadata: dict[str, Any],
) -> None:
    summary_columns = [
        "model",
        "family",
        "validation_selection_score",
        *sorted(column for column in comparison if column.startswith("validation_")),
        *sorted(column for column in comparison if column.startswith("test_")),
    ]
    summary = comparison.loc[:, list(dict.fromkeys(summary_columns))]
    table = summary.to_html(
        index=False,
        table_id="metrics-table",
        float_format=lambda value: f"{value:.5f}",
    )
    split_strategy = escape(str(split_metadata["strategy"]))
    actual_test_size = float(split_metadata["actual_test_size"])
    (output_dir / "report.html").write_text(
        "<html><head><title>ADMET model benchmark</title>"
        "<style>body{font-family:system-ui;margin:2rem;max-width:1300px;line-height:1.5}"
        "table{border-collapse:collapse;font-size:.9rem}th,td{padding:.45rem .6rem;"
        "border:1px solid #ddd}th{background:#f3f4f6;cursor:pointer;user-select:none}"
        "th:hover{background:#e5e7eb}a{margin-right:1rem}</style></head><body>"
        f"<h1>{target}: {task} model benchmark</h1>"
        f"<p><strong>Selection metric:</strong> {selection_metric}<br>"
        f"<strong>Selected by validation:</strong> {selected_model}</p>"
        f"<p><strong>Train/test strategy:</strong> {split_strategy} "
        f"({actual_test_size:.1%} held out)</p>"
        "<p>Hyperparameter search and model selection use development data only. "
        "Every row has been refit and evaluated on the held-out test split; those "
        "test values were not used to select a model.</p>"
        "<p>Click a column header to sort the table. "
        "<a href='model_comparison.csv'>comparison CSV</a>"
        "<a href='validation_predictions.csv'>validation predictions</a>"
        "<a href='held_out_test_predictions.csv'>held-out predictions</a>"
        "<a href='evaluation/report.html'>evaluation report</a>"
        "<a href='data/split_manifest.json'>split manifest</a>"
        "<a href='data/validation_split_manifest.json'>validation split manifest</a>"
        "<a href='resolved_config.yaml'>resolved configuration</a></p>"
        f"{table}"
        "<script>"
        "const table=document.getElementById('metrics-table');let ascending={};"
        "for(const [column,header] of [...table.tHead.rows[0].cells].entries()){"
        "header.title='Sort by this column';header.addEventListener('click',()=>{"
        "const rows=[...table.tBodies[0].rows];const direction=ascending[column]?-1:1;"
        "rows.sort((left,right)=>{const a=left.cells[column].textContent.trim();"
        "const b=right.cells[column].textContent.trim();const an=Number(a),bn=Number(b);"
        "const comparison=Number.isNaN(an)||Number.isNaN(bn)?a.localeCompare(b):an-bn;"
        "return direction*comparison;});rows.forEach(row=>table.tBodies[0].append(row));"
        "ascending[column]=!ascending[column];});}"
        "</script></body></html>"
    )


def run_model_benchmark(config: dict[str, Any], output_dir: str | Path) -> BenchmarkResult:
    """Benchmark classical and neural molecular models under one honest protocol."""

    benchmark_config = config["benchmark"]
    task = config["task"]
    seed = int(config.get("seed", 42))
    dataset_config = config["dataset"]
    target = str(dataset_config["target"])
    smiles_column = str(dataset_config["smiles_column"])
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)

    chemical_filter = _chemical_filter_from_config(config)
    loaded_splits = load_data_split_bundle(config)
    raw_train, raw_test = loaded_splits.train, loaded_splits.test
    train, train_audit, train_summary = _prepare_split(
        raw_train,
        task=task,
        target=target,
        smiles_column=smiles_column,
        name_column=dataset_config.get("name_column"),
        target_transform=dataset_config.get("target_transform", "none"),
        chemical_filter=chemical_filter,
    )
    test, test_audit, test_summary = _prepare_split(
        raw_test,
        task=task,
        target=target,
        smiles_column=smiles_column,
        name_column=dataset_config.get("name_column"),
        target_transform=dataset_config.get("target_transform", "none"),
        chemical_filter=chemical_filter,
    )
    validation_config = dict(benchmark_config.get("validation") or {})
    validation_size = float(
        validation_config.get("size", benchmark_config.get("validation_size", 0.2))
    )
    development, validation, validation_metadata = _split_development_data(
        train,
        task,
        validation_size,
        seed,
        smiles_column,
        validation_config,
    )
    data_dir = output / "data"
    data_dir.mkdir(exist_ok=True)
    write_json(data_dir / "split_manifest.json", loaded_splits.metadata)
    write_json(data_dir / "validation_split_manifest.json", validation_metadata)
    for name, frame in {
        "development": development,
        "validation": validation,
        "held_out_test": test,
        "train_filter_audit": train_audit,
        "test_filter_audit": test_audit,
    }.items():
        frame.to_csv(data_dir / f"{name}.csv", index=False)

    features_config = config.get("features", {})
    feature_options = {
        "radius": int(features_config.get("radius", 2)),
        "n_bits": int(features_config.get("n_bits", 1024)),
        "include_rdkit_descriptors": bool(features_config.get("include_rdkit_descriptors", True)),
        "representation": features_config.get("representation"),
        "descriptor_set": str(features_config.get("descriptor_set", "compact")),
    }
    all_smiles = pd.concat([development[smiles_column], validation[smiles_column], test[smiles_column]])
    feature_set = build_features(all_smiles.tolist(), **feature_options)
    features = feature_set.values
    write_json(
        output / "feature_schema.json",
        {"feature_count": len(feature_set.names), "feature_names": feature_set.names},
    )
    development_end = len(development)
    validation_end = development_end + len(validation)
    x_development = features[:development_end]
    x_validation = features[development_end:validation_end]
    x_test = features[validation_end:]
    x_train = np.vstack([x_development, x_validation])
    y_development = development["target_model"].to_numpy()
    y_validation = validation["target_model"].to_numpy()
    y_test = test["target_model"].to_numpy()
    y_train = np.concatenate([y_development, y_validation])
    selection_metric = str(benchmark_config["selection_metric"])
    outcomes: list[ModelOutcome] = []

    model_config = benchmark_config["models"]
    classical_config = benchmark_config.get("classical_hpo", {})
    for spec in model_specs(task, list(model_config.get("classical", [])), seed):
        model_dir = output / "models" / spec.name
        model_dir.mkdir(exist_ok=True)
        outcome = _classical_outcome(
            spec=spec,
            task=task,
            x_development=x_development,
            y_development=y_development,
            x_validation=x_validation,
            y_validation=y_validation,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            hpo_config=classical_config,
            inner_strategy=str(validation_metadata["strategy"]),
            cv_groups=(
                development["benchmark_split_group"].to_numpy()
                if "benchmark_split_group" in development
                else None
            ),
            selection_metric=selection_metric,
            seed=seed,
            output_dir=model_dir,
        )
        outcomes.append(outcome)

    neural_config = benchmark_config.get("neural", {})
    neural_models = list(model_config.get("neural", []))
    accelerator: str | None = None
    if neural_models:
        accelerator = _chemprop_accelerator(str(neural_config.get("accelerator", "auto")))
        assert accelerator is not None
    for model_name in neural_models:
        candidates = neural_config.get(model_name, {}).get("candidates", [])
        if not candidates:
            raise ValueError(f"No HPO candidates configured for neural model {model_name!r}")
        outcomes.append(
            _neural_outcome(
                model_name=model_name,
                candidates=list(candidates),
                task=task,
                development=development,
                validation=validation,
                train=train,
                test=test,
                smiles_column=smiles_column,
                selection_metric=selection_metric,
                seed=seed,
                inner_validation_size=float(neural_config.get("inner_validation_size", 0.15)),
                inner_split_config=dict(
                    neural_config.get("inner_split") or validation_config
                ),
                accelerator=accelerator,
                output_dir=output / "models" / model_name,
            )
        )

    rows = []
    for outcome in outcomes:
        rows.append(
            {
                "model": outcome.name,
                "family": outcome.family,
                "hpo_inner_cv_score": outcome.hpo_score,
                "validation_selection_score": outcome.validation_selection_score,
                "parameters": json.dumps(outcome.parameters, sort_keys=True),
                **{f"validation_{key}": value for key, value in outcome.validation_metrics.items()},
                **{f"test_{key}": value for key, value in outcome.test_metrics.items()},
            }
        )
    comparison = pd.DataFrame(rows).sort_values("validation_selection_score", ascending=False)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    selected_model = str(comparison.iloc[0]["model"])
    _write_prediction_table(validation, smiles_column, outcomes, "validation_prediction", output / "validation_predictions.csv")
    _write_prediction_table(test, smiles_column, outcomes, "test_prediction", output / "held_out_test_predictions.csv")
    evaluation_config = dict(benchmark_config.get("evaluation") or {})
    if bool(evaluation_config.get("enabled", True)):
        protocol_name = (
            f"outer_{loaded_splits.metadata['strategy']}__"
            f"validation_{validation_metadata['strategy']}"
        )
        _write_evaluation_bundle(
            output_dir=output,
            project_name=str(
                config.get("project_name")
                or config.get("run_name")
                or f"{target} model arena"
            ),
            task=task,
            target=target,
            smiles_column=smiles_column,
            target_transform=str(dataset_config.get("target_transform", "none")),
            protocol_name=protocol_name,
            validation=validation,
            test=test,
            outcomes=outcomes,
            evaluation_config=evaluation_config,
            seed=seed,
        )
    write_json(
        output / "benchmark_metadata.json",
        {
            "task": task,
            "target": target,
            "benchmark_mode": str(benchmark_config.get("mode", "screening")),
            "selection_metric": selection_metric,
            "selected_model": selected_model,
            "selection_protocol": "development HPO, common validation selection, final test only",
            "accelerator": accelerator or "not_used",
            "split": loaded_splits.metadata,
            "validation_split": validation_metadata,
            "train": train_summary,
            "test": test_summary,
        },
    )
    resolved_config = {key: value for key, value in config.items() if key != "_config_dir"}
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False)
    )
    _write_benchmark_report(
        output,
        comparison,
        selected_model,
        task,
        target,
        selection_metric,
        loaded_splits.metadata,
    )
    return BenchmarkResult(output_dir=output, selected_model=selected_model, comparison=comparison)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune and compare molecular models on shared validation and test splits."
    )
    parser.add_argument("config", type=Path, help="Model-arena YAML configuration")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; overrides output_dir in the YAML file",
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.output_dir is not None:
        # Command-line paths follow normal shell semantics and are relative to
        # the current working directory. YAML paths stay relative to the YAML.
        output = arguments.output_dir.expanduser().resolve()
    elif config.get("output_dir") is not None:
        output = resolve_config_path(config["output_dir"], config).resolve()
    else:
        raise ValueError("Set output_dir in YAML or pass --output-dir")
    result = run_model_benchmark(config, output)
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "selected_model": result.selected_model,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
