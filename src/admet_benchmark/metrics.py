"""Task metrics used after a model has been selected on validation only."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return challenge-aligned and rank-correlation regression metrics."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    baseline_mae = float(np.mean(np.abs(y_true - np.mean(y_true))))
    return {
        "mae": mae,
        "relative_absolute_error": mae / baseline_mae if baseline_mae else float("nan"),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman_r": _spearman_r(y_true, y_pred),
        "kendall_tau": _kendall_tau(y_true, y_pred),
    }


def _kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Use SciPy lazily so metric-only consumers need no eager import."""

    from scipy.stats import kendalltau

    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    result = kendalltau(y_true, y_pred)
    return float(result.statistic)


def _spearman_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Use tie-aware ranks rather than positional ranks from ``argsort``."""

    from scipy.stats import spearmanr

    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    result = spearmanr(y_true, y_pred)
    return float(result.statistic)


def classification_metrics(
    y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Return threshold-free and thresholded binary-classification metrics."""

    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if not 0 <= threshold <= 1:
        raise ValueError("Classification threshold must be between 0 and 1")
    y_pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    negative_predictive_value = tn / (tn + fn) if tn + fn else float("nan")
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "negative_predictive_value": float(negative_predictive_value),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
