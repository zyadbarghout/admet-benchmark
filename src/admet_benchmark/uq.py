"""Explicit calibration primitives for fold-isolated predictive uncertainty."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression


def conformal_quantile(scores, confidence=.9):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError('Calibration scores must be a nonempty finite nonnegative vector')
    if not 0 < confidence < 1:
        raise ValueError('Confidence must be between zero and one')
    rank = int(np.ceil((len(scores) + 1) * confidence))
    if rank > len(scores):
        raise ValueError('Insufficient calibration support for a finite quantile at this confidence')
    return float(np.sort(scores)[rank-1])


def logit(probability):
    p = np.clip(np.asarray(probability), 1e-7, 1-1e-7)
    return np.log(p/(1-p)).reshape(-1, 1)


class ProbabilityCalibrator:
    """Sigmoid mapping fitted only to the dedicated probability-calibration role."""
    def fit(self, probability, target):
        if set(np.unique(target)) != {0, 1}:
            raise ValueError('Probability calibration requires both classes')
        self.model = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000).fit(logit(probability), target)
        return self

    def predict(self, probability):
        return self.model.predict_proba(logit(probability))[:, 1]


def regression_calibration(target, prediction, dispersion, confidence, floor):
    residual = np.abs(np.asarray(target)-prediction)
    return {'confidence': confidence, 'floor': float(floor),
            'absolute_q': conformal_quantile(residual, confidence),
            'normalized_q': conformal_quantile(residual/np.maximum(dispersion, floor), confidence),
            'n_calibration': len(residual)}


def regression_intervals(prediction, dispersion, calibration):
    width = calibration['normalized_q'] * np.maximum(dispersion, calibration['floor'])
    return prediction-width, prediction+width


def classification_calibration(target, probability, confidence):
    target = np.asarray(target, dtype=int)
    score = np.where(target == 1, 1-probability, probability)
    return {'confidence': confidence, 'class_q': [conformal_quantile(score[target==c], confidence) for c in (0,1)],
            'class_support': [int(np.sum(target==c)) for c in (0,1)]}


def prediction_sets(probability, calibration):
    p = np.asarray(probability)
    return np.column_stack([p <= calibration['class_q'][0], 1-p <= calibration['class_q'][1]])


def interval_score(target, lower, upper, confidence):
    y = np.asarray(target)
    return np.asarray(upper)-lower + 2/(1-confidence)*(np.maximum(lower-y,0)+np.maximum(y-upper,0))
