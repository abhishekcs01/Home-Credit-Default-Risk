from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class CalibrationResult:
    method: str
    brier_score: float
    roc_auc: float
    model: object


def fit_platt_scaler(y_true: np.ndarray, y_prob: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(y_prob.reshape(-1, 1), y_true)
    return lr


def predict_platt(model: LogisticRegression, y_prob: np.ndarray) -> np.ndarray:
    return model.predict_proba(y_prob.reshape(-1, 1))[:, 1]


def evaluate_calibration_methods(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[CalibrationResult, CalibrationResult, str]:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob, y_true)
    iso_pred = iso.predict(y_prob)

    platt = fit_platt_scaler(y_true, y_prob)
    platt_pred = predict_platt(platt, y_prob)

    iso_result = CalibrationResult(
        method="isotonic",
        brier_score=float(brier_score_loss(y_true, iso_pred)),
        roc_auc=float(roc_auc_score(y_true, iso_pred)),
        model=iso,
    )
    platt_result = CalibrationResult(
        method="platt",
        brier_score=float(brier_score_loss(y_true, platt_pred)),
        roc_auc=float(roc_auc_score(y_true, platt_pred)),
        model=platt,
    )

    best = iso_result if iso_result.brier_score <= platt_result.brier_score else platt_result
    return iso_result, platt_result, best.method


def calibration_curve_data(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10) -> dict:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return {
        "fraction_of_positives": prob_true.tolist(),
        "mean_predicted_value": prob_pred.tolist(),
    }
