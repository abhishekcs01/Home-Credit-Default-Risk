from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.predict import (
    EnsembleInferenceEngine,
    _locked_cat_predict_proba,
    _max_fold_workers,
    predict_with_ensemble,
)


class _Prep:
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [
                np.asarray(df["F1"], dtype=float),
                np.asarray(df["F2"], dtype=float),
            ]
        )


class _Lgb:
    best_iteration = 1

    def predict(self, x: np.ndarray, num_iteration=None) -> np.ndarray:
        return np.clip(0.1 + 0.2 * x[:, 0], 0, 1)


class _Cat:
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.clip(0.15 + 0.1 * x[:, 1], 0, 1)
        return np.column_stack([1 - p, p])


class _Xgb:
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.clip(0.2 + 0.05 * x[:, 0] + 0.05 * x[:, 1], 0, 1)
        return np.column_stack([1 - p, p])


class _Stack:
    def predict_proba(self, x_meta: np.ndarray) -> np.ndarray:
        p = np.clip(np.mean(x_meta, axis=1), 0, 1)
        return np.column_stack([1 - p, p])


class _Cal:
    def predict(self, p: np.ndarray) -> np.ndarray:
        return np.clip(p * 0.95, 0, 1)


def test_locked_cat_predict_and_worker_cap(monkeypatch):
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = _locked_cat_predict_proba(_Cat(), x)
    assert out.shape == (2,)
    assert np.all((out >= 0) & (out <= 1))

    class _Cfg:
        class api:
            max_concurrent_inferences = 2

    monkeypatch.setattr("src.models.predict.load_config", lambda: _Cfg())
    assert _max_fold_workers(1) == 1
    assert _max_fold_workers(5) == 2


def test_predict_with_ensemble_old_bundle_format():
    test_df = pd.DataFrame({"SK_ID_CURR": [10, 11], "F1": [1.0, 2.0], "F2": [0.5, 1.5]})
    bundle = {
        "preprocessor": _Prep(),
        "models": [_Lgb(), _Lgb()],
        "feature_keep_indices": np.array([0], dtype=int),
    }
    pred = predict_with_ensemble(bundle, test_df)
    assert list(pred.columns) == ["SK_ID_CURR", "TARGET"]
    assert pred.shape == (2, 2)
    assert np.all((pred["TARGET"].to_numpy() >= 0) & (pred["TARGET"].to_numpy() <= 1))


def test_predict_with_ensemble_new_bundle_with_stack_and_calibration():
    test_df = pd.DataFrame({"SK_ID_CURR": [20, 21, 22], "F1": [1.0, 2.0, 3.0], "F2": [1.5, 0.5, 1.0]})
    fold = {
        "preprocessor": _Prep(),
        "feature_keep_indices": None,
        "lgb_model": _Lgb(),
        "cat_model": _Cat(),
        "xgb_model": _Xgb(),
    }
    bundle = {
        "fold_models": [fold, fold],
        "blend_weights": {"lgb": 0.4, "cat": 0.4, "xgb": 0.2},
        "stacking_model": _Stack(),
        "calibration_model": _Cal(),
    }
    pred = predict_with_ensemble(bundle, test_df)
    assert pred.shape == (3, 2)
    assert np.all((pred["TARGET"].to_numpy() >= 0) & (pred["TARGET"].to_numpy() <= 1))


def test_ensemble_engine_lazy_bundle_load(monkeypatch, tmp_path):
    load_calls = {"n": 0}

    def fake_load_pickle(_):
        load_calls["n"] += 1
        return {
            "fold_models": [
                {
                    "preprocessor": _Prep(),
                    "feature_keep_indices": None,
                    "lgb_model": _Lgb(),
                    "cat_model": None,
                    "xgb_model": None,
                }
            ],
            "blend_weights": {"lgb": 1.0, "cat": 0.0, "xgb": 0.0},
        }

    monkeypatch.setattr("src.models.predict.load_pickle", fake_load_pickle)

    engine = EnsembleInferenceEngine(tmp_path / "bundle.pkl")
    assert engine.bundle_loaded is False
    features = pd.DataFrame({"SK_ID_CURR": [1], "F1": [2.0], "F2": [1.0]})
    out1 = engine.predict_dataframe(features)
    out2 = engine.predict_dataframe(features)
    assert engine.bundle_loaded is True
    assert load_calls["n"] == 1
    assert out1.equals(out2)
