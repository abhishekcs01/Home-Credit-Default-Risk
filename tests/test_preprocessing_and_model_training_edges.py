from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

import src.models.train as mt
from src.data.preprocess import DesignMatrixPreprocessor
from src.models.predict import predict_with_ensemble


class _Prep:
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.asarray(df["F1"], dtype=float), np.asarray(df["F2"], dtype=float)])


class _Model:
    best_iteration = 1

    def predict(self, x: np.ndarray, num_iteration=None):
        return np.clip(0.4 + 0.1 * x[:, 0], 0, 1)


def test_preprocessor_edge_paths_and_empty_output():
    with pytest.raises(RuntimeError):
        DesignMatrixPreprocessor().transform(pd.DataFrame({"A": [1]}))

    with pytest.raises(ValueError):
        DesignMatrixPreprocessor().fit(pd.DataFrame({"SK_ID_CURR": [1, 2], "A": [1.0, 2.0]}))

    prep = DesignMatrixPreprocessor()
    prep.drop_cols_ = []
    prep.raw_columns_ = ["Z_DUMMY"]
    prep.cat_cols_ = []
    prep.num_cols_ = []
    prep.medians_ = {}
    prep.target_encoding_cols_ = []
    prep.target_encoding_maps_ = {}
    prep.target_prior_ = 0.5
    prep.low_card_ = []
    prep.ohe_ = None
    out = prep.transform(pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 1]}))
    assert out.shape[1] == 0


def test_inference_old_bundle_threadpool_path(monkeypatch):
    class _Cfg:
        class api:
            max_concurrent_inferences = 4

    monkeypatch.setattr("src.models.predict.load_config", lambda: _Cfg())
    test_df = pd.DataFrame({"SK_ID_CURR": [1, 2], "F1": [1.0, 2.0], "F2": [0.5, 1.5]})
    bundle = {"preprocessor": _Prep(), "models": [_Model(), _Model(), _Model()], "feature_keep_indices": None}
    pred = predict_with_ensemble(bundle, test_df)
    assert pred.shape == (2, 2)


def test_model_training_optional_paths(monkeypatch):
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(60),
            "TARGET": rng.integers(0, 2, size=60),
            "F1": rng.normal(size=60),
            "F2": rng.normal(size=60),
            "ORGANIZATION_TYPE": np.array([f"O{i%7}" for i in range(60)], dtype=object),
        }
    )

    monkeypatch.setattr(mt, "CatBoostClassifier", None)
    monkeypatch.setattr(mt, "xgb", None)
    ev = mt.evaluate_models(df, random_state=7, ohe_max_categories=4, miss_drop_threshold=0.95)
    assert ev["auc_cat"] is None

    monkeypatch.setattr(mt, "_train_lgb", lambda x_tr, y_tr, x_va, y_va, random_state, **kwargs: types.SimpleNamespace(
        best_iteration=1,
        predict=lambda x, num_iteration=None: np.clip(0.3 + 0.2 * x[:, 0], 0, 1),
        feature_importance=lambda importance_type="gain": np.ones(x_tr.shape[1], dtype=float),
    ))

    out = mt.train_kfold_lightgbm_ensemble(
        df,
        n_splits=3,
        random_state=11,
        ohe_max_categories=4,
        miss_drop_threshold=0.95,
        use_xgboost=False,
        enable_stacking=False,
        enable_calibration=False,
        enable_optuna=False,
        enable_shap=False,
    )
    assert out["oof_pred_xgb"] is not None
    assert out["oof_auc_blend"] >= 0.0


def test_train_xgboost_real_path(monkeypatch):
    class _XGBCls:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, x_train, y_train, eval_set=None, verbose=False):
            self.n = x_train.shape[0]
            return self

        def predict_proba(self, x):
            p = np.clip(0.2 + 0.1 * np.tanh(x[:, 0]), 0, 1)
            return np.column_stack([1 - p, p])

    monkeypatch.setattr(mt, "xgb", types.SimpleNamespace(XGBClassifier=_XGBCls))
    x = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=float)
    y = np.array([0, 1, 0], dtype=int)
    model = mt._train_xgboost(x, y, x, y, random_state=42)
    pred = mt._predict_xgboost(model, x)
    assert pred.shape == (3,)
