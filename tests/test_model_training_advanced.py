from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

import src.models.train as mt


class _DummyLGB:
    def __init__(self, n_features: int):
        self.best_iteration = 1
        self._n_features = n_features

    def predict(self, x: np.ndarray, num_iteration=None) -> np.ndarray:
        base = np.asarray(x[:, 0], dtype=float)
        return np.clip((base - np.nanmin(base)) / (np.nanmax(base) - np.nanmin(base) + 1e-6), 0, 1)

    def feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        assert importance_type == "gain"
        return np.linspace(1.0, float(self._n_features), self._n_features, dtype=float)


class _DummyCat:
    def __init__(self, n_features: int):
        self._n_features = n_features

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.clip(0.2 + 0.1 * np.tanh(x[:, 0]), 0, 1)
        return np.column_stack([1 - p, p])

    def get_feature_importance(self, type: str = "FeatureImportance") -> np.ndarray:
        assert type == "FeatureImportance"
        return np.linspace(1.0, float(self._n_features), self._n_features, dtype=float)


class _DummyXGB:
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.clip(0.3 + 0.1 * np.tanh(x[:, 1]), 0, 1)
        return np.column_stack([1 - p, p])

    @property
    def feature_importances_(self):
        return np.array([0.2, 0.8], dtype=float)


def _wide_training_df(n: int = 120, m: int = 95) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    data = {"SK_ID_CURR": np.arange(n), "TARGET": rng.integers(0, 2, size=n)}
    for i in range(m):
        data[f"F_{i:03d}"] = rng.normal(loc=0.0, scale=1.0, size=n)
    data["ORGANIZATION_TYPE"] = np.array([f"ORG_{i % 40}" for i in range(n)], dtype=object)
    data["NAME_INCOME_TYPE"] = np.array([f"INC_{i % 5}" for i in range(n)], dtype=object)
    return pd.DataFrame(data)


def test_helper_metrics_and_feature_selection_paths():
    y = np.array([0, 1, 1, 0, 1, 0], dtype=int)
    p = np.array([0.1, 0.9, 0.8, 0.2, 0.6, 0.4], dtype=float)
    best = mt._compute_threshold_metrics(y, p)
    assert 0.05 <= best["threshold"] <= 0.95
    assert 0.0 <= best["f1"] <= 1.0

    stack_model, stack_pred = mt._fit_stacker(y, p, p * 0.95, p * 1.05)
    assert stack_model is not None
    assert stack_pred.shape == y.shape

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    mt._accumulate_importance(totals, counts, ["a", "b"], np.array([1.0, 2.0]))
    assert totals["a"] == 1.0 and counts["b"] == 1

    assert mt._select_stable_features({}, {}, {}, {}, prune_bottom_frac=0.1) is None
    assert mt._select_stable_features({"x": 1.0}, {"x": 1}, {"x": 1.0}, {"x": 1}, prune_bottom_frac=0.0) is None

    many = {f"f{i}": float(i + 1) for i in range(100)}
    sel = mt._select_stable_features(many, {k: 1 for k in many}, many, {k: 1 for k in many}, prune_bottom_frac=0.1)
    assert sel is not None and len(sel) >= 60


def test_optional_train_predict_paths_and_errors(monkeypatch):
    x = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=float)
    y = np.array([0, 1], dtype=int)

    monkeypatch.setattr(mt, "CatBoostClassifier", None)
    assert mt._train_catboost(x, y, x, y, random_state=42) is None
    with pytest.raises(NotFittedError):
        mt._predict_catboost(None, x)

    monkeypatch.setattr(mt, "xgb", None)
    assert mt._train_xgboost(x, y, x, y, random_state=42) is None
    with pytest.raises(NotFittedError):
        mt._predict_xgboost(None, x)


def test_optuna_and_shap_paths(monkeypatch):
    x = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=float)
    y = np.array([0, 1, 0, 1], dtype=int)

    assert mt._optuna_tune_lgb(x, y, x, y, random_state=42, n_trials=2) is None if mt.optuna is None else True

    class _FakeStudy:
        best_params = {"learning_rate": 0.05}

        def optimize(self, objective, n_trials: int, show_progress_bar: bool):
            objective(types.SimpleNamespace(suggest_float=lambda *a, **k: 0.05, suggest_int=lambda *a, **k: 32))

    class _FakeOptuna:
        @staticmethod
        def create_study(direction: str):
            assert direction == "maximize"
            return _FakeStudy()

    monkeypatch.setattr(mt, "optuna", _FakeOptuna())
    monkeypatch.setattr(mt.lgb, "Dataset", lambda data, label, reference=None: {"data": data, "label": label})
    monkeypatch.setattr(mt.lgb, "early_stopping", lambda stopping_rounds: None)
    monkeypatch.setattr(mt.lgb, "log_evaluation", lambda period: None)
    monkeypatch.setattr(mt.lgb, "train", lambda params, dtr, num_boost_round, valid_sets, valid_names, callbacks: _DummyLGB(dtr["data"].shape[1]))
    tuned = mt._optuna_tune_lgb(x, y, x, y, random_state=7, n_trials=1)
    assert tuned is not None and "learning_rate" in tuned

    assert mt._compute_shap_summary(_DummyLGB(2), np.empty((0, 2)), ["f1", "f2"]) is None

    fake_shap = types.SimpleNamespace(
        TreeExplainer=lambda model: types.SimpleNamespace(shap_values=lambda sample: np.abs(sample))
    )
    monkeypatch.setitem(sys.modules, "shap", fake_shap)
    shap_summary = mt._compute_shap_summary(_DummyLGB(2), x, ["f1", "f2"])
    assert shap_summary is not None and set(shap_summary).issubset({"f1", "f2"})
    assert "mean_abs" in shap_summary["f1"]

    fake_bad_shap = types.SimpleNamespace(TreeExplainer=lambda model: (_ for _ in ()).throw(RuntimeError("bad")))
    monkeypatch.setitem(sys.modules, "shap", fake_bad_shap)
    assert mt._compute_shap_summary(_DummyLGB(2), x, ["f1", "f2"]) is None


def test_incremental_study_and_kfold_training(monkeypatch):
    train_df = _wide_training_df()

    monkeypatch.setattr(mt, "merge_stage", lambda app, key, **aux: app.copy())
    monkeypatch.setattr(mt, "engineer_application_features", lambda df: df.copy())
    monkeypatch.setattr(
        mt,
        "evaluate_models",
        lambda *a, **k: {
            "auc_lgb": 0.7,
            "train_auc_lgb": 0.8,
            "auc_lr": 0.65,
            "mean_cv_lr": 0.64,
            "feature_names": ["f1", "f2"],
        },
    )
    study = mt.run_incremental_table_study(
        train_df,
        bureau=pd.DataFrame(),
        bureau_balance=pd.DataFrame(),
        previous_application=pd.DataFrame(),
        installments_payments=pd.DataFrame(),
        pos_cash_balance=pd.DataFrame(),
        credit_card_balance=pd.DataFrame(),
    )
    assert len(study) == 6

    monkeypatch.setattr(mt, "CatBoostClassifier", object())
    monkeypatch.setattr(mt, "xgb", object())
    monkeypatch.setattr(mt, "_train_lgb", lambda x_tr, y_tr, x_va, y_va, random_state, **kwargs: _DummyLGB(x_tr.shape[1]))
    monkeypatch.setattr(mt, "_train_catboost", lambda x_tr, y_tr, x_va, y_va, random_state, **kwargs: _DummyCat(x_tr.shape[1]))
    monkeypatch.setattr(mt, "_train_xgboost", lambda x_tr, y_tr, x_va, y_va, random_state, **kwargs: _DummyXGB())
    monkeypatch.setattr(mt, "_compute_shap_summary", lambda *a, **k: {"F_000": {"mean_abs": 1.0, "mean_signed": 0.5}})
    monkeypatch.setattr(mt, "_optuna_tune_lgb", lambda *a, **k: {"learning_rate": 0.05})

    monkeypatch.setattr(mt.lgb, "Dataset", lambda data, label, reference=None: {"data": data, "label": label})
    monkeypatch.setattr(mt.lgb, "early_stopping", lambda stopping_rounds: None)
    monkeypatch.setattr(mt.lgb, "log_evaluation", lambda period: None)
    monkeypatch.setattr(mt.lgb, "train", lambda params, dtr, num_boost_round, valid_sets, valid_names, callbacks: _DummyLGB(dtr["data"].shape[1]))

    out = mt.train_kfold_lightgbm_ensemble(
        train_df,
        n_splits=3,
        random_state=42,
        ohe_max_categories=5,
        miss_drop_threshold=0.95,
        prune_bottom_frac=0.1,
        blend_weights=(0.3, 0.3, 0.4),
        use_xgboost=True,
        enable_stacking=True,
        enable_calibration=True,
        enable_optuna=True,
        enable_shap=True,
    )
    assert len(out["fold_models"]) == 3
    assert out["oof_auc_blend"] >= 0.0
    assert out["oof_pred_stack"] is not None
    assert out["oof_pred_calibrated"] is not None
    assert out["shap_importance_top"] is not None
    assert out["optuna_best_params"] is not None
