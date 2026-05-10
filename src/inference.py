from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from src.utils import load_pickle

# CatBoost predict_proba on the *same* classifier instance is not thread-safe; serialize per object id.
# Different folds use different instances → different locks → folds still run in parallel safely.
_cat_predict_locks: dict[int, threading.Lock] = {}
_cat_predict_locks_guard = threading.Lock()


def _locked_cat_predict_proba(cat_model: Any, x_test: np.ndarray) -> np.ndarray:
    key = id(cat_model)
    with _cat_predict_locks_guard:
        lock = _cat_predict_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _cat_predict_locks[key] = lock
    with lock:
        return cat_model.predict_proba(x_test)[:, 1]


def _max_fold_workers(num_folds: int) -> int:
    """ThreadPool size for fold (or legacy model list) inference.

    Default is **1** (sequential): lowest overhead for typical single-row API calls.
    Set ``HOME_CREDIT_INFERENCE_FOLD_WORKERS`` to:
      - a positive integer — cap parallel fold workers
      - ``auto`` / ``max`` / ``0`` — use ``min(n_folds, cpu_count())``
    """
    env = os.getenv("HOME_CREDIT_INFERENCE_FOLD_WORKERS")
    if env is None or not str(env).strip():
        return 1
    raw = str(env).strip().lower()
    if raw in {"auto", "max", "0"}:
        cpu = os.cpu_count() or 4
        return max(1, min(num_folds, cpu))
    return max(1, min(num_folds, int(env)))


def _predict_single_fold(
    fm: dict[str, Any],
    merged_test: pd.DataFrame,
    w_lgb: float,
    w_cat: float,
) -> np.ndarray:
    prep = fm["preprocessor"]
    x_test = prep.transform(merged_test)
    keep_idx = fm.get("feature_keep_indices")
    if keep_idx is not None:
        x_test = x_test[:, keep_idx]
    lgb_model = fm["lgb_model"]
    lgb_pred = lgb_model.predict(x_test, num_iteration=lgb_model.best_iteration)
    cat_model = fm.get("cat_model")
    if cat_model is not None:
        cat_pred = _locked_cat_predict_proba(cat_model, x_test)
    else:
        cat_pred = np.zeros_like(lgb_pred)
    return w_lgb * lgb_pred + w_cat * cat_pred


def predict_with_ensemble(model_bundle: dict, merged_test: pd.DataFrame) -> pd.DataFrame:
    fold_models = model_bundle.get("fold_models")
    if fold_models is None:
        # Backward-compatibility with older bundle format.
        prep = model_bundle["preprocessor"]
        models = model_bundle["models"]
        keep_idx = model_bundle.get("feature_keep_indices")
        x_test = prep.transform(merged_test)
        if keep_idx is not None:
            x_test = x_test[:, keep_idx]
        n_models = len(models)
        workers = _max_fold_workers(n_models)

        def _one(m: Any) -> np.ndarray:
            return m.predict(x_test, num_iteration=m.best_iteration)

        if workers <= 1 or n_models <= 1:
            cols = [_one(m) for m in models]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                cols = list(pool.map(_one, models))
        ensemble_test_preds = np.column_stack(cols)
        predictions = ensemble_test_preds.mean(axis=1)
        return pd.DataFrame({"SK_ID_CURR": merged_test["SK_ID_CURR"], "TARGET": predictions})

    blend = model_bundle.get("blend_weights", {"lgb": 0.5, "cat": 0.5})
    w_lgb = float(blend.get("lgb", 0.5))
    w_cat = float(blend.get("cat", 0.5))
    n_folds = len(fold_models)
    workers = _max_fold_workers(n_folds)
    worker_fn = partial(_predict_single_fold, merged_test=merged_test, w_lgb=w_lgb, w_cat=w_cat)

    if workers <= 1 or n_folds <= 1:
        fold_preds = [worker_fn(fm) for fm in fold_models]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fold_preds = list(pool.map(worker_fn, fold_models))

    predictions = np.mean(np.column_stack(fold_preds), axis=1)
    return pd.DataFrame({"SK_ID_CURR": merged_test["SK_ID_CURR"], "TARGET": predictions})


class EnsembleInferenceEngine:
    """Lightweight helper for API/Locust integration."""

    def __init__(self, model_bundle_path):
        self.model_bundle_path = model_bundle_path
        self._bundle = None
        self._bundle_load_lock = threading.Lock()

    @property
    def bundle(self):
        if self._bundle is None:
            with self._bundle_load_lock:
                if self._bundle is None:
                    self._bundle = load_pickle(self.model_bundle_path)
        return self._bundle

    @property
    def bundle_loaded(self) -> bool:
        return self._bundle is not None

    def predict_dataframe(self, features: pd.DataFrame) -> pd.DataFrame:
        return predict_with_ensemble(self.bundle, features)
