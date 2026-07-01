from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.config import load_config
from src.models.calibration import predict_platt
from src.utils import get_logger, init_logging, load_pickle

LOGGER = get_logger("predict")

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
    cfg = load_config()
    if num_folds <= 1:
        return 1
    return max(1, min(num_folds, cfg.api.max_concurrent_inferences))


def _unwrap_model_bundle(model_bundle: dict[str, Any]) -> dict[str, Any]:
    if "model" in model_bundle and isinstance(model_bundle["model"], dict):
        return model_bundle["model"]
    return model_bundle


def _predict_single_fold(
    fm: dict[str, Any],
    merged_test: pd.DataFrame,
    w_lgb: float,
    w_cat: float,
    w_xgb: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    xgb_model = fm.get("xgb_model")
    if xgb_model is not None:
        xgb_pred = xgb_model.predict_proba(x_test)[:, 1].astype(np.float64)
    else:
        xgb_pred = np.zeros_like(lgb_pred)
    blend = w_lgb * lgb_pred + w_cat * cat_pred + w_xgb * xgb_pred
    return lgb_pred, cat_pred, xgb_pred, blend


def predict_with_ensemble(model_bundle: dict[str, Any], merged_test: pd.DataFrame) -> pd.DataFrame:
    active_bundle = _unwrap_model_bundle(model_bundle)
    fold_models = active_bundle.get("fold_models")
    if fold_models is None:
        prep = active_bundle["preprocessor"]
        models = active_bundle["models"]
        keep_idx = active_bundle.get("feature_keep_indices")
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

    blend = active_bundle.get("blend_weights", {"lgb": 0.5, "cat": 0.5, "xgb": 0.0})
    w_lgb = float(blend.get("lgb", 0.5))
    w_cat = float(blend.get("cat", 0.5))
    w_xgb = float(blend.get("xgb", 0.0))
    n_folds = len(fold_models)
    workers = _max_fold_workers(n_folds)
    worker_fn = partial(_predict_single_fold, merged_test=merged_test, w_lgb=w_lgb, w_cat=w_cat, w_xgb=w_xgb)

    if workers <= 1 or n_folds <= 1:
        fold_parts = [worker_fn(fm) for fm in fold_models]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fold_parts = list(pool.map(worker_fn, fold_models))

    lgb_mean = np.mean(np.column_stack([p[0] for p in fold_parts]), axis=1)
    cat_mean = np.mean(np.column_stack([p[1] for p in fold_parts]), axis=1)
    xgb_mean = np.mean(np.column_stack([p[2] for p in fold_parts]), axis=1)
    predictions = np.mean(np.column_stack([p[3] for p in fold_parts]), axis=1)

    stacking_model = active_bundle.get("stacking_model")
    if stacking_model is not None:
        x_meta = np.column_stack([lgb_mean, cat_mean, xgb_mean])
        predictions = stacking_model.predict_proba(x_meta)[:, 1]
    calibration_model = active_bundle.get("calibration_model")
    calibration_method = active_bundle.get("calibration_method", "isotonic")
    if calibration_model is not None:
        if calibration_method == "platt":
            predictions = predict_platt(calibration_model, predictions)
        else:
            predictions = calibration_model.predict(predictions)
    return pd.DataFrame({"SK_ID_CURR": merged_test["SK_ID_CURR"], "TARGET": predictions})


class EnsembleInferenceEngine:
    def __init__(self, model_bundle_path: Path):
        self.model_bundle_path = model_bundle_path
        self._bundle: dict[str, Any] | None = None
        self._bundle_load_lock = threading.Lock()

    @property
    def bundle(self) -> dict[str, Any]:
        if self._bundle is None:
            with self._bundle_load_lock:
                if self._bundle is None:
                    LOGGER.info("Loading model bundle from %s", self.model_bundle_path)
                    self._bundle = load_pickle(self.model_bundle_path)
        return self._bundle

    @property
    def bundle_loaded(self) -> bool:
        return self._bundle is not None

    def predict_dataframe(self, features: pd.DataFrame) -> pd.DataFrame:
        LOGGER.info("Running inference for %d records", len(features))
        return predict_with_ensemble(self.bundle, features)


def run_submission(*, output_path: Path | None = None) -> Path:
    if not config.MODEL_BUNDLE_PATH.exists():
        from src.data.pipeline import ensure_model_bundle

        ensure_model_bundle()
    if not config.MERGED_TEST_PATH.exists():
        from src.data.pipeline import ensure_processed_datasets

        ensure_processed_datasets()
    model_bundle = load_pickle(config.MODEL_BUNDLE_PATH)
    merged_test = pd.read_pickle(config.MERGED_TEST_PATH)
    submission = predict_with_ensemble(model_bundle, merged_test)
    target_path = output_path or config.SUBMISSION_PATH
    submission.to_csv(target_path, index=False)
    LOGGER.info("Saved submission: %s (rows=%d)", target_path, len(submission))
    return target_path


def main() -> None:
    init_logging()
    run_submission()


if __name__ == "__main__":
    main()
