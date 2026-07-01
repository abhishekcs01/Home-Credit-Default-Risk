from __future__ import annotations

import json
import subprocess
import warnings
from contextlib import redirect_stderr
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from io import StringIO
from logging import Logger
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src import config
from src.config import load_config
from src.data.aggregation import merge_stage
from src.data.preprocess import DesignMatrixPreprocessor
from src.features.build_features import engineer_application_features
from src.features.feature_schema import FeatureSchema
from src.features.selection import rank_features
from src.models.calibration import evaluate_calibration_methods, predict_platt
from src.models.evaluate import plot_ensemble_feature_importance, plot_pr_curve, plot_roc_curve
from src.reporting import generate_all_reports
from src.utils import ensure_dir, get_logger, init_logging, save_pickle, timer

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None

try:
    import xgboost as _xgb
except ImportError:  # pragma: no cover
    _xgb = None

xgb: Any = _xgb

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TrainingDevices:
    lgb_device: str
    catboost_task_type: str
    catboost_devices: str | None
    xgb_device: str
    gpu_device_id: int
    requested_device: str


def _nvidia_gpu_available() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


@lru_cache(maxsize=1)
def _lightgbm_cuda_available() -> bool:
    """Return True when LightGBM was built with CUDA and a minimal GPU train succeeds."""
    if not _nvidia_gpu_available():
        return False
    try:
        x = np.array([[1.0, 0.5], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        y = np.array([0, 1, 0], dtype=np.int32)
        ds = lgb.Dataset(x, label=y)
        params = {
            "objective": "binary",
            "device": "cuda",
            "gpu_device_id": 0,
            "verbose": -1,
            "num_leaves": 4,
            "num_threads": 1,
        }
        with redirect_stderr(StringIO()):
            lgb.train(params, ds, num_boost_round=1)
        return True
    except lgb.basic.LightGBMError:
        return False
    except Exception:
        return False


def resolve_training_devices(requested_device: str, gpu_device_id: int = 0) -> TrainingDevices:
    normalized = str(requested_device).strip().lower()
    if normalized not in ("auto", "cpu", "cuda"):
        raise ValueError(f"training.device must be one of auto, cpu, cuda; got {requested_device!r}")

    if normalized == "cpu":
        return TrainingDevices("cpu", "CPU", None, "cpu", gpu_device_id, normalized)

    use_gpu = normalized == "cuda" or (normalized == "auto" and _nvidia_gpu_available())
    if use_gpu:
        if not _lightgbm_cuda_available():
            raise RuntimeError(
                "LightGBM CUDA is required for GPU training but this LightGBM build does not support CUDA. "
                "Install a CUDA-enabled build with: "
                "pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_CUDA=ON"
            )
        return TrainingDevices(
            lgb_device="cuda",
            catboost_task_type="GPU",
            catboost_devices=str(gpu_device_id),
            xgb_device="cuda",
            gpu_device_id=gpu_device_id,
            requested_device=normalized,
        )
    return TrainingDevices("cpu", "CPU", None, "cpu", gpu_device_id, normalized)


def _training_devices_from_config() -> TrainingDevices:
    cfg = load_config()
    return resolve_training_devices(cfg.training.device, cfg.training.gpu_device_id)


def log_training_devices(logger: Logger, devices: TrainingDevices) -> None:
    logger.info("LIGHTGBM BACKEND: %s", "CUDA" if devices.lgb_device == "cuda" else "CPU")
    logger.info("CATBOOST BACKEND: %s", "GPU" if devices.catboost_task_type == "GPU" else "CPU")
    logger.info("XGBOOST BACKEND: %s", "CUDA" if devices.xgb_device == "cuda" else "CPU")
    if devices.requested_device in ("auto", "cuda") and devices.lgb_device == "cpu":
        logger.warning("GPU requested but no NVIDIA GPU detected; using CPU for all boosters.")


def _lgb_binary_params(devices: TrainingDevices, scale_pos_weight: float, random_state: int) -> dict:
    params = dict(config.LGBM_BASE_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight
    params["random_state"] = random_state
    params["device"] = devices.lgb_device
    if devices.lgb_device == "cuda":
        params.pop("force_col_wise", None)
        params["gpu_device_id"] = devices.gpu_device_id
    else:
        params["force_col_wise"] = True
    return params


def _catboost_binary_params(devices: TrainingDevices, scale_pos_weight: float, random_state: int) -> dict:
    from src import config as project_config

    project_config.CATBOOST_TMP_DIR.mkdir(parents=True, exist_ok=True)
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 1200,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 8.0,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.85,
        "random_strength": 0.5,
        "auto_class_weights": None,
        "scale_pos_weight": scale_pos_weight,
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,
        "train_dir": str(project_config.CATBOOST_TMP_DIR),
        "task_type": devices.catboost_task_type,
    }
    if devices.catboost_task_type == "GPU" and devices.catboost_devices is not None:
        params["devices"] = devices.catboost_devices
    return params


def _xgb_binary_params(devices: TrainingDevices, scale_pos_weight: float, random_state: int) -> dict:
    return {
        "n_estimators": 1400,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 4.0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": random_state,
        "n_jobs": 1,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "device": devices.xgb_device,
    }


def _train_lgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    random_state: int,
    devices: TrainingDevices | None = None,
) -> lgb.Booster:
    training_devices = devices or _training_devices_from_config()
    pos, neg = (y_train == 1).sum(), (y_train == 0).sum()
    params = _lgb_binary_params(training_devices, neg / pos if pos else 1.0, random_state)
    dtr = lgb.Dataset(x_train, label=y_train)
    dva = lgb.Dataset(x_valid, label=y_valid, reference=dtr)
    return lgb.train(
        params,
        dtr,
        num_boost_round=5000,
        valid_sets=[dva],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=120), lgb.log_evaluation(period=0)],
    )


def _train_catboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    random_state: int,
    devices: TrainingDevices | None = None,
):
    if CatBoostClassifier is None:
        return None
    training_devices = devices or _training_devices_from_config()
    pos, neg = (y_train == 1).sum(), (y_train == 0).sum()
    model = CatBoostClassifier(**_catboost_binary_params(training_devices, neg / pos if pos else 1.0, random_state))
    model.fit(x_train, y_train, eval_set=(x_valid, y_valid), use_best_model=True)
    return model


def _predict_catboost(model, x: np.ndarray) -> np.ndarray:
    if model is None:
        raise NotFittedError("CatBoost model is unavailable.")
    return model.predict_proba(x)[:, 1].astype(np.float64)


def _train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    random_state: int,
    devices: TrainingDevices | None = None,
):
    if xgb is None:
        return None
    training_devices = devices or _training_devices_from_config()
    pos, neg = (y_train == 1).sum(), (y_train == 0).sum()
    model = xgb.XGBClassifier(**_xgb_binary_params(training_devices, neg / pos if pos else 1.0, random_state))
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    return model


def _predict_xgboost(model, x: np.ndarray) -> np.ndarray:
    if model is None:
        raise NotFittedError("XGBoost model is unavailable.")
    return model.predict_proba(x)[:, 1].astype(np.float64)


def _compute_threshold_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    best = {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    for thr in np.linspace(0.05, 0.95, 19):
        pred = (probs >= thr).astype(int)
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if f1 > best["f1"]:
            best = {
                "threshold": float(thr),
                "f1": f1,
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
            }
    return best


def _optimize_blend_weights(
    oof_lgb: np.ndarray,
    oof_cat: np.ndarray,
    oof_xgb: np.ndarray,
    y_true: np.ndarray,
    *,
    use_xgb: bool,
) -> tuple[float, float, float]:
    best_auc, best = -1.0, (0.5, 0.5, 0.0)
    grid = np.linspace(0.0, 1.0, 21)
    for w_lgb in grid:
        for w_cat in grid:
            w_xgb = 1.0 - w_lgb - w_cat
            if w_xgb < -1e-9:
                continue
            if not use_xgb:
                w_xgb = 0.0
                total = w_lgb + w_cat
                if total <= 0:
                    continue
                w_lgb, w_cat = w_lgb / total, w_cat / total
            blend = w_lgb * oof_lgb + w_cat * oof_cat + w_xgb * oof_xgb
            auc = float(roc_auc_score(y_true, blend))
            if auc > best_auc:
                best_auc, best = auc, (float(w_lgb), float(w_cat), float(w_xgb))
    return best


def _fit_stacker(y_true: np.ndarray, oof_lgb: np.ndarray, oof_cat: np.ndarray, oof_xgb: np.ndarray):
    x_meta = np.column_stack([oof_lgb, oof_cat, oof_xgb])
    model = LogisticRegression(max_iter=1200, solver="lbfgs")
    model.fit(x_meta, y_true)
    pred = model.predict_proba(x_meta)[:, 1]
    return model, pred


def _optuna_tune_lgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    random_state: int,
    devices: TrainingDevices | None = None,
    n_trials: int = 20,
) -> dict | None:
    if optuna is None:
        return None
    training_devices = devices or _training_devices_from_config()

    def objective(trial):
        pos, neg = (y_train == 1).sum(), (y_train == 0).sum()
        params = _lgb_binary_params(training_devices, neg / pos if pos else 1.0, random_state)
        params.update(
            {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 24, 96),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 2.0, log=True),
            }
        )
        dtr = lgb.Dataset(x_train, label=y_train)
        dva = lgb.Dataset(x_valid, label=y_valid, reference=dtr)
        model = lgb.train(
            params,
            dtr,
            num_boost_round=2500,
            valid_sets=[dva],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(stopping_rounds=80), lgb.log_evaluation(period=0)],
        )
        pred = model.predict(x_valid, num_iteration=model.best_iteration)
        return roc_auc_score(y_valid, pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _compute_shap_summary(
    lgb_model: lgb.Booster, x_sample: np.ndarray, feature_names: list[str]
) -> dict[str, dict[str, float]] | None:
    if x_sample.size == 0:
        return None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="numpy.ufunc size changed.*")
            warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"shap\..*")
            import shap  # type: ignore

        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(x_sample)
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[-1], dtype=float)
        else:
            values = np.asarray(shap_values, dtype=float)
        mean_abs = np.mean(np.abs(values), axis=0)
        mean_signed = np.mean(values, axis=0)
        order = np.argsort(mean_abs)[::-1][:20]
        return {
            feature_names[i]: {
                "mean_abs": float(mean_abs[i]),
                "mean_signed": float(mean_signed[i]),
            }
            for i in order
            if i < len(feature_names)
        }
    except Exception:
        return None


@dataclass
class FoldPreparedData:
    fold_id: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    preprocessor: DesignMatrixPreprocessor
    feature_names: list[str]
    x_train: np.ndarray
    y_train: np.ndarray
    x_valid: np.ndarray
    y_valid: np.ndarray


def _logreg_holdout_and_cv(
    x_train: np.ndarray, x_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, *, random_state: int
) -> tuple[float, float, float]:
    scaler_lr = StandardScaler()
    x_train_lr = scaler_lr.fit_transform(x_train)
    x_val_lr = scaler_lr.transform(x_val)
    log_reg = LogisticRegression(max_iter=600, solver="liblinear", dual=False, C=0.1, random_state=random_state)
    log_reg.fit(x_train_lr, y_train)
    auc_lr = float(roc_auc_score(y_val, log_reg.predict_proba(x_val_lr)[:, 1]))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    fold_aucs = []
    for tr_idx, va_idx in skf.split(x_train, y_train):
        x_tr, x_va = x_train[tr_idx], x_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        sc = StandardScaler()
        x_tr_s, x_va_s = sc.fit_transform(x_tr), sc.transform(x_va)
        m = LogisticRegression(max_iter=600, solver="liblinear", dual=False, C=0.1, random_state=random_state)
        m.fit(x_tr_s, y_tr)
        fold_aucs.append(roc_auc_score(y_va, m.predict_proba(x_va_s)[:, 1]))
    return auc_lr, float(np.mean(fold_aucs)), float(np.std(fold_aucs))


def _prepare_fold_data(
    df: pd.DataFrame,
    *,
    n_splits: int,
    random_state: int,
    ohe_max_categories: int,
    miss_drop_threshold: float,
) -> tuple[list[FoldPreparedData], np.ndarray]:
    y_all = df["TARGET"].astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    prepared: list[FoldPreparedData] = []
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(df)), y_all), start=1):
        df_tr = df.iloc[tr_idx].copy()
        df_va = df.iloc[va_idx].copy()
        prep = DesignMatrixPreprocessor(
            ohe_max_categories=ohe_max_categories,
            miss_drop_threshold=miss_drop_threshold,
        ).fit(df_tr)
        x_tr = prep.transform(df_tr)
        x_va = prep.transform(df_va)
        prepared.append(
            FoldPreparedData(
                fold_id=fold_id,
                train_idx=tr_idx,
                valid_idx=va_idx,
                preprocessor=prep,
                feature_names=list(prep.feature_names_),
                x_train=x_tr,
                y_train=y_all[tr_idx],
                x_valid=x_va,
                y_valid=y_all[va_idx],
            )
        )
    return prepared, y_all


def _accumulate_importance(
    totals: dict[str, float],
    counts: dict[str, int],
    feature_names: list[str],
    values: np.ndarray,
) -> None:
    for name, v in zip(feature_names, values, strict=False):
        totals[name] = totals.get(name, 0.0) + float(v)
        counts[name] = counts.get(name, 0) + 1


def _select_stable_features(
    lgb_totals: dict[str, float],
    lgb_counts: dict[str, int],
    cat_totals: dict[str, float],
    cat_counts: dict[str, int],
    *,
    prune_bottom_frac: float,
) -> set[str] | None:
    if prune_bottom_frac <= 0:
        return None
    mean_scores: dict[str, float] = {}
    all_names = set(lgb_totals) | set(cat_totals)
    for name in all_names:
        lgb_mean = lgb_totals.get(name, 0.0) / max(1, lgb_counts.get(name, 0))
        cat_mean = cat_totals.get(name, 0.0) / max(1, cat_counts.get(name, 0))
        mean_scores[name] = 0.7 * lgb_mean + 0.3 * cat_mean
    if len(mean_scores) < 80:
        return None
    ordered = sorted(mean_scores.items(), key=lambda kv: kv[1])
    n_drop = max(1, int(len(ordered) * prune_bottom_frac))
    selected = {k for k, _ in ordered[n_drop:]}
    return selected if len(selected) >= max(60, int(0.42 * len(ordered))) else None


def evaluate_models(
    df_merged: pd.DataFrame,
    *,
    random_state: int = config.RANDOM_STATE,
    ohe_max_categories: int = config.OHE_MAX_CATEGORIES,
    test_size: float = 0.2,
    miss_drop_threshold: float = config.MISS_DROP_THRESHOLD,
    prune_bottom_frac: float = 0.0,
    devices: TrainingDevices | None = None,
) -> dict:
    _ = prune_bottom_frac
    training_devices = devices or _training_devices_from_config()
    train_df, val_df = train_test_split(
        df_merged,
        test_size=test_size,
        random_state=random_state,
        stratify=df_merged["TARGET"],
    )
    prep = DesignMatrixPreprocessor(
        ohe_max_categories=ohe_max_categories,
        miss_drop_threshold=miss_drop_threshold,
    ).fit(train_df)
    x_train, y_train = prep.transform(train_df), train_df["TARGET"].astype(int).to_numpy()
    x_val, y_val = prep.transform(val_df), val_df["TARGET"].astype(int).to_numpy()

    auc_lr, mean_cv_lr, std_cv_lr = _logreg_holdout_and_cv(x_train, x_val, y_train, y_val, random_state=random_state)
    gbm = _train_lgb(x_train, y_train, x_val, y_val, random_state=random_state, devices=training_devices)
    best_it = gbm.best_iteration if gbm.best_iteration is not None else -1
    prob_val_lgb = np.asarray(gbm.predict(x_val, num_iteration=best_it), dtype=np.float64)
    prob_train_lgb = np.asarray(gbm.predict(x_train, num_iteration=best_it), dtype=np.float64)

    cat_model = _train_catboost(
        x_train, y_train, x_val, y_val, random_state=random_state, devices=training_devices
    )
    if cat_model is None:
        prob_val_cat = np.zeros_like(prob_val_lgb)
        prob_train_cat = np.zeros_like(prob_train_lgb)
        blend_w_lgb, blend_w_cat = 1.0, 0.0
    else:
        prob_val_cat = np.asarray(_predict_catboost(cat_model, x_val), dtype=np.float64)
        prob_train_cat = np.asarray(_predict_catboost(cat_model, x_train), dtype=np.float64)
        blend_w_lgb, blend_w_cat = 0.5, 0.5

    prob_val_blend = blend_w_lgb * prob_val_lgb + blend_w_cat * prob_val_cat
    prob_train_blend = blend_w_lgb * prob_train_lgb + blend_w_cat * prob_train_cat

    return {
        "auc_lgb": float(roc_auc_score(y_val, prob_val_lgb)),
        "auc_cat": float(roc_auc_score(y_val, prob_val_cat)) if cat_model is not None else None,
        "auc_blend": float(roc_auc_score(y_val, prob_val_blend)),
        "auc_lr": auc_lr,
        "mean_cv_lr": mean_cv_lr,
        "std_cv_lr": std_cv_lr,
        "train_auc_lgb": float(roc_auc_score(y_train, prob_train_lgb)),
        "train_auc_blend": float(roc_auc_score(y_train, prob_train_blend)),
        "best_iteration": best_it,
        "gbm": gbm,
        "cat_model": cat_model,
        "feature_names": list(prep.feature_names_),
        "feature_keep_indices": None,
        "x_val": x_val,
        "y_val": y_val,
        "val_pred": prob_val_blend,
        "val_pred_lgb": prob_val_lgb,
        "val_pred_cat": prob_val_cat,
    }


def run_incremental_table_study(
    application_train: pd.DataFrame,
    *,
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame,
    previous_application: pd.DataFrame,
    installments_payments: pd.DataFrame,
    pos_cash_balance: pd.DataFrame,
    credit_card_balance: pd.DataFrame,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    stages = [
        ("1) application only", "app"),
        ("2) + bureau & bureau_balance", "bureau"),
        ("3) + previous_application", "previous"),
        ("4) + installments_payments", "installments"),
        ("5) + POS_CASH_balance", "pos_cash"),
        ("6) + credit_card_balance", "credit_card"),
    ]
    aux = dict(
        bureau=bureau,
        bureau_balance=bureau_balance,
        previous_application=previous_application,
        installments_payments=installments_payments,
        pos_cash_balance=pos_cash_balance,
        credit_card_balance=credit_card_balance,
    )

    rows = []
    for label, key in stages:
        merged = engineer_application_features(merge_stage(application_train.copy(), key, **aux))
        ev = evaluate_models(
            merged,
            random_state=random_state,
            ohe_max_categories=config.OHE_MAX_CATEGORIES,
            miss_drop_threshold=config.MISS_DROP_THRESHOLD,
            prune_bottom_frac=0.0,
        )
        rows.append(
            {
                "Stage": label,
                "ROC-AUC (LGBM, holdout)": round(ev["auc_lgb"], 4),
                "Train AUC (LGBM)": round(ev["train_auc_lgb"], 4),
                "Train-val gap": round(ev["train_auc_lgb"] - ev["auc_lgb"], 4),
                "ROC-AUC (LogReg, holdout)": round(ev["auc_lr"], 4),
                "LogReg 5-fold mean": round(ev["mean_cv_lr"], 4),
                "n_features": len(ev["feature_names"]),
            }
        )
    return pd.DataFrame(rows)


def train_kfold_lightgbm_ensemble(
    merged_train: pd.DataFrame,
    *,
    feature_keep_indices: np.ndarray | None = None,
    n_splits: int = config.N_FOLD_ENSEMBLE,
    random_state: int = config.RANDOM_STATE,
    ohe_max_categories: int = config.OHE_MAX_CATEGORIES,
    miss_drop_threshold: float = config.MISS_DROP_THRESHOLD,
    prune_bottom_frac: float = config.PRUNE_BOTTOM_FRAC,
    blend_weights: tuple[float, ...] = (0.5, 0.5),
    use_xgboost: bool = config.ENABLE_XGBOOST,
    enable_stacking: bool = config.ENABLE_STACKING,
    enable_calibration: bool = config.ENABLE_CALIBRATION,
    enable_optuna: bool = config.ENABLE_OPTUNA,
    enable_shap: bool = config.ENABLE_SHAP,
    devices: TrainingDevices | None = None,
) -> dict:
    training_devices = devices or _training_devices_from_config()
    prepared_folds, y_all = _prepare_fold_data(
        merged_train,
        n_splits=n_splits,
        random_state=random_state,
        ohe_max_categories=ohe_max_categories,
        miss_drop_threshold=miss_drop_threshold,
    )
    lgb_totals: dict[str, float] = {}
    lgb_counts: dict[str, int] = {}
    cat_totals: dict[str, float] = {}
    cat_counts: dict[str, int] = {}

    # Feature selection uses fold-0 importance only (single training pass).
    selected_names = None
    if prune_bottom_frac > 0 and prepared_folds:
        fd0 = prepared_folds[0]
        probe_lgb = _train_lgb(
            fd0.x_train, fd0.y_train, fd0.x_valid, fd0.y_valid, random_state=random_state, devices=training_devices
        )
        _accumulate_importance(
            lgb_totals,
            lgb_counts,
            fd0.feature_names,
            np.asarray(probe_lgb.feature_importance(importance_type="gain"), dtype=float),
        )
        probe_cat = _train_catboost(
            fd0.x_train, fd0.y_train, fd0.x_valid, fd0.y_valid, random_state=random_state, devices=training_devices
        )
        if probe_cat is not None:
            _accumulate_importance(
                cat_totals,
                cat_counts,
                fd0.feature_names,
                np.asarray(probe_cat.get_feature_importance(type="FeatureImportance"), dtype=float),
            )
        selected_names = _select_stable_features(
            lgb_totals,
            lgb_counts,
            cat_totals,
            cat_counts,
            prune_bottom_frac=prune_bottom_frac,
        )

    oof_lgb = np.zeros(len(y_all), dtype=np.float64)
    oof_cat = np.zeros(len(y_all), dtype=np.float64)
    oof_blend = np.zeros(len(y_all), dtype=np.float64)
    fold_models = []
    fold_metrics = []
    ensemble_importance_rows = []

    if len(blend_weights) >= 3:
        raw_w_lgb, raw_w_cat, raw_w_xgb = blend_weights[0], blend_weights[1], blend_weights[2]
    elif len(blend_weights) == 2:
        raw_w_lgb, raw_w_cat, raw_w_xgb = blend_weights[0], blend_weights[1], 0.0
    else:
        raw_w_lgb, raw_w_cat, raw_w_xgb = 1.0, 0.0, 0.0
    if CatBoostClassifier is None:
        raw_w_cat = 0.0
    if xgb is None or not use_xgboost:
        raw_w_xgb = 0.0
    w_sum = raw_w_lgb + raw_w_cat + raw_w_xgb if (raw_w_lgb + raw_w_cat + raw_w_xgb) > 0 else 1.0
    w_lgb, w_cat, w_xgb = raw_w_lgb / w_sum, raw_w_cat / w_sum, raw_w_xgb / w_sum

    tuned_lgb_params: dict | None = None
    if enable_optuna and prepared_folds:
        fd0 = prepared_folds[0]
        tuned_lgb_params = _optuna_tune_lgb(
            fd0.x_train,
            fd0.y_train,
            fd0.x_valid,
            fd0.y_valid,
            random_state=random_state,
            devices=training_devices,
            n_trials=12,
        )

    for fd in prepared_folds:
        keep_idx = None
        names = fd.feature_names
        x_tr, x_va = fd.x_train, fd.x_valid
        if feature_keep_indices is not None:
            keep_idx = feature_keep_indices
            x_tr, x_va = x_tr[:, keep_idx], x_va[:, keep_idx]
            names = [names[i] for i in keep_idx]
        elif selected_names is not None:
            keep_idx = np.array([i for i, n in enumerate(names) if n in selected_names], dtype=int)
            if keep_idx.size >= max(20, int(0.25 * len(names))):
                x_tr, x_va = x_tr[:, keep_idx], x_va[:, keep_idx]
                names = [names[i] for i in keep_idx]
            else:
                keep_idx = None

        if tuned_lgb_params:
            params = _lgb_binary_params(
                training_devices,
                ((fd.y_train == 0).sum() / max((fd.y_train == 1).sum(), 1)),
                random_state,
            )
            params.update(tuned_lgb_params)
            dtr = lgb.Dataset(x_tr, label=fd.y_train)
            dva = lgb.Dataset(x_va, label=fd.y_valid, reference=dtr)
            lgb_model = lgb.train(
                params,
                dtr,
                num_boost_round=5000,
                valid_sets=[dva],
                valid_names=["valid"],
                callbacks=[lgb.early_stopping(stopping_rounds=120), lgb.log_evaluation(period=0)],
            )
        else:
            lgb_model = _train_lgb(
                x_tr, fd.y_train, x_va, fd.y_valid, random_state=random_state, devices=training_devices
            )
        pred_lgb = lgb_model.predict(x_va, num_iteration=lgb_model.best_iteration)
        if CatBoostClassifier is not None:
            cat_model = _train_catboost(
                x_tr, fd.y_train, x_va, fd.y_valid, random_state=random_state, devices=training_devices
            )
            pred_cat = _predict_catboost(cat_model, x_va)
            cat_imp = np.asarray(cat_model.get_feature_importance(type="FeatureImportance"), dtype=float)
        else:
            cat_model = None
            pred_cat = np.zeros_like(pred_lgb)
            cat_imp = np.zeros(len(names), dtype=float)
        if xgb is not None and use_xgboost:
            xgb_model = _train_xgboost(
                x_tr, fd.y_train, x_va, fd.y_valid, random_state=random_state, devices=training_devices
            )
            pred_xgb = _predict_xgboost(xgb_model, x_va)
            xgb_imp = np.asarray(getattr(xgb_model, "feature_importances_", np.zeros(len(names))), dtype=float)
        else:
            xgb_model = None
            pred_xgb = np.zeros_like(pred_lgb)
            xgb_imp = np.zeros(len(names), dtype=float)
        pred_blend = (
            w_lgb * np.asarray(pred_lgb, dtype=np.float64)
            + w_cat * np.asarray(pred_cat, dtype=np.float64)
            + w_xgb * np.asarray(pred_xgb, dtype=np.float64)
        )

        va_idx = fd.valid_idx
        oof_lgb[va_idx] = pred_lgb
        oof_cat[va_idx] = pred_cat
        if "oof_xgb" not in locals():
            oof_xgb = np.zeros(len(y_all), dtype=np.float64)
        oof_xgb[va_idx] = pred_xgb
        oof_blend[va_idx] = pred_blend
        fold_metrics.append(
            {
                "fold": fd.fold_id,
                "auc_lgb": float(roc_auc_score(fd.y_valid, pred_lgb)),
                "auc_cat": float(roc_auc_score(fd.y_valid, pred_cat)) if cat_model is not None else None,
                "auc_xgb": float(roc_auc_score(fd.y_valid, pred_xgb)) if xgb_model is not None else None,
                "auc_blend": float(roc_auc_score(fd.y_valid, pred_blend)),
                "n_features": int(x_tr.shape[1]),
            }
        )
        fold_models.append(
            {
                "preprocessor": fd.preprocessor,
                "feature_names": names,
                "feature_keep_indices": keep_idx,
                "lgb_model": lgb_model,
                "cat_model": cat_model,
                "xgb_model": xgb_model,
                "fold_id": fd.fold_id,
            }
        )
        lgb_imp = np.asarray(lgb_model.feature_importance(importance_type="gain"), dtype=float)
        for i, nm in enumerate(names):
            ensemble_importance_rows.append(
                {
                    "feature": nm,
                    "fold": fd.fold_id,
                    "lgb_gain": float(lgb_imp[i]),
                    "cat_importance": float(cat_imp[i]) if i < len(cat_imp) else 0.0,
                    "xgb_importance": float(xgb_imp[i]) if i < len(xgb_imp) else 0.0,
                }
            )
    if "oof_xgb" not in locals():
        oof_xgb = np.zeros(len(y_all), dtype=np.float64)

    opt_w_lgb, opt_w_cat, opt_w_xgb = _optimize_blend_weights(
        oof_lgb, oof_cat, oof_xgb, y_all, use_xgb=(xgb is not None and use_xgboost)
    )
    oof_blend = opt_w_lgb * oof_lgb + opt_w_cat * oof_cat + opt_w_xgb * oof_xgb
    w_lgb, w_cat, w_xgb = opt_w_lgb, opt_w_cat, opt_w_xgb

    stack_model = None
    oof_stack = None
    if enable_stacking:
        stack_model, oof_stack = _fit_stacker(y_all, oof_lgb, oof_cat, oof_xgb)
    calibrator = None
    calibration_method = None
    calibration_report: dict[str, Any] = {}
    oof_calibrated = None
    if enable_calibration:
        base_probs = oof_stack if oof_stack is not None else oof_blend
        iso_result, platt_result, best_method = evaluate_calibration_methods(y_all, base_probs)
        calibration_report = {
            "isotonic_brier": iso_result.brier_score,
            "isotonic_auc": iso_result.roc_auc,
            "platt_brier": platt_result.brier_score,
            "platt_auc": platt_result.roc_auc,
            "best_method": best_method,
        }
        calibration_method = best_method
        calibrator = iso_result.model if best_method == "isotonic" else platt_result.model
        oof_calibrated = (
            calibrator.predict(base_probs)
            if best_method == "isotonic"
            else predict_platt(calibrator, base_probs)
        )
    threshold_metrics = _compute_threshold_metrics(y_all, oof_calibrated if oof_calibrated is not None else oof_blend)
    shap_summary = None
    if enable_shap and fold_models:
        first = fold_models[0]
        sample_x = prepared_folds[0].x_valid[: min(512, len(prepared_folds[0].x_valid))]
        if first.get("feature_keep_indices") is not None:
            sample_x = sample_x[:, first["feature_keep_indices"]]
        shap_summary = _compute_shap_summary(first["lgb_model"], sample_x, first["feature_names"])

    return {
        "fold_models": fold_models,
        "blend_weights": {"lgb": float(w_lgb), "cat": float(w_cat), "xgb": float(w_xgb)},
        "feature_keep_indices": feature_keep_indices,
        "selected_feature_names": sorted(selected_names) if selected_names is not None else None,
        "feature_names_final": fold_models[0]["feature_names"] if fold_models else [],
        "y_all": y_all,
        "oof_pred_lgb": oof_lgb,
        "oof_pred_cat": oof_cat,
        "oof_pred_xgb": oof_xgb,
        "oof_pred_blend": oof_blend,
        "oof_pred_stack": oof_stack,
        "oof_pred_calibrated": oof_calibrated,
        "oof_auc_lgb": float(roc_auc_score(y_all, oof_lgb)),
        "oof_auc_cat": float(roc_auc_score(y_all, oof_cat)) if CatBoostClassifier is not None else None,
        "oof_auc_xgb": float(roc_auc_score(y_all, oof_xgb)) if (xgb is not None and use_xgboost) else None,
        "oof_auc_blend": float(roc_auc_score(y_all, oof_blend)),
        "oof_auc_stack": float(roc_auc_score(y_all, oof_stack)) if oof_stack is not None else None,
        "oof_auc_calibrated": float(roc_auc_score(y_all, oof_calibrated)) if oof_calibrated is not None else None,
        "fold_metrics": fold_metrics,
        "fold_artifacts": [
            {
                "fold": int(m.get("fold") or 0),
                "n_features": int(m.get("n_features") or 0),
                "auc_blend": float(m.get("auc_blend") or 0.0),
            }
            for m in fold_metrics
        ],
        "stacking_model": stack_model,
        "calibration_model": calibrator,
        "calibration_method": calibration_method,
        "calibration_report": calibration_report,
        "threshold_analysis": threshold_metrics,
        "shap_importance_top": shap_summary,
        "optuna_best_params": tuned_lgb_params,
        "ensemble_feature_importance": pd.DataFrame(ensemble_importance_rows),
    }


def run_training(*, demo_mode: bool | None = None) -> Path:
    cfg = load_config()
    logger = get_logger("train_model")
    if demo_mode is None:
        from src.data.load_data import using_synthetic_data

        demo_mode = using_synthetic_data()

    if not config.MERGED_TRAIN_PATH.exists():
        logger.warning(
            "Processed train dataset missing at %s; running preprocessing first.",
            config.MERGED_TRAIN_PATH,
        )
        from src.data.pipeline import ensure_processed_datasets

        ensure_processed_datasets(force_recompute=demo_mode)

    training_devices = resolve_training_devices(cfg.training.device, cfg.training.gpu_device_id)
    log_training_devices(logger, training_devices)
    if demo_mode:
        logger.warning(
            "Demo/synthetic training mode enabled (fewer folds, SHAP disabled) because Kaggle raw data is absent."
        )
    if not config.MERGED_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing processed train dataset at {config.MERGED_TRAIN_PATH}.")
    logger.info("Training started")

    n_splits = min(3, cfg.training.n_folds) if demo_mode else cfg.training.n_folds
    enable_shap = False if demo_mode else cfg.training.enable_shap

    with timer("Load processed training data"):
        merged_train = pd.read_pickle(config.MERGED_TRAIN_PATH)

    with timer("Train k-fold ensemble"):
        training_output = train_kfold_lightgbm_ensemble(
            merged_train,
            random_state=cfg.training.random_state,
            ohe_max_categories=cfg.training.ohe_max_categories,
            miss_drop_threshold=cfg.training.miss_drop_threshold,
            prune_bottom_frac=cfg.training.prune_bottom_frac,
            n_splits=n_splits,
            blend_weights=(
                cfg.training.blend_weight_lgb,
                cfg.training.blend_weight_cat,
                cfg.training.blend_weight_xgb,
            ),
            use_xgboost=cfg.training.enable_xgboost,
            enable_stacking=cfg.training.enable_stacking,
            enable_calibration=cfg.training.enable_calibration,
            enable_optuna=cfg.training.enable_optuna,
            enable_shap=enable_shap,
            devices=training_devices,
        )

    schema = FeatureSchema.from_dataframe(merged_train, version="train-v1")
    feature_ranking = None
    if training_output.get("fold_models"):
        fm0 = training_output["fold_models"][0]
        prep0 = fm0["preprocessor"]
        sample_df = merged_train.sample(n=min(8000, len(merged_train)), random_state=cfg.training.random_state)
        x_sample = prep0.transform(sample_df)
        keep = fm0.get("feature_keep_indices")
        if keep is not None:
            x_sample = x_sample[:, keep]
        names = fm0["feature_names"]
        y_sample = sample_df["TARGET"].astype(int).to_numpy()
        ranking = rank_features(
            x_sample,
            y_sample,
            names,
            lgb_model=fm0["lgb_model"],
            shap_summary=training_output.get("shap_importance_top"),
            prune_bottom_frac=0.0,
            random_state=cfg.training.random_state,
        )
        feature_ranking = ranking.to_dataframe()

    metrics = {
        "oof_auc_blend": training_output["oof_auc_blend"],
        "oof_auc_lgb": training_output["oof_auc_lgb"],
        "oof_auc_cat": training_output["oof_auc_cat"],
        "oof_auc_xgb": training_output.get("oof_auc_xgb"),
        "oof_auc_stack": training_output.get("oof_auc_stack"),
        "oof_auc_calibrated": training_output.get("oof_auc_calibrated"),
        "n_features_final": len(training_output["feature_names_final"]),
        "blend_weights": training_output["blend_weights"],
        "calibration_method": training_output.get("calibration_method"),
        "version": cfg.api.version,
    }
    threshold = float(training_output["threshold_analysis"]["threshold"])
    bundle = {
        "model": {
            "fold_models": training_output["fold_models"],
            "blend_weights": training_output["blend_weights"],
            "stacking_model": training_output.get("stacking_model"),
            "calibration_model": training_output.get("calibration_model"),
            "calibration_method": training_output.get("calibration_method"),
        },
        "preprocessor": {"kind": "fold_preprocessors", "count": len(training_output["fold_models"])},
        "feature_names": list(training_output["feature_names_final"]),
        "metrics": metrics,
        "threshold": threshold,
        "version": cfg.api.version,
        "schema_version": "1.0",
        "trained_at": datetime.now(UTC).isoformat(),
        "feature_schema": schema.model_dump(),
        # backward-compatible fields for existing prediction code paths
        **training_output,
    }

    ensure_dir(config.MODELS_DIR)
    ensure_dir(config.FIGURES_DIR)
    ensure_dir(config.METRICS_DIR)
    save_pickle(bundle, config.MODEL_BUNDLE_PATH)
    plot_roc_curve(training_output["y_all"], training_output["oof_pred_blend"], output_path=config.ROC_CURVE_PATH)
    plot_pr_curve(training_output["y_all"], training_output["oof_pred_blend"], output_path=config.PR_CURVE_PATH)
    _, fi_table = plot_ensemble_feature_importance(
        training_output["ensemble_feature_importance"], output_path=config.FEATURE_IMPORTANCE_PATH
    )

    config.TRAINING_METRICS_JSON_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    config.TRAINING_CONFIG_SNAPSHOT_PATH.write_text(
        json.dumps(
            {
                "random_state": cfg.training.random_state,
                "device": cfg.training.device,
                "gpu_device_id": cfg.training.gpu_device_id,
                "lgb_device": training_devices.lgb_device,
                "catboost_task_type": training_devices.catboost_task_type,
                "xgb_device": training_devices.xgb_device,
                "n_folds": n_splits,
                "blend_weights": training_output["blend_weights"],
                "enable_stacking": cfg.training.enable_stacking,
                "enable_calibration": cfg.training.enable_calibration,
                "enable_optuna": cfg.training.enable_optuna,
                "enable_shap": enable_shap,
                "enable_xgboost": cfg.training.enable_xgboost,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    fi_table.head(15).to_csv(config.FEATURE_IMPORTANCE_TOP15_CSV_PATH, index=False)
    pd.DataFrame(training_output["fold_metrics"]).to_csv(config.FOLD_METRICS_CSV_PATH, index=False)
    if training_output.get("shap_importance_top"):
        (config.METRICS_DIR / "shap_summary.json").write_text(
            json.dumps(training_output["shap_importance_top"], indent=2),
            encoding="utf-8",
        )
    generate_all_reports(
        training_output,
        metrics,
        calibration_report=training_output.get("calibration_report"),
        feature_ranking=feature_ranking,
        bundle_meta={
            "version": cfg.api.version,
            "schema_version": "1.0",
            "trained_at": bundle["trained_at"],
            "feature_count": metrics["n_features_final"],
            "oof_auc_calibrated": metrics.get("oof_auc_calibrated"),
        },
    )
    logger.info("Training completed. Saved model bundle to %s", config.MODEL_BUNDLE_PATH)
    return config.MODEL_BUNDLE_PATH


def main() -> None:
    init_logging()
    run_training()


__all__ = [
    "TrainingDevices",
    "evaluate_models",
    "log_training_devices",
    "resolve_training_devices",
    "run_incremental_table_study",
    "run_training",
    "train_kfold_lightgbm_ensemble",
]


if __name__ == "__main__":
    main()
