from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src import config
from src.aggregation import merge_stage
from src.feature_engineering import engineer_application_features
from src.preprocessing import DesignMatrixPreprocessor, dataframe_to_design_matrix


def _lgb_binary_params(scale_pos_weight: float, random_state: int) -> dict:
    params = dict(config.LGBM_BASE_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight
    params["random_state"] = random_state
    return params


def _indices_after_gain_prune(gbm: lgb.Booster, bottom_frac: float) -> np.ndarray | None:
    if bottom_frac <= 0:
        return None
    imp = np.asarray(gbm.feature_importance(importance_type="gain"), dtype=float)
    n = len(imp)
    if n < 80:
        return None
    n_drop = max(1, int(n * bottom_frac))
    worst = set(np.argsort(imp)[:n_drop].tolist())
    keep = np.array([i for i in range(n) if i not in worst], dtype=int)
    if keep.size < max(60, int(0.42 * n)):
        return None
    return keep


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


def _lgb_train_val_prune(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    names: list[str],
    *,
    scale_pos_weight: float,
    random_state: int,
    prune_bottom_frac: float,
) -> tuple[lgb.Booster, list[str], np.ndarray | None]:
    params = _lgb_binary_params(scale_pos_weight, random_state)
    train_set = lgb.Dataset(x_train, label=y_train)
    val_set = lgb.Dataset(x_val, label=y_val, reference=train_set)
    gbm = lgb.train(
        params,
        train_set,
        num_boost_round=3000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=120), lgb.log_evaluation(period=0)],
    )

    feat_idx = _indices_after_gain_prune(gbm, prune_bottom_frac)
    if feat_idx is not None:
        x_train, x_val = x_train[:, feat_idx], x_val[:, feat_idx]
        names = [names[i] for i in feat_idx]
        train_set = lgb.Dataset(x_train, label=y_train)
        val_set = lgb.Dataset(x_val, label=y_val, reference=train_set)
        gbm = lgb.train(
            params,
            train_set,
            num_boost_round=3000,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(stopping_rounds=120), lgb.log_evaluation(period=0)],
        )
    return gbm, names, feat_idx


def evaluate_models(
    df_merged: pd.DataFrame,
    *,
    random_state: int = config.RANDOM_STATE,
    ohe_max_categories: int = config.OHE_MAX_CATEGORIES,
    test_size: float = 0.2,
    miss_drop_threshold: float = config.MISS_DROP_THRESHOLD,
    prune_bottom_frac: float = 0.0,
) -> dict:
    prep = DesignMatrixPreprocessor(ohe_max_categories=ohe_max_categories, miss_drop_threshold=miss_drop_threshold)
    prep.fit(df_merged)
    x_full, y_full = prep.X_train_, prep.y_
    names = list(prep.feature_names_)
    x_train, x_val, y_train, y_val = train_test_split(
        x_full, y_full, test_size=test_size, random_state=random_state, stratify=y_full
    )

    auc_lr, mean_cv_lr, std_cv_lr = _logreg_holdout_and_cv(x_train, x_val, y_train, y_val, random_state=random_state)
    pos, neg = (y_train == 1).sum(), (y_train == 0).sum()
    scale_pos_weight = neg / pos if pos else 1.0

    gbm, names, feat_idx = _lgb_train_val_prune(
        x_train,
        y_train,
        x_val,
        y_val,
        names,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        prune_bottom_frac=prune_bottom_frac,
    )
    x_train_eval = x_train[:, feat_idx] if feat_idx is not None else x_train
    x_val_eval = x_val[:, feat_idx] if feat_idx is not None else x_val
    it = gbm.best_iteration if gbm.best_iteration is not None else -1
    prob_val_lgb = gbm.predict(x_val_eval, num_iteration=it)
    prob_train_lgb = gbm.predict(x_train_eval, num_iteration=it)

    return {
        "auc_lgb": float(roc_auc_score(y_val, prob_val_lgb)),
        "auc_lr": auc_lr,
        "mean_cv_lr": mean_cv_lr,
        "std_cv_lr": std_cv_lr,
        "train_auc_lgb": float(roc_auc_score(y_train, prob_train_lgb)),
        "best_iteration": gbm.best_iteration,
        "scale_pos_weight": scale_pos_weight,
        "gbm": gbm,
        "feature_names": names,
        "feature_keep_indices": feat_idx,
        "x_val": x_val_eval,
        "y_val": y_val,
        "val_pred": prob_val_lgb,
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
) -> dict:
    prep = DesignMatrixPreprocessor(
        ohe_max_categories=config.OHE_MAX_CATEGORIES, miss_drop_threshold=config.MISS_DROP_THRESHOLD
    ).fit(merged_train)
    x_all, y_all = prep.X_train_, prep.y_
    if feature_keep_indices is not None:
        x_all = x_all[:, feature_keep_indices]
        feature_names = [prep.feature_names_[i] for i in feature_keep_indices]
    else:
        feature_names = list(prep.feature_names_)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    models, oof_pred, fold_val_aucs = [], np.zeros(len(y_all), dtype=np.float64), []

    for tr_idx, va_idx in skf.split(x_all, y_all):
        x_tr, x_va = x_all[tr_idx], x_all[va_idx]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]
        pos_f, neg_f = (y_tr == 1).sum(), (y_tr == 0).sum()
        params_f = _lgb_binary_params(neg_f / pos_f if pos_f else 1.0, random_state)
        dtr = lgb.Dataset(x_tr, label=y_tr)
        dva = lgb.Dataset(x_va, label=y_va, reference=dtr)
        gbm_f = lgb.train(
            params_f,
            dtr,
            num_boost_round=5000,
            valid_sets=[dva],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(stopping_rounds=120), lgb.log_evaluation(period=0)],
        )
        models.append(gbm_f)
        p_va = gbm_f.predict(x_va, num_iteration=gbm_f.best_iteration)
        oof_pred[va_idx] = p_va
        fold_val_aucs.append(float(roc_auc_score(y_va, p_va)))

    return {
        "models": models,
        "preprocessor": prep,
        "feature_keep_indices": feature_keep_indices,
        "feature_names_final": feature_names,
        "x_all": x_all,
        "y_all": y_all,
        "oof_auc": float(roc_auc_score(y_all, oof_pred)),
        "fold_val_aucs": fold_val_aucs,
    }


__all__ = [
    "dataframe_to_design_matrix",
    "evaluate_models",
    "run_incremental_table_study",
    "train_kfold_lightgbm_ensemble",
]

