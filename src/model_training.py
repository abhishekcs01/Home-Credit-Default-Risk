from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src import config
from src.aggregation import merge_stage
from src.feature_engineering import engineer_application_features
from src.preprocessing import DesignMatrixPreprocessor

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover - optional in test environment
    CatBoostClassifier = None


def _lgb_binary_params(scale_pos_weight: float, random_state: int) -> dict:
    params = dict(config.LGBM_BASE_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight
    params["random_state"] = random_state
    return params


def _catboost_binary_params(scale_pos_weight: float, random_state: int) -> dict:
    return {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 1200,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 8.0,
        "subsample": 0.85,
        "random_strength": 0.5,
        "auto_class_weights": None,
        "scale_pos_weight": scale_pos_weight,
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,
    }


def _train_lgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    random_state: int,
) -> lgb.Booster:
    pos, neg = (y_train == 1).sum(), (y_train == 0).sum()
    params = _lgb_binary_params(neg / pos if pos else 1.0, random_state)
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
):
    if CatBoostClassifier is None:
        return None
    pos, neg = (y_train == 1).sum(), (y_train == 0).sum()
    model = CatBoostClassifier(**_catboost_binary_params(neg / pos if pos else 1.0, random_state))
    model.fit(x_train, y_train, eval_set=(x_valid, y_valid), use_best_model=True)
    return model


def _predict_catboost(model, x: np.ndarray) -> np.ndarray:
    if model is None:
        raise NotFittedError("CatBoost model is unavailable.")
    return model.predict_proba(x)[:, 1].astype(np.float64)


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
) -> dict:
    _ = prune_bottom_frac  # kept for backward-compatible function signature
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
    gbm = _train_lgb(x_train, y_train, x_val, y_val, random_state=random_state)
    best_it = gbm.best_iteration if gbm.best_iteration is not None else -1
    prob_val_lgb = gbm.predict(x_val, num_iteration=best_it)
    prob_train_lgb = gbm.predict(x_train, num_iteration=best_it)

    cat_model = _train_catboost(x_train, y_train, x_val, y_val, random_state=random_state)
    if cat_model is None:
        prob_val_cat = np.zeros_like(prob_val_lgb)
        prob_train_cat = np.zeros_like(prob_train_lgb)
        blend_w_lgb, blend_w_cat = 1.0, 0.0
    else:
        prob_val_cat = _predict_catboost(cat_model, x_val)
        prob_train_cat = _predict_catboost(cat_model, x_train)
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
    blend_weights: tuple[float, float] = (0.5, 0.5),
) -> dict:
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

    first_pass = []
    for fd in prepared_folds:
        lgb_model = _train_lgb(fd.x_train, fd.y_train, fd.x_valid, fd.y_valid, random_state=random_state)
        lgb_pred = lgb_model.predict(fd.x_valid, num_iteration=lgb_model.best_iteration)
        _accumulate_importance(
            lgb_totals,
            lgb_counts,
            fd.feature_names,
            np.asarray(lgb_model.feature_importance(importance_type="gain"), dtype=float),
        )
        cat_model = _train_catboost(fd.x_train, fd.y_train, fd.x_valid, fd.y_valid, random_state=random_state)
        if cat_model is not None:
            cat_imp = np.asarray(cat_model.get_feature_importance(type="FeatureImportance"), dtype=float)
            _accumulate_importance(cat_totals, cat_counts, fd.feature_names, cat_imp)
            cat_pred = _predict_catboost(cat_model, fd.x_valid)
        else:
            cat_pred = np.zeros_like(lgb_pred)
        first_pass.append((fd, lgb_pred, cat_pred))

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

    raw_w_lgb, raw_w_cat = blend_weights
    if CatBoostClassifier is None:
        raw_w_lgb, raw_w_cat = 1.0, 0.0
    w_sum = raw_w_lgb + raw_w_cat if (raw_w_lgb + raw_w_cat) > 0 else 1.0
    w_lgb, w_cat = raw_w_lgb / w_sum, raw_w_cat / w_sum

    for fd, _, _ in first_pass:
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

        lgb_model = _train_lgb(x_tr, fd.y_train, x_va, fd.y_valid, random_state=random_state)
        pred_lgb = lgb_model.predict(x_va, num_iteration=lgb_model.best_iteration)
        if CatBoostClassifier is not None:
            cat_model = _train_catboost(x_tr, fd.y_train, x_va, fd.y_valid, random_state=random_state)
            pred_cat = _predict_catboost(cat_model, x_va)
            cat_imp = np.asarray(cat_model.get_feature_importance(type="FeatureImportance"), dtype=float)
        else:
            cat_model = None
            pred_cat = np.zeros_like(pred_lgb)
            cat_imp = np.zeros(len(names), dtype=float)
        pred_blend = w_lgb * pred_lgb + w_cat * pred_cat

        va_idx = fd.valid_idx
        oof_lgb[va_idx] = pred_lgb
        oof_cat[va_idx] = pred_cat
        oof_blend[va_idx] = pred_blend
        fold_metrics.append(
            {
                "fold": fd.fold_id,
                "auc_lgb": float(roc_auc_score(fd.y_valid, pred_lgb)),
                "auc_cat": float(roc_auc_score(fd.y_valid, pred_cat)) if cat_model is not None else None,
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
                }
            )

    return {
        "fold_models": fold_models,
        "blend_weights": {"lgb": float(w_lgb), "cat": float(w_cat)},
        "feature_keep_indices": feature_keep_indices,
        "selected_feature_names": sorted(selected_names) if selected_names is not None else None,
        "feature_names_final": fold_models[0]["feature_names"] if fold_models else [],
        "y_all": y_all,
        "oof_pred_lgb": oof_lgb,
        "oof_pred_cat": oof_cat,
        "oof_pred_blend": oof_blend,
        "oof_auc_lgb": float(roc_auc_score(y_all, oof_lgb)),
        "oof_auc_cat": float(roc_auc_score(y_all, oof_cat)) if CatBoostClassifier is not None else None,
        "oof_auc_blend": float(roc_auc_score(y_all, oof_blend)),
        "fold_metrics": fold_metrics,
        "ensemble_feature_importance": pd.DataFrame(ensemble_importance_rows),
    }


__all__ = [
    "evaluate_models",
    "run_incremental_table_study",
    "train_kfold_lightgbm_ensemble",
]

