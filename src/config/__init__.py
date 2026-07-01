from __future__ import annotations

from pathlib import Path

from src.config.settings import AppConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
OUTPUTS_DIR = ARTIFACTS_DIR
SUBMISSIONS_DIR = ARTIFACTS_DIR / "submissions"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
CATBOOST_TMP_DIR = ARTIFACTS_DIR / "tmp" / "catboost"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"

APPLICATION_TRAIN_PATH = RAW_DATA_DIR / "application_train.csv"
APPLICATION_TEST_PATH = RAW_DATA_DIR / "application_test.csv"
BUREAU_PATH = RAW_DATA_DIR / "bureau.csv"
BUREAU_BALANCE_PATH = RAW_DATA_DIR / "bureau_balance.csv"
PREVIOUS_APPLICATION_PATH = RAW_DATA_DIR / "previous_application.csv"
INSTALLMENTS_PAYMENTS_PATH = RAW_DATA_DIR / "installments_payments.csv"
POS_CASH_BALANCE_PATH = RAW_DATA_DIR / "POS_CASH_balance.csv"
CREDIT_CARD_BALANCE_PATH = RAW_DATA_DIR / "credit_card_balance.csv"

MERGED_TRAIN_PATH = PROCESSED_DATA_DIR / "merged_train.pkl"
MERGED_TEST_PATH = PROCESSED_DATA_DIR / "merged_test.pkl"
MERGED_TRAIN_CSV_PATH = PROCESSED_DATA_DIR / "merged_train.csv"
MERGED_TEST_CSV_PATH = PROCESSED_DATA_DIR / "merged_test.csv"

MODEL_BUNDLE_PATH = MODELS_DIR / "ensemble_model.pkl"
ROC_CURVE_PATH = FIGURES_DIR / "roc_curve.png"
PR_CURVE_PATH = FIGURES_DIR / "pr_curve.png"
FEATURE_IMPORTANCE_PATH = FIGURES_DIR / "feature_importance.png"
TRAINING_METRICS_JSON_PATH = METRICS_DIR / "training_metrics.json"
FOLD_METRICS_CSV_PATH = METRICS_DIR / "fold_metrics.csv"
FEATURE_IMPORTANCE_TOP15_CSV_PATH = METRICS_DIR / "feature_importance_top15.csv"
TRAINING_METADATA_JSON_PATH = METRICS_DIR / "training_metadata.json"
TRAINING_CONFIG_SNAPSHOT_PATH = METRICS_DIR / "config_snapshot.json"

SUBMISSION_PATH = SUBMISSIONS_DIR / "submission.csv"

RANDOM_STATE = 42
SENTINEL_DAYS = 365_243
OHE_MAX_CATEGORIES = 15
MISS_DROP_THRESHOLD = 0.70
PRUNE_BOTTOM_FRAC = 0.06
N_FOLD_ENSEMBLE = 5
BLEND_WEIGHT_LGB = 0.5
BLEND_WEIGHT_CAT = 0.5
BLEND_WEIGHT_XGB = 0.0
ENABLE_XGBOOST = False
ENABLE_STACKING = True
ENABLE_CALIBRATION = True
ENABLE_OPTUNA = False
ENABLE_SHAP = False

LGBM_BASE_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.022,
    "num_leaves": 42,
    "max_depth": -1,
    "min_child_samples": 70,
    "subsample": 0.75,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.35,
    "reg_lambda": 0.35,
    "min_gain_to_split": 0.02,
    "verbosity": -1,
    "force_col_wise": True,
}

__all__ = [
    "AppConfig",
    "load_config",
]
