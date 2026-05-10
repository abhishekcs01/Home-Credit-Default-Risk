"""Central configuration: paths, constants, and model hyperparameters.

All scripts and ``src`` modules resolve filesystem locations through this module.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (repository root = directory containing this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

MODELS_DIR = PROJECT_ROOT / "models"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
METRICS_DIR = OUTPUTS_DIR / "metrics"
LOGS_DIR = OUTPUTS_DIR / "logs"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"

# Raw CSV inputs
APPLICATION_TRAIN_PATH = RAW_DATA_DIR / "application_train.csv"
APPLICATION_TEST_PATH = RAW_DATA_DIR / "application_test.csv"
BUREAU_PATH = RAW_DATA_DIR / "bureau.csv"
BUREAU_BALANCE_PATH = RAW_DATA_DIR / "bureau_balance.csv"
PREVIOUS_APPLICATION_PATH = RAW_DATA_DIR / "previous_application.csv"
INSTALLMENTS_PAYMENTS_PATH = RAW_DATA_DIR / "installments_payments.csv"
POS_CASH_BALANCE_PATH = RAW_DATA_DIR / "POS_CASH_balance.csv"
CREDIT_CARD_BALANCE_PATH = RAW_DATA_DIR / "credit_card_balance.csv"

# Processed artifacts
MERGED_TRAIN_PATH = PROCESSED_DATA_DIR / "merged_train.pkl"
MERGED_TEST_PATH = PROCESSED_DATA_DIR / "merged_test.pkl"
MERGED_TRAIN_CSV_PATH = PROCESSED_DATA_DIR / "merged_train.csv"
MERGED_TEST_CSV_PATH = PROCESSED_DATA_DIR / "merged_test.csv"

# Training outputs
MODEL_BUNDLE_PATH = MODELS_DIR / "lightgbm_model.pkl"
ROC_CURVE_PATH = FIGURES_DIR / "roc_curve.png"
PR_CURVE_PATH = FIGURES_DIR / "pr_curve.png"
FEATURE_IMPORTANCE_PATH = FIGURES_DIR / "feature_importance.png"
TRAINING_METRICS_JSON_PATH = METRICS_DIR / "training_metrics.json"
FOLD_METRICS_CSV_PATH = METRICS_DIR / "fold_metrics.csv"
FEATURE_IMPORTANCE_TOP15_CSV_PATH = METRICS_DIR / "feature_importance_top15.csv"
TRAINING_METADATA_JSON_PATH = METRICS_DIR / "training_metadata.json"
TRAINING_CONFIG_SNAPSHOT_PATH = METRICS_DIR / "config_snapshot.json"

# Inference — canonical submission location (+ timestamped copies beside it)
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission.csv"

# ---------------------------------------------------------------------------
# Modeling / preprocessing constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
SENTINEL_DAYS = 365_243
OHE_MAX_CATEGORIES = 15
MISS_DROP_THRESHOLD = 0.70
PRUNE_BOTTOM_FRAC = 0.06
N_FOLD_ENSEMBLE = 5
BLEND_WEIGHT_LGB = 0.5
BLEND_WEIGHT_CAT = 0.5

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
