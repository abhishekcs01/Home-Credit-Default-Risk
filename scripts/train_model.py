from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.evaluation import plot_ensemble_feature_importance, plot_pr_curve, plot_roc_curve
from src.model_training import train_kfold_lightgbm_ensemble
from src.utils import ensure_dir, get_logger, init_logging, save_pickle, timer


def _config_snapshot() -> dict:
    snapshot_keys = [
        "RANDOM_STATE",
        "OHE_MAX_CATEGORIES",
        "MISS_DROP_THRESHOLD",
        "PRUNE_BOTTOM_FRAC",
        "N_FOLD_ENSEMBLE",
        "BLEND_WEIGHT_LGB",
        "BLEND_WEIGHT_CAT",
        "LGBM_BASE_PARAMS",
    ]
    out = {}
    for key in snapshot_keys:
        out[key] = getattr(config, key, None)
    return out


def run_training() -> Path:
    logger = get_logger("train_model")
    if not config.MERGED_TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Missing processed train dataset at {config.MERGED_TRAIN_PATH}. "
            "Run scripts/preprocess_data.py first."
        )

    with timer("Load processed training data"):
        logger.info("Loading processed training data")
        merged_train = pd.read_pickle(config.MERGED_TRAIN_PATH)
        logger.info("Train shape: %s", merged_train.shape)

    with timer("K-fold ensemble training (leakage-safe preprocessing + blending)"):
        logger.info("Training %d-fold LightGBM+CatBoost ensemble", config.N_FOLD_ENSEMBLE)
        bundle = train_kfold_lightgbm_ensemble(
            merged_train,
            random_state=config.RANDOM_STATE,
            ohe_max_categories=config.OHE_MAX_CATEGORIES,
            miss_drop_threshold=config.MISS_DROP_THRESHOLD,
            prune_bottom_frac=config.PRUNE_BOTTOM_FRAC,
            n_splits=config.N_FOLD_ENSEMBLE,
            blend_weights=(config.BLEND_WEIGHT_LGB, config.BLEND_WEIGHT_CAT),
        )

    ensure_dir(config.MODELS_DIR)
    ensure_dir(config.FIGURES_DIR)
    ensure_dir(config.METRICS_DIR)

    with timer("Save model + plots + metrics"):
        save_pickle(bundle, config.MODEL_BUNDLE_PATH)
        plot_roc_curve(bundle["y_all"], bundle["oof_pred_blend"], output_path=config.ROC_CURVE_PATH)
        plot_pr_curve(bundle["y_all"], bundle["oof_pred_blend"], output_path=config.PR_CURVE_PATH)
        _, fi_table = plot_ensemble_feature_importance(
            bundle["ensemble_feature_importance"],
            output_path=config.FEATURE_IMPORTANCE_PATH,
        )
        fold_df = pd.DataFrame(bundle["fold_metrics"])
        fold_df.to_csv(config.FOLD_METRICS_CSV_PATH, index=False)
        bundle["ensemble_feature_importance"].to_csv(config.METRICS_DIR / "ensemble_feature_importance_all_folds.csv", index=False)

        metrics = {
            "oof_auc_blend": bundle["oof_auc_blend"],
            "oof_auc_lgb": bundle["oof_auc_lgb"],
            "oof_auc_cat": bundle["oof_auc_cat"],
            "n_features_first_fold_final": len(bundle["feature_names_final"]),
            "blend_weights": bundle["blend_weights"],
        }
        config.TRAINING_METRICS_JSON_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        fi_table.head(15).to_csv(config.FEATURE_IMPORTANCE_TOP15_CSV_PATH, index=False)
        metadata = {
            "trained_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "n_rows_train": int(len(merged_train)),
            "n_columns_train": int(merged_train.shape[1]),
            "model_bundle_path": str(config.MODEL_BUNDLE_PATH),
        }
        config.TRAINING_METADATA_JSON_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        config.TRAINING_CONFIG_SNAPSHOT_PATH.write_text(json.dumps(_config_snapshot(), indent=2), encoding="utf-8")

    logger.info("Saved model bundle: %s", config.MODEL_BUNDLE_PATH)
    logger.info("Saved ROC curve: %s", config.ROC_CURVE_PATH)
    logger.info("Saved PR curve: %s", config.PR_CURVE_PATH)
    logger.info("Saved feature importance figure: %s", config.FEATURE_IMPORTANCE_PATH)
    logger.info("Saved metrics: %s", config.TRAINING_METRICS_JSON_PATH)
    logger.info("Saved feature importance table: %s", config.FEATURE_IMPORTANCE_TOP15_CSV_PATH)
    return config.MODEL_BUNDLE_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Home Credit LightGBM model from processed train data.")
    return parser.parse_args()


def main() -> None:
    init_logging()
    _ = parse_args()
    run_training()


if __name__ == "__main__":
    main()
