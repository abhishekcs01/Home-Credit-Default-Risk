from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.evaluation import plot_feature_importance, plot_roc_curve
from src.model_training import evaluate_models, train_kfold_lightgbm_ensemble
from src.utils import ensure_dir, get_logger, init_logging, save_pickle, timer


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

    with timer("Validation split + feature pruning"):
        logger.info("Running validation and feature pruning")
        validation = evaluate_models(
            merged_train,
            random_state=config.RANDOM_STATE,
            ohe_max_categories=config.OHE_MAX_CATEGORIES,
            miss_drop_threshold=config.MISS_DROP_THRESHOLD,
            prune_bottom_frac=config.PRUNE_BOTTOM_FRAC,
        )

    with timer("K-fold LightGBM ensemble training"):
        logger.info("Training %d-fold LightGBM ensemble", config.N_FOLD_ENSEMBLE)
        bundle = train_kfold_lightgbm_ensemble(
            merged_train,
            feature_keep_indices=validation["feature_keep_indices"],
            n_splits=config.N_FOLD_ENSEMBLE,
            random_state=config.RANDOM_STATE,
        )

    ensure_dir(config.MODELS_DIR)
    ensure_dir(config.FIGURES_DIR)
    ensure_dir(config.METRICS_DIR)

    with timer("Save model + plots + metrics"):
        save_pickle(bundle, config.MODEL_BUNDLE_PATH)
        plot_roc_curve(validation["y_val"], validation["val_pred"], output_path=config.ROC_CURVE_PATH)
        _, fi_table = plot_feature_importance(
            bundle["models"][-1],
            bundle["feature_names_final"],
            output_path=config.FEATURE_IMPORTANCE_PATH,
        )

        metrics = {
            "holdout_auc_lgb": validation["auc_lgb"],
            "holdout_auc_logreg": validation["auc_lr"],
            "oof_auc_lgb_kfold": bundle["oof_auc"],
            "best_iteration_holdout": validation["best_iteration"],
            "n_features_final": len(bundle["feature_names_final"]),
        }
        config.TRAINING_METRICS_JSON_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        fi_table.head(15).to_csv(config.FEATURE_IMPORTANCE_TOP15_CSV_PATH, index=False)

    logger.info("Saved model bundle: %s", config.MODEL_BUNDLE_PATH)
    logger.info("Saved ROC curve: %s", config.ROC_CURVE_PATH)
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
