from __future__ import annotations

from pathlib import Path

from src import config
from src.data.load_data import using_synthetic_data
from src.utils import get_logger

LOGGER = get_logger("data.pipeline")


def ensure_processed_datasets(*, force_recompute: bool = False) -> tuple[Path, Path]:
    """Ensure merged train/test pickles exist, preprocessing with synthetic data when raw CSVs are absent."""
    if (
        not force_recompute
        and config.MERGED_TRAIN_PATH.exists()
        and config.MERGED_TEST_PATH.exists()
    ):
        return config.MERGED_TRAIN_PATH, config.MERGED_TEST_PATH

    if using_synthetic_data():
        LOGGER.warning(
            "Kaggle raw data not found; preprocessing will use synthetic sample tables "
            "so the pipeline can run offline."
        )

    from scripts.preprocess_data import run_preprocessing

    return run_preprocessing(use_cache=not force_recompute)


def ensure_model_bundle(*, force_retrain: bool = False) -> Path:
    """Ensure the ensemble model bundle exists, training on synthetic data when needed."""
    if not force_retrain and config.MODEL_BUNDLE_PATH.is_file():
        return config.MODEL_BUNDLE_PATH

    if using_synthetic_data():
        LOGGER.warning(
            "Model bundle missing and raw Kaggle data is unavailable; "
            "running a compact demo training pass on synthetic sample data."
        )

    ensure_processed_datasets(force_recompute=force_retrain)
    from src.models.train import run_training

    return run_training(demo_mode=using_synthetic_data())
