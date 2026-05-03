from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.aggregation import merge_stage
from src.data_loading import load_application_test, load_application_train, load_auxiliary_tables
from src.feature_engineering import engineer_application_features
from src.preprocessing import replace_day_sentinel
from src.utils import ensure_dir, get_logger, init_logging, timer


def run_preprocessing(*, optimize_memory: bool = True, export_csv: bool = False) -> tuple[Path, Path]:
    logger = get_logger("preprocess")
    with timer("Load raw tables"):
        logger.info("Loading raw training and test data")
        train_df = load_application_train(optimize_memory=optimize_memory)
        test_df = load_application_test(optimize_memory=optimize_memory)
        aux_tables = load_auxiliary_tables(optimize_memory=optimize_memory)

    train_df = replace_day_sentinel(train_df, [c for c in train_df.columns if "DAYS" in c.upper()])
    test_df = replace_day_sentinel(test_df, [c for c in test_df.columns if "DAYS" in c.upper()])

    with timer("Aggregate tables + feature engineering"):
        logger.info("Running full-stage table aggregation and feature engineering")
        merged_train = engineer_application_features(merge_stage(train_df.copy(), "full", **aux_tables))
        merged_test = engineer_application_features(merge_stage(test_df.copy(), "full", **aux_tables))

    ensure_dir(config.PROCESSED_DATA_DIR)
    with timer("Save processed pickles"):
        merged_train.to_pickle(config.MERGED_TRAIN_PATH)
        merged_test.to_pickle(config.MERGED_TEST_PATH)
        logger.info("Saved processed train: %s (shape=%s)", config.MERGED_TRAIN_PATH, merged_train.shape)
        logger.info("Saved processed test: %s (shape=%s)", config.MERGED_TEST_PATH, merged_test.shape)

    if export_csv:
        with timer("Export merged CSV (optional)"):
            merged_train.to_csv(config.MERGED_TRAIN_CSV_PATH, index=False)
            merged_test.to_csv(config.MERGED_TEST_CSV_PATH, index=False)
            logger.info("Wrote CSV exports: %s, %s", config.MERGED_TRAIN_CSV_PATH, config.MERGED_TEST_CSV_PATH)

    return config.MERGED_TRAIN_PATH, config.MERGED_TEST_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Home Credit raw data into merged train/test artifacts.")
    parser.add_argument(
        "--no-memory-opt",
        action="store_true",
        help="Disable memory optimization while loading raw tables.",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Also write merged_train.csv and merged_test.csv under data/processed/ (large files).",
    )
    return parser.parse_args()


def main() -> None:
    init_logging()
    args = parse_args()
    run_preprocessing(optimize_memory=not args.no_memory_opt, export_csv=args.export_csv)


if __name__ == "__main__":
    main()
