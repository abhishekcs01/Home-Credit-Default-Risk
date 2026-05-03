from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.inference import predict_with_ensemble
from src.utils import ensure_dir, get_logger, init_logging, load_pickle, timer


def run_submission(*, output_path: Path | None = None, write_timestamped_copy: bool = True) -> tuple[Path, Path | None]:
    logger = get_logger("generate_submission")
    if not config.MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Missing model artifact at {config.MODEL_BUNDLE_PATH}. Run scripts/train_model.py first."
        )
    if not config.MERGED_TEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing processed test dataset at {config.MERGED_TEST_PATH}. Run scripts/preprocess_data.py first."
        )

    ensure_dir(config.SUBMISSIONS_DIR)

    with timer("Load model + test data"):
        logger.info("Loading model bundle and processed test data")
        model_bundle = load_pickle(config.MODEL_BUNDLE_PATH)
        merged_test = pd.read_pickle(config.MERGED_TEST_PATH)

    with timer("Ensemble inference"):
        submission = predict_with_ensemble(model_bundle, merged_test)

    primary = output_path or config.SUBMISSION_PATH
    with timer("Write submission files"):
        submission.to_csv(primary, index=False)
        logger.info("Saved submission: %s (rows=%d)", primary, len(submission))

    stamped: Path | None = None
    if write_timestamped_copy:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stamped = config.SUBMISSIONS_DIR / f"submission_{stamp}.csv"
        submission.to_csv(stamped, index=False)
        logger.info("Saved timestamped copy: %s", stamped)

    return primary, stamped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission.csv from trained model and processed test data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output path for submission CSV (default: {config.SUBMISSION_PATH}).",
    )
    parser.add_argument(
        "--no-timestamp-copy",
        action="store_true",
        help="Do not write an additional submission_YYYYMMDD_HHMMSS.csv file.",
    )
    return parser.parse_args()


def main() -> None:
    init_logging()
    args = parse_args()
    run_submission(output_path=args.output, write_timestamped_copy=not args.no_timestamp_copy)


if __name__ == "__main__":
    main()
