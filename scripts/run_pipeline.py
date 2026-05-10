"""Run preprocessing → training → submission generation in one process."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config  # noqa: E402
from src.utils import get_logger, init_logging  # noqa: E402


def _import_script(module_label: str, relative_script: Path):
    spec = importlib.util.spec_from_file_location(module_label, relative_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load script at {relative_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_full_pipeline(
    *,
    optimize_memory: bool = True,
    export_csv: bool = False,
    force_recompute: bool = False,
    submission_output: Path | None = None,
    timestamped_submission_copy: bool = True,
    log_file: Path | None = None,
) -> dict[str, object]:
    """Execute all stages sequentially; returns simple result metadata."""
    init_logging(log_file)
    log = get_logger("run_pipeline")
    log.info("Project root: %s", project_config.PROJECT_ROOT)
    log.info("Starting full ML pipeline (preprocess → train → submit)")

    t0 = time.perf_counter()
    stages: dict[str, float] = {}

    script_dir = project_config.SCRIPTS_DIR
    preprocess_mod = _import_script("pipeline_preprocess", script_dir / "preprocess_data.py")
    train_mod = _import_script("pipeline_train", script_dir / "train_model.py")
    submit_mod = _import_script("pipeline_submit", script_dir / "generate_submission.py")

    s = time.perf_counter()
    log.info("=== Stage 1/3: Preprocessing ===")
    preprocess_mod.run_preprocessing(
        optimize_memory=optimize_memory,
        export_csv=export_csv,
        use_cache=not force_recompute,
    )
    stages["preprocess_sec"] = time.perf_counter() - s

    s = time.perf_counter()
    log.info("=== Stage 2/3: Training ===")
    train_mod.run_training()
    stages["train_sec"] = time.perf_counter() - s

    s = time.perf_counter()
    log.info("=== Stage 3/3: Submission ===")
    primary, stamped = submit_mod.run_submission(
        output_path=submission_output,
        write_timestamped_copy=timestamped_submission_copy,
    )
    stages["submission_sec"] = time.perf_counter() - s

    total = time.perf_counter() - t0
    stages["total_sec"] = total

    log.info(
        "Pipeline complete in %.1fs (preprocess %.1fs, train %.1fs, submission %.1fs)",
        total,
        stages["preprocess_sec"],
        stages["train_sec"],
        stages["submission_sec"],
    )

    return {
        "submission_path": primary,
        "submission_timestamped_path": stamped,
        "timing": stages,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run end-to-end Home Credit ML pipeline.")
    p.add_argument("--no-memory-opt", action="store_true", help="Disable memory optimization when loading raw CSVs.")
    p.add_argument(
        "--export-csv",
        action="store_true",
        help="During preprocessing, also write large merged_train.csv / merged_test.csv exports.",
    )
    p.add_argument(
        "--submission-output",
        type=Path,
        default=None,
        help="Override path for final submission.csv (default: outputs/submissions/submission.csv).",
    )
    p.add_argument(
        "--no-timestamp-copy",
        action="store_true",
        help="Skip writing submission_YYYYMMDD_HHMMSS.csv alongside the primary file.",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Append logs to this file (default: outputs/logs/pipeline_YYYYMMDD_HHMMSS.log).",
    )
    p.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached merged_train.pkl / merged_test.pkl and recompute preprocessing from raw CSVs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    default_log = project_config.LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = args.log_file or default_log
    run_full_pipeline(
        optimize_memory=not args.no_memory_opt,
        export_csv=args.export_csv,
        force_recompute=args.force_recompute,
        submission_output=args.submission_output,
        timestamped_submission_copy=not args.no_timestamp_copy,
        log_file=log_path,
    )


if __name__ == "__main__":
    main()
