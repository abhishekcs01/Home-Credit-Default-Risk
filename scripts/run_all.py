from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.api_runtime import ensure_local_api, stop_managed_api
from src.config import load_config


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"> {' '.join(cmd)}")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=merged_env)


def main() -> int:
    cfg = load_config()
    python_bin = sys.executable

    # Offline ML pipeline
    _run([python_bin, "scripts/preprocess_data.py"])
    _run([python_bin, "scripts/train_model.py"])
    _run([python_bin, "scripts/generate_submission.py"])

    # Full tests with terminal coverage percentage
    _run(
        [
            python_bin,
            "-m",
            "pytest",
            "tests",
            "-q",
            "--cov=src/api",
            "--cov=src/config",
            "--cov=src/features",
            "--cov=src/utils",
            "--cov-report=term-missing",
            "--cov-report=html",
        ],
        env={"COVERAGE_FILE": ".coverage.all"},
    )

    # API startup + smoke test
    api_proc, started_here = ensure_local_api(
        PROJECT_ROOT,
        cfg.load_test.host,
        startup_timeout_seconds=120.0,
        require_inference=True,
    )
    try:
        _run([python_bin, "scripts/smoke_test_api.py", "--no-start"])
    finally:
        stop_managed_api(api_proc, started_here=started_here)

    print("All core flows completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
