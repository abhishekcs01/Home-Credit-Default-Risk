"""Single entry point: ML pipeline → temporary API → smoke test → optional Locust load test."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config  # noqa: E402
from src.runtime_config import load_runtime_settings  # noqa: E402
from src.utils import get_logger, init_logging  # noqa: E402


def _load_run_pipeline():
    path = PROJECT_ROOT / "scripts" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("pipeline_run_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_full_pipeline


def _wait_for_health(base_url: str, *, timeout_sec: float, log) -> None:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("Install httpx (pip install httpx) for API smoke checks.") from exc

    deadline = time.monotonic() + timeout_sec
    last_err: str | None = None
    url = f"{base_url.rstrip('/')}/health"
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=5.0)
            if resp.status_code == 200:
                log.info("API healthy: %s", url)
                return
            last_err = f"HTTP {resp.status_code}"
        except Exception as exc:
            last_err = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"API did not become healthy within {timeout_sec}s (last error: {last_err})")


def _smoke_predict(base_url: str, payload_path: Path, log) -> None:
    import httpx

    body = json.loads(payload_path.read_text(encoding="utf-8"))
    url = f"{base_url.rstrip('/')}/predict"
    resp = httpx.post(url, json=body, timeout=120.0)
    resp.raise_for_status()
    data = resp.json()
    if "predictions" not in data:
        raise RuntimeError(f"Unexpected predict response: {data!r}")
    log.info("Smoke POST /predict OK (%d prediction row(s))", len(data.get("predictions", [])))


def _run_locust(base_url: str, *, users: int, spawn_rate: int, run_time: str, log) -> None:
    locustfile = PROJECT_ROOT / "tests" / "load" / "locustfile.py"
    if not locustfile.is_file():
        raise FileNotFoundError(f"Missing Locust file: {locustfile}")
    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(locustfile),
        "--host",
        base_url.rstrip("/"),
        "--headless",
        "--users",
        str(users),
        "--spawn-rate",
        str(spawn_rate),
        "--run-time",
        run_time,
    ]
    log.info("Running Locust: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def run_full_stack(
    *,
    optimize_memory: bool,
    export_csv: bool,
    force_recompute: bool,
    submission_output: Path | None,
    timestamped_submission_copy: bool,
    log_file: Path | None,
    pipeline_only: bool,
    skip_load_test: bool,
    api_startup_timeout_sec: float,
    load_users: int,
    load_spawn_rate: int,
    load_run_time: str,
) -> None:
    init_logging(log_file)
    log = get_logger("run_all")

    run_full_pipeline = _load_run_pipeline()
    run_full_pipeline(
        optimize_memory=optimize_memory,
        export_csv=export_csv,
        force_recompute=force_recompute,
        submission_output=submission_output,
        timestamped_submission_copy=timestamped_submission_copy,
        log_file=log_file,
    )

    if pipeline_only:
        log.info("Pipeline-only mode: skipping API, smoke, and load test.")
        return

    settings = load_runtime_settings()
    port = settings.api.port
    base_url = f"http://127.0.0.1:{port}"
    payload_path = PROJECT_ROOT / "examples" / "payloads" / "minimal_request.json"
    if not payload_path.is_file():
        raise FileNotFoundError(f"Missing smoke payload: {payload_path}")

    env = os.environ.copy()
    env.setdefault("HOME_CREDIT_API_HOST", "127.0.0.1")
    env.setdefault("HOME_CREDIT_API_PORT", str(port))
    env.setdefault("HOME_CREDIT_API_STARTUP_LOAD_MODEL", "true")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_health(base_url, timeout_sec=api_startup_timeout_sec, log=log)
        _smoke_predict(base_url, payload_path, log=log)
        if not skip_load_test:
            _run_locust(
                base_url,
                users=load_users,
                spawn_rate=load_spawn_rate,
                run_time=load_run_time,
                log=log,
            )
        log.info("Full stack finished successfully.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run preprocess → train → submission, then API smoke (+ optional Locust) in one command.",
    )
    p.add_argument("--no-memory-opt", action="store_true", help="Disable memory optimization when loading raw CSVs.")
    p.add_argument("--export-csv", action="store_true", help="Also export merged_train.csv / merged_test.csv.")
    p.add_argument("--force-recompute", action="store_true", help="Force preprocessing from raw data (ignore merged pickles).")
    p.add_argument("--submission-output", type=Path, default=None, help="Override submission CSV path.")
    p.add_argument("--no-timestamp-copy", action="store_true", help="Skip timestamped submission copy.")
    p.add_argument("--log-file", type=Path, default=None, help="Pipeline log file path.")
    p.add_argument(
        "--pipeline-only",
        action="store_true",
        help="Only run ML pipeline (preprocess → train → submission); skip API and tests.",
    )
    p.add_argument("--skip-load-test", action="store_true", help="After smoke test, skip Locust.")
    p.add_argument(
        "--api-startup-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for /health after starting uvicorn (model load can be slow).",
    )
    p.add_argument("--load-users", type=int, default=5, help="Locust users (default: small quick run).")
    p.add_argument("--load-spawn-rate", type=int, default=2, help="Locust spawn rate.")
    p.add_argument("--load-run-time", type=str, default="30s", help="Locust headless run time (e.g. 30s, 2m).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    default_log = project_config.LOGS_DIR / f"run_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = args.log_file or default_log
    run_full_stack(
        optimize_memory=not args.no_memory_opt,
        export_csv=args.export_csv,
        force_recompute=args.force_recompute,
        submission_output=args.submission_output,
        timestamped_submission_copy=not args.no_timestamp_copy,
        log_file=log_path,
        pipeline_only=args.pipeline_only,
        skip_load_test=args.skip_load_test,
        api_startup_timeout_sec=args.api_startup_timeout,
        load_users=args.load_users,
        load_spawn_rate=args.load_spawn_rate,
        load_run_time=args.load_run_time,
    )


if __name__ == "__main__":
    main()
