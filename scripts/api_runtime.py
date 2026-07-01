from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from src import config


def wait_for_api(
    base_url: str,
    *,
    timeout_seconds: float = 90.0,
    require_inference: bool = False,
) -> dict[str, Any] | None:
    deadline = time.time() + timeout_seconds
    url = f"{base_url.rstrip('/')}/health"
    with httpx.Client(timeout=5.0) as client:
        while time.time() < deadline:
            try:
                resp = client.get(url)
                if resp.status_code != 200:
                    time.sleep(0.5)
                    continue
                body = resp.json()
                if require_inference and not body.get("inference_ready"):
                    time.sleep(1.0)
                    continue
                return body
            except (httpx.HTTPError, ValueError):
                time.sleep(0.5)
    return None


def ensure_local_api(
    project_root: Path,
    base_url: str,
    *,
    startup_timeout_seconds: float = 120.0,
    require_inference: bool = True,
) -> tuple[subprocess.Popen[str] | None, bool]:
    existing = wait_for_api(
        base_url,
        timeout_seconds=3.0,
        require_inference=require_inference,
    )
    if existing is not None:
        return None, False

    proc = subprocess.Popen(
        [sys.executable, "scripts/run_api.py"],
        cwd=project_root,
    )
    ready = wait_for_api(
        base_url,
        timeout_seconds=startup_timeout_seconds,
        require_inference=require_inference,
    )
    if ready is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise RuntimeError(
            f"API did not become ready at {base_url.rstrip('/')}/health "
            f"within {startup_timeout_seconds:.0f}s"
        )
    return proc, True


def stop_managed_api(proc: subprocess.Popen[str] | None, *, started_here: bool) -> None:
    if not started_here or proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def ensure_model_artifact() -> None:
    if config.MODEL_BUNDLE_PATH.is_file():
        return
    from src.data.pipeline import ensure_model_bundle

    ensure_model_bundle()
