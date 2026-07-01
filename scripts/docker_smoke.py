from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config

DOCKER_IMAGE = "abhishek-ml-project"
DOCKER_CONTAINER = "abhishek-ml-project"


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=check, text=True, capture_output=True)


def _pids_on_port(port: int) -> list[int]:
    try:
        proc = _run(["ss", "-tlnp"], check=False)
    except FileNotFoundError:
        return []
    pids: list[int] = []
    needle = f":{port}"
    for line in proc.stdout.splitlines():
        if needle not in line or "users:" not in line:
            continue
        chunk = line.split("users:((", 1)[-1]
        for token in chunk.replace(")", "").split(","):
            token = token.strip().strip('"')
            if token.startswith("pid="):
                try:
                    pids.append(int(token.split("=", 1)[1]))
                except ValueError:
                    pass
    return sorted(set(pids))


def free_port(port: int) -> None:
    _run(["docker", "rm", "-f", DOCKER_CONTAINER], check=False)
    for pid in _pids_on_port(port):
        subprocess.run(["kill", str(pid)], check=False)
        time.sleep(0.2)
    deadline = time.time() + 5.0
    while time.time() < deadline and _pids_on_port(port):
        time.sleep(0.2)


def docker_build() -> None:
    _run(["docker", "build", "-t", DOCKER_IMAGE, "."])


def docker_start(port: int = 8000) -> None:
    artifacts = PROJECT_ROOT / "artifacts"
    _run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "-p",
            f"{port}:8000",
            "--name",
            DOCKER_CONTAINER,
            "-v",
            f"{artifacts}:/app/artifacts:ro",
            DOCKER_IMAGE,
        ]
    )


def docker_logs(tail: int = 40) -> str:
    proc = _run(["docker", "logs", DOCKER_CONTAINER], check=False)
    lines = (proc.stdout + proc.stderr).splitlines()
    return "\n".join(lines[-tail:])


def wait_for_docker_api(
    base_url: str,
    *,
    timeout_seconds: float = 240.0,
) -> dict[str, Any] | None:
    deadline = time.time() + timeout_seconds
    url = f"{base_url.rstrip('/')}/health"
    timeout = httpx.Timeout(connect=5.0, read=15.0, write=15.0, pool=5.0)
    with httpx.Client(timeout=timeout) as client:
        while time.time() < deadline:
            try:
                resp = client.get(url)
                if resp.status_code != 200:
                    time.sleep(1.0)
                    continue
                body = resp.json()
                model_path = str(body.get("model_path", ""))
                if not body.get("inference_ready"):
                    time.sleep(1.0)
                    continue
                if not model_path.startswith("/app/artifacts"):
                    time.sleep(1.0)
                    continue
                return body
            except (httpx.HTTPError, ValueError):
                time.sleep(1.0)
    return None


def run_smoke_predict(base_url: str, *, timeout_seconds: float = 180.0) -> None:
    from tests.load.payload_factory import build_payload_pool

    payload = build_payload_pool(pool_size=1, seed=42)["minimal"][0]
    timeout = httpx.Timeout(connect=10.0, read=timeout_seconds, write=30.0, pool=10.0)
    with httpx.Client(timeout=timeout) as client:
        health = client.get(f"{base_url.rstrip('/')}/health")
        health.raise_for_status()
        body = health.json()
        if not body.get("inference_ready"):
            raise RuntimeError(f"Inference not ready: {body}")
        if not str(body.get("model_path", "")).startswith("/app/artifacts"):
            raise RuntimeError(f"Expected Docker API, got model_path={body.get('model_path')}")
        predict = client.post(f"{base_url.rstrip('/')}/predict", json=payload)
        predict.raise_for_status()
        result = predict.json()
        if not result.get("predictions"):
            raise RuntimeError(f"Predict response missing predictions: {result}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build, start Docker API, and smoke test it.")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not config.MODEL_BUNDLE_PATH.is_file():
        print(f"Missing model bundle at {config.MODEL_BUNDLE_PATH}. Run `make train` first.", file=sys.stderr)
        return 1

    base_url = f"http://127.0.0.1:{args.port}"
    try:
        print(f"Freeing port {args.port}...")
        free_port(args.port)
        if not args.skip_build:
            print(f"Building Docker image {DOCKER_IMAGE}...")
            docker_build()
        print(f"Starting container {DOCKER_CONTAINER}...")
        docker_start(args.port)
        print("Waiting for Docker API to become inference-ready...")
        ready = wait_for_docker_api(base_url, timeout_seconds=240.0)
        if ready is None:
            print("Docker API did not become ready in time.", file=sys.stderr)
            print(docker_logs(), file=sys.stderr)
            return 1
        print(f"API ready: {ready.get('model_path')}")
        run_smoke_predict(base_url)
    except (httpx.HTTPError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Docker smoke test failed: {exc}", file=sys.stderr)
        print(docker_logs(), file=sys.stderr)
        return 1

    print("Docker smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
