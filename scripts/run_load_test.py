from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.api_runtime import ensure_local_api, stop_managed_api, wait_for_api
from src.config import load_config


def _api_kind(base_url: str) -> str | None:
    ready = wait_for_api(base_url, timeout_seconds=3.0, require_inference=True)
    if ready is None:
        return None
    model_path = str(ready.get("model_path", ""))
    if model_path.startswith("/app/artifacts"):
        return "docker"
    return "local"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Locust load test (UI mode by default).")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode instead of Locust Web UI.",
    )
    parser.add_argument(
        "--web-host",
        default="127.0.0.1",
        help="Locust Web UI host (UI mode only). Default: 127.0.0.1",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=8089,
        help="Locust Web UI port (UI mode only). Default: 8089",
    )
    parser.add_argument(
        "--use-docker",
        action="store_true",
        help="Start the API in Docker before the load test (frees port 8000 first).",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    cfg = load_config()
    locustfile = PROJECT_ROOT / "tests" / "load" / "simple_locustfile.py"
    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(locustfile),
        "--host",
        cfg.load_test.host,
    ]
    if args.headless:
        cmd.extend(
            [
                "--headless",
                "--users",
                str(cfg.load_test.users),
                "--spawn-rate",
                str(cfg.load_test.spawn_rate),
                "--run-time",
                cfg.load_test.run_time,
                "--only-summary",
            ]
        )
    else:
        cmd.extend(["--web-host", args.web_host, "--web-port", str(args.web_port)])

    api_proc: subprocess.Popen[str] | None = None
    started_here = False
    kind = _api_kind(cfg.load_test.host)
    if args.use_docker:
        from scripts.docker_smoke import docker_build, docker_start, free_port, wait_for_docker_api

        print("Preparing Docker API for load test...")
        free_port(8000)
        docker_build()
        docker_start(8000)
        if wait_for_docker_api(cfg.load_test.host, timeout_seconds=240.0) is None:
            print("Docker API did not become ready.", file=sys.stderr)
            return 1
        kind = "docker"
    elif kind is None:
        try:
            api_proc, started_here = ensure_local_api(
                PROJECT_ROOT,
                cfg.load_test.host,
                startup_timeout_seconds=120.0,
                require_inference=True,
            )
            kind = "local"
        except RuntimeError as exc:
            print(str(exc))
            return 1
    else:
        print(f"[load-test] Using existing {kind} API at {cfg.load_test.host}")

    try:
        if not args.headless:
            print(f"Locust UI: http://{args.web_host}:{args.web_port}")
            print("Set users/spawn rate from the Locust Web UI and click Start swarming.")
            if kind == "local":
                print("[load-test] Tip: use `make load-test USE_DOCKER=1` to test the Docker image.")
        return subprocess.run(cmd, cwd=PROJECT_ROOT, check=False).returncode
    finally:
        stop_managed_api(api_proc, started_here=started_here)


if __name__ == "__main__":
    raise SystemExit(main())
