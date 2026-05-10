"""Start Locust with the web UI (not headless). Target API must already be running."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime_config import load_runtime_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open Locust web UI to drive load tests against the inference API.",
    )
    parser.add_argument(
        "--target-host",
        type=str,
        default=None,
        help="API base URL for Locust (default: load_test.host from configs/config.yaml or HOME_CREDIT_LOAD_HOST).",
    )
    parser.add_argument(
        "--web-host",
        type=str,
        default=None,
        help="Bind address for Locust UI (default: config load_test.locust_web_host).",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=None,
        help="Port for Locust UI (default: config load_test.locust_web_port, usually 8089).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser tab automatically.",
    )
    args = parser.parse_args()

    settings = load_runtime_settings()
    lt = settings.load_test
    target = args.target_host or os.getenv("HOME_CREDIT_LOAD_HOST", lt.host)
    web_host = args.web_host or lt.locust_web_host
    web_port = args.web_port if args.web_port is not None else lt.locust_web_port

    locustfile = PROJECT_ROOT / "tests" / "load" / "locustfile.py"
    if not locustfile.is_file():
        raise FileNotFoundError(f"Missing {locustfile}")

    ui_url = f"http://{web_host if web_host != '0.0.0.0' else '127.0.0.1'}:{web_port}"
    print(f"Locust UI: {ui_url}")
    print(f"Target API (--host): {target}")
    print("Ensure the API is running (e.g. python -m uvicorn app.main:app --host 127.0.0.1 --port 8000).")
    print()

    if not args.no_browser:
        try:
            webbrowser.open(ui_url)
        except Exception:
            pass

    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(locustfile),
        "--host",
        target,
        "--web-host",
        web_host,
        "--web-port",
        str(web_port),
    ]
    os.chdir(PROJECT_ROOT)
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
