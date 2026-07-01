from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.api_runtime import ensure_local_api, ensure_model_artifact, stop_managed_api, wait_for_api
from src.config import load_config
from tests.load.payload_factory import build_payload_pool


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test the inference API.")
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="Do not auto-start a local API process when the target host is down.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    cfg = load_config()
    base = cfg.load_test.host.rstrip("/")

    try:
        ensure_model_artifact()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    api_proc = None
    started_here = False
    try:
        if args.no_start:
            ready = wait_for_api(base, timeout_seconds=30.0, require_inference=True)
            if ready is None:
                print(
                    f"API is not reachable at {base}/health. "
                    "Start it with `make serve`, `make docker-run`, or `make docker-smoke`.",
                    file=sys.stderr,
                )
                return 1
        else:
            try:
                api_proc, started_here = ensure_local_api(
                    PROJECT_ROOT,
                    base,
                    startup_timeout_seconds=max(120.0, float(cfg.api.queue_wait_seconds) * 0.2 + 90.0),
                    require_inference=True,
                )
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 1

        data = build_payload_pool(pool_size=1, seed=42)["minimal"][0]
        timeout_s = max(180.0, float(cfg.api.inference_timeout_seconds) + float(cfg.api.queue_wait_seconds) * 0.25)
        client_timeout = httpx.Timeout(connect=10.0, read=timeout_s, write=30.0, pool=10.0)
        with httpx.Client(timeout=client_timeout) as client:
            health = client.get(f"{base}/health")
            health.raise_for_status()
            health_body = health.json()
            if not health_body.get("inference_ready"):
                print(
                    f"API health OK but inference is not ready: {health_body}",
                    file=sys.stderr,
                )
                return 1
            predict = client.post(f"{base}/predict", json=data)
            predict.raise_for_status()
            body = predict.json()
            if not body.get("predictions"):
                print(f"Predict response missing predictions: {body}", file=sys.stderr)
                return 1
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        print(f"HTTP {exc.response.status_code} from {exc.request.url}: {detail}", file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        print(
            f"Could not reach API at {base}: {exc}. "
            "Start it with `make serve` or `make docker-run`.",
            file=sys.stderr,
        )
        return 1
    finally:
        stop_managed_api(api_proc, started_here=started_here)

    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
