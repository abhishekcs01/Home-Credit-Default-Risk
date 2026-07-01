from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


def main() -> None:
    cfg = load_config()
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--host",
        cfg.api.host,
        "--port",
        str(cfg.api.port),
        "--workers",
        str(cfg.api.workers),
        "--limit-concurrency",
        str(cfg.api.limit_concurrency),
        "--backlog",
        str(cfg.api.backlog),
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
