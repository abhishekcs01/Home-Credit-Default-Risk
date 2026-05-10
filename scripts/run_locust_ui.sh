#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python scripts/run_locust_ui.py "$@"
