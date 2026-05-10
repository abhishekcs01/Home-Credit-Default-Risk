#!/usr/bin/env bash
set -euo pipefail

HOST="${HOME_CREDIT_LOAD_HOST:-}"
USERS="${HOME_CREDIT_LOAD_USERS:-}"
SPAWN_RATE="${HOME_CREDIT_LOAD_SPAWN_RATE:-}"
RUN_TIME="${HOME_CREDIT_LOAD_RUN_TIME:-}"

if [[ -z "${HOST}" || -z "${USERS}" || -z "${SPAWN_RATE}" || -z "${RUN_TIME}" ]]; then
  eval "$(python - <<'PY'
from src.runtime_config import load_runtime_settings
s = load_runtime_settings()
print(f'DEFAULT_HOST="{s.load_test.host}"')
print(f'DEFAULT_USERS="{s.load_test.users}"')
print(f'DEFAULT_SPAWN_RATE="{s.load_test.spawn_rate}"')
print(f'DEFAULT_RUN_TIME="{s.load_test.run_time}"')
PY
)"
fi

HOST="${HOST:-$DEFAULT_HOST}"
USERS="${USERS:-$DEFAULT_USERS}"
SPAWN_RATE="${SPAWN_RATE:-$DEFAULT_SPAWN_RATE}"
RUN_TIME="${RUN_TIME:-$DEFAULT_RUN_TIME}"

exec locust -f tests/load/locustfile.py --host "${HOST}" --users "${USERS}" --spawn-rate "${SPAWN_RATE}" --run-time "${RUN_TIME}" --headless
