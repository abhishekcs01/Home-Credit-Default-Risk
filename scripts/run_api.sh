#!/usr/bin/env bash
set -euo pipefail

HOST="${HOME_CREDIT_API_HOST:-}"
PORT="${HOME_CREDIT_API_PORT:-}"
WORKERS="${HOME_CREDIT_API_WORKERS:-}"

if [[ -z "${HOST}" || -z "${PORT}" || -z "${WORKERS}" ]]; then
  eval "$(python - <<'PY'
from src.runtime_config import load_runtime_settings
s = load_runtime_settings()
print(f'DEFAULT_HOST="{s.api.host}"')
print(f'DEFAULT_PORT="{s.api.port}"')
print(f'DEFAULT_WORKERS="{s.api.workers}"')
PY
)"
fi

HOST="${HOST:-$DEFAULT_HOST}"
PORT="${PORT:-$DEFAULT_PORT}"
WORKERS="${WORKERS:-$DEFAULT_WORKERS}"

exec uvicorn app.main:app --host "${HOST}" --port "${PORT}" --workers "${WORKERS}"
