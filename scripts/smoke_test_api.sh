#!/usr/bin/env bash
set -euo pipefail

API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8000}"
PAYLOAD_PATH="${PAYLOAD_PATH:-examples/payloads/minimal_request.json}"

echo "Health check: ${API_BASE_URL}/health"
curl --fail --silent --show-error "${API_BASE_URL}/health" | python -m json.tool

echo "Prediction smoke test: ${API_BASE_URL}/predict"
curl --fail --silent --show-error \
  -X POST "${API_BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  --data "@${PAYLOAD_PATH}" | python -m json.tool

echo "Smoke test passed."
