#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/mnt/c/Python/python.exe}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" scripts/preprocess_data.py
"${PYTHON_BIN}" scripts/train_model.py
