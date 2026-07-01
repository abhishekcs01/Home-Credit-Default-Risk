#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/mnt/c/Python/python.exe}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

export COVERAGE_FILE="${COVERAGE_FILE:-.coverage.run_tests}"
"${PYTHON_BIN}" -m pytest tests -q --cov=src/api --cov=src/config --cov=src/features --cov=src/utils --cov-report=term-missing --cov-report=html
