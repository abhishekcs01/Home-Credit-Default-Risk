#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${ROOT_DIR}"

echo "Installing dependencies from requirements.txt..."
"${PYTHON_BIN}" -m pip install -r requirements.txt

if "${PYTHON_BIN}" -c "import lightgbm" >/dev/null 2>&1; then
  if [[ "${INSTALL_LIGHTGBM:-}" == "cuda" || "${FORCE_REINSTALL_LIGHTGBM:-}" == "cuda" ]]; then
    echo "Installing CUDA-enabled LightGBM from source..."
    "${PYTHON_BIN}" -m pip uninstall -y lightgbm
    "${PYTHON_BIN}" -m pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_CUDA=ON
  else
    echo "LightGBM already installed — not modified (CUDA builds are preserved)."
    echo "  Rebuild CUDA LightGBM: INSTALL_LIGHTGBM=cuda make install"
  fi
elif [[ "${INSTALL_LIGHTGBM:-}" == "cpu" ]]; then
  echo "Installing CPU LightGBM wheel..."
  "${PYTHON_BIN}" -m pip install "lightgbm>=4.5.0"
elif [[ "${INSTALL_LIGHTGBM:-}" == "cuda" ]]; then
  echo "Installing CUDA-enabled LightGBM from source..."
  "${PYTHON_BIN}" -m pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_CUDA=ON
else
  echo "LightGBM not installed."
  echo "  GPU: INSTALL_LIGHTGBM=cuda make install"
  echo "  CPU/CI: INSTALL_LIGHTGBM=cpu make install"
fi

echo "Dependency install complete."
