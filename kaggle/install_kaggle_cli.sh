#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TRM_DIR:-}" ]]; then
  echo "TRM_DIR must be set (try: source kaggle/kaggle_env_exports.sh)" >&2
  exit 1
fi

VENV_DIR="${VENV_DIR:-${TRM_DIR}/.venv}"
PY_BIN="${PY:-python3.10}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[install_kaggle_cli] Creating venv at ${VENV_DIR} (python=${PY_BIN})"
  "${PY_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip
pip install kaggle

echo "[install_kaggle_cli] Kaggle CLI installed in ${VENV_DIR}"
