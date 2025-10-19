#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: package_trm_checkpoint.sh [OPTIONS]

Packages the latest TRM checkpoint for Kaggle publishing. Requires the TRM
training run directory to contain COMMANDS.txt and TRM_COMMIT.txt as generated
by the paper-faithful runbook.

Options:
  --checkpoint PATH   Explicit checkpoint to package.
  --run-dir PATH      Override run directory (defaults to $TRM_DIR/runs/$RUN_NAME).
  --no-upload         Skip Kaggle upload even if PUBLISH_TO_KAGGLE=1.
  -h, --help          Show this help text.
EOF
}

CHECKPOINT_OVERRIDE=""
RUN_DIR_OVERRIDE=""
DO_UPLOAD=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT_OVERRIDE="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR_OVERRIDE="$2"
      shift 2
      ;;
    --no-upload)
      DO_UPLOAD=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

source "${SCRIPT_DIR}/kaggle_env_exports.sh"

VENV_DIR="${VENV_DIR:-${TRM_DIR}/.venv}"
RUN_DIR="${RUN_DIR_OVERRIDE:-${TRM_DIR}/runs/${RUN_NAME}}"
CHECKPOINT=""

if [[ -n "${CHECKPOINT_OVERRIDE}" ]]; then
  CHECKPOINT="${CHECKPOINT_OVERRIDE}"
else
  if [[ -d "${RUN_DIR}/checkpoints" ]]; then
    CHECKPOINT="$(ls -t "${RUN_DIR}/checkpoints" 2>/dev/null | head -n1)"
    if [[ -n "${CHECKPOINT}" ]]; then
      CHECKPOINT="${RUN_DIR}/checkpoints/${CHECKPOINT}"
    fi
  fi
  if [[ -z "${CHECKPOINT}" ]]; then
    CHECKPOINT="$(ls -t "${RUN_DIR}"/*.ckpt 2>/dev/null | head -n1 || true)"
  fi
fi

if [[ -z "${CHECKPOINT}" ]]; then
  echo "[package_trm_checkpoint] No checkpoint found. Use --checkpoint to specify one." >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[package_trm_checkpoint] Checkpoint path not found: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${KAGGLE_DS_DIR}"

cp "${SCRIPT_DIR}/dataset_template/README.md" "${KAGGLE_DS_DIR}/README.md"
sed \
  -e "s/__RUN_NAME__/${RUN_NAME}/g" \
  -e "s/__KAGGLE_USERNAME__/${KAGGLE_USERNAME}/g" \
  -e "s/__KAGGLE_DATASET_SLUG__/${KAGGLE_DATASET_SLUG}/g" \
  "${SCRIPT_DIR}/dataset_template/dataset-metadata.json.tmpl" \
  > "${KAGGLE_DS_DIR}/dataset-metadata.json"

cp "${CHECKPOINT}" "${KAGGLE_DS_DIR}/model.ckpt"

[[ -f "${RUN_DIR}/TRM_COMMIT.txt" ]] && cp "${RUN_DIR}/TRM_COMMIT.txt" "${KAGGLE_DS_DIR}/"
[[ -f "${RUN_DIR}/COMMANDS.txt" ]] && cp "${RUN_DIR}/COMMANDS.txt" "${KAGGLE_DS_DIR}/"

ENV_LOG_PATH="${TRM_DIR}/.repro/env_${TARGET_DATASET}_${RUN_NAME}.txt"
SUMMARY_PATH="${TRM_DIR}/.repro/summary_${RUN_NAME}.txt"

[[ -f "${ENV_LOG_PATH}" ]] && cp "${ENV_LOG_PATH}" "${KAGGLE_DS_DIR}/ENVIRONMENT.txt"
[[ -f "${SUMMARY_PATH}" ]] && cp "${SUMMARY_PATH}" "${KAGGLE_DS_DIR}/RUN_SUMMARY.txt"

cat <<EOF > "${KAGGLE_DS_DIR}/MANIFEST.txt"
CHECKPOINT=$(basename "${CHECKPOINT}")
RUN_DIR=${RUN_DIR}
TRM_COMMIT=$(cat "${RUN_DIR}/TRM_COMMIT.txt" 2>/dev/null || echo "unknown")
PACKAGED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

echo "[package_trm_checkpoint] Packaged dataset at ${KAGGLE_DS_DIR}"

if [[ "${PUBLISH_TO_KAGGLE}" != "1" ]]; then
  echo "[package_trm_checkpoint] PUBLISH_TO_KAGGLE!=1; skipping upload."
  exit 0
fi

if [[ "${DO_UPLOAD}" -ne 1 ]]; then
  echo "[package_trm_checkpoint] Upload disabled by --no-upload."
  exit 0
fi

if ! command -v kaggle >/dev/null 2>&1; then
  if [[ -d "${VENV_DIR}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"
  fi
fi

if ! command -v kaggle >/dev/null 2>&1; then
  echo "[package_trm_checkpoint] Kaggle CLI not found in PATH. Install it before uploading." >&2
  exit 1
fi

if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
  echo "[package_trm_checkpoint] Missing ${HOME}/.kaggle/kaggle.json; cannot authenticate." >&2
  exit 1
fi

chmod 600 "${HOME}/.kaggle/kaggle.json"

if kaggle datasets create -p "${KAGGLE_DS_DIR}" >/tmp/kaggle_create.log 2>&1; then
  echo "[package_trm_checkpoint] Kaggle dataset created: ${KAGGLE_USERNAME}/${KAGGLE_DATASET_SLUG}"
else
  echo "[package_trm_checkpoint] create failed (likely already exists); attempting version push."
  kaggle datasets version -p "${KAGGLE_DS_DIR}" -m "update ${RUN_NAME}"
  echo "[package_trm_checkpoint] Kaggle dataset versioned: ${KAGGLE_USERNAME}/${KAGGLE_DATASET_SLUG}"
fi
