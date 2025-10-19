#!/usr/bin/env bash
# Environment exports for Kaggle publishing workflow.
# Source this file before running dataset packaging scripts so that all
# required variables are guaranteed to exist.

# Core TRM run parameters ---------------------------------------------------
# Leave existing values untouched if they are already set in the environment.
export TARGET_DATASET="${TARGET_DATASET:-arc2}"
export TRM_DIR="${TRM_DIR:-$HOME/TinyRecursiveModels}"
export RUN_NAME="${RUN_NAME:-trm_arc2_paper_repro}"

# Kaggle publishing toggles -------------------------------------------------
# Set PUBLISH_TO_KAGGLE=0 to skip dataset upload entirely.
export PUBLISH_TO_KAGGLE="${PUBLISH_TO_KAGGLE:-1}"

# The Kaggle username must match the account that owns ~/.kaggle/kaggle.json.
export KAGGLE_USERNAME="${KAGGLE_USERNAME:-alexthuth}"

# The dataset slug is stable so downstream notebooks can reference it safely.
export KAGGLE_DATASET_SLUG="${KAGGLE_DATASET_SLUG:-trm-offline-wheels-py311}"

# Directory that will hold the packaged dataset prior to upload.
export KAGGLE_DS_DIR="${KAGGLE_DS_DIR:-${TRM_DIR}/kaggle_dataset_trm_offline_wheels_v2}"

# Torch nightly index URL defaults to CUDA 12.6 wheels; override if needed.
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu126}"

# Convenience helper to display the current configuration.
if [[ "${1:-}" == "print" ]]; then
  cat <<EOF
TARGET_DATASET=${TARGET_DATASET}
TRM_DIR=${TRM_DIR}
RUN_NAME=${RUN_NAME}
PUBLISH_TO_KAGGLE=${PUBLISH_TO_KAGGLE}
KAGGLE_USERNAME=${KAGGLE_USERNAME}
KAGGLE_DATASET_SLUG=${KAGGLE_DATASET_SLUG}
KAGGLE_DS_DIR=${KAGGLE_DS_DIR}
TORCH_INDEX_URL=${TORCH_INDEX_URL}
EOF
fi
