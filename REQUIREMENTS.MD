#!/usr/bin/env bash
# ==============================================================================
# TRM (Tiny Recursive Models) — Paper-Faithful Reproduction & Kaggle Publish
# Single-file runbook for a coding agent to do everything needed to:
#   1) Reproduce training "as written" in the TRM repo (ARC-AGI-1 or ARC-AGI-2)
#   2) (Optional) Publish trained weights as a Kaggle Dataset
#
# Sources:
# - Official TRM repo README (requirements, dataset builders, training commands)
#   https://github.com/SamsungSAILMontreal/TinyRecursiveModels  (used verbatim)
# - TRM paper (for context): https://arxiv.org/abs/2510.04871
#
# This script is:
#   - Idempotent (re-running skips completed steps)
#   - Defensive (checks GPU, Python, pip, Kaggle CLI)
#   - Paper-faithful (no code changes; uses upstream commands as-is)
#
# Notes the README states explicitly:
# - Python 3.10, CUDA 12.6 (or similar); install torch for your CUDA version
# - Install requirements + 'adam-atan2', optional W&B login
# - Dataset builders for ARC-AGI-1 *or* ARC-AGI-2 (do not train on both)
# - ARC training: arch=trm, L_layers=2, H_cycles=3, L_cycles=4, ema=True
# ==============================================================================

set -euo pipefail

# --------------------------- USER CONFIGURATION -------------------------------
# Choose which ARC dataset to reproduce: 'arc1' or 'arc2'
TARGET_DATASET="${TARGET_DATASET:-arc1}"   # allowed: arc1 | arc2

# Where to clone TRM
TRM_DIR="${TRM_DIR:-$HOME/TinyRecursiveModels}"

# Python executable (must be Python 3.10 per README)
PY="${PY:-python3.10}"

# Create & use a virtual environment
USE_VENV="${USE_VENV:-1}"
VENV_DIR="${VENV_DIR:-$TRM_DIR/.venv}"

# PyTorch install index for your CUDA version (README example uses cu126 nightly)
# Adjust if your system uses a different CUDA (see: https://pytorch.org/get-started/)
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu126}"

# W&B login token (optional). If empty, W&B is skipped.
WANDB_API_KEY="${WANDB_API_KEY:-}"

# Run name label used by TRM to name output folder under runs/
RUN_NAME="${RUN_NAME:-trm_${TARGET_DATASET}_paper_repro}"

# GPUs: if >=4 present, we mirror the README and use 4-way torchrun; otherwise single-GPU
MAX_GPUS_TO_USE="${MAX_GPUS_TO_USE:-4}"

# (Optional) Kaggle publishing
PUBLISH_TO_KAGGLE="${PUBLISH_TO_KAGGLE:-0}"       # 1 to enable
KAGGLE_USERNAME="${KAGGLE_USERNAME:-}"            # required if publishing
KAGGLE_DATASET_SLUG="${KAGGLE_DATASET_SLUG:-trm-${TARGET_DATASET}-weights-${RUN_NAME}}"
KAGGLE_DS_DIR="${KAGGLE_DS_DIR:-$TRM_DIR/kaggle_dataset_${TARGET_DATASET}_${RUN_NAME}}"

# ----------------------------- HELPER FUNCTIONS -------------------------------
log()  { echo -e "\n[TRM] $*"; }
die()  { echo -e "\n[TRM][FATAL] $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

gpu_count() { nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' '; }

activate_venv() {
  if [[ "${USE_VENV}" == "1" ]]; then
    if [[ ! -d "${VENV_DIR}" ]]; then
      log "Creating venv at ${VENV_DIR} with ${PY} ..."
      ${PY} -m venv "${VENV_DIR}" || die "Failed to create venv"
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    log "Using venv: $(which python)"
  else
    log "VENV disabled; using system Python: $(which ${PY})"
  fi
}

ensure_prereqs() {
  have git || die "git is required"
  have ${PY} || die "Python 3.10 required (PY=${PY} not found). Install Python 3.10."
  have pip || die "pip is required"
  if ! have nvidia-smi; then
    log "Warning: nvidia-smi not found; continuing (CPU-only or container without NVIDIA tools)"
  fi
}

clone_trm() {
  if [[ ! -d "${TRM_DIR}/.git" ]]; then
    log "Cloning TRM into ${TRM_DIR} ..."
    git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git "${TRM_DIR}" \
      || die "Failed to clone TinyRecursiveModels"
  else
    log "TRM repo already exists at ${TRM_DIR}; pulling latest ..."
    (cd "${TRM_DIR}" && git pull --ff-only) || die "git pull failed"
  fi
  log "TRM commit: $(cd "${TRM_DIR}" && git rev-parse --short HEAD)"
}

install_env() {
  activate_venv
  cd "${TRM_DIR}"

  # Upgrade pip tooling
  pip install --upgrade pip wheel setuptools

  # Install torch/torchvision/torchaudio per README (for your CUDA)
  # README example: cu126 nightly index; change TORCH_INDEX_URL if needed.
  log "Installing PyTorch from ${TORCH_INDEX_URL} ..."
  pip install --pre --upgrade torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}" \
    || die "PyTorch install failed (check CUDA/Index URL)"

  # Install repo requirements
  [[ -f requirements.txt ]] || die "requirements.txt missing in TRM repo"
  pip install -r requirements.txt || die "pip install -r requirements.txt failed"

  # Install adam-atan2 as in README
  pip install --no-cache-dir --no-build-isolation adam-atan2 || die "adam-atan2 install failed"

  # Optional: W&B login
  if [[ -n "${WANDB_API_KEY}" ]]; then
    pip install wandb || true
    WANDB_API_KEY="${WANDB_API_KEY}" wandb login || die "W&B login failed"
  else
    log "WANDB_API_KEY not set; skipping wandb login (logging remains local)"
  fi

  # Record environment for reproducibility
  mkdir -p "${TRM_DIR}/.repro"
  (python -V; pip list; python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())") \
    | tee "${TRM_DIR}/.repro/env_${TARGET_DATASET}_${RUN_NAME}.txt"
}

build_datasets() {
  cd "${TRM_DIR}"
  # Dataset builders from README:
  # ARC-AGI-1:
  #   python -m dataset.build_arc_dataset --input-file-prefix kaggle/combined/arc-agi \
  #     --output-dir data/arc1concept-aug-1000 --subsets training evaluation concept --test-set-name evaluation
  # ARC-AGI-2:
  #   python -m dataset.build_arc_dataset --input-file-prefix kaggle/combined/arc-agi \
  #     --output-dir data/arc2concept-aug-1000 --subsets training2 evaluation2 concept --test-set-name evaluation2
  # NOTE in README: Don't train on both ARC-AGI-1 and ARC-AGI-2 simultaneously
  # because ARC-AGI-2 training contains some ARC-AGI-1 eval data.

  case "${TARGET_DATASET}" in
    arc1)
      OUT="data/arc1concept-aug-1000"
      if [[ ! -d "${OUT}" ]]; then
        log "Building ARC-AGI-1 dataset at ${OUT} ..."
        python -m dataset.build_arc_dataset \
          --input-file-prefix kaggle/combined/arc-agi \
          --output-dir "${OUT}" \
          --subsets training evaluation concept \
          --test-set-name evaluation
      else
        log "ARC-AGI-1 dataset already present at ${OUT} (skipping)"
      fi
      ;;

    arc2)
      OUT="data/arc2concept-aug-1000"
      if [[ ! -d "${OUT}" ]]; then
        log "Building ARC-AGI-2 dataset at ${OUT} ..."
        python -m dataset.build_arc_dataset \
          --input-file-prefix kaggle/combined/arc-agi \
          --output-dir "${OUT}" \
          --subsets training2 evaluation2 concept \
          --test-set-name evaluation2
      else
        log "ARC-AGI-2 dataset already present at ${OUT} (skipping)"
      fi
      ;;

    *)
      die "TARGET_DATASET must be arc1 or arc2 (got: ${TARGET_DATASET})"
      ;;
  esac

  # Sanity checks
  [[ -d "${OUT}" ]] || die "Dataset build failed; directory ${OUT} not found"
  find "${OUT}" -maxdepth 2 -type f | head -n 5 | sed 's/^/[TRM][DATA] /'
}

train_arc() {
  cd "${TRM_DIR}"
  local n_gpus="$(gpu_count || echo 0)"
  local data_path
  if [[ "${TARGET_DATASET}" == "arc1" ]]; then
    data_path="data/arc1concept-aug-1000"
  else
    data_path="data/arc2concept-aug-1000"
  fi
  [[ -d "${data_path}" ]] || die "Data path ${data_path} missing"

  log "GPU(s) detected: ${n_gpus}"
  log "Starting TRM training for ${TARGET_DATASET} (run_name=${RUN_NAME})"

  # Training commands from README (ARC-AGI-1/2), paper-faithful:
  # - arch=trm
  # - arch.L_layers=2
  # - arch.H_cycles=3
  # - arch.L_cycles=4
  # - +run_name=${run_name}
  # - ema=True
  # (Runtime indicated as ~3 days on 4x H100 in README; actual runtime depends on hardware.)

  if [[ "${n_gpus}" -ge 4 ]]; then
    # Limit to MAX_GPUS_TO_USE if needed
    local use_gpus="${MAX_GPUS_TO_USE}"
    if [[ "${n_gpus}" -lt "${MAX_GPUS_TO_USE}" ]]; then use_gpus="${n_gpus}"; fi

    log "Using torchrun with ${use_gpus} GPUs ..."
    torchrun --nproc-per-node "${use_gpus}" --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
      arch=trm \
      data_paths="[${data_path}]" \
      arch.L_layers=2 \
      arch.H_cycles=3 arch.L_cycles=4 \
      +run_name="${RUN_NAME}" ema=True
  else
    log "Using single-GPU (or CPU) python training ..."
    python pretrain.py \
      arch=trm \
      data_paths="[${data_path}]" \
      arch.L_layers=2 \
      arch.H_cycles=3 arch.L_cycles=4 \
      +run_name="${RUN_NAME}" ema=True
  fi

  # Capture outputs
  local run_dir="${TRM_DIR}/runs/${RUN_NAME}"
  [[ -d "${run_dir}" ]] || die "Run output folder ${run_dir} not found"
  (cd "${TRM_DIR}" && git rev-parse HEAD) > "${run_dir}/TRM_COMMIT.txt" || true
  log "Training complete. Outputs in: ${run_dir}"
  find "${run_dir}" -maxdepth 2 -type f | sed 's/^/[TRM][RUN] /' | head -n 20
}

# Helper to locate a likely checkpoint file (last/best)
find_ckpt() {
  local run_dir="${TRM_DIR}/runs/${RUN_NAME}"
  if [[ -d "${run_dir}/checkpoints" ]]; then
    ls -t "${run_dir}/checkpoints"/* 2>/dev/null | head -n 1
  else
    ls -t "${run_dir}"/*.ckpt 2>/dev/null | head -n 1
  fi
}

publish_kaggle_dataset() {
  [[ "${PUBLISH_TO_KAGGLE}" == "1" ]] || { log "Kaggle publish disabled (PUBLISH_TO_KAGGLE=0). Skipping."; return 0; }
  [[ -n "${KAGGLE_USERNAME}" ]] || die "KAGGLE_USERNAME is required to publish to Kaggle"
  have kaggle || { log "Installing Kaggle CLI ..."; pip install kaggle >/dev/null 2>&1 || die "pip install kaggle failed"; }

  # Ensure token exists (~/.kaggle/kaggle.json)
  if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
    die "Kaggle API token not found at ~/.kaggle/kaggle.json (create from Kaggle Account -> API)"
  fi
  chmod 600 "$HOME/.kaggle/kaggle.json" || true

  local ckpt_file
  ckpt_file="$(find_ckpt || true)"
  [[ -n "${ckpt_file}" ]] || die "Could not find a checkpoint in runs/${RUN_NAME}. Did training finish?"

  rm -rf "${KAGGLE_DS_DIR}"
  mkdir -p "${KAGGLE_DS_DIR}"
  cp -v "${ckpt_file}" "${KAGGLE_DS_DIR}/model.ckpt"

  # Minimal metadata
  cat > "${KAGGLE_DS_DIR}/dataset-metadata.json" <<EOF
{
  "title": "TRM ${TARGET_DATASET^^} Weights (${RUN_NAME})",
  "id": "${KAGGLE_USERNAME}/${KAGGLE_DATASET_SLUG}",
  "licenses": [{"name": "cc-by-4.0"}]
}
EOF

  # Optional README with provenance
  cat > "${KAGGLE_DS_DIR}/README.md" <<'EOF'
# TRM Weights (Paper-Faithful Run)
- Source: SamsungSAILMontreal/TinyRecursiveModels (paper-faithful training)
- Contents: `model.ckpt` (final checkpoint)
- Notes: Use TRM repo `pretrain.py` or your own inference harness to load and evaluate.
EOF

  # Create dataset (or version if exists)
  log "Publishing dataset to Kaggle: ${KAGGLE_USERNAME}/${KAGGLE_DATASET_SLUG}"
  if kaggle datasets create -p "${KAGGLE_DS_DIR}"; then
    log "Kaggle dataset created."
  else
    log "Dataset may already exist; pushing a new version ..."
    kaggle datasets version -p "${KAGGLE_DS_DIR}" -m "update ${RUN_NAME}" || die "Kaggle dataset version failed"
  fi
}

# ------------------------------- MAIN FLOW ------------------------------------
main() {
  log "=== TRM Paper-Faithful Reproduction ==="
  log "Dataset: ${TARGET_DATASET} | Repo: ${TRM_DIR} | Run: ${RUN_NAME}"

  ensure_prereqs
  clone_trm
  install_env
  build_datasets
  train_arc

  # Optional Kaggle publishing of the trained weights as a dataset
  publish_kaggle_dataset

  log "=== DONE ==="
  log "Outputs under: ${TRM_DIR}/runs/${RUN_NAME}"
  log "If PUBLISH_TO_KAGGLE=1, dataset pushed: ${KAGGLE_USERNAME}/${KAGGLE_DATASET_SLUG}"
}

main "$@"
