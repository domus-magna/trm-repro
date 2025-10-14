
# TRM (Tiny Recursive Models) — Paper-Faithful Reproduction on ARC-AGI-2 → Publish Weights to Kaggle
# Single-agent playbook. One agent performs the entire workflow end-to-end.

> Scope: Reproduce upstream TRM training **as written** on **ARC-AGI-2** and publish the final checkpoint as a **Kaggle Dataset** for use in a Kaggle Notebook.  
> No source edits. No hyperparameter changes. Do **not** mix ARC-AGI-1 with ARC-AGI-2.

---

## 0) Global configuration (export these first)
Set these in your shell. Adjust paths/usernames as needed.

```bash
# --- Required knobs ---
export TARGET_DATASET=arc2                              # fixed for this runbook
export TRM_DIR="$HOME/TinyRecursiveModels"              # clone target for TRM repo
export PY="python3.10"                                  # must be Python 3.10
export USE_VENV=1
export VENV_DIR="$TRM_DIR/.venv"
export TORCH_INDEX_URL="https://download.pytorch.org/whl/nightly/cu126"  # adjust for your CUDA
export RUN_NAME="trm_arc2_paper_repro"
export MAX_GPUS_TO_USE=4

# --- Optional logging ---
export WANDB_API_KEY=""                                 # set to enable W&B login; leave empty to skip

# --- Kaggle publishing (optional but in scope here) ---
export PUBLISH_TO_KAGGLE=1                              # 1 = publish; 0 = skip
export KAGGLE_USERNAME="<your_kaggle_username>"         # required if publishing
export KAGGLE_DATASET_SLUG="trm-arc2-weights-${RUN_NAME}"
export KAGGLE_DS_DIR="$TRM_DIR/kaggle_dataset_arc2_${RUN_NAME}"
````

**Invariant checks (must hold throughout):**

* `TARGET_DATASET` **must equal** `arc2`.
* Do **not** touch model code or flags beyond those specified below.
* If `PUBLISH_TO_KAGGLE=1`: `~/.kaggle/kaggle.json` exists and has `chmod 600`.

---

## 1) System sanity & folders

```bash
set -euo pipefail

command -v git >/dev/null || { echo "git not found"; exit 1; }
command -v "$PY" >/dev/null || { echo "Python 3.10 not found (PY=$PY)"; exit 1; }
command -v pip >/dev/null || { echo "pip not found"; exit 1; }

mkdir -p "$TRM_DIR" ".repro" "artifacts" "logs"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L || true; fi
```

---

## 2) Clone upstream TRM (no modifications)

```bash
if [ ! -d "$TRM_DIR/.git" ]; then
  git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git "$TRM_DIR"
else
  (cd "$TRM_DIR" && git pull --ff-only)
fi
(cd "$TRM_DIR" && git rev-parse --short HEAD) | tee ".repro/trm_commit.txt"
```

---

## 3) Python environment & dependencies

```bash
# venv (recommended)
if [ "${USE_VENV:-1}" = "1" ]; then
  "$PY" -m venv "$VENV_DIR"
  # shellcheck disable=SC1090
  . "$VENV_DIR/bin/activate"
fi

# pip tooling
pip install --upgrade pip wheel setuptools

# PyTorch triplet (match your CUDA; replace index if needed)
pip install --pre --upgrade torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"

# TRM repo requirements
pip install -r "$TRM_DIR/requirements.txt"

# adam-atan2 (called out by TRM)
pip install --no-cache-dir --no-build-isolation adam-atan2

# Optional: Weights & Biases
if [ -n "${WANDB_API_KEY}" ]; then
  pip install wandb
  WANDB_API_KEY="$WANDB_API_KEY" wandb login
fi

# Record environment
{ python -V; pip list; python - <<'PY'
import torch, sys
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
PY
} | tee ".repro/env_${TARGET_DATASET}_${RUN_NAME}.txt"
```

**If torch install fails:**

* Pick the correct index for your CUDA (e.g., cu121, cu118) from PyTorch’s site.
* For CPU-only, use `--index-url https://download.pytorch.org/whl/nightly/cpu`.

---

## 4) Build ARC-AGI-2 dataset exactly as upstream

> This uses TRM’s dataset builder and **must** write to `data/arc2concept-aug-1000/`.
> The input prefix `kaggle/combined/arc-agi` must exist as per the TRM README’s data acquisition step.

```bash
cd "$TRM_DIR"

if [ ! -d "data/arc2concept-aug-1000" ]; then
  python -m dataset.build_arc_dataset \
    --input-file-prefix kaggle/combined/arc-agi \
    --output-dir data/arc2concept-aug-1000 \
    --subsets training2 evaluation2 concept \
    --test-set-name evaluation2
fi

[ -d "data/arc2concept-aug-1000" ] || { echo "ARC-AGI-2 dataset missing"; exit 2; }
find "data/arc2concept-aug-1000" -maxdepth 2 -type f | head -n 8
```

**If the builder can’t find inputs:**
Stop. Obtain the ARC-AGI-2 sources exactly as documented upstream, ensure the prefix path exists, then re-run.

---

## 5) Train TRM on ARC-AGI-2 (paper-faithful)

Hyperparameters/flags are **fixed** to upstream ARC examples.

```bash
cd "$TRM_DIR"
NGPUS=$( (nvidia-smi -L 2>/dev/null | wc -l) || echo 0 )
echo "GPUs detected: ${NGPUS}"

if [ "${NGPUS:-0}" -ge 4 ]; then
  # Multi-GPU (preferred if >=4 GPUs)
  torchrun --nproc-per-node "${MAX_GPUS_TO_USE}" \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
    pretrain.py \
      arch=trm \
      data_paths="[data/arc2concept-aug-1000]" \
      arch.L_layers=2 \
      arch.H_cycles=3 arch.L_cycles=4 \
      +run_name="${RUN_NAME}" ema=True
else
  # Single-GPU (longer runtime)
  python pretrain.py \
    arch=trm \
    data_paths="[data/arc2concept-aug-1000]" \
    arch.L_layers=2 \
    arch.H_cycles=3 arch.L_cycles=4 \
    +run_name="${RUN_NAME}" ema=True
fi
```

**Notes & guardrails:**

* Do **not** alter batch size, LR, WD, precision, or EMA flags unless upstream says so.
* If OOM on single GPU: stop and report; do **not** silently change flags.
* If crash mid-run and a resume flag exists upstream, use their documented semantics; otherwise re-start.

---

## 6) Verify artifacts & provenance

```bash
RUN_DIR="$TRM_DIR/runs/$RUN_NAME"
[ -d "$RUN_DIR" ] || { echo "Run folder missing: $RUN_DIR"; exit 3; }

CKPT=$(ls -t "$RUN_DIR"/checkpoints/* 2>/dev/null | head -n1 || true)
if [ -z "$CKPT" ]; then
  CKPT=$(ls -t "$RUN_DIR"/*.ckpt 2>/dev/null | head -n1 || true)
fi
[ -n "$CKPT" ] || { echo "No checkpoint found in $RUN_DIR"; exit 4; }

(cd "$TRM_DIR" && git rev-parse HEAD) > "$RUN_DIR/TRM_COMMIT.txt"
echo "pretrain.py arch=trm data_paths=[data/arc2concept-aug-1000] arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 +run_name=$RUN_NAME ema=True" \
  > "$RUN_DIR/COMMANDS.txt"

printf "RUN_NAME=%s\nCKPT=%s\nTRM_COMMIT=%s\n" \
  "$RUN_NAME" "$CKPT" "$(cat "$RUN_DIR/TRM_COMMIT.txt")" \
  | tee ".repro/summary_${RUN_NAME}.txt"

# Quick manifest
mkdir -p artifacts
printf "%s\n" \
  "RUN_DIR=$RUN_DIR" \
  "CKPT=$CKPT" \
  "ENV_LOG=.repro/env_${TARGET_DATASET}_${RUN_NAME}.txt" \
  "SUMMARY=.repro/summary_${RUN_NAME}.txt" \
  > artifacts/MANIFEST.txt
```

---

## 7) Publish weights to Kaggle as a Dataset (optional but in scope)

> This step creates (or versions) a Kaggle Dataset containing `model.ckpt`.

```bash
if [ "${PUBLISH_TO_KAGGLE:-0}" = "1" ]; then
  command -v kaggle >/dev/null || pip install kaggle
  [ -f "$HOME/.kaggle/kaggle.json" ] || { echo "Missing ~/.kaggle/kaggle.json"; exit 5; }
  chmod 600 "$HOME/.kaggle/kaggle.json" || true

  rm -rf "$KAGGLE_DS_DIR"
  mkdir -p "$KAGGLE_DS_DIR"
  cp -v "$CKPT" "$KAGGLE_DS_DIR/model.ckpt"

  # Dataset metadata
  cat > "$KAGGLE_DS_DIR/dataset-metadata.json" <<EOF
{
  "title": "TRM ARC-AGI-2 Weights (${RUN_NAME})",
  "id": "${KAGGLE_USERNAME}/${KAGGLE_DATASET_SLUG}",
  "licenses": [{"name": "cc-by-4.0"}]
}
EOF

  # Optional README
  cat > "$KAGGLE_DS_DIR/README.md" <<'EOF'
# TRM Weights (Paper-Faithful, ARC-AGI-2)
- Source: SamsungSAILMontreal/TinyRecursiveModels (upstream, no modifications)
- Contents: `model.ckpt` (final checkpoint)
- Usage: In a Kaggle Notebook, attach this Dataset, clone TRM, and load the checkpoint for inference/eval.
EOF

  # Create or version the dataset
  if kaggle datasets create -p "$KAGGLE_DS_DIR"; then
    echo "Kaggle dataset created: ${KAGGLE_USERNAME}/${KAGGLE_DATASET_SLUG}"
  else
    kaggle datasets version -p "$KAGGLE_DS_DIR" -m "update ${RUN_NAME}"
    echo "Kaggle dataset versioned: ${KAGGLE_USERNAME}/${KAGGLE_DATASET_SLUG}"
  fi
fi
```

---

## 8) Done — handoff to Kaggle Notebook

**Notebook steps (human-operated in Kaggle UI):**

1. New Notebook → **Add Data** → attach `/${KAGGLE_USERNAME}/${KAGGLE_DATASET_SLUG}`.
2. (Recommended) `git clone` the TRM repo in the notebook, `pip install -r requirements.txt`.
3. Load `/kaggle/input/<slug>/model.ckpt` with TRM’s code and run evaluation/inference.

---

## Troubleshooting (single-agent)

* **Torch install mismatch**: choose the correct `TORCH_INDEX_URL` for your CUDA; or install CPU wheels if no GPU.
* **Python 3.10 missing**: install Python 3.10; do not proceed with 3.11/3.12.
* **Dataset builder input missing**: provide the ARC-AGI-2 inputs exactly as per TRM README; do not substitute ARC-AGI-1.
* **CUDA OOM**: on multi-GPU, reduce `MAX_GPUS_TO_USE` but keep flags intact; on single-GPU OOM, stop and report (do not change batch unless upstream provides a flag).
* **Kaggle auth error (401/403)**: re-download `kaggle.json` from Kaggle → Account → API, place at `~/.kaggle/kaggle.json`, `chmod 600`.
* **No checkpoint found**: training likely failed early; check logs and re-run step 5.

---

## Repro checklist (tick all)

* [ ] `TARGET_DATASET=arc2` and **never changed**.
* [ ] TRM cloned, commit recorded in `runs/$RUN_NAME/TRM_COMMIT.txt`.
* [ ] `data/arc2concept-aug-1000/` built via the **exact** builder command.
* [ ] Training command exactly matches step 5 flags.
* [ ] Final checkpoint path captured; `COMMANDS.txt` written.
* [ ] Environment log saved under `.repro/`.
* [ ] Kaggle Dataset created/versioned (if `PUBLISH_TO_KAGGLE=1`).
* [ ] `artifacts/MANIFEST.txt` written with key paths.

```


