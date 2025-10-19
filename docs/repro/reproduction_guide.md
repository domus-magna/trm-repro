# TRM ARC-AGI-2 Reproduction Guide

This document walks through reproducing the ARC-AGI-2 Tiny Recursive Models training run and preparing evaluation artifacts. It assumes access to 8× H200-class GPUs (or equivalent) but provides single-GPU fallbacks.

## 0. Prerequisites
- Linux host with CUDA 12.6 drivers and Python 3.10.
- NVIDIA GPUs (8× recommended, single GPU acceptable for longer training).
- Kaggle account with API token (`~/.kaggle/kaggle.json`, chmod 600).
- Optional: Hugging Face token for publishing (`hf auth login`).

## 1. Environment Setup
```bash
python3.10 -m venv ~/.venvs/trm
source ~/.venvs/trm/bin/activate
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
pip install -r TinyRecursiveModels/requirements.txt
pip install --no-cache-dir --no-build-isolation adam-atan2
```

> If using Weights & Biases, run `WANDB_API_KEY=... wandb login`.

## 2. Fetch Source & Data
```bash
git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git
cd TinyRecursiveModels
git checkout e7b68717f0a6c4cbb4ce6fbef787b14f42083bd9
```

Build ARC-AGI-2 dataset (requires `kaggle/combined/arc-agi` input tree as described in the upstream README):
```bash
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2
```

## 3. Training
### Multi-GPU (8× H200)
```bash
torchrun --nproc-per-node 8 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  pretrain.py \
    arch=trm \
    data_paths="[data/arc2concept-aug-1000]" \
    arch.L_layers=2 \
    arch.H_cycles=3 arch.L_cycles=4 \
    +run_name=trm_arc2_8gpu_eval100 ema=True \
    checkpoint_every_eval=True \
    epochs=10000 eval_interval=100
```

Resume from the provided checkpoint to reach step 72,385:
```bash
torchrun --nproc-per-node 8 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  pretrain.py \
    arch=trm \
    data_paths="[data/arc2concept-aug-1000]" \
    arch.L_layers=2 \
    arch.H_cycles=3 arch.L_cycles=4 \
    +run_name=trm_arc2_8gpu_eval100 ema=True \
    checkpoint_every_eval=True \
    epochs=10000 eval_interval=100 \
    +load_checkpoint=checkpoints/Arc2concept-aug-1000-ACT-torch/trm_arc2_8gpu_eval100/step_62976
```

### Single-GPU Fallback
```bash
python pretrain.py \
  arch=trm \
  data_paths="[data/arc2concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=trm_arc2_1gpu ema=True
```

If out-of-memory occurs, stop and scale out; do not alter batch size or precision.

## 4. Evaluation & Kaggle Submission
### Local Evaluation
```bash
python evaluators/run_arc_eval.py \
  --checkpoint /path/to/model.ckpt \
  --output-dir runs/trm_arc2_eval
```

Inspect `submission.json` and metrics under the output directory.

### Kaggle Notebook
1. Create new GPU notebook (T4 or A100).
2. Attach datasets:
   - `seconds0/trm-arc2-weights-trm_arc2_8gpu_eval100`
   - `seconds0/trm-offline-wheels-py311`
   - `seconds0/trm-repo-clean`
   - `arc-prize-2025`
3. Copy contents of `kaggle/trm_arc2_inference_notebook.py`.
4. Run all cells; expect accuracy ≈0.628 on public evaluation.
5. Manually upload `submission.json` (or configure Kaggle CLI with `kaggle.json`).

## 5. Publishing Artifacts
- **Hugging Face**: Upload `huggingface_release/trm_arc2_8gpu` via `hf upload-large-folder` after logging in with `hf auth login`.
- **Kaggle Dataset**: Package `artifacts/checkpoints/kaggle_dataset_8gpu_step_72385/` with `kaggle datasets version -p ... -m "update"`.

## 6. Provenance Checklist
- ✅ `COMMANDS.txt` and `ENVIRONMENT.txt` captured for each run.
- ✅ W&B metrics exported to `artifacts/logs/raw/`.
- ✅ Upstream commit recorded in `TRM_COMMIT.txt`.
- ✅ Sponsor acknowledgement included in `README.md` and model card.

For additional context, see:
- [artifacts/README.md](../../artifacts/README.md)
- [huggingface_release/README.md](../../huggingface_release/trm_arc2_8gpu/README.md)
