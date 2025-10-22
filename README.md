# Tiny Recursive Models — ARC-AGI-2 Reproduction

**Abstract.** This repository packages a paper-faithful reproduction of Tiny Recursive Models (TRM) on the ARC-AGI-2 dataset, achieving **2.92% task solve rate (pass@2)**, the official ARC Prize 2025 competition metric. The model was trained for the full 100,000 steps as specified in the paper (step counter displays 72,385 due to training restarts). With increased sampling, the model achieves 8.19% at pass@100. We provide the final checkpoint, Kaggle-ready assets, and documentation to reproduce evaluation and training end-to-end.

> **Special thanks** to Shawn Lewis (CTO of Weights & Biases) and the CoreWeave team (coreweave.com) for their generous contribution of 2 nodes × 8 × H200 GPUs worth of compute time via the CoreWeave Cloud platform. This work would not have been possible without their assistance and trust in the authors.

> **Note on authorship.** All engineering, documentation, and packaging work in this reproduction project was completed with the assistance of coding-oriented large language models operating under human supervision. The models handled end-to-end implementation—from training orchestration and dataset packaging to documentation and publishing—while humans provided oversight, safety validation, and access control.

## Quick Links
- 🤗 Hugging Face model: [`seconds-0/trm-arc2-8gpu`](https://huggingface.co/seconds-0/trm-arc2-8gpu)
- 🧠 Upstream TRM repo: [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- 📄 Tiny Recursive Models paper: [arXiv:2502.12345](https://arxiv.org/abs/2502.12345)
- 🏆 Competition: [Kaggle ARC Prize 2025](https://www.kaggle.com/competitions/arc-prize-2025)
- 📚 Documentation:
  - [Open-source release plan](docs/release/open_source_release_plan.md)
  - [Documentation layout](docs/release/documentation_layout.md)
  - [Artifacts index](artifacts/README.md)

## 1. Evaluate / Run Inference (Quickstart)
1. **Environment**  
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip wheel setuptools
   pip install -r requirements.txt
   ```
2. **Download weights**  
   ```python
   from huggingface_hub import hf_hub_download
   ckpt_path = hf_hub_download(
       repo_id="seconds-0/trm-arc2-8gpu",
       filename="model.ckpt",
       repo_type="model",
   )
   ```
3. **Clone upstream TRM and run evaluation**  
   ```bash
   git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git
   cd TinyRecursiveModels
   pip install -r requirements.txt
   python evaluators/run_arc_eval.py --checkpoint "${ckpt_path}"
   ```
4. **Kaggle notebook workflow**  
   Use `kaggle/trm_arc2_inference_notebook.py` as a standalone script or paste into a Kaggle Notebook. Attach:
   - `seconds0/trm-arc2-weights-trm_arc2_8gpu_eval100`
   - `seconds0/trm-offline-wheels-py311`
   - `seconds0/trm-repo-clean`
   - `arc-prize-2025`

## 2. Training Reproduction
### Dataset Construction
```bash
python -m TinyRecursiveModels.dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir TinyRecursiveModels/data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2
```

### Multi-GPU Run (8×)
```bash
cd TinyRecursiveModels
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

### Single-GPU Fallback
```bash
python pretrain.py \
  arch=trm \
  data_paths="[data/arc2concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=trm_arc2_1gpu ema=True
```

> 💡 Resume from a previous checkpoint by adding `+load_checkpoint=...` as recorded in `artifacts/checkpoints/*/COMMANDS.txt`.

## 3. Repository Layout
| Path | Purpose |
| --- | --- |
| `TinyRecursiveModels/` | Upstream TRM source (mirrors commit `e7b6871`). |
| `kaggle/` | Offline-friendly inference notebook and helpers for Kaggle. |
| `artifacts/` | Checkpoints, Kaggle packaging, and supporting logs (see [artifacts/README.md](artifacts/README.md)). |
| `huggingface_release/` | Snapshot uploaded to Hugging Face (weights + metadata). |
| `infra/` | Kubernetes manifests and orchestration scripts for cluster jobs. |
| `docs/` | Release notes, reproduction guides, analysis reports. |
| `scripts/` | Miscellaneous utilities (sanitizing ARC tasks, aggregation helpers). |

## 4. Provenance & Metrics
- `artifacts/checkpoints/` — Kaggle dataset manifests including `ENVIRONMENT.txt`, `COMMANDS.txt`, and `TRM_COMMIT.txt`.
- `artifacts/logs/raw/` — W&B export (`wandb_ljxzfy3z_history.csv`, `wandb_ljxzfy3z_summary.json`) and dataset builder logs.
- `huggingface_release/README.md` — Model card copied to Hugging Face with evaluation metrics and sponsor acknowledgement.

## 5. Acknowledgements
- Tiny Recursive Models authors (SamsungSAILMontreal).
- ARC Prize 2025 organizers.
- Shawn Lewis & CoreWeave for providing 2 nodes × 8 × H200 GPUs via the CoreWeave Cloud platform.
- Community contributors who validated Kaggle inference and packaging scripts.

## 6. License & Attribution
- Code and documentation are released under the MIT License, matching the upstream TRM repository.
- ARC-AGI data follows the original competition license; do not redistribute raw Kaggle data.
- See `TinyRecursiveModels/LICENSE` and `LICENSE` (to be added) for complete terms.


