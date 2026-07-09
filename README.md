# Tiny Recursive Models — ARC-AGI-2 Reproduction

**Abstract.** This repository packages a **partial reproduction** of Tiny Recursive Models (TRM) on the ARC-AGI-2 dataset. On the 120-task public evaluation set, our run scores **2.92% pass@2** against the paper's **7.8%** on the same metric — roughly 37% of the published result. We reproduce the paper's architecture, dataset construction, and hyperparameters, but not its score. We provide the final checkpoint, Kaggle-ready assets, a [per-task breakdown](results/README.md) of what the model solved and failed to solve, and documentation to reproduce evaluation and training end-to-end. Training ran for approximately 135,361 cumulative steps; the W&B step counter displays 72,385 because `TrainState.step` resets on resume (`step_offset = 62,976`).

> **Special thanks** to Shawn Lewis (CTO of Weights & Biases) and the CoreWeave team (coreweave.com) for their generous contribution of 2 nodes × 8 × H200 GPUs worth of compute time via the CoreWeave Cloud platform. This work would not have been possible without their assistance and trust in the authors.

> **Note on authorship.** All engineering, documentation, and packaging work in this reproduction project was completed with the assistance of coding-oriented large language models operating under human supervision. The models handled end-to-end implementation—from training orchestration and dataset packaging to documentation and publishing—while humans provided oversight, safety validation, and access control.

## Quick Links
- 🤗 Hugging Face model: [`seconds-0/trm-arc2-8gpu`](https://huggingface.co/seconds-0/trm-arc2-8gpu)
- 🧠 Upstream TRM repo: [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- 📄 Tiny Recursive Models paper: [arXiv:2510.04871](https://arxiv.org/abs/2510.04871) — *Less is More: Recursive Reasoning with Tiny Networks*
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

## 4. Results & What the Metrics Mean

Evaluation of checkpoint step 72,385 on the ARC-AGI-2 public evaluation set (120 tasks, 167 test outputs). Full breakdown of which tasks were solved: [`results/README.md`](results/README.md).

| Metric | Value | Submittable? |
| --- | --- | --- |
| `ARC/pass@1` | 1.67% | yes |
| `ARC/pass@2` | **2.92%** | yes — this is the headline number |
| `ARC/pass@5` | 5.00% | no |
| `ARC/pass@10` | 5.83% | no |
| `ARC/pass@100` | 8.19% | no |
| `ARC/pass@1000` | 13.75% | no |

**`pass@k` for k > 2 is an oracle metric, not a score.** In TRM's evaluator, `k` is a rank cutoff into the pool of distinct candidate grids produced across ~1000 test-time augmentations, ranked by vote count (`evaluators/arc.py`, `for h, stats in p_map[:k]`). `pass@100` asks whether the correct grid appears anywhere in the top 100 candidates — answering that requires the ground truth, so it cannot be submitted. `submission_K = 2` caps a real submission at two grids. **Do not compare our `pass@100` of 8.19% to the paper's 7.8%**, which is a `pass@2`. The comparable figures are 2.92% (ours) and 7.8% (paper).

**`ARC/pass@2` is not the official ARC Prize metric,** despite the name. TRM computes a per-task macro-average with fractional credit: `mean over tasks of (test outputs solved / test outputs in task)`. The ARC Prize scores a micro-average over all test outputs. Both are reported by [`results/grade_submission.py`](results/grade_submission.py). On this run they give:

- TRM metric (comparable to the paper): **2.92%** — 3.5 tasks' worth of credit out of 120
- Official ARC Prize metric: **2.99%** — 5 of 167 test outputs

**`all/accuracy = 0.7035` is per-cell token accuracy, not a solve rate.** A model that predicts the output grid's shape and copies most of the input scores highly on it. It is not indicative of puzzle-solving; `all/exact_accuracy = 0.0118` is the per-example exact-match rate.

Two further caveats on `pass@k`. Upstream sets `aggregated_voting=True`, which makes `begin_eval()` a no-op, so the candidate pool accumulates votes across successive evaluation rounds rather than being scored from a single pass — the reported `pass@k` pools roughly 11 rounds of EMA checkpoints. And the evaluation is transductive: the demonstration pairs of all 120 evaluation tasks are in the training split with learned per-task puzzle embeddings. This is permitted under ARC Prize rules, but the number is not zero-shot generalization to unseen tasks. Test output grids never enter the training split.

## 5. Provenance
- `results/` — Per-task predictions and scoring for the reported run.
- `artifacts/checkpoints/` — Kaggle dataset manifests including `ENVIRONMENT.txt`, `COMMANDS.txt`, and `TRM_COMMIT.txt`.
- `artifacts/logs/raw/` — W&B export (`wandb_ljxzfy3z_history.csv`, `wandb_ljxzfy3z_summary.json`) and dataset builder logs.
- `huggingface_release/README.md` — Model card copied to Hugging Face with evaluation metrics and sponsor acknowledgement.

> ⚠️ **The Hugging Face checkpoint is not the checkpoint reported above.** [`seconds-0/trm-arc2-8gpu`](https://huggingface.co/seconds-0/trm-arc2-8gpu) ships **step 119,432** (see its `MANIFEST.txt`), from a later resume run, which scores 0.83% pass@1 — its inference path generated duplicate candidates, collapsing `pass@2` onto `pass@1`. The step 72,385 checkpoint that produced the 2.92% figure is not currently released. The bundled `wandb_ljxzfy3z_summary.json` describes step 72,385 and therefore does not describe the shipped weights.

## 6. Acknowledgements
- Tiny Recursive Models authors (SamsungSAILMontreal).
- ARC Prize 2025 organizers.
- Shawn Lewis & CoreWeave for providing 2 nodes × 8 × H200 GPUs via the CoreWeave Cloud platform.
- Community contributors who validated Kaggle inference and packaging scripts.

## 7. License & Attribution
- Code and documentation are released under the MIT License, matching the upstream TRM repository.
- ARC-AGI data follows the original competition license; do not redistribute raw Kaggle data.
- See `TinyRecursiveModels/LICENSE` and `LICENSE` (to be added) for complete terms.


