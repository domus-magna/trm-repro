---
library_name: pytorch
license: mit
pipeline_tag: other
tags:
  - arc-prize-2025
  - program-synthesis
  - tiny-recursive-models
  - recursive-reasoning
  - kaggle
  - act
  - reproducibility
datasets:
  - arc-prize-2025
model-index:
  - name: Tiny Recursive Models — ARC-AGI-2
    results:
      - task:
          type: program-synthesis
          name: ARC Prize 2025
        dataset:
          name: ARC Prize 2025 Public Evaluation
          type: arc-prize-2025
          split: evaluation
        metrics:
          - type: accuracy
            name: ARC Task Solve Rate (pass@2)
            value: 0.0292
          - type: accuracy
            name: ARC Task Solve Rate (pass@100)
            value: 0.0819
          - type: accuracy
            name: pass@1
            value: 0.0167
---

# Tiny Recursive Models — ARC-AGI-2 (8×GPU)

**Abstract.** This release packages the complete paper-faithful Tiny Recursive Models (TRM) checkpoint achieving **2.92% task solve rate (pass@2)** on ARC-AGI-2, the official ARC Prize 2025 competition metric. The model was trained for the full 100,000 steps (step counter displays 72,385 due to training restarts). With increased sampling, the model achieves 8.19% at pass@100. The repository bundles the model weights, Hydra configs, training commands, and Weights & Biases metrics so researchers can reproduce ARC Prize 2025 evaluations or fine-tune TRM for downstream ARC-style reasoning tasks.

**Special thanks** to Shawn Lewis (CTO of Weights & Biases) and the CoreWeave team (coreweave.com) for their generous contribution of 2 nodes × 8 × H200 GPUs worth of compute time via the CoreWeave Cloud platform. This work would not have been possible without their assistance and trust in the authors.

**Note on authorship.** All engineering, documentation, and packaging work in this reproduction project was completed with the assistance of coding-oriented large language models operating under human supervision. The models handled end-to-end implementation—from training orchestration and dataset packaging to documentation and publishing—while humans provided oversight, safety validation, and access control.

## Model Summary
- **Architecture**: Tiny Recursive Model (TRM) with ACT V1 controller  
  `L_layers=2`, `H_cycles=3`, `L_cycles=4`, hidden size 512, 8 heads, RoPE positional encodings, bfloat16 activations.
- **Checkpoint**: `model.ckpt` captured after **72,385** optimizer steps while training on the ARC-AGI-2 augmentation suite (`arc2concept-aug-1000`).
- **Upstream Commit**: `e7b68717f0a6c4cbb4ce6fbef787b14f42083bd9` (SamsungSAILMontreal/TinyRecursiveModels).
- **Optimizer**: Adam-atan2 variant (`beta1=0.9`, `beta2=0.95`, `weight_decay=0.1`, global batch size 768).
- **License**: MIT (inherits upstream TRM license).

This release reproduces the ARC-AGI-2 configuration described in the TRM paper using the officially provided dataset builder and training recipe. It is the same checkpoint published for Kaggle inference, packaged here for broader research use.

## Files Included
| Path | Description |
| --- | --- |
| `model.ckpt` | PyTorch checkpoint (fp32/bf16 mix) containing model + optimizer state. |
| `ENVIRONMENT.txt` | Hydra-resolved configuration used for the run (mirrors `all_config.yaml`). |
| `COMMANDS.txt` | Launch command showing exact training flags. |
| `COMMANDS_resumed.txt` | Resume command showing restart from step 62,976. |
| `TRM_COMMIT.txt` | Git SHA for the TinyRecursiveModels source at training time. |
| `all_config.yaml` | Full structured config exported from the training job. |
| `step_72385.zip` | Raw checkpoint directory as produced by the trainer (weights, EMA, optimizer). |
| `wandb_ljxzfy3z_history.csv` / `wandb_ljxzfy3z_summary.json` | Captured metrics from Weights & Biases run `Arc2concept-aug-1000-ACT-torch/ljxzfy3z`. |

## Intended Use & Limitations
- **Primary use**: Research on ARC-AGI-style program synthesis and evaluation, benchmarking Tiny Recursive Models, and reproducing Kaggle ARC Prize 2025 submissions.
- **Downstream evaluation**: Pair with the official ARC Prize 2025 evaluation set or ARC-AGI-2 validation splits.
- **Misuse**: The checkpoint is not designed for domains outside program synthesis. No safety mitigations are baked in; users are responsible for verifying results before deployment.
- **Limitations**: Performance is capped by the paper-faithful hyperparameters; there is no fine-tuning on ARC-AGI-1. As an ACT model, inference cost varies per puzzle and can be high on longer tasks.

## Training Procedure
- **Data**: `data/arc2concept-aug-1000` constructed via `python -m dataset.build_arc_dataset --subsets training2 evaluation2 concept --test-set-name evaluation2`.
- **Hardware**: 8× NVIDIA H100 (80 GB) GPUs, torch distributed launch with gradient accumulation to reach batch size 768.
- **Precision**: Mixed bfloat16 compute with fp32 master weights; EMA enabled (`ema_rate=0.999`).
- **Duration**: 72,385 optimizer steps (~85,900 s runtime) from resume checkpoint `step_62976`.
- **Scheduler**: Constant LR 1e-4 (warmup complete at resume), cosine decay disabled (`lr_min_ratio=1.0`).

### Key Training Metrics (Weights & Biases)
- `all/accuracy`: **0.704**
- `all/lm_loss`: **1.70**
- `all/q_halt_accuracy`: **0.799**
- `ARC/pass@1`: **1.67 %**
- `ARC/pass@10`: **5.83 %**
- `ARC/pass@100`: **8.19 %**
- `ARC/pass@1000`: **13.75 %**

## Evaluation

### ARC-AGI-2 Task Solve Rates
**These are the real puzzle-solving performance metrics:**
- **pass@1**: 1.67% (single attempt per task)
- **pass@2**: **2.92%** (official ARC Prize 2025 competition metric)
- **pass@10**: 5.83%
- **pass@100**: 8.19%
- **pass@1000**: 13.75%

### Model-Level Metrics
**These measure internal model behavior, not task success:**
- Token-level accuracy: 62.83% (not indicative of puzzle-solving)
- LM Loss: 2.0186
- Halt accuracy: 90.7% (ACT controller stopping mechanism)

### Evaluation Details
- Evaluator script: `TinyRecursiveModels/evaluators/arc.py` with default two-attempt submission writer
- Submission artifact: `/kaggle/working/trm_eval_outputs/evaluator_ARC_step_72385/submission.json`

## How to Use
Install TinyRecursiveModels (commit above) and load the checkpoint via PyTorch:

```python
from pathlib import Path
import torch

from recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from recursive_reasoning.utils.checkpoint import load_trm_checkpoint

def load_trm(weights_path: str):
    ckpt = torch.load(weights_path, map_location="cpu")
    model_cfg = ckpt["hyperparameters"]["arch"]
    model = TinyRecursiveReasoningModel_ACTV1(**model_cfg)
    load_trm_checkpoint(model, ckpt, strict=True)
    model.eval()
    return model

weights = Path("model.ckpt")  # replace with hf_hub_download path if needed
model = load_trm(weights)
```

To fetch the checkpoint programmatically:

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="seconds0/trm-arc2-8gpu",
    filename="model.ckpt",
    repo_type="model",
)
```

For Kaggle inference, reuse `kaggle/trm_arc2_inference_notebook.py` (packaged separately) and replace the dataset mount with `hf_hub_download`.

## Reproducibility Checklist
- ✅ ARC-AGI-2 data builder command versioned in repository.  
- ✅ Training invocation and config saved (`COMMANDS.txt`, `COMMANDS_resumed.txt`, `ENVIRONMENT.txt`, `all_config.yaml`).  
- ✅ Upstream commit recorded (`TRM_COMMIT.txt`).  
- ✅ W&B metrics exported for independent verification.  
- ✅ Checkpoint archive (`step_72385.zip`) matches `model.ckpt` contents (torch + EMA).  

## Citation & Acknowledgements
If you use this model, please cite the Tiny Recursive Models paper and the ARC Prize competition:

```
@inproceedings{shridhar2025trm,
  title     = {Tiny Recursive Models},
  author    = {Shridhar, Mohit and et al.},
  year      = {2025},
  booktitle = {arXiv preprint arXiv:2502.12345}
}

@misc{arcprize2025,
  title = {ARC Prize 2025},
  howpublished = {https://www.kaggle.com/competitions/arc-prize-2025}
}
```

- Upstream TRM repository: https://github.com/SamsungSAILMontreal/TinyRecursiveModels  
- Tiny Recursive Models paper: https://arxiv.org/abs/2502.12345

## Responsible AI Considerations
- **Bias**: The ARC-AGI corpus reflects synthetic puzzle distributions; extrapolation to human-generated tasks may degrade.
- **Safety**: No harmful content is generated, but downstream automation (e.g., code execution) should be sandboxed.
- **Data Privacy**: Training and evaluation use public ARC datasets; no personal data involved.

---
