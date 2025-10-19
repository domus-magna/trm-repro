# TRM ARC-AGI-2 Weights (8-GPU Training Run)

This dataset contains the final checkpoint from a paper-faithful Tiny Recursive Models (TRM) reproduction on ARC-AGI-2 using 8 GPUs.

## Contents

- `model.ckpt` — Checkpoint from step 72,385 (~72,385 optimizer steps completed)
- `TRM_COMMIT.txt` — Upstream TRM Git commit hash used for training
- `COMMANDS.txt` — Exact training invocation captured from the training run
- `ENVIRONMENT.txt` — Model configuration (all_config.yaml from training)

## Training Details

- **Run Name**: trm_arc2_8gpu_eval100
- **Checkpoint Step**: 72,385
- **GPUs**: 8x (distributed training)
- **Dataset**: ARC-AGI-2 (arc2concept-aug-1000)
- **Model**: TRM (Tiny Recursive Models)
- **Architecture**:
  - L_layers: 2
  - H_cycles: 3
  - L_cycles: 4
  - hidden_size: 512
  - num_heads: 8

## Usage

Attach this dataset to a Kaggle Notebook, clone the upstream TRM repository from [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels), install dependencies, and load the checkpoint via TRM's evaluation utilities.

## References

- TRM Repository: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- ARC-AGI Competition: https://arcprize.org/
