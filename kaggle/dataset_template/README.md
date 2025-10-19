# TRM ARC-AGI-2 Weights

This dataset bundles the final checkpoint from a paper-faithful Tiny Recursive Models (TRM) reproduction on ARC-AGI-2.

## Contents

- `model.ckpt` — latest checkpoint from the training run.
- `TRM_COMMIT.txt` — upstream TRM Git commit hash used for training.
- `COMMANDS.txt` — exact training invocation captured from Hydra.
- `ENVIRONMENT.txt` — Python package inventory and torch/cuda availability.

## Usage

Attach this dataset to a Kaggle Notebook, clone the upstream TRM repository, install dependencies, and load the checkpoint via TRM's evaluation utilities. Refer to the reproduction notes for dataset-specific evaluation scripts.
