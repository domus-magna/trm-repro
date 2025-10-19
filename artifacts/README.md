# Artifacts Directory

This folder collects derived assets generated while reproducing the TRM ARC-AGI-2 runs. Nothing here is required to build the project from source, but the subdirectories provide provenance for checkpoints, Kaggle packaging, and evaluation logs.

## Layout
- `checkpoints/` – Kaggle dataset bundles for the 8-GPU and 4-GPU runs (each includes `model.ckpt`, `ENVIRONMENT.txt`, and provenance docs).
- `kaggle_support/` – Offline wheels and helper assets used to execute Kaggle notebooks without internet access (`kaggle_dataset_trm_offline_wheels_v2/`, temporary wheel caches).
- `logs/` – Curated logs (currently `raw/` contains W&B export CSV/JSON and dataset builder output).
- `8gpu_checkpoint_latest/` – Legacy checkpoint directory harvested from the training cluster (kept for reference).
- `built_wheels/` – Early experiments while assembling the offline wheels dataset (kept for reference).
- (Removed) `kaggle_kernel_*` – Intermediate Kaggle notebook exports; see `docs/analysis/kaggle_wheels_review.md` for context.
- `TinyRecursiveModels_clean/` – Sanitized zip of the upstream TRM repository packaged for Kaggle.

## Next Steps
- Trim redundant wheel archives prior to final release.
- Add `MANIFEST.md` files inside subdirectories with checksums and usage notes.
- Ensure `.gitignore` excludes any future large or sensitive artifacts generated locally.
