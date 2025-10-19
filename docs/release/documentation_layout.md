# Documentation Layout Plan

Goal: define the documentation architecture for the open-source TRM ARC-AGI-2 release, ensuring clear entry points for users, researchers, and operators.

## Top-Level `README.md` Outline
1. **Project Overview**
   - Short abstract + sponsor acknowledgement.
   - Links to Hugging Face model, Kaggle competition, upstream TRM repo/paper.
2. **Quickstart (Inference/Evaluation)**
   - Instructions for downloading weights from Hugging Face.
   - Kaggle notebook usage with attached datasets.
3. **Training Reproduction**
   - Environment setup (Python 3.10, PyTorch nightly index).
   - Dataset builder command for ARC-AGI-2.
   - Distributed training command (8-GPU & single-GPU fallback).
4. **Repository Layout**
   - Table summarizing directories (`TinyRecursiveModels/`, `kaggle/`, `docs/`, `artifacts/`, `infra/`, etc.).
5. **Provenance & Artifacts**
   - Link to manifests (in `docs/release/`) and W&B exports.
6. **Acknowledgements & Licensing**
   - Sponsor thank you, upstream TRM attribution, license summary.
7. **Support & Contributions**
   - How to file issues, point to Beads tracker if published, contribution guidelines (future).

## `docs/` Structure
- `docs/release/`
  - `open_source_release_plan.md` – research references.
  - `repo_audit.md` – inventory and actions.
  - `release_notes.md` (to add) – changelog per public release.
- `docs/repro/`
  - `training.md` – step-by-step training reproduction.
  - `evaluation.md` – Kaggle inference + submission guidance.
- `docs/analysis/`
  - Postmortems, evaluation studies (`HPC_VERIFICATION_PROBLEM_ANALYSIS.md`, etc.).
- `docs/ops/`
  - Cloud/Kubernetes procedures (`backblaze_restore`, support ticket notes).
- `docs/drafts/`
  - Retain WIP content; add README clarifying draft status.

## Artifact Documentation
- Add `artifacts/README.md` summarizing subdirectories:
  - `artifacts/checkpoints/` → Kaggle-ready datasets (step manifests).
  - `artifacts/kaggle_support/` → Offline wheels, repo zips.
  - `artifacts/logs/` → Curated logs (move from `logs/` once trimmed).
- Each artifact subfolder should include a `MANIFEST.md` listing files, size, and usage.

## Operational Assets
- Create `infra/` directory for Kubernetes manifests and launch scripts:
  - `infra/kubernetes/` → `trm-*.yaml`, config maps.
  - `infra/scripts/` → `monitor_*.sh`, `smart_launcher.sh`, `backblaze_restore/`.
  - Add `infra/README.md` explaining deployment assumptions and sensitive values to omit.

## Additional Files
- `CONTRIBUTING.md` (new) – outline issue reporting, code style, testing expectations.
- `LICENSE` – confirm MIT; add SPDX headers to scripts where feasible.
- `.gitignore` – update to exclude datasets, wheels, scratch files, `.env`.
- `CODE_OF_CONDUCT.md` – optional but recommended for public release.

This layout will drive the cleanup (trm-repro-32) and documentation drafting (trm-repro-33/35).
