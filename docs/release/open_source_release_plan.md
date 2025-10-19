# TRM ARC-AGI-2 Open-Source Release – Research Notes

## Overview
Objective: prepare the TRM ARC-AGI-2 reproduction repository for a public open-source release that aligns with Hugging Face Hub guidance and general best practices for research codebases.

## Key Requirements & Best Practices

### Hugging Face Hub Expectations
- Provide a rich `README.md`/model card with YAML metadata (`library_name`, `license`, `tags`, `model-index`) to populate the right-hand panel and leaderboards.citeturn1search0
- Include evaluation results via `model-index` entries to surface metrics in Hub search and pinned cards.citeturn1search0
- Keep large artifacts tracked with Git LFS or upload via `huggingface-cli upload-large-folder` to avoid Git issues.citeturn5open0
- Document hardware, datasets, evaluation flow, and intended use/limitations within the model card.citeturn1search0

### Repository Hygiene
- Adopt a clear top-level layout separating source (`TinyRecursiveModels/`), scripts (`scripts/`), notebooks (`kaggle/`), and release assets (`huggingface_release/`).
- Provide a reproducibility checklist covering data acquisition, training invocation, config integrity, and environment capture.
- Remove transient artifacts (temporary logs, scratch wheels) or relocate them under `artifacts/` with README context.
- Ensure `.gitignore` covers generated data (datasets, checkpoints) to keep repo lightweight while linking to hosted copies (Kaggle/HF).

### Documentation Essentials
- Top-level `README.md` should include:
  1. Project summary + abstract.
  2. Quick start for evaluation/inference using Hugging Face checkpoint.
  3. Training reproduction instructions with explicit commands and environment setup.
  4. Links to original TRM repo, paper, Kaggle competition, and Hugging Face model page.
  5. Acknowledgements section (sponsors, contributors).citeturn3search0
- Create dedicated guides under `docs/`:
  - `docs/repro/` – step-by-step training/evaluation.
  - `docs/releases/` – change logs, provenance manifests.
  - Keep existing drafts (`docs/drafts/`) but mark as WIP if not ready for public consumption.

### Licensing & Attribution
- Retain upstream MIT license with clear reference to SamsungSAILMontreal/TinyRecursiveModels.citeturn3search0
- Add sponsor acknowledgement (Shawn Lewis, CoreWeave) wherever training resources are described (README, model card).
- Ensure third-party assets (Kaggle materials, W&B exports) comply with their licenses or are referenced via links instead of raw inclusion.

### Release Checklist Draft
1. Finalize repo layout (source, docs, scripts, artifacts).
2. Prune or relocate temporary files (`tmp_wheels*`, raw logs) into `artifacts/` or ignore list.
3. Update `.gitignore` for generated content.
4. Rewrite `README.md` following above template.
5. Create reproduction guide linking Hugging Face weights + Kaggle notebook.
6. Verify licensing & acknowledgements.
7. Run doc lint / link checks.
8. Tag release commit and cross-link from Hugging Face model card.

These notes feed the downstream Beads tasks (audit, cleanup, documentation, QA) to reach a publish-ready repository.
