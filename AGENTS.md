# AGENTS.md

Status: Resuming TRM training (CoreWeave)

Directives for any agents/tools operating in this repository:

- Ignore README files: Treat any `README*` content as outdated and non‑authoritative. Do not use README text for decisions, configuration, or claims verification.
- Source of truth: Prefer data and runtime artifacts over prose. Concretely: `.beads/`, `artifacts/`, `huggingface_release/`, `infra/`, `kaggle/`, and any live logs/metrics produced by jobs.
- Current focus: Resume training from the most recent usable checkpoint on CoreWeave. Prefer single‑node H200 pods initially; scale only after stability checks.
- Resume safety: Always launch `trm-train-arc2-{8,4}gpu-resume.yaml` (not the baseline jobs). These manifests mount `trm-pyshim` and export `RESUME_*` so checkpoints hydrate correctly. Never run the plain `trm-train-arc2-{8,4}gpu` jobs unless you intentionally want a fresh run.
- Resume verification: The bootstrap script patches `pretrain.py` to seed `TrainState.step` from `RESUME_EXPECTED_STEP`. Every resume must show `[resume] initializing train_state.step to <expected>` in pod logs and the first W&B datapoint should be at or above that step with mature `train/lm_loss` (~0.3 for 8×, ~0.25 for 4×). If either signal is missing, stop the job immediately and investigate.
- Config guardrails: Do not remove or bypass the `trm-common-script` / `trm-pyshim` runtime patches—they enforce the resume guard and W&B metadata. Any future rollback scripts must reapply these ConfigMaps before creating jobs.
- Kaggle submissions: Always build the ARC **test** split (240 puzzles). Never switch notebooks to the evaluation subset when preparing leaderboard files; doing so causes scoring failures.
- ARC dataset builders: Training checkpoints created before 2025‑10‑26 used the legacy shuffled identifier order in `artifacts/TinyRecursiveModels_clean/dataset/build_arc_dataset.py`. Kaggle inference now uses the sorted identifier mapping in `TinyRecursiveModels/dataset/build_arc_dataset.py`. Keep these pipelines separate—resume training with the legacy builder, but evaluate/export with the sorted builder unless the checkpoint has been remapped. Document any remapping work explicitly and log the identifier table SHA256 before each run.
- W&B monitoring: To confirm resumes are loading the intended checkpoints, run `python scripts/monitor_wandb_runs.py --entity <team> --project trm-arc2 --run-name <resume_name>` (requires `WANDB_API_KEY`). This prints the latest loss/exact-accuracy/ARC pass@K stats reported by the run.
- Kaggle identifier mode: Until a remapped checkpoint exists, set `ARC_IDENTIFIER_MODE=legacy` (default) so the inference notebook copies `dataset/build_arc_dataset_legacy.py` over the sorted builder; once a sorted-compatible checkpoint is available, set `ARC_IDENTIFIER_MODE=sorted`.
- Copy-mode diagnostics: Activate the repo venv (`source .venv/bin/activate`) and run `PYTHONPATH=TinyRecursiveModels python scripts/debug_eval_cpu.py --checkpoint <ckpt> --dataset <dataset_dir> --batch-size 8 --max-examples 120`. Baseline outputs live at `artifacts/diagnostics/copy_mode_legacy.txt` (legacy builder) and `artifacts/diagnostics/copy_mode_sorted.txt` (remapped checkpoint); both currently report 0/120 grid matches, so further model-side remediation is required before switching Kaggle to the sorted mapping.
- Code changes: Keep edits minimal and scoped. Do not “fix” unrelated behavior. Update docs if behavior changes.
- Security/secrets: Never print tokens/keys from kubeconfigs or environment files. Redact credentials in logs and commits.

Scope: This file applies to the entire repository.
