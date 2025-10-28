# AGENTS.md

Status: Resuming TRM training (CoreWeave)

Directives for any agents/tools operating in this repository:

- Ignore README files: Treat any `README*` content as outdated and non‑authoritative. Do not use README text for decisions, configuration, or claims verification.
- Source of truth: Prefer data and runtime artifacts over prose. Concretely: `.beads/`, `artifacts/`, `huggingface_release/`, `infra/`, `kaggle/`, and any live logs/metrics produced by jobs.
- Current focus: Resume training from the most recent usable checkpoint on CoreWeave. Prefer single‑node H200 pods initially; scale only after stability checks.
- Kaggle submissions: Always build the ARC **test** split (240 puzzles). Never switch notebooks to the evaluation subset when preparing leaderboard files; doing so causes scoring failures.
- ARC dataset builders: Training checkpoints created before 2025‑10‑26 used the legacy shuffled identifier order in `artifacts/TinyRecursiveModels_clean/dataset/build_arc_dataset.py`. Kaggle inference now uses the sorted identifier mapping in `TinyRecursiveModels/dataset/build_arc_dataset.py`. Keep these pipelines separate—resume training with the legacy builder, but evaluate/export with the sorted builder unless the checkpoint has been remapped. Document any remapping work explicitly and log the identifier table SHA256 before each run.
- W&B monitoring: To confirm resumes are loading the intended checkpoints, run `python scripts/monitor_wandb_runs.py --entity <team> --project trm-arc2 --run-name <resume_name>` (requires `WANDB_API_KEY`). This prints the latest loss/exact-accuracy/ARC pass@K stats reported by the run.
- Kaggle identifier mode: Until a remapped checkpoint exists, set `ARC_IDENTIFIER_MODE=legacy` (default) so the inference notebook copies `dataset/build_arc_dataset_legacy.py` over the sorted builder; once a sorted-compatible checkpoint is available, set `ARC_IDENTIFIER_MODE=sorted`.
- Copy-mode diagnostics: Activate the repo venv (`source .venv/bin/activate`) and run `PYTHONPATH=TinyRecursiveModels python scripts/debug_eval_cpu.py --checkpoint <ckpt> --dataset <dataset_dir> --batch-size 8 --max-examples 120`. Baseline outputs live at `artifacts/diagnostics/copy_mode_legacy.txt` (legacy builder) and `artifacts/diagnostics/copy_mode_sorted.txt` (remapped checkpoint); both currently report 0/120 grid matches, so further model-side remediation is required before switching Kaggle to the sorted mapping.
- Code changes: Keep edits minimal and scoped. Do not “fix” unrelated behavior. Update docs if behavior changes.
- Security/secrets: Never print tokens/keys from kubeconfigs or environment files. Redact credentials in logs and commits.

Scope: This file applies to the entire repository.
