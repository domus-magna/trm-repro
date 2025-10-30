# TRM ARC-AGI-2 State Report — 2025-10-30 22:25 UTC

## Snapshot
- 8× H200 resume job `trm-train-arc2-8gpu-resume-8jjbk` relaunched from checkpoint step 249 575 with guard instrumentation; pod logs show the shim validating `RESUME_*` and `train_state.step` lifting to the expected value, yet ARC evaluator metrics remain flat at 0/172 matches. Evidence: `infra/kubernetes/trm-train-8gpu-resume.yaml`, `artifacts/diagnostics/arc_eval_step249575_pod.log`.
- Kaggle evaluation and submission kernels (latest v14/v4) still emit duplicate predictions for every puzzle (172/172 eval, 259/259 test) with ARC/pass@1 ≈ 0.004 (eval) and 0.0 (test). Evidence: `submission_download_eval_step249575_run_v14/trm-arc-agi-2-eval-step249575-run.log`, `submission_download_inference_step249575_run/trm-arc-agi-2-inference-step249575-run.log`, `artifacts/validation/kgl_fix/eval_validation_cli.txt`, `artifacts/validation/kgl_fix/test_validation_cli.txt`.
- Offline diagnostics continue to report near-zero grid matches: legacy vs sorted copy-mode both 0/120, CPU debug eval 1/120, despite embedding norms looking sane. Evidence: `artifacts/diagnostics/copy_mode_legacy.txt`, `artifacts/diagnostics/copy_mode_sorted.txt`, `artifacts/diagnostics/debug_eval_step249575.log`, `artifacts/diagnostics/embedding_norms.json`.
- Training/evaluator configs remain constrained to 16 ACT steps and deterministic argmax decoding, yielding exactly one candidate per puzzle. Attempts to raise `ARC_HALTING_MAX_STEPS` to 448 or enable stochastic sampling on Kaggle still collapse to a single candidate. Evidence: `artifacts/diagnostics/all_config_step249575.yaml`, `TinyRecursiveModels/evaluators/arc_utils.py`, Kaggle issue notes (`KGL-0012`).
- Beads board highlights 7 in-progress and 4 blocked items tied to resume reliability, Kaggle dataset packaging, and CoreWeave operations; downstream documentation and dataset tasks remain unassigned. Evidence: `.beads/trm-repro.db` (`RSM-EPIC-0001`, `KGL-0012`, `KGL-0014`, `RSM-0010`, `RSM-0011`, `RSM-0003`, `KGL-0008`, etc.).

## Training Resume Status

### 8× ARC2 resume (step 249 575)
- **Manifest & guard plumbing**: `infra/kubernetes/trm-train-8gpu-resume.yaml` exports `RESUME_CHECKPOINT_PATH=/workspace/TinyRecursiveModels/checkpoints/Arc2concept-aug-1000-ACT-torch/trm_arc2_8gpu_resume_step115815_plus100k_v2/step_249575`, `RESUME_EXPECTED_STEP=249575`, and mounts `trm-common-script`/`trm-pyshim`.
- **Guard behaviour**: `artifacts/diagnostics/arc_eval_step249575_pod.log` records repeated `[PYSHIM] resume env validated` and `[resume] initializing train_state.step to 249575` banners across ranks, confirming the runtime patches take effect before training loops start.
- **W&B**: the shim installs a wandb log wrapper before launch (`[PYSHIM] wandb.log wrapper installed`); job reports to run `0r6vbfiz` with resume metadata (per `RSM-EPIC-0001` notes). Previous resume baseline (`wandb_ljxzfy3z_summary.json`) plateaued at `_step=72385`, `ARC/pass@1=0.0167`.
- **Config**: `artifacts/diagnostics/all_config_step249575.yaml` shows `halt_max_steps=16`, `global_batch_size=768`, deterministic loss head, and `checkpoint_every_eval=True`.
- **Evaluator output**: the pod log sampler (`Processing batch … Completed inference in 16 steps`) plus `debug_eval_step249575.log` demonstrate ACT halting in ≤16 iterations with only 1/120 grid matches on CPU.
- **Open items (RSM-EPIC-0001)**: verify train_state hydrate > step_249575 via monitor script, capture W&B ARC metrics, and explore guard-induced load_checkpoint failure modes if performance remains flat.

### 4× ARC2 resume status
- `infra/kubernetes/trm-train-4gpu-resume.yaml` mirrors the guard workflow with `RESUME_EXPECTED_STEP=53428`. Tasks `trm-repro-7`, `RSM-0003`, and `RSM-0007` remain `open/blocked` while focus stays on stabilising the 8× job.
- Recent guard rollouts surfaced `ModuleNotFoundError: adam_atan2_backend` until `trm-common-script` rebuilt the wheel at runtime; resolved in current ConfigMap but highlighted in `RSM-EPIC-0001` log.
- Operator follow-up needed before relaunching the 4× canary once 8× metrics improve.

### Resume guard instrumentation
- `infra/kubernetes/trm-common-script-cm.yaml` patches `pretrain.py` to seed `TrainState.step` from `RESUME_EXPECTED_STEP` (and checkpoint suffix), asserts resume steps on rank 0, and injects `resume_expected_step`/`resume_checkpoint_path` into wandb config.
- `infra/kubernetes/trm-pyshim-cm.yaml` enforces presence/type of `RESUME_*` env vars, aborts on mismatches, wraps `wandb.log` to emit resume metadata, and preserves torch inference_mode hooks.
- Guard trip history (via `RSM-EPIC-0001` notes) shows earlier attempts failing when `train_state.step` stayed zero; current build passes but exposes real performance regression.

## Inference & Evaluator Diagnostics
- `TinyRecursiveModels/evaluators/arc.py` and `TinyRecursiveModels/evaluators/arc_utils.py` now rank candidates by average halt probability and count, yet every evaluation batch feeds exactly one candidate per puzzle from the deterministic `ACTLossHead` pipeline. Kaggle logs (e.g. `submission_download_eval_step249575_run_v14/trm-arc-agi-2-eval-step249575-run.log`) confirm `candidate_stats` count=1, producing duplicate padding.
- `KGL-0012` notes capture multiple Kaggle reruns (v2–v5) with overrides:
  - `ARC_HALTING_MAX_STEPS=448` → runtime increases (~80 min on CPU) but pass@1 remains 0.0042 (eval) / 0.0 (test).
  - Stochastic decoding experiments (Gumbel-softmax, sample count up to 8, temperature ≤1000) still result in a single candidate per puzzle.
- Offline copy-mode diagnostics:
  - `artifacts/diagnostics/copy_mode_legacy.txt`: 0/120 grid matches, mean halt prob 0.0587.
  - `artifacts/diagnostics/copy_mode_sorted.txt`: 0/120 grid matches, mean halt prob 0.0480.
  - `artifacts/diagnostics/debug_eval_step249575.log`: 1/120 grid matches, halt prob mean 0.0931, highlighting widespread divergence vs labels.
- `kaggle/notes/evaluation.txt` enumerates 120 evaluation puzzle IDs still failing; aligns with debug eval samples in diagnostics logs.
- Root cause hypotheses (from Beads notes): deterministic decoding + single aug path + ACT cap at 16 prevents exploration; need to reintroduce stochastic sampling or multiple dataset augmentations inside inference.

## Kaggle Pipeline & Offline Testing
- **Checkpoint packages**:
  - `artifacts/checkpoints/kaggle_dataset_8gpu_step119432/MANIFEST.txt` — SHA256 `2bc8bb3a…`, packaged 2025-10-28.
  - `artifacts/checkpoints/kaggle_dataset_8gpu_step249575/MANIFEST.txt` — SHA256 `3a46d78f…`, packaged 2025-10-29; `dataset-metadata.json` targets private dataset `seconds0/trm-arc2-weights-trm-arc2-8gpu-step249575`.
- **Kernel outputs** (all private / no leaderboard submissions):
  - Eval v14: `ARC/pass@1=0.004166…`, all higher pass@K identical; duplicates 172/172 (`artifacts/validation/kgl_fix/eval_validation_cli.txt`).
  - Submission v1–v5: identical duplicate rate 259/259, pass@1=0 (`submission_download_inference_step249575_run*/trm-arc-agi-2-inference-step249575-run.log`).
  - Submission artifacts hashed (e.g. SHA256 `fabca683…` for step 249 575 run) stored under `submission_download_inference_step249575_run`.
- **Repo snapshots**: `seconds0/trm-repo-clean` updated 2025-10-30 12:03 UTC with evaluator diagnostics. Despite patching candidate sorter, logs still show `[ARC DEBUG]` with `candidates=1`.
- **Dataset builder log**: `artifacts/logs/raw/dataset-builder.out` reports augmentation shortfalls (numerous puzzles capped at 576/72 variants) but overall dataset size `Total puzzle IDs: 1 191 731`.
- **Validation scripts**: `artifacts/validation/kgl_fix/augmentation_test.log`, `.../eval_validation_cli.txt`, `.../test_validation_cli.txt` confirm schema integrity yet persistent duplicate attempts.

## Dataset & Identifier Mapping
- Legacy vs sorted identifier resources live in `artifacts/diagnostics/identifier_mappings/`. Training checkpoints prior to 2025-10-26 rely on the legacy shuffled ordering (`TinyRecursiveModels_clean/dataset/build_arc_dataset.py`), while evaluation/export pipelines after remap expect sorted mapping (`TinyRecursiveModels/dataset/build_arc_dataset.py`).
- Hash guardrails (2025-10-29):
  - Legacy evaluation split SHA256: `c364837393c2428e40c6116692fb1b66bf011108ec9930475df306cd779bbfd1`.
  - Legacy test split SHA256: `9af3f07ab5c05320e2da99c85ad76086f7cbabe2159b5cf694da01aa7e33546f`.
- `artifacts/diagnostics/embedding_norms.json` shows identical norm statistics between legacy and sorted embeddings (mean ≈ 3.2455, no zero rows), reducing likelihood of embedding corruption after remap.
- `KGL-0014` (in progress) tracks remapped checkpoint `artifacts/checkpoints/kaggle_dataset_8gpu_step149833_sorted/model.ckpt` and associated diagnostics; copy-mode parity remains unsolved (0/120 matches).
- `scripts/debug_eval_cpu.py` supports `ARC_IDENTIFIER_MODE` toggles (`legacy` vs `sorted`) for offline evaluation; ensure `PYTHONPATH=TinyRecursiveModels` and `ARC_IDENTIFIER_MODE=legacy` before running inference notebooks until remap issues resolved.

## Diagnostics & Tooling Inventory
- Training logs: `artifacts/diagnostics/arc_eval_step249575_pod.log`, `artifacts/logs/raw/wandb_ljxzfy3z_history.csv`, `huggingface_release/trm_arc2_8gpu/COMMANDS_resumed.txt`.
- Checkpoint configs: `artifacts/8gpu_checkpoint_resume_plus100k_v2/all_config.yaml`, `huggingface_release/trm_arc2_8gpu/all_config.yaml`.
- Monitoring scripts: `scripts/monitor_wandb_runs.py` (requires `WANDB_API_KEY`), `scripts/debug_eval_cpu.py`, `infra/scripts/run_eval_job.sh`, `infra/scripts/trm-dataset-build.sh`.
- Tests: `tests/test_arc_evaluator.py` exercises new candidate scoring logic.

## Active Beads Workstreams (2025-10-30)

### In progress
- `RSM-EPIC-0001` — Resume guard rollout & verification; latest note logs successful guard but unmet accuracy targets.
- `RSM-0010` — Extend TRM runs by +100k steps; 8× job active (`pod trm-train-arc2-8gpu-resume-bp6tn` earlier, superseded by step 249 575 run).
- `RSM-0011` — Checkpoint scheduling; cronjob `trm-checkpoint-prune` keeps last 10 snapshots (review if cadence sufficient for +100k steps).
- `RSM-0005` — Guard mapping/embedding alignment (no notes yet, likely tied to identifier audits).
- `KGL-0012` — Kaggle submission kernel validation; multiple private runs logged, duplicates unresolved.
- `KGL-0014` — Remapped checkpoint pipeline; embedding norms captured, awaiting inference fix.
- `KGL-0010` — Dataset with evaluation solutions on CoreWeave; job re-applied but pending completion evidence.

### Blocked
- `trm-repro-4` — ARC1 launch deprioritised until ARC2 stabilises.
- `KGL-0008` — CoreWeave eval job requires operator to fetch logs (`infra/scripts/run_eval_job.sh`).
- `RSM-0003` / `RSM-0007` — 4× canary resume monitoring paused until guard validation finishes.

### High-priority open items
- `trm-repro-5` — Cancel stray ARC2 4× jobs if any remain.
- `trm-repro-6` — Launch ARC1 8× job once ARC2 pipeline healthy.
- `trm-repro-19/20/21` — Document dataset, checkpoint, and evaluation progression (feed this report back into Beads).
- `trm-repro-22/23` — Resume/restart history + environment capture.
- `KGL-0005` — Stage GPU eval plan (still unexecuted).
- `KGL-0009` — Refresh Kaggle dataset with latest repo archive after evaluator fix.

## Key Risks & Unknowns
- **Model performance regression**: Even with resume guard verified, ARC metrics stagnate. Need to determine whether training data, optimizer state, or evaluation config diverged from the productive regime pre-step 249 575.
- **Candidate collapse**: Deterministic decoding yields single candidate per puzzle, nullifying duplicate mitigation and pass@K sampling. Requires architectural or sampling change (temperature noise, multiple forward passes, data augmentation).
- **Identifier remap**: Sorted mapping pipeline remains unusable; switching inference without remapped checkpoints risks total failure. Legacy mode must stay active until copy-mode accuracy improves.
- **Operational dependencies**: CoreWeave job control (cancel/resume/eval) depends on operator availability; blocked tasks hinder validation loop.
- **Checkpoint retention**: Current prune cronjob keeps 10 checkpoints; ensure this covers future +100k step runs without deleting necessary resumes.

## Immediate Next Steps (proposed)
1. Run `scripts/monitor_wandb_runs.py --entity <team> --project trm-arc2 --run-name trm_arc2_8gpu_resume_step115815_plus100k_v2` to capture up-to-date ARC metrics and confirm resume offsets.
2. Capture a fresh `debug_eval_cpu` sweep with `--halt-max-steps 448` and extended samples to quantify whether longer ACT loops or stochasticity change candidate counts.
3. Prototype multi-sample decoding in training evaluator (e.g. set `ARC_SAMPLING_COUNT>1`) and sync the same change into Kaggle notebooks, then rerun evaluation to observe candidate histograms.
4. Revisit dataset augmentation pipeline to restore multiple augmented views per puzzle for inference (check `artifacts/logs/raw/dataset-builder.out` shortfalls).
5. Coordinate with operators to unblock `KGL-0008` and `RSM-0003/0007`, ensuring 4× canary launch once 8× metrics improve.
6. Update Beads tickets (`trm-repro-19/20/21`) with this report, then plan follow-up experiments (halting schedule, sampling diversity, optimizer resume checks).
