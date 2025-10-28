# CLAUDE.md

## Monitoring W&B Runs

- Use `python scripts/monitor_wandb_runs.py --entity <team-or-username> --project trm-arc2 --run-name trm_arc2_8gpu_resume_step115815_plus100k_v2` to verify resume jobs.  
- The script prints key metrics (`train/lm_loss`, `train/exact_accuracy`, `ARC/pass@K`, etc.) and confirms the checkpoint path from `config.load_checkpoint`.  
- `WANDB_API_KEY` must be available in the environment (run `wandb login` first if needed).

## Resume Pipeline Checklist

- Always apply the `trm-common-script`, `trm-sitecustomize`, and `trm-pyshim` ConfigMaps before launching resume jobs; `infra/kubernetes/trm-train-{8,4}gpu-resume.yaml` assumes those patches are present.  
- Pod logs must include:  
  1. `Loading checkpoint .../step_<N>` (from the overlay loader).  
  2. `[resume] initializing train_state.step to <N>` (from the bootstrap patch).  
  3. No `Resume guard` exceptions.  
  If any of these are missing, the run is invalid—delete the job and fix the configuration.  
- After launch, check W&B: the first `_step` should equal the expected resume step and `train/lm_loss` should be in the ~0.2–0.4 band. Losses near 2.6 indicate a fresh start.
- Never run the plain `trm-train-arc2-{8,4}gpu` manifests unless you explicitly want to start from scratch; they do not hydrate the resume step.

## Identifier Mapping Guardrails

- Training continues to rely on the legacy shuffled builder (under `artifacts/TinyRecursiveModels_clean`).  
- The Kaggle inference notebook honours `ARC_IDENTIFIER_MODE` (`legacy` by default). When set to `legacy`, it copies `dataset/build_arc_dataset_legacy.py` over the sorted builder before dataset construction; switch to `sorted` once a remapped checkpoint is available.  
- If you build datasets manually, run `python scripts/check_identifier_mapping.py --dataset <dir> --expect <legacy|sorted>` to confirm the hash matches the chosen mode.
- Remapping tool: `source .venv/bin/activate && python scripts/remap_puzzle_embeddings.py --input-ckpt <legacy_ckpt> --output-ckpt <sorted_ckpt> --permutation artifacts/diagnostics/identifier_mappings/identifier_permutation.json --direction legacy_to_sorted`. The current remapped artifact is `artifacts/checkpoints/kaggle_dataset_8gpu_step149833_sorted/model.ckpt`, but diagnostics in `artifacts/diagnostics/copy_mode_sorted.txt` still show 0/120 matches, so keep Kaggle runs on `ARC_IDENTIFIER_MODE=legacy` until model accuracy improves.
