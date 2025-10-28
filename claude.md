# CLAUDE.md

## Monitoring W&B Runs

- Use `python scripts/monitor_wandb_runs.py --entity <team-or-username> --project trm-arc2 --run-name trm_arc2_8gpu_resume_step115815_plus100k_v2` to verify resume jobs.  
- The script prints key metrics (`train/lm_loss`, `train/exact_accuracy`, `ARC/pass@K`, etc.) and confirms the checkpoint path from `config.load_checkpoint`.  
- `WANDB_API_KEY` must be available in the environment (run `wandb login` first if needed).

## Identifier Mapping Guardrails

- Training continues to rely on the legacy shuffled builder (under `artifacts/TinyRecursiveModels_clean`).  
- The Kaggle inference notebook honours `ARC_IDENTIFIER_MODE` (`legacy` by default). When set to `legacy`, it copies `dataset/build_arc_dataset_legacy.py` over the sorted builder before dataset construction; switch to `sorted` once a remapped checkpoint is available.  
- If you build datasets manually, run `python scripts/check_identifier_mapping.py --dataset <dir> --expect <legacy|sorted>` to confirm the hash matches the chosen mode.
- Remapping tool: `source .venv/bin/activate && python scripts/remap_puzzle_embeddings.py --input-ckpt <legacy_ckpt> --output-ckpt <sorted_ckpt> --permutation artifacts/diagnostics/identifier_mappings/identifier_permutation.json --direction legacy_to_sorted`. The current remapped artifact is `artifacts/checkpoints/kaggle_dataset_8gpu_step149833_sorted/model.ckpt`, but diagnostics in `artifacts/diagnostics/copy_mode_sorted.txt` still show 0/120 matches, so keep Kaggle runs on `ARC_IDENTIFIER_MODE=legacy` until model accuracy improves.
