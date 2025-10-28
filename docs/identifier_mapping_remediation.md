# Identifier Mapping Remediation Plan

## Context

Kaggle evaluation kernels (and anything under `TinyRecursiveModels/`) now build ARC datasets with identifiers sorted by puzzle ID (see `TinyRecursiveModels/dataset/build_arc_dataset.py`). Training checkpoints produced prior to 2025‑10‑26 were built with the legacy shuffled order preserved in `artifacts/TinyRecursiveModels_clean/dataset/build_arc_dataset.py`. Using a checkpoint trained with one ordering and evaluating with the other scrambles the sparse puzzle embedding and yields 0 % exact matches.

Relevant references:

- `KAGGLE_EVALUATION_FIXES.md` — Bug #2 (“Identifier Mapping Shuffle Bug”).
- Kaggle notebook: `kaggle/trm_arc2_inference_notebook.py`.
- Training builder snapshot: `artifacts/TinyRecursiveModels_clean/dataset/build_arc_dataset.py`.

## Investigation Plan

1. **Snapshot identifier tables**  
   - Regenerate evaluation datasets with both builders (sorted vs. shuffled).  
   - Persist `identifiers.json` and derivable permutation matrices in `artifacts/diagnostics/identifier_mappings/`.

2. **Checkpoint alignment audit**  
   - For each active checkpoint bundle, record which builder produced the data (via MANIFEST notes or hash logs).  
   - Compute the permutation that maps the legacy ordering to the sorted ordering.

3. **Runtime validation**  
   - Extend `scripts/debug_eval_cpu.py` (or GPU analogue) to accept a permutation and verify pass@K after remapping embeddings.  
   - Automate a sanity check that compares prediction hashes before/after permutation.

## Remediation Path (Chosen)

**Option A (Dual Pipelines)**  
We will keep the builders split for now:

- **Training / resume jobs** mount the legacy builder (`artifacts/TinyRecursiveModels_clean/dataset/build_arc_dataset.py`).  
- **Evaluation / Kaggle inference** can run in two modes: `ARC_IDENTIFIER_MODE=legacy` (default) copies `dataset/build_arc_dataset_legacy.py` over the sorted builder to remain compatible with legacy checkpoints; once a remapped checkpoint exists, switch to `ARC_IDENTIFIER_MODE=sorted`.  
- The permutation materialized in `artifacts/diagnostics/identifier_mappings/identifier_permutation.json` is used to remap checkpoints when/if we copy weights between the two worlds. A simple hash check (see below) guards against mixing identifiers accidentally.

Rationale: avoids touching validated checkpoints while giving us a deterministic way to migrate later. Once a new checkpoint is trained end‑to‑end with the sorted mapping we can sunset the legacy path.

## Test Strategy

1. **Unit tests**  
   - Add regression tests ensuring sorted builder output matches permutation expectations (e.g., puzzle `0934a4d8` index).  
   - Verify old builder still matches checkpoint MANIFEST.

2. **Integration tests**  
   - Before pushing a Kaggle kernel, run evaluation twice: (a) straight sorted builder, (b) sorted builder plus applied permutation to puzzle embedding → expect identical hash matches after fix.  
   - Record metrics in beads/diagnostics for traceability.

3. **Monitoring**  
   - Update W&B / Kaggle logs template to print the identifier ordering hash (e.g., SHA256 of `identifiers.json`) so we can spot mismatches immediately. The planned sanity check should exit early if the hash does not match the expected builder for that run. Use `scripts/check_identifier_mapping.py` for offline verification, and rely on the runtime guard added to `kaggle/trm_arc2_inference_notebook.py`.

## Next Actions

- Decide between Option A (dual pipeline with explicit remap) vs. Option B (permanent checkpoint remap) before the next Kaggle submission.  
- Implement the chosen flow and document the process directly in the Kaggle repo README substitute and AGENTS.md.  
- Backfill existing checkpoints with metadata identifying their builder provenance.
