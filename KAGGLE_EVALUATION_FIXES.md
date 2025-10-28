# Kaggle Evaluation Bug Fixes

## Summary

Investigation of the Kaggle zero-score issue revealed **3 critical bugs** that have been fixed:

## Bug #1: Hash Lookup Padding Bug (CRITICAL)

**File:** `TinyRecursiveModels/evaluators/arc.py:152-161`

**Issue:** When top-K voted prediction hashes weren't found in any worker's hashmap, the code would skip them and pad with duplicates of the first prediction. This caused `attempt_1 == attempt_2` for most puzzles, reducing pass@2 effectiveness to near-zero.

**Original Code:**
```python
for h, stats in p_map[:self.submission_K]:  # Only check top-K
    for hmap, preds in global_hmap_preds:
        if h in hmap:
            pred_grids.append(hmap[h])
            break

while len(pred_grids) < self.submission_K:
    pred_grids.append(pred_grids[0])  # Duplicate!
```

**Fix:** Iterate through ALL predictions until K grids are found:
```python
for h, stats in p_map:  # Check ALL candidates, not just top-K
    if len(pred_grids) >= self.submission_K:
        break
    for hmap, preds in global_hmap_preds:
        if h in hmap:
            pred_grids.append(hmap[h])
            break

while len(pred_grids) < self.submission_K:
    pred_grids.append(pred_grids[0])
```

**Impact:** Should eliminate duplicate attempts and improve pass@2 scores.

---

## Bug #2: Identifier Mapping Shuffle Bug (CRITICAL)

**File:** `TinyRecursiveModels/dataset/build_arc_dataset.py:200-202`

**Issue:** Puzzle IDs were randomly shuffled before being assigned integer identifiers for the embedding table. If different random seeds were used when building the training vs. evaluation datasets, the identifier mappings would be different. This caused:
- Model trained with embedding #100 for puzzle "abc123"
- Evaluation dataset has embedding #150 for puzzle "abc123"
- Model uses wrong embedding #100 during evaluation → predictions for wrong puzzle!

**Original Code:**
```python
# Shuffle puzzles
puzzles = list(puzzles.items())
np.random.shuffle(puzzles)  # DEPENDS ON RANDOM SEED!
```

**Fix:** Sort puzzles deterministically by ID:
```python
# Sort puzzles deterministically by ID to ensure consistent identifier mapping
# (Previously used random shuffle which caused identifier drift between datasets)
puzzles = sorted(puzzles.items(), key=lambda x: x[0])
```

**Impact:** This was likely the PRIMARY cause of zero scores. Non-zero token accuracy but zero exact matches makes perfect sense if the model is generating reasonable grids for the wrong puzzles.

---

## Bug #3: Diagnostic Logging Added

**File:** `TinyRecursiveModels/evaluators/arc.py`

**Enhancement:** Added extensive diagnostic logging to help debug future issues:

- Verbose mode shows top-3 prediction hashes vs. ground truth for first 10 examples
- Tracks statistics:
  - Total test examples processed
  - Examples with no predictions
  - Examples with hash matches
  - Examples with duplicate attempts (attempt_1 == attempt_2)

**Usage:**
```python
evaluator = ARC(data_path, eval_metadata, verbose=True)
```

---

## Augmentation Inverse Tests (VERIFIED WORKING)

**File:** `scripts/test_aug_simple.py`

**Result:** ✅ ALL TESTS PASSED

Verified that:
- All 8 dihedral transforms and their inverses work correctly
- Aug/inverse_aug round trips restore original grids
- Color 0 (black) is preserved in color permutations
- Grid hashing is deterministic and collision-free

**Conclusion:** Augmentation/inverse logic is NOT the source of the bug.

---

## EMA Weights (VERIFIED WORKING)

**Analysis:** Code inspection of `pretrain.py:619-642` confirmed:

```python
if config.ema:
    print("SWITCH TO EMA")
    train_state_eval = copy.deepcopy(train_state)
    train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)  # Apply EMA
else:
    train_state_eval = train_state

# ... run evaluation ...

save_train_state(config, train_state_eval)  # Saves EMA weights if EMA enabled
```

EMA weights ARE being used correctly during training/evaluation and saved in checkpoints.

**Conclusion:** EMA handling is NOT the source of the bug.

---

## Next Steps

1. ✅ Apply all fixes to codebase (DONE)
2. ⏳ Rebuild evaluation dataset with fixed code
3. ⏳ Run local validation on evaluation split
4. ⏳ Update Kaggle inference kernel with fixes
5. ⏳ Submit to Kaggle and verify leaderboard score

---

## Expected Outcome

With these fixes, we expect:

1. **Identifier mapping fix** → Predictions should now be for the correct puzzles → pass@K > 0
2. **Hash lookup fix** → attempt_1 ≠ attempt_2 → improved pass@2 scores
3. **Diagnostic logging** → Can verify which examples match and identify remaining issues

The identifier mapping bug was almost certainly the root cause of zero scores. The hash lookup bug would have further degraded pass@2 performance even if the identifier bug was fixed.

---

## Files Modified

- `TinyRecursiveModels/evaluators/arc.py` - Hash lookup fix + diagnostic logging
- `TinyRecursiveModels/dataset/build_arc_dataset.py` - Identifier mapping sort fix
- `scripts/test_aug_simple.py` - Augmentation unit tests (new)
- `scripts/identifier_mapping_analysis.md` - Bug analysis (new)

---

## Verification Commands

```bash
# Test augmentation inverse
python3.11 scripts/test_aug_simple.py

# Rebuild dataset with fixes (requires proper environment)
# python3 TinyRecursiveModels/dataset/build_arc_dataset.py --input-file-prefix <path> ...

# Run evaluation with verbose logging
# Add verbose=True to evaluator config in pretrain config

# Verify identifier mapping before packaging
python3 scripts/check_identifier_mapping.py --dataset artifacts/diagnostics/identifier_mappings/sorted --expect sorted

# When validating a checkpoint package
python3 scripts/validate_checkpoint_for_kaggle.py \
    --package-dir artifacts/checkpoints/kaggle_dataset_8gpu_step115815 \
    --dataset-dir /path/to/built_arc_dataset
```

## Recent Guardrails

- `scripts/check_identifier_mapping.py` compares `identifiers.json` hashes against the sorted inference builder to prevent mismatched embeddings.
- `scripts/validate_checkpoint_for_kaggle.py` accepts `--dataset-dir`/`--identifier-mode` and enforces the matching hash when packaging uploads.
- `kaggle/trm_arc2_inference_notebook.py` now honours `ARC_IDENTIFIER_MODE` (`legacy` by default) and aborts if the runtime hash differs from the expected mapping, catching accidental regressions.
