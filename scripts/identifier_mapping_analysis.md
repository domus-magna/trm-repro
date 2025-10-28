# Identifier Mapping Analysis

## CRITICAL BUG FOUND

### Issue: Non-Deterministic Identifier Mapping

**Location:** `TinyRecursiveModels/dataset/build_arc_dataset.py:202`

```python
# Shuffle puzzles
puzzles = list(puzzles.items())
np.random.shuffle(puzzles)  # <-- DEPENDS ON RANDOM SEED!
```

### Impact

The identifier mapping (which maps puzzle names to integer IDs for the embedding table) is created based on the **order that puzzles are processed**. Since puzzles are shuffled with `np.random.shuffle()` before processing, the identifier mapping depends on the random seed.

**If different seeds were used when building:**
1. Training dataset → Seed A → Identifier map A
2. Evaluation dataset → Seed B → Identifier map B

Then the puzzle embeddings would be **misaligned**:
- Model trained with embedding #100 for puzzle "abc123"
- Evaluation dataset has embedding #150 for puzzle "abc123"
- Model uses embedding #100 during evaluation → **wrong puzzle!**

### Code Flow

```
load_puzzles_arcagi() [line 167]
  ↓
  np.random.shuffle(puzzles) [line 202]
  ↓
  convert_single_arc_puzzle() [line 126]
  ↓
convert_dataset() [line 225]
  ↓
  np.random.seed(config.seed) [line 226]
  ↓
  identifier_map creation [lines 232-240]
  - Iterates through data in processing order
  - Assigns sequential IDs based on order encountered
```

### Root Cause

The identifier mapping is created by iterating through puzzles in **shuffle order** and assigning sequential IDs:

```python
identifier_map = {}
for split_name, split in data.items():
    for subset_name, subset in split.items():
        for group in subset:
            for puzzle in group:
                if puzzle.id not in identifier_map:
                    identifier_map[puzzle.id] = num_identifiers
                    num_identifiers += 1
```

Since `data` contains puzzles in shuffled order, the IDs depend on the shuffle seed.

### Verification Needed

1. Check what seed was used when building the training dataset
2. Check what seed was used when building the evaluation dataset for Kaggle
3. If seeds differ, this explains the zero-score issue

### Fix Options

**Option 1: Sort puzzle IDs alphabetically** (RECOMMENDED)
```python
# Instead of random shuffle, sort deterministically
puzzles = sorted(puzzles.items(), key=lambda x: x[0])  # Sort by puzzle ID
```

**Option 2: Use same seed everywhere**
- Document and enforce seed=42 (or any fixed value) for all dataset builds
- Still fragile if anyone forgets

**Option 3: Store identifier map with checkpoint**
- Save identifier_map during training
- Load it during evaluation
- Ensures exact alignment

### Test Case

To verify this bug:

1. Build dataset with seed=1
2. Build dataset with seed=2
3. Compare identifier maps
4. Expected: DIFFERENT mappings
5. This proves the bug

### Related Files

- `TinyRecursiveModels/dataset/build_arc_dataset.py` - Dataset builder with shuffle
- `TinyRecursiveModels/evaluators/arc.py` - Uses identifier_map to look up puzzle names
- `kaggle/trm_arc2_inference_notebook.py` - Builds evaluation dataset at runtime

### Severity

**CRITICAL** - This bug would cause:
- All predictions to be for wrong puzzles
- Zero exact matches even if model is working correctly
- Explains "non-zero token accuracy but zero pass@K"
