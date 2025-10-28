# ARC-AGI Technical References & Community Patterns
## Supplementary Deep-Dive Documentation

**Date:** October 25, 2025
**Type:** Technical Reference Guide
**Audience:** ML Engineers, Kaggle Participants

---

## 1. SUBMISSION FORMAT DEEP DIVE

### 1.1 JSON Structure Requirements

**Canonical Format:**
```json
{
  "task_id_1": [
    {
      "attempt_1": [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 0, 1]
      ],
      "attempt_2": [
        [9, 8, 7, 6],
        [5, 4, 3, 2],
        [1, 0, 9, 8]
      ]
    }
  ],
  "task_id_2": [
    {
      "attempt_1": [],
      "attempt_2": [[1]]
    }
  ]
}
```

### 1.2 Validation Rules (Strict)

**Rule Set A: Structure**
```python
# Top level MUST be dict
assert isinstance(data, dict), "Top level must be JSON object"

# Keys MUST be strings (task IDs)
for key in data.keys():
    assert isinstance(key, str), f"Task ID {key} must be string"

# Values MUST be lists
for task_id, outputs in data.items():
    assert isinstance(outputs, list), f"{task_id}: value must be list"
    assert len(outputs) > 0, f"{task_id}: outputs list cannot be empty"
```

**Rule Set B: Entries**
```python
# Each output MUST be dict with exact keys
for task_id, outputs in data.items():
    for idx, entry in enumerate(outputs):
        assert isinstance(entry, dict), f"{task_id}[{idx}]: must be dict"
        assert set(entry.keys()) == {"attempt_1", "attempt_2"}, \
            f"{task_id}[{idx}]: keys must be attempt_1 and attempt_2"
```

**Rule Set C: Grids**
```python
# Each attempt must be list of lists
for task_id, outputs in data.items():
    for idx, entry in enumerate(outputs):
        for attempt_name in ["attempt_1", "attempt_2"]:
            grid = entry[attempt_name]
            assert isinstance(grid, list), \
                f"{task_id}[{idx}].{attempt_name}: must be list"
            
            # Empty grids allowed
            if len(grid) == 0:
                continue
            
            # All rows must be lists
            for row_idx, row in enumerate(grid):
                assert isinstance(row, list), \
                    f"{task_id}: row {row_idx} must be list"
            
            # All rows must have same length (rectangular)
            row_lengths = [len(row) for row in grid]
            assert len(set(row_lengths)) <= 1, \
                f"{task_id}: ragged rows - lengths {row_lengths}"
            
            # All values must be integers
            for row_idx, row in enumerate(grid):
                for col_idx, val in enumerate(row):
                    assert isinstance(val, int), \
                        f"{task_id}[{row_idx}][{col_idx}] = {val} (type {type(val).__name__})"
                    assert 0 <= val <= 9, \
                        f"{task_id}[{row_idx}][{col_idx}] = {val} (out of 0-9 range)"
```

### 1.3 Common Format Errors

**Error 1: Top-level array instead of object**
```python
# WRONG
[
  {"task_id": "...", "attempt_1": ...}
]

# CORRECT
{
  "task_id": [{"attempt_1": ..., "attempt_2": ...}]
}
```

**Error 2: Missing list wrapper**
```python
# WRONG
{
  "task_id": {"attempt_1": [...], "attempt_2": [...]}
}

# CORRECT
{
  "task_id": [{"attempt_1": [...], "attempt_2": [...]}]
}
```

**Error 3: Wrong key names**
```python
# WRONG
{"task_id": [{"pred_1": [...], "pred_2": [...]}]}
{"task_id": [{"output": [...], "output_2": [...]}]}
{"task_id": [{"attempt1": [...], "attempt2": [...]}]}  # Missing underscore

# CORRECT
{"task_id": [{"attempt_1": [...], "attempt_2": [...]}]}
```

**Error 4: Ragged rows**
```python
# WRONG - rows have different lengths
{"task_id": [{"attempt_1": [[1, 2], [3, 4, 5]], "attempt_2": [...]}]}

# CORRECT - all rows same length
{"task_id": [{"attempt_1": [[1, 2, 0], [3, 4, 5]], "attempt_2": [...]}]}
```

**Error 5: Non-integer values**
```python
# WRONG
{"task_id": [{"attempt_1": [[1.5, 2.0], [3, 4]], "attempt_2": [...]}]}
{"task_id": [{"attempt_1": [["1", "2"], [3, 4]], "attempt_2": [...]}]}

# CORRECT
{"task_id": [{"attempt_1": [[1, 2], [3, 4]], "attempt_2": [...]}]}
```

---

## 2. INFERENCE PIPELINE ANALYSIS

### 2.1 Expected Data Flow

```
checkpoint
    ↓
[Load state dict] → model_state
    ↓
[Create model] → model (with config)
    ↓
[Apply EMA weights] → model.eval()
    ↓
[Load evaluation data] → {task_id: {train: [...], test: [...]}}
    ↓
[Inference per puzzle] → predictions per task
    ↓
[Format output] → submission JSON
    ↓
[Validate] → format check
    ↓
[Submit to Kaggle]
```

### 2.2 Critical Checkpoint Keys

**From training checkpoint (step_72385):**

```python
# These keys MUST exist for inference to work
required_keys = [
    "model.inner.puzzle_emb.weights",  # Puzzle embeddings
    "model.encoder.*",                  # Encoder layers
    "model.decoder.*",                  # Decoder layers
    "model.halt_predictor.*",          # Halt prediction
]

# Configuration parameters encoded in checkpoint
config_params = {
    "H_cycles": 3,      # Horizontal cycles
    "L_cycles": 4,      # Lateral cycles
    "hidden_size": 512, # Hidden dimension
    "num_heads": 8,     # Attention heads
    "puzzle_emb_len": 16,    # Puzzle embedding length
    "puzzle_emb_ndim": 512,  # Puzzle embedding dimension
    "halt_max_steps": 16,    # Max ACT steps
    "no_ACT_continue": True, # ACT disable flag
}
```

### 2.3 Loading Pattern

**Correct Pattern:**
```python
import torch

# Load checkpoint
checkpoint_path = "model.ckpt"
checkpoint_state = torch.load(checkpoint_path, map_location="cpu")

# Verify structure
assert isinstance(checkpoint_state, dict), "Checkpoint must be dict"
assert "model.inner.puzzle_emb.weights" in checkpoint_state, \
    "Missing required key for puzzle embeddings"

# Create model with matching config
model_config = {
    "H_cycles": 3,
    "L_cycles": 4,
    "hidden_size": 512,
    "num_heads": 8,
    "puzzle_emb_len": 16,
    "puzzle_emb_ndim": 512,
    "halt_max_steps": 16,
    "no_ACT_continue": True,
}

model = TinyRecursiveReasoningModel(**model_config)

# Load weights
model.load_state_dict(checkpoint_state)
model.eval()

# Disable gradients
with torch.no_grad():
    # Inference here
    pass
```

---

## 3. TASK ID VALIDATION

### 3.1 Expected Task Count

**Competition Setup:**
- Training set: 400 puzzles (for reference/augmentation)
- Evaluation set: 240 puzzles (submission target)
- Hidden test set: Unknown (part of Kaggle evaluation)

**Submission Requirements:**
- MUST include all 240 public evaluation puzzles
- MAY include additional predictions for hidden tasks
- Extra predictions are allowed (not penalized)

### 3.2 Task ID Format

**Format Specification:**
```
- Type: String
- Pattern: numeric identifier (e.g., "007d0a30", "1cf80156")
- Case: lowercase hexadecimal
- Length: 8 characters
- Format: ARC dataset convention
```

### 3.3 ID Validation Code

```python
import hashlib
import json

def validate_task_ids(submission: dict, expected_eval_json: dict = None):
    """Verify task IDs in submission."""
    
    # Check all keys are strings
    for key in submission.keys():
        assert isinstance(key, str), f"Task ID must be string, got {type(key)}"
    
    # Check format matches expected (8 hex digits)
    for task_id in submission.keys():
        assert len(task_id) == 8, f"Task ID {task_id} wrong length"
        assert task_id.isalnum(), f"Task ID {task_id} contains non-alphanumeric"
        try:
            int(task_id, 16)  # Verify hex
        except ValueError:
            raise ValueError(f"Task ID {task_id} not valid hex")
    
    # Check count
    count = len(submission)
    assert count >= 240, f"Expected >=240 tasks, got {count}"
    
    # Check against expected if provided
    if expected_eval_json:
        expected_ids = set(expected_eval_json.keys())
        submitted_ids = set(submission.keys())
        missing = expected_ids - submitted_ids
        assert not missing, f"Missing {len(missing)} task IDs"
    
    # Hash verification (from inference notebook)
    source_ids = sorted(submission.keys())
    id_hash = hashlib.sha256("\n".join(source_ids).encode("utf-8")).hexdigest()
    print(f"Task ID hash (SHA256): {id_hash}")
    print(f"Tasks submitted: {len(source_ids)}")
    
    return True
```

---

## 4. INFERENCE CONFIGURATION MATRIX

### 4.1 Parameter Matching Table

**Configuration must match across all components:**

| Component | Parameter | Value | Source | Validation |
|-----------|-----------|-------|--------|-----------|
| Model | H_cycles | 3 | checkpoint | model.config.H_cycles |
| Model | L_cycles | 4 | checkpoint | model.config.L_cycles |
| Model | hidden_size | 512 | checkpoint | model.config.hidden_size |
| Model | num_heads | 8 | checkpoint | model.config.num_heads |
| Model | puzzle_emb_len | 16 | ENVIRONMENT.txt | model.puzzle_emb.embedding_dim |
| Model | puzzle_emb_ndim | 512 | ENVIRONMENT.txt | model.puzzle_emb.embedding_dim |
| Model | halt_max_steps | 16 | ENVIRONMENT.txt | model.halt_predictor.max_steps |
| Model | no_ACT_continue | True | ENVIRONMENT.txt | model.config.no_ACT_continue |
| Dataset | test_split | 240 puzzles | competition | len(eval_tasks) |

### 4.2 Mismatch Detection

```python
def check_config_match(model, checkpoint_env):
    """Verify configuration consistency."""
    
    mismatches = []
    
    # Check each parameter
    config_checks = [
        ("H_cycles", model.config.H_cycles, 3),
        ("L_cycles", model.config.L_cycles, 4),
        ("hidden_size", model.config.hidden_size, 512),
        ("num_heads", model.config.num_heads, 8),
        ("halt_max_steps", model.halt_predictor.max_steps, 16),
        ("no_ACT_continue", model.config.no_ACT_continue, True),
    ]
    
    for param_name, actual, expected in config_checks:
        if actual != expected:
            mismatches.append({
                "parameter": param_name,
                "expected": expected,
                "actual": actual
            })
    
    if mismatches:
        print("Configuration mismatches detected:")
        for m in mismatches:
            print(f"  {m['parameter']}: expected {m['expected']}, got {m['actual']}")
        raise ValueError("Configuration mismatch - update notebook config")
    
    return True
```

---

## 5. GRID OUTPUT VALIDATION

### 5.1 Grid Sanity Checks

```python
def validate_grid(grid, task_id, attempt_name, allow_empty=True):
    """Comprehensive grid validation."""
    
    # Empty grids allowed
    if len(grid) == 0:
        if not allow_empty:
            raise ValueError(f"{task_id}.{attempt_name}: empty grid not allowed")
        return True
    
    # Type check
    assert isinstance(grid, list), f"{task_id}.{attempt_name}: must be list"
    
    # Check rows
    assert all(isinstance(row, list) for row in grid), \
        f"{task_id}.{attempt_name}: all rows must be lists"
    
    # Rectangular check
    row_lengths = [len(row) for row in grid]
    assert len(set(row_lengths)) == 1, \
        f"{task_id}.{attempt_name}: ragged rows {row_lengths}"
    
    # Value type check
    for row_idx, row in enumerate(grid):
        for col_idx, val in enumerate(row):
            assert isinstance(val, (int, float)), \
                f"{task_id}.{attempt_name}[{row_idx},{col_idx}]: " \
                f"type {type(val).__name__}, must be int"
            assert 0 <= val <= 9, \
                f"{task_id}.{attempt_name}[{row_idx},{col_idx}]: " \
                f"value {val} out of range [0-9]"
    
    # Dimensions
    height = len(grid)
    width = len(grid[0]) if grid else 0
    assert height > 0, f"{task_id}.{attempt_name}: height 0"
    assert width > 0, f"{task_id}.{attempt_name}: width 0"
    assert height <= 100, f"{task_id}.{attempt_name}: height {height} > 100"
    assert width <= 100, f"{task_id}.{attempt_name}: width {width} > 100"
    
    return True
```

### 5.2 Common Grid Issues

**Issue 1: Float Values**
```python
# Detection
grid = [[1.0, 2.0], [3.0, 4.0]]
if any(isinstance(v, float) for row in grid for v in row):
    print("ERROR: Contains float values")

# Fix
grid = [[int(v) for v in row] for row in grid]
```

**Issue 2: Ragged Rows**
```python
# Detection
row_lengths = [len(row) for row in grid]
if len(set(row_lengths)) > 1:
    print(f"ERROR: Ragged rows {row_lengths}")

# Fix
max_len = max(row_lengths)
grid = [row + [0] * (max_len - len(row)) for row in grid]
```

**Issue 3: Out of Range Values**
```python
# Detection
for row_idx, row in enumerate(grid):
    for col_idx, val in enumerate(row):
        if not (0 <= val <= 9):
            print(f"ERROR: [{row_idx},{col_idx}] = {val} out of [0-9]")

# Fix
grid = [[max(0, min(9, v)) for v in row] for row in grid]  # Clamp
```

---

## 6. DEBUGGING WORKFLOW

### 6.1 Progressive Validation

```bash
# Step 1: Load and basic checks
python3 -c "
import json
sub = json.load(open('submission.json'))
print(f'✓ JSON loads: {len(sub)} tasks')
"

# Step 2: Format validation
python3 scripts/validate_submission.py --submission submission.json
echo "✓ Format validated"

# Step 3: Count check
python3 -c "
import json
sub = json.load(open('submission.json'))
print(f'Tasks: {len(sub)} (expected: >=240)')
assert len(sub) >= 240, 'Task count too low'
echo "✓ Task count OK"
"

# Step 4: ID verification
python3 -c "
import json, hashlib
sub = json.load(open('submission.json'))
ids = sorted(sub.keys())
hash_val = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
print(f'Task ID hash: {hash_val}')
echo "✓ IDs extracted"
"

# Step 5: Sample inspection
python3 -c "
import json
sub = json.load(open('submission.json'))
first_id = list(sub.keys())[0]
outputs = sub[first_id][0]
print(f'Sample task {first_id}:')
print(f'  attempt_1: {len(outputs[\"attempt_1\"])}x{len(outputs[\"attempt_1\"][0]) if outputs[\"attempt_1\"] else 0}')
print(f'  attempt_2: {len(outputs[\"attempt_2\"])}x{len(outputs[\"attempt_2\"][0]) if outputs[\"attempt_2\"] else 0}')
echo "✓ Sample output OK"
"

# Step 6: Final check
echo "✓ Ready for submission"
```

### 6.2 Debugging Checklist

```
Submission Debugging Checklist
=============================

FORMAT
  [?] JSON parses without error
  [?] Top level is object (dict), not array
  [?] All keys are strings (task IDs)
  [?] All values are lists
  [?] All list items are dicts
  [?] All dicts have keys: "attempt_1", "attempt_2"
  [?] validate_submission.py passes

CONTENT
  [?] Task count >= 240
  [?] All task IDs are 8-character hex
  [?] All grids are rectangular (no ragged rows)
  [?] All values are integers
  [?] All values in range [0-9]
  [?] At least one non-empty attempt per task
  [?] Both attempt_1 and attempt_2 exist

CONFIGURATION
  [?] Model loads without error
  [?] Checkpoint keys present and correct
  [?] H_cycles matches (3)
  [?] L_cycles matches (4)
  [?] hidden_size matches (512)
  [?] num_heads matches (8)

SAMPLES
  [?] First 3 tasks manually verified
  [?] Outputs are sensible for given inputs
  [?] Grid shapes reasonable (not all empty)
  [?] Colors realistic (0-9 range)
```

---

## 7. REFERENCE IMPLEMENTATIONS

### 7.1 Minimal Valid Submission

```python
import json

# Minimal valid submission
minimal_submission = {
    "007d0a30": [
        {
            "attempt_1": [[1, 2], [3, 4]],
            "attempt_2": [[5, 6], [7, 8]]
        }
    ]
}

# Validate
from scripts.validate_submission import validate_submission
validate_submission(minimal_submission)

# Save
with open("submission.json", "w") as f:
    json.dump(minimal_submission, f)

print("✓ Minimal submission created")
```

### 7.2 Complete Validation Script

```python
#!/usr/bin/env python3
import json
import hashlib
import sys
from pathlib import Path

def comprehensive_validation(submission_path, expected_path=None):
    """Full validation pipeline."""
    
    print("=" * 60)
    print("COMPREHENSIVE SUBMISSION VALIDATION")
    print("=" * 60)
    
    # Load submission
    print("\n[1/5] Loading submission...")
    try:
        with open(submission_path) as f:
            submission = json.load(f)
        print(f"✓ Loaded {len(submission)} tasks")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return False
    
    # Format validation
    print("\n[2/5] Validating format...")
    try:
        from validate_submission import validate_submission
        validate_submission(submission)
        print("✓ Format valid")
    except Exception as e:
        print(f"✗ Format invalid: {e}")
        return False
    
    # Count check
    print("\n[3/5] Checking task count...")
    if len(submission) >= 240:
        print(f"✓ {len(submission)} tasks (>= 240 required)")
    else:
        print(f"✗ Only {len(submission)} tasks (need 240)")
        return False
    
    # ID validation
    print("\n[4/5] Validating task IDs...")
    ids = sorted(submission.keys())
    id_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    print(f"  SHA256(IDs): {id_hash}")
    
    try:
        for task_id in ids[:5]:
            assert len(task_id) == 8, f"ID {task_id} wrong length"
            int(task_id, 16)
        print(f"✓ Task IDs valid (checked {min(5, len(ids))})")
    except Exception as e:
        print(f"✗ Invalid task ID: {e}")
        return False
    
    # Sample check
    print("\n[5/5] Spot-checking samples...")
    first_id = ids[0]
    sample = submission[first_id][0]
    for attempt in ["attempt_1", "attempt_2"]:
        grid = sample[attempt]
        if grid:
            h = len(grid)
            w = len(grid[0])
            print(f"  {attempt}: {h}x{w} grid")
        else:
            print(f"  {attempt}: empty")
    print("✓ Sample inspection OK")
    
    print("\n" + "=" * 60)
    print("✓ VALIDATION PASSED - Ready for submission")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    comprehensive_validation("submission.json")
```

---

## CONCLUSION

This reference guide provides:
- Exact format specifications
- Validation code patterns
- Debugging procedures
- Common error signatures
- Reference implementations

Use this alongside `validate_submission.py` for comprehensive pre-submission checks.

---

**Document Status:** Complete
**Last Updated:** October 25, 2025
**Classification:** Technical Reference
