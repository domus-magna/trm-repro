# Results — Per-Task Breakdown

Predictions and scoring for the run reported in the top-level README: W&B run `ljxzfy3z`
(`trm_arc2_8gpu_eval100`), evaluator checkpoint **step 72,385**, on the ARC-AGI-2 public
evaluation set (120 tasks / 167 test outputs).

- `submission_step72385.json` — the evaluator's `attempt_1` / `attempt_2` grids for all 167 test outputs.
- `grade_submission.py` — scores that file against the evaluation set under both metrics.

Reproduce:

```bash
git clone --depth 1 https://github.com/arcprize/ARC-AGI-2.git
python results/grade_submission.py \
    --submission results/submission_step72385.json \
    --eval-dir ARC-AGI-2/data/evaluation
```

This reproduces `ARC/pass@1 = 0.016666667` and `ARC/pass@2 = 0.029166667` exactly, matching
`huggingface_release/trm_arc2_8gpu/wandb_ljxzfy3z_summary.json`.

## What the model solved

Three tasks solved outright, plus one of two test outputs on a fourth.

| Task | Test outputs solved | Solved by | Output dims | Distinct input colors | Train examples |
| --- | --- | --- | --- | --- | --- |
| `71e489b6` | 2 / 2 | `attempt_1` | 16×16, 18×19 | 2 | 3 |
| `7666fa5d` | 1 / 1 | `attempt_1` | 16×16 | 2 | 2 |
| `1818057f` | 1 / 1 | `attempt_2` only | 22×22 | 2 | 3 |
| `b6f77b65` | 1 / 2 | `attempt_2` only | 12×12 | 10 | 5 |

Every solved test output is a **same-shape** task — output grid dimensions equal input grid
dimensions. All three fully solved tasks have exactly **2 distinct input colors**, against a set
median of 6. Two of the five solved outputs required `attempt_2`.

## What the model failed

| Failure mode | Count |
| --- | --- |
| Reshape tasks (output dims ≠ input dims) solved | **0 / 48** |
| Correct output *shape* on reshape outputs (`attempt_1`) | 6 / 48 (12.5%) |
| Correct output *shape* on same-shape outputs (`attempt_1`) | 98 / 119 (82.4%) |
| Prediction was a verbatim copy of the test input | 37 / 167 |
| Prediction was a single flat color | 29 / 167 |
| Prediction was empty or malformed | 7 / 167 |

**Output-shape inference is the dominant failure.** 48 of 167 test outputs (29%) require the model
to produce a grid of different dimensions than its input. It gets that shape right 12.5% of the
time, so those outputs are lost before any cell is scored. On same-shape outputs it gets the
geometry right 82.4% of the time.

**The model does slightly better than copying its input, but not by much.** On the 98 same-shape
outputs where `attempt_1` had the correct dimensions, mean per-cell accuracy is 88.07% against
84.13% for a baseline that simply echoes the test input — a +3.94pp lift (47 wins / 39 ties /
12 losses; sign test p ≈ 5e-6). The lift is statistically real but small, and this comparison is
generous to the model because it conditions on outputs the model already sized correctly. Per-cell
accuracy on this task family is largely explained by input-echoing and should not be read as
partial understanding.

## Where the headroom is

The run's full `pass@k` curve, from `wandb_ljxzfy3z_summary.json`:

| k | 1 | 2 | 5 | 10 | 100 | 1000 |
| --- | --- | --- | --- | --- | --- | --- |
| `ARC/pass@k` | 1.67% | 2.92% | 5.00% | 5.83% | 8.19% | 13.75% |

`pass@1000 = 13.75%` means the correct grid is present in the model's candidate pool for roughly
one task in seven, but ranked below the top two by augmentation vote count. The gap between
`pass@2` and `pass@1000` is a **candidate-ranking** problem, not a capacity problem. Together with
output-shape inference, that is where the distance to the paper's 7.8% most plausibly lives.

See the top-level README, section 4, for what `pass@k` does and does not mean.
