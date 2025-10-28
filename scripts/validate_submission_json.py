#!/usr/bin/env python3
"""
Lightweight validator for ARC Prize 2025 submission.json produced by TRM kernels.

Checks:
- JSON structure matches {puzzle_id: [ {attempt_1: grid, attempt_2: grid, ...}, ... ]}
- All grids are 2D, shapes <= 30x30, values in [0,9]
- Reports diversity: fraction of test examples with attempt_1 != attempt_2 (for K>=2)
- Reports basic counts: puzzles, total test examples, attempts per example

Usage:
  python3 scripts/validate_submission_json.py --path submission.json --k 2 --expected-puzzles 240

Exit code 0 on pass; non-zero on validation failure.
"""
import argparse
import json
import sys
from typing import Any, Dict, List


def is_valid_grid(g: Any) -> bool:
    if not isinstance(g, list) or not g:
        return False
    nrows = len(g)
    if nrows < 1 or nrows > 30:
        return False
    ncols = None
    for row in g:
        if not isinstance(row, list) or not row:
            return False
        if ncols is None:
            ncols = len(row)
            if ncols < 1 or ncols > 30:
                return False
        elif len(row) != ncols:
            return False
        for v in row:
            if not isinstance(v, int) or v < 0 or v > 9:
                return False
    return True


def grids_equal(a: List[List[int]], b: List[List[int]]) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for va, vb in zip(ra, rb):
            if va != vb:
                return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to submission.json")
    ap.add_argument("--k", type=int, default=2, help="Number of attempts per example (expected)")
    ap.add_argument("--expected-puzzles", type=int, default=240, help="Expected number of puzzles in test split")
    args = ap.parse_args()

    data: Dict[str, Any] = json.load(open(args.path))
    if not isinstance(data, dict):
        print("ERROR: submission root must be an object mapping puzzle_id -> list of test entries", file=sys.stderr)
        return 2

    n_puzzles = len(data)
    total_examples = 0
    bad_grids = 0
    bad_format = 0
    attempts_hist: Dict[int, int] = {}
    dup_examples = 0

    for pid, examples in data.items():
        if not isinstance(examples, list) or len(examples) < 1:
            print(f"ERROR: puzzle {pid} has no test entries", file=sys.stderr)
            bad_format += 1
            continue
        for ex in examples:
            if not isinstance(ex, dict):
                bad_format += 1
                continue
            # Count attempts present
            attempts = [k for k in ex.keys() if k.startswith("attempt_")]
            attempts_hist[len(attempts)] = attempts_hist.get(len(attempts), 0) + 1
            # Validate each grid
            grids: List[List[List[int]]] = []
            for i in range(1, args.k + 1):
                key = f"attempt_{i}"
                if key not in ex:
                    print(f"WARN: puzzle {pid} example missing {key}")
                    continue
                grid = ex[key]
                if not is_valid_grid(grid):
                    bad_grids += 1
                grids.append(grid)
            # Duplicate detection for K>=2
            if len(grids) >= 2 and grids_equal(grids[0], grids[1]):
                dup_examples += 1
            total_examples += 1

    # Summary
    print("=== submission.json validation ===")
    print(f"puzzles: {n_puzzles}")
    print(f"total test examples: {total_examples}")
    print(f"attempts histogram (per example): {attempts_hist}")
    print(f"invalid grids: {bad_grids}")
    print(f"format errors: {bad_format}")
    if total_examples > 0:
        print(f"duplicate attempt_1==attempt_2: {dup_examples} ({dup_examples/total_examples*100:.1f}%)")

    # Hard checks
    rc = 0
    if n_puzzles != args.expected_puzzles:
        print(f"ERROR: expected {args.expected_puzzles} puzzles, found {n_puzzles}", file=sys.stderr)
        rc = 3
    if bad_grids > 0 or bad_format > 0:
        rc = 4
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

