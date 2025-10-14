#!/usr/bin/env python3
"""Sanitize evaluation2 solutions: deduplicate and truncate to match test counts."""
import json
import sys

CHALL = "kaggle/combined/arc-agi_evaluation2_challenges.json"
SOL_CANDIDATES = [
    "kaggle/combined/arc-agi_evaluation2-solutions.json",
    "kaggle/combined/arc-agi_evaluation2_solutions.json",
    "kaggle/combined/arc-agi_evaluation-solutions.json",
    "kaggle/combined/arc-agi_evaluation_solutions.json"
]
OUT = "kaggle/combined/arc-agi_evaluation2-solutions.sanitized.json"
AUDIT = "kaggle/combined/arc-agi_evaluation2-solutions.sanitized.audit.txt"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# Find solutions file
SOL = None
for cand in SOL_CANDIDATES:
    try:
        with open(cand):
            SOL = cand
            break
    except FileNotFoundError:
        continue

if not SOL:
    print("ERROR: No evaluation2 solutions file found")
    sys.exit(2)

print(f"Using solutions file: {SOL}")

challenges = load_json(CHALL)
solutions = load_json(SOL)

affected = []
fixed = 0
deduped_count = 0

for puzzle_id in challenges.keys():
    if puzzle_id not in solutions:
        print(f"WARNING: {puzzle_id} in challenges but not in solutions")
        continue

    challenge_tests = challenges[puzzle_id]["test"]
    solution_list = solutions[puzzle_id]

    # Deduplicate exact solutions (by JSON string)
    seen = set()
    dedup_list = []
    for sol in solution_list:
        key = json.dumps(sol, sort_keys=True)
        if key in seen:
            deduped_count += 1
            continue
        seen.add(key)
        dedup_list.append(sol)

    solution_list = dedup_list

    n_challenge = len(challenge_tests)
    n_solution = len(solution_list)

    if n_solution > n_challenge:
        # Truncate to match test count
        solution_list = solution_list[:n_challenge]
        affected.append((puzzle_id, n_challenge, n_solution))
        fixed += 1

    # Write back
    solutions[puzzle_id] = solution_list

# Write sanitized solutions
with open(OUT, "w") as f:
    json.dump(solutions, f, ensure_ascii=False, indent=2)

# Write audit log
with open(AUDIT, "w") as f:
    f.write("Sanitized evaluation2 solutions\n")
    f.write(f"De-duplicated entries: {deduped_count}\n")
    f.write(f"Truncated puzzles: {fixed}\n\n")
    for pid, nc, ns in affected:
        f.write(f"{pid}: kept {nc} of {ns} solutions\n")

print(f"\nWrote {OUT}")
print(f"Wrote {AUDIT}\n")
print("Audit:")
with open(AUDIT, "r") as f:
    print(f.read())
