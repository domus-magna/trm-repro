import json, os

os.chdir("/workspace/TinyRecursiveModels")

challenges_path = "kaggle/combined/arc-agi_evaluation2_challenges.json"
sol_path = "kaggle/combined/arc-agi_evaluation2_solutions.json"

with open(challenges_path) as f:
    challenges = json.load(f)

with open(sol_path) as f:
    solutions = json.load(f)

print(f"Challenges: {len(challenges)} puzzles")
print(f"Solutions: {len(solutions)} puzzles")
print(f"Key overlap: {len(set(challenges.keys()) & set(solutions.keys()))}")

print("\n=== Mismatch Analysis ===")
mismatches = []
for k in list(challenges.keys())[:20]:
    n_challenge_tests = len(challenges[k]["test"])
    n_solution_grids = len(solutions[k])
    if n_challenge_tests != n_solution_grids:
        mismatches.append((k, n_challenge_tests, n_solution_grids))

print(f"Found {len(mismatches)} mismatches in first 20 puzzles:")
for m in mismatches:
    print(f"  {m[0]}: {m[1]} test cases, {m[2]} solutions")

print("\n=== Solution Structure Sample ===")
first_id = list(challenges.keys())[0]
print(f"Puzzle {first_id}:")
print(f"  Challenge test count: {len(challenges[first_id]['test'])}")
print(f"  Solutions type: {type(solutions[first_id])}")
print(f"  Solutions length: {len(solutions[first_id])}")
print(f"  First solution type: {type(solutions[first_id][0])}")
