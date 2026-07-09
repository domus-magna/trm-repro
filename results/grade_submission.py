#!/usr/bin/env python3
"""Grade a TRM submission.json against the ARC-AGI-2 public evaluation set.

Reports both the metric TRM's evaluator logs to W&B (`ARC/pass@k`, a per-task
macro-average with fractional credit) and the official ARC Prize metric (a
micro-average over test outputs). These are different estimators and produce
different numbers from the same predictions.

Usage:
    git clone --depth 1 https://github.com/arcprize/ARC-AGI-2.git
    python results/grade_submission.py \
        --submission results/submission_step72385.json \
        --eval-dir ARC-AGI-2/data/evaluation
"""

import argparse
import glob
import json
import os


def load_eval(eval_dir):
    tasks = {}
    for path in glob.glob(os.path.join(eval_dir, "*.json")):
        tasks[os.path.basename(path)[:-5]] = json.load(open(path))
    if not tasks:
        raise SystemExit(f"no tasks found in {eval_dir}")
    return tasks


def grade(submission, tasks):
    per_task = {}
    n_outputs = 0
    hits_at_1 = 0
    hits_at_2 = 0

    for task_id, task in tasks.items():
        truths = [pair["output"] for pair in task["test"]]
        preds = submission.get(task_id, [])
        if len(preds) != len(truths):
            raise SystemExit(
                f"{task_id}: {len(preds)} predictions for {len(truths)} test outputs. "
                "Check that the submission was generated against this evaluation set."
            )

        solved_1 = solved_2 = 0
        for i, truth in enumerate(truths):
            attempt_1 = preds[i].get("attempt_1")
            attempt_2 = preds[i].get("attempt_2")
            solved_1 += attempt_1 == truth
            solved_2 += truth in (attempt_1, attempt_2)

        n_outputs += len(truths)
        hits_at_1 += solved_1
        hits_at_2 += solved_2
        per_task[task_id] = (solved_1, solved_2, len(truths))

    n_tasks = len(tasks)
    # TRM's evaluator: mean over tasks of (test outputs solved / test outputs in task).
    trm_pass_1 = sum(s1 / n for s1, _, n in per_task.values()) / n_tasks
    trm_pass_2 = sum(s2 / n for _, s2, n in per_task.values()) / n_tasks
    return per_task, n_outputs, hits_at_1, hits_at_2, trm_pass_1, trm_pass_2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--eval-dir", required=True)
    args = ap.parse_args()

    tasks = load_eval(args.eval_dir)
    submission = json.load(open(args.submission))
    per_task, n_outputs, hits_1, hits_2, trm_1, trm_2 = grade(submission, tasks)

    print(f"{len(tasks)} tasks / {n_outputs} test outputs\n")
    print("TRM evaluator metric (per-task macro-average, fractional credit):")
    print(f"  ARC/pass@1 = {trm_1:.9f}")
    print(f"  ARC/pass@2 = {trm_2:.9f}\n")
    print("Official ARC Prize metric (micro-average over test outputs):")
    print(f"  pass@1 = {hits_1}/{n_outputs} = {100 * hits_1 / n_outputs:.2f}%")
    print(f"  pass@2 = {hits_2}/{n_outputs} = {100 * hits_2 / n_outputs:.2f}%\n")

    credited = {t: v for t, v in per_task.items() if v[1] > 0}
    print(f"Tasks with any credit (pass@2): {len(credited)}")
    for task_id, (s1, s2, n) in sorted(credited.items()):
        full = "fully solved" if s2 == n else f"partial ({s2}/{n})"
        attempt = "attempt_1" if s1 == s2 else "attempt_2 needed"
        print(f"  {task_id}  {s2}/{n} test outputs  {full}, {attempt}")


if __name__ == "__main__":
    main()
