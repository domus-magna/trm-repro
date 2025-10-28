#!/usr/bin/env python3
"""
Fetch and display key metrics for active TRM runs on Weights & Biases.

Examples
--------
python scripts/monitor_wandb_runs.py --entity arc-prize --project trm-arc2 --run-name trm_arc2_8gpu_resume_step115815_plus100k_v2
python scripts/monitor_wandb_runs.py --entity arc-prize --project trm-arc2 --limit 5

Requirements
------------
- WANDB_API_KEY must be set (or `wandb login` performed on this machine).
- wandb>=0.16 is already listed in the training images; install locally if missing.
"""
from __future__ import annotations

import argparse
import datetime as dt
from typing import Iterable, Sequence

import wandb


SUMMARY_FIELDS: Sequence[str] = (
    "train/lm_loss",
    "train/exact_accuracy",
    "train/steps",
    "all/accuracy",
    "all/lm_loss",
    "ARC/pass@1",
    "ARC/pass@2",
    "ARC/pass@5",
)


def format_metric(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int,)):
        return str(value)
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def iter_runs(path: str, run_name: str | None, limit: int | None) -> Iterable[wandb.apis.public.Run]:
    api = wandb.Api()
    filters = {}
    if run_name:
        # Case-sensitive exact match; use regex if glob-like
        if any(ch in run_name for ch in "*?"):
            regex = run_name.replace("*", ".*").replace("?", ".")
            filters["config.run_name"] = {"$regex": f"^{regex}$"}
        else:
            filters["config.run_name"] = run_name
    try:
        runs = api.runs(path=path, filters=filters, order="-created_at")
    except ValueError as exc:
        raise SystemExit(f"W&B project not found: {path!r}") from exc
    count = 0
    for run in runs:
        yield run
        count += 1
        if limit is not None and count >= limit:
            break


def describe_run(run: wandb.apis.public.Run) -> None:
    cfg_obj = run.config or {}
    if hasattr(cfg_obj, "items"):
        cfg = dict(cfg_obj.items())
    elif isinstance(cfg_obj, dict):
        cfg = dict(cfg_obj)
    else:
        cfg = {}
    run_name = cfg.get("run_name", run.name)
    ckpt = cfg.get("load_checkpoint")
    started = run.created_at or dt.datetime.min
    if hasattr(started, "isoformat"):
        started_str = f"{started.isoformat()}Z"
    else:
        started_str = str(started)
    print(f"\nRun: {run_name}")
    print(f"  id/state     : {run.id} / {run.state}")
    print(f"  started      : {started_str}")
    if ckpt:
        print(f"  load_checkpoint: {ckpt}")
    summary_obj = run.summary or {}
    summary = {}
    if summary_obj:
        if isinstance(summary_obj, dict):
            summary = dict(summary_obj)
        elif hasattr(summary_obj, "items"):
            try:
                summary = dict(summary_obj.items())
            except Exception:
                summary = {}
    if summary:
        print("  metrics:")
        for key in SUMMARY_FIELDS:
            print(f"    {key:<16} {format_metric(summary.get(key))}")
    else:
        print("  metrics: (summary not available yet)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", help="Weights & Biases entity/username (required)")
    parser.add_argument("--project", required=True, help="Weights & Biases project name (e.g., trm-arc2)")
    parser.add_argument("--run-name", dest="run_name", help="Filter by exact run_name (wildcards * and ? supported)")
    parser.add_argument("--limit", type=int, help="Number of runs to display (default: all matching)")
    args = parser.parse_args()

    if not args.entity:
        raise SystemExit("--entity is required (set to the W&B team or username)")

    project_path = f"{args.entity}/{args.project}"
    print(f"Querying W&B project {project_path!r}")
    if args.run_name:
        print(f"Filtering run_name={args.run_name!r}")
    if args.limit:
        print(f"Limiting to {args.limit} runs")

    found = False
    for run in iter_runs(project_path, args.run_name, args.limit):
        describe_run(run)
        found = True
    if not found:
        print("No runs matched the given filters.")


if __name__ == "__main__":
    main()
