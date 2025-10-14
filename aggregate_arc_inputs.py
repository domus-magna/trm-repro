#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def aggregate_dir(src_dir: Path) -> dict:
    agg = {}
    # Accept nested files (ConceptARC has subdirs); walk recursively
    for p in src_dir.rglob('*.json'):
        try:
            with p.open('r') as f:
                data = json.load(f)
            # Use filename stem as puzzle id (e.g., 0934a4d8, AboveBelow1)
            agg[p.stem] = data
        except Exception as e:
            raise RuntimeError(f"Failed reading {p}: {e}")
    if not agg:
        raise RuntimeError(f"No JSON files found under {src_dir}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trm-root', type=Path, required=True, help='Path to TinyRecursiveModels repo root')
    ap.add_argument('--prefix', default='kaggle/combined/arc-agi', help='Input prefix relative to TRM root')
    args = ap.parse_args()

    root = args.trm_root.resolve()
    base = root / args.prefix

    subsets = {
        'training2': base / 'training2',
        'evaluation2': base / 'evaluation2',
        'concept': base / 'concept',
    }

    outputs = {}
    for name, src in subsets.items():
        if not src.exists():
            raise RuntimeError(f"Missing input dir: {src}")
        outputs[name] = aggregate_dir(src)

    # Write aggregated challenges files at the exact paths TRM expects
    for name, payload in outputs.items():
        out_path = root / f"kaggle/combined/arc-agi_{name}_challenges.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w') as f:
            json.dump(payload, f)
        print(f"Wrote {out_path} ({len(payload)} puzzles)")


if __name__ == '__main__':
    main()

