#!/usr/bin/env python3
"""
Remap TRM puzzle embeddings between identifier orderings.

Use this to convert a legacy checkpoint (shuffled identifiers) into one that
matches the sorted identifier mapping employed by Kaggle inference.

Example:
    python scripts/remap_puzzle_embeddings.py \
        --input-ckpt artifacts/checkpoints/kaggle_dataset_8gpu_step115815/model.ckpt \
        --output-ckpt artifacts/checkpoints/kaggle_dataset_8gpu_step115815_sorted/model.ckpt \
        --permutation artifacts/diagnostics/identifier_mappings/identifier_permutation.json \
        --direction legacy_to_sorted
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch


PUZZLE_KEYS = [
    "_orig_mod.model.inner.puzzle_emb.weights",
    "model.inner.puzzle_emb.weights",
]


def load_mapping(path: Path, direction: str) -> List[Optional[int]]:
    payload = json.loads(path.read_text())
    if direction == "legacy_to_sorted":
        key = "legacy_to_sorted"
    else:
        key = "sorted_to_legacy"
    if key not in payload:
        raise KeyError(f"{key} not found in permutation file {path}")
    mapping = payload[key]
    if not isinstance(mapping, list):
        raise TypeError(f"{key} must be a list in permutation file {path}")
    return mapping


def remap_tensor(
    tensor: torch.Tensor,
    mapping: List[Optional[int]],
    direction: str,
) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D embedding weights, got shape {tuple(tensor.shape)}")
    if len(mapping) > tensor.shape[0]:
        raise ValueError(
            f"Permutation length {len(mapping)} exceeds embedding rows {tensor.shape[0]}"
        )
    if len(mapping) < tensor.shape[0]:
        # Allow partial permutations: pad the trailing entries with None so they remain unchanged.
        padding = tensor.shape[0] - len(mapping)
        mapping = list(mapping) + [None] * padding
    result = tensor.clone()
    if direction == "legacy_to_sorted":
        result = torch.zeros_like(tensor)
        for legacy_idx, sorted_idx in enumerate(mapping):
            if sorted_idx is None:
                result[legacy_idx] = tensor[legacy_idx]
            else:
                result[sorted_idx] = tensor[legacy_idx]
    else:  # sorted_to_legacy
        result = torch.zeros_like(tensor)
        for sorted_idx, legacy_idx in enumerate(mapping):
            if legacy_idx is None:
                result[sorted_idx] = tensor[sorted_idx]
            else:
                result[legacy_idx] = tensor[sorted_idx]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ckpt", required=True, type=Path, help="Path to source checkpoint")
    parser.add_argument("--output-ckpt", required=True, type=Path, help="Path to write remapped checkpoint")
    parser.add_argument(
        "--permutation",
        required=True,
        type=Path,
        help="JSON file containing legacy_to_sorted / sorted_to_legacy arrays",
    )
    parser.add_argument(
        "--direction",
        choices=("legacy_to_sorted", "sorted_to_legacy"),
        default="legacy_to_sorted",
        help="Whether to convert legacy checkpoints to sorted order (default) or the reverse",
    )
    args = parser.parse_args()

    state = torch.load(args.input_ckpt, map_location="cpu")

    mapping = load_mapping(args.permutation, args.direction)

    updated = False
    for key in PUZZLE_KEYS:
        if key in state:
            print(f"Remapping embedding tensor: {key}")
            state[key] = remap_tensor(state[key], mapping, args.direction)
            updated = True
    if not updated:
        raise KeyError(f"None of the expected puzzle embedding keys found in {args.input_ckpt}")

    out_path = args.output_ckpt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_path)
    print(f"Saved remapped checkpoint to {out_path}")


if __name__ == "__main__":
    main()
