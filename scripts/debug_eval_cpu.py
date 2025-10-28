#!/usr/bin/env python3
"""
CPU-only diagnostic runner for the Tiny Recursive Model ARC checkpoint.

It loads a baked dataset with ARC evaluation labels, runs the TRM forward pass,
and reports how often predicted grids match the ground truth as well as halt probabilities.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "TinyRecursiveModels"))

# ---------------------------------------------------------------------------
# Compatibility shims
# ---------------------------------------------------------------------------
import torch.nn as nn

if not hasattr(nn, "Buffer"):
    class _CompatBuffer(nn.Parameter):  # type: ignore[name-defined]
        def __new__(cls, data, persistent=True):  # type: ignore[override]
            param = nn.Parameter(data, requires_grad=False)
            param._persistent = persistent  # type: ignore[attr-defined]
            return param

    nn.Buffer = _CompatBuffer  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Environment guards
# ---------------------------------------------------------------------------
os.environ.setdefault("DISABLE_COMPILE", "1")

try:
    import adam_atan2  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    import torch.optim as optim

    adam_stub = types.ModuleType("adam_atan2")

    class AdamATan2(optim.Adam):  # type: ignore[name-defined]
        """Torch Adam substitute for diagnostics."""

    adam_stub.AdamATan2 = AdamATan2  # type: ignore[attr-defined]
    sys.modules["adam_atan2"] = adam_stub

# Project imports (after path tweaks)
from dataset.build_arc_dataset import grid_hash, inverse_aug  # type: ignore  # noqa: E402
from models.losses import ACTLossHead  # type: ignore  # noqa: E402
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1  # type: ignore  # noqa: E402
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig  # type: ignore  # noqa: E402


ARC_MAX_GRID = 30


def crop_grid(flat_grid: np.ndarray) -> np.ndarray:
    """Replica of evaluator _crop without numba dependency."""
    grid = flat_grid.reshape(ARC_MAX_GRID, ARC_MAX_GRID)
    max_area = 0
    max_rows, max_cols = 0, 0
    num_c = ARC_MAX_GRID
    for num_r in range(1, ARC_MAX_GRID + 1):
        for c in range(1, num_c + 1):
            val = grid[num_r - 1, c - 1]
            if (val < 2) or (val > 11):
                num_c = c - 1
                break
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_rows, max_cols = num_r, num_c
    cropped = grid[:max_rows, :max_cols]
    return (cropped - 2).astype(np.uint8)


@dataclass
class ExampleStats:
    puzzle_id: str
    test_index: int
    hash_pred: str
    hash_label: str
    hash_input: str
    match_grid: bool
    halt_prob: float
    frac_diff_vs_input: float
    frac_diff_vs_label: float


def build_model(num_identifiers: int, batch_size: int) -> ACTLossHead:
    cfg = dict(
        batch_size=batch_size,
        seq_len=ARC_MAX_GRID * ARC_MAX_GRID,
        puzzle_emb_ndim=512,
        num_puzzle_identifiers=num_identifiers,
        vocab_size=12,  # PAD + EOS + digits (0-9) shifted by +2
        H_cycles=3,
        L_cycles=4,
        H_layers=0,
        L_layers=2,
        hidden_size=512,
        expansion=4,
        num_heads=8,
        pos_encodings="rope",
        halt_max_steps=16,
        halt_exploration_prob=0.1,
        forward_dtype="float32",
        mlp_t=False,
        puzzle_emb_len=16,
        no_ACT_continue=True,
    )
    inner = TinyRecursiveReasoningModel_ACTV1(cfg)
    model = ACTLossHead(inner, loss_type="stablemax_cross_entropy")
    return model


def strip_prefix(state: Dict[str, torch.Tensor], prefix: str = "_orig_mod.") -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state):
        return state
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}


def run(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()

    identifier_map: List[str] = json.loads((dataset_path / "identifiers.json").read_text())
    test_puzzles: Dict[str, Dict] = json.loads((dataset_path / "test_puzzles.json").read_text())

    puzzle_counts: Dict[str, int] = defaultdict(int)

    dataset_cfg = PuzzleDatasetConfig(
        seed=0,
        dataset_paths=[str(dataset_path)],
        global_batch_size=args.batch_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    eval_dataset = PuzzleDataset(dataset_cfg, split="test")
    eval_loader = DataLoader(eval_dataset, batch_size=None)

    # Align puzzle embedding size with checkpoint
    ckpt_state = torch.load(checkpoint_path, map_location="cpu")
    puzzle_key = next(k for k in ckpt_state if "puzzle_emb.weights" in k)
    num_identifiers = ckpt_state[puzzle_key].shape[0]

    model = build_model(num_identifiers=num_identifiers, batch_size=args.batch_size)
    model.eval()

    normalized_state = strip_prefix(ckpt_state)
    missing, unexpected = model.load_state_dict(normalized_state, strict=False)
    if missing:
        print("[WARN] Missing keys:", missing)
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected)

    stats: List[ExampleStats] = []
    max_examples = args.max_examples if args.max_examples is not None else float("inf")

    with torch.inference_mode():
        for set_name, batch, _ in eval_loader:
            batch = {k: v.to(torch.long) for k, v in batch.items()}

            carry = model.initial_carry(batch)  # type: ignore[attr-defined]
            while True:
                carry, _, _, outputs, all_finish = model(
                    carry=carry,
                    batch=batch,
                    return_keys=["preds", "q_halt_logits"],
                )
                if all_finish:
                    break

            preds = outputs["preds"]
            halt_logits = outputs["q_halt_logits"]

            for idx in range(preds.shape[0]):
                if len(stats) >= max_examples:
                    break

                identifier = int(batch["puzzle_identifiers"][idx])
                if identifier == 0:
                    continue  # padded slot

                puzzle_name = identifier_map[identifier]
                orig_name, inverse_fn = inverse_aug(puzzle_name)
                assert orig_name == puzzle_name, "Evaluation split should not include augmented identifiers."

                test_idx = puzzle_counts[puzzle_name]
                puzzle_counts[puzzle_name] += 1

                label_grid = np.array(test_puzzles[puzzle_name]["test"][test_idx]["output"], dtype=np.uint8)
                raw_input_grid = np.array(test_puzzles[puzzle_name]["test"][test_idx]["input"], dtype=np.uint8)

                seq_pred = preds[idx].cpu().numpy()
                seq_input = batch["inputs"][idx].cpu().numpy()

                grid_pred = inverse_fn(crop_grid(seq_pred))
                grid_input = inverse_fn(crop_grid(seq_input))

                if grid_pred.shape == grid_input.shape:
                    diff_vs_input = float(np.mean(grid_pred != grid_input))
                else:
                    diff_vs_input = 1.0
                if grid_pred.shape == label_grid.shape:
                    diff_vs_label = float(np.mean(grid_pred != label_grid))
                else:
                    diff_vs_label = 1.0

                stats.append(
                    ExampleStats(
                        puzzle_id=puzzle_name,
                        test_index=test_idx,
                        hash_pred=grid_hash(grid_pred),
                        hash_label=grid_hash(label_grid),
                        hash_input=grid_hash(grid_input),
                        match_grid=np.array_equal(grid_pred, label_grid),
                        halt_prob=float(torch.sigmoid(halt_logits[idx]).cpu().item()),
                        frac_diff_vs_input=diff_vs_input,
                        frac_diff_vs_label=diff_vs_label,
                    )
                )
            if len(stats) >= max_examples:
                break
        else:
            # Completed loop without hitting max_examples
            pass

    total = len(stats)
    matches = sum(1 for s in stats if s.match_grid)
    matching_puzzles = len({(s.puzzle_id, s.test_index) for s in stats if s.match_grid})
    unique_hashes = len({s.hash_pred for s in stats})
    input_matches = sum(1 for s in stats if s.hash_pred == s.hash_input)

    halt_probs = np.array([s.halt_prob for s in stats], dtype=np.float32)
    diff_vs_input = np.array([s.frac_diff_vs_input for s in stats], dtype=np.float32)
    diff_vs_label = np.array([s.frac_diff_vs_label for s in stats], dtype=np.float32)

    print(f"Total test examples processed: {total}")
    print(f"Grid matches: {matches} ({matches / max(1, total):.2%}) across {matching_puzzles} puzzles")
    print(f"Unique prediction hashes: {unique_hashes}")
    print(f"Predictions identical to inputs: {input_matches} ({input_matches / max(1, total):.2%})")
    print(
        "q_halt probability stats: "
        f"min={halt_probs.min():.4f}, max={halt_probs.max():.4f}, mean={halt_probs.mean():.4f}, std={halt_probs.std():.4f}"
    )
    if total:
        print(
            "Fraction differing cells vs input: "
            f"min={diff_vs_input.min():.4f}, max={diff_vs_input.max():.4f}, "
            f"mean={diff_vs_input.mean():.4f}, std={diff_vs_input.std():.4f}"
        )
        print(
            "Fraction differing cells vs label: "
            f"min={diff_vs_label.min():.4f}, max={diff_vs_label.max():.4f}, "
            f"mean={diff_vs_label.mean():.4f}, std={diff_vs_label.std():.4f}"
        )
    preview = min(5, len(stats))
    for sample in stats[:preview]:
        print(
            f"Sample {sample.puzzle_id}[{sample.test_index}]: halt_prob={sample.halt_prob:.4f}, "
            f"pred_hash={sample.hash_pred[:16]}..., label_hash={sample.hash_label[:16]}..., "
            f"diff_vs_input={sample.frac_diff_vs_input:.4f}, diff_vs_label={sample.frac_diff_vs_label:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect TRM ARC checkpoint predictions on CPU.")
    parser.add_argument("--checkpoint", required=True, help="Path to model.ckpt")
    parser.add_argument("--dataset", required=True, help="Path to built ARC dataset (with labels)")
    parser.add_argument("--batch-size", type=int, default=16, help="Global batch size used during iteration")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of evaluation examples")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
