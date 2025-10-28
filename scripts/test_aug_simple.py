#!/usr/bin/env python3
"""
Simplified augmentation inverse test - standalone without dependencies.
"""

import numpy as np
import hashlib


# Copied from dataset/common.py
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """8 dihedral symmetries by rotate, flip and mirror"""

    if tid == 0:
        return arr  # identity
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)  # horizontal reflection
    elif tid == 5:
        return np.flipud(arr)  # vertical reflection
    elif tid == 6:
        return np.fliplr(np.rot90(arr, k=3))  # diagonal reflection
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))  # anti-diagonal reflection
    else:
        return arr


def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    return dihedral_transform(arr, DIHEDRAL_INVERSE[tid])


# Copied from dataset/build_arc_dataset.py
PuzzleIdSeparator = "|"


def grid_hash(grid: np.ndarray):
    assert grid.ndim == 2
    assert grid.dtype == np.uint8

    buffer = [x.to_bytes(1, byteorder='big') for x in grid.shape]
    buffer.append(grid.tobytes())

    return hashlib.sha256(b"".join(buffer)).hexdigest()


def aug(name: str):
    # Augment plan
    trans_id = np.random.randint(0, 8)
    mapping = np.concatenate([np.arange(0, 1, dtype=np.uint8), np.random.permutation(np.arange(1, 10, dtype=np.uint8))])  # Permute colors, Excluding "0" (black)

    name_with_aug_repr = f"{name}{PuzzleIdSeparator}t{trans_id}{PuzzleIdSeparator}{''.join(str(x) for x in mapping)}"

    def _map_grid(grid: np.ndarray):
        return dihedral_transform(mapping[grid], trans_id)

    return name_with_aug_repr, _map_grid


def inverse_aug(name: str):
    # Inverse the "aug" function
    if PuzzleIdSeparator not in name:
        return name, lambda x: x

    trans_id, perm = name.split(PuzzleIdSeparator)[-2:]
    trans_id = int(trans_id[1:])  # Remove "t" letter
    inv_perm = np.argsort(list(perm)).astype(np.uint8)

    def _map_grid(grid: np.ndarray):
        return inv_perm[inverse_dihedral_transform(grid, trans_id)]

    return name.split(PuzzleIdSeparator)[0], _map_grid


# Test code
def main():
    print("=" * 70)
    print("AUGMENTATION INVERSE TESTS (Simplified)")
    print("=" * 70)
    print()

    # Test 1: Dihedral transforms
    print("Test 1: Dihedral transforms and inverses")
    test_grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.uint8)

    all_pass = True
    for tid in range(8):
        transformed = dihedral_transform(test_grid, tid)
        restored = inverse_dihedral_transform(transformed, tid)

        if np.array_equal(restored, test_grid):
            print(f"  ✓ Transform {tid} OK")
        else:
            print(f"  ✗ Transform {tid} FAILED")
            all_pass = False

    print()

    # Test 2: Aug/inverse_aug round trip
    print("Test 2: Aug/inverse_aug round trip")

    test_grids = [
        np.array([[1, 2], [3, 4]], dtype=np.uint8),
        np.arange(10, dtype=np.uint8).reshape(2, 5),
        np.random.randint(0, 10, size=(8, 8), dtype=np.uint8),
    ]

    for grid_idx, test_grid in enumerate(test_grids):
        for trial in range(3):
            puzzle_name = f"puzzle_{grid_idx}_{trial}"

            # Aug and inverse
            aug_name, aug_fn = aug(puzzle_name)
            augmented = aug_fn(test_grid)

            restored_name, inv_fn = inverse_aug(aug_name)
            restored = inv_fn(augmented)

            if np.array_equal(restored, test_grid) and restored_name == puzzle_name:
                if trial == 0:
                    print(f"  ✓ Grid {grid_idx} OK")
            else:
                print(f"  ✗ Grid {grid_idx} trial {trial} FAILED")
                print(f"    Aug name: {aug_name}")
                if not np.array_equal(restored, test_grid):
                    print(f"    Grid mismatch: {np.sum(test_grid != restored)} cells differ")
                if restored_name != puzzle_name:
                    print(f"    Name mismatch: {restored_name} != {puzzle_name}")
                all_pass = False

    print()

    # Test 3: Color permutation preserves color 0
    print("Test 3: Color 0 (black) is always preserved")

    for trial in range(5):
        mapping = np.concatenate([np.arange(0, 1, dtype=np.uint8), np.random.permutation(np.arange(1, 10, dtype=np.uint8))])

        if mapping[0] == 0:
            if trial == 0:
                print(f"  ✓ Color 0 preserved in mapping")
        else:
            print(f"  ✗ Color 0 NOT preserved: mapping[0] = {mapping[0]}")
            all_pass = False
            break

    print()

    # Test 4: Hash consistency
    print("Test 4: Hash consistency")

    grid1 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    hash1a = grid_hash(grid1)
    hash1b = grid_hash(grid1)

    grid2 = np.array([[1, 2], [3, 5]], dtype=np.uint8)
    hash2 = grid_hash(grid2)

    if hash1a == hash1b:
        print(f"  ✓ Hash is deterministic")
    else:
        print(f"  ✗ Hash is NOT deterministic")
        all_pass = False

    if hash1a != hash2:
        print(f"  ✓ Different grids have different hashes")
    else:
        print(f"  ✗ Different grids have SAME hash")
        all_pass = False

    print()
    print("=" * 70)

    if all_pass:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
