#!/usr/bin/env python3
"""
Unit tests for augmentation and inverse augmentation functions.
Tests that aug() and inverse_aug() are proper inverses of each other.
"""

import sys
import os
import numpy as np

# Add TinyRecursiveModels to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TinyRecursiveModels'))

from dataset.build_arc_dataset import aug, inverse_aug, grid_hash, PuzzleIdSeparator
from dataset.common import dihedral_transform, inverse_dihedral_transform, DIHEDRAL_INVERSE


def test_dihedral_inverse():
    """Test that dihedral transforms and their inverses are correct."""
    print("Testing dihedral transforms...")

    # Create a test grid with distinct values
    test_grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.uint8)

    errors = []
    for tid in range(8):
        # Apply transform then inverse
        transformed = dihedral_transform(test_grid, tid)
        restored = inverse_dihedral_transform(transformed, tid)

        if not np.array_equal(restored, test_grid):
            errors.append(f"Transform {tid}: inverse did not restore original")
            print(f"  ✗ Transform {tid} FAILED")
            print(f"    Original:\n{test_grid}")
            print(f"    Transformed:\n{transformed}")
            print(f"    Restored:\n{restored}")
        else:
            print(f"  ✓ Transform {tid} OK")

    if errors:
        print(f"\n❌ Dihedral transform test FAILED: {len(errors)} errors")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("✓ All dihedral transforms passed!\n")
        return True


def test_color_permutation_inverse():
    """Test that color permutation inverse works correctly."""
    print("Testing color permutation inverse...")

    # Create a grid with all colors 0-9
    test_grid = np.arange(10, dtype=np.uint8).reshape(2, 5)

    errors = []
    for trial in range(20):
        # Generate random permutation (preserving color 0)
        mapping = np.concatenate([
            np.arange(0, 1, dtype=np.uint8),  # Color 0 unchanged
            np.random.permutation(np.arange(1, 10, dtype=np.uint8))
        ])

        # Apply permutation
        permuted = mapping[test_grid]

        # Compute inverse permutation
        inv_perm = np.argsort(mapping).astype(np.uint8)

        # Apply inverse
        restored = inv_perm[permuted]

        if not np.array_equal(restored, test_grid):
            errors.append(f"Trial {trial}: inverse permutation failed")
            print(f"  ✗ Trial {trial} FAILED")
            print(f"    Mapping: {mapping}")
            print(f"    Inverse: {inv_perm}")
            print(f"    Original: {test_grid.flatten()}")
            print(f"    Permuted: {permuted.flatten()}")
            print(f"    Restored: {restored.flatten()}")
        else:
            if trial < 3:
                print(f"  ✓ Trial {trial} OK (mapping={mapping.tolist()})")

    if errors:
        print(f"\n❌ Color permutation test FAILED: {len(errors)} errors")
        return False
    else:
        print("✓ All color permutation trials passed!\n")
        return True


def test_aug_inverse_round_trip():
    """Test that aug() followed by inverse_aug() restores the original grid."""
    print("Testing aug/inverse_aug round trip...")

    # Create various test grids
    test_grids = [
        # Simple grid
        np.array([[1, 2], [3, 4]], dtype=np.uint8),

        # Grid with all colors
        np.arange(10, dtype=np.uint8).reshape(2, 5),

        # Larger grid
        np.random.randint(0, 10, size=(10, 10), dtype=np.uint8),

        # Grid with only black (color 0)
        np.zeros((5, 5), dtype=np.uint8),

        # Grid with repeated colors
        np.array([[1, 1, 2, 2], [3, 3, 4, 4], [5, 5, 6, 6]], dtype=np.uint8),
    ]

    errors = []
    for grid_idx, test_grid in enumerate(test_grids):
        print(f"  Testing grid {grid_idx+1}/{len(test_grids)} (shape={test_grid.shape})...")

        # Run multiple trials with different random augmentations
        for trial in range(5):
            puzzle_name = f"test_puzzle_{grid_idx}_{trial}"

            # Apply augmentation
            aug_name, aug_fn = aug(puzzle_name)
            augmented_grid = aug_fn(test_grid)

            # Extract augmentation parameters from name
            parts = aug_name.split(PuzzleIdSeparator)
            if len(parts) != 3:
                errors.append(f"Grid {grid_idx} trial {trial}: invalid augmented name format")
                continue

            orig_name_check, trans_id_str, perm_str = parts
            if orig_name_check != puzzle_name:
                errors.append(f"Grid {grid_idx} trial {trial}: original name mismatch")

            # Apply inverse augmentation
            restored_name, inv_fn = inverse_aug(aug_name)
            restored_grid = inv_fn(augmented_grid)

            # Verify restoration
            if restored_name != puzzle_name:
                errors.append(f"Grid {grid_idx} trial {trial}: restored name '{restored_name}' != '{puzzle_name}'")

            if not np.array_equal(restored_grid, test_grid):
                errors.append(f"Grid {grid_idx} trial {trial}: grid not restored")
                print(f"    ✗ Trial {trial} FAILED")
                print(f"      Aug name: {aug_name}")
                print(f"      Original hash: {grid_hash(test_grid)[:16]}...")
                print(f"      Augmented hash: {grid_hash(augmented_grid)[:16]}...")
                print(f"      Restored hash: {grid_hash(restored_grid)[:16]}...")
                print(f"      Original:\n{test_grid}")
                print(f"      Restored:\n{restored_grid}")
                print(f"      Diff (!=0):\n{test_grid != restored_grid}")
            else:
                if trial == 0:
                    print(f"    ✓ Trial {trial} OK (aug_name={aug_name})")

    if errors:
        print(f"\n❌ Aug/inverse_aug round trip test FAILED: {len(errors)} errors")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("✓ All aug/inverse_aug round trips passed!\n")
        return True


def test_identity_inverse():
    """Test that inverse_aug works correctly on non-augmented names."""
    print("Testing inverse_aug on non-augmented names...")

    test_grid = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    puzzle_name = "test_puzzle_no_aug"

    restored_name, inv_fn = inverse_aug(puzzle_name)
    restored_grid = inv_fn(test_grid)

    if restored_name != puzzle_name:
        print(f"  ✗ Name mismatch: '{restored_name}' != '{puzzle_name}'")
        return False

    if not np.array_equal(restored_grid, test_grid):
        print(f"  ✗ Grid was modified by identity inverse")
        return False

    print("  ✓ Identity inverse works correctly\n")
    return True


def test_hash_consistency():
    """Test that grid hashes are consistent and unique for different grids."""
    print("Testing grid hash consistency...")

    test_grid = np.array([[1, 2], [3, 4]], dtype=np.uint8)

    # Hash should be deterministic
    hash1 = grid_hash(test_grid)
    hash2 = grid_hash(test_grid)

    if hash1 != hash2:
        print(f"  ✗ Hash not deterministic: {hash1} != {hash2}")
        return False

    # Different grids should have different hashes
    different_grid = np.array([[1, 2], [3, 5]], dtype=np.uint8)
    hash3 = grid_hash(different_grid)

    if hash1 == hash3:
        print(f"  ✗ Different grids have same hash")
        return False

    # Same content but different shape should have different hash
    reshaped_grid = np.array([[1, 2, 3, 4]], dtype=np.uint8)
    hash4 = grid_hash(reshaped_grid)

    if hash1 == hash4:
        print(f"  ✗ Different shapes have same hash")
        return False

    print("  ✓ Hash consistency checks passed\n")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("AUGMENTATION INVERSE UNIT TESTS")
    print("=" * 70)
    print()

    results = []
    results.append(("Dihedral inverse", test_dihedral_inverse()))
    results.append(("Color permutation inverse", test_color_permutation_inverse()))
    results.append(("Identity inverse", test_identity_inverse()))
    results.append(("Hash consistency", test_hash_consistency()))
    results.append(("Aug/inverse_aug round trip", test_aug_inverse_round_trip()))

    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
