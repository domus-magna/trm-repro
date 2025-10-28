#!/usr/bin/env python3
"""
Verify that identifier mappings are consistent between training and evaluation datasets.

This checks:
1. Identifier map order is deterministic
2. Puzzle names map to consistent indices
3. Evaluation dataset identifiers match training identifiers
"""

import json
import os
import sys

def load_identifier_map(data_path):
    """Load identifier map from a dataset."""
    identifiers_file = os.path.join(data_path, "identifiers.json")

    if not os.path.exists(identifiers_file):
        return None

    with open(identifiers_file, "r") as f:
        identifier_map = json.load(f)

    return identifier_map


def analyze_identifier_map(identifier_map, name="Dataset"):
    """Analyze and print statistics about an identifier map."""
    print(f"\n{name} Identifier Map Analysis:")
    print(f"  Total identifiers: {len(identifier_map)}")

    # Check for special tokens
    blank_count = sum(1 for id in identifier_map if id == "<blank>" or "blank" in id.lower())
    print(f"  Blank identifiers: {blank_count}")

    # Show first 10 identifiers
    print(f"  First 10 identifiers:")
    for idx in range(min(10, len(identifier_map))):
        print(f"    [{idx}] {identifier_map[idx]}")

    # Show last 5 identifiers
    if len(identifier_map) > 10:
        print(f"  Last 5 identifiers:")
        for idx in range(max(10, len(identifier_map) - 5), len(identifier_map)):
            print(f"    [{idx}] {identifier_map[idx]}")

    return identifier_map


def compare_identifier_maps(map1, map2, name1="Map 1", name2="Map 2"):
    """Compare two identifier maps for consistency."""
    print(f"\nComparing {name1} vs {name2}:")

    if map1 is None or map2 is None:
        print("  ⚠ One or both maps are None, cannot compare")
        return False

    # Check lengths
    if len(map1) != len(map2):
        print(f"  ✗ Different lengths: {len(map1)} vs {len(map2)}")
        return False
    else:
        print(f"  ✓ Same length: {len(map1)}")

    # Check order
    mismatches = []
    for idx in range(len(map1)):
        if map1[idx] != map2[idx]:
            mismatches.append((idx, map1[idx], map2[idx]))

    if mismatches:
        print(f"  ✗ Found {len(mismatches)} mismatches:")
        for idx, id1, id2 in mismatches[:10]:  # Show first 10
            print(f"    [{idx}] {id1} != {id2}")
        if len(mismatches) > 10:
            print(f"    ... and {len(mismatches) - 10} more")
        return False
    else:
        print(f"  ✓ All identifiers match in order")
        return True


def find_dataset_paths():
    """Find all dataset paths in the repo."""
    # Common locations for datasets
    search_paths = [
        "data",
        "artifacts",
        "kaggle/datasets",
        "TinyRecursiveModels/data",
    ]

    dataset_paths = []
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue

        # Walk through directory
        for root, dirs, files in os.walk(search_path):
            if "identifiers.json" in files:
                dataset_paths.append(root)

    return dataset_paths


def main():
    print("=" * 80)
    print("IDENTIFIER MAPPING VERIFICATION")
    print("=" * 80)

    # Find all datasets
    dataset_paths = find_dataset_paths()

    if not dataset_paths:
        print("\n⚠ No datasets with identifiers.json found")
        print("Searched in: data, artifacts, kaggle/datasets, TinyRecursiveModels/data")
        return 1

    print(f"\nFound {len(dataset_paths)} datasets with identifier maps:")
    for idx, path in enumerate(dataset_paths):
        print(f"  {idx+1}. {path}")

    # Load all identifier maps
    identifier_maps = {}
    for path in dataset_paths:
        identifier_map = load_identifier_map(path)
        if identifier_map:
            dataset_name = os.path.basename(path) or path
            identifier_maps[dataset_name] = identifier_map
            analyze_identifier_map(identifier_map, dataset_name)

    # Compare pairs
    if len(identifier_maps) > 1:
        print("\n" + "=" * 80)
        print("CROSS-DATASET COMPARISON")
        print("=" * 80)

        names = list(identifier_maps.keys())
        all_match = True

        for i in range(len(names) - 1):
            for j in range(i + 1, len(names)):
                match = compare_identifier_maps(
                    identifier_maps[names[i]],
                    identifier_maps[names[j]],
                    names[i],
                    names[j]
                )
                if not match:
                    all_match = False

        print("\n" + "=" * 80)
        if all_match:
            print("✓ ALL IDENTIFIER MAPS ARE CONSISTENT")
            return 0
        else:
            print("✗ IDENTIFIER MAP MISMATCHES DETECTED")
            print("\nThis could cause evaluation failures because:")
            print("- Puzzle embeddings would map to wrong puzzles")
            print("- Model predictions would be for different puzzles than expected")
            return 1
    else:
        print("\nOnly one dataset found, cannot compare")
        return 0


if __name__ == "__main__":
    sys.exit(main())
