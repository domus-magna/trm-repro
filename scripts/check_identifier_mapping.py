#!/usr/bin/env python3
"""
Quick sanity check for ARC identifier ordering.

Usage:
    python scripts/check_identifier_mapping.py --dataset artifacts/diagnostics/identifier_mappings/sorted --expect sorted
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "sorted": "f3fe1a1f0b27b36fd53166ac17faf980e6c7ff9e73ee16d884095a6c860637a5",
    "legacy": "c364837393c2428e40c6116692fb1b66bf011108ec9930475df306cd779bbfd1",
}


def load_sha(dataset: Path) -> str:
    identifiers = dataset / "identifiers.json"
    if not identifiers.exists():
        raise FileNotFoundError(f"{identifiers} not found")
    return hashlib.sha256(identifiers.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path containing identifiers.json")
    parser.add_argument("--expect", choices=EXPECTED.keys(), help="Expected mapping name (optional)")
    args = parser.parse_args()

    sha = load_sha(Path(args.dataset))
    print(f"identifiers.json sha256: {sha}")
    if args.expect:
        expected_sha = EXPECTED[args.expect]
        if sha != expected_sha:
            raise SystemExit(f"Hash mismatch: expected {expected_sha}, got {sha}")
        print("Identifier mapping matches expected schema.")


if __name__ == "__main__":
    main()
