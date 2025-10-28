#!/usr/bin/env python3
"""
Validate TRM checkpoint package for Kaggle upload.

This script performs comprehensive validation of a packaged checkpoint
to ensure it meets all Kaggle requirements and is ready for upload.

Usage:
    python3 scripts/validate_checkpoint_for_kaggle.py \
        --package-dir artifacts/checkpoints/kaggle_dataset_8gpu_step_72385
"""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


IDENTIFIER_HASHES = {
    "legacy": "c364837393c2428e40c6116692fb1b66bf011108ec9930475df306cd779bbfd1",
    "sorted": "f3fe1a1f0b27b36fd53166ac17faf980e6c7ff9e73ee16d884095a6c860637a5",
}


def validate_metadata(metadata_path: Path) -> Dict:
    """Validate dataset-metadata.json against Kaggle requirements."""
    print("\n📋 Validating Kaggle Metadata")
    print("=" * 60)

    if not metadata_path.exists():
        raise ValidationError(f"Metadata file not found: {metadata_path}")

    try:
        with open(metadata_path) as f:
            meta = json.load(f)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in metadata: {e}")

    # Check required fields
    required_fields = ['title', 'id', 'licenses']
    missing = [f for f in required_fields if f not in meta]
    if missing:
        raise ValidationError(f"Missing required fields: {missing}")

    # Validate title length (6-50 chars)
    title = meta['title']
    title_len = len(title)
    if not (6 <= title_len <= 50):
        raise ValidationError(f"Title length {title_len} invalid (must be 6-50 chars)")
    print(f"✅ Title: '{title}' ({title_len} chars)")

    # Validate ID format (username/slug)
    dataset_id = meta['id']
    if '/' not in dataset_id:
        raise ValidationError(f"ID format invalid: '{dataset_id}' (must be username/slug)")
    username, slug = dataset_id.split('/', 1)
    print(f"✅ ID: '{dataset_id}' (username: {username}, slug: {slug})")

    # Validate licenses
    licenses = meta['licenses']
    if not licenses or not isinstance(licenses, list) or len(licenses) == 0:
        raise ValidationError("At least one license required")
    license_name = licenses[0].get('name', '')
    print(f"✅ License: {license_name}")

    return meta


def validate_checkpoint(checkpoint_path: Path) -> Tuple[int, int]:
    """Validate model.ckpt file integrity."""
    print("\n🔍 Validating Checkpoint File")
    print("=" * 60)

    if not checkpoint_path.exists():
        raise ValidationError(f"Checkpoint file not found: {checkpoint_path}")

    # Check file size
    size_bytes = checkpoint_path.stat().st_size
    size_gb = size_bytes / (1024 ** 3)
    print(f"File size: {size_gb:.2f} GB ({size_bytes:,} bytes)")

    # Warn if over 20GB (Kaggle limit)
    if size_gb > 20:
        raise ValidationError(f"Checkpoint exceeds 20GB Kaggle limit: {size_gb:.2f} GB")
    elif size_gb > 15:
        print(f"⚠️  WARNING: Checkpoint is {size_gb:.2f} GB (close to 20GB limit)")

    # Validate ZIP structure
    try:
        with zipfile.ZipFile(checkpoint_path, 'r') as zf:
            print("✅ Valid ZIP archive")

            num_files = len(zf.namelist())
            print(f"✅ Contains {num_files} files")

            # Test for corruption
            bad_file = zf.testzip()
            if bad_file is not None:
                raise ValidationError(f"Corrupted file in archive: {bad_file}")
            print("✅ No file corruption detected")

            # Check for PyTorch checkpoint structure
            has_data_pkl = any('data.pkl' in name for name in zf.namelist())
            has_version = any('version' in name for name in zf.namelist())

            if has_data_pkl and has_version:
                print("✅ Valid PyTorch checkpoint structure")
            else:
                print("⚠️  WARNING: Unexpected checkpoint structure")
                if not has_data_pkl:
                    print("   - Missing data.pkl")
                if not has_version:
                    print("   - Missing version file")

    except zipfile.BadZipFile:
        raise ValidationError(f"Checkpoint is not a valid ZIP file: {checkpoint_path}")

    return size_bytes, num_files


def validate_package_files(package_dir: Path) -> Dict[str, Path]:
    """Validate all required package files exist."""
    print("\n📦 Validating Package Files")
    print("=" * 60)

    required_files = {
        'model.ckpt': 'PyTorch checkpoint',
        'dataset-metadata.json': 'Kaggle metadata',
        'README.md': 'Documentation',
        'COMMANDS.txt': 'Training invocation',
        'ENVIRONMENT.txt': 'Model configuration',
        'TRM_COMMIT.txt': 'Git provenance'
    }

    found_files = {}
    missing_files = []

    for filename, description in required_files.items():
        filepath = package_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / (1024**3):.2f} GB"
            elif size > 1024:
                size_str = f"{size / 1024:.2f} KB"
            else:
                size_str = f"{size} bytes"

            print(f"✅ {filename:25} {size_str:>12} - {description}")
            found_files[filename] = filepath
        else:
            print(f"❌ {filename:25} MISSING - {description}")
            missing_files.append(filename)

    if missing_files:
        raise ValidationError(f"Missing required files: {', '.join(missing_files)}")

    return found_files


def validate_identifier_mapping(dataset_dir: Path, mode: str) -> None:
    """Ensure identifiers.json matches the expected identifier mapping."""
    print("\n🧭 Validating Identifier Mapping")
    print("=" * 60)
    identifiers_path = dataset_dir / "identifiers.json"
    if not identifiers_path.exists():
        raise ValidationError(f"identifiers.json not found in {dataset_dir}")
    sha = hashlib.sha256(identifiers_path.read_bytes()).hexdigest()
    print(f"Hash: {sha}")
    expected = IDENTIFIER_HASHES[mode]
    if sha != expected:
        raise ValidationError(
            "Dataset identifier mapping does not match the expected schema.\n"
            f"Expected {expected}, got {sha}. "
            "Rebuild with the correct dataset builder or choose a different identifier mode."
        )
    print(f"✅ Identifier mapping matches the '{mode}' schema.")


def validate_provenance(files: Dict[str, Path]):
    """Validate provenance files contain expected content."""
    print("\n📜 Validating Provenance")
    print("=" * 60)

    # Check TRM_COMMIT.txt
    commit_path = files['TRM_COMMIT.txt']
    commit_hash = commit_path.read_text().strip()
    if len(commit_hash) != 40:
        raise ValidationError(f"Invalid git commit hash: {commit_hash}")
    print(f"✅ Git commit: {commit_hash}")

    # Check COMMANDS.txt
    commands_path = files['COMMANDS.txt']
    commands = commands_path.read_text()
    if 'pretrain.py' not in commands:
        raise ValidationError("COMMANDS.txt doesn't contain pretrain.py invocation")
    if 'arch=trm' not in commands:
        raise ValidationError("COMMANDS.txt doesn't contain arch=trm")
    print("✅ Training commands present")

    # Check ENVIRONMENT.txt
    env_path = files['ENVIRONMENT.txt']
    env_config = env_path.read_text()
    required_keys = ['arch:', 'H_cycles:', 'L_cycles:', 'hidden_size:']
    missing_keys = [k for k in required_keys if k not in env_config]
    if missing_keys:
        raise ValidationError(f"ENVIRONMENT.txt missing keys: {missing_keys}")
    print("✅ Environment configuration complete")

    # Check README.md
    readme_path = files['README.md']
    readme = readme_path.read_text()
    if len(readme) < 100:
        print("⚠️  WARNING: README.md seems short (< 100 chars)")
    else:
        print("✅ README.md present")


def calculate_totals(package_dir: Path) -> Tuple[int, int]:
    """Calculate total size and file count."""
    total_size = 0
    file_count = 0

    for item in package_dir.iterdir():
        if item.is_file():
            total_size += item.stat().st_size
            file_count += 1

    return total_size, file_count


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--package-dir',
        type=Path,
        required=True,
        help='Path to packaged checkpoint directory'
    )
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        help='Optional ARC dataset directory to verify identifier mapping'
    )
    parser.add_argument(
        '--identifier-mode',
        choices=('legacy', 'sorted'),
        default='sorted',
        help="Expected identifier ordering when --dataset-dir is provided (default: sorted)",
    )

    args = parser.parse_args()
    package_dir = args.package_dir
    dataset_dir = args.dataset_dir

    if not package_dir.exists():
        print(f"❌ Package directory not found: {package_dir}")
        sys.exit(1)

    if not package_dir.is_dir():
        print(f"❌ Path is not a directory: {package_dir}")
        sys.exit(1)

    print("=" * 60)
    print("🔬 TRM Checkpoint Validation for Kaggle Upload")
    print("=" * 60)
    print(f"Package: {package_dir}")

    try:
        # Validate package files
        files = validate_package_files(package_dir)

        if dataset_dir:
            if not dataset_dir.exists():
                raise ValidationError(f"Dataset directory not found: {dataset_dir}")
            validate_identifier_mapping(dataset_dir, args.identifier_mode)

        # Validate metadata
        metadata = validate_metadata(files['dataset-metadata.json'])

        # Validate checkpoint
        ckpt_size, ckpt_files = validate_checkpoint(files['model.ckpt'])

        # Validate provenance
        validate_provenance(files)

        # Calculate totals
        total_size, total_files = calculate_totals(package_dir)

        # Summary
        print("\n" + "=" * 60)
        print("✅ VALIDATION PASSED")
        print("=" * 60)
        print(f"Dataset ID:     {metadata['id']}")
        print(f"Title:          {metadata['title']}")
        print(f"Total size:     {total_size / (1024**3):.2f} GB")
        print(f"Total files:    {total_files}")
        print(f"Checkpoint:     {ckpt_size / (1024**3):.2f} GB ({ckpt_files} internal files)")
        print()
        print("✅ Package is ready for Kaggle upload")
        print()
        print("Next steps:")
        print("  1. Upload: kaggle datasets version -p", package_dir, "-m 'version message'")
        print("  2. Verify: Check dataset page on Kaggle")
        print("  3. Test: Attach to inference notebook and run")

        return 0

    except ValidationError as e:
        print("\n" + "=" * 60)
        print(f"❌ VALIDATION FAILED: {e}")
        print("=" * 60)
        return 1
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ UNEXPECTED ERROR: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())
