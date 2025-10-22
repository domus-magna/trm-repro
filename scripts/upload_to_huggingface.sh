#!/usr/bin/env bash
#
# upload_to_huggingface.sh
#
# Uploads files or directories to the seconds-0/trm-arc2-8gpu Hugging Face repository.
#
# Usage:
#   ./scripts/upload_to_huggingface.sh [OPTIONS]
#
# Options:
#   --full        Upload entire huggingface_release/trm_arc2_8gpu directory
#   --readme      Upload only README.md (default)
#   --commit-msg  Custom commit message (optional)
#
# Requirements:
#   - huggingface_hub CLI installed: pip install huggingface_hub
#   - Authenticated: hf auth login
#

set -euo pipefail

# Configuration
REPO_ID="seconds-0/trm-arc2-8gpu"
RELEASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/huggingface_release/trm_arc2_8gpu"

# Default options
MODE="readme"
COMMIT_MSG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --full)
      MODE="full"
      shift
      ;;
    --readme)
      MODE="readme"
      shift
      ;;
    --commit-msg)
      COMMIT_MSG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--full|--readme] [--commit-msg \"message\"]"
      exit 1
      ;;
  esac
done

# Check authentication
if ! hf auth whoami &> /dev/null; then
  echo "❌ Error: Not authenticated with Hugging Face"
  echo "Run: hf auth login"
  exit 1
fi

echo "🤗 Uploading to Hugging Face repository: $REPO_ID"
echo "📂 Source directory: $RELEASE_DIR"
echo "📝 Mode: $MODE"

cd "$RELEASE_DIR" || exit 1

if [[ "$MODE" == "full" ]]; then
  echo "📤 Uploading entire directory..."
  if [[ -n "$COMMIT_MSG" ]]; then
    hf upload "$REPO_ID" . --exclude ".cache/*" --exclude ".gitignore" --commit-message "$COMMIT_MSG"
  else
    hf upload "$REPO_ID" . --exclude ".cache/*" --exclude ".gitignore"
  fi
else
  echo "📤 Uploading README.md..."
  if [[ -n "$COMMIT_MSG" ]]; then
    hf upload "$REPO_ID" README.md README.md --commit-message "$COMMIT_MSG"
  else
    hf upload "$REPO_ID" README.md README.md
  fi
fi

echo "✅ Upload complete!"
echo "🔗 View at: https://huggingface.co/$REPO_ID"
