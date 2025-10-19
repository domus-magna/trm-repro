#!/usr/bin/env bash
set -euo pipefail

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${BACKBLAZE_KEY_ID:?BACKBLAZE_KEY_ID must be set}"
: "${BACKBLAZE_APPLICATION_KEY:?BACKBLAZE_APPLICATION_KEY must be set}"
BACKBLAZE_BUCKET="${BACKBLAZE_BUCKET:-trm-arc2-checkpoints}"

kubectl delete secret trm-backblaze --ignore-not-found >/dev/null 2>&1 || true

kubectl create secret generic trm-backblaze \
  --from-literal=RCLONE_CONFIG_TRMB2_TYPE=b2 \
  --from-literal=RCLONE_CONFIG_TRMB2_ACCOUNT="${BACKBLAZE_KEY_ID}" \
  --from-literal=RCLONE_CONFIG_TRMB2_KEY="${BACKBLAZE_APPLICATION_KEY}" \
  --from-literal=B2_BUCKET="${BACKBLAZE_BUCKET}"

echo "Created/updated trm-backblaze secret (bucket=${BACKBLAZE_BUCKET})"
