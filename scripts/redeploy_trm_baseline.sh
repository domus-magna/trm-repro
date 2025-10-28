#!/usr/bin/env bash
set -euo pipefail

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl not found in PATH; aborting." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEGACY_DIR="${ROOT_DIR}/infra/legacy_revert"

for required in trm-common-script-cm.yaml trm-sitecustomize-cm.yaml trm-train-8gpu.yaml trm-train-4gpu.yaml; do
  if [[ ! -f "${LEGACY_DIR}/${required}" ]]; then
    echo "Missing ${LEGACY_DIR}/${required}; aborting." >&2
    exit 2
  fi
done

NAMESPACE="${TRM_NAMESPACE:-trm}"

echo "[rollback] applying legacy ConfigMaps to namespace ${NAMESPACE}"
kubectl apply -n "${NAMESPACE}" -f "${LEGACY_DIR}/trm-common-script-cm.yaml"
kubectl apply -n "${NAMESPACE}" -f "${LEGACY_DIR}/trm-sitecustomize-cm.yaml"

echo "[rollback] removing existing training jobs (ignore if absent)"
kubectl delete job -n "${NAMESPACE}" trm-train-arc2-8gpu-resume trm-train-arc2-4gpu-resume trm-train-arc2-8gpu trm-train-arc2-4gpu --ignore-not-found

echo "[rollback] launching legacy 8× job"
kubectl apply -n "${NAMESPACE}" -f "${LEGACY_DIR}/trm-train-8gpu.yaml"

echo "[rollback] launching legacy 4× job"
kubectl apply -n "${NAMESPACE}" -f "${LEGACY_DIR}/trm-train-4gpu.yaml"

echo "[rollback] current pods:"
kubectl get pods -n "${NAMESPACE}" | grep trm-train-arc2 || true
