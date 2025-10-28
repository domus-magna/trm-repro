#!/usr/bin/env bash
set -euo pipefail

# Usage: infra/scripts/run_eval_job.sh <namespace>
NS=${1:-trm}

echo "[run-eval] Applying dataset builder with solutions (if needed)" >&2
kubectl apply -n "$NS" -f infra/kubernetes/trm-dataset-script-solutions-cm.yaml
kubectl apply -n "$NS" -f infra/kubernetes/trm-dataset-build-rwx-solutions.yaml
echo "[run-eval] Waiting for dataset job to complete..." >&2
kubectl wait -n "$NS" --for=condition=complete --timeout=30m job/trm-dataset-build-rwx-solutions || true
echo "[run-eval] Applying single-GPU eval job" >&2
kubectl apply -n "$NS" -f infra/kubernetes/trm-eval-1gpu.yaml
echo "[run-eval] Streaming logs (Ctrl-C to stop)" >&2
kubectl logs -n "$NS" -f job/trm-eval-arc2-1gpu-diagnostic || true

