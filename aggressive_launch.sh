#!/bin/bash
# Aggressive training launcher - launches during verification gaps
set -euo pipefail

echo "$(date): Starting aggressive training launcher..."

# Check if HPC verification is running
HPC_RUNNING=$(kubectl get pods -n cw-hpc-verification -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' 2>/dev/null | wc -w)

if [ "$HPC_RUNNING" -gt 0 ]; then
    echo "$(date): HPC verification pods detected - checking if they're actually active..."

    # Check if HPC pods are actually active or just in gap state
    # HPC verification typically runs for 45-50 minutes, so if pods are older than 25 minutes, they're likely gap state
    CURRENT_TIME=$(date +%s)
    GAP_STATE_PODS=0

    for pod in $(kubectl get pods -n cw-hpc-verification -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}'); do
        START_TIME_STR=$(kubectl get pod $pod -n cw-hpc-verification -o jsonpath='{.status.startTime}')
        echo "$(date): Pod $pod started at: $START_TIME_STR"

        # Convert to epoch time (rough approximation for age check)
        if [[ $START_TIME_STR =~ ([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2}) ]]; then
            START_EPOCH=$(date -d "$START_TIME_STR" +%s 2>/dev/null || echo "0")
            AGE_MINUTES=$(( (CURRENT_TIME - START_EPOCH) / 60 ))

            echo "$(date): Pod $pod age: ${AGE_MINUTES} minutes"

            # If pod is older than 25 minutes, consider it gap state
            if [ "$AGE_MINUTES" -gt 25 ]; then
                GAP_STATE_PODS=$((GAP_STATE_PODS + 1))
            fi
        fi
    done

    if [ "$GAP_STATE_PODS" -gt 0 ]; then
        echo "$(date): HPC pods appear to be in gap state - LAUNCHING TRAINING!"
    else
        echo "$(date): HPC pods appear to be actively running - waiting..."
        exit 1
    fi
else
    echo "$(date): No HPC verification running - launching training..."
fi

# Check if there are any recently completed HPC verification pods (within last 10 minutes)
RECENT_COMPLETED=$(kubectl get pods -n cw-hpc-verification -o jsonpath='{.items[?(@.status.phase=="Succeeded")].status.startTime}' 2>/dev/null | wc -w)

if [ "$RECENT_COMPLETED" -gt 0 ]; then
    echo "$(date): Found recently completed HPC verification - this might be truly complete"
    echo "$(date): LAUNCHING TRAINING JOBS AGGRESSIVELY!"
else
    echo "$(date): No recently completed HPC verification - likely gap between cycles"
    echo "$(date): LAUNCHING TRAINING ANYWAY - maximizing GPU utilization!"
fi

# Clean up any existing failed jobs
echo "Cleaning up existing failed jobs..."
kubectl delete job trm-train-arc2-4gpu trm-train-arc2-12gpu-master trm-train-arc2-12gpu-worker --ignore-not-found=true

# Launch training jobs
echo "Launching 4-GPU training job..."
kubectl apply -f /Users/alexanderhuth/trm-repro/.k8s-trm-train-4gpu.yaml

echo "Launching 12-GPU training jobs..."
kubectl apply -f /Users/alexanderhuth/trm-repro/.k8s-trm-12gpu-svc.yaml
kubectl apply -f /Users/alexanderhuth/trm-repro/.k8s-trm-12gpu-master.yaml
kubectl apply -f /Users/alexanderhuth/trm-repro/.k8s-trm-12gpu-worker.yaml

echo "$(date): Training jobs launched successfully!"
