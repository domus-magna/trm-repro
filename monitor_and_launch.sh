#!/bin/bash
set -euo pipefail

echo "$(date): Starting HPC verification monitor..."

# Function to check if HPC verification pods are still running
check_hpc_pods() {
    # Check if any HPC verification pods are still running
    RUNNING_PODS=$(kubectl get pods -n cw-hpc-verification -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' 2>/dev/null | wc -w)
    return $RUNNING_PODS
}

# Function to check if we should wait longer (HPC verification runs in cycles)
should_wait_longer() {
    # HPC verification runs in cycles: 45-50 min active, then 1.5-2 hour gaps
    # We need to distinguish between "gap between cycles" vs "truly complete"

    # Check if there are any completed HPC verification pods recently
    RECENT_COMPLETED=$(kubectl get pods -n cw-hpc-verification -o jsonpath='{.items[?(@.status.phase=="Succeeded")].status.startTime}' 2>/dev/null | wc -w)

    if [ "$RECENT_COMPLETED" -gt 0 ]; then
        echo "$(date): Found recently completed HPC verification pods - this might be truly complete"
        return 1  # Don't wait longer, launch jobs
    else
        echo "$(date): No recently completed HPC verification pods - likely just a gap between cycles"
        return 0  # Wait longer
    fi
}

# Function to launch training jobs when HPC verification completes
launch_training_jobs() {
    echo "$(date): HPC verification pods completed! Launching training jobs..."

    # Delete any existing failed jobs first
    echo "Cleaning up any existing failed jobs..."
    kubectl delete job trm-train-arc2-4gpu --ignore-not-found
    kubectl delete job trm-train-arc2-12gpu-master --ignore-not-found
    kubectl delete job trm-train-arc2-12gpu-worker --ignore-not-found

    # Launch 4-GPU training job
    echo "Launching 4-GPU training job..."
    kubectl apply -f /Users/alexanderhuth/trm-repro/.k8s-trm-train-4gpu.yaml

    # Launch 12-GPU training jobs (8+4 across two nodes)
    echo "Launching 12-GPU training jobs..."
    kubectl apply -f /Users/alexanderhuth/trm-repro/.k8s-trm-12gpu-svc.yaml
    kubectl apply -f /Users/alexanderhuth/trm-repro/.k8s-trm-12gpu-master.yaml
    kubectl apply -f /Users/alexanderhuth/trm-repro/.k8s-trm-12gpu-worker.yaml

    echo "$(date): All training jobs launched successfully!"
    echo "Monitor with: kubectl get pods -l job-name=trm-train-arc2-4gpu"
    echo "Monitor with: kubectl get pods -l app=trm-12gpu-agent"
}

# Main monitoring loop
while true; do
    echo "$(date): Checking HPC verification status..."

    if check_hpc_pods; then
        echo "$(date): HPC verification still running, waiting 30 seconds..."
        sleep 30
    else
        echo "$(date): No HPC verification pods running..."

        if ! should_wait_longer; then
            echo "$(date): HPC verification appears to be truly complete - launching training jobs!"
            launch_training_jobs
            break
        else
            echo "$(date): Likely just a gap between HPC verification cycles, waiting 60 seconds..."
            sleep 60
        fi
    fi
done

echo "$(date): Monitor completed."
