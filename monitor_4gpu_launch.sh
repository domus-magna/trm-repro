#!/bin/bash
# Monitor for HPC verification completion and launch 4-GPU job on gd956ec

while true; do
    echo "$(date): Checking HPC verification status..."

    # Check if HPC verification is running on gd956ec (where we want to run 4-GPU)
    HPC_ON_956EC=$(kubectl get pods -n cw-hpc-verification -o jsonpath='{.items[?(@.spec.nodeName=="gd956ec")].metadata.name}' 2>/dev/null | wc -w)

    if [ "$HPC_ON_956EC" -eq 0 ]; then
        echo "$(date): No HPC verification on gd956ec - checking if 4-GPU job is running..."

        # Check if 4-GPU job is already running
        GPU4_RUNNING=$(kubectl get pods -l job-name=trm-train-arc2-4gpu -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' 2>/dev/null | wc -w)

        if [ "$GPU4_RUNNING" -eq 0 ]; then
            echo "$(date): 4-GPU job not running - LAUNCHING 4-GPU TRAINING JOB!"
            kubectl apply -f /Users/alexanderhuth/trm-repro/.k8s-trm-train-4gpu.yaml
            echo "$(date): 4-GPU job launched successfully!"
        else
            echo "$(date): 4-GPU job already running - monitoring continues..."
        fi
    else
        echo "$(date): HPC verification still running on gd956ec - waiting..."
    fi

    sleep 60
done
