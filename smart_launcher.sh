#!/usr/bin/env bash
set -euo pipefail

echo "$(date): Starting smart GPU availability check..."

nodes=(gd90782 gd956ec)
free=0

for n in "${nodes[@]}"; do
    echo "$(date): Checking node $n..."

    # Get capacity and allocatable GPUs
    cap=$(kubectl get node "$n" -o json | jq -r '.status.capacity["nvidia.com/gpu"] // "0"')
    alloc=$(kubectl get node "$n" -o json | jq -r '.status.allocatable["nvidia.com/gpu"] // "0"')

    echo "$(date): Node $n - Capacity: $cap, Allocatable: $alloc"

    # Fall back to capacity if allocatable missing but device plugin up
    if [[ ! "$alloc" =~ ^[0-9]+$ ]] || [ "$alloc" -eq 0 ]; then
        alloc=$cap
        echo "$(date): Using capacity as allocatable for node $n"
    fi

    # Count "free" as allocatable (this is what scheduler sees)
    free=$(( free + alloc ))
done

echo "$(date): Total free GPUs across all nodes: $free"

# Check if we have enough GPUs for our jobs
if (( free >= 12 )); then
    echo "$(date): OK: >=12 GPUs allocatable, can launch 12-GPU job"
    # Uncomment when ready to launch:
    # kubectl apply -f /Users/alexanderhuth/trm-repro/.k8s-trm-12gpu-svc.yaml
    # kubectl apply -f /Users/alexanderhuth/trm-repro/.k8s-trm-12gpu-master.yaml
    # kubectl apply -f /Users/alexanderhuth/trm-repro/.k8s-trm-12gpu-worker.yaml
elif (( free >= 4 )); then
    echo "$(date): OK: >=4 GPUs allocatable, can launch 4-GPU job"
    # Uncomment when ready to launch:
    # kubectl apply -f /Users/alexanderhuth/trm-repro/.k8s-trm-train-4gpu.yaml
else
    echo "$(date): BLOCKED: Only $free GPUs allocatable (need >=4 for 4-GPU job or >=12 for 12-GPU job)"
    exit 1
fi

echo "$(date): Smart launcher check completed successfully!"
