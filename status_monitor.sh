#!/bin/bash
# HPC Verification and Training Status Monitor

echo "=== HPC Verification & Training Status Monitor ==="
echo "$(date): Starting monitoring..."

while true; do
    echo "----------------------------------------"
    echo "$(date): Status Update"
    echo ""

    # Check HPC verification pods
    echo "🔍 HPC Verification Status:"
    kubectl get pods -n cw-hpc-verification --no-headers 2>/dev/null | head -5 || echo "  ❌ Cannot access HPC verification namespace"

    echo ""
    echo "💻 Node GPU Status:"
    kubectl get nodes -l gpu.nvidia.com/model=H200 -o custom-columns=NAME:metadata.name,STATUS:status.conditions[0].status,GPU_AVAIL:status.allocatable.nvidia.com/gpu --no-headers 2>/dev/null || echo "  ❌ Cannot access nodes"

    echo ""
    echo "🚀 Training Jobs:"
    kubectl get jobs --no-headers 2>/dev/null | grep trm-train || echo "  No training jobs found"

    echo ""
    echo "🎯 Active Training Pods:"
    kubectl get pods -l job-name --no-headers 2>/dev/null | grep trm-train || echo "  No training pods found"

    echo ""
    echo "⏱️  Next update in 2 minutes..."
    echo ""

    sleep 120
done
