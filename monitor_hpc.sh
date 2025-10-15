#!/bin/bash
# Simple script to check HPC verification status
echo "=== HPC Verification Status ==="
kubectl get pods -n cw-hpc-verification -o wide
echo ""
echo "=== GPU Node Status ==="
kubectl get nodes -l gpu.nvidia.com/model=H200 -o wide
echo ""
echo "=== Available GPU Resources ==="
kubectl describe nodes gd90782 | grep -A 5 "Allocated resources" | tail -6
kubectl describe nodes gd956ec | grep -A 5 "Allocated resources" | tail -6
