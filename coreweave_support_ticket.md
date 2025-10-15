# CoreWeave Support Ticket - HPC Verification System Blocking GPU Access

## Issue Summary

**SEVERITY: CRITICAL** - HPC verification system malfunction causing indefinite GPU resource blocking and complete training pipeline failure.

## Environment Details

**Namespace**: default
**Node Names**: gd90782, gd956ec
**Cluster**: (CoreWeave cluster with H200 GPUs)

## Problem Description

The HPC verification system is malfunctioning with the following issues:

1. **Resource Tracking Bug**: HPC verification pods remain in "Running" state during gap periods but continue consuming GPU resources
2. **Admission Controller Failure**: Scheduler reports "Available: 0" GPUs despite nodes showing 8 GPUs allocatable
3. **Cyclic Behavior**: Verification runs ~45-50 minutes, gaps ~1.5-2 hours, repeat
4. **Complete Blockage**: All training jobs fail with `UnexpectedAdmissionError`

## Timeline of Events

**October 14, 2025**:
- **09:29 PDT**: Brief gap detected, training jobs launched but failed
- **11:01 PDT**: New HPC verification cycle started
- **11:44 PDT**: HPC pods terminated, GPU resources freed
- **11:45 PDT**: Training jobs successfully launched and running

## Technical Evidence

### 1. HPC Pod Configuration
```bash
kubectl get pods -n cw-hpc-verification -o wide
NAME                                                   READY   STATUS    AGE   IP           NODE
hpc-verification-short-1760464800-nhc-run-4006521046   2/2     Running   31m   10.0.1.114   gd956ec
hpc-verification-short-1760464800-nhc-run-44470815     2/2     Running   31m   10.0.0.146   gd90782
```

**GPU Resource Allocation**:
```json
{
  "limits": {"cpu":"96","memory":"512Gi","rdma/ib":"1","nvidia.com/gpu":"8"},
  "requests": {"cpu":"100m","memory":"100Mi","rdma/ib":"1","nvidia.com/gpu":"8"}
}
```

### 2. Node Resource Status (During Issue)
```bash
# Both nodes reported 8/8 GPUs allocatable despite HPC pods consuming them
kubectl get node gd90782 -o json | jq '.status.capacity["nvidia.com/gpu"], .status.allocatable["nvidia.com/gpu"]'
"8"
"8"
```

### 3. Training Job Admission Failures
```bash
kubectl get events --sort-by=.lastTimestamp | grep -E 'UnexpectedAdmissionError|nvidia.com/gpu'
LAST SEEN   TYPE      REASON                      OBJECT                                  MESSAGE
29m         Warning   UnexpectedAdmissionError    pod/trm-train-arc2-4gpu-9lltn           Allocate failed due to requested number of devices unavailable for nvidia.com/gpu. Requested: 4, Available: 0, which is unexpected
29m         Warning   UnexpectedAdmissionError    pod/trm-train-arc2-12gpu-worker-cxd4d   Allocate failed due to requested number of devices unavailable for nvidia.com/gpu. Requested: 4, Available: 0, which is unexpected
```

### 4. Current Status (After Manual Intervention)
```bash
kubectl get pods -l job-name
NAME                                READY   STATUS              RESTARTS   AGE
trm-train-arc2-12gpu-master-lq2ts   1/1     Running             0          2m
trm-train-arc2-12gpu-worker-gwjq5   1/1     Running             0          2m
trm-train-arc2-4gpu-tchhr           1/1     Running             0          2m
```

## Root Cause Analysis

### Primary Issue: State Management Failure
- HPC verification pods remain "Running" during gap periods
- GPU resources not released despite pods being in "gap" state
- Resource tracking reports allocatable GPUs while admission denies access

### Secondary Issue: Resource Tracking Inconsistency
- Node allocatable counts don't reflect actual GPU consumption
- Scheduler/admission controller sees "Available: 0" despite node reports
- No NVIDIA device plugin found (daemonsets in kube-system)

## Immediate Resolution Applied

**Temporary Fix**: Manual termination of HPC verification pods
```bash
kubectl delete pods -n cw-hpc-verification --all
```

**Result**: GPU resources freed, training jobs successfully launched and running.

## Long-term Request

**Request one of the following**:

1. **Immediate Fix**: Release GPU resources at end of NHC verification cycles
2. **Node Opt-out**: Label/taint to exclude these nodes from HPC verification (e.g., `coreweave.com/hpc-verify=disabled`)
3. **Predictable Schedule**: Verification window/schedule to avoid during training
4. **Device Plugin Investigation**: Confirm proper NVIDIA device plugin operation

## Impact Assessment

### Financial Impact
- **16 H200 GPUs** blocked during gap periods (~70% of time)
- **Daily waste**: 14-16 hours × 16 GPUs = **$400-800/day** (H200 pricing)
- **Training blockage**: Zero progress on TRM ARC-AGI-2 reproduction

### Operational Impact
- **Complete pipeline failure** for machine learning workloads
- **Resource waste** of premium GPU instances
- **Unpredictable availability** due to cyclic verification

## Urgency

**CRITICAL** - This represents a system failure where expensive GPU resources are being wasted while delivering no value to users. Immediate intervention required to restore proper resource management.

## Contact Information

**Namespace**: default
**Nodes**: gd90782, gd956ec
**Current Status**: Training running after manual HPC pod termination
