# Infrastructure Assets

This directory groups the operational material that supported TRM ARC-AGI-2 training.

- `kubernetes/` – Manifest files (Deployments, ConfigMaps, CronJobs) used to launch training, evaluation, and maintenance jobs on the cluster. Secrets and environment-specific values have been removed; add your own before applying.
- `scripts/` – Utility scripts for monitoring and job orchestration (e.g., `monitor_4gpu_launch.sh`, `smart_launcher.sh`) plus the `backblaze_restore/` helper used to recover checkpoints from cold storage.

> **Note:** These assets assume a CoreWeave-style Kubernetes environment with access to NVLink-connected GPU nodes. Review each file carefully before reuse and replace credentials or image references with your own.
