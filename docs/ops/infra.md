# Infra‑Only Migration (Docker Hub, CoreWeave, Backblaze, Checkpoints, W&B)

This guide covers only the infrastructure pieces you asked to retain while rebuilding code: Docker Hub image management, CoreWeave access basics, Backblaze (B2) setup, checkpoint preservation practices, and Weights & Biases connectivity.

---

## Docker Hub

- Repository:
  - Create or reuse `docker.io/<namespace>/<repo>` for your training image.
  - Prefer pinning deployments by digest; use tags for human‑readable pointers.
- CI (GitHub Actions):
  - Use `.github/workflows/build-dockerhub.yml` with repo secrets `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`.
  - Tags policy: `latest` on default branch, `branch-<sha>` for traceability.
- Local build/push (optional):
  - `docker login -u "$DOCKERHUB_USERNAME" -p "$DOCKERHUB_TOKEN"`
  - `docker buildx build -f Dockerfile.dockerhub -t <ns>/<repo>:latest .`
  - `docker push <ns>/<repo>:latest`
- Verify:
  - `docker run --rm <ns>/<repo>:latest python -c "import torch; print('ok')"`
  - `docker manifest inspect <ns>/<repo>:latest`

---

## CoreWeave (Kubernetes)

- Access:
  - Ensure your kubeconfig is set (this repo includes `.kube-config`; export if needed):
    - `export KUBECONFIG="$PWD/.kube-config"`
  - Verify cluster:
    - `kubectl get nodes --show-labels | rg -i 'gpu|product|h200'`
    - Typical GPU label: `nvidia.com/gpu.product=NVIDIA-H200-SXM5-141GB`
- Observability (no new Jobs defined here):
  - Pods: `kubectl get pods -A | rg -i 'trm|prod'`
  - Logs: `kubectl logs -f <pod-name>`
  - GPU metrics snapshot via `nvidia-smi` (read‑only):
    - `kubectl exec <pod-name> -- nvidia-smi --query-gpu=index,gpu_name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits`
- Secrets (recommended):
  - W&B token: `kubectl create secret generic wandb-api-key --from-literal=token='<YOUR_WANDB_API_KEY>'`
  - B2/rclone: store `rclone.conf` as a Secret or mount from your home dir in helper pods.

---

## Backblaze (B2) via rclone

- Create a B2 bucket (example): `trm-arc2-checkpoints`.
- Configure rclone (one‑time):
  - `rclone config` → `n` → name `b2` → provider Backblaze B2 → enter `keyID` and `applicationKey`.
  - Test: `rclone lsd b2:`
- Folder structure (example):
  - `b2:trm-arc2-checkpoints/<run_name>/step_XXXXX`
  - Optionally include evaluator outputs and a copy of `all_config.yaml`.
- Upload (local → B2):
  - `rclone copy checkpoints/<run_name> b2:trm-arc2-checkpoints/<run_name> --update --checksum --retries 3 --transfers 4`
- Verify:
  - `rclone size b2:trm-arc2-checkpoints/<run_name>`
  - Spot‑check a single file: `rclone md5sum b2:trm-arc2-checkpoints/<run_name> | head`
- Optional lifecycle:
  - Configure bucket rules for object versions/retention using Backblaze console (keep N latest steps). 

---

## Checkpoint Preservation (Process)

- Naming:
  - Use stable run names and monotonic `step_<N>` files. Keep `all_config.yaml` alongside checkpoints for reproducibility.
- Where:
  - On cluster: use a mounted PVC path, e.g., `/workspace/checkpoints/<run_name>`.
  - Locally: mirror under `checkpoints/<run_name>`.
- Backup loop (manual or cron):
  - Discover latest remote step: `kubectl exec <pod> -- ls -1 /workspace/checkpoints/<run_name> | grep '^step_' | sort -V | tail -1`
  - Copy from pod: `kubectl cp <pod>:/workspace/checkpoints/<run_name>/<file> checkpoints/<run_name>/<file>`
  - Sync to B2: `rclone copy checkpoints/<run_name> b2:trm-arc2-checkpoints/<run_name> --update --checksum`
- Integrity:
  - Check non‑zero sizes after `kubectl cp`.
  - Optionally compute a local `sha256sum` and log alongside the filename.

---

## Weights & Biases (W&B) Connection

- Default posture:
  - Keep disabled unless explicitly needed on cluster (Kaggle is offline).
- Env vars:
  - `WANDB_API_KEY`: secret token.
  - `WANDB_PROJECT`: e.g., `trm-arc2`.
  - `WANDB_ENTITY`: your team/user (optional).
  - `WANDB_MODE`: `online` (to enable) or omit; image defaults can set disabled.
  - `WANDB_DISABLED`: set to empty to enable; set to `true` to disable.
- Secret (CoreWeave):
  - `kubectl create secret generic wandb-api-key --from-literal=token='<YOUR_WANDB_API_KEY>'`
- Run hygiene:
  - Add clear `run_name` and (optionally) `group`/`job_type` for comparisons.
  - Do not log raw puzzle contents if privacy is a concern.

---

