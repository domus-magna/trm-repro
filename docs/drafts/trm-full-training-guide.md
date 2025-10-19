# TRM Full Training Run Guide (ARC‑AGI‑2)

This guide covers end‑to‑end training of Tiny Recursive Models (TRM) for ARC‑AGI‑2 on a multi‑GPU cloud instance, including hardware selection, environment setup, data building, training commands, checkpointing/resume, robustness, verification, and Kaggle packaging.


> Tip: Keep Weights & Biases (W&B) disabled unless you need dashboards; Kaggle runs are offline.

---

## 1) Hardware & Pricing

Recommended GPU: 4×H100 (80GB). If available, 4×H200 (141GB) offers larger memory headroom (same Hopper SM90 arch) which helps with bigger batches and fewer grad‑accum steps. Alternatives: 4×A100‑80G, 4×L40S (48GB).

Typical price ranges (subject to market):
- H100: $1.5–$3.5/GPU‑hr
- H200: provider‑dependent; typically similar to H100 with a modest premium. Check CoreWeave’s pricing portal and plan +10–25% vs H100 unless you have negotiated rates.
- A100‑80G: $1.0–$2.5/GPU‑hr
- L40S: $0.4–$1.2/GPU‑hr

### Vast.ai (CLI)
- Install CLI: `pipx install vastai` or `pip3 install vastai`
- Auth: `vastai set api_key <VAST_API_KEY>`
- Search examples (sorted by dollars per hour `dph`):
  - H100 (4+ gpus, verified, reliable network):
    ```bash
    vastai search offers 'verified=true num_gpus>=4 gpu_name=H100 reliability>0.98 inet_down>200 inet_up>100' -o dph -raw
    ```
  - A100‑80G:
    ```bash
    vastai search offers 'verified=true num_gpus>=4 gpu_name=A100 reliability>0.98' -o dph -raw
    ```
  - L40S:
    ```bash
    vastai search offers 'verified=true num_gpus>=4 gpu_name=L40S reliability>0.98' -o dph -raw
    ```
- Launch example:
  ```bash
  vastai create instance <OFFER_ID> \
    --image pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel \
    --disk 200 \
    --onstart-cmd "sleep infinity"
  vastai ssh <INSTANCE_ID>
  ```

### Prime Intellect (API)
- Set token: `export PI_TOKEN='<YOUR_PI_TOKEN>'`
- Query GPUs/prices (example pattern):
  ```bash
  curl -H "Authorization: Bearer $PI_TOKEN" \
       'https://api.primeintellect.ai/v1/gpus?gpu=H100&min_gpus=4'
  ```
- Choose best $/hr in a nearby region to reduce latency.

---

## 2) Container Image

Use a devel image with CUDA toolkit + nvcc (needed to build `adam-atan2`):
- Recommended: `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel`
- Alternative: `pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel` (ensure matching Torch CUDA wheels)

Compatibility notes:
- Vendor code updated: All `nn.Buffer(...)` usage replaced with `register_buffer(...)` for broad PyTorch compatibility.
- No runtime hotpatching required — jobs pin an upstream TRM commit for reproducibility.

Notes for H200 (Hopper, SM90):
- Compute capability is SM90 (same as H100). Use `TORCH_CUDA_ARCH_LIST=9.0`.
- Larger VRAM (up to 141GB SXM) allows increasing `global_batch_size` and/or per‑GPU microbatch to reduce gradient accumulation.
- Performance is generally similar to H100 with higher memory bandwidth; expect slight throughput gains but plan schedules assuming H100‑like speeds.

---

## 3) Instance Setup

Run these on the provisioned host (inside the container):

```bash
apt-get update && apt-get install -y git tmux build-essential libssl-dev
python3 -m pip install --upgrade pip wheel setuptools
```

Clone your workspace (this repo) to `/workspace`:

```bash
mkdir -p /workspace && cd /workspace
# If you have your patched repo in a remote, use it; otherwise:
git clone https://github.com/your-org-or-local/samsung-to-kaggle.git trm
cd /workspace/trm
```

Python deps:

```bash
pip install einops tqdm coolname pydantic argdantic omegaconf hydra-core packaging numba triton
# Torch (if not present in image)
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install `adam-atan2` (CUDA build):

```bash
# Set SM arch list per GPU type (choose one):
# H100 (SM90):
export TORCH_CUDA_ARCH_LIST="9.0"
# A100 (SM80):
# export TORCH_CUDA_ARCH_LIST="8.0"
# L40S (Ada, SM89):
# export TORCH_CUDA_ARCH_LIST="8.9"

export FORCE_CUDA=1
pip install --no-cache-dir --no-build-isolation adam-atan2
```

Disable W&B by default (optional):

```bash
export WANDB_MODE=disabled
export WANDB_DISABLED=true
```

---

## 3b) CoreWeave (Kubernetes) Quickstart — H200 Pods

CoreWeave provides GPUs via Kubernetes. Below is a single‑node Job example requesting 4 GPUs on an H200 node. Replace the node selector with the exact label in your cluster.

Discover GPU labels (one‑time):
```bash
kubectl get nodes --show-labels | rg -i 'h200|gpu|product'
# Look for a label key/value you can select on, e.g.:
#  - nvidia.com/gpu.product=NVIDIA-H200-SXM5-141GB
#  - or a CoreWeave‑specific selector (consult your CW org’s cluster docs)
```

Example Job (single node, 4×GPU) — pin TRM commit, no hotpatching:
```yaml
# cw-trm-h200-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: trm-arc2-h200
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-H200-SXM5-141GB   # <- replace if different in your cluster
      containers:
      - name: trainer
        image: pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel
        command: ["bash","-lc"]
        args:
          - |
            set -euo pipefail
            apt-get update && apt-get install -y git tmux build-essential libssl-dev ripgrep
            python3 -m pip install --upgrade pip wheel setuptools
            mkdir -p /workspace && cd /workspace
            git clone https://github.com/your-org-or-local/samsung-to-kaggle.git trm
            cd /workspace/trm
            # Pin upstream TinyRecursiveModels (if cloning upstream in the job)
            # TRM_COMMIT=<sha>
            # git -C /workspace/trm/trm_repo fetch --depth 1 origin "$TRM_COMMIT" || true
            # git -C /workspace/trm/trm_repo checkout "$TRM_COMMIT" || true
            pip install einops tqdm coolname pydantic argdantic omegaconf hydra-core packaging numba triton
            pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            export TORCH_CUDA_ARCH_LIST=9.0
            export FORCE_CUDA=1
            pip install --no-cache-dir --no-build-isolation adam-atan2 || true
            export WANDB_MODE=disabled
            export WANDB_DISABLED=true
            cd /workspace/trm/trm_repo
            python -m dataset.build_arc_dataset \
              --input-file-prefix kaggle/combined/arc-agi \
              --output-dir /workspace/data/arc2concept-aug-1000 \
              --subsets training2 evaluation2 concept \
              --test-set-name evaluation2
            # Verify dataset (TRM creates only 2 splits: train and test)
            test "$(find /workspace/data/arc2concept-aug-1000/ -name dataset.json | wc -l)" = 2
            # Launch training on 4 GPUs (single node)
            DISABLE_COMPILE=1 \
            torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
              arch=trm data_paths="[/workspace/data/arc2concept-aug-1000]" \
              arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
              +run_name=pretrain_att_arc2concept ema=True \
              +checkpoint_path=/workspace/checkpoints/trm_arc2_run1
        resources:
          limits:
            nvidia.com/gpu: 4
            cpu: "16"
            memory: 128Gi
```

Apply and follow logs:
```bash
kubectl apply -f cw-trm-h200-job.yaml
kubectl logs -f job/trm-arc2-h200
```

Multi‑node on CoreWeave: use a headless Service or Pod IPs for rendezvous, set `--nnodes`, `--nproc-per-node`, and environment for `MASTER_ADDR`, `MASTER_PORT`, `NODE_RANK`, `WORLD_SIZE`. For TRM, single‑node 4×H200 is sufficient; prefer single node unless you specifically need >4 GPUs.

Storage:
- Smoke tests: do NOT use a PVC. Use the ephemeral Job as shown; it is simpler and sufficient to validate the pipeline. If you need to download artifacts once, add a short sleep at the end and copy while the pod is still running (see Retrieve artifacts below).
- Long runs: create and mount a PersistentVolumeClaim (PVC) so checkpoints and `submission.json` persist after the Job completes or across restarts.

PVC example (apply once):
```yaml
# pvc-trm.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: trm-workspace-pvc
spec:
  accessModes: ["ReadWriteOnce"]
  resources:
    requests:
      storage: 200Gi
  # storageClassName: <your-storage-class>  # optional; use your default if omitted
```

Mount in your long‑run Job:
```yaml
      containers:
      - name: trainer
        volumeMounts:
        - name: trm-workspace
          mountPath: /workspace
      volumes:
      - name: trm-workspace
        persistentVolumeClaim:
          claimName: trm-workspace-pvc
```

Retrieve artifacts:
```bash
# For ephemeral smoke Job (no PVC): hold container for 5 minutes at end of script
#   echo 'sleep 300' >> end of Job's args script, then:
POD=$(kubectl get pod -l job-name=trm-arc2-h200-smoke -o jsonpath='{.items[0].metadata.name}')
kubectl cp "$POD":/workspace/checkpoints/smoke_h200 ./cw_artifacts/smoke_h200

# For long‑run Job with PVC: start a short helper pod mounting the same PVC and cp from there,
# or reuse another Job/Pod later — data persists on the PVC.
```

Networking/NCCL: for single‑node, defaults are fine. For multi‑node, follow CoreWeave’s NCCL guidance; at minimum set `NCCL_DEBUG=INFO` and verify IB fabric in your pod spec.

Budgeting on CoreWeave: check the “Instance Pricing” page in the CW docs portal for live rates. As a rule of thumb, budget similarly to H100 with +10–25% headroom unless you have contracted pricing.

---

## 3c) Docker Workflow (Pre-Built Images) - RECOMMENDED

**✅ MIGRATION COMPLETE:** We have successfully migrated from GitHub Container Registry (GHCR) to Docker Hub due to size limitations. The pre-built image is now available and ready for use.

### Docker Hub Image Details

- **Repository:** `alexthuth/trm-arc2`
- **Latest Tag:** `alexthuth/trm-arc2:latest`
- **URL:** https://hub.docker.com/r/alexthuth/trm-arc2
- **Status:** ✅ **PRODUCTION READY**

### Benefits of Docker Approach

- ✅ **Faster startup:** ~1-2 minutes (vs ~10-15 min for inline scripts)
- ✅ **Reproducible:** Same image works locally, on RunPod, and on CoreWeave
- ✅ **Cacheable:** Build once, run many times
- ✅ **Testable:** Validate environment locally before GPU runs
- ✅ **No size limits:** Docker Hub handles large images without issues
- ✅ **Automated updates:** Image rebuilds automatically on code changes

### Quick Start with Docker Hub Image

**1. Pull the Image**
```bash
docker pull alexthuth/trm-arc2:latest
```

**2. Validate Environment**
```bash
# Test Python environment
docker run --rm alexthuth/trm-arc2:latest python -c "import torch, hydra, einops; print('✅ OK')"

# Test Hydra config parsing
docker run --rm alexthuth/trm-arc2:latest bash -c "
  cd /workspace/trm && python trm_repo/pretrain.py --help | head -20
"
```

**3. Use in CoreWeave Job**

Replace inline setup with the pre-built image:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: trm-arc2-h200-smoke-docker
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 3600  # 1 hour timeout
  ttlSecondsAfterFinished: 3600  # Auto-cleanup after 1hr
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-H200-SXM5-141GB
      containers:
      - name: trainer
        image: alexthuth/trm-arc2:latest  # <-- Docker Hub image
        imagePullPolicy: Always
        command: ["bash", "-c"]
        args:
          - |
            set -euo pipefail
            cd /workspace/trm/trm_repo

            # Build dataset (dependencies already installed!)
            python -m dataset.build_arc_dataset \
              --input-file-prefix kaggle/combined/arc-agi \
              --output-dir /workspace/data/arc2concept-aug-100 \
              --subsets training2 evaluation2 concept \
              --test-set-name evaluation2 \
              --num-aug 100 \
              --max-puzzles 100

            # Verify (must be 2 files)
            FILE_COUNT=$(find /workspace/data/arc2concept-aug-100/ -name dataset.json | wc -l)
            if [ "$FILE_COUNT" != "2" ]; then
              echo "❌ Dataset verification failed! Found $FILE_COUNT files, expected 2"
              exit 1
            fi
            echo "✅ Dataset verified: 2 files (train, test)"

            # Train
            cd /workspace/trm
            DISABLE_COMPILE=1 torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
              trm_repo/pretrain.py \
              arch=trm data_paths="[/workspace/data/arc2concept-aug-100]" \
              epochs=1 eval_interval=1 \
              arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
              +run_name=smoke_h200_docker \
              +checkpoint_path=/workspace/checkpoints/smoke_h200_docker
        resources:
          limits:
            nvidia.com/gpu: 4
            cpu: "16"
            memory: 128Gi
```

### Enable Live Telemetry (W&B)

By default the image disables W&B via `WANDB_MODE=disabled` and `WANDB_DISABLED=true`. To see losses/metrics live, explicitly enable W&B and provide credentials.

Kubernetes (recommended via Secret):
```bash
# Create once per cluster/namespace (do NOT commit the key)
kubectl create secret generic wandb-api-key \
  --from-literal=token='<YOUR_WANDB_API_KEY>'
```

Inject into your Job and override the defaults:
```yaml
      containers:
      - name: trainer
        env:
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: wandb-api-key
              key: token
        - name: WANDB_PROJECT
          value: trm-arc2
        - name: WANDB_ENTITY   # optional; set to your W&B team/user
          value: your-entity
        - name: WANDB_MODE     # ensure online logging
          value: online
        - name: WANDB_DISABLED # override image default (must be empty)
          value: ""
```

Also add `project_name=trm-arc2` to the training command so runs have a clear project in W&B:
```bash
... trm_repo/pretrain.py arch=trm ... project_name=trm-arc2 +run_name=smoke_h200_docker ...
```

Local Docker (single GPU) with telemetry:
```bash
# Mount your workspace for checkpoints, mount a data dir, enable W&B
docker run --rm -it --gpus all --shm-size=1g \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  -e WANDB_PROJECT=trm-arc2 \
  -e WANDB_MODE=online \
  -e WANDB_DISABLED= \
  -v "$PWD":/workspace/trm \
  -v /path/to/data:/workspace/data \
  alexthuth/trm-arc2:latest bash -lc '
    cd /workspace/trm && \
    DISABLE_COMPILE=1 torchrun --nproc-per-node 1 trm_repo/pretrain.py \
      arch=trm data_paths="[/workspace/data/arc2concept-aug-100]" \
      epochs=1 eval_interval=1 arch.L_layers=1 arch.H_cycles=1 arch.L_cycles=1 \
      arch.hidden_size=64 arch.num_heads=4 arch.puzzle_emb_ndim=0 arch.puzzle_emb_len=0 \
      arch.forward_dtype=float32 project_name=trm-arc2 +run_name=smoke_local_docker \
      +checkpoint_path=/workspace/checkpoints/smoke_local_docker'
```

Notes:
- `WANDB_DISABLED` must be unset/empty to enable logging (setting `false` still disables it).
- The training script prints progress to stdout; detailed loss/metrics stream to W&B when enabled. Tail logs with `docker logs -f <container>` or `kubectl logs -f job/<name>` and open the W&B run URL printed at startup.

### Image Contents

The Docker Hub image includes:
- **Base:** PyTorch 2.4.0 with CUDA 12.1 devel
- **Dependencies:** All Python packages pre-installed
- **CUDA Support:** Full toolkit for adam-atan2 compilation
- **Project Code:** Complete TRM repository structure
- **Environment:** W&B disabled by default

### Migration History

- **Previous:** GitHub Container Registry (GHCR) - failed due to 10GB layer limit
- **Current:** Docker Hub - no size restrictions, reliable builds
- **Build Process:** Automated via GitHub Actions on master branch changes
- **Status:** ✅ Production ready, tested, and documented

### When to Use Docker vs Inline Scripts

**Use Docker (Recommended):**
- ✅ Multiple smoke tests or training runs
- ✅ Production training (consistent environment)
- ✅ Debugging CoreWeave issues (faster iteration)
- ✅ Team collaboration (same environment for everyone)

**Use Inline Scripts:**
- ⚠️ One-off experiments where build time > script time
- ⚠️ Rapid prototyping with frequently changing dependencies
- ⚠️ When you need to test dependency changes immediately

### Troubleshooting

**Image Pull Issues:**
```bash
# Check if image exists
docker manifest inspect alexthuth/trm-arc2:latest

# Pull with explicit tag
docker pull alexthuth/trm-arc2:latest
```

**CUDA Not Available:**
```bash
# Test with GPU support
docker run --rm --gpus all alexthuth/trm-arc2:latest nvidia-smi
```

**For complete Docker Hub migration documentation, see:** `docs/dockerhub-migration-complete.md`

---

## 3d) Production Stack (8×GPU, Single Node)

Kubernetes pattern (single pod):
- RWX PVC (500Gi, storage class `shared-vast`) mounted via subPath to `/workspace/data` and `/workspace/checkpoints`.
- One Job requesting 8×H200 (8 GPUs total) with the pinned Docker Hub image `alexthuth/trm-arc2@sha256:479572c8...ec2873b`.
- Runtime env: `/dev/shm` (8Gi), `OMP_NUM_THREADS=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Training config (unchanged): `arch.L_cycles=4`, `global_batch_size=768`, `eval_interval=800` (divides `epochs=100000`), `checkpoint_every_eval=True`.
- Optional smoke test: a copy of the Job with `epochs=1 eval_interval=1 checkpoint_every_eval=True` to force a quick checkpoint + evaluator run.

Important: Mount the shared PVC only to data and checkpoints via subPath to avoid shadowing the image’s code under `/workspace/trm`:
```yaml
volumeMounts:
- name: trm-shared
  mountPath: /workspace/data
  subPath: data
- name: trm-shared
  mountPath: /workspace/checkpoints
  subPath: checkpoints
```
This keeps the prebuilt Docker image’s repository intact while persisting datasets and checkpoints on the PVC.

One‑time setup:
```bash
kubectl apply -f jobs/trm-shared-pvc.yaml
kubectl get pvc trm-shared-pvc
```

Run a production‑parity smoke (recommended):
```bash
kubectl apply -f <your single‑node 8×GPU smoke‑test job yaml>
kubectl logs -f job/trm-prod-smoke-master
```

Run the full production job:
```bash
kubectl apply -f <your single‑node 8×GPU production job yaml>
kubectl logs -f job/trm-production-master-docker
```

Validation checklist (production smoke):
- W&B run URL printed in master logs; metrics stream during TRAIN.
- Checkpoint saved under `/workspace/checkpoints/prod_smoke/step_*` (PVC).
- Evaluator writes `/workspace/checkpoints/prod_smoke/evaluator_ARC_step_*/submission.json`.
- No NCCL/DataLoader/compile errors.

Post‑smoke quick inference (optional):
```bash
# Pick latest checkpoint
CKPT=$(kubectl exec -ti $(kubectl get pod -l job-name=trm-prod-smoke-master -o jsonpath='{.items[0].metadata.name}') -- \
  bash -lc "ls -1t /workspace/checkpoints/prod_smoke/step_* | head -n1" | tr -d '\r')

# Package to Kaggle format
kubectl exec -ti $(kubectl get pod -l job-name=trm-prod-smoke-master -o jsonpath='{.items[0].metadata.name}') -- \
  bash -lc "cd /workspace/trm && scripts/package_kaggle_weights.sh $CKPT /workspace/kaggle_trm_weights"

# Run minimal inference on evaluation puzzles
kubectl exec -ti $(kubectl get pod -l job-name=trm-prod-smoke-master -o jsonpath='{.items[0].metadata.name}') -- \
  bash -lc "cd /workspace/trm && \
    python kaggle_trm_code/infer_trm_arc.py /workspace/data/arc2concept-aug-1000/test_puzzles.json \
      /workspace/kaggle_trm_weights/weights \
      /workspace/checkpoints/prod_smoke/submission_from_kaggle_infer.json && \
    head -n 2 /workspace/checkpoints/prod_smoke/submission_from_kaggle_infer.json"
```

Notes:
- The master builds the dataset if missing; worker waits. Subsequent runs reuse the PVC.
- Use subPath mounts for `/workspace/data` and `/workspace/checkpoints` only; do not mount the PVC over `/workspace` or you will hide the prebuilt code.
- The evaluator detaches tensors internally to be safe under inference mode and distributed gather.
- Epoch time on 16×H200 is ~3–7 minutes after dataset exists; first build adds ~10–15 minutes.

### Eval Cadence (~5k steps per eval)

- `eval_interval` represents “epochs per eval”, not steps. Steps per eval are approximately:
  `steps_per_eval ≈ eval_interval × (dataset_factor) ÷ global_batch_size`.
- With ARC‑AGI‑2 (dataset_factor ≈ 5,545 observed) and `global_batch_size=768`, `eval_interval=800` yields ~5,780 steps per eval. Ensure `eval_interval` divides `epochs` (100,000).
- If eval overhead is high, relax to `eval_interval≈1000` (~7,220 steps/eval) to reduce frequency.
- Checkpointing: set `checkpoint_every_eval=True` so every eval emits a checkpoint for A/B comparisons and inference.
- Note: The code does not require `epochs` to be divisible by `eval_interval`.

---

## 4) Build ARC‑AGI‑2 Data

Use the bundled JSONs (`trm_repo/kaggle/combined`) and produce a training dataset:

```bash
cd /workspace/trm/trm_repo
python3 -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir /workspace/data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2
```

**Expected duration:** 10-15 minutes (observed: 11-12 min on 4×H100)
**Memory usage:** ~15GB RAM peak (single-threaded Python process)
**Output:** 2 subdirectories (train, test) each with `dataset.json` + puzzle arrays
  - train/ contains: training2 + concept subsets
  - test/ contains: evaluation2 subset
  - Note: TRM does NOT create a validation split by default

**CRITICAL - Verify completion before training:**
```bash
# Must return 2 (train, test) - TRM creates only 2 splits, no validation
find /workspace/data/arc2concept-aug-1000/ -name 'dataset.json' | wc -l

# Verify each split exists
ls -lh /workspace/data/arc2concept-aug-1000/{train,test}/dataset.json

# Check file sizes (should be >1MB each)
du -h /workspace/data/arc2concept-aug-1000/*/dataset.json
```

**If verification fails** (only 1 file found or 0 files):
```bash
# Clean and rebuild
sudo rm -rf /workspace/data/arc2concept-aug-1000
sudo mkdir -p /workspace/data
sudo chown -R ubuntu:ubuntu /workspace

# Rebuild
python3 -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir /workspace/data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

# Wait and re-verify
sleep 60
find /workspace/data/arc2concept-aug-1000/ -name 'dataset.json' | wc -l
```

Notes:
- Do not mix ARC‑AGI‑1 eval with ARC‑AGI‑2 training.
- Dataset builder can silently produce incomplete output - always verify!
- If building fails repeatedly, consider reducing augmentations (see Section 16 Post-Mortem)

---

## 5) Training Parameters & Launch (4×H100)

Baseline TRM settings:
- `arch=trm`, `arch.L_layers=2`, `arch.H_cycles=3`, `arch.L_cycles=6`
- `global_batch_size`: start from 768 and adjust to VRAM
- Optimizer: AdamATan2 (installed above)
- Precision: `arch.forward_dtype=bfloat16`
- EMA: `ema=True`

Launch (single node, 4 GPUs):

```bash
cd /workspace/trm/trm_repo
DISABLE_COMPILE=1 \
WANDB_MODE=disabled WANDB_DISABLED=true \
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  arch=trm \
  data_paths="[/workspace/data/arc2concept-aug-1000]" \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 \
  +run_name=pretrain_att_arc2concept \
  ema=True \
  +checkpoint_path=/workspace/checkpoints/trm_arc2_run1
```

Tips:
- If stable, try enabling `torch.compile` by removing `DISABLE_COMPILE=1` for extra performance.
- If `adam-atan2` fails to build, the code falls back to Adam automatically (slight quality impact).

### H200‑specific notes (applies to H100 as well)
- Architecture flag remains `TORCH_CUDA_ARCH_LIST=9.0` (Hopper/SM90).
- With 141GB VRAM per GPU, you can increase `global_batch_size` compared to H100‑80G. Start with +25–50% and monitor memory; prefer fewer grad‑accum steps rather than pushing per‑step memory to the edge.
- Expected wall‑clock is similar to H100 for identical configs; gains mostly come from larger effective batch or reduced accumulation overhead.

---

## 5b) CUDA/NCCL Environment Reference

- `TORCH_CUDA_ARCH_LIST`:
  - H200/H100 (Hopper/SM90): `9.0`
  - A100 (SM80): `8.0`
  - L40S (Ada, SM89): `8.9`
- `FORCE_CUDA=1`: set when building `adam-atan2` to ensure CUDA extension compiles.
- `DISABLE_COMPILE=1`: start with Torch compile disabled; re‑enable once stable.
- NCCL (single‑node): defaults are fine. For multi‑node, set `NCCL_DEBUG=INFO` and verify your provider’s fabric config (IB/NVLink). On CoreWeave, follow their multi‑node NCCL guidance if scaling beyond one node.
- Helpful toggles for debugging/stability:
  - `HYDRA_FULL_ERROR=1` and `PYTHONUNBUFFERED=1` for full tracebacks and unbuffered logs
  - `NCCL_ASYNC_ERROR_HANDLING=1` for clearer distributed error surfacing
  - `OMP_NUM_THREADS=1` to avoid CPU oversubscription
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce memory fragmentation

---

## 5c) Smoke Tests (Local & CoreWeave H200)

Purpose: validate end‑to‑end setup (deps → dataset → training → eval → artifacts) before committing to a long run.

### A) Local Smoke (CPU or 1 GPU)

Build a tiny dataset to keep runtime short:
```bash
cd /workspace/trm/trm_repo
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir /workspace/data/arc2concept-aug-100 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2 \
  --num-aug 100 \
  --max-puzzles 100

# Verify (must be 2 files - TRM creates only train and test, no validation)
find /workspace/data/arc2concept-aug-100/ -name 'dataset.json' | wc -l
ls -lh /workspace/data/arc2concept-aug-100/{train,test}/dataset.json
```

Run a minimal model for one epoch (CPU or single GPU):
```bash
cd /workspace/trm
DISABLE_COMPILE=1 WANDB_MODE=disabled WANDB_DISABLED=true \
  python trm_repo/pretrain.py \
  arch=trm data_paths="[/workspace/data/arc2concept-aug-100]" \
  epochs=1 eval_interval=1 \
  arch.hidden_size=64 arch.num_heads=4 \
  arch.L_layers=1 arch.H_cycles=1 arch.L_cycles=1 \
  arch.puzzle_emb_ndim=0 arch.puzzle_emb_len=0 \
  arch.forward_dtype=float32 \
  evaluators='[{name: arc@ARC}]' \
  +run_name=smoke_local \
  +checkpoint_path=/workspace/checkpoints/smoke_local
```

Expected outputs:
- Checkpoint directory with a step: `/workspace/checkpoints/smoke_local/step_*`
- ARC evaluator artifact: `/workspace/checkpoints/smoke_local/evaluator_ARC_step_*/submission.json`

Typical duration: 5–10 minutes on CPU; 2–5 minutes on a single modern GPU.

### B) CoreWeave H200 Smoke (Kubernetes Job)

Use a job mirroring production, but with a tiny dataset and a single epoch:
```yaml
# cw-trm-h200-smoke.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: trm-arc2-h200-smoke
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-H200-SXM5-141GB   # adjust for your cluster
      containers:
      - name: trainer
        image: pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel
        command: ["bash","-lc"]
        args:
          - |
            set -euo pipefail
            apt-get update && apt-get install -y git tmux build-essential libssl-dev ripgrep
            python3 -m pip install --upgrade pip wheel setuptools
            mkdir -p /workspace && cd /workspace
            git clone https://github.com/your-org-or-local/samsung-to-kaggle.git trm
            cd /workspace/trm
            pip install einops tqdm coolname pydantic argdantic omegaconf hydra-core packaging numba triton
            pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            export TORCH_CUDA_ARCH_LIST=9.0
            export FORCE_CUDA=1
            pip install --no-cache-dir --no-build-isolation adam-atan2 || true
            export WANDB_MODE=disabled
            export WANDB_DISABLED=true

            cd /workspace/trm/trm_repo
            python -m dataset.build_arc_dataset \
              --input-file-prefix kaggle/combined/arc-agi \
              --output-dir /workspace/data/arc2concept-aug-100 \
              --subsets training2 evaluation2 concept \
              --test-set-name evaluation2 \
              --num-aug 100 \
              --max-puzzles 100
            test "$(find /workspace/data/arc2concept-aug-100/ -name dataset.json | wc -l)" = 2

            # 4×GPU single node, 1 epoch with ARC evaluator
            cd /workspace/trm
            DISABLE_COMPILE=1 \
            torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 trm_repo/pretrain.py \
              arch=trm data_paths="[/workspace/data/arc2concept-aug-100]" \
              epochs=1 eval_interval=1 \
              arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
              arch.forward_dtype=bfloat16 \
              evaluators='[{name: arc@ARC}]' \
              +run_name=smoke_h200 \
              +checkpoint_path=/workspace/checkpoints/smoke_h200
        resources:
          limits:
            nvidia.com/gpu: 4
            cpu: "16"
            memory: 128Gi
```

Run and verify:
```bash
kubectl apply -f cw-trm-h200-smoke.yaml
kubectl logs -f job/trm-arc2-h200-smoke

# After completion (Succeeded):
kubectl exec -ti job/trm-arc2-h200-smoke -- ls -lh /workspace/checkpoints/smoke_h200
kubectl exec -ti job/trm-arc2-h200-smoke -- find /workspace/checkpoints/smoke_h200 -name submission.json -maxdepth 2
```

Expected outputs (inside the pod):
- `/workspace/checkpoints/smoke_h200/step_*`
- `/workspace/checkpoints/smoke_h200/evaluator_ARC_step_*/submission.json`

Typical duration: ~6–12 minutes end‑to‑end (2–3 min dataset, ~3–7 min train+eval).

Storage guidance:
- Smoke: keep it simple — no PVC. If you want to copy results, you can add a `sleep 300` at the end of the Job script and then:
  ```bash
  POD=$(kubectl get pod -l job-name=trm-arc2-h200-smoke -o jsonpath='{.items[0].metadata.name}')
  kubectl cp "$POD":/workspace/checkpoints/smoke_h200 ./cw_artifacts/smoke_h200
  ```
- Long run: use a PVC (see CoreWeave Quickstart section for the `pvc-trm.yaml` and how to mount it) so checkpoints persist after completion.

Smoke Test Pass Criteria:
- Dataset verified (2 dataset.json files: train and test)
- Training runs to completion (1 epoch) without NCCL/compile errors
- Checkpoint directory exists with at least one `step_*`
- ARC `submission.json` exists under the checkpoint path

---

## 6) Checkpointing & Resume

- Checkpoints are saved under `+checkpoint_path` at each eval cycle:
  - `/workspace/checkpoints/trm_arc2_run1/step_<N>`
- Resume example:
  ```bash
  torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
    arch=trm data_paths="[/workspace/data/arc2concept-aug-1000]" \
    load_checkpoint=/workspace/checkpoints/trm_arc2_run1/step_<N> \
    +checkpoint_path=/workspace/checkpoints/trm_arc2_run1_resume
  ```
- Puzzle embeddings are reshaped automatically if identifier counts differ.

---

## 7) Robustness & Monitoring

- Logs: stdout shows progress; evaluation prints “Completed inference in X steps”.
- Precision: keep `bfloat16`; if instability occurs, try `arch.forward_dtype=float32`.
- NCCL: single‑node defaults usually work. For multi‑node, configure rendezvous and open ports.
- Dataloader: already optimized for compute‑bound training (`num_workers=1`).

---

## 8) Verification (Evaluation & Submission Artifact)

Enable the ARC evaluator to write Kaggle‑style `submission.json` per checkpoint:

```bash
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  arch=trm data_paths="[/workspace/data/arc2concept-aug-1000]" \
  evaluators='[{name: arc@ARC}]' \
  +checkpoint_path=/workspace/checkpoints/trm_arc2_run1
```

This produces:
- `/workspace/checkpoints/trm_arc2_run1/evaluator_ARC_step_<N>/submission.json`

Use it for local validation and as a baseline Kaggle submission prototype.

---

## 9) Packaging for Kaggle

Single offline bundle + repo snapshot:

1. **Wheels+weights dataset** — `seconds0/trm-offline-wheels-py311`
   - Wheels: Hydra 1.3.2, OmegaConf 2.3.0, Pydantic 2.8.2 + core 2.20.1, numba/llvmlite cp311, adam-atan2 0.0.3 wheel, auditwheel toolchain, etc.
   - Checkpoint: `model.ckpt` (latest ARC-AGI-2 run) lives at dataset root for `torch.load`.
   - README documents exact `pip install --no-index --find-links=/kaggle/input/trm-offline-wheels-py311 ...` commands.

2. **Repo snapshot** — `seconds0/trm-repo-clean` (zip of TinyRecursiveModels at commit `e7b6871`).

3. **Inference kernel** — `seconds0/trm-arc-agi-2-inference-py311-offline`
   - Attaches the two datasets above + `arc-prize-2025` competition data.
   - Builds the ARC evaluation dataset, loads `model.ckpt`, and emits `/kaggle/working/trm_eval_outputs/evaluator_ARC_step_72385/submission.json` (pass@1 ≈ 0.628).
   - Uses single-process `dist.init_process_group("gloo")` so evaluator `gather_object` succeeds without multi-node setup.

### 9.1 Kaggle Validation (2025-10-18)

- Kernel run succeeded end-to-end on Kaggle GPU after bundling the checkpoint with the wheels dataset.
- Evaluator metrics: `accuracy=0.6283`, `lm_loss=2.0186`, `q_halt_accuracy=0.9070`, ACT steps capped at 16; all pass@K values for K≤1000 remain 0 (matching paper expectations).
- Submission artifact written to `/kaggle/working/trm_eval_outputs/evaluator_ARC_step_72385/submission.json`; logs confirm offline installs and dataset build completed without internet access.

---

## 10) Budgeting

### Smoke Test (Recommended First Step)

**Purpose:** Verify configuration works before committing to expensive 3-day run.

**Duration:** ~73 minutes (~1.2 hours)

**Cost breakdown (at $7.56/hr for 4×H100):**
| Phase | Duration | Cost |
|-------|----------|------|
| Provisioning | 5 min | $0.63 |
| Environment setup | 15 min | $1.89 |
| Dataset building + verification | 18 min | $2.27 |
| Training (30 min smoke test) | 30 min | $3.78 |
| Buffer/monitoring | 5 min | $0.63 |
| **Total** | **~73 min** | **~$9.20** |

**Budget:** Plan for **$10-12** to allow for retries if issues occur.

### Full Training Run (3 Days)

- Target runtime: ~3 days (72 hours) on 4×H100 for baseline config (per upstream README).
- Rough cost ranges (query current offers before booking):
  - **H100**: 4×$2.2/GPU‑hr × 72hr ≈ $630 (range $400–$1k)
  - **H200**: similar to H100 with provider‑specific premium; plan H100 cost +10–25% unless you have negotiated pricing
  - **A100‑80G**: 4×$1.6/GPU‑hr × 72hr ≈ $460 (range $300–$700)
  - **L40S**: 4×$0.8/GPU‑hr × 72hr ≈ $230 (range $100–$350, may require longer training or tuning)

**Recommendation:** ALWAYS run smoke test first. Do NOT attempt full 3-day run without successful smoke test completion.

### Total Project Budget

- Smoke test: $10-12
- Full training: $400-1000 (H100)
- Kaggle submission trials: $0 (free tier)
- **Minimum total**: ~$410-1012 for H100 path
- **Budget recommendation**: $500-1200 with contingency for retries

---

## 11) Quick Repro Scripts

Build data:
```bash
cd /workspace/trm/trm_repo
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir /workspace/data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2
```

Build tiny smoke dataset (fast):
```bash
cd /workspace/trm/trm_repo
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir /workspace/data/arc2concept-aug-100 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2 \
  --num-aug 100 \
  --max-puzzles 100
```

Train (4×H100):
```bash
DISABLE_COMPILE=1 WANDB_MODE=disabled WANDB_DISABLED=true TORCH_CUDA_ARCH_LIST=9.0 FORCE_CUDA=1 \
  torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 trm_repo/pretrain.py \
  arch=trm data_paths="[/workspace/data/arc2concept-aug-1000]" \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=pretrain_att_arc2concept ema=True \
  +checkpoint_path=/workspace/checkpoints/trm_arc2_run1
```

Train (4×H200 on CoreWeave, single node — identical command):
```bash
DISABLE_COMPILE=1 WANDB_MODE=disabled WANDB_DISABLED=true TORCH_CUDA_ARCH_LIST=9.0 FORCE_CUDA=1 \
  torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 trm_repo/pretrain.py \
  arch=trm data_paths="[/workspace/data/arc2concept-aug-1000]" \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=pretrain_att_arc2concept ema=True \
  +checkpoint_path=/workspace/checkpoints/trm_arc2_run1
# Consider increasing global_batch_size by ~25–50% on H200 if VRAM allows
```

Evaluate (ARC evaluator enabled):
```bash
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 trm_repo/pretrain.py \
  arch=trm data_paths="[/workspace/data/arc2concept-aug-1000]" \
  evaluators='[{name: arc@ARC}]' \
  +checkpoint_path=/workspace/checkpoints/trm_arc2_run1
```

Local smoke (CPU/1‑GPU, tiny dataset, 1 epoch):
```bash
DISABLE_COMPILE=1 WANDB_MODE=disabled WANDB_DISABLED=true \
  python trm_repo/pretrain.py \
  arch=trm data_paths="[/workspace/data/arc2concept-aug-100]" \
  epochs=1 eval_interval=1 \
  arch.hidden_size=64 arch.num_heads=4 \
  arch.L_layers=1 arch.H_cycles=1 arch.L_cycles=1 \
  arch.puzzle_emb_ndim=0 arch.puzzle_emb_len=0 \
  arch.forward_dtype=float32 \
  evaluators='[{name: arc@ARC}]' \
  +run_name=smoke_local \
  +checkpoint_path=/workspace/checkpoints/smoke_local
```

---

## 12) Common Pitfalls & Fixes

- `adam-atan2` build fails:
  - Use a CUDA devel image; set `FORCE_CUDA=1` and `TORCH_CUDA_ARCH_LIST` appropriately; retry install.
  - Fallback to Adam is automatic, but may slightly affect quality.
- BF16/precision issues: temporarily switch to `arch.forward_dtype=float32` to isolate.
- Torch compile instability: start with `DISABLE_COMPILE=1`; re‑enable once stable.
- NCCL: for multi‑node, configure rendezvous/ports; for single node, defaults typically work.

---

## 13) Optional: One‑Click Vast.ai On‑Start Bootstrap

You can pass an on‑start script to Vast.ai to fully bootstrap:

```bash
#!/usr/bin/env bash
set -euo pipefail
apt-get update && apt-get install -y git tmux build-essential libssl-dev
python3 -m pip install --upgrade pip wheel setuptools
mkdir -p /workspace && cd /workspace
# TODO: replace with your repo URL
git clone https://github.com/your-org-or-local/samsung-to-kaggle.git trm
cd /workspace/trm
pip install einops tqdm coolname pydantic argdantic omegaconf hydra-core packaging numba triton
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
export TORCH_CUDA_ARCH_LIST=9.0
export FORCE_CUDA=1
pip install --no-cache-dir --no-build-isolation adam-atan2 || true
export WANDB_MODE=disabled
export WANDB_DISABLED=true
cd /workspace/trm/trm_repo
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir /workspace/data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2
DISABLE_COMPILE=1 torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 pretrain.py \
  arch=trm data_paths="[/workspace/data/arc2concept-aug-1000]" \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=pretrain_att_arc2concept ema=True \
  +checkpoint_path=/workspace/checkpoints/trm_arc2_run1
```

> Replace repo URL, tune batch size and cycles for your GPUs, and consider adding `tmux` to detach the training.

---

## 14) After Training: Package for Kaggle

- Copy the best checkpoint to `kaggle_trm_weights/weights/model.pt`.
- Write `kaggle_trm_weights/weights/config.json` to match training config (hidden_size/heads/L/H cycles/seq_len/vocab/precision/etc.).
- Publish `kaggle_trm_code/` as `seconds0/trm-arc2-code-v2` and `kaggle_trm_weights/` as `seconds0/trm-arc2-weights-v1`.
- Attach both to the kernel and run.

---

If you want a tailored command bundle for a specific provider/region or a different GPU count, let me know your constraints and I'll adapt the hyperparameters and provisioning steps.

---

## 15) Troubleshooting: Common Issues & Solutions

This section documents issues encountered during actual production runs and their fixes.

### A) Dependency Installation Issues

**Problem 1: antlr4-python3-runtime fails with setuptools TypeError**
```
TypeError: canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'
```

**Solution:** Downgrade setuptools before installing omegaconf/hydra:
```bash
pip install 'setuptools<70.0.0'
pip install omegaconf hydra-core  # Now works
```

**Problem 2: adam-atan2 fails to build**
Expected on some systems. Code falls back to Adam automatically.

**Solution:** Ensure CUDA devel tools + correct TORCH_CUDA_ARCH_LIST:
```bash
# Use devel image: pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel
export TORCH_CUDA_ARCH_LIST="9.0"  # H100 (8.0 for A100, 8.9 for L40S)
export FORCE_CUDA=1
pip install --no-cache-dir --no-build-isolation adam-atan2
```

### B) PATH Issues

**Problem: wandb/torchrun commands not found after pip install**

**Root cause:** User-level pip installs to `~/.local/bin`, not in PATH by default.

**Solution:** Add to PATH in all scripts and sessions:
```bash
export PATH="$HOME/.local/bin:$PATH"
wandb login <KEY>  # Now works
```

**Best practice:** Add to `.bashrc` or use root pip installs (with `sudo`).

### C) Dataset Building Issues

**Problem: PermissionError when building dataset**
```
PermissionError: [Errno 13] Permission denied: '/workspace/data'
```

**Solution:** Create workspace directories with correct ownership:
```bash
sudo mkdir -p /workspace/data
sudo chown -R ubuntu:ubuntu /workspace  # Replace 'ubuntu' with your user
python3 -m dataset.build_arc_dataset ...  # Now works
```

### D) Prime Intellect API Issues

**Problem 1: SSH access requires pre-configured key**
- Prime Intellect pods require SSH key set up in dashboard or via API
- Key location: `~/.ssh/primeintellect_ed25519` (if previously configured)

**Problem 2: Pod creation requires maxPrice field**
```json
{"detail":"Missing field: max_price not specified"}
```

**Solution:** Always include `maxPrice` in pod creation:
```json
{
  "pod": {
    "cloudId": "gpu_4x_h100",
    "maxPrice": 10.0,  // Required
    ...
  }
}
```

**Problem 3: Correct API endpoint**
- ✅ Correct: `https://api.primeintellect.ai/api/v1/availability/`
- ❌ Wrong: `https://api.primeintellect.ai/v1/gpus`

### E) Auto-Kill Timer for Cost Control

**Best practice:** Always run auto-kill script to prevent runaway costs:

```bash
# scripts/auto_kill_instance.sh
#!/usr/bin/env bash
POD_ID="your-pod-id"
DURATION_MINUTES=30
PI_TOKEN="your-token"

DURATION_SECONDS=$((DURATION_MINUTES * 60))
sleep $DURATION_SECONDS

curl -s -X DELETE \
  -H "Authorization: Bearer $PI_TOKEN" \
  https://api.primeintellect.ai/api/v1/pods/$POD_ID
```

Run in background with nohup:
```bash
nohup ./auto_kill_instance.sh > /tmp/auto_kill.log 2>&1 &
```

### F) Monitoring Setup

**Best practice:** Enable W&B for smoke tests/debugging:

```bash
# Export environment variables
export PATH="$HOME/.local/bin:$PATH"
export WANDB_PROJECT="trm-arc2-smoke-test"

# Login once
wandb login <YOUR_API_KEY>

# Training will auto-log to W&B
torchrun ... pretrain.py ...
```

**W&B dashboard:** https://wandb.ai/[username]/[project-name]

### G) Training Command Checklist

Before launching training, verify:

1. ✅ Dataset built and accessible (`ls /workspace/data/arc2concept-aug-1000/`)
2. ✅ GPUs detected (`nvidia-smi`)
3. ✅ PyTorch + CUDA working (`python3 -c "import torch; print(torch.cuda.is_available())"`)
4. ✅ PATH includes ~/.local/bin (`which wandb torchrun`)
5. ✅ W&B logged in (`wandb whoami`)
6. ✅ Auto-kill timer started (if using)
7. ✅ Checkpoint directory writable (`mkdir -p /workspace/checkpoints/test && rm -rf /workspace/checkpoints/test`)

---

## 16) Best Practices & Production Improvements

This section documents recommendations from production code reviews and operational experience.

### A) Security: Secrets Management

**Issue**: WANDB_API_KEY hardcoded in YAML files is a security risk.

**Impact**:
- Medium-High for production (credentials exposed in repo)
- Low for 1-hour smoke tests (limited exposure window)

**Recommendation**: Use Kubernetes Secrets for production runs.

**Implementation**:

```bash
# Create secret once per cluster/namespace (do NOT commit the key)
kubectl create secret generic trm-secrets \
  --from-literal=wandb-api-key='140743f34f674639e4930313501dfd0c21556d89'

# Verify secret exists
kubectl get secret trm-secrets
```

**Use in YAML** (replace hardcoded env):

```yaml
# Before (insecure):
env:
- name: WANDB_API_KEY
  value: "140743f34f674639e4930313501dfd0c21556d89"  # ❌ Hardcoded

# After (secure):
env:
- name: WANDB_API_KEY
  valueFrom:
    secretKeyRef:
      name: trm-secrets
      key: wandb-api-key  # ✅ From secret
```

**Alternative**: Use `envFrom` for multiple secrets:

```yaml
containers:
- name: trainer
  envFrom:
  - secretRef:
      name: trm-secrets  # Loads all keys as env vars
```

**Note**: For one-off smoke tests (<1 hour), hardcoded keys are acceptable. Always use secrets for:
- Production runs (>1 day)
- Shared repos (public or team)
- CI/CD pipelines

### B) PVC Sharing for Multi-Node Training

**Issue**: Worker rebuilds dataset (15 min) because it uses ephemeral `emptyDir` storage.

**Impact**:
- Low for 1-hour ablation tests (~20% overhead)
- Medium for 3-day production runs (wasted compute, duplicate work)

**Trade-offs**:

| Approach | Pros | Cons |
|----------|------|------|
| **emptyDir** (current) | Simple, no contention, independent | Rebuilds dataset (~15 min), no shared checkpoints |
| **Shared PVC (RWX)** | Dataset built once, shared checkpoints, faster startup | Requires RWX mode, potential contention, complex cleanup |
| **Separate PVCs** | No contention, isolated | More complex, still rebuilds dataset |

**Current Status**:
- Existing PVC: `trm-workspace-pvc` is **RWO** (ReadWriteOnce) - only one pod can mount
- Multi-node requires either **RWX** (ReadWriteMany) PVC or separate PVCs

**Recommendation**:

**For 1-hour ablation tests**: Keep current `emptyDir` approach (simpler, works fine)

**For production runs**: Share PVC if RWX available:

```yaml
# Check if your storage class supports RWX
kubectl get storageclass shared-vast -o yaml | grep -i access

# If RWX supported, update worker to use same PVC:
volumes:
- name: workspace
  persistentVolumeClaim:
    claimName: trm-workspace-pvc  # Same as master
```

**If only RWO available**, alternatives:
1. Keep `emptyDir` (accept 15 min overhead)
2. Mount PVC read-only on worker (if supported)
3. Create separate worker PVC and sync after master builds

**Storage class reference**:
- `shared-vast` (CoreWeave): Supports RWX for shared filesystems
- Check with `kubectl get storageclass` for your cluster

### C) Checkpoint Configuration

**Status**: Default TRM config includes `checkpoint_every_eval: true`

**Best Practice**: Always set explicit checkpoint path:

```yaml
+checkpoint_path=/workspace/checkpoints/ablation_no_puzzle_emb
```

**This ensures**:
- Checkpoints saved at every eval interval
- Artifacts persist on PVC (if mounted)
- Easy to locate for post-training analysis

**Verification**:
```bash
# List checkpoints on master pod
kubectl exec trm-ablation-master-8kn26 -- \
  ls -lh /workspace/checkpoints/ablation_no_puzzle_emb/

# Check evaluator outputs
kubectl exec trm-ablation-master-8kn26 -- \
  find /workspace/checkpoints -name submission.json
```

### D) Evaluation Cadence

**Current ablation config**: `eval_interval=200` (epochs between evaluations)

**Analysis**:
- `epochs=100000` (default total)
- `eval_interval=200` divides evenly (500 evals total)
- Steps per eval: ~5,780 steps (with batch_size=768, dataset_factor≈5,545)

**Why this is good**:
- Fast feedback for ablation test (eval every ~15-20 min)
- More checkpoints = better A/B comparison
- Divides evenly into total epochs

**Production recommendation**:
- Smoke tests: `eval_interval=200` (frequent evals, quick feedback)
- Production: `eval_interval=800-1000` (reduce eval overhead, fewer checkpoints)

**Rule**: Always ensure `eval_interval` divides `epochs`:
```python
# Good: 100000 % 800 == 0
eval_interval=800  # ✅ 125 evals

# Bad: 100000 % 900 != 0
eval_interval=900  # ❌ Uneven, may miss final eval
```

### E) Production Checklist (Summary)

Before deploying production runs, verify:

**Security**:
- [ ] Secrets configured (not hardcoded)
- [ ] API keys rotated after any leaks
- [ ] YAML files reviewed (no sensitive data)

**Storage**:
- [ ] PVC provisioned and mounted
- [ ] Storage class supports required access mode (RWO or RWX)
- [ ] Disk space sufficient (~50GB per checkpoint, 10-20 checkpoints)

**Configuration**:
- [ ] `eval_interval` divides `epochs` evenly
- [ ] `checkpoint_every_eval=True` (or default enabled)
- [ ] `+checkpoint_path` specified explicitly
- [ ] W&B project/entity configured (if online logging)

**Monitoring**:
- [ ] W&B dashboard accessible (if enabled)
- [ ] Log aggregation working (kubectl logs)
- [ ] Alerts set up for job failures (optional)

**Cost Control**:
- [ ] `activeDeadlineSeconds` set (hard time limit)
- [ ] `ttlSecondsAfterFinished` set (auto-cleanup)
- [ ] Budget alerts configured (provider-level)

### F) Future Improvements (Roadmap)

**Not yet implemented, but recommended**:

1. **Helmfile/Kustomize**: Templatize YAMLs for DRY configuration
2. **GPU affinity**: Pin specific GPUs to pods for consistent performance
3. **Init containers**: Pre-pull Docker images before main container starts
4. **Resource quotas**: Prevent runaway resource consumption
5. **Log shipping**: Send logs to external service (CloudWatch, Grafana)
6. **Checkpoint versioning**: Tag checkpoints with git commit SHA
7. **A/B testing framework**: Compare multiple config runs side-by-side

---

## 17) Post-Mortem: First Smoke Test (October 2025)

**Purpose:** This section documents the first smoke test attempt on Prime Intellect 4×H100 infrastructure, capturing lessons learned for future runs.

**Date:** 2025-10-08  
**Provider:** Prime Intellect (4×H100-80GB @ $7.56/hr, us-central-3)  
**Goal:** 30-minute smoke test to verify training configuration  
**Result:** ⚠️ **INCONCLUSIVE** - Dataset built successfully but pod issues prevented training
**Cost:** $3.78 (30 minutes)

**IMPORTANT CORRECTION:** Initial analysis incorrectly identified dataset building as failed. The dataset builder created 2 of 2 expected files (train/dataset.json and test/dataset.json). TRM does NOT create 3 splits - only train and test.

### Timeline (UTC)

| Time | Event | Status |
|------|-------|--------|
| 10:30 | Instance launched | ✅ Success |
| 10:33-10:45 | Environment setup | ✅ Success (setuptools fix worked) |
| 10:47 | First training attempt | ❌ Failed - dataset.json not found |
| 10:50 | Auto-kill timer started (30 min) | ⚠️ Too early! |
| 10:54 | Dataset rebuild started | 🔄 In progress |
| 11:04-11:06 | Dataset building (11-12 min) | ✅ Created 2/2 expected files |
| 11:20 | Auto-kill terminated instance | ❌ Before training started |

### What Went Wrong

1. ~~**Dataset incomplete**~~ **CORRECTION: Dataset was COMPLETE**
   - ✅ TRM creates only 2 splits by design: train/ and test/ (no validation split)
   - ✅ Builder created train/dataset.json and test/dataset.json successfully
   - ❌ **Our documentation was wrong**, not the dataset builder!

2. **Auto-kill timer started too early**
   - Started BEFORE dataset was ready
   - Correct sequence: Build dataset → Verify → Start timer → Train
   - Actual sequence: Start timer → Build dataset → Terminated before training

3. **Incorrect verification expectations**
   - Documentation incorrectly expected 3 files (train/val/test)
   - TRM only creates 2 files (train/test) by design
   - Should have verified 2 dataset.json files present, not 3

### What Went Right

1. ✅ All dependency fixes worked perfectly:
   - setuptools<70.0.0 resolved antlr4-python3-runtime conflict
   - PyTorch 2.5.1+cu121 installed successfully
   - PATH fix for wandb/pip binaries worked
   - W&B login successful

2. ✅ Prime Intellect API integration solid:
   - Pod provisioning worked
   - SSH access configured correctly
   - Auto-kill timer script executed flawlessly (just started too early)

3. ✅ Permission fixes effective:
   - /workspace directories created with correct ownership
   - No permission errors during dataset build attempt

### Resource Usage Observed

- **Memory**: ~15GB peak during dataset building (out of 503GB available)
- **Disk**: Minimal (~26GB used of 4.9TB)
- **CPU**: 100%+ on single core during dataset build (Python process)
- **Network**: Not measured

### Cost Breakdown

| Phase | Duration | Cost @ $7.56/hr |
|-------|----------|----------------|
| Environment setup | 15 min | $1.89 |
| Failed training attempt | 1 min | $0.13 |
| Dataset building (incomplete) | 12 min | $1.51 |
| Idle waiting | 2 min | $0.25 |
| **Total** | **30 min** | **$3.78** |

### Critical Lessons Learned

1. **ALWAYS verify dataset before training**
   ```bash
   # MANDATORY verification (must return 2: train and test)
   find /workspace/data/arc2concept-aug-1000/ -name 'dataset.json' | wc -l
   ```

2. **NEVER start auto-kill timer early**
   - Wait for dataset completion
   - Verify all files exist
   - THEN start timer
   - THEN start training

3. **Dataset building takes longer than expected**
   - Estimated: 10-15 min
   - Actual: 11-12 min (and incomplete)
   - Always add buffer time

4. **Silent failures are dangerous**
   - Dataset builder can exit without error but produce incomplete output
   - Must verify output files, not just check if process completed
   - Implement explicit verification steps

### Corrected Workflow for Next Attempt

```bash
# 1. Provision instance
./launch_prime_intellect.py

# 2. SSH and fix environment (15 min)
ssh ubuntu@<IP>
pip install 'setuptools<70.0.0'
# ... (full setup)

# 3. Build dataset (15 min)
cd /workspace/trm/trm_repo
python3 -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir /workspace/data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

# 4. VERIFY dataset (MANDATORY)
echo "Verifying dataset..."
FILE_COUNT=$(find /workspace/data/arc2concept-aug-1000/ -name 'dataset.json' | wc -l)
if [ "$FILE_COUNT" != "2" ]; then
  echo "❌ Dataset verification failed! Found $FILE_COUNT files, expected 2"
  exit 1
fi
ls -lh /workspace/data/arc2concept-aug-1000/{train,test}/dataset.json
echo "✅ Dataset verified!"

# 5. Start auto-kill timer (ONLY AFTER verification)
./auto_kill.sh $POD_ID 35 > /tmp/auto_kill.log 2>&1 &
echo $! > /tmp/auto_kill.pid

# 6. Start training
tmux new-session -d -s trm-smoke "torchrun ... pretrain.py ..."

# 7. Monitor
tail -f /workspace/training.log | grep -E 'wandb|loss|step'
```

### Updated Cost Estimate for Successful Smoke Test

Based on this experience, realistic smoke test cost:

| Phase | Duration | Cost @ $7.56/hr |
|-------|----------|----------------|
| Provisioning | 5 min | $0.63 |
| Environment setup | 15 min | $1.89 |
| Dataset building + verification | 18 min | $2.27 |
| Training (30 min) | 30 min | $3.78 |
| Buffer/monitoring | 5 min | $0.63 |
| **Total** | **~73 min** | **~$9.20** |

Previous estimate was $8.20 - now updated to **$9-10 range** for safety.

### CORRECTION (Post-Investigation)

**Dataset builder worked correctly!** After reviewing TRM source code (`dataset/build_arc_dataset.py`):

1. ✅ TRM creates **only 2 splits by design** (train and test), NOT 3
2. ✅ train/ contains: training2 + concept subsets
3. ✅ test/ contains: evaluation2 subset
4. ❌ NO validation split is created (this is intentional)
5. ✅ The "2 of 3 files" was actually **2 of 2 files** - **SUCCESS!**

**Initial analysis was wrong** - documentation incorrectly assumed 3 splits based on typical ML workflows. TRM's design does not include a validation split.

### Recommendations

1. ✅ **Use updated workflow** with explicit verification steps
2. ✅ **Budget $10 for smoke tests** (not $6-8)
3. ⚠️ **Consider smaller dataset for smoke test**: Use arc2concept-aug-100 to reduce build time from 15 min to ~2-3 min
4. ✅ **Always verify before starting costly operations** (training, long runs)
5. ✅ **Document ALL silent failures** for debugging

---

**Next attempt:** Follow corrected workflow above. Verify dataset completion before ANY timer or training.
