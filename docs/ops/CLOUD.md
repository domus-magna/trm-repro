# Cloud Operations (CoreWeave) — Training Is Resuming

Status: We are resuming TRM training on CoreWeave. All README content is considered outdated; do not rely on READMEs for procedures or claims.

Authoritative artifacts to use going forward:

- Kubernetes manifests: `infra/kubernetes/`
- Persistent workspace PVC: managed in the `trm` namespace
- Checkpoints and logs: `huggingface_release/`, `artifacts/`, and job outputs
- Kaggle packaging: `kaggle/` (for inference-only workflows)

Operator notes:

- Kube access is configured via the repo kubeconfig in `infra/kubernetes/archive/kube-config.yaml`. Do not commit tokens or print them in logs.
- Preferred starting shape: single node, 4× or 8× H200, RWX PVC mounted at a stable path. Expand to multi-node only after stability checks.
- All new tasks and changes should be tracked in `.beads/` and committed alongside code when appropriate.

Do not treat README files as current; this page and `AGENTS.md` supersede them.

