# Repository Audit – TRM ARC-AGI-2 Release Prep

Date: 2025-10-19  
Scope: Top-level inventory of the `trm-repro` workspace with recommended actions before open-sourcing.

## Legend
- **Keep** – required for public release with minimal changes.
- **Refine** – keep but needs cleanup (docs, renames, slim down).
- **Relocate** – move under a structured subdirectory (e.g., `artifacts/archives/`).
- **Drop/Ignore** – remove from repo or ensure `.gitignore` covers it.

## Directory & File Classification

### Core Source & Documentation
| Path | Status | Notes |
| --- | --- | --- |
| `TinyRecursiveModels/` | Keep | Upstream source clone; ensure pinned commit hash, no local modifications before release. |
| `kaggle/` | Refine | Contains inference notebook sources; document usage and ensure filenames standardized. |
| `docs/` | Refine | Organize into `docs/release/`, `docs/repro/`, migrate drafts or tag as WIP. |
| `huggingface_release/` | Refine | Promote to `releases/huggingface/` or similar, add README describing contents. |
| `scripts/` | Refine | Retain reusable automation; add descriptions or usage in README. |

### Experiment & Artifact Outputs
| Path | Status | Notes |
| --- | --- | --- |
| `artifacts/` | Refine | Contains multiple Kaggle helper archives; keep but add subfolders + index README. |
| `logs/` | Refine | Preserve W&B exports and dataset builder logs; slim to essential files and document. |
| `kaggle_dataset_8gpu_step_72385/` | Relocate | Move under `artifacts/checkpoints/` with README pointing to Kaggle/HF mirrors. |
| `kaggle_dataset_arc2_trm_arc2_4gpu_eval100/` | Relocate | Same treatment as above; maintain provenance file. |
| `kaggle_dataset_trm_offline_wheels_v2/` | Relocate | Place under `artifacts/kaggle_support/`; confirm redundant wheels trimmed. |
| `tmp_wheels/`, `tmp_wheels_py311/` | Drop/Ignore | Generated wheel staging areas; delete or add to `.gitignore`. |
| `kaggle (1).json` | Drop/Ignore | Likely stray Kaggle API token backup; remove before release. |

### Configuration / Deployment Assets
| Path | Status | Notes |
| --- | --- | --- |
| `run.sh`, `monitor_*.sh`, `smart_launcher.sh`, `monitor_and_launch.sh` | Refine | Evaluate which launch scripts remain relevant; document usage. |
| `trm-*.yaml`, `trm-*.cm.yaml` | Refine | Kubernetes configs; group under `infra/` or `deploy/` for clarity. |
| `backblaze_restore/`, `coreweave_support_ticket.md`, `infra.md` | Relocate | Archive under `docs/ops/` or `ops/` referencing operational processes. |

### Analyses & Drafts
| Path | Status | Notes |
| --- | --- | --- |
| `AGENTS.MD`, `CLAUDE.MD`, `.claude/`, `.cursor/` | Drop/Ignore | Internal agent transcripts; remove or keep private. |
| `HPC_VERIFICATION_PROBLEM_ANALYSIS.md`, `TRM_8GPU_EVALUATION_FAILURE_REPORT.md` | Refine | Keep as appendix in `docs/analysis/`; ensure sensitive info removed. |
| `docs/drafts/` | Refine | Review drafts, promote finalized content to `docs/release/` or `docs/repro/`. |

### Miscellaneous
| Path | Status | Notes |
| --- | --- | --- |
| `package.json`, `package-lock.json` | Refine | Confirm if Node tooling is still needed; otherwise remove. |
| `REQUIREMENTS.MD`, `aggregate_arc_inputs.py`, `sanitize_solutions.py`, `triage_solutions.py` | Refine | Document usage or move to `tools/`. |
| `.env`, `.k8s-*`, `.kube-config` | Drop/Ignore | Sensitive or environment-specific; ensure excluded via `.gitignore`. |

## Immediate Cleanup Recommendations
1. Delete or git-ignore temporary wheel directories and stray Kaggle credential files.
2. Consolidate artifacts under `artifacts/` with descriptive subdirs (`checkpoints/`, `kaggle_support/`, `ops/`).
3. Move Kubernetes manifests into `infra/` and add overview README.
4. Establish `docs/` structure (`release`, `repro`, `analysis`) and move relevant notes.
5. Prepare to rewrite top-level `README.md` highlighting repo layout, release links, acknowledgements.

This audit informs downstream cleanup tasks (trm-repro-31/32) before public release.
