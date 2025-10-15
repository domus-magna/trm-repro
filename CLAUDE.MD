# TRM (Tiny Recursive Models) — Paper-Faithful Reproduction on ARC-AGI-2 → Publish Weights to Kaggle

> We track work in Beads instead of Markdown. Run `bd quickstart` to see how.

## Key Instructions

**Single-agent playbook.** One agent performs the entire workflow end-to-end.

**Scope:** Reproduce upstream TRM training **as written** on **ARC-AGI-2** and publish the final checkpoint as a **Kaggle Dataset** for use in a Kaggle Notebook.

**Critical invariants:**
- **TARGET_DATASET must be `arc2`** at all times.
- **No source edits** to TRM. Do not change hyperparameters beyond those explicitly documented for ARC-AGI-2 in upstream examples.
- **Do not mix** ARC-AGI-1 and ARC-AGI-2 inputs.

## Quick Reference

1. **Global configuration** - Set environment variables for the run
2. **System sanity** - Check dependencies and create directories
3. **Clone TRM** - Get upstream repository (no modifications)
4. **Python environment** - Set up Python 3.10 virtual environment and dependencies
5. **Build ARC-AGI-2 dataset** - Use TRM's dataset builder
6. **Train TRM** - Run training with exact upstream flags
7. **Verify artifacts** - Check checkpoint and record provenance
8. **Publish to Kaggle** - Create/version Kaggle Dataset (optional)

## Workflow Commands

See AGENTS.MD for the complete step-by-step workflow with exact commands and flags.
