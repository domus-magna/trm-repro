# Final QA Checklist (Pre-Open-Source)

Date: 2025-10-19

## Completed
- ✅ Verified repository tree after cleanup (`ls -1`) — top-level now limited to source, docs, artifacts, infra, scripts.
- ✅ Confirmed Kaggle datasets and offline wheels moved under `artifacts/`.
- ✅ Added MIT `LICENSE` and ensured acknowledgements present in `README.md` & Hugging Face model card.
- ✅ Created documentation scaffolding (`docs/release`, `docs/repro`, `docs/ops`, `docs/analysis`).
- ✅ Ran `rg -n "TODO"` to locate remaining TODOs (isolated inside artifact wheel sources and draft docs).
- ✅ Checked `git status -sb` to understand move/rename impact (new directories untracked pending commit).

## Follow-Up Before Release
- [x] Review `artifacts/kaggle_kernel_build_adam_v*/` wheel trees—removed prototypes; see docs/analysis/kaggle_wheels_review.md.
- [ ] Address draft TODO in `docs/drafts/trm-full-training-guide.md` or mark as WIP.
- [ ] Update `.gitignore` to exclude `.secrets/` and other private files.
- [ ] Run markdown lint / link check (optional).
- [ ] Finalize CONTRIBUTING.md and CODE_OF_CONDUCT.md (placeholders noted in documentation plan).
