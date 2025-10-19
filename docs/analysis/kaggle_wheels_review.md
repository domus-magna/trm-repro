# Kaggle Offline Wheels Build Attempts

During the ARC-AGI-2 reproduction we iterated on several Kaggle “offline wheels” builds to supply dependencies inside the no-internet notebook environment. The directories removed in this cleanup (`artifacts/kaggle_kernel_build_adam*`, `artifacts/kaggle_kernel_inference*`) captured intermediate work products from those attempts.

## Summary of Attempts
- **v1–v3**: Initial scaffolding notebooks that packaged `adam-atan2` sources but failed due to missing binary wheels for `numpy==2.0.2` and `pydantic-core==2.20.1`. Pip reported “No matching distribution found,” blocking installation.
- **v4–v6**: Added source tarballs for dependencies, but `adam-atan2==0.0.3` still failed to build because of CUDA toolkit paths on Kaggle (missing `CUDA_HOME`). These runs also suffered from the same numpy/pydantic gaps.
- **v7–v9**: Ensured CUDA env vars were set; wheels built successfully in isolation, but the resulting artifact bundle was superseded by the curated dataset `seconds0/trm-offline-wheels-py311`, which includes vetted wheels plus the TRM repo snapshot.
- **Inference kernels**: `kaggle_kernel_inference*` directories corresponded to early notebook exports. The final, validated script lives at `kaggle/trm_arc2_inference_notebook.py`, rendering the intermediate kernels obsolete.

## Decision
- Archive all intermediate wheel-build directories as they add noise without aiding reproducibility.
- Retain only `artifacts/kaggle_support/kaggle_dataset_trm_offline_wheels_v2`, which matches the published Kaggle dataset and has been smoke-tested.
- Document future updates in `docs/repro/reproduction_guide.md` if the Kaggle packaging needs revision.

This note remains in `docs/analysis/` as historical context; the raw notebooks/logs can be restored from git history when needed.
