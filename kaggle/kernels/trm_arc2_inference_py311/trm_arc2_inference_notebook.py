# %% [markdown]
"""
# TRM ARC-AGI-2 Inference Notebook

Paper-faithful Tiny Recursive Model (TRM) inference on the ARC Prize 2025 evaluation set.

**What this notebook does**

1. Installs prepackaged dependencies from the attached wheels dataset (no internet).
2. Unpacks a clean snapshot of the TinyRecursiveModels repo.
3. Builds the ARC evaluation dataset in TRM format (no augmentation).
4. Loads `model.ckpt` from `seconds0/trm-offline-wheels-py311`.
5. Runs the ARC evaluator to produce `submission.json` for leaderboard submission.

**Before you run**

- Attach these datasets:
  - `seconds0/trm-offline-wheels-py311`
  - `seconds0/trm-repo-clean`
  - `arc-prize-2025` (competition data)
- Switch the runtime to **GPU** (Settings → Accelerator → GPU).
- Execution order matters: run each cell sequentially.
"""

# %% [markdown]
"""
## 0. Offline bootstrap (install wheels & unpack repo)
"""

# %%
import os
import shutil
from pathlib import Path
import subprocess
import sys

import numpy as np

try:
    import torch.distributed as dist
except ImportError:
    dist = None

from packaging.version import Version, InvalidVersion

INPUT_ROOT = Path("/kaggle/input")


def resolve_dataset(
    primary_slug: str,
    filename: str | None = None,
    aliases: tuple[str, ...] = (),
    display_slug: str | None = None,
) -> Path:
    """Return path to an attached dataset (optionally a file within it)."""
    candidate_slugs = (primary_slug, *aliases)
    for slug in candidate_slugs:
        base = INPUT_ROOT / slug
        target = base / filename if filename is not None else base
        if target.exists():
            return target
    attached = ", ".join(sorted(p.name for p in INPUT_ROOT.iterdir()))
    missing = display_slug or primary_slug
    msg = f"Attach dataset {missing}. Currently mounted: {attached or 'none'}"
    raise FileNotFoundError(msg)


WHEELS_DATASET = resolve_dataset(
    "trm-offline-wheels-py311",
    display_slug="seconds0/trm-offline-wheels-py311",
    aliases=("seconds0-trm-offline-wheels-py311", "trm-offline-wheels"),
)
if WHEELS_DATASET.is_file():
    wheels_extract_dir = Path("/kaggle/working/trm_offline_wheels")
    if wheels_extract_dir.exists():
        shutil.rmtree(wheels_extract_dir)
    wheels_extract_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(WHEELS_DATASET, wheels_extract_dir)
    WHEELS = wheels_extract_dir
else:
    WHEELS = WHEELS_DATASET
REPO_DATASET = resolve_dataset(
    "trm-repo-clean",
    display_slug="seconds0/trm-repo-clean",
    aliases=("seconds0-trm-repo-clean",),
)
REPO_ZIP = REPO_DATASET / "TinyRecursiveModels_clean.zip"
REPO_DIR_CANDIDATE = REPO_DATASET / "TinyRecursiveModels_clean"
REPO_FROM_DIR = False
if not REPO_ZIP.exists():
    if REPO_DIR_CANDIDATE.exists():
        REPO_FROM_DIR = True
    else:
        zip_candidates = sorted(REPO_DATASET.glob("**/*.zip"))
        if len(zip_candidates) == 1:
            REPO_ZIP = zip_candidates[0]
        elif not zip_candidates:
            listing = "\n".join(sorted(p.name for p in REPO_DATASET.iterdir()))
            raise FileNotFoundError(
                f"No zip archive found in {REPO_DATASET}. Expected TinyRecursiveModels_clean.zip.\n"
                f"Contents: {listing or '[empty]'}"
            )
        else:
            names = ", ".join(str(z.relative_to(REPO_DATASET)) for z in zip_candidates)
            raise FileNotFoundError(
                f"Multiple zip archives found in {REPO_DATASET}: {names}. "
                "Set REPO_ZIP manually in the first cell."
            )
REPO_DIR = Path("/kaggle/working/TinyRecursiveModels")

try:
    import importlib.metadata as metadata
except ImportError:  # Python <3.8 fallback (not expected on Kaggle)
    import importlib_metadata as metadata  # type: ignore


def is_requirement_satisfied(requirement: str) -> bool:
    """Return True if the exact requirement is already installed."""
    name, _, version = requirement.partition("==")
    try:
        installed_version = metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    if not version:
        return True
    try:
        return Version(installed_version) >= Version(version)
    except InvalidVersion:
        return installed_version == version


def find_distribution(dist_name: str, version: str) -> Path | None:
    """Best-effort locate an offline wheel or sdist for dist_name==version."""
    normalized = dist_name.replace("-", "_")
    search_patterns = [
        f"{dist_name}-{version}-*.whl",
        f"{normalized}-{version}-*.whl",
        f"{dist_name}-{version}.whl",
        f"{normalized}-{version}.whl",
        f"{dist_name}-{version}.tar.gz",
        f"{normalized}-{version}.tar.gz",
        f"{dist_name}-{version}.zip",
        f"{normalized}-{version}.zip",
    ]
    for pattern in search_patterns:
        matches = list(WHEELS.rglob(pattern))
        if matches:
            return matches[0]
    dir_patterns = [
        f"{dist_name}-{version}",
        f"{normalized}-{version}",
    ]
    for pattern in dir_patterns:
        matches = list(WHEELS.rglob(pattern))
        dirs = [m for m in matches if m.is_dir()]
        if dirs:
            setup_dirs = [
                d for d in dirs
                if (d / "setup.py").exists()
                or (d / "pyproject.toml").exists()
                or (d / "setup.cfg").exists()
            ]
            if setup_dirs:
                setup_dirs.sort(key=lambda p: (len(p.parts), str(p)))
                return setup_dirs[0]
            dirs.sort(key=lambda p: (len(p.parts), str(p)))
            return dirs[0]
    return None


def pip_install_requirement(requirement: str, *, mandatory: bool = False, env: dict | None = None, extra_args: list[str] | None = None) -> None:
    """Install requirement from offline wheels, optionally skipping if missing."""
    if is_requirement_satisfied(requirement):
        print(f"{requirement} already satisfied; skipping install.")
        return
    name, _, version = requirement.partition("==")
    location = find_distribution(name, version)
    if location is None:
        message = f"[WARN] {requirement} not found in {WHEELS}; skipping install."
        if mandatory:
            available = []
            if WHEELS.exists():
                try:
                    # Show a handful of available artifacts to aid debugging
                    available = sorted(
                        str(p.relative_to(WHEELS))
                        for p in WHEELS.rglob("*.*")
                        if p.is_file()
                    )[:20]
                except Exception:
                    available = []
            raise FileNotFoundError(
                f"Missing offline artifact for {requirement} in {WHEELS}.\n"
                f"Sample contents: {available}"
            )
        print(message)
        return

    cmd: list[str]
    if location.is_dir():
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            str(location),
        ]
    elif location.suffix == ".whl":
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(WHEELS),
            requirement,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            str(location),
        ]
    if extra_args:
        cmd.extend(extra_args)
    print("Installing", requirement, "from", location.name)
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        if mandatory:
            raise
        print(f"[WARN] Installation failed for {requirement}: {exc}. Continuing with existing environment.")


pip_install_requirement("hydra-core==1.3.2")
pip_install_requirement("omegaconf==2.3.0")
pip_install_requirement("annotated-types==0.7.0")
pip_install_requirement("pydantic-core==2.20.1")
pip_install_requirement("pydantic==2.8.2")
pip_install_requirement("typing-extensions==4.15.0")
pip_install_requirement("typing-inspection==0.4.0")
pip_install_requirement("pydantic-settings==2.4.0")
pip_install_requirement("python-dotenv==1.0.1")
pip_install_requirement("argdantic==1.3.3")
pip_install_requirement("coolname==2.2.0")
pip_install_requirement("einops==0.8.0")
pip_install_requirement("numba==0.60.0")
pip_install_requirement("llvmlite==0.43.0")
pip_install_requirement("antlr4-python3-runtime==4.9.3")
try:
    pip_install_requirement(
        "adam-atan2==0.0.3",
        mandatory=True,
    )
except Exception as exc:
    # Fallback: provide a stub optimizer so evaluation can proceed without the extension
    import types
    import torch.optim as optim

    print("[WARN] adam-atan2 installation failed:", exc)
    print("[WARN] Falling back to torch.optim.Adam as AdamATan2 stub (evaluation only).")
    adam_stub = types.ModuleType("adam_atan2")

    class AdamATan2(optim.Adam):
        pass

    adam_stub.AdamATan2 = AdamATan2
    sys.modules["adam_atan2"] = adam_stub

if REPO_FROM_DIR:
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    shutil.copytree(REPO_DIR_CANDIDATE, REPO_DIR)
    print("Repository copied from dataset directory to", REPO_DIR)
else:
    REPO_DIR.mkdir(exist_ok=True)
    shutil.unpack_archive(REPO_ZIP, REPO_DIR, format="zip")
    print("Repository unpacked from", REPO_ZIP.name, "to", REPO_DIR)

if not (REPO_DIR / "dataset").exists():
    nested_candidates = [
        d for d in REPO_DIR.iterdir()
        if d.is_dir() and d.name.lower().startswith("tinyrecursivemodels")
    ]
    for nested in nested_candidates:
        print("Flattening nested repo directory", nested.name)
        for item in nested.iterdir():
            destination = REPO_DIR / item.name
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(item), REPO_DIR)
        shutil.rmtree(nested)
        if (REPO_DIR / "dataset").exists():
            break

if not (REPO_DIR / "dataset").exists():
    raise FileNotFoundError(f"dataset/ not found in {REPO_DIR}. Check the repo dataset attachment.")

print("Repo root entries:", sorted(p.name for p in REPO_DIR.iterdir())[:12])

# %% [markdown]
"""
## 1. Build ARC evaluation dataset
Converts competition JSON into TRM's expected format (no augmentation).
"""

os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.append(str(REPO_DIR))

DATA_DIR = Path("/kaggle/working/arc_dataset")
DATA_DIR.mkdir(exist_ok=True)
ARC_PREFIX = "/kaggle/input/arc-prize-2025/arc-agi"

print("Building evaluation dataset …")
subprocess.run(
    [
        "python3",
        str(REPO_DIR / "dataset/build_arc_dataset.py"),
        "--input-file-prefix",
        ARC_PREFIX,
        "--output-dir",
        str(DATA_DIR),
        "--subsets",
        "evaluation",
        "--test-set-name",
        "evaluation",
        "--num-aug",
        "0",
    ],
    check=True,
    env={**os.environ, "PYTHONPATH": str(REPO_DIR)},
)

# %% [markdown]
"""
## 2. Load TRM components and checkpoint
"""

# %%
import json
import torch
from torch.utils.data import DataLoader

try:
    from TinyRecursiveModels.dataset.common import PuzzleDatasetMetadata
    from TinyRecursiveModels.pretrain import (
        ArchConfig,
        EvaluatorConfig,
        LossConfig,
        PretrainConfig,
        TrainState,
        create_evaluators,
        create_model,
        evaluate,
    )
    from TinyRecursiveModels.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
except ModuleNotFoundError:
    from dataset.common import PuzzleDatasetMetadata
    from pretrain import (
        ArchConfig,
        EvaluatorConfig,
        LossConfig,
        PretrainConfig,
        TrainState,
        create_evaluators,
        create_model,
        evaluate,
    )
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

if not torch.cuda.is_available():
    raise RuntimeError("GPU runtime required. Enable GPU for this notebook.")

if dist is not None and dist.is_available() and not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29400")
    dist.init_process_group("gloo", rank=0, world_size=1)

CHECKPOINT_PATH = WHEELS / "model.ckpt"
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"model.ckpt missing from {WHEELS}. "
        "Re-attach seconds0/trm-offline-wheels-py311 or refresh dataset version."
    )

checkpoint_state = torch.load(CHECKPOINT_PATH, map_location="cpu")
puzzle_vocab_size = checkpoint_state["model.inner.puzzle_emb.weights"].shape[0]
del checkpoint_state
EVAL_SAVE_DIR = Path("/kaggle/working/trm_eval_outputs")
EVAL_SAVE_DIR.mkdir(exist_ok=True)

loss_cfg = LossConfig(
    name="losses@ACTLossHead",
    loss_type="stablemax_cross_entropy",
)
arch_cfg = ArchConfig(
    name="recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1",
    loss=loss_cfg,
    mlp_t=False,
    H_cycles=3,
    L_cycles=4,
    H_layers=0,
    L_layers=2,
    hidden_size=512,
    num_heads=8,
    expansion=4,
    puzzle_emb_ndim=512,
    puzzle_emb_len=16,
    forward_dtype="float32",
    pos_encodings="rope",
    halt_max_steps=16,
    halt_exploration_prob=0.1,
    no_ACT_continue=True,
)
eval_cfg = EvaluatorConfig(
    name="arc@ARC",
    submission_K=2,
    pass_Ks=[1, 2, 5, 10, 100, 1000],
    aggregated_voting=True,
)

cfg = PretrainConfig(
    arch=arch_cfg,
    data_paths=[str(DATA_DIR)],
    data_paths_test=[str(DATA_DIR)],
    evaluators=[eval_cfg],
    global_batch_size=64,
    epochs=1,
    lr=1e-4,
    lr_min_ratio=1.0,
    lr_warmup_steps=1,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    puzzle_emb_lr=0.01,
    puzzle_emb_weight_decay=0.1,
    project_name="Arc2concept-aug-1000-ACT-torch",
    run_name="trm_arc2_8gpu_eval100",
    load_checkpoint=str(CHECKPOINT_PATH),
    checkpoint_path=str(EVAL_SAVE_DIR),
    checkpoint_every_eval=False,
    eval_interval=1,
    min_eval_interval=0,
    eval_save_outputs=[],
    ema=True,
    ema_rate=0.999,
    freeze_weights=False,
    seed=0,
)

train_dataset = PuzzleDataset(
    PuzzleDatasetConfig(
        seed=cfg.seed,
        dataset_paths=cfg.data_paths,
        global_batch_size=cfg.global_batch_size,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    ),
    split="train",
)
train_metadata: PuzzleDatasetMetadata = train_dataset.metadata
train_metadata.num_puzzle_identifiers = puzzle_vocab_size
del train_dataset

eval_dataset = PuzzleDataset(
    PuzzleDatasetConfig(
        seed=cfg.seed,
        dataset_paths=cfg.data_paths_test or cfg.data_paths,
        global_batch_size=cfg.global_batch_size,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    ),
    split="test",
)
eval_loader = DataLoader(eval_dataset, batch_size=None)

model, optimizers, optimizer_lrs = create_model(cfg, train_metadata, rank=0, world_size=1)
model.eval()

train_state = TrainState(
    model=model,
    optimizers=optimizers,
    optimizer_lrs=optimizer_lrs,
    carry=None,
    step=72385,
    total_steps=0,
)

evaluators = create_evaluators(cfg, eval_dataset.metadata)
metrics = evaluate(
    cfg,
    train_state,
    eval_loader,
    eval_dataset.metadata,
    evaluators,
    rank=0,
    world_size=1,
    cpu_group=None,
)

submission_dirs = sorted(EVAL_SAVE_DIR.glob("evaluator_ARC_step_*"))
if not submission_dirs:
    raise FileNotFoundError("Expected evaluator output not found.")
submission_path = submission_dirs[-1] / "submission.json"
if not submission_path.exists():
    raise FileNotFoundError(f"{submission_path} missing.")

shutil.copy(submission_path, "/kaggle/working/submission.json")

print("Saved submission:", submission_path)
print("Copied to /kaggle/working/submission.json")
print("Evaluator metrics:")
def _json_default(obj):
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    return obj

print(json.dumps(metrics, indent=2, default=_json_default))

# %% [markdown]
"""
## 3. (Optional) Submit to ARC Prize 2025
Uncomment the line below when you’re ready to submit.
"""

# %%
# !kaggle competitions submit -c arc-prize-2025 \
#     -f /kaggle/working/submission.json \
#     -m "TRM ARC-AGI-2 checkpoint inference"
