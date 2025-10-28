# %% [code]
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
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import hashlib

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
try:
    WEIGHTS_DATASET = resolve_dataset(
        "trm-arc2-weights-trm-arc2-8gpu-eval100",
        display_slug="seconds0/trm-arc2-weights-trm-arc2-8gpu-eval100",
        aliases=(
            "seconds0-trm-arc2-weights-trm-arc2-8gpu-eval100",
            "trm-arc2-weights-trm_arc2_8gpu_eval100",
            "trm-arc2-weights-trm-arc2-8gpu-resume",
            "trm-arc2-weights",
        ),
    )
except FileNotFoundError:
    print("[WARN] Checkpoint dataset not attached; falling back to wheels dataset for weights.")
    WEIGHTS_DATASET = WHEELS

CHECKPOINT_MANIFEST = WEIGHTS_DATASET / "MANIFEST.txt"
CHECKPOINT_STEP = 72385
if CHECKPOINT_MANIFEST.exists():
    manifest: dict[str, str] = {}
    for line in CHECKPOINT_MANIFEST.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        manifest[key.strip()] = value.strip()
    if "CHECKPOINT_STEP" in manifest:
        try:
            CHECKPOINT_STEP = int(manifest["CHECKPOINT_STEP"])
        except ValueError:
            print(f"[WARN] Invalid CHECKPOINT_STEP in {CHECKPOINT_MANIFEST}: {manifest['CHECKPOINT_STEP']}")
    print(f"Checkpoint manifest: step={CHECKPOINT_STEP}")

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
ARC_DATA_ROOT = Path("/kaggle/input/arc-prize-2025")
ARC_PREFIX = str(ARC_DATA_ROOT / "arc-agi")
ARC_SUBSET = "test"

print("Building test dataset …")
subprocess.run(
    [
        "python3",
        str(REPO_DIR / "dataset/build_arc_dataset.py"),
        "--input-file-prefix",
        ARC_PREFIX,
        "--output-dir",
        str(DATA_DIR),
        "--subsets",
        ARC_SUBSET,
        "--test-set-name",
        ARC_SUBSET,
        "--num-aug",
        "0",
    ],
    check=True,
    env={**os.environ, "PYTHONPATH": str(REPO_DIR)},
)

# %% [markdown]
"""
### 1.1 Dataset validators
Confirm the build targets the ARC competition evaluation split (120 puzzles).
"""

# %%
with open(DATA_DIR / "test_puzzles.json") as f:
    test_puzzles = json.load(f)

if isinstance(test_puzzles, dict):
    puzzle_ids = sorted(test_puzzles.keys())
else:
    puzzle_ids = sorted(str(pid) for pid in test_puzzles)

source_file = ARC_DATA_ROOT / f"arc-agi_{ARC_SUBSET}_challenges.json"
if not source_file.exists():
    raise FileNotFoundError(
        f"{source_file.name} missing from competition data. "
        "Confirm that the official ARC Prize dataset is attached."
    )

with open(source_file) as f:
    evaluation_source = json.load(f)

expected_eval_puzzles = len(evaluation_source)

with open(DATA_DIR / "test" / "dataset.json") as f:
    dataset_meta = json.load(f)

if dataset_meta["total_puzzles"] != expected_eval_puzzles:
    raise RuntimeError(
        f"Unexpected evaluation puzzle count: {dataset_meta['total_puzzles']} (expected {expected_eval_puzzles}). "
        "Check that arc-prize-2025/arc-agi is attached instead of the sample dataset."
    )

source_ids = sorted(evaluation_source.keys())
if puzzle_ids != source_ids:
    missing = sorted(set(source_ids) - set(puzzle_ids))
    extra = sorted(set(puzzle_ids) - set(source_ids))
    raise RuntimeError(
        "Evaluation puzzle IDs do not match the competition file.\n"
        f"Missing IDs: {missing[:5]}\nExtra IDs: {extra[:5]}"
    )

id_hash = hashlib.sha256("\n".join(source_ids).encode("utf-8")).hexdigest()
print(
    f"Validators: {ARC_SUBSET} split confirmed "
    f"({expected_eval_puzzles} puzzles, SHA256={id_hash})."
)

# %% [markdown]
"""
## 2. Load TRM components and checkpoint
"""

# %%
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

CHECKPOINT_PATH = WEIGHTS_DATASET / "model.ckpt"
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"model.ckpt missing from {WEIGHTS_DATASET}. "
        "Attach seconds0/trm-arc2-weights-trm-arc2-8gpu-eval100 (or matching alias) "
        "alongside the offline wheels dataset."
    )

checkpoint_state = torch.load(CHECKPOINT_PATH, map_location="cpu")
def _resolve_puzzle_key(state_dict: dict[str, "torch.Tensor"]) -> str:
    primary = "model.inner.puzzle_emb.weights"
    alternate = "_orig_mod.model.inner.puzzle_emb.weights"
    if primary in state_dict:
        return primary
    if alternate in state_dict:
        return alternate
    raise KeyError(f"Puzzle embedding weights not found in checkpoint keys: {list(state_dict)[:5]}")

puzzle_emb_key = _resolve_puzzle_key(checkpoint_state)
puzzle_vocab_size = checkpoint_state[puzzle_emb_key].shape[0]
ckpt_has_attention_keys = any("self_attn.qkv_proj.weight" in k for k in checkpoint_state)

def _strip_prefix(state_dict: dict[str, "torch.Tensor"], prefix: str = "_orig_mod.") -> dict[str, "torch.Tensor"]:
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items() }

normalized_checkpoint_state = _strip_prefix(checkpoint_state)
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
cfg.load_checkpoint = None  # manual load to support remapped checkpoint keys

arch_summary = {
    "name": arch_cfg.name,
    "mlp_t": arch_cfg.mlp_t,
    "num_heads": arch_cfg.num_heads,
    "pos_encodings": arch_cfg.pos_encodings,
    "forward_dtype": arch_cfg.forward_dtype,
    "halt_max_steps": arch_cfg.halt_max_steps,
}
print("ARCH_CFG:", arch_summary)
print("CHECKPOINT_PATH:", CHECKPOINT_PATH)
print("CHECKPOINT_STEP:", CHECKPOINT_STEP)
print("CKPT_HAS_ATTENTION_KEYS:", ckpt_has_attention_keys)

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

incompatible = model.load_state_dict(normalized_checkpoint_state, strict=False)
if incompatible.missing_keys:
    print(f"[WARN] Missing keys when loading checkpoint: {sorted(incompatible.missing_keys)[:5]}")
if incompatible.unexpected_keys:
    print(f"[WARN] Unexpected keys when loading checkpoint: {sorted(incompatible.unexpected_keys)[:5]}")
del checkpoint_state
del normalized_checkpoint_state

inner_model = getattr(model, "model", None)
loss_head = getattr(model, "loss", None)
model_classes = {
    "model": type(model).__name__,
    "inner": type(inner_model).__name__ if inner_model is not None else "None",
    "loss_head": type(loss_head).__name__ if loss_head is not None else "None",
}
print("MODEL_CLASSES:", model_classes)

train_state = TrainState(
    model=model,
    optimizers=optimizers,
    optimizer_lrs=optimizer_lrs,
    carry=None,
    step=CHECKPOINT_STEP,
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
## 3. Housekeeping (keep only submission.json)
"""

# %%
cleanup_targets = [
    EVAL_SAVE_DIR,
    REPO_DIR,
    DATA_DIR,
    Path("/kaggle/working/trm_offline_wheels"),
]

for target in cleanup_targets:
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            try:
                target.unlink()
            except FileNotFoundError:
                pass

for leftover in Path("/kaggle/working").iterdir():
    if leftover.name in {"submission.json"}:
        continue
    if leftover.is_dir():
        shutil.rmtree(leftover, ignore_errors=True)
    else:
        try:
            leftover.unlink()
        except FileNotFoundError:
            pass

# %% [markdown]
