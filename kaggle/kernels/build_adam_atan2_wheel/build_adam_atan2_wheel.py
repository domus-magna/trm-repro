"""
Kaggle script to build adam-atan2 wheel for Python 3.11 / CUDA 12.5.

Expected datasets:
  - seconds0/trm-offline-wheels (contains adam_atan2-0.0.3.tar.gz and helper wheels)
Outputs:
  - adam_atan2-0.0.3-cp311-cp311-manylinux2014_x86_64.whl
  - build_log.txt
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def main() -> None:
    wheels_root = Path("/kaggle/input/trm-offline-wheels-py311")
    if not wheels_root.exists():
        raise FileNotFoundError("Attach dataset seconds0/trm-offline-wheels.")

    output_dir = Path("/kaggle/working/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "build_log.txt"
    with log_path.open("w") as log_file:
        log_file.write("Kaggle adam-atan2 build log\n")

    run_dir = Path("/kaggle/working")
    sdist_candidates = sorted(wheels_root.glob("adam_atan2-0.0.3*.tar.gz"))
    if sdist_candidates:
        run(["tar", "xf", str(sdist_candidates[0]), "-C", str(run_dir)])
    else:
        source_dir = wheels_root / "adam_atan2-0.0.3"
        if not source_dir.exists():
            raise FileNotFoundError(
                "adam_atan2-0.0.3 source missing from wheels dataset (no tar.gz or directory)."
            )
        shutil.copytree(source_dir, run_dir / "adam_atan2-0.0.3", dirs_exist_ok=True)

    src_dir = run_dir / "adam_atan2-0.0.3"
    if not src_dir.exists():
        raise FileNotFoundError("Unpacked source directory not found.")

    nested_src = src_dir / "adam_atan2-0.0.3"
    if nested_src.exists() and not (src_dir / "setup.py").exists():
        for item in nested_src.iterdir():
            target = src_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

    setup_path = src_dir / "setup.py"
    text = setup_path.read_text()
    if '{"60", "75", "80", "86", "89", "90"}' not in text:
        text = text.replace(
            '{"80", "86", "89", "90"}',
            '{"60", "75", "80", "86", "89", "90"}',
        )
        setup_path.write_text(text)

    env = os.environ.copy()
    env.update(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "MAX_JOBS": "4",
            "TORCH_CUDA_ARCH_LIST": "6.0;7.5;8.0;8.6;8.9;9.0",
            "SETUPTOOLS_SCM_PRETEND_VERSION": "0.0.3",
            "USE_NINJA": "1",
        }
    )

    # Confirm torch availability
    run(
        [
            sys.executable,
            "-c",
            "import torch, setuptools, wheel; print('torch', torch.__version__)",
        ],
        env=env,
    )

    # Install build prerequisites from offline wheels
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            f"--find-links={wheels_root}",
            "setuptools==75.2.0",
            "setuptools-scm==8.1.0",
            "wheel==0.45.1",
            "ninja==1.11.1.1",
            "auditwheel==6.1.0",
            "pyelftools==0.32",
        ],
        env=env,
    )

    dist_dir = src_dir / "dist"
    if dist_dir.exists():
        shutil.rmtree(dist_dir)

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "-w",
            "dist",
            "--no-deps",
        ],
        cwd=src_dir,
        env=env,
    )

    built_wheels = sorted(dist_dir.glob("adam_atan2-0.0.3-*.whl"))
    if not built_wheels:
        raise FileNotFoundError("Wheel build succeeded but no wheel found.")

    local_wheel = built_wheels[-1]
    print("Built wheel:", local_wheel.name)

    wheel_path = local_wheel
    try:
        run(["auditwheel", "--version"], env=env)
        run(["auditwheel", "repair", str(local_wheel), "-w", str(dist_dir)], env=env)
        repaired_wheels = sorted(
            dist_dir.glob("adam_atan2-0.0.3*-manylinux*.whl"),
            key=lambda p: p.stat().st_mtime,
        )
        if repaired_wheels:
            wheel_path = repaired_wheels[-1]
            print("Repaired wheel:", wheel_path.name)
        else:
            print("[WARN] auditwheel repair ran but produced no manylinux wheel. Using original build.")
    except subprocess.CalledProcessError as exc:
        print("[WARN] auditwheel repair failed; using original linux_x86_64 wheel.", exc)

    shutil.copy2(wheel_path, output_dir / wheel_path.name)

    # Smoke test: install wheel and run minimal optimizer step
    run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheel_path)],
        env=env,
    )
    smoke_test = """
import sys
import torch
from adam_atan2 import AdamATan2

model = torch.nn.Linear(8, 4).cuda()
optim = AdamATan2(model.parameters(), lr=1e-3)
data = torch.randn(16, 8, device='cuda')
target = torch.randn(16, 4, device='cuda')
loss = torch.nn.functional.mse_loss(model(data), target)
loss.backward()
optim.step()
print('adam_atan2_backend present:', 'adam_atan2_backend' in sys.modules)
"""
    run([sys.executable, "-c", smoke_test], env=env)

    print("Wheel build and smoke test completed.")


if __name__ == "__main__":
    main()
