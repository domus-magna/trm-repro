set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -y >/dev/null
apt-get install -y --no-install-recommends \
  python3 python3-distutils python3-dev python3-pip python3-venv \
  git ca-certificates build-essential curl >/dev/null
update-ca-certificates >/dev/null || true
python3 -V
python3 -m pip install --upgrade pip wheel setuptools >/dev/null
# Set CUDA env explicitly for extension builds
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
# Prepare TRM repo
if [ -d /workspace/TinyRecursiveModels/.git ]; then
  cd /workspace/TinyRecursiveModels && git remote set-url origin https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git || true
  git fetch --depth=1 origin main && git reset --hard FETCH_HEAD
else
  mkdir -p /workspace/TinyRecursiveModels
  cd /workspace/TinyRecursiveModels
  git init
  git remote add origin https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git || true
  git fetch --depth=1 origin main
  git reset --hard FETCH_HEAD
fi
# Install stable PyTorch cu121 + deps + wandb
python3 -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 >/dev/null
python3 -m pip install -r requirements.txt >/dev/null || true
python3 -m pip install --no-build-isolation adam-atan2 wandb >/dev/null
# Verify dataset exists
[ -d data/arc2concept-aug-1000 ] || { echo 'Dataset missing on RWX PVC'; ls -la data || true; exit 2; }
