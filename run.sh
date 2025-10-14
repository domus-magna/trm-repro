#!/usr/bin/env bash
set -euo pipefail
apt-get update -y >/dev/null && apt-get install -y --no-install-recommends git ca-certificates >/dev/null && rm -rf /var/lib/apt/lists/*
python3 -V
pip install --upgrade pip wheel setuptools >/dev/null
pip install numpy pydantic argdantic >/dev/null
# Clone TRM into workspace if missing
if [ ! -d /workspace/TinyRecursiveModels/.git ]; then
  git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git /workspace/TinyRecursiveModels
fi
cd /workspace/TinyRecursiveModels
mkdir -p kaggle/combined/arc-agi/{training2,evaluation2,concept}
# Fetch ARC-AGI-2 and ConceptARC
tmp_arc2=$(mktemp -d); git clone --depth=1 https://github.com/arcprize/ARC-AGI-2.git "$tmp_arc2"
cp -v "$tmp_arc2/data/training"/*.json kaggle/combined/arc-agi/training2/ || true
cp -v "$tmp_arc2/data/evaluation"/*.json kaggle/combined/arc-agi/evaluation2/ || true
rm -rf "$tmp_arc2"
tmp_concept=$(mktemp -d); git clone --depth=1 https://github.com/victorvikram/ConceptARC.git "$tmp_concept"
find "$tmp_concept/corpus" -type f -name '*.json' -print0 | xargs -0 -I{} cp -v {} kaggle/combined/arc-agi/concept/ || true
rm -rf "$tmp_concept"
# Aggregate
python - <<'PY'
import json
from pathlib import Path
root = Path('/workspace/TinyRecursiveModels')
base = root / 'kaggle/combined/arc-agi'

def aggregate(src: Path):
    out = {}
    for p in src.rglob('*.json'):
        out[p.stem] = json.loads(p.read_text())
    print(src, '->', len(out), 'puzzles')
    return out

for subset in ['training2','evaluation2','concept']:
    data = aggregate(base / subset)
    (root / f'kaggle/combined/arc-agi_{subset}_challenges.json').write_text(json.dumps(data))
PY
# Ensure no conflicting solutions are present
mkdir -p kaggle/combined/_backup_solutions
for f in kaggle/combined/arc-agi_*_solutions.json; do
  [ -f "$f" ] && mv "$f" kaggle/combined/_backup_solutions/ || true
done
# Build dataset
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2
