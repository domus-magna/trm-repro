#!/usr/bin/env bash
set -euo pipefail
echo "[dataset] starting builder"
apt-get update -y >/dev/null && apt-get install -y --no-install-recommends git ca-certificates >/dev/null && rm -rf /var/lib/apt/lists/*
python3 -V
python3 -m pip install --upgrade pip wheel setuptools >/dev/null
python3 -m pip install numpy pydantic argdantic >/dev/null
cd /workspace
if [ -d TinyRecursiveModels ] && [ ! -d TinyRecursiveModels/.git ]; then
  rm -rf TinyRecursiveModels
fi
if [ ! -d TinyRecursiveModels/.git ]; then
  git clone --depth=1 https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git TinyRecursiveModels
fi
cd TinyRecursiveModels
echo "[dataset] using prebuilt combined mapping JSONs in repo"
ls -la kaggle/combined | sed -n '1,80p'
echo "[dataset] sanitizing evaluation2 solutions to match challenges"
python3 - <<'PYSANITIZE'
import json
from pathlib import Path
base=Path('kaggle/combined')
challenges=json.loads((base/'arc-agi_evaluation2_challenges.json').read_text())
solutions=json.loads((base/'arc-agi_evaluation2_solutions.json').read_text())
affected=[]; fixed=0; deduped=0
for pid, ch in challenges.items():
    if pid not in solutions: 
        continue
    tests=ch.get('test',[])
    sols=solutions[pid]
    seen=set(); dedup=[]
    for s in sols:
        k=json.dumps(s, sort_keys=True)
        if k in seen: deduped+=1; continue
        seen.add(k); dedup.append(s)
    sols=dedup
    if len(sols)>len(tests):
        affected.append((pid,len(tests),len(sols)))
        sols=sols[:len(tests)]; fixed+=1
    solutions[pid]=sols
(base/'arc-agi_evaluation2_solutions.json').write_text(json.dumps(solutions,ensure_ascii=False,indent=2))
print('Sanitized:', 'deduped=',deduped,' truncated=',fixed,' puzzles=',len(affected))
PYSANITIZE
echo "[dataset] building TRM dataset"
python3 -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2
echo "[dataset] fingerprint"
python3 - <<'PY'
from pathlib import Path
import json, hashlib
base=Path('data/arc2concept-aug-1000')
counts={}
for split in ['train','test']:
    p=base/split/'puzzles.json'
    if p.exists():
        data=p.read_bytes()
        counts[split]=len(json.loads(data))
        (base/f'{split}.sha256').write_text(hashlib.sha256(data).hexdigest())
print(json.dumps({'counts':counts},indent=2))
PY
echo "[dataset] done"
