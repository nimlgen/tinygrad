#!/bin/bash
set -euo pipefail
cd /home/nimlgen/k31-update
ram=/home/nimlgen/kimi-prefetch-study/model-ram/k31-update
fork=/home/nimlgen/k3q/atomic-llama
py=/home/nimlgen/k3q/venv/bin/python
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=1
"$py" -u convert_trunk.py
"$fork/build/bin/llama-quantize" --tensor-type indexer=f16 "$ram/trunk-bf16.gguf" "$ram/trunk-q4_k.gguf" Q4_K_M 32
"$py" trunk_names.py "$ram/trunk-q4_k.gguf" "$ram/Kimi-K3.1-Pro-trunk-q4_k-pr.gguf" "$fork/gguf-py"
"$py" - <<'PY'
import hashlib, json, pathlib, sys
sys.path.insert(0, '/home/nimlgen/k3q/atomic-llama/gguf-py')
import gguf
p = pathlib.Path('/home/nimlgen/kimi-prefetch-study/model-ram/k31-update/Kimi-K3.1-Pro-trunk-q4_k-pr.gguf')
r = gguf.GGUFReader(p)
indexers = [t for t in r.tensors if '.indexer.' in t.name]
assert len(indexers) == 120, len(indexers)
assert r.fields['kimi-k3.attention.indexer.top_k'].contents() == 2048
assert r.fields['kimi-k3.attention.indexer.kpool'].contents() == 1
with p.open('rb') as f: sha = hashlib.file_digest(f, 'sha256').hexdigest()
pathlib.Path('trunk-complete.json').write_text(json.dumps({'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha,
 'indexer_tensors': len(indexers), 'tensors': len(r.tensors)}))
print('TRUNK COMPLETE', sha, flush=True)
PY
