"""Reconstruct complete expert matrices from GPU parts and compare with the prior GGUF quantizer."""
import json, pathlib, sys
import numpy as np

layer = int(sys.argv[1])
base = pathlib.Path('/home/nimlgen/k31-update' if layer <= 52 else '/home/nimlgen/kimi-prefetch-study/model-ram/k31-update') / 'experts'
meta = json.loads((base / f'L{layer:03}.json').read_text())
h = meta['header']
sys.argv = ['quant_experts.py', '/raid/downloads/nimlgen/Kimi-K3.1-Pro', '/dev/null', 'IQ3_XXS', '/home/nimlgen/k3q/imatrix.dat', '0']
legacy = {}
src = pathlib.Path('/home/nimlgen/k3q/quant_experts.py').read_text().split('w = gguf.GGUFWriter')[0]
exec(compile(src, '/home/nimlgen/k3q/quant_experts.py', 'exec'), legacy)
from gguf.quants import dequantize

for expert in [0, h['n_experts'] // 2, h['n_experts'] - 1]:
  gates, ups, downs = [], [], []
  for g, (units, size) in enumerate(zip(h['units'], h['part_bytes'])):
    with (base / f'L{layer:03}.g{g}.bin').open('rb') as f:
      f.seek(expert * size)
      raw = f.read(size)
    row_up, row_down = h['lat'] // 256 * 98, units // 256 * 98
    gu, dn = units * row_up, h['lat'] * row_down
    gates.append(np.frombuffer(raw[:gu], np.uint8).reshape(units, row_up))
    ups.append(np.frombuffer(raw[gu:gu*2], np.uint8).reshape(units, row_up))
    downs.append(np.frombuffer(raw[gu*2:gu*2+dn], np.uint8).reshape(h['lat'], row_down))
    assert not any(raw[gu*2+dn:])
  arrays = [np.concatenate(gates), np.concatenate(ups), np.concatenate(downs, axis=1)]
  for array, (hf_name, gg_name, rows, cols) in zip(arrays, legacy['MATS']):
    expected = legacy['quant_expert'](layer, hf_name, gg_name, expert).reshape(rows, -1)
    np.testing.assert_array_equal(array, expected)
    prefix = f'language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.{hf_name}.'
    original = legacy['dequant'](legacy['st'](prefix + 'weight_packed'), legacy['st'](prefix + 'weight_scale'))
    restored = dequantize(array, legacy['QT']).reshape(rows, cols)
    assert np.isfinite(restored).all()
    relative = np.linalg.norm(restored - original) / np.linalg.norm(original)
    print(f'layer={layer} expert={expert} matrix={hf_name}: exact packed bytes; relative quantization error={relative:.5f}', flush=True)
print('PASS: 3 experts x 3 matrices, every GPU split and padding checked against legacy quantizer', flush=True)
