"""Convert verified HF shards into the k3_rebased IQ3_XXS per-GPU store layout.

Outputs are per layer so installation can resume and read back each whole layer.
Source checkpoints are retained. Existing K3 artifacts are never overwritten.
"""
import concurrent.futures, ctypes, hashlib, json, os, pathlib, shutil, struct, sys, time, zlib
import numpy as np

STATE = pathlib.Path('/home/nimlgen/k31-update')
HF = pathlib.Path('/raid/downloads/nimlgen/Kimi-K3.1-Pro')
RAM = pathlib.Path('/home/nimlgen/kimi-prefetch-study/model-ram/k31-update')
FORK = pathlib.Path('/home/nimlgen/k3q/atomic-llama')
sys.path.insert(0, str(FORK / 'gguf-py'))
import gguf

source = json.loads((STATE / 'source-manifest.json').read_text())
config = json.loads((HF / 'config.json').read_text())['text_config']
index = json.loads((HF / 'model.safetensors.index.json').read_text())['weight_map']
NE, LAT, FF = config['num_experts'], config['routed_expert_hidden_size'], config['moe_intermediate_size']
LAYERS = list(range(config['first_k_dense_replace'], config['num_hidden_layers']))
UNITS = [768] * 4
assert (NE, LAT, FF, LAYERS) == (896, 3584, 3072, list(range(1, 93)))
QT = gguf.GGMLQuantizationType.IQ3_XXS
BLK, TSZ = gguf.GGML_QUANT_SIZES[QT]
PARTS = [((3 * u * LAT // BLK * TSZ + 4095) // 4096) * 4096 for u in UNITS]
MATS = [('w1', 'ffn_gate_exps', FF, LAT), ('w3', 'ffn_up_exps', FF, LAT), ('w2', 'ffn_down_exps', LAT, FF)]
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
lib = ctypes.CDLL(str(FORK / 'build/bin/libggml-base.so'))
lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                  ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
lib.ggml_quantize_chunk.restype = ctypes.c_size_t
lib.ggml_quantize_init.argtypes = [ctypes.c_int]
lib.ggml_quantize_init(int(QT))
imreader = gguf.GGUFReader('/home/nimlgen/k3q/imatrix.dat')
imat = {t.name: np.asarray(t.data) for t in imreader.tensors}
header = {'model': 'kimi-k3.1-pro', 'expert_type': QT.name, 'block': [BLK, TSZ], 'units': UNITS, 'lat': LAT, 'ff': FF,
          'n_experts': NE, 'layers': LAYERS, 'part_bytes': PARTS,
          'source': {'repo': source['id'], 'revision': source['sha'], 'quantization': QT.name,
                     'imatrix_sha256': hashlib.file_digest(open('/home/nimlgen/k3q/imatrix.dat', 'rb'), 'sha256').hexdigest()}}
(STATE / 'store-header.json').write_text(json.dumps(header, indent=2))

def quant_expert(layer, expert, h, base, mm):
  mats, no_imatrix = [], 0
  for hf_name, gg_name, rows, cols in MATS:
    prefix = f'language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.{hf_name}.'
    arrays = []
    for suffix, shape in [('weight_packed', (rows, cols // 2)), ('weight_scale', (rows, cols // 32))]:
      t = h[prefix + suffix]
      assert t['dtype'] == 'U8' and tuple(t['shape']) == shape
      begin, end = t['data_offsets']
      arrays.append(mm[base + begin:base + end].reshape(shape))
    p, scale = arrays
    x = np.empty((rows, cols), np.float32)
    x[:, 0::2], x[:, 1::2] = E2M1[p & 15], E2M1[p >> 4]
    x.reshape(rows, -1, 32)[:] *= np.ldexp(np.float32(1), scale.astype(np.int32) - 127)[..., None]
    key = f'blk.{layer}.{gg_name}.weight'
    s, count = imat.get(key + '.in_sum2'), imat.get(key + '.counts')
    importance = None
    if s is not None and count is not None and count.reshape(-1)[expert] > 0:
      importance = np.ascontiguousarray(s.reshape(NE, -1)[expert] / count.reshape(-1)[expert], dtype=np.float32)
    else: no_imatrix += 1
    out = np.empty((rows, cols // BLK * TSZ), np.uint8)
    n = lib.ggml_quantize_chunk(int(QT), x.ctypes.data, out.ctypes.data, 0, rows, cols,
                               importance.ctypes.data if importance is not None else None)
    assert n == out.nbytes
    mats.append(out)
  gate, up, down = mats
  result, start = [], 0
  for units, size in zip(UNITS, PARTS):
    b0, b1 = start // BLK * TSZ, (start + units) // BLK * TSZ
    part = b''.join([gate[start:start + units].tobytes(), up[start:start + units].tobytes(), down[:, b0:b1].tobytes()])
    assert len(part) <= size
    result.append(part.ljust(size, b'\0'))
    start += units
  return result, no_imatrix

def ready_layers():
  pending = set(LAYERS)
  while pending:
    ready = [l for l in sorted(pending) if (HF / index[f'language_model.model.layers.{l}.block_sparse_moe.experts.0.w1.weight_packed']).exists()]
    if not ready:
      print(f'waiting for verified shards: {len(pending)} layers remain', flush=True)
      time.sleep(30)
    for layer in ready:
      yield layer
      pending.remove(layer)

if __name__ == '__main__':
  for layer in ready_layers():
    outdir = (STATE if layer <= 52 else RAM) / 'experts'
    outdir.mkdir(parents=True, exist_ok=True)
    meta = outdir / f'L{layer:03}.json'
    if meta.exists():
      prior = json.loads(meta.read_text())
      assert prior['header'] == header
      for g, size in enumerate(PARTS):
        p = outdir / f'L{layer:03}.g{g}.bin'
        assert p.stat().st_size == NE * size
        with p.open('rb') as f: assert hashlib.file_digest(f, 'sha256').hexdigest() == prior['sha256'][g]
      print(f'layer {layer}: existing output verified', flush=True)
      continue
    shards = {index[f'language_model.model.layers.{layer}.block_sparse_moe.experts.{e}.{name}.{suffix}']
              for e in range(NE) for name, _, _, _ in MATS for suffix in ['weight_packed', 'weight_scale']}
    assert len(shards) == 1
    shard = HF / shards.pop()
    print(f'layer {layer}: waiting for verified {shard.name}', flush=True)
    while not shard.exists(): time.sleep(15)
    assert shutil.disk_usage(outdir).free > NE * sum(PARTS) + (30 << 30)
    with shard.open('rb') as f:
      n = struct.unpack('<Q', f.read(8))[0]
      h, base = json.loads(f.read(n)), 8 + n
    mm = np.memmap(shard, dtype=np.uint8, mode='r')
    crcs, hashes, missing = [[] for _ in UNITS], [hashlib.sha256() for _ in UNITS], 0
    files = [(outdir / f'L{layer:03}.g{g}.partial').open('wb') for g in range(4)]
    started = time.monotonic()
    try:
      with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('K31_WORKERS', '112'))) as pool:
        # Bounded batches keep temporary arrays and completed parts below a few GB.
        for first in range(0, NE, 112):
          jobs = [pool.submit(quant_expert, layer, e, h, base, mm) for e in range(first, min(first + 112, NE))]
          for job in jobs:
            parts, absent = job.result()
            missing += absent
            for g, raw in enumerate(parts):
              files[g].write(raw)
              hashes[g].update(raw)
              crcs[g].append(zlib.crc32(raw))
          print(f'layer {layer}: {first + len(jobs)}/{NE} experts, {time.monotonic()-started:.1f}s', flush=True)
      for f in files: f.flush(); os.fsync(f.fileno())
    finally:
      for f in files: f.close()
    for g, size in enumerate(PARTS):
      p = outdir / f'L{layer:03}.g{g}.partial'
      assert p.stat().st_size == NE * size
      p.rename(outdir / f'L{layer:03}.g{g}.bin')
    info = {'layer': layer, 'header': header, 'crc': crcs, 'sha256': [h.hexdigest() for h in hashes], 'without_imatrix': missing}
    meta.with_suffix('.json.partial').write_text(json.dumps(info))
    meta.with_suffix('.json.partial').rename(meta)
    del mm
    print(f'layer {layer}: COMPLETE, {time.monotonic()-started:.1f}s, {missing} matrices without imatrix', flush=True)
  (STATE / 'quantize-complete.json').write_text(json.dumps(header))
