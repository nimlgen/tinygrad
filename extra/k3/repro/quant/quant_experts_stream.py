# (stream variant: waits for each input shard to arrive) stream K3 routed experts: mxfp4 hf safetensors -> ggml quant type (with imatrix) into the second shard of a 2-shard split gguf
import sys, os, json, struct, ctypes, time, numpy as np, concurrent.futures as cf
FORK = "/home/nimlgen/k3q/atomic-llama"
sys.path.insert(0, f"{FORK}/gguf-py")
import gguf
hf, out, qname, imat_path, n_trunk = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
QT = gguf.GGMLQuantizationType[qname]
lib = ctypes.CDLL(f"{FORK}/build/bin/libggml-base.so")
lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
lib.ggml_quantize_chunk.restype = ctypes.c_size_t
lib.ggml_quantize_init.argtypes = [ctypes.c_int]
lib.ggml_quantize_init(int(QT))
BLK, TSZ = gguf.GGML_QUANT_SIZES[QT]

cfg = json.load(open(f"{hf}/config.json"))["text_config"]
NE, LAT, FF = cfg["num_experts"], cfg["routed_expert_hidden_size"], cfg["moe_intermediate_size"]
LAYERS = list(range(cfg["first_k_dense_replace"], cfg["num_hidden_layers"]))
MATS = [("w1", "ffn_gate_exps", FF, LAT), ("w3", "ffn_up_exps", FF, LAT), ("w2", "ffn_down_exps", LAT, FF)]  # hf name, gguf name, rows, n_per_row

idx, hdrs = json.load(open(f"{hf}/model.safetensors.index.json"))["weight_map"], {}
def st(name) -> np.ndarray:
  f = idx[name]
  while f not in hdrs and not os.path.exists(f"{hf}/{f}"): time.sleep(2)  # (streamed in by a fetcher, written as .part then renamed)
  if f not in hdrs:
    with open(f"{hf}/{f}", "rb") as fh: n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
    hdrs[f] = (h, 8 + n, np.memmap(f"{hf}/{f}", dtype=np.uint8, mode="r"))
  h, base, mm = hdrs[f]; s, e = h[name]["data_offsets"]
  return mm[base+s:base+e].reshape(h[name]["shape"])

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
def dequant(p:np.ndarray, s:np.ndarray) -> np.ndarray:  # packed e2m1 (low nibble first) + e8m0 per 32 -> f32
  v = np.empty((p.shape[0], p.shape[1] * 2), np.float32)
  v[:, 0::2], v[:, 1::2] = E2M1[p & 15], E2M1[p >> 4]
  return (v.reshape(p.shape[0], -1, 32) * np.ldexp(np.float32(1), s.astype(np.int32) - 127)[..., None]).reshape(v.shape)

imat = {t.name: np.asarray(t.data) for t in gguf.GGUFReader(imat_path).tensors}
def importance(gname:str, e:int) -> np.ndarray|None:  # mean squared activation per input column for expert e
  s, c = imat.get(gname + ".in_sum2"), imat.get(gname + ".counts")
  if s is None or (c := c.reshape(-1)[e]) <= 0: return None
  return np.ascontiguousarray(s.reshape(NE, -1)[e] / c, dtype=np.float32)

def quant_expert(L:int, hfname:str, gname:str, e:int) -> np.ndarray:
  p = f"language_model.model.layers.{L}.block_sparse_moe.experts.{e}.{hfname}."
  x = np.ascontiguousarray(dequant(st(p + "weight_packed"), st(p + "weight_scale")))
  rows, n = x.shape
  q, im = np.empty(rows * (n // BLK) * TSZ, np.uint8), importance(f"blk.{L}.{gname}.weight", e)
  if im is None and qname.startswith("IQ2"): im = np.ones(n, np.float32)  # (the iq2 quants need an importance: uniform for experts the imatrix never saw)
  assert lib.ggml_quantize_chunk(int(QT), x.ctypes.data, q.ctypes.data, 0, rows, n, im.ctypes.data if im is not None else None) == q.nbytes
  return q

w = gguf.GGUFWriter(out, "kimi-k3")
w.add_uint16(gguf.Keys.Split.LLM_KV_SPLIT_NO, 1); w.add_uint16(gguf.Keys.Split.LLM_KV_SPLIT_COUNT, 2)
w.add_int32(gguf.Keys.Split.LLM_KV_SPLIT_TENSORS_COUNT, n_trunk + len(LAYERS) * len(MATS))
for L in LAYERS:
  for _, gname, rows, n in MATS:
    w.add_tensor_info(f"blk.{L}.{gname}.weight", (NE, rows, n // BLK * TSZ), np.dtype(np.uint8), NE * rows * n // BLK * TSZ, raw_dtype=QT)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
# DELETE_CONSUMED=1: remove an input shard once every expert tensor in it is quantized (input + output don't both fit in ram)
last_layer = {}
for name, f in idx.items():
  if ".block_sparse_moe.experts." in name: last_layer[f] = max(last_layer.get(f, -1), int(name.split(".layers.")[1].split(".")[0]))
nomat = 0
with cf.ThreadPoolExecutor(int(os.environ.get("THREADS", "56"))) as ex:
  for L in LAYERS:
    t0 = time.perf_counter()
    for hfname, gname, rows, n in MATS:
      nomat += sum(importance(f"blk.{L}.{gname}.weight", e) is None for e in range(NE))
      w.write_tensor_data(np.concatenate(list(ex.map(lambda e: quant_expert(L, hfname, gname, e), range(NE)))))
    if os.environ.get("DELETE_CONSUMED"):
      for f in [f for f, last in last_layer.items() if last == L]:
        hdrs.pop(f, None); os.remove(f"{hf}/{f}")
    print(f"layer {L} {time.perf_counter()-t0:.1f}s experts-without-imatrix so far {nomat}", flush=True)
w.close()
