"""Save the captured training JIT in the untimed setup (dev_beam) and load it in the timed run, like openpilot's compile3.

The pickled graph is the post-jit_lower linear: device state (timelines, signals, cmdbufs, rdma wires) is still placeholders
bound at link time, so the only real buffers in it are model/optimizer state, planned intermediates and small constants.
State buffers are pickled by name and rebound to the loading process's tensors; buffers the graph writes are reallocated
empty; everything else (constants the capture baked in) keeps its contents."""
import pickle, time, re
from tinygrad import Tensor
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import getenv

def named_buffers(named:dict[str, Tensor]) -> dict[str, Buffer]:
  out:dict[str, Buffer] = {}
  for name, t in named.items():
    # storage_base: a sharded tensor is UNSHARD over its MultiBuffer, whose .base is the UNSHARD itself
    if (u:=t.uop.storage_base).op.name != "BUFFER" or (b:=u.arg.buffer) is None: continue
    for i, x in enumerate(b.bufs if isinstance(b, MultiBuffer) else (b,)): out[f"{name}#{i}"] = x.base
  return out

class _Saver(pickle.Pickler):
  def __init__(self, f, names:dict[int, str], written:set[int]):
    super().__init__(f, protocol=pickle.HIGHEST_PROTOCOL)
    self.names, self.written, self.stats = names, written, {"named": 0, "fresh": 0, "fresh_bytes": 0, "copied": 0, "copied_bytes": 0}
    self.live:dict[int, str] = {}
    if getenv("GPTOSS_JIT_DEBUG"): # buffers live tensors hold: unnamed ones are state the loader can't rebind
      from tinygrad.tensor import all_tensors
      for tref in list(all_tensors):
        if (t:=tref()) is None or (u:=t.uop.storage_base).op.name != "BUFFER" or (b:=u.arg.buffer) is None: continue
        for x in (b.bufs if isinstance(b, MultiBuffer) else (b,)): self.live[id(x.base)] = f"{t.shape} {t.dtype}"
  def persistent_id(self, obj):
    if type(obj) is not Buffer or obj._base is not None: return None # views pickle as (base, offset); the base comes back here
    if (name:=self.names.get(id(obj))) is not None:
      self.stats["named"] += 1
      return ("named", name)
    # the jit's written set skips buffers a kernel reads and writes (in-place outputs), so size decides too: constants the capture
    # baked in are small, anything big and unnamed is an intermediate whose contents the graph rewrites
    if id(obj) in self.written or obj.nbytes > getenv("GPTOSS_JIT_COPY_MAX", 1 << 20):
      self.stats["fresh"] += 1; self.stats["fresh_bytes"] += obj.nbytes # noqa: E702
      return ("fresh", obj.device, obj.size, obj.dtype, obj.options)
    self.stats["copied"] += 1; self.stats["copied_bytes"] += obj.nbytes # noqa: E702
    if getenv("GPTOSS_JIT_DEBUG") and (held:=self.live.get(id(obj))): print(f"  copied live {obj.device} {obj.nbytes} {obj.dtype} {held}")
    return None

class _Loader(pickle.Unpickler):
  def __init__(self, f, bufs:dict[str, Buffer]):
    super().__init__(f)
    self.bufs = bufs
  def persistent_load(self, pid):
    if pid[0] == "named": return self.bufs[pid[1]]
    _, device, size, dtype, options = pid
    return Buffer(device, size, dtype, options=options)

def save_jit(jit:TinyJit, named:dict[str, Tensor], path:str):
  assert jit.captured is not None, "capture the jit before saving it"
  st = time.perf_counter()
  bufs = named_buffers(named)
  written = {id(x.base) for u in jit.captured._written_uops if u.op.name == "BUFFER" and (b:=u.arg.buffer) is not None
             for x in (b.bufs if isinstance(b, MultiBuffer) else (b,))}
  from tinygrad.runtime.ops_rdma import rdma_qp
  pairs = sorted(set(rdma_qp.cache_keys())) if hasattr(rdma_qp, "cache_keys") else _rdma_pairs(jit)
  with open(path, "wb") as f:
    (s:=_Saver(f, {id(b): n for n, b in bufs.items()}, written)).dump((jit, pairs))
  print(f"gptoss_jitcache: saved {path} in {time.perf_counter()-st:.1f}s: {s.stats['named']} named, {s.stats['fresh']} fresh "
        f"({s.stats['fresh_bytes']/1e9:.1f} GB), {s.stats['copied']} copied ({s.stats['copied_bytes']/1e6:.1f} MB)", flush=True)

# rdma placeholder names: rdma_<gpu pair>_<qp buffer>, see ops_rdma.rdma_mem
RDMA_TAG = re.compile(r"^rdma_((?:amd(?:_\d+)?_)+)(?:sq|rq|scq|rcq|sq_seq|rq_seq|rq_done|psn|db)$")

def _rdma_pairs(jit:TinyJit) -> list[tuple[str, ...]]:
  assert jit.captured is not None
  # device 0's canonical name is bare "AMD", so its part of the tag has no number
  return sorted({tuple(f"AMD:{n}" if n else "AMD" for n in re.findall(r"amd(?:_(\d+))?_", m.group(1))) for u in jit.captured._linear.toposort()
                 if u.op.name == "PARAM" and isinstance(u.tag, str) and (m:=RDMA_TAG.match(u.tag))})

def open_rdma_pairs(pairs:list[tuple[str, ...]]):
  # queue pairs and their placeholder bindings are made while lowering, which a loaded jit skipped
  if not pairs: return
  from tinygrad.runtime.ops_rdma import rdma_qp
  for pair in pairs: rdma_qp(pair)

def load_jit(named:dict[str, Tensor], path:str) -> TinyJit:
  st = time.perf_counter()
  with open(path, "rb") as f: jit, pairs = _Loader(f, named_buffers(named)).load()
  open_rdma_pairs(pairs)
  print(f"gptoss_jitcache: loaded {path} in {time.perf_counter()-st:.1f}s ({len(pairs)} rdma pairs)", flush=True)
  # link now (setup, no data touched) instead of in the first training call
  st = time.perf_counter()
  assert jit.captured is not None
  _ = jit.captured._written_uops
  print(f"gptoss_jitcache: linked in {time.perf_counter()-st:.1f}s", flush=True)
  return jit
