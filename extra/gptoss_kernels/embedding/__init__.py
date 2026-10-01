from __future__ import annotations
import functools, pathlib
from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import getenv, ALLREDUCE_NODE_NDEVS
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType, sint
from tinygrad.renderer import Estimates
from extra.llama_kernels import alloc_like, compile_hip

VOCAB, EMBED = 128256, 2880
FWD_THREADS, ROWS_PER_WG = 512, 8

@functools.cache
def _custom_embedding_fwd(out:UOp, idx:UOp, weight:UOp) -> UOp:
  tokens = idx.numel()
  threads, workgroups = UOp.special(FWD_THREADS, "lidx0"), UOp.special(tokens // ROWS_PER_WG, "gidx0")
  sink = UOp.sink(out.base, idx.base, weight.base, threads, workgroups,
                  arg=KernelInfo(f"gptoss_embedding_fwd_{tokens}_{VOCAB}_{EMBED}_v16_t{FWD_THREADS}_nt",
                                 estimates=Estimates(mem=tokens*4 + tokens*EMBED*4)))
  src = (pathlib.Path(__file__).parent/"embedding_fwd.cpp").read_text()
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, []))))

def gptoss_embedding_fwd(weight:Tensor, idx:Tensor) -> Tensor:
  out_shape = idx.shape + (EMBED,)
  out = alloc_like(out_shape, dtypes.bfloat16, idx.device, idx.uop.axis).clone()
  out, *_ = Tensor.custom_kernel(out, idx.reshape(-1), weight, fxn=_custom_embedding_fwd)
  return out

THREADS = 256

@functools.cache
def _custom_init_heads(head:UOp) -> UOp:
  vocab = head.numel()
  threads, workgroups = UOp.special(THREADS, "lidx0"), UOp.special((vocab+THREADS-1)//THREADS, "gidx0")
  sink = UOp.sink(head.base, threads, workgroups,
                  arg=KernelInfo(f"embedding_bwd_init_heads_{vocab}", estimates=Estimates(mem=vocab*4)))
  src = (pathlib.Path(__file__).parent/"embedding_bwd.cpp").read_text()
  defines = [f"-DVOCAB={vocab}", f"-DTHREADS={THREADS}", "-DINIT_HEADS=1"]
  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

@functools.cache
def _custom_build_links(next_idx:UOp, head:UOp, idx:UOp) -> UOp:
  tokens, vocab = idx.numel(), head.numel()
  threads, workgroups = UOp.special(THREADS, "lidx0"), UOp.special((tokens+THREADS-1)//THREADS, "gidx0")
  sink = UOp.sink(next_idx.base, head.base, idx.base, threads, workgroups,
                  arg=KernelInfo(f"embedding_bwd_build_links_{tokens}_{vocab}", estimates=Estimates(ops=tokens, mem=3*tokens*4)))
  src = (pathlib.Path(__file__).parent/"embedding_bwd.cpp").read_text()
  defines = [f"-DTOKENS={tokens}", f"-DVOCAB={vocab}", f"-DTHREADS={THREADS}", "-DBUILD_LINKS=1"]
  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

@functools.cache
def _custom_reduce(out:UOp, grad_emb:UOp, head:UOp, next_idx:UOp, *row_offset:UOp) -> UOp:
  vocab, embed = out.shape
  tokens = next_idx.numel()
  threads = UOp.special(THREADS, "lidx0")
  workgroups = UOp.special(vocab*((embed+THREADS-1)//THREADS), "gidx0")
  sink = UOp.sink(out.base, grad_emb.base, head.base, next_idx.base, *(x.base for x in row_offset), threads, workgroups,
                  arg=KernelInfo(f"embedding_bwd_owner_reduce_{tokens}_{vocab}_{embed}",
                                 estimates=Estimates(ops=tokens*embed, mem=tokens*embed*2+vocab*embed*2)))
  src = (pathlib.Path(__file__).parent/"embedding_bwd.cpp").read_text()
  defines = [f"-DTOKENS={tokens}", f"-DVOCAB={vocab}", f"-DEMBED={embed}", f"-DTHREADS={THREADS}"]
  if row_offset: defines.append("-DSHARDED_VOCAB=1")
  return UOp(Ops.PROGRAM,
             src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

@functools.cache
def _vocab_row_offsets(device:tuple[str, ...], vocab:int, ranks:tuple[int, ...]) -> Tensor:
  return Tensor([rank*(vocab//len(device)) for rank in ranks], dtype=dtypes.int32).shard(device, 0).realize()

def _shards(t:Tensor) -> UOp:
  # Keep logical views while removing the distributed axis marker. A token slice may still
  # reference an 8193-token physical allocation; dropping its SHRINK changes the token count.
  u = t.uop
  if u.op is Ops.UNSHARD: return u.src[0]
  parent = Tensor(u.src[0], device=t.device)
  raw = _shards(parent)
  if u.op is Ops.AFTER: return raw.after(*u.src[1:])
  if u.op is Ops.RESHAPE: return raw.reshape(u.shard_shape)
  if u.op is Ops.SHRINK:
    axis = parent.uop.axis
    assert axis is not None and u.marg[axis] == (0, parent.shape[axis]), "partial sharded-axis slice unsupported"
    return raw._mop(Ops.SHRINK, tuple((0, raw.shape[i]) if i == axis else v for i,v in enumerate(u.marg)))
  if u.op is Ops.PERMUTE: return raw.permute(u.marg)
  if u.op is Ops.CAST: return raw.cast(u.dtype)
  if u.op is Ops.CONTIGUOUS: return raw.contiguous()
  raise NotImplementedError(f"unsupported shard view {u.op}")

@functools.cache
def _node_gather_fxn(param:UOp, n:int) -> UOp:
  t = Tensor(param)
  devs, raw = t.device, _shards(t)
  assert t.uop.axis == 0
  nodes = [devs[i:i+n] for i in range(0, len(devs), n)]
  per_dev = [Tensor.cat(*[Tensor(raw.mselect(devs.index(s))).to(d) for s in node]) for node in nodes for d in node]
  return UOp.mstack(*[x.uop for x in per_dev])

def _node_gather(t:Tensor, n:int) -> Tensor:
  # Bind the complete logical view at a call boundary before selecting physical lanes.
  # Peeling a caller's view directly bypasses its PARAM identity in an enclosing function.
  out = _node_gather_fxn(t.uop.param_like(0), n)
  return Tensor(out.call_with_output(t.uop, name="embedding_node_gather", precompile=True), device=t.device)

def embedding_bwd_two_nodes(grad_emb:Tensor, idx:Tensor, vocab:sint, n:int) -> Tensor:
  # each node reduces its own tokens for its rows and for its rank peers' rows: 46 MB crosses the nic instead of every token's grad
  devs = grad_emb.device
  assert isinstance(devs, tuple) and len(devs) == 2 * n
  ge, ix = _node_gather(grad_emb, n), _node_gather(idx, n)
  peer = [(i + n) % len(devs) for i in range(len(devs))]
  own, theirs = (_shards(embedding_bwd_owner(ge, ix, vocab, shard_output=True, ranks=r)) for r in (tuple(range(len(devs))), tuple(peer)))
  out = [own.mselect(i).alu(Ops.ADD, theirs.mselect(peer[i]).copy_to_device(devs[i])) for i in range(len(devs))]
  return Tensor(UOp.mstack(*out).unshard(0, UOp.range(len(devs), -1, AxisType.DEVICE)), device=devs)

def embedding_bwd_owner(grad_emb:Tensor, idx:Tensor, vocab:sint, *, shard_output:bool=False, ranks:tuple[int, ...]|None=None) -> Tensor:
  grad_emb = grad_emb.reshape(idx.numel(), grad_emb.shape[-1])
  device = grad_emb.device
  head = alloc_like((vocab,), dtypes.int32, device)
  next_idx = alloc_like((idx.numel(),), dtypes.int32, device)
  offsets = ()
  if shard_output:
    assert isinstance(device, tuple) and vocab % len(device) == 0
    offsets = (_vocab_row_offsets(device, vocab, ranks or tuple(range(len(device)))),)
  out = alloc_like((vocab, grad_emb.shape[-1]), dtypes.bfloat16, device, 0 if shard_output else None)
  head, *_ = Tensor.custom_kernel(head, fxn=_custom_init_heads)
  next_idx, *_ = Tensor.custom_kernel(next_idx, head, idx.reshape(-1), fxn=_custom_build_links)
  out, *_ = Tensor.custom_kernel(out, grad_emb, head, next_idx, *offsets, fxn=_custom_reduce)
  return out

@functools.cache
def _embedding_fwd_fxn(wp:UOp, ip:UOp, device:str|tuple[str, ...]) -> Tensor:
  return gptoss_embedding_fwd(Tensor(wp, device=device), Tensor(ip, device=device))

def _embedding_bwd(grad_emb:UOp, call:UOp) -> tuple:
  weight, idx = call.src[1:3]
  device = Tensor(weight).device
  def gather(u:UOp) -> Tensor:
    t = Tensor(u, device=device)
    if not isinstance(device, tuple) or t.uop.axis is None: return t
    return Tensor(t.uop.copy_to_device(device)).contiguous()
  shard_output = isinstance(device, tuple) and getenv("ZERO_OPTIM", 0) and getenv("ZERO2", 0) and getenv("GPTOSS_ZERO2_EMBEDDING", 0)
  if shard_output and Tensor(grad_emb).uop.axis is not None and 0 < (n:=ALLREDUCE_NODE_NDEVS.value) and len(device) == 2 * n:
    return embedding_bwd_two_nodes(Tensor(grad_emb, device=device), Tensor(idx, device=device), weight.shape[0], n).uop, None
  return embedding_bwd_owner(gather(grad_emb), gather(idx), weight.shape[0], shard_output=bool(shard_output)).uop, None

class GPTOSSEmbedding(nn.Embedding):
  def __call__(self, idx:Tensor) -> Tensor:
    fxn = _embedding_fwd_fxn(self.weight.as_param(0).uop, idx.as_param(1).uop, self.weight.device)
    return Tensor.call(self.weight, idx, fxn=fxn, grad_fxn=_embedding_bwd)
