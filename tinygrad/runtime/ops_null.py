import inspect, functools, itertools
from tinygrad.device import BufferStorage, BufferSpec, Buffer, Compiled, HostAllocator, MMIOInterface, Program, ProfileGraphEntry, ProfileGraphEvent
from tinygrad.renderer import Renderer, cstyle, nir, ptx, llvmir, wgsl, x86
from tinygrad.renderer.cstyle import CStyleLanguage
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, KernelInfo, uopfunc
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv, dedup, prod, panic, cpu_events, perf_counter_us, NULL_ALLOW_COPYOUT, PROFILE
from tinygrad.engine.realize import get_call_kernel_args
from tinygrad.runtime.support.hcq2 import layout_args, pack_args, pkt, select, assemble, encode_cmdbuf, addr
from tinygrad.runtime.support.hcq2 import pm_hcq

class NullRenderer(CStyleLanguage):
  has_local = False
  float4 = "float4"
  barrier = "// BARRIER"
  type_map = {**CStyleLanguage.type_map, **{dt:dt.name for dt in dtypes.fp8s}}
  code_for_op = {**CStyleLanguage.code_for_op, Ops.THREEFRY: lambda a,b,dtype: f"threefry({a},{b})", Ops.MAX: lambda a,b,dtype: f"max({a},{b})"}
  def asm(self, prg: UOp, lin: UOp) -> bytes:
    assert self.target.arch.startswith("gfx"), "only amd supports assembly"
    from tinygrad.renderer.amd.elf import assemble_linear
    return assemble_linear(prg, lin, self.target.arch)

EXEC, COPY, WAIT, STORE, TIMESTAMP = range(5)
null_events:dict[tuple[str, str, bytes|None], int] = {}

@uopfunc
def null_submit(cmdbuf:UOp, doorbell:UOp) -> UOp: return doorbell.index(0).store(cmdbuf.index(0).load()).sink()

def null_cmd(dev:str, op:int, *args:UOp|int, runs:UOp|None=None) -> list[UOp]:
  return [pkt(op, *[a.getaddr(dev) if isinstance(a, UOp) else UOp.const(a, dtypes.uint64) for a in (op, *args, 0, 0, 0)][:4], runs=runs)]
def null_event(device:str, name:str, key:bytes|None=None) -> int: return null_events.setdefault((device, name, key), len(null_events))

def null_exec(ctx:str, call:UOp, prg:UOp) -> list[UOp]:
  args = get_call_kernel_args(call, prg, (ctx,))
  kernargs = UOp(Ops.LINEAR, src=tuple(pack_args(layout_args(args), 8 * max(len(args), 1))), arg="kernargs")
  return null_cmd(ctx, EXEC, kernargs, len(args), null_event(ctx, prg.src[0].arg.function_name, prg.key), runs=call)

pm_null = PatternMatcher([
  (UPat(Ops.PROGRAM, name="prg").call(name="call", allow_any_len=True), null_exec),
  (UPat(Ops.STORE).call(UPat.var("dst"), UPat.var("src"), name="call", allow_any_len=True),
   lambda ctx, call, dst, src: null_cmd(ctx, COPY, dst, src, null_event(f"{src.device}:SDMA:0", f"{src.device} -> {dst.device}"), runs=call)),
  (UPat.var("buf").index(UPat.var("i")).load().backedge(UPat(Ops.RANGE), UPat(Ops.LOAD) < UPat.var("v")),
   lambda ctx, buf, i, v: null_cmd(ctx, WAIT, addr(buf, i, ctx), v, 0)),
  (UPat.var("buf").index(UPat.var("i")).load().backedge(UPat(Ops.RANGE), UPat(Ops.LOAD) != UPat.var("v")),
   lambda ctx, buf, i, v: null_cmd(ctx, WAIT, addr(buf, i, ctx), v, 1)),
  (UPat.var("buf").index(UPat.var("i")).store(UPat.native("gettime").call()), lambda ctx, buf, i: null_cmd(ctx, TIMESTAMP, addr(buf, i, ctx))),
  (UPat.var("buf").index(UPat.var("i")).store(UPat.var("v")), lambda ctx, buf, i, v: null_cmd(ctx, STORE, addr(buf, i, ctx), v)),
]) + pm_hcq

def null_encode(s:UOp) -> UOp:
  (call, *deps), q = s.src, s.src[0].src[1]
  cmdbuf = encode_cmdbuf(*assemble(select(call.body.src, pm_null, q.device)), q, deps)
  return null_submit(cmdbuf, UOp.alloc((1,), dtypes.uint8, device=q.device, name="doorbell"))

class NullProgram(Program['NullDevice']):
  def __init__(self, dev, obj, links=None): self.streams = [(i,prod(s)) for i,(n,_,_,s) in enumerate(obj.signature) if (n or "").startswith("cmdbuf")]
  def __call__(self, *args, **kwargs):
    st, words = perf_counter_us(), [w for i, n in self.streams for w in MMIOInterface(args[i], n, fmt='Q')[:]]
    # timestamps are emulated: every exec and copy takes 1us
    for op, at, done in zip(words[0::4], words[1::4], itertools.accumulate(op in (EXEC, COPY) for op in words[0::4])):
      if op == TIMESTAMP: MMIOInterface(at, 8, fmt='Q')[0] = int((st + done) * 1000)
    descs = [list(null_events)[event] for op, event in zip(words[0::4], words[3::4]) if op in (EXEC, COPY)] if PROFILE else []
    sigs = [st + sum(x[0] == d[0] for x in descs[:i]) + k for i, d in enumerate(descs) for k in (0, 1)]
    if descs: cpu_events.append(ProfileGraphEvent([ProfileGraphEntry(d, n, 2*i, 2*i+1, k) for i, (d, n, k) in enumerate(descs)], sigs))

class NullAllocator(HostAllocator):
  def _alloc(self, size:int, options, va=itertools.count(1 << 40, 1 << 32)) -> BufferStorage:
    return super()._alloc(size, options) if options.host or options.cpu_access else BufferStorage(next(va))
  def _copyin(self, dest, src:memoryview): pass
  def _copyout(self, dest:memoryview, src): NULL_ALLOW_COPYOUT or panic(RuntimeError, "no copyout on NULL")
  def _map(self, buf:Buffer) -> BufferStorage: return BufferStorage(buf._buf if buf.device.startswith("NULL") else buf.host.addr)

Compiled.pm_encode += PatternMatcher([
  (UPat.sink(arg=KernelInfo("null_compute_submit")).call(UPat()).after(allow_any_len=True, name="s"), null_encode),
  (UPat.sink(arg=KernelInfo("null_copy_submit")).call(UPat()).after(allow_any_len=True, name="s"), null_encode),
])

class NullDevice(Compiled):
  renderers = dedup([NullRenderer] + [r for m in [cstyle, nir, ptx, llvmir, wgsl, x86] for r in m.__dict__.values()
                                      if inspect.isclass(r) and issubclass(r, Renderer)])
  host = property(lambda self: self.device)
  timeline = functools.cached_property(lambda self: self.host_buffer(16))
  error_state = functools.cached_property(lambda self: self.host_buffer(8))

  def __init__(self, device:str):
    assert (emu:=getenv("EMULATE", "")) == "", \
      "EMULATE is deprecated, use DEV=NULL:HIP:"+{"AMD":"gfx1100", "AMD_RDNA4":"gfx1201", "AMD_CDNA4":"gfx950"}.get(emu, "<arch>")
    super().__init__(device, NullAllocator(self), NullProgram)

  # its memory is fake: every placeholder on it gets host memory
  def host_buffer(self, n:int) -> Buffer: return Buffer(self.device, n, opaque=memoryview(bytearray(n)), options=BufferSpec(external_ptr=1))
