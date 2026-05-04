from __future__ import annotations
from typing import cast, Callable, Type, TypeVar, Generic
from dataclasses import dataclass
import functools, itertools
from tinygrad.helpers import DEV, select_first_inited, select_by_name
from tinygrad.device import Device, Buffer, BufferSpec, Compiled
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, PatternMatcher, UPat, UOp, buffers
from tinygrad.runtime.support.hcq import HCQCompiled, HCQBuffer, HCQAllocator, HCQAllocatorBase, HCQSignal, HWQueue
from tinygrad.runtime.support.memory import BumpAllocator
from tinygrad.engine.realize import unwrap_multi, resolve_params, get_call_arg_uops, get_runtime, track_stats
from tinygrad.renderer import Renderer

SignalType = TypeVar('SignalType', bound='HCQSignal')

@dataclass
class HCQEncodeCtx:
  dev: HCQ2Compiled
  q: HWQueue

def hcq_encode(ctx:HCQEncodeCtx, linear:UOp) -> UOp:
  for op in linear.src: ctx.dev.pm_encode.rewrite(op, ctx=ctx)
  return UOp(Ops.BINARY, dtypes.void, arg=ctx.q)

def hcq_schedule_copy(call:UOp, ast:UOp) -> UOp|None:
  ops = []
  for bufs, _ in unwrap_multi(call, resolved:=resolve_params(call, ())):
    if Device.canonicalize(bufs[0].device) != Device.canonicalize(bufs[1].device): return None
    dev = Device[bufs[0].device]
    if dev.hw_copy_queue_t is None: return None
    blob = hcq_encode(HCQEncodeCtx(dev, dev.hw_copy_queue_t()), UOp.linear(
      UOp(Ops.WAIT,  dtypes.void, src=(dev.timeline_uop, dev.timeline_var - 1)),
      UOp(Ops.COPY,  dtypes.void, src=tuple(resolved)),
      UOp(Ops.STORE, dtypes.void, src=(dev.timeline_uop, dev.timeline_var)))).replace(tag=ast)
    ops.append(UOp(Ops.CUSTOM_FUNCTION, dtypes.void, src=(blob,), arg="invoke", tag=dev.sdma_submit).call(*resolved))
  return UOp.linear(*ops)

def hcq_schedule_program(call:UOp, ast:UOp) -> UOp|None:
  ops = []
  for bufs, _ in unwrap_multi(call, resolved:=resolve_params(call, ())):
    dev = Device[ast.src[1].arg]
    rt = get_runtime(dev.device, ast)
    args_state = rt.fill_kernargs(tuple(bufs[i].ensure_allocated()._buf for i in ast.arg.globals),
                                  tuple(None if v.expr in ast.arg.runtimevars else v for v in ast.arg.vars))
    blob = hcq_encode(HCQEncodeCtx(dev, dev.hw_compute_queue_t()), UOp.linear(
      UOp(Ops.WAIT,    dtypes.void, src=(dev.timeline_uop, dev.timeline_var - 1)),
      UOp(Ops.BARRIER, dtypes.void),
      UOp(Ops.PROGRAM, dtypes.void, src=(dev.wrap_buffer_uop(rt.lib_gpu), dev.wrap_buffer_uop(args_state.buf)),
                                    arg=(ast.arg.global_size, ast.arg.local_size or (1, 1, 1), rt, args_state)),
      UOp(Ops.STORE,   dtypes.void, src=(dev.timeline_uop, dev.timeline_var)))).replace(tag=ast)
    ops.append(UOp(Ops.CUSTOM_FUNCTION, dtypes.void, src=(blob,), arg="invoke", tag=dev.compute_submit).call(*resolved))
  return UOp.linear(*ops)

pm_hcq_schedule = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="ast"),), name="call", allow_any_len=True, device=("AMD2",)), hcq_schedule_program),
  (UPat(Ops.CALL, src=(UPat(Ops.COPY, name="ast"),), name="call", allow_any_len=True, device=("AMD2",)), hcq_schedule_copy),
])

class HCQ2Compiled(Compiled, Generic[SignalType]):
  pm_encode: PatternMatcher

  def __init__(self, device:str, allocator:HCQAllocatorBase, compilers:list[type[Renderer]], runtime, signal_t:Type[SignalType]|None=None,
               comp_queue_t:Callable[..., HWQueue]|None=None, copy_queue_t:Callable[..., HWQueue]|None=None, kernargs_size=(16 << 20),
               sigalloc_size=0x1000, can_recover:bool=False, arch=None):
    self.device_id:int = int(device.split(":")[1]) if ":" in device else 0
    super().__init__(device, allocator, compilers, runtime, arch=arch)

    # Reuse HCQCompiled signal pools so HCQSignal.__del__ keeps working unchanged.
    self.peer_group = getattr(getattr(self, 'iface', None), 'peer_group', device.split(":")[0])
    HCQCompiled.peer_groups[self.peer_group].append(self)

    self.signal_t, self.hw_compute_queue_t, self.hw_copy_queue_t = signal_t, comp_queue_t, copy_queue_t

    self.timeline_value:int = 1
    self.prof_exec_counter:int = 0
    self.prof_prg_counter = itertools.count(0)

    if signal_t is not None:
      for sig_page in HCQCompiled.signal_pages[self.peer_group]: cast(HCQAllocator, self.allocator).map(sig_page)
      self.sigalloc_size = sigalloc_size
      self.timeline_signal, self._shadow_timeline_signal = self.new_signal(value=0, is_timeline=True), self.new_signal(value=0, is_timeline=True)

    if comp_queue_t is not None:
      self.kernargs_buf:HCQBuffer = self.allocator.alloc(kernargs_size, BufferSpec(cpu_access=True))
      self.kernargs_offset_allocator:BumpAllocator = BumpAllocator(self.kernargs_buf.size, wrap=True)

    self.can_recover = can_recover
    self.error_state:Exception|None = None

  def count(self) -> int: return self.iface.count if hasattr(self, 'iface') else 1

  def synchronize(self, timeout:int|None=None):
    if self.error_state is not None: raise self.error_state
    if not hasattr(self, 'timeline_signal'): return
    try: self.timeline_signal.wait(self.timeline_value - 1, timeout=timeout if timeout is not None and self.can_recover else None)
    except RuntimeError as e:
      self.error_state = e
      if hasattr(self, 'on_device_hang'): self.on_device_hang()
      raise e
    if self.timeline_value > (1 << 31): self._wrap_timeline_signal()

  def next_timeline(self):
    self.timeline_value += 1
    return self.timeline_value - 1

  def new_signal(self, **kwargs) -> SignalType:
    assert self.signal_t is not None, "Device does not support signals"
    if not HCQCompiled.signal_pool[pg:=self.peer_group]:
      HCQCompiled.signal_pages[pg].append(alc:=self.allocator.alloc(self.sigalloc_size, BufferSpec(host=True, uncached=True, cpu_access=True)))
      HCQCompiled.signal_pool[pg] += [alc.offset(offset=off, size=16) for off in range(0, alc.size, 16)]
      for dev in HCQCompiled.peer_groups[pg]: cast(HCQAllocator, dev.allocator).map(alc)
    return self.signal_t(base_buf=HCQCompiled.signal_pool[pg].pop(), owner=self, **kwargs)

  def _wrap_timeline_signal(self):
    self.timeline_signal, self._shadow_timeline_signal, self.timeline_value = self._shadow_timeline_signal, self.timeline_signal, 1
    self.timeline_signal.value = 0
    cast(HCQAllocatorBase, self.allocator).b_timeline = [0] * len(cast(HCQAllocatorBase, self.allocator).b)

  def _realloc(self, oldbuf:HCQBuffer|None, new_size:int, options:BufferSpec|None=None, force=False) -> tuple[HCQBuffer, bool]:
    if oldbuf is not None: self.allocator.free(oldbuf, oldbuf.size, options=options)
    try: buf, realloced = self.allocator.alloc(new_size, options=options), True
    except MemoryError:
      if force: raise
      buf, realloced = self.allocator.alloc(oldbuf.size if oldbuf is not None else new_size, options=options), False
    return buf, realloced

  def _select_iface(self):
    assert hasattr(self, "ifaces"), "must have ifaces to select an iface"
    t = DEV.target(dev:=type(self).__name__[:-6])
    filtered = select_by_name(self.ifaces, lambda i: i.__name__[:-5], t.interface, f"{dev} has no interface {t.interface!r}")
    filtered = [i for i in filtered if t.interface.startswith("MOCK") or not i.__name__[:-5].startswith("MOCK")]
    return select_first_inited([functools.partial(cast(Callable, iface), self, self.device_id) for iface in filtered],
                               f"No interface for {dev}:{self.device_id} is available")

  def _is_cpu(self) -> bool: return hasattr(self, 'device') and self.device.split(":")[0] == "CPU"

  def finalize(self):
    try: self.synchronize()
    except RuntimeError as e: print(f"{self.device} synchronization failed before finalizing: {e}")
    if hasattr(self, 'iface') and hasattr(self.iface, 'device_fini'): self.iface.device_fini()

  # *** HCQ2-specific: UOp-based scheduling glue ***

  def wrap_buffer_uop(self, hcqbuf:HCQBuffer) -> UOp:
    uop = UOp.new_buffer(self.device, hcqbuf.size, dtypes.uint8)
    buffers[uop] = Buffer(self.device, hcqbuf.size, dtypes.uint8, opaque=hcqbuf)
    return uop

  @property
  def timeline_uop(self) -> UOp:
    if (cur:=getattr(self, '_timeline_uop', None)) is not None: return cur
    self._timeline_uop = self.wrap_buffer_uop(self.timeline_signal.base_buf)
    return self._timeline_uop

  @property
  def timeline_var(self) -> UOp:
    if (cur:=getattr(self, '_timeline_var', None)) is not None: return cur
    self._timeline_var = UOp.variable(f"_hcq_tl_{self.device.replace(':', '_')}", 0, 1<<31, dtype=dtypes.uint32)
    return self._timeline_var

  def _invoke(self, ctx, call, ast):
    q, var_vals = ast.src[0].arg, {**ctx.var_vals, self.timeline_var.expr: self.next_timeline()}
    bufs = [u.buffer for u in get_call_arg_uops(call) if u.op is Ops.BUFFER]
    with track_stats(ctx, call, self.device, bufs, ctx.var_vals): q.submit(self, var_vals)
  compute_submit = sdma_submit = _invoke
