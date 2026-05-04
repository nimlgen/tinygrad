from __future__ import annotations
import contextlib, functools, mmap
from tinygrad.runtime.autogen import kfd, hsa
from tinygrad.runtime.autogen.am import am
from tinygrad.runtime.support.amd import AMDIP, import_module, import_soc
from tinygrad.runtime.support.hcq2 import HCQ2Compiled
from tinygrad.runtime.ops_amd import AMDSignal, AMDComputeQueue, AMDCopyQueue, AMDProgram, AMDAllocator
from tinygrad.runtime.ops_amd import KFDIface, PCIIface, USBIface, _mock
from tinygrad.device import Device
from tinygrad.uop.ops import Ops, PatternMatcher, UPat, UOp
from tinygrad.dtype import dtypes
from tinygrad.helpers import round_up, unwrap, DEBUG, getenv, ceildiv
from tinygrad.renderer.cstyle import HIPRenderer, HIPCCRenderer
from tinygrad.renderer.llvmir import AMDLLVMRenderer

def _signal(b:UOp) -> AMDSignal: return AMDSignal(base_buf=b.buffer.ensure_allocated()._buf, virt=True)
def _buf(u:UOp): return u.buffer.ensure_allocated()._buf

pm_amd_encode = PatternMatcher([
  (UPat(Ops.WAIT,    name="w"), lambda ctx, w: ctx.q.wait(_signal(w.src[0]), w.src[1])),
  (UPat(Ops.STORE,   name="s"), lambda ctx, s: ctx.q.signal(_signal(s.src[0]), s.src[1])),
  (UPat(Ops.BARRIER),           lambda ctx:    ctx.q.memory_barrier()),
  (UPat(Ops.PROGRAM, name="p"), lambda ctx, p: ctx.q.exec(p.arg[2], p.arg[3], p.arg[0], p.arg[1])),
  (UPat(Ops.COPY,    name="c"), lambda ctx, c: ctx.q.copy(_buf(c.src[0]), _buf(c.src[1]), c.src[0].buffer.nbytes)),
])

# *** AMD2 device: minimal HCQ2Compiled AMD device (no graphs, no profiling, no AQL) ***

class AMD2Device(HCQ2Compiled):
  ifaces = [KFDIface, PCIIface, USBIface, _mock(KFDIface, "MOCKIface"), _mock(KFDIface), _mock(PCIIface), _mock(USBIface)]
  pm_encode = pm_amd_encode

  def is_am(self) -> bool: return isinstance(self.iface, (PCIIface, USBIface))
  def is_usb(self) -> bool: return isinstance(self.iface, USBIface)

  def __init__(self, device:str=""):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0

    self.iface = self._select_iface()

    self.target:tuple[int, ...] = ((trgt:=self.iface.props['gfx_target_version']) // 10000, (trgt // 100) % 100, trgt % 100)
    self.arch = "gfx%d%x%x" % self.target
    assert (self.target in ((9,4,2),(9,5,0))) or self.target[0] in (11, 12), f"Unsupported arch: {self.arch}"
    if DEBUG >= 1: print(f"AMD2Device: opening {self.device_id} with target {self.target} arch {self.arch}")

    self.xccs = 1
    self.is_aql = False
    self.se_cnt = self.iface.props['array_count'] // self.iface.props['simd_arrays_per_engine']
    self.cu_cnt = self.iface.props['simd_count'] // self.iface.props['simd_per_cu']
    self.waves_per_cu = self.iface.props['max_waves_per_simd'] * self.iface.props['simd_per_cu']
    self.wave_cnt = self.cu_cnt * self.waves_per_cu if self.target[0] != 9 else min(self.cu_cnt * 40, self.se_cnt * 512)

    import importlib
    self.ip_off = importlib.import_module(f"tinygrad.runtime.autogen.am.{'vega' if self.target[0] == 9 else 'navi'}_offsets")
    self.soc = import_soc(self.target)
    self.pm4 = importlib.import_module(f"tinygrad.runtime.autogen.am.pm4_{'soc15' if self.target[0] == 9 else 'nv'}")
    self.sdma = import_module('sdma', min(self.iface.ip_versions[am.SDMA0_HWIP], (6, 0, 0)))
    self.gc = AMDIP('gc', self.iface.ip_versions[am.GC_HWIP],
                    bases={i: tuple(getattr(self.ip_off, f'GC_BASE__INST{i}_SEG{s}', 0) for s in range(6)) for i in range(6)})
    self.nbio = AMDIP('nbio' if self.target[0] < 12 else 'nbif', self.iface.ip_versions[am.NBIF_HWIP],
                      bases={i: tuple(getattr(self.ip_off, f'NBIO_BASE__INST{i}_SEG{s}', 0) for s in range(9)) for i in range(6)})

    # Disable optional features for the simple runtime.
    self.sqtt_enabled = False
    self.pmc_enabled = False

    self.compute_queue = self._create_queue(kfd.KFD_IOC_QUEUE_TYPE_COMPUTE, 16 << 20, eop_buffer_size=0x1000, ctx_save_restore_size=0)

    self.max_copy_size = 0x40000000 if self.iface.ip_versions[am.SDMA0_HWIP][0] >= 5 else 0x400000
    self.sdma_queues:dict = {}
    self.has_sdma_queue = self.sdma_queue(0) is not None

    super().__init__(device, AMDAllocator(self), [HIPRenderer, AMDLLVMRenderer, HIPCCRenderer], functools.partial(AMDProgram, self), AMDSignal,
                     functools.partial(AMDComputeQueue, self),
                     functools.partial(AMDCopyQueue, self, max_copy_size=self.max_copy_size) if self.has_sdma_queue else None,
                     kernargs_size=(16 << 20), sigalloc_size=0x1000, can_recover=self.is_am(), arch=self.arch)

    self.max_private_segment_size = 0
    # HCQ2 pre-bakes queues at compile time, so they encode dev.scratch.va_addr. Reallocating
    # scratch later would invalidate those baked addresses. Pre-allocate the worst-case size
    # (16 KB/thread) up front so dev.scratch never moves.
    self._ensure_has_local_memory(16 * 1024)

  def _create_queue(self, queue_type, ring_size, ctx_save_restore_size=0, eop_buffer_size=0, ctl_stack_size=0, debug_memory_size=0, idx=0):
    ring = self.iface.alloc(ring_size, uncached=True, cpu_access=True)
    gart = self.iface.alloc(0x100, uncached=True, cpu_access=True)
    cwsr_buffer_size = round_up((ctx_save_restore_size + debug_memory_size), mmap.PAGESIZE)
    cwsr_buffer = self.iface.alloc(cwsr_buffer_size) if ctx_save_restore_size else None
    eop_buffer = self.iface.alloc(eop_buffer_size) if eop_buffer_size else None
    qd = self.iface.create_queue(queue_type, ring, gart, rptr=getattr(hsa.amd_queue_t, 'read_dispatch_id').offset,
                                 wptr=getattr(hsa.amd_queue_t, 'write_dispatch_id').offset, eop_buffer=eop_buffer, cwsr_buffer=cwsr_buffer,
                                 ctx_save_restore_size=ctx_save_restore_size, ctl_stack_size=ctl_stack_size, idx=idx)
    qd.ring_buf = ring
    return qd

  def sdma_queue(self, idx:int):
    if getenv("AMD_DISABLE_SDMA"): return None
    if idx in self.sdma_queues: return self.sdma_queues[idx]
    with contextlib.suppress(OSError):
      self.sdma_queues[idx] = self._create_queue(kfd.KFD_IOC_QUEUE_TYPE_SDMA, 16 << 20, idx=idx)
    return self.sdma_queues.get(idx, None)

  def _ensure_has_local_memory(self, private_segment_size):
    if self.max_private_segment_size >= private_segment_size: return
    lanes_per_wave = 64
    mem_alignment_size = 256 if self.target[0] != 9 else 1024
    size_per_thread = round_up(private_segment_size, mem_alignment_size // lanes_per_wave)
    size_per_xcc = size_per_thread * lanes_per_wave * self.iface.props['max_slots_scratch_cu'] * self.cu_cnt
    self.scratch, ok = self._realloc(getattr(self, 'scratch', None), size_per_xcc)
    if ok:
      max_scratch_waves = self.cu_cnt * self.iface.props['max_slots_scratch_cu']
      wave_scratch = ceildiv(lanes_per_wave * size_per_thread, mem_alignment_size)
      num_waves = (size_per_xcc // (wave_scratch * mem_alignment_size)) // (self.se_cnt if self.target[0] != 9 else 1)
      tmpring_t = getattr(hsa, f'union_COMPUTE_TMPRING_SIZE{"_GFX"+str(self.target[0]) if self.target[0] != 9 else ""}_bitfields')
      self.tmpring_size = int.from_bytes(tmpring_t(WAVES=min(num_waves, max_scratch_waves), WAVESIZE=wave_scratch), 'little')
      self.max_private_segment_size = private_segment_size

  def hw_compute_queues(self): return [(None, self.hw_compute_queue_t)] if self.hw_compute_queue_t is not None else []
  def hw_copy_queues(self): return [(f"SDMA:{i}", functools.partial(unwrap(self.hw_copy_queue_t), queue_idx=i)) for i in self.sdma_queues]

  def synchronize(self, timeout:int|None=None):
    super().synchronize(timeout)
    if self.is_am() and not self.is_usb() and self.error_state is None:
      self.iface._collect_interrupts(reset=False, drain_only=True)

  def on_device_hang(self): self.iface.on_device_hang()
  def device_props(self): return self.iface.props
  def invalidate_caches(self):
    unwrap(self.hw_compute_queue_t)().memory_barrier().signal(self.timeline_signal, self.next_timeline()).submit(self)
    self.synchronize()
