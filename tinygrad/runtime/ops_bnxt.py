from __future__ import annotations
from typing import cast
import functools
from tinygrad.device import Allocator, Buffer, BufferSpec, BufferStorage, Compiled, Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import round_up, getenv
from tinygrad.runtime.support.am.amdev import AMMemoryManager
from tinygrad.runtime.support.bnxt import BNXTDev, BNXTQP
from tinygrad.runtime.support.memory import AddrSpace, MMIOInterface, VirtMapping
from tinygrad.runtime.support.system import PCIIfaceBase, PCIAllocationMeta, System
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat

class BNXTIface(PCIIfaceBase):
  def __init__(self, dev:BNXTDevice, index:int):
    self.dev, self.vram_bar = dev, 2
    self.pci_dev = System.pci_probe_device("BNXT", index, 0x14e4, ((0xffff, (0x1760,)),), 0x02)
    if self.is_local(): System.reserve_va(AMMemoryManager.va_allocator.base, AMMemoryManager.va_allocator.size)
    self.dev_impl = BNXTDev(self.pci_dev, ip=getenv("BNXT_IP", f"10.0.0.{index + 1}"))

  def is_bar_small(self) -> bool: return False

  def buffer(self, mem:MMIOInterface, paddrs:list[int], snooped:bool=True) -> Buffer:
    va = AMMemoryManager.alloc_vaddr(size:=round_up(mem.nbytes, 0x1000), 0x1000)
    mapping = VirtMapping(va, size, [(p, 0x1000) for p in paddrs], AddrSpace.SYS, uncached=True, snooped=snooped)
    return Buffer(self.dev.device, mem.nbytes, dtypes.uint8, opaque=BufferStorage(va, PCIAllocationMeta(mapping, True), mem))

  def counter(self) -> Buffer:
    mem, paddrs = self.pci_dev.alloc_sysmem(0x1000)
    mem[:8] = bytes(8)
    return self.buffer(mem, paddrs).view(1, dtypes.uint64, 0).ensure_allocated()

  @functools.cached_property
  def doorbell(self) -> Buffer:
    off = self.dev_impl.db_off & ~0xfff
    return self.buffer(self.pci_dev.map_bar(2, off=off, size=0x1000), [self.pci_dev.bar_info(2)[0] + off], snooped=False)

class BNXTAllocator(Allocator):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage: raise RuntimeError("BNXT only maps buffers")
  def _map(self, buf:Buffer) -> BufferStorage:
    iface = getattr(Device[buf.device], "iface", None)
    if not isinstance(iface, PCIIfaceBase) or iface.peer_group != self.dev.peer_group: raise RuntimeError("BNXT requires memory on its node")
    mapping = buf.meta.mapping
    paddrs = mapping.paddrs if mapping.aspace is AddrSpace.SYS else iface.p2p_paddrs(mapping.paddrs)[0]
    page = (2 << 20) if buf._buf % (2 << 20) == 0 and all(p % (2 << 20) == 0 and s % (2 << 20) == 0 for p, s in paddrs) else 0x1000
    key = self.dev.iface.dev_impl.register_mem([p + off for p, size in paddrs for off in range(0, size, page)],
                                              mapping.size, page.bit_length() - 1, va=buf._buf)
    return BufferStorage(key, key)
  def _offset(self, buf, size:int, offset:int): return buf
  def _unmap(self, storage:BufferStorage): self.dev.iface.dev_impl.unregister_mem(storage.meta)

class BNXTDevice(Compiled):
  has_copy_queue = False

  def __init__(self, device:str):
    self.iface = BNXTIface(self, int(device.split(":")[1]) if ":" in device else 0)
    self.qps:dict[tuple[str, str], BNXTQP] = {}
    self.bufs:dict[tuple[tuple[str, str], str], Buffer] = {}
    super().__init__(device, BNXTAllocator(self), [], None)
    self.pm_bufferize = PatternMatcher([(UPat(Ops.PARAM, name="b"),
      lambda ctx, b: ctx.bufs[b.tag[1:]] if isinstance(b.tag, tuple) and b.tag[0] == "rdma" else None)]) + self.pm_bufferize

  def synchronize(self, timeout:int|None=None):
    for d in {d for pair in self.qps for d in pair if Device[d].peer_group == self.peer_group}: Device[d].synchronize(timeout)

  def qp(self, local, peer) -> BNXTQP:
    pair = tuple(sorted((local.device, peer.device)))
    if pair not in self.qps:
      other = cast(BNXTDevice, peer.nic)
      for nic in (self, other):
        nic.qps[pair] = q = BNXTQP(nic.iface.dev_impl)
        for name in ("sq", "rq", "scq", "rcq"):
          ring = getattr(q, name)
          nic.bufs[pair, name] = nic.iface.buffer(ring["mem"], ring["paddrs"])
        for name in ("sq_prod", "sq_psn", "rq_prod", "scq_cons", "rcq_cons"): nic.bufs[pair, name] = nic.iface.counter()
        nic.bufs[pair, "db"] = nic.iface.doorbell
      for a, b in ((self, other), (other, self)):
        a.qps[pair].connect(b.qps[pair].qpn, b.iface.dev_impl.local_gid, b.iface.dev_impl.mac)
    return self.qps[pair]

  def arg(self, pair:tuple[str, str], name:str) -> UOp:
    b = self.bufs[pair, name]
    return UOp.placeholder((b.size,), b.dtype, 0, device=(self.device,), volatile=True, tag=("rdma", pair, name))
