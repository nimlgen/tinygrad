from __future__ import annotations
from typing import cast
import functools, struct, operator, re, itertools
from tinygrad.device import Allocator, Buffer, BufferSpec, BufferStorage, Compiled, Device
from tinygrad.dtype import dtypes, DType
from tinygrad.helpers import round_up, ceildiv, unwrap, to_tuple
from tinygrad.engine.realize import get_call_arg_uops
from tinygrad.runtime.autogen import bnxt
from tinygrad.runtime.support.rdma.bnxtdev import BNXTDev, BNXTQP, db_value, send_wqe, recv_wqe, WQE_SIZE, RING_ENTRIES, CQ_ENTRIES, MTU
from tinygrad.runtime.support.hcq2 import unwrap_view, to_name, ins, copy_half, HWQueue, COPY_HALVES
import tinygrad.runtime.support.hcq2 as hcq2
from tinygrad.runtime.support.memory import AddrSpace, MemoryManager
from tinygrad.runtime.support.system import PCIIfaceBase, System, filter_visible_devices
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat

RDMA_CHUNK = 1 << 30 # a wqe length is 32 bits

BNXT_IDS = (0x14e4, ((0xffff, (0x1760,)),), 0x02) # vendor, (mask, device ids), base class

class BNXTIface(PCIIfaceBase):
  def __init__(self, dev:RDMADevice, index:int):
    super().__init__(dev, index, *BNXT_IDS[:2], vram_bar=2, va_start=MemoryManager.va_allocator.base, va_size=MemoryManager.va_allocator.size,
                     dev_impl_t=BNXTDev, base_class=BNXT_IDS[2])
  def device_fini(self): self.dev_impl.fini()

  @functools.cached_property
  def doorbell(self) -> Buffer:
    off = self.dev_impl.db_off & ~0xfff
    return self.buffer(self.pci_dev.map_bar(2, off=off, size=0x1000), [self.pci_dev.bar_info(2)[0] + off], snooped=False)

class BNXTAllocator(Allocator):
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage: # sysmem: the counters, the batch slots, the timeline
    return self.dev.iface.storage(*self.dev.iface.pci_dev.alloc_sysmem(round_up(size, 0x1000)))
  def _copyin(self, dest:BufferStorage, src:memoryview): unwrap(dest.host)[:len(src)] = src
  def _map(self, buf:Buffer) -> BufferStorage: # a memory region over the buffer's pages, keyed at the gpu virtual address
    iface = getattr(Device[buf.device], "iface", None)
    if not isinstance(iface, PCIIfaceBase) or iface.peer_group != self.dev.peer_group: raise RuntimeError("RDMA requires memory on its node")
    mapping = buf.meta.mapping # the nic reaches vram over pcie: the bar, even where gpus reach each other over xgmi
    paddrs = mapping.paddrs if mapping.aspace is AddrSpace.SYS else PCIIfaceBase.p2p_paddrs(iface, mapping.paddrs)[0]
    align = buf._buf | functools.reduce(operator.or_, (p | s for p, s in paddrs)) # every address a multiple of the page: fewer pbl entries
    log_page = max(l for l in (12, 13, 16, 18, 20, 21, 22, 30) if not align & ((1 << l) - 1)) # the page sizes the nic has
    key = self.dev.iface.dev_impl.register_mem([p + off for p, size in paddrs for off in range(0, size, 1 << log_page)],
                                              mapping.size, log_page, va=buf._buf)
    return BufferStorage(key, key)
  def _offset(self, buf, size:int, offset:int): return buf
  def _unmap(self, storage:BufferStorage): self.dev.iface.dev_impl.unregister_mem(storage.meta)

@functools.cache
def rdma_nic_for(dev, anchor) -> RDMADevice|None:
  def node(s:str) -> str: return ":".join(s.split(":")[:3]) if s.startswith("remote:") else ""
  def bus(s:str) -> int: return int(re.findall(r":([0-9a-f]{2}):[0-9a-f]{2}\.[0-7]", s)[-1], 16)
  gpu = dev.iface.pci_dev.pcibus
  try: nics = [(i, n) for i, (_, n) in enumerate(filter_visible_devices(System.list_devices(*BNXT_IDS), "RDMA")) if node(n) == node(gpu)]
  except RuntimeError: return None # no pcie on this machine

  # the closest nic to the anchor on dev's node
  return cast(RDMADevice, Device[f"RDMA:{min(nics, key=lambda x: abs(bus(x[1]) - bus(anchor.iface.pci_dev.pcibus)))[0]}"]) if nics else None

class RDMADevice(Compiled):
  ifaces = [BNXTIface]

  def __init__(self, device:str):
    self.iface = self._select_iface(device)
    super().__init__(device, BNXTAllocator(self), [], None)

# *****************
# UOps implementation

@functools.cache
def rdma_qp(pair:tuple[str, str]) -> dict[str, BNXTQP]:
  # one qp per gpu pair
  nics = [unwrap(rdma_nic_for(Device[d], Device[min(pair)])) for d in pair]
  qps = {nic.device: BNXTQP(nic.iface.dev_impl) for nic in nics}
  for nic, q in zip(nics, qps.values()):
    bufs = {name: nic.iface.buffer(getattr(q, name).ring, getattr(q, name).paddrs) for name in ("sq", "rq", "scq", "rcq")}
    bufs |= {name: Buffer(nic.device, 8, initial_value=bytes(8)) for name in ("sq_seq", "rq_seq", "psn")} | {"db": nic.iface.doorbell}
    Compiled.link_bufs |= {to_name(nic.device, *pair, n): b for n, b in bufs.items()}
  for a, b in (nics, nics[::-1]): qps[a.device].connect(qps[b.device].qpn, b.iface.dev_impl.local_gid, b.iface.dev_impl.mac)
  return qps

def rdma_mem(nic:str, pair:tuple[str, str], name:str, size:int, dtype:DType=dtypes.uint8) -> UOp:
  return UOp.alloc((size,), dtype, 0, device=nic, name=to_name(*pair, name))
def rdma_ring(nic:str, pair:tuple[str, str], is_recv:bool) -> UOp:
  return rdma_mem(nic, pair, "rq" if is_recv else "sq", RING_ENTRIES * (WQE_SIZE if is_recv else WQE_SIZE + 8))
def rdma_cq(nic:str, pair:tuple[str, str], is_recv:bool) -> UOp: return rdma_mem(nic, pair, "rcq" if is_recv else "scq", CQ_ENTRIES * 32)
def rdma_seq(nic:str, pair:tuple[str, str], is_recv:bool) -> UOp: return rdma_mem(nic, pair, "rq_seq" if is_recv else "sq_seq", 1, dtypes.uint64)
def rdma_psn(nic:str, pair:tuple[str, str]) -> UOp: return rdma_mem(nic, pair, "psn", 1, dtypes.uint64) # the next psn of the sends
def rdma_db(nic:str, pair:tuple[str, str]) -> UOp: return rdma_mem(nic, pair, "db", 0x1000)

def rdma_wire(call:UOp) -> UOp|None:
  if call.op is not Ops.CALL: return None
  return next((b for b in get_call_arg_uops(call) if b.device is not None and to_tuple(b.device)[0].startswith("RDMA")), None)
def pair_of(buf:UOp, wire:UOp) -> tuple[str, str]: return (min(gp:=(to_tuple(buf.device)[0], wire.base.arg.name)), max(gp))

# 1. prep: a copy over a wire becomes copy halves per wqe (a chunk), each completed before the next

def split_rdma_copies(lin:UOp) -> UOp|None:
  if any(c.op is Ops.LINEAR for c in lin.src) or not any(c.src[0].op is Ops.STORE and rdma_wire(c) is not None for c in lin.src): return None
  counters:dict[UOp, UOp] = {} # a stream's next slot and the next psn of the sends of a pair, in nic memory, after the copies so far
  out:list[UOp] = []
  for c in lin.src:
    if c.src[0].op is not Ops.STORE or (wire:=rdma_wire(c)) is None:
      out.append(c)
      continue
    (dst, src), is_recv = get_call_arg_uops(c), wire is get_call_arg_uops(c)[1]
    buf, pair = dst if is_recv else src, pair_of(dst if is_recv else src, wire)
    seq0, psn0 = rdma_seq(wire.device, pair, is_recv), rdma_psn(wire.device, pair)
    chunks = [min(RDMA_CHUNK, buf.nbytes() - off) for off in range(0, buf.nbytes(), RDMA_CHUNK)]
    packets = list(itertools.accumulate((ceildiv(size, MTU) for size in chunks), initial=0))
    seq = counters[seq0] = (s:=counters.get(seq0, seq0)).after(s.index(0).store(s.index(0).load() + len(chunks)))
    if not is_recv: counters[psn0] = (q:=counters.get(psn0, psn0)).after(q.index(0).store(q.index(0).load() + packets[-1]))
    n, p = seq.index(0).load() - len(chunks), counters.get(psn0, psn0).index(0).load() - (0 if is_recv else packets[-1])
    out += [UOp.custom_function(name).call(buf, wire, seq0, UOp.const(0, dtypes.u64), n + i, p + packets[i], UOp.const(off, dtypes.uint64),
                                           UOp.const(size, dtypes.uint32), UOp.const(is_recv, dtypes.bool))
            for i, (off, size) in enumerate(zip(range(0, buf.nbytes(), RDMA_CHUNK), chunks)) for name in COPY_HALVES]
  return lin.replace(src=tuple(out))
hcq2.pm_insert_copy_staging = PatternMatcher([(UPat(Ops.LINEAR, name="lin"), split_rdma_copies)]) + hcq2.pm_insert_copy_staging

# 2. encode: a half becomes the commands of its wqe on the gpu queue

def rdma_wqe(call:UOp) -> tuple[list[UOp], list[UOp]]: # the ops that post a wqe and ring the doorbell, and the ops that wait for its cqe
  (buf, wire, _, _, n, p, off, size, is_recv) = call.src[1:]
  devs, pair, nic = (to_tuple(buf.device)[0],), pair_of(buf, wire), cast(RDMADevice, Device[wire.device])
  qp, ring, cq = rdma_qp(pair)[nic.device], rdma_ring(nic.device, pair, is_recv.val), rdma_cq(nic.device, pair, is_recv.val)
  ring_addr, cq_addr = ring.getaddr(devs), cq.getaddr(devs)
  db = rdma_db(nic.device, pair).getaddr(devs) + (nic.iface.dev_impl.db_off & 0xfff)
  ring_db = db_value(qp.qpn, bnxt.DBC_DBC_TYPE_RQ if is_recv.val else bnxt.DBC_DBC_TYPE_SQ, 0, 0)
  cq_db = db_value(qp.rcq_id if is_recv.val else qp.scq_id, bnxt.DBC_DBC_TYPE_CQ, 0, 0)

  # sdma fills in the wqe
  hdr, key = struct.unpack("<8I", (recv_wqe if is_recv.val else send_wqe)(0, 0, size.val)[:32]), unwrap_view(buf)[0].getaddr(nic.device)
  issue = [ins("write", ring_addr + (n % RING_ENTRIES) * WQE_SIZE, *hdr, buf.getaddr(devs) + off, key.cast(dtypes.uint32), size)]

  # a send also fills in its msn entry: the slot, the psn after it (a psn per packet), its first psn
  if not is_recv.val: issue += [ins("write", ring_addr + RING_ENTRIES * WQE_SIZE + (n % RING_ENTRIES) * 8,
                                    ((n % RING_ENTRIES) << 48) | (((p + ceildiv(size.val, MTU)) & 0xffffff) << 24) | (p & 0xffffff))]

  # rings the doorbell: the slot after the wqe and the epoch of its pass
  issue += [ins("write", db, ((n + 1) % RING_ENTRIES | ((n + 1) // RING_ENTRIES & 1) << bnxt.BNXT_QPLIB_DBR_EPOCH_SHIFT) | ring_db)]

  # waits for the cqe, acks the cq, and invalidates the gpu caches on recv
  complete = [ins("wait_eq", cq_addr + (n % CQ_ENTRIES) * 32 + 24, (n // CQ_ENTRIES & 1 ^ 1 | (2 if is_recv.val else 0)).cast(dtypes.uint16)),
              ins("write", db, ((n + 1) % CQ_ENTRIES | ((n + 1) // CQ_ENTRIES & 1) << bnxt.BNXT_QPLIB_DBR_EPOCH_SHIFT) | cq_db)]
  return issue, complete + ([ins("memory_barrier")] if is_recv.val else [])

HWQueue.q_rewrite = PatternMatcher([(UPat(Ops.CALL, name="c"),
  lambda ctx, c: ctx.lower(rdma_wqe(c)[copy_half(c) == "copy_complete"]) if copy_half(c) and rdma_wire(c) is not None else None)]) + HWQueue.q_rewrite
