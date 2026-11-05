from __future__ import annotations
import functools, os, glob, pathlib, struct
from typing import cast
from tinygrad.device import Allocator, Buffer, BufferSpec, BufferStorage, Compiled, Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import ceildiv, round_up, to_tuple, flatten, wait_cond
from tinygrad.engine.realize import get_call_arg_uops
from tinygrad.runtime.autogen import nvme, pci
from tinygrad.runtime.support.hcq2 import to_name, ins, unwrap_view, copy_half, HWQueue, COPY_HALVES
import tinygrad.runtime.support.hcq2 as hcq2
from tinygrad.runtime.support.memory import AddrSpace, MemoryManager, MMIOInterface
from tinygrad.runtime.support.system import PCIDevice, PCIIfaceBase, System, filter_visible_devices
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat

NVME_IDS = (0x15b7, ((0xffff, (0x5030, 0x5050)),), 0x01) # vendor, (mask, device ids), base class
SQE, CQE = 1 << nvme.NVME_NVM_IOSQES, 1 << nvme.NVME_NVM_IOCQES
ENTRIES = 0x1000 // SQE # a queue is one page, so it's contiguous even without an iommu

class NVMEQueue:
  def __init__(self, dev:NVMEDev, qid:int):
    self.dev, self.qid, self.tail, self.phase, self.entries = dev, qid, 0, 1, ENTRIES
    (self.sq, (self.sq_addr,)), (self.cq, (self.cq_addr,)) = dev.pci_dev.alloc_sysmem(0x1000), dev.pci_dev.alloc_sysmem(0x1000)
    self.sq_db, self.cq_db = nvme.NVME_REG_DBS + 2 * qid * dev.db_stride, nvme.NVME_REG_DBS + (2 * qid + 1) * dev.db_stride

  def submit(self, opcode:int, nsid:int=0, prp1:int=0, cdw10:int=0, cdw11:int=0):
    slot, self.tail = self.tail, (self.tail + 1) % self.entries
    self.sq[slot * SQE:(slot + 1) * SQE] = struct.pack("<IIQQQQII16x", opcode | slot << 16, nsid, 0, 0, prp1, 0, cdw10, cdw11)
    System.memory_barrier()
    self.dev.regs[self.sq_db // 4] = self.tail
    wait_cond(lambda: self.cq.view(slot * CQE + 12, 4, fmt='I')[0] >> 16 & 1, value=self.phase, msg=f"nvme queue {self.qid}")
    if (dw3:=self.cq.view(slot * CQE + 12, 4, fmt='I')[0]) >> 17:
      raise RuntimeError(f"nvme {self.dev.pci_dev.pcibus}: queue {self.qid} command {opcode:#x} failed, status {dw3 >> 17:#x}")
    self.phase ^= int(self.tail == 0)
    self.dev.regs[self.cq_db // 4] = self.tail

class NVMEDev:
  def __init__(self, pci_dev:PCIDevice):
    self.pci_dev = pci_dev
    pci_dev.write_config(pci.PCI_COMMAND, pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER, 2)
    self.regs = pci_dev.map_bar(0, fmt='I')
    cap = self.regs[nvme.NVME_REG_CAP // 4] | self.regs[nvme.NVME_REG_CAP // 4 + 1] << 32
    self.db_stride, timeout_ms = 4 << (cap >> 32 & 0xf), (cap >> 24 & 0xff) * 500

    self.regs[nvme.NVME_REG_CC // 4] = 0
    wait_cond(lambda: self.regs[nvme.NVME_REG_CSTS // 4] & nvme.NVME_CSTS_RDY, value=0, timeout_ms=timeout_ms, msg="nvme disable")
    self.admin_q = NVMEQueue(self, 0)
    self.regs[nvme.NVME_REG_AQA // 4] = (self.admin_q.entries - 1) << 16 | (self.admin_q.entries - 1)
    for reg, addr in ((nvme.NVME_REG_ASQ, self.admin_q.sq_addr), (nvme.NVME_REG_ACQ, self.admin_q.cq_addr)):
      self.regs[reg // 4], self.regs[reg // 4 + 1] = addr & 0xffffffff, addr >> 32
    self.regs[nvme.NVME_REG_CC // 4] = nvme.NVME_CC_IOCQES | nvme.NVME_CC_IOSQES | nvme.NVME_CC_CSS_NVM | nvme.NVME_CC_ENABLE
    wait_cond(lambda: self.regs[nvme.NVME_REG_CSTS // 4] & nvme.NVME_CSTS_RDY, value=1, timeout_ms=timeout_ms, msg="nvme enable")

    ident, (ident_addr,) = pci_dev.alloc_sysmem(0x1000)
    self.admin_q.submit(nvme.nvme_admin_identify, 0, ident_addr, 1)
    self.max_xfer = 0x1000 << min(ident[77] or 8, 8) # at most prp1 and a 256 entry list
    self.admin_q.submit(nvme.nvme_admin_identify, 1, ident_addr, 0)
    self.lba_shift = ident[128 + 4 * (ident[26] & 0xf) + 2]
    self.size = struct.unpack_from("<Q", bytes(ident[0:8]))[0] << self.lba_shift
    self.admin_q.submit(nvme.nvme_admin_set_features, 0, 0, nvme.NVME_FEAT_NUM_QUEUES, 0xfffe << 16 | 0xfffe)
    self.queues:list[NVMEQueue] = []

  def new_queue(self) -> NVMEQueue:
    self.queues.append(q:=NVMEQueue(self, len(self.queues) + 1))
    self.admin_q.submit(nvme.nvme_admin_create_cq, 0, q.cq_addr, (q.entries - 1) << 16 | q.qid, 1)
    self.admin_q.submit(nvme.nvme_admin_create_sq, 0, q.sq_addr, (q.entries - 1) << 16 | q.qid, q.qid << 16 | 1)
    return q

def prp_table(dev:NVMEDev, pages:list[int]) -> tuple[MMIOInterface, list[int]]:
  lists, list_addrs = dev.pci_dev.alloc_sysmem(ceildiv(len(pages) + 1, 256) * 0x1000)
  for j in range(len(list_addrs)):
    chunk = pages[256 * j:256 * j + 512] # list pages overlap, so a prp list never crosses a page and needs no chaining
    lists[j * 0x1000:j * 0x1000 + 8 * len(chunk)] = struct.pack(f"<{len(chunk)}Q", *chunk)
  table, table_addrs = dev.pci_dev.alloc_sysmem(round_up(16 * len(pages), 0x1000))
  # each page and a pointer to the prp list of the pages after it
  entries = flatten((page, list_addrs[(p + 1) // 256] + (p + 1) % 256 * 8) for p, page in enumerate(pages))
  table[:8 * len(entries)] = struct.pack(f"<{len(entries)}Q", *entries)
  return table, table_addrs

class NVMEIface(PCIIfaceBase):
  def __init__(self, dev:NVMEDevice, index:int):
    # make sure the drive isn't mounted or used by md/lvm
    pcibus = filter_visible_devices(System.list_devices(*NVME_IDS), "NVME")[index][1]
    mounted = {line.split()[0] for line in pathlib.Path("/proc/mounts").read_text().splitlines()}
    for b in [b for b in glob.glob("/sys/class/block/*") if f"/{pcibus}/" in os.path.realpath(b)]:
      assert f"/dev/{os.path.basename(b)}" not in mounted and not os.listdir(f"{b}/holders"), f"{dev.device} ({pcibus}) is in use by the system: {b}"
    super().__init__(dev, index, *NVME_IDS[:2], vram_bar=0, va_start=MemoryManager.va_allocator.base, va_size=MemoryManager.va_allocator.size,
                     dev_impl_t=NVMEDev, base_class=NVME_IDS[2])

class NVMEAllocator(Allocator['NVMEDevice']):
  lru = False
  def _alloc(self, size:int, options:BufferSpec) -> BufferStorage: return BufferStorage(0)
  def _offset(self, buf:int, size:int, offset:int) -> int: return buf + offset
  def _map(self, buf:Buffer) -> BufferStorage:
    iface = getattr(Device[buf.device], "iface", None)
    if not isinstance(iface, PCIIfaceBase) or iface.peer_group != self.dev.peer_group: raise RuntimeError(f"NVME can't reach {buf.device}")
    mapping = buf.meta.mapping
    paddrs = mapping.paddrs if mapping.aspace is AddrSpace.SYS else PCIIfaceBase.p2p_paddrs(iface, mapping.paddrs)[0]
    table = self.dev.iface.buffer(*prp_table(self.dev.nvme, [p + off for p, size in paddrs for off in range(0, size, 0x1000)]))
    return BufferStorage(table.get_buf(buf.device), table)

class NVMEDevice(Compiled):
  ifaces = [NVMEIface]

  def __init__(self, device:str):
    self.iface = self._select_iface(device)
    self.nvme = cast(NVMEDev, self.iface.dev_impl)
    super().__init__(device, NVMEAllocator(self), [], None)
  def synchronize(self, timeout:int|None=None): pass

# *****************
# UOps implementation

@functools.cache
def nvme_queue(drive:str, gpu:str) -> NVMEQueue: # the reads of a drive into a gpu share a queue
  dev = cast(NVMEDevice, Device[drive])
  q, (seq, seq_addrs), bar0 = dev.nvme.new_queue(), dev.nvme.pci_dev.alloc_sysmem(0x1000), dev.iface.pci_dev.bar_info(0)[0]
  bufs = {"sq": dev.iface.buffer(q.sq, [q.sq_addr]), "cq": dev.iface.buffer(q.cq, [q.cq_addr]), "seq": dev.iface.buffer(seq.view(0, 8), seq_addrs),
          "regs": dev.iface.buffer(dev.nvme.regs, [bar0 + o for o in range(0, dev.nvme.regs.nbytes, 0x1000)], snooped=False)}
  Compiled.link_bufs |= {to_name(drive, gpu, n): b for n, b in bufs.items()}
  return q

def nvme_drive(c:UOp) -> str|None: # the drive a copy reads from
  return d if c.op is Ops.CALL and (d:=to_tuple(get_call_arg_uops(c)[1].device)[0]).startswith("NVME") else None

# 1. prep: a read becomes copy halves per wave, a loop over the full waves then the rest. a wave is drained before the next, so cids stay unique

def split_nvme(lin:UOp) -> UOp|None:
  if any(c.op is Ops.LINEAR for c in lin.src) or not any(c.src[0].op is Ops.STORE and nvme_drive(c) for c in lin.src): return None
  seqs:dict[tuple[str, str], UOp] = {} # the slot counter of a drive on a gpu after the reads so far: each read takes its slots in order
  out:list[UOp] = []
  for c in lin.src:
    if c.src[0].op is not Ops.STORE or (drive:=nvme_drive(c)) is None:
      out.append(c)
      continue
    (dst, src), dev, gpu = get_call_arg_uops(c), cast(NVMEDevice, Device[drive]).nvme, to_tuple(get_call_arg_uops(c)[0].device)[0]
    (_, doff), soff = unwrap_view(dst), unwrap_view(src)[1]
    assert doff % 0x1000 == 0 and (soff | src.nbytes()) % (1 << dev.lba_shift) == 0 and soff + src.nbytes() <= dev.size, \
      f"{drive} reads whole blocks of {1 << dev.lba_shift} bytes into page aligned memory, not {src.nbytes()} bytes at {soff} to {doff}"
    full, rem = divmod(src.nbytes(), dev.max_xfer)
    waves, last = full // (ENTRIES - 1), full % (ENTRIES - 1) + (rem > 0)
    seq = seqs.setdefault((drive, gpu), UOp.alloc((1,), dtypes.uint64, 0, device=drive, name=to_name(gpu, "seq")))
    seqs[(drive, gpu)] = bumped = seq.after(seq.index(0).store(seq.index(0).load() + (n:=full + (rem > 0))))
    def halves(k:UOp, size:int) -> list[UOp]:
      return [UOp.custom_function(name).call(dst, src, seq.without_after, k, bumped.index(0).load() - n, UOp.const(size, dtypes.uint32))
              for name in COPY_HALVES]
    out += (halves(UOp.range(waves, next(UOp.unique_num), dtype=dtypes.u64), ENTRIES - 1) if waves else []) + \
           (halves(UOp.const(waves, dtypes.u64), last) if last else [])
  return lin.replace(src=tuple(out))
hcq2.pm_insert_copy_staging = PatternMatcher([(UPat(Ops.LINEAR, name="lin"), split_nvme)]) + hcq2.pm_insert_copy_staging

# 2. encode: a half becomes the commands of its wave on the gpu queue

def nvme_wave(call:UOp) -> tuple[list[UOp], list[UOp]]: # the ops that post a wave of reads and ring the doorbell, and the ops that wait for it
  (dst, src, _, k, first, size) = call.src[1:]
  drive, gpu = to_tuple(src.device)[0], to_tuple(dst.device)[0]
  dev, q = cast(NVMEDevice, Device[drive]).nvme, nvme_queue(drive, gpu)
  def mem(name:str, size:int) -> UOp: return UOp.alloc((size,), dtypes.uint8, 0, device=drive, name=to_name(gpu, name)).getaddr(gpu)
  sq, cq, regs = mem("sq", 0x1000), mem("cq", 0x1000), mem("regs", dev.regs.nbytes)
  (dbase, doff), (full, rem) = unwrap_view(dst), divmod(src.nbytes(), dev.max_xfer)
  prps, lba0 = dbase.getaddr(drive) + doff // 0x1000 * 16, src.getaddr(drive) >> dev.lba_shift

  ops = []
  for j in range(size.val): # write the sqe, then copy prp1 and prp2 (the next page or a list) from the table
    c = (int(k.vmin) if k.vmin == k.vmax else k) * (ENTRIES - 1) + j
    slot, nb = (first + c) % ENTRIES, rem if isinstance(c, int) and c == full else dev.max_xfer
    entry, lba, page = sq + slot * SQE, lba0 + c * (dev.max_xfer >> dev.lba_shift), c * (dev.max_xfer // 0x1000)
    ops += [ins("write", entry, ((slot + 1) << 16 | nvme.nvme_cmd_read).cast(dtypes.uint32), 1, *[0] * 8, lba.cast(dtypes.uint32),
                (lba >> 32).cast(dtypes.uint32), (nb >> dev.lba_shift) - 1, 0, 0, 0),
            ins("copy", entry + 24, prps + page * 16, 16 if ceildiv(nb, 0x1000) > 2 else 8)] + \
           ([ins("copy", entry + 32, prps + (page + 1) * 16, 8)] if ceildiv(nb, 0x1000) == 2 else [])
  cqe = cq + ((first + c) % ENTRIES) * CQE + 12 # the wave's last cqe, cleared before ringing the doorbell
  end = ((first + c + 1) % ENTRIES).cast(dtypes.uint32)
  issue, complete = ops + [ins("write", cqe, 0), ins("write", regs + q.sq_db, end)], [ins("wait", cqe, 1), ins("write", regs + q.cq_db, end)]
  return issue, complete + [ins("memory_barrier")]

HWQueue.q_rewrite = PatternMatcher([(UPat(Ops.CALL, name="c"),
  lambda ctx, c: ctx.lower(nvme_wave(c)[copy_half(c) == "copy_complete"]) if copy_half(c) and nvme_drive(c) else None)]) + HWQueue.q_rewrite
