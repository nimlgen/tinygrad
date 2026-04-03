from __future__ import annotations
import mmap, struct, functools
from typing import cast
from tinygrad.helpers import round_up, DEBUG
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocatorBase, HCQAllocator, HWQueue, HCQBuffer, FileIOInterface
from tinygrad.runtime.support.system import System, PCIIfaceBase, PCIAllocationMeta
from tinygrad.runtime.support.memory import VirtMapping, AddrSpace
from tinygrad.runtime.support.mlx.mlxdev import MLXDev, MLXQP, to_be
from tinygrad.runtime.ops_amd import AMDComputeQueue, AMDDevice

class RDMAAllocator(HCQAllocatorBase):
  def __init__(self, dev:RDMADevice): super().__init__(dev, batch_cnt=0)
  def _map(self, buf): pass
  def _transfer(self, dest, src, sz, src_dev, dest_dev):
    src_paddr = src.meta.mapping.paddrs[0][0] + src_dev.iface.pci_dev.bar_info(src_dev.iface.vram_bar)[0]
    dest_paddr = dest.meta.mapping.paddrs[0][0] + dest_dev.iface.pci_dev.bar_info(dest_dev.iface.vram_bar)[0]
    RDMACopyQueue(self.dev).copy(dest_paddr, src_paddr, sz, dest_dev, src_dev).submit(self.dev)

class MLXIface(PCIIfaceBase):
  def __init__(self, dev:RDMADevice, dev_id:int):
    cl, pcibus = System.list_devices(vendor=0x15b3, devices=((0xffff, (0x101b,)),))[dev_id]
    self.dev = dev
    self.pci_dev = cl("mlx", pcibus)
    self.mlx_dev = MLXDev(self.pci_dev, ip=f"10.0.0.{dev_id}")
    self.uar_buf = self._buf([self.mlx_dev.pci_dev.bar_info(0)[0] + self.mlx_dev.uar * 0x1000])
    self.dbr_buf = self._buf(self.mlx_dev.dbr_paddrs)

  def is_bar_small(self) -> bool: return False

  def _buf(self, paddrs:list[int]) -> HCQBuffer:
    va = FileIOInterface.anon_mmap(0, size:=len(paddrs) * 0x1000, 0, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, 0)
    mapping = VirtMapping(va, size, [(p, 0x1000) for p in paddrs], AddrSpace.SYS, uncached=True, snooped=True)
    return HCQBuffer(va, size, meta=PCIAllocationMeta(mapping, has_cpu_mapping=False), owner=self.dev)

  @functools.cache
  def connect(self, remote_nic:RDMADevice) -> tuple[MLXQP, MLXQP, HCQBuffer]:
    src_qp, dest_qp = MLXQP(self.mlx_dev, log_sq_size=7, log_rq_size=7), MLXQP(remote_nic.iface.mlx_dev, log_sq_size=7, log_rq_size=7)
    src_qp.connect(dest_qp)
    dest_qp.connect(src_qp)
    return src_qp, dest_qp, remote_nic.iface._buf(dest_qp.cq_paddrs)

class RDMACopyQueue(HWQueue):
  def __init__(self, dev:RDMADevice):
    self.dev = dev
    super().__init__()

  def copy(self, dest_paddr, src_paddr, copy_size, dest_dev:AMDDevice, src_dev:AMDDevice):
    mkey = self.dev.iface.mlx_dev.mkey
    recv_data = struct.pack('>IIQ', copy_size, mkey, dest_paddr)
    send_data = struct.pack('>IIQ', copy_size, mkey, src_paddr)
    src_wait = src_dev.timeline_value - 1
    dest_wait = dest_dev.timeline_value - 1
    src_sig = src_dev.next_timeline()
    dest_sig = dest_dev.next_timeline()
    self._q.append((src_dev, dest_dev, recv_data, send_data, src_wait, src_sig, dest_wait, dest_sig))
    return self

  def _submit(self, dev:RDMADevice):
    for src_dev, dest_dev, recv_data, send_data, src_wait, src_sig, dest_wait, dest_sig in self._q:
      remote_nic = dest_dev.rdma_dev()
      src_qp, dest_qp, cq_buf = dev.iface.connect(remote_nic)

      # ensure GPU page tables are set up
      for buf in [dev.iface.uar_buf, dev.iface.dbr_buf]: cast(HCQAllocator, src_dev.allocator).map(buf)
      for buf in [remote_nic.iface.dbr_buf, cq_buf]: cast(HCQAllocator, dest_dev.allocator).map(buf)

      # recv WQE on dest QP's RQ
      rq_wqe = dest_qp.qp_buf.view((dest_qp.rq_head & ((1 << dest_qp.log_rq_size) - 1)) * 16, 16)
      rq_wqe[:] = recv_data
      dest_qp.rq_head += 1

      # send WQE on src QP's SQ (opcode 0x0a = SEND, ds_count=2, no CE)
      wqe = src_qp.qp_buf.view(src_qp.sq_offset + (src_qp.sq_head & ((1 << src_qp.log_sq_size) - 1)) * 64, 64)
      wqe[:] = bytes(64)
      wqe[0:8] = struct.pack('>II', (src_qp.sq_head << 8) | 0x0a, (src_qp.qp_info['qpn'] << 8) | 2)
      wqe[16:32] = send_data
      src_qp.sq_head += 1
      doorbell_val = to_be('Q', int.from_bytes(bytes(wqe[0:8]), 'big'))

      # recv CQ owner bit poll
      cq_ci = dest_qp.cq_ci
      cq_owner = (1 if (cq_ci & dest_qp.cq_size) else 0) << 24

      # sender: fire doorbell and signal immediately
      src_q = AMDComputeQueue(src_dev)
      src_q.wait(src_dev.timeline_signal, src_wait)
      src_q.write(dev.iface.dbr_buf.offset(src_qp.qp_dbr + 4), to_be('I', src_qp.sq_head))
      src_q.write(dev.iface.uar_buf.offset(0x800), doorbell_val, b64=True)
      src_q.signal(src_dev.timeline_signal, src_sig)
      src_q.submit(src_dev)

      # receiver: post recv, poll recv CQ
      dest_q = AMDComputeQueue(dest_dev)
      dest_q.wait(dest_dev.timeline_signal, dest_wait)
      dest_q.write(remote_nic.iface.dbr_buf.offset(dest_qp.qp_dbr), to_be('I', dest_qp.rq_head))
      dest_q.poll(cq_buf.offset((cq_ci & (dest_qp.cq_size - 1)) * 64 + 60), cq_owner, mask=0x01000000)
      dest_q.write(remote_nic.iface.dbr_buf.offset(dest_qp.cq_dbr), to_be('I', (cq_ci + 1) & 0xFFFFFF))
      dest_q.signal(dest_dev.timeline_signal, dest_sig)
      dest_q.submit(dest_dev)

      dest_qp.cq_ci += 1

class RDMADevice(HCQCompiled):
  def __init__(self, device:str=""):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.iface = MLXIface(self, self.device_id)
    super().__init__(device, RDMAAllocator(self), [], None, signal_t=None)
