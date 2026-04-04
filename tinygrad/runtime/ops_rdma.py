from __future__ import annotations
import mmap, struct, functools
from typing import cast
from tinygrad.uop.ops import sint
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocatorBase, HCQAllocator, HWQueue, HCQBuffer, FileIOInterface
from tinygrad.runtime.support.system import System, PCIIfaceBase, PCIAllocationMeta
from tinygrad.runtime.support.memory import VirtMapping, AddrSpace
from tinygrad.runtime.support.mlx.mlxdev import MLXDev, MLXQP, to_be
from tinygrad.runtime.ops_amd import AMDComputeQueue, AMDDevice

class RDMAAllocator(HCQAllocatorBase):
  def __init__(self, dev:RDMADevice): super().__init__(dev, batch_cnt=0)

  def _map(self, buf):
    base = buf._base or buf
    bar_base = base.owner.iface.pci_dev.bar_info(base.owner.iface.vram_bar)[0]
    paddrs = base.meta.mapping.paddrs
    page_sz = (2 << 20) if min(sz for _, sz in paddrs) >= (2 << 20) else (4 << 10)
    pages = [bar_base + p + off for p, sz in paddrs for off in range(0, sz, page_sz)]
    mkey = self.dev.iface.mlx_dev.register_mem(pages, len(pages) * page_sz, {(4 << 10): 12, (2 << 20): 21}[page_sz])
    return HCQBuffer(bar_base + paddrs[0][0], base.size, meta=mkey, owner=self.dev)

  def _transfer(self, dest, src, sz, src_dev, dest_dev):
    q = RDMACopyQueue(self.dev)
    src_q, dest_q = AMDComputeQueue(src_dev), AMDComputeQueue(dest_dev)
    remote_nic = dest_dev.rdma_dev()
    src_qp, dest_qp, _, _ = self.dev.iface.connect(remote_nic)
    qpn = src_qp.qp_info['qpn']

    src_q.wait(src_dev.timeline_signal, src_dev.timeline_value - 1)
    dest_q.wait(dest_dev.timeline_signal, dest_dev.timeline_value - 1)

    q.prepare_transfer(dest, src, sz, src_dev, dest_dev, src_q, dest_q,
      sq_dbr_val=to_be('I', src_qp.sq_head + 1),
      uar_db_val=to_be('Q', ((src_qp.sq_head << 8) | 0x0a) << 32 | ((qpn << 8) | 2)),
      src_cq_dbr_val=to_be('I', (src_qp.cq_ci + 1) & 0xFFFFFF),
      rq_dbr_val=to_be('I', dest_qp.rq_head + 1),
      dest_cq_dbr_val=to_be('I', (dest_qp.cq_ci + 1) & 0xFFFFFF),
      src_cq_ci=src_qp.cq_ci, dest_cq_ci=dest_qp.cq_ci)

    q._submit(self.dev)
    src_q.signal(src_dev.timeline_signal, src_dev.next_timeline()).submit(src_dev)
    dest_q.signal(dest_dev.timeline_signal, dest_dev.next_timeline()).submit(dest_dev)

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
  def connect(self, remote_nic:RDMADevice) -> tuple[MLXQP, MLXQP, HCQBuffer, HCQBuffer]:
    src_qp, dest_qp = MLXQP(self.mlx_dev, log_sq_size=7, log_rq_size=7), MLXQP(remote_nic.iface.mlx_dev, log_sq_size=7, log_rq_size=7)
    src_qp.connect(dest_qp)
    dest_qp.connect(src_qp)
    return src_qp, dest_qp, remote_nic.iface._buf(dest_qp.cq_paddrs), self._buf(src_qp.cq_paddrs)

class RDMACopyQueue(HWQueue):
  def __init__(self, dev:RDMADevice):
    self.dev = dev
    super().__init__()

  @staticmethod
  def build_rdma_mec(rdma_dev:RDMADevice, remote_nic:RDMADevice, src_qp:MLXQP, dest_qp:MLXQP,
                     src_cq_buf:HCQBuffer, cq_buf:HCQBuffer, src_dev:AMDDevice, dest_dev:AMDDevice,
                     src_q:AMDComputeQueue, dest_q:AMDComputeQueue,
                     sq_dbr_val:sint, uar_db_val:sint, src_cq_dbr_val:sint, rq_dbr_val:sint, dest_cq_dbr_val:sint,
                     src_cq_ci:sint, dest_cq_ci:sint):
    for buf in [rdma_dev.iface.uar_buf, rdma_dev.iface.dbr_buf, src_cq_buf]: cast(HCQAllocator, src_dev.allocator).map(buf)
    for buf in [remote_nic.iface.dbr_buf, cq_buf]: cast(HCQAllocator, dest_dev.allocator).map(buf)

    # sender: ring SQ doorbell + UAR, poll send CQ completion, advance CQ CI
    src_q.write(rdma_dev.iface.dbr_buf.offset(src_qp.qp_dbr + 4), sq_dbr_val)
    src_q.write(rdma_dev.iface.uar_buf.offset(0x800), uar_db_val, b64=True)
    src_q.poll(HCQBuffer(src_cq_buf.va_addr + ((src_cq_ci & (src_qp.cq_size - 1)) * 64 + 60), 4),
               (((src_cq_ci >> 7) & 1) << 24), mask=0x01000000)
    src_q.write(rdma_dev.iface.dbr_buf.offset(src_qp.cq_dbr), src_cq_dbr_val)

    # receiver: post RQ doorbell, poll recv CQ completion, advance CQ CI
    dest_q.write(remote_nic.iface.dbr_buf.offset(dest_qp.qp_dbr), rq_dbr_val)
    dest_q.poll(HCQBuffer(cq_buf.va_addr + ((dest_cq_ci & (dest_qp.cq_size - 1)) * 64 + 60), 4),
                (((dest_cq_ci >> 7) & 1) << 24), mask=0x01000000)
    dest_q.write(remote_nic.iface.dbr_buf.offset(dest_qp.cq_dbr), dest_cq_dbr_val)

  def prepare_transfer(self, dest_buf:HCQBuffer, src_buf:HCQBuffer, sz:int,
                       src_dev:AMDDevice, dest_dev:AMDDevice, src_q:AMDComputeQueue, dest_q:AMDComputeQueue,
                       sq_dbr_val:sint, uar_db_val:sint, src_cq_dbr_val:sint, rq_dbr_val:sint, dest_cq_dbr_val:sint,
                       src_cq_ci:sint, dest_cq_ci:sint):
    rdma_dev = self.dev
    remote_nic = dest_dev.rdma_dev()

    # src_buf lkey must be on sender's NIC, dest_buf lkey must be on receiver's NIC
    rdma_dev.allocator.map(src_buf)
    remote_nic.allocator.map(dest_buf)
    src_mb = (src_buf._base or src_buf).mappings[rdma_dev]
    dest_mb = (dest_buf._base or dest_buf).mappings[remote_nic]

    src_qp, dest_qp, cq_buf, src_cq_buf = rdma_dev.iface.connect(remote_nic)

    src_off = src_mb.va_addr + (src_buf.va_addr - (src_buf._base or src_buf).va_addr)
    dest_off = dest_mb.va_addr + (dest_buf.va_addr - (dest_buf._base or dest_buf).va_addr)
    send_data = struct.pack('>IIQ', sz, src_mb.meta, src_off)
    recv_data = struct.pack('>IIQ', sz, dest_mb.meta, dest_off)

    self._q.append((src_qp, dest_qp, send_data, recv_data))

    RDMACopyQueue.build_rdma_mec(rdma_dev, remote_nic, src_qp, dest_qp, src_cq_buf, cq_buf,
                                 src_dev, dest_dev, src_q, dest_q,
                                 sq_dbr_val, uar_db_val, src_cq_dbr_val, rq_dbr_val, dest_cq_dbr_val,
                                 src_cq_ci, dest_cq_ci)

  def _submit(self, dev:RDMADevice):
    for src_qp, dest_qp, send_data, recv_data in self._q:
      assert src_qp.sq_head + 1 - src_qp.cq_ci <= (1 << src_qp.log_sq_size), "SQ ring full, need bigger ring"
      assert dest_qp.rq_head + 1 - dest_qp.cq_ci <= (1 << dest_qp.log_rq_size), "RQ ring full, need bigger ring"

      # recv WQE on dest QP's RQ
      rq_wqe = dest_qp.qp_buf.view((dest_qp.rq_head & ((1 << dest_qp.log_rq_size) - 1)) * 16, 16)
      rq_wqe[:] = recv_data
      dest_qp.rq_head += 1

      # send WQE on src QP's SQ (opcode 0x0a = SEND, ds_count=2)
      wqe = src_qp.qp_buf.view(src_qp.sq_offset + (src_qp.sq_head & ((1 << src_qp.log_sq_size) - 1)) * 64, 64)
      wqe[:] = b'\x00' * 64
      wqe[0:8] = struct.pack('>II', (src_qp.sq_head << 8) | 0x0a, (src_qp.qp_info['qpn'] << 8) | 2)
      wqe[11] = 0x08  # CE: signal completion
      wqe[16:32] = send_data
      src_qp.sq_head += 1

      src_qp.cq_ci += 1
      dest_qp.cq_ci += 1

class RDMADevice(HCQCompiled):
  def __init__(self, device:str=""):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.iface = MLXIface(self, self.device_id)
    super().__init__(device, RDMAAllocator(self), [], None, signal_t=None)
