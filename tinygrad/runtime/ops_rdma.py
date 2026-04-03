from __future__ import annotations
import struct
from tinygrad.helpers import round_up, DEBUG
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocatorBase, HWQueue
from tinygrad.runtime.support.system import System, PCIIfaceBase
from tinygrad.runtime.support.memory import AddrSpace
from tinygrad.runtime.support.mlx.mlxdev import MLXDev, MLXQP, to_be
from tinygrad.runtime.ops_amd import AMDComputeQueue, AMDDevice, WAIT_REG_MEM_FUNCTION_EQ

def map_phys_to_gpu(gpu, paddrs, size):
  if isinstance(paddrs, int): paddrs = [paddrs]
  size = round_up(size, 0x1000)
  va = gpu.iface.dev_impl.mm.alloc_vaddr(size, align=0x1000)
  gpu.iface.dev_impl.mm.map_range(va, size, [(p, 0x1000) for p in paddrs], aspace=AddrSpace.SYS, snooped=True, uncached=True)
  return va

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
    self.pci_dev = cl("mlx", pcibus)

class RDMACopyQueue(HWQueue):
  def __init__(self, dev:RDMADevice):
    self.dev = dev
    super().__init__()

  def copy(self, dest_paddr, src_paddr, copy_size, dest_dev:AMDDevice, src_dev:AMDDevice):
    mkey = self.dev.mlx_dev.mkey

    recv_data = struct.pack('>IIQ', copy_size, mkey, dest_paddr)
    send_data = struct.pack('>IIQ', copy_size, mkey, src_paddr)

    # pre-compute timeline values (capture both waits before any increment to avoid loopback deadlock)
    src_wait = src_dev.timeline_value - 1
    dest_wait = dest_dev.timeline_value - 1
    src_sig = src_dev.next_timeline()
    dest_sig = dest_dev.next_timeline()

    self._q.append((src_dev, dest_dev, recv_data, send_data, src_wait, src_sig, dest_wait, dest_sig))
    return self

  def _submit(self, dev:RDMADevice):
    for src_dev, dest_dev, recv_data, send_data, src_wait, src_sig, dest_wait, dest_sig in self._q:
      (src_qp, src_m), (dest_qp, dest_m) = dev.get_qp_pair(src_dev, dest_dev)

      # === CPU: write WQEs to MLX5 queues ===

      # recv WQE on dest QP's RQ (16 bytes scatter entry)
      rq_mask = (1 << 4) - 1  # log_rq_size=4
      rq_wqe = dest_qp.qp_buf.view((dest_qp.rq_head & rq_mask) * 16, 16)
      rq_wqe[:] = recv_data
      dest_qp.rq_head += 1

      # send WQE on src QP's SQ (64 bytes, opcode 0x0a = SEND, ds_count=2)
      sq_mask = (1 << src_qp.log_sq_size) - 1
      wqe = src_qp.qp_buf.view(src_qp.sq_offset + (src_qp.sq_head & sq_mask) * 64, 64)
      wqe[:] = bytes(64)
      wqe[0:8] = struct.pack('>II', (src_qp.sq_head << 8) | 0x0a, (src_qp.qp_info['qpn'] << 8) | 2)
      wqe[11] = 0x08  # CE: signal completion
      wqe[16:32] = send_data
      src_qp.sq_head += 1
      doorbell_val = to_be('Q', int.from_bytes(bytes(wqe[0:8]), 'big'))

      # CQ addresses for polling owner bit at cqe byte 63
      cq0_ci = src_qp.cq_ci
      cq0_addr = src_m['cq_gpu_va'] + (cq0_ci & (src_qp.cq_size - 1)) * 64 + 60
      cq0_owner = (1 if (cq0_ci & src_qp.cq_size) else 0) << 24
      cq1_ci = (dest_qp.cq_ci + 1) if src_dev == dest_dev else dest_qp.cq_ci  # loopback: same CQ, next entry
      cq1_addr = dest_m['cq_gpu_va'] + (cq1_ci & (dest_qp.cq_size - 1)) * 64 + 60
      cq1_owner = (1 if (cq1_ci & dest_qp.cq_size) else 0) << 24

      if src_dev == dest_dev:
        # loopback: single queue — recv doorbell before send to avoid RNR deadlock
        q = AMDComputeQueue(src_dev)
        q.wait(src_dev.timeline_signal, src_wait)
        q.release_mem(dest_m['dbr_gpu_va'] + dest_qp.qp_dbr, to_be('I', dest_qp.rq_head),
                      q.pm4.data_sel__mec_release_mem__send_32_bit_low, q.pm4.int_sel__mec_release_mem__none)
        q.release_mem(src_m['dbr_gpu_va'] + src_qp.qp_dbr + 4, to_be('I', src_qp.sq_head),
                      q.pm4.data_sel__mec_release_mem__send_32_bit_low, q.pm4.int_sel__mec_release_mem__none)
        q.release_mem(src_m['uar_gpu_va'] + 0x800, doorbell_val,
                      q.pm4.data_sel__mec_release_mem__send_64_bit_data, q.pm4.int_sel__mec_release_mem__none)
        q.wait_reg_mem(cq0_owner, mask=0x01000000, mem=cq0_addr, op=WAIT_REG_MEM_FUNCTION_EQ)
        q.wait_reg_mem(cq1_owner, mask=0x01000000, mem=cq1_addr, op=WAIT_REG_MEM_FUNCTION_EQ)
        q.signal(src_dev.timeline_signal, dest_sig)
        q.submit(src_dev)
      else:
        # multi-device: separate queues, each device controls its own
        src_q = AMDComputeQueue(src_dev)
        src_q.wait(src_dev.timeline_signal, src_wait)
        src_q.release_mem(src_m['dbr_gpu_va'] + src_qp.qp_dbr + 4, to_be('I', src_qp.sq_head),
                          src_q.pm4.data_sel__mec_release_mem__send_32_bit_low, src_q.pm4.int_sel__mec_release_mem__none)
        src_q.release_mem(src_m['uar_gpu_va'] + 0x800, doorbell_val,
                          src_q.pm4.data_sel__mec_release_mem__send_64_bit_data, src_q.pm4.int_sel__mec_release_mem__none)
        src_q.wait_reg_mem(cq0_owner, mask=0x01000000, mem=cq0_addr, op=WAIT_REG_MEM_FUNCTION_EQ)
        src_q.signal(src_dev.timeline_signal, src_sig)
        src_q.submit(src_dev)

        dest_q = AMDComputeQueue(dest_dev)
        dest_q.wait(dest_dev.timeline_signal, dest_wait)
        dest_q.release_mem(dest_m['dbr_gpu_va'] + dest_qp.qp_dbr, to_be('I', dest_qp.rq_head),
                           dest_q.pm4.data_sel__mec_release_mem__send_32_bit_low, dest_q.pm4.int_sel__mec_release_mem__none)
        dest_q.wait_reg_mem(cq1_owner, mask=0x01000000, mem=cq1_addr, op=WAIT_REG_MEM_FUNCTION_EQ)
        dest_q.signal(dest_dev.timeline_signal, dest_sig)
        dest_q.submit(dest_dev)

      src_qp.cq_ci += 1
      dest_qp.cq_ci += 1

class RDMADevice(HCQCompiled):
  def __init__(self, device:str=""):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.iface = MLXIface(self, self.device_id)
    self.mlx_dev = MLXDev(self.iface.pci_dev, ip=f"10.0.0.{self.device_id}")
    self.qp_pairs: dict[tuple[int, int], tuple[tuple[MLXQP, dict], tuple[MLXQP, dict]]] = {}

    super().__init__(device, RDMAAllocator(self), [], None, signal_t=None)

  def _map_qp_to_gpu(self, qp:MLXQP, gpu_dev:AMDDevice) -> dict:
    uar_paddr = self.mlx_dev.pci_dev.bar_info(0)[0] + self.mlx_dev.uar * 0x1000
    uar_gpu_va = map_phys_to_gpu(gpu_dev, uar_paddr, 0x1000)
    dbr_gpu_va = map_phys_to_gpu(gpu_dev, self.mlx_dev.dbr_paddrs[0], 0x1000)
    cq_gpu_va = map_phys_to_gpu(gpu_dev, qp.cq_paddrs, round_up(qp.cq_size * 64, 0x1000))
    for i in range(qp.cq_size): qp.cq_mem[i * 64 + 63] = 0x01
    return {'uar_gpu_va': uar_gpu_va, 'dbr_gpu_va': dbr_gpu_va, 'cq_gpu_va': cq_gpu_va}

  def get_qp_pair(self, src_dev:AMDDevice, dest_dev:AMDDevice) -> tuple[tuple[MLXQP, dict], tuple[MLXQP, dict]]:
    key = (src_dev.device_id, dest_dev.device_id)
    if key not in self.qp_pairs:
      gid = int.from_bytes(self.mlx_dev.local_gid, 'big')
      if src_dev.device_id == dest_dev.device_id:
        qp = MLXQP(self.mlx_dev)
        qp.connect(qp.qp_info['qpn'], self.mlx_dev.mac, gid)
        m = self._map_qp_to_gpu(qp, src_dev)
        self.qp_pairs[key] = ((qp, m), (qp, m))
        if DEBUG >= 1: print(f"RDMA: loopback QP 0x{qp.qp_info['qpn']:x} for GPU {src_dev.device_id}")
      else:
        src_qp, dest_qp = MLXQP(self.mlx_dev), MLXQP(self.mlx_dev)
        src_qp.connect(dest_qp.qp_info['qpn'], self.mlx_dev.mac, gid)
        dest_qp.connect(src_qp.qp_info['qpn'], self.mlx_dev.mac, gid)
        src_m, dest_m = self._map_qp_to_gpu(src_qp, src_dev), self._map_qp_to_gpu(dest_qp, dest_dev)
        self.qp_pairs[key] = ((src_qp, src_m), (dest_qp, dest_m))
        if DEBUG >= 1: print(f"RDMA: QP pair 0x{src_qp.qp_info['qpn']:x}<->0x{dest_qp.qp_info['qpn']:x} "
                             f"for GPU {src_dev.device_id}<->GPU {dest_dev.device_id}")
    return self.qp_pairs[key]
