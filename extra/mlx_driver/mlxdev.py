import struct, time, random, json, sys, socket, ctypes, os, functools
from tinygrad.helpers import getenv
from tinygrad.runtime.support.system import PCIDevice
from tinygrad.runtime.autogen import mlx5

MLX_DEBUG = getenv("MLX_DEBUG", 0)
CMD_IF_REV, MBOX_SZ, MBOX_STRIDE = 5, 512, 1024

def swap32(v): return ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v >> 8) & 0xFF00) | ((v >> 24) & 0xFF)

def ifc_get(buf, bit_off, width):
  byte_off, bit_in, n = bit_off // 8, bit_off % 8, (bit_off % 8 + width + 7) // 8
  return (int.from_bytes(buf[byte_off:byte_off + n], 'big') >> (n * 8 - bit_in - width)) & ((1 << width) - 1)

def ifc_set(buf, bit_off, width, value):
  byte_off, bit_in, n = bit_off // 8, bit_off % 8, (bit_off % 8 + width + 7) // 8
  shift, val = n * 8 - bit_in - width, int.from_bytes(buf[byte_off:byte_off + n], 'big')
  buf[byte_off:byte_off + n] = ((val & ~(((1 << width) - 1) << shift)) | ((value & ((1 << width) - 1)) << shift)).to_bytes(n, 'big')

@functools.cache
def ifc_fields(ifc_struct): return {name: (off, ctypes.sizeof(typ)) for name, typ, off in ifc_struct._real_fields_ if not name.startswith('reserved')}

def fill_ifc(buf, ifc_struct, base=0, **kwargs):
  for name, val in kwargs.items(): ifc_set(buf, base + (f:=ifc_fields(ifc_struct)[name])[0], f[1], val)

def read_ifc(buf, ifc_struct, field, base=0): return ifc_get(buf, base + (f:=ifc_fields(ifc_struct)[field])[0], f[1])

def pack_pas(buf, offset, paddrs):
  for i, pa in enumerate(paddrs): struct.pack_into('>Q', buf, offset + i * 8, pa)

class MLXDev:
  def __init__(self, pci_dev:PCIDevice):
    self.pci_dev, self.bar, self.token = pci_dev, pci_dev.map_bar(0, fmt='I'), 0
    fw_rev, cmdif_sub = self.rreg(0), self.rreg(4)
    print(f"mlx5: firmware {fw_rev >> 16}.{fw_rev & 0xFFFF}.{cmdif_sub & 0xFFFF}")
    assert (cmdif_sub >> 16) == CMD_IF_REV

    self.wait_fw_init()
    self._setup_cmd()
    self.wait_fw_init()
    self.cmd_exec(mlx5.MLX5_CMD_OP_ENABLE_HCA)
    self._set_issi()
    self._satisfy_pages(mlx5.MLX5_BOOT_PAGES)
    self._access_reg(mlx5.MLX5_REG_HOST_ENDIANNESS, bytearray(16))
    self._set_hca_caps()
    self._satisfy_pages(mlx5.MLX5_INIT_PAGES)
    self._init_hca()
    print("mlx5: INIT_HCA OK")

    self._query_hca_caps_post_init()
    self.uar = self._alloc_uar()
    self.pd, self.td = self._alloc_24(mlx5.MLX5_CMD_OP_ALLOC_PD), self._alloc_24(mlx5.MLX5_CMD_OP_ALLOC_TRANSPORT_DOMAIN)
    sc_out = self.cmd_exec(mlx5.MLX5_CMD_OP_QUERY_SPECIAL_CONTEXTS, out_sz=16)
    self.resd_lkey, self.null_mkey = struct.unpack_from('>I', sc_out, 4)[0], struct.unpack_from('>I', sc_out, 8)[0]
    self.mac = bytes(self.cmd_exec(mlx5.MLX5_CMD_OP_QUERY_NIC_VPORT_CONTEXT, out_sz=256)[8 + 0xF6:8 + 0xFC])
    self._enable_roce()

    self.dbr_mem, self.dbr_paddrs = self.pci_dev.alloc_sysmem(0x1000)
    self.dbr_offset = 0
    self.eq = self._create_eq()
    self.cq = self._create_cq()
    self.mkey = self._create_mkey()
    self.qpn = self._create_qp()
    self.cq_ci, self.sq_head, self.rq_head = 0, 0, 0
    if MLX_DEBUG >= 1:
      print(f"mlx5: MAC={':'.join(f'{b:02x}' for b in self.mac)} UAR={self.uar} PD=0x{self.pd:x} TD=0x{self.td:x}")
      print(f"mlx5: EQ={self.eq} CQ=0x{self.cq:x} MKEY=0x{self.mkey:x} QPN=0x{self.qpn:x}")

  def rreg(self, off): return swap32(self.bar[off // 4])
  def wreg(self, off, val): self.bar[off // 4] = swap32(val)

  def wait_fw_init(self, timeout=10.0):
    t = time.monotonic()
    while self.rreg(0x1FC) & 0x80000000:
      if time.monotonic() - t > timeout: raise TimeoutError("FW init timeout")
      time.sleep(0.001)

  def _dma_phys(self, off): return self.dma_paddrs[off // 0x1000] + (off % 0x1000)

  def _setup_mbox(self, n, base, tok):
    if n == 0: return 0
    for i in range(n):
      off = 0x1000 + (base + i) * MBOX_STRIDE
      self.dma.mv[off:off + 576] = bytes(576)
      struct.pack_into('>I', self.dma.mv, off + 568, i)
      self.dma.mv[off + 573] = tok
      if i < n - 1: struct.pack_into('>Q', self.dma.mv, off + 560, self._dma_phys(0x1000 + (base + i + 1) * MBOX_STRIDE))
    return self._dma_phys(0x1000 + base * MBOX_STRIDE)

  def _mbox_write(self, data, base, tok):
    for i in range((len(data) + MBOX_SZ - 1) // MBOX_SZ):
      off = 0x1000 + (base + i) * MBOX_STRIDE
      chunk = data[i * MBOX_SZ:(i + 1) * MBOX_SZ]
      self.dma.mv[off:off + len(chunk)] = chunk
      self.dma.mv[off + 573] = tok
      struct.pack_into('>I', self.dma.mv, off + 568, i)

  def _mbox_read(self, size, base):
    return b''.join(bytes(self.dma.mv[(off:=0x1000 + i * MBOX_STRIDE):off + min(MBOX_SZ, size - i * MBOX_SZ)])
                    for i in range((size + MBOX_SZ - 1) // MBOX_SZ))

  def _setup_cmd(self):
    from tinygrad.runtime.autogen import pci
    self.pci_dev.write_config(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER, 2)
    cmd_l = self.rreg(0x14) & 0xFF
    self.log_sz, self.log_stride, self.max_reg_cmds = (cmd_l >> 4) & 0xF, cmd_l & 0xF, (1 << ((cmd_l >> 4) & 0xF)) - 1
    self.dma, self.dma_paddrs = self.pci_dev.alloc_sysmem(0x1000 + 128 * MBOX_STRIDE)
    self.wreg(0x10, self.dma_paddrs[0] >> 32)
    self.wreg(0x14, (self.dma_paddrs[0] & 0xFFFFFFFF) | cmd_l)
    self.fw_pages = []

  def cmd_exec(self, opcode, op_mod=0, inp=b'', out_sz=0, page_queue=False):
    self.token = (self.token % 255) + 1
    tok, slot = self.token, self.max_reg_cmds if page_queue else 0
    inlen, outlen = max(16, 8 + len(inp)), 16 + out_sz

    hdr = bytearray(16)
    struct.pack_into('>HH', hdr, 0, opcode, 0)
    struct.pack_into('>HH', hdr, 4, 0, op_mod)
    hdr[8:8 + min(8, len(inp))] = inp[:min(8, len(inp))]

    mbox_in = inp[8:] if len(inp) > 8 else b''
    n_in = (len(mbox_in) + MBOX_SZ - 1) // MBOX_SZ if mbox_in else 0
    n_out = (out_sz + MBOX_SZ - 1) // MBOX_SZ if out_sz > 0 else 0
    in_ptr, out_ptr = self._setup_mbox(n_in, 0, tok), self._setup_mbox(n_out, n_in, tok)
    if mbox_in: self._mbox_write(mbox_in, 0, tok)

    lay = slot << self.log_stride
    self.dma.mv[lay:lay + 64] = bytes(64)
    self.dma.mv[lay] = mlx5.MLX5_PCI_CMD_XPORT
    struct.pack_into('>I', self.dma.mv, lay + 4, inlen)
    struct.pack_into('>Q', self.dma.mv, lay + 8, in_ptr)
    self.dma.mv[lay + 16:lay + 32] = hdr
    struct.pack_into('>Q', self.dma.mv, lay + 48, out_ptr)
    struct.pack_into('>I', self.dma.mv, lay + 56, outlen)
    self.dma.mv[lay + 60], self.dma.mv[lay + 63] = tok, mlx5.CMD_OWNER_HW

    xor_val = 0
    for i in range(64): xor_val ^= self.dma.mv[lay + i]
    self.dma.mv[lay + 61] = (~xor_val) & 0xFF

    if MLX_DEBUG >= 2:
      d = bytes(self.dma.mv[lay:lay + 64])
      print(f"  CMD[{slot}] op=0x{opcode:04x} mod=0x{op_mod:04x} inlen={inlen} outlen={outlen} tok={tok}")
      print(f"  LAY: {d[:32].hex(' ')}"); print(f"  LAY: {d[32:].hex(' ')}")

    self.wreg(0x18, 1 << slot)
    t = time.monotonic()
    while self.dma.mv[lay + 63] & mlx5.CMD_OWNER_HW:
      if time.monotonic() - t > 60.0: raise TimeoutError(f"cmd 0x{opcode:04x} timeout")
      time.sleep(0.0001)

    delivery = self.dma.mv[lay + 63] >> 1
    out_hdr = [struct.unpack_from('>I', self.dma.mv, lay + 32 + i * 4)[0] for i in range(4)]
    status, syndrome = out_hdr[0] >> 24, out_hdr[1]
    if MLX_DEBUG >= 2: print(f"  DONE[{slot}] status=0x{status:02x} syn=0x{syndrome:08x} delivery=0x{delivery:x}")
    assert delivery == 0, f"cmd 0x{opcode:04x} delivery error 0x{delivery:x}"
    assert status == 0, f"cmd 0x{opcode:04x} failed status=0x{status:x} syn=0x{syndrome:08x}"
    return bytearray(struct.pack('>II', out_hdr[2], out_hdr[3])) + (self._mbox_read(out_sz, n_in) if out_sz > 0 else b'')

  def _query_cap(self, cap_type, mode): return bytearray(self.cmd_exec(mlx5.MLX5_CMD_OP_QUERY_HCA_CAP, op_mod=(cap_type << 1) | mode, out_sz=4096)[8:])
  def _set_cap(self, cap_type, data): self.cmd_exec(mlx5.MLX5_CMD_OP_SET_HCA_CAP, op_mod=cap_type << 1, inp=bytearray(8) + data)
  def _alloc_24(self, op): return struct.unpack_from('>I', self.cmd_exec(op), 0)[0] & 0xFFFFFF

  def _set_issi(self):
    if struct.unpack_from('>I', self.cmd_exec(mlx5.MLX5_CMD_OP_QUERY_ISSI, out_sz=96), 100)[0] & 2:
      inp = bytearray(8); struct.pack_into('>I', inp, 0, 1); self.cmd_exec(mlx5.MLX5_CMD_OP_SET_ISSI, inp=inp)

  def _satisfy_pages(self, mode):
    npages = struct.unpack_from('>i', self.cmd_exec(mlx5.MLX5_CMD_OP_QUERY_PAGES, op_mod=mode), 4)[0]
    if MLX_DEBUG >= 1: print(f"mlx5: {'boot' if mode == mlx5.MLX5_BOOT_PAGES else 'init'} pages: {npages}")
    if npages <= 0: return
    mem, paddrs = self.pci_dev.alloc_sysmem(npages * 0x1000)
    self.fw_pages.append((mem, paddrs))
    inp = bytearray(8 + npages * 8)
    struct.pack_into('>I', inp, 4, npages)
    pack_pas(inp, 8, paddrs)
    self.cmd_exec(mlx5.MLX5_CMD_OP_MANAGE_PAGES, op_mod=mlx5.MLX5_PAGES_GIVE, inp=inp, page_queue=True)

  def _access_reg(self, reg_id, data, write=True):
    inp = bytearray(8 + len(data))
    struct.pack_into('>HH', inp, 0, 0, reg_id)
    inp[8:8 + len(data)] = data
    return self.cmd_exec(mlx5.MLX5_CMD_OP_ACCESS_REG, op_mod=int(not write), inp=inp, out_sz=len(data))[8:]

  def _set_hca_caps(self):
    CAP = mlx5.struct_mlx5_ifc_cmd_hca_cap_bits
    gen_max, gen_cur = self._query_cap(mlx5.MLX5_CAP_GENERAL, 0), self._query_cap(mlx5.MLX5_CAP_GENERAL, 1)
    self.gen_caps, cap = gen_cur, bytearray(gen_cur)
    fill_ifc(cap, CAP, pkey_table_size=0, cmdif_checksum=0, log_uar_page_sz=os.sysconf('SC_PAGE_SIZE').bit_length() - 1 - 12,
             log_max_qp=min(18, read_ifc(gen_max, CAP, 'log_max_qp')))
    if read_ifc(gen_max, CAP, 'uar_4k') and os.sysconf('SC_PAGE_SIZE') > 4096: fill_ifc(cap, CAP, uar_4k=1)
    if read_ifc(gen_max, CAP, 'cache_line_128byte'):
      try: fill_ifc(cap, CAP, cache_line_128byte=1 if os.sysconf('SC_LEVEL1_DCACHE_LINESIZE') >= 128 else 0)
      except (ValueError, OSError): fill_ifc(cap, CAP, cache_line_128byte=0)
    for f in ['dct', 'pci_sync_for_fw_update_event', 'pci_sync_for_fw_update_with_driver_unload',
              'pcie_reset_using_hotreset_method', 'release_all_pages', 'mkey_by_name', 'abs_native_port_num']:
      if read_ifc(gen_max, CAP, f): fill_ifc(cap, CAP, **{f: 1})
    if (vhca_ports:=read_ifc(gen_max, CAP, 'num_vhca_ports')): fill_ifc(cap, CAP, num_vhca_ports=vhca_ports)
    if read_ifc(gen_max, CAP, 'vhca_state'):
      fill_ifc(cap, CAP, event_on_vhca_state_teardown_request=1, event_on_vhca_state_in_use=1,
               event_on_vhca_state_active=1, event_on_vhca_state_allocated=1)
    if (max_msix:=read_ifc(gen_max, CAP, 'num_total_dynamic_vf_msix')): fill_ifc(cap, CAP, num_total_dynamic_vf_msix=max_msix)
    if read_ifc(gen_cur, CAP, 'roce_rw_supported') and read_ifc(gen_max, CAP, 'roce'): fill_ifc(cap, CAP, roce=1)
    self._set_cap(mlx5.MLX5_CAP_GENERAL, cap)

    if read_ifc(gen_cur, CAP, 'atomic'):
      self._query_cap(mlx5.MLX5_CAP_ATOMIC, 0); ATOM = mlx5.struct_mlx5_ifc_atomic_caps_bits
      if read_ifc(self._query_cap(mlx5.MLX5_CAP_ATOMIC, 1), ATOM, 'supported_atomic_req_8B_endianness_mode_1'):
        cap = bytearray(4096); fill_ifc(cap, ATOM, atomic_req_8B_endianness_mode=1); self._set_cap(mlx5.MLX5_CAP_ATOMIC, cap)

    if read_ifc(gen_cur, CAP, 'pg'):
      odp_max, odp_cur = self._query_cap(mlx5.MLX5_CAP_ODP, 0), self._query_cap(mlx5.MLX5_CAP_ODP, 1)
      cap, modified = bytearray(odp_cur), False
      if ifc_get(odp_max, 0x600, 1) and ifc_get(odp_max, 0x245, 1): ifc_set(cap, 0x600, 1, 1); modified = True
      else:
        for off in [0xC5, 0x85, 0xE5, 0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0x105, 0x100, 0x101, 0x102, 0x103, 0x104]:
          if ifc_get(odp_max, off, 1): ifc_set(cap, off, 1, 1); modified = True
      if modified: self._set_cap(mlx5.MLX5_CAP_ODP, cap)

    ROCE = mlx5.struct_mlx5_ifc_roce_cap_bits
    roce_max, roce_cur = self._query_cap(mlx5.MLX5_CAP_ROCE, 0), self._query_cap(mlx5.MLX5_CAP_ROCE, 1)
    if not read_ifc(roce_cur, ROCE, 'sw_r_roce_src_udp_port') and read_ifc(roce_max, ROCE, 'sw_r_roce_src_udp_port'):
      cap = bytearray(roce_cur); fill_ifc(cap, ROCE, sw_r_roce_src_udp_port=1)
      if read_ifc(roce_max, ROCE, 'qp_ooo_transmit_default'): fill_ifc(cap, ROCE, qp_ooo_transmit_default=1)
      self._set_cap(mlx5.MLX5_CAP_ROCE, cap)

  def _init_hca(self):
    inp = bytearray(24)
    if read_ifc(self.gen_caps, mlx5.struct_mlx5_ifc_cmd_hca_cap_bits, 'sw_owner_id'):
      for i in range(4): struct.pack_into('>I', inp, 8 + i * 4, random.getrandbits(32))
    self.cmd_exec(mlx5.MLX5_CMD_OP_INIT_HCA, inp=inp)

  def _query_hca_caps_post_init(self):
    self.gen_caps, CAP = self._query_cap(mlx5.MLX5_CAP_GENERAL, 1), mlx5.struct_mlx5_ifc_cmd_hca_cap_bits
    for cap, field in [(mlx5.MLX5_CAP_ATOMIC, 'atomic'), (mlx5.MLX5_CAP_ODP, 'pg'), (mlx5.MLX5_CAP_ROCE, 'roce')]:
      if read_ifc(self.gen_caps, CAP, field): self._query_cap(cap, 1)

  def _enable_roce(self):
    inp = bytearray(504)
    fill_ifc(inp, mlx5.struct_mlx5_ifc_modify_nic_vport_field_select_bits, base=4*8, roce_en=1)
    fill_ifc(inp, mlx5.struct_mlx5_ifc_nic_vport_context_bits, base=248*8, roce_en=1)
    self.cmd_exec(mlx5.MLX5_CMD_OP_MODIFY_NIC_VPORT_CONTEXT, inp=inp)

  def _alloc_uar(self):
    uar = struct.unpack_from('>I', self.cmd_exec(mlx5.MLX5_CMD_OP_ALLOC_UAR), 0)[0] & 0xFFFFFF
    self.uar_page = self.pci_dev.map_bar(0, off=uar * (ps:=os.sysconf('SC_PAGE_SIZE')), size=ps, fmt='I')
    return uar

  def _alloc_dbr(self):
    assert self.dbr_offset + 8 <= 0x1000
    phys = self.dbr_paddrs[0] + self.dbr_offset; self.dbr_offset += 8
    return phys

  def _create_eq(self, log_eq_size=7):
    n_pages = ((1 << log_eq_size) * 64 + 0xFFF) // 0x1000
    self.eq_mem, eq_paddrs = self.pci_dev.alloc_sysmem(n_pages * 0x1000)
    for i in range(1 << log_eq_size): self.eq_mem.mv[i * 64 + 31] = 0x01
    inp = bytearray(264 + n_pages * 8)
    fill_ifc(memoryview(inp)[8:72], mlx5.struct_mlx5_ifc_eqc_bits, log_eq_size=log_eq_size, uar_page=self.uar, log_page_size=0)
    pack_pas(inp, 264, eq_paddrs)
    return struct.unpack_from('>I', self.cmd_exec(mlx5.MLX5_CMD_OP_CREATE_EQ, inp=inp), 0)[0] & 0xFF

  def _create_cq(self, log_cq_size=7):
    n_pages = ((1 << log_cq_size) * 64 + 0xFFF) // 0x1000
    self.cq_mem, cq_paddrs = self.pci_dev.alloc_sysmem(n_pages * 0x1000)
    for i in range(1 << log_cq_size): self.cq_mem.mv[i * 64 + 63] = 0x01
    self.cq_dbr_phys = self._alloc_dbr()
    inp = bytearray(264 + n_pages * 8)
    fill_ifc(memoryview(inp)[8:72], mlx5.struct_mlx5_ifc_cqc_bits,
             log_cq_size=log_cq_size, uar_page=self.uar, c_eqn_or_apu_element=self.eq, log_page_size=0, dbr_addr=self.cq_dbr_phys)
    pack_pas(inp, 264, cq_paddrs)
    return struct.unpack_from('>I', self.cmd_exec(mlx5.MLX5_CMD_OP_CREATE_CQ, inp=inp), 0)[0] & 0xFFFFFF

  def _create_mkey(self):
    inp = bytearray(264)
    fill_ifc(memoryview(inp)[8:72], mlx5.struct_mlx5_ifc_mkc_bits,
             access_mode_1_0=0, rw=1, rr=1, lw=1, lr=1, qpn=0xFFFFFF, mkey_7_0=0x42, length64=1, pd=self.pd)
    return (struct.unpack_from('>I', self.cmd_exec(mlx5.MLX5_CMD_OP_CREATE_MKEY, inp=inp), 0)[0] & 0xFFFFFF) << 8 | 0x42

  def _qp_cmd(self, opcode, qpn, qpc_kwargs=None, ads_kwargs=None, extra_inp=None):
    inp = extra_inp if extra_inp is not None else bytearray(264)
    struct.pack_into('>I', inp, 0, qpn)
    qpc = memoryview(inp)[16:248]
    if qpc_kwargs: fill_ifc(qpc, mlx5.struct_mlx5_ifc_qpc_bits, st=0, pm_state=3, pd=self.pd, cqn_snd=self.cq, cqn_rcv=self.cq, **qpc_kwargs)
    if ads_kwargs: fill_ifc(qpc, mlx5.struct_mlx5_ifc_ads_bits, base=0xC0, **ads_kwargs)
    self.cmd_exec(opcode, inp=inp)

  def _create_qp(self, log_sq_size=4, log_rq_size=4):
    rq_sz, n_pages = (1 << log_rq_size) << 4, ((1 << log_rq_size) * 16 + (1 << log_sq_size) * 64 + 0xFFF) // 0x1000
    self.qp_buf, qp_paddrs = self.pci_dev.alloc_sysmem(n_pages * 0x1000)
    self.qp_dbr_phys, self.sq_offset = self._alloc_dbr(), rq_sz
    inp = bytearray(264 + n_pages * 8)
    fill_ifc(memoryview(inp)[16:248], mlx5.struct_mlx5_ifc_qpc_bits, st=0, pm_state=3, pd=self.pd, cqn_snd=self.cq, cqn_rcv=self.cq,
             log_msg_max=30, log_rq_size=log_rq_size, log_rq_stride=0, log_sq_size=log_sq_size,
             rlky=1, uar_page=self.uar, log_page_size=0, dbr_addr=self.qp_dbr_phys)
    pack_pas(inp, 264, qp_paddrs)
    qpn = struct.unpack_from('>I', self.cmd_exec(mlx5.MLX5_CMD_OP_CREATE_QP, inp=inp), 0)[0] & 0xFFFFFF
    self._qp_cmd(mlx5.MLX5_CMD_OP_RST2INIT_QP, qpn, qpc_kwargs=dict(log_ack_req_freq=8), ads_kwargs=dict(pkey_index=0, vhca_port_num=1))
    if MLX_DEBUG >= 1: print(f"mlx5: QP 0x{qpn:x} RST -> INIT")
    return qpn

  @staticmethod
  def _ipv4_to_gid(ip): return bytes(10) + b'\xff\xff' + socket.inet_aton(ip)

  @staticmethod
  def _calc_udp_sport(lqpn, rqpn):
    v = (lqpn * rqpn ^ ((lqpn * rqpn) >> 20) ^ ((lqpn * rqpn) >> 40)) & 0xFFFFF
    return ((v & 0x3FFF) ^ ((v & 0xFC000) >> 14)) | 0xC000

  def set_roce_address(self, gid_index, ip):
    inp = bytearray(40)
    struct.pack_into('>H', inp, 0, gid_index); inp[3] = 1
    ra = memoryview(inp)[8:]
    ra[0:16] = self._ipv4_to_gid(ip)
    ra[18:20], ra[20:24] = self.mac[0:2], self.mac[2:6]
    fill_ifc(ra, mlx5.struct_mlx5_ifc_roce_addr_layout_bits, roce_l3_type=0, roce_version=2)
    self.cmd_exec(mlx5.MLX5_CMD_OP_SET_ROCE_ADDRESS, inp=inp)
    self.local_gid = self._ipv4_to_gid(ip)

  def connection_info(self): return {"qpn": self.qpn, "mac": self.mac.hex(), "gid": self.local_gid.hex()}

  def init2rtr(self, remote_qpn, remote_mac, remote_gid):
    if isinstance(remote_mac, str): remote_mac = bytes.fromhex(remote_mac)
    if isinstance(remote_gid, str): remote_gid = bytes.fromhex(remote_gid)
    inp = bytearray(264)
    struct.pack_into('>I', inp, 0, self.qpn); struct.pack_into('>I', inp, 8, 0x1A)
    qpc = memoryview(inp)[16:248]
    fill_ifc(qpc, mlx5.struct_mlx5_ifc_qpc_bits, st=0, pm_state=3, pd=self.pd, cqn_snd=self.cq, cqn_rcv=self.cq,
             mtu=3, log_msg_max=read_ifc(self.gen_caps, mlx5.struct_mlx5_ifc_cmd_hca_cap_bits, 'log_max_msg'),
             remote_qpn=remote_qpn, log_ack_req_freq=8, log_rra_max=3, rre=1, rwe=1, min_rnr_nak=12, next_rcv_psn=0)
    fill_ifc(qpc, mlx5.struct_mlx5_ifc_ads_bits, base=0xC0, pkey_index=0, src_addr_index=0, hop_limit=64,
             udp_sport=self._calc_udp_sport(self.qpn, remote_qpn), vhca_port_num=1,
             rmac_47_32=int.from_bytes(remote_mac[0:2], 'big'), rmac_31_0=int.from_bytes(remote_mac[2:6], 'big'))
    qpc[(0xC0 + 0x80) // 8:(0xC0 + 0x80) // 8 + 16] = remote_gid
    self.cmd_exec(mlx5.MLX5_CMD_OP_INIT2RTR_QP, inp=inp)
    if MLX_DEBUG >= 1: print(f"mlx5: QP 0x{self.qpn:x} INIT -> RTR (remote_qpn=0x{remote_qpn:x})")

  def rtr2rts(self):
    self._qp_cmd(mlx5.MLX5_CMD_OP_RTR2RTS_QP, self.qpn,
      qpc_kwargs=dict(log_ack_req_freq=8, next_send_psn=0, log_sra_max=3, retry_count=7, rnr_retry=7),
      ads_kwargs=dict(ack_timeout=14, vhca_port_num=1))
    if MLX_DEBUG >= 1: print(f"mlx5: QP 0x{self.qpn:x} RTR -> RTS")

  def _post_sq(self, opcode, ds_count, segs):
    wqe = memoryview(self.qp_buf.mv)[(wo:=self.sq_offset + (self.sq_head & 0xF) * 64):wo + 64]
    wqe[:64] = bytes(64)
    struct.pack_into('>II', wqe, 0, (self.sq_head << 8) | opcode, (self.qpn << 8) | ds_count)
    wqe[11] = 0x08
    for off, data in segs: wqe[off:off + len(data)] = data
    self.sq_head += 1
    struct.pack_into('>I', self.dbr_mem.mv, self.qp_dbr_phys - self.dbr_paddrs[0] + 4, self.sq_head)
    self.uar_page[0x800 // 4] = swap32(struct.unpack_from('>I', wqe, 0)[0])
    self.uar_page[0x800 // 4 + 1] = swap32(struct.unpack_from('>I', wqe, 4)[0])
    self._poll_cq()

  def rdma_write(self, remote_addr, rkey, local_addr, lkey, length):
    self._post_sq(0x08, 3, [(16, struct.pack('>QI4x', remote_addr, rkey)), (32, struct.pack('>IIQ', length, lkey, local_addr))])

  def send(self, addr, lkey, length): self._post_sq(0x0a, 2, [(16, struct.pack('>IIQ', length, lkey, addr))])

  def post_recv(self, addr, lkey, length):
    struct.pack_into('>IIQ', self.qp_buf.mv, (self.rq_head & 0xF) * 16, length, lkey, addr)
    self.rq_head += 1
    struct.pack_into('>I', self.dbr_mem.mv, self.qp_dbr_phys - self.dbr_paddrs[0], self.rq_head & 0xFFFF)

  def _poll_cq(self, timeout=5.0):
    t, cq_size = time.monotonic(), 1 << 7
    while True:
      cqe = bytes(self.cq_mem.mv[(idx:=self.cq_ci & (cq_size - 1)) * 64:(idx + 1) * 64])
      opcode, owner = cqe[63] >> 4, cqe[63] & 1
      if opcode != 0x0F and owner == (1 if (self.cq_ci & cq_size) else 0):
        self.cq_ci += 1
        struct.pack_into('>I', self.dbr_mem.mv, self.cq_dbr_phys - self.dbr_paddrs[0], self.cq_ci & 0xFFFFFF)
        if opcode in (13, 14): raise RuntimeError(f"CQE error: opcode={opcode} syndrome=0x{cqe[55]:02x} vendor=0x{cqe[54]:02x}")
        return opcode
      if time.monotonic() - t > timeout: raise TimeoutError("CQ poll timeout")
      time.sleep(0.0001)

if __name__ == "__main__":
  dev = MLXDev(PCIDevice("mlx5", getenv("MLX_PCI", "0000:41:00.0")))
  ip = sys.argv[sys.argv.index("--ip") + 1] if "--ip" in sys.argv else None
  if "--server" in sys.argv:
    dev.set_roce_address(0, ip)
    print(json.dumps(dev.connection_info()), flush=True)
    remote = json.loads(input())
    dev.init2rtr(remote["qpn"], remote["mac"], remote["gid"])
    dev.rtr2rts()
    print("connected", flush=True)
    target_mem, target_paddrs = dev.pci_dev.alloc_sysmem(0x1000)
    print(json.dumps({"target_addr": target_paddrs[0], "rkey": dev.mkey}), flush=True)
    input()
    data = bytes(target_mem[i] for i in range(64))
    print(f"RECEIVED: {data.hex(' ')}\nAS TEXT: {data.rstrip(b'\\x00').decode('ascii', errors='replace')}", flush=True)
