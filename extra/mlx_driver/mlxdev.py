import struct, time, random, json, sys, socket, ctypes, os, functools
from tinygrad.helpers import getenv, wait_cond, next_power2
from tinygrad.runtime.support.system import PCIDevice
from tinygrad.runtime.autogen import mlx5, pci

MLX_DEBUG = getenv("MLX_DEBUG", 0)

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
def fill_ifc(buf, ifc_struct, base=0, **kw):
  for name, val in kw.items():
    off, st = base, ifc_struct
    for part in name.split('__'):
      f = ifc_fields(st)
      if part == name.split('__')[-1]: ifc_set(buf, off + f[part][0], f[part][1], val)
      else: off += f[part][0]; st = _nested_type(st, part)
@functools.cache
def _nested_type(parent, field_name):
  for name, typ, off in parent._real_fields_:
    if name == field_name: return typ
  raise KeyError(f"no nested field '{field_name}' in {parent}")
def read_ifc(buf, ifc_struct, field, base=0): return ifc_get(buf, base + (f:=ifc_fields(ifc_struct)[field])[0], f[1])
def ifc_decode(buf, ifc_struct, base=0): return {name: ifc_get(buf, base + off, w) for name, (off, w) in ifc_fields(ifc_struct).items()}

class MLXCmdQueue:
  MBOX_SZ, MBOX_STRIDE = mlx5.MLX5_CMD_DATA_BLOCK_SIZE, next_power2(ctypes.sizeof(mlx5.struct_mlx5_cmd_prot_block))

  def __init__(self, dev):
    self.dev, self.token = dev, 0

    cmd_l = dev.iseg_r('cmdq_addr_l_sz') & 0xFF
    self.log_sz, self.log_stride, self.max_reg_cmds = (cmd_l >> 4) & 0xF, cmd_l & 0xF, (1 << ((cmd_l >> 4) & 0xF)) - 1

    self.dma, self.dma_paddrs = dev.pci_dev.alloc_sysmem(0x1000 + 128 * MLXCmdQueue.MBOX_STRIDE)
    dev.iseg_w('cmdq_addr_h', self.dma_paddrs[0] >> 32)
    dev.iseg_w('cmdq_addr_l_sz', (self.dma_paddrs[0] & 0xFFFFFFFF) | cmd_l)

  def _dma_phys(self, off): return self.dma_paddrs[off // 0x1000] + (off % 0x1000)

  def _setup_mbox(self, n, base, tok):
    if n == 0: return 0
    for i in range(n):
      off = 0x1000 + (base + i) * self.MBOX_STRIDE
      self.dma.mv[off:off + 576] = bytes(576)
      struct.pack_into('>I', self.dma.mv, off + 568, i)
      self.dma.mv[off + 573] = tok
      if i < n - 1: struct.pack_into('>Q', self.dma.mv, off + 560, self._dma_phys(0x1000 + (base + i + 1) * self.MBOX_STRIDE))
    return self._dma_phys(0x1000 + base * self.MBOX_STRIDE)

  def _mbox_write(self, data, base, tok):
    for i in range((len(data) + mlx5.MLX5_CMD_DATA_BLOCK_SIZE - 1) // mlx5.MLX5_CMD_DATA_BLOCK_SIZE):
      off = 0x1000 + (base + i) * self.MBOX_STRIDE
      chunk = data[i * mlx5.MLX5_CMD_DATA_BLOCK_SIZE:(i + 1) * mlx5.MLX5_CMD_DATA_BLOCK_SIZE]
      self.dma.mv[off:off + len(chunk)] = chunk
      self.dma.mv[off + 573] = tok
      struct.pack_into('>I', self.dma.mv, off + 568, i)

  def _mbox_read(self, size, base):
    return b''.join(bytes(self.dma.mv[(off:=0x1000 + (base + i) * self.MBOX_STRIDE):off + min(mlx5.MLX5_CMD_DATA_BLOCK_SIZE, size - i * mlx5.MLX5_CMD_DATA_BLOCK_SIZE)])
                    for i in range((size + mlx5.MLX5_CMD_DATA_BLOCK_SIZE - 1) // mlx5.MLX5_CMD_DATA_BLOCK_SIZE))

  def exec(self, opcode, op_mod=0, inp=b'', out_sz=0, page_queue=False, in_struct=None, out_struct=None, _payload=b'', **kw):
    if in_struct is not None:
      fields = ifc_fields(in_struct)
      inp_sz = max(0, (max((off + w for off, w in fields.values()), default=0) + 7) // 8 - 8)
      inp = bytearray(inp_sz + len(_payload))
      if kw: fill_ifc(inp, in_struct, base=-0x40, **kw)
      if _payload: inp[inp_sz:] = _payload
    if out_struct is not None: out_sz = max(0, ctypes.sizeof(out_struct) - 16)
    self.token = (self.token % 255) + 1
    tok, slot = self.token, self.max_reg_cmds if page_queue else 0
    inlen, outlen = max(16, 8 + len(inp)), 16 + out_sz

    hdr = bytearray(16)
    struct.pack_into('>HH', hdr, 0, opcode, 0)
    struct.pack_into('>HH', hdr, 4, 0, op_mod)
    hdr[8:8 + min(8, len(inp))] = inp[:min(8, len(inp))]

    mbox_in = inp[8:] if len(inp) > 8 else b''
    n_in = (len(mbox_in) + mlx5.MLX5_CMD_DATA_BLOCK_SIZE - 1) // mlx5.MLX5_CMD_DATA_BLOCK_SIZE if mbox_in else 0
    n_out = (out_sz + mlx5.MLX5_CMD_DATA_BLOCK_SIZE - 1) // mlx5.MLX5_CMD_DATA_BLOCK_SIZE if out_sz > 0 else 0
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

    self.dev.iseg_w('cmd_dbell', 1 << slot)
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
    raw = bytearray(struct.pack('>II', out_hdr[2], out_hdr[3])) + (self._mbox_read(out_sz, n_in) if out_sz > 0 else b'')
    return ifc_decode(raw, out_struct, base=-0x40) if out_struct else raw

class MLXDev:
  def __init__(self, pci_dev:PCIDevice):
    self.pci_dev, self.bar = pci_dev, pci_dev.map_bar(0, fmt='I')
    fw_rev, cmdif_sub = self.iseg_r('fw_rev'), self.iseg_r('cmdif_rev_fw_sub')
    print(f"mlx5: firmware {fw_rev >> 16}.{fw_rev & 0xFFFF}.{cmdif_sub & 0xFFFF}")
    assert (cmdif_sub >> 16) == 5

    self.fw_pages = []
    wait_cond(lambda: self.iseg_r('initializing') & 0x80000000, value=0, msg="FW init timeout")
    self.pci_dev.write_config(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER, 2)
    self.cmd = MLXCmdQueue(self)
    wait_cond(lambda: self.iseg_r('initializing') & 0x80000000, value=0, msg="FW init timeout")

    self.cmd.exec(mlx5.MLX5_CMD_OP_ENABLE_HCA)
    self._set_issi()
    self._satisfy_pages(mlx5.MLX5_BOOT_PAGES)
    self._access_reg(mlx5.MLX5_REG_HOST_ENDIANNESS, bytearray(16))
    self._set_hca_caps()
    self._satisfy_pages(mlx5.MLX5_INIT_PAGES)
    self._init_hca()
    print("mlx5: INIT_HCA OK")

    self._query_hca_caps_post_init()
    self.uar = self._alloc_uar()
    self.pd = self.cmd.exec(mlx5.MLX5_CMD_OP_ALLOC_PD, out_struct=mlx5.struct_mlx5_ifc_alloc_pd_out_bits)['pd']
    self.td = self.cmd.exec(mlx5.MLX5_CMD_OP_ALLOC_TRANSPORT_DOMAIN, out_struct=mlx5.struct_mlx5_ifc_alloc_transport_domain_out_bits)['transport_domain']
    sc = self.cmd.exec(mlx5.MLX5_CMD_OP_QUERY_SPECIAL_CONTEXTS, out_struct=mlx5.struct_mlx5_ifc_query_special_contexts_out_bits)
    self.resd_lkey, self.null_mkey = sc['resd_lkey'], sc['null_mkey']
    self.mac = bytes(self.cmd.exec(mlx5.MLX5_CMD_OP_QUERY_NIC_VPORT_CONTEXT, out_sz=256)[8 + 0xF6:8 + 0xFC])
    self._enable_roce()
    self.dbr_mem, self.dbr_paddrs, self.dbr_offset = *self.pci_dev.alloc_sysmem(0x1000), 0
    self.mkey = self._create_mkey()
    if MLX_DEBUG >= 1: print(f"mlx5: MAC={':'.join(f'{b:02x}' for b in self.mac)} UAR={self.uar} PD=0x{self.pd:x} MKEY=0x{self.mkey:x}")

  def rreg(self, off): return swap32(self.bar[off // 4])
  def wreg(self, off, val): self.bar[off // 4] = swap32(val)
  def iseg_r(self, field): return self.rreg(getattr(mlx5.struct_mlx5_init_seg, field).offset)
  def iseg_w(self, field, val): self.wreg(getattr(mlx5.struct_mlx5_init_seg, field).offset, val)

  def _alloc_dbr(self):
    assert self.dbr_offset + 8 <= 0x1000
    phys = self.dbr_paddrs[0] + self.dbr_offset; self.dbr_offset += 8; return phys

  def _set_issi(self):
    if self.cmd.exec(mlx5.MLX5_CMD_OP_QUERY_ISSI, out_struct=mlx5.struct_mlx5_ifc_query_issi_out_bits)['supported_issi_dw0'] & 2:
      self.cmd.exec(mlx5.MLX5_CMD_OP_SET_ISSI, in_struct=mlx5.struct_mlx5_ifc_set_issi_in_bits, current_issi=1)

  def _satisfy_pages(self, mode):
    if (npages:=self.cmd.exec(mlx5.MLX5_CMD_OP_QUERY_PAGES, out_struct=mlx5.struct_mlx5_ifc_query_pages_out_bits, op_mod=mode)['num_pages']) <= 0: return
    mem, paddrs = self.pci_dev.alloc_sysmem(npages * 0x1000)
    self.fw_pages.append((mem, paddrs))
    self.cmd.exec(mlx5.MLX5_CMD_OP_MANAGE_PAGES, in_struct=mlx5.struct_mlx5_ifc_manage_pages_in_bits,
                  op_mod=mlx5.MLX5_PAGES_GIVE, page_queue=True, input_num_entries=npages,
                  _payload=struct.pack(f'>{npages}Q', *paddrs))

  def _access_reg(self, reg_id, data, write=True):
    inp = bytearray(8 + len(data)); struct.pack_into('>HH', inp, 0, 0, reg_id); inp[8:8 + len(data)] = data
    return self.cmd.exec(mlx5.MLX5_CMD_OP_ACCESS_REG, op_mod=int(not write), inp=inp, out_sz=len(data))[8:]

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
    if (v:=read_ifc(gen_max, CAP, 'num_vhca_ports')): fill_ifc(cap, CAP, num_vhca_ports=v)
    if read_ifc(gen_max, CAP, 'vhca_state'):
      fill_ifc(cap, CAP, event_on_vhca_state_teardown_request=1, event_on_vhca_state_in_use=1,
               event_on_vhca_state_active=1, event_on_vhca_state_allocated=1)
    if (v:=read_ifc(gen_max, CAP, 'num_total_dynamic_vf_msix')): fill_ifc(cap, CAP, num_total_dynamic_vf_msix=v)
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

  def _query_cap(self, cap_type, mode): return bytearray(self.cmd.exec(mlx5.MLX5_CMD_OP_QUERY_HCA_CAP, op_mod=(cap_type << 1) | mode, out_sz=4096)[8:])
  def _set_cap(self, cap_type, data): self.cmd.exec(mlx5.MLX5_CMD_OP_SET_HCA_CAP, op_mod=cap_type << 1, inp=bytearray(8) + data)

  def _init_hca(self):
    kw = dict(sw_owner_id=random.getrandbits(128)) if read_ifc(self.gen_caps, mlx5.struct_mlx5_ifc_cmd_hca_cap_bits, 'sw_owner_id') else {}
    self.cmd.exec(mlx5.MLX5_CMD_OP_INIT_HCA, in_struct=mlx5.struct_mlx5_ifc_init_hca_in_bits, **kw)

  def _query_hca_caps_post_init(self):
    self.gen_caps, CAP = self._query_cap(mlx5.MLX5_CAP_GENERAL, 1), mlx5.struct_mlx5_ifc_cmd_hca_cap_bits
    for cap, field in [(mlx5.MLX5_CAP_ATOMIC, 'atomic'), (mlx5.MLX5_CAP_ODP, 'pg'), (mlx5.MLX5_CAP_ROCE, 'roce')]:
      if read_ifc(self.gen_caps, CAP, field): self._query_cap(cap, 1)

  def _enable_roce(self):
    self.cmd.exec(mlx5.MLX5_CMD_OP_MODIFY_NIC_VPORT_CONTEXT, in_struct=mlx5.struct_mlx5_ifc_modify_nic_vport_context_in_bits,
                  field_select__roce_en=1, nic_vport_context__roce_en=1)

  def _alloc_uar(self):
    uar = self.cmd.exec(mlx5.MLX5_CMD_OP_ALLOC_UAR, out_struct=mlx5.struct_mlx5_ifc_alloc_uar_out_bits)['uar']
    self.uar_page = self.pci_dev.map_bar(0, off=uar * (ps:=os.sysconf('SC_PAGE_SIZE')), size=ps, fmt='I')
    return uar

  def _create_mkey(self):
    res = self.cmd.exec(mlx5.MLX5_CMD_OP_CREATE_MKEY, in_struct=mlx5.struct_mlx5_ifc_create_mkey_in_bits,
                        out_struct=mlx5.struct_mlx5_ifc_create_mkey_out_bits,
                        memory_key_mkey_entry__access_mode_1_0=0, memory_key_mkey_entry__rw=1, memory_key_mkey_entry__rr=1,
                        memory_key_mkey_entry__lw=1, memory_key_mkey_entry__lr=1, memory_key_mkey_entry__qpn=0xFFFFFF,
                        memory_key_mkey_entry__mkey_7_0=0x42, memory_key_mkey_entry__length64=1, memory_key_mkey_entry__pd=self.pd)
    return (res['mkey_index'] << 8) | 0x42

  def set_roce_address(self, gid_index, ip):
    self.cmd.exec(mlx5.MLX5_CMD_OP_SET_ROCE_ADDRESS, in_struct=mlx5.struct_mlx5_ifc_set_roce_address_in_bits,
                  roce_address_index=gid_index, vhca_port_num=1,
                  roce_address__roce_l3_type=0, roce_address__roce_version=2,
                  roce_address__source_l3_address=int.from_bytes(MLXQP._ipv4_to_gid(ip), 'big'),
                  roce_address__source_mac_47_32=int.from_bytes(self.mac[0:2], 'big'),
                  roce_address__source_mac_31_0=int.from_bytes(self.mac[2:6], 'big'))
    self.local_gid = MLXQP._ipv4_to_gid(ip)

class MLXQP:
  def __init__(self, dev:MLXDev, log_sq_size=4, log_rq_size=4, log_eq_size=7, log_cq_size=7):
    self.dev, self.cq_size = dev, 1 << log_cq_size

    def _create_hw_q(log_size, entry_sz, owner_off, opcode, in_struct, out_struct, out_field, **ctx_kw):
      mem, paddrs = dev.pci_dev.alloc_sysmem((n:=((1 << log_size) * entry_sz + 0xFFF) // 0x1000) * 0x1000)
      for i in range(1 << log_size): mem.mv[i * entry_sz + owner_off] = 0x01
      res = dev.cmd.exec(opcode, in_struct=in_struct, out_struct=out_struct, _payload=struct.pack(f'>{n}Q', *paddrs), **ctx_kw)
      return mem, res[out_field]

    self.eq_mem, self.eqn = _create_hw_q(log_eq_size, 64, 31, mlx5.MLX5_CMD_OP_CREATE_EQ,
                                          mlx5.struct_mlx5_ifc_create_eq_in_bits, mlx5.struct_mlx5_ifc_create_eq_out_bits, 'eq_number',
                                          eq_context_entry__log_eq_size=log_eq_size, eq_context_entry__uar_page=dev.uar, eq_context_entry__log_page_size=0)
    self.cq_dbr_phys = dev._alloc_dbr()
    self.cq_mem, self.cqn = _create_hw_q(log_cq_size, 64, 63, mlx5.MLX5_CMD_OP_CREATE_CQ,
                                          mlx5.struct_mlx5_ifc_create_cq_in_bits, mlx5.struct_mlx5_ifc_create_cq_out_bits, 'cqn',
                                          cq_context__log_cq_size=log_cq_size, cq_context__uar_page=dev.uar,
                                          cq_context__c_eqn_or_apu_element=self.eqn, cq_context__log_page_size=0, cq_context__dbr_addr=self.cq_dbr_phys)
    rq_sz = (1 << log_rq_size) << 4
    self.qp_buf, qp_paddrs = dev.pci_dev.alloc_sysmem((n_qp:=((rq_sz + (1 << log_sq_size) * 64 + 0xFFF) // 0x1000)) * 0x1000)
    self.qp_dbr_phys, self.sq_offset = dev._alloc_dbr(), rq_sz
    res = dev.cmd.exec(mlx5.MLX5_CMD_OP_CREATE_QP, in_struct=mlx5.struct_mlx5_ifc_create_qp_in_bits,
                       out_struct=mlx5.struct_mlx5_ifc_create_qp_out_bits, _payload=struct.pack(f'>{n_qp}Q', *qp_paddrs),
                       qpc__st=0, qpc__pm_state=3, qpc__pd=dev.pd, qpc__cqn_snd=self.cqn, qpc__cqn_rcv=self.cqn,
                       qpc__log_msg_max=30, qpc__log_rq_size=log_rq_size, qpc__log_rq_stride=0, qpc__log_sq_size=log_sq_size,
                       qpc__rlky=1, qpc__uar_page=dev.uar, qpc__log_page_size=0, qpc__dbr_addr=self.qp_dbr_phys)
    self.qpn = res['qpn']
    self._qp_transition(mlx5.MLX5_CMD_OP_RST2INIT_QP, qpc_kw=dict(log_ack_req_freq=8), ads_kw=dict(pkey_index=0, vhca_port_num=1))
    self.cq_ci = self.sq_head = self.rq_head = 0
    if MLX_DEBUG >= 1: print(f"mlx5: QP 0x{self.qpn:x} (EQ={self.eqn} CQ=0x{self.cqn:x})")

  def _dbr_write(self, phys, val): struct.pack_into('>I', self.dev.dbr_mem.mv, phys - self.dev.dbr_paddrs[0], val)

  _QP_STRUCTS = {mlx5.MLX5_CMD_OP_RST2INIT_QP: mlx5.struct_mlx5_ifc_rst2init_qp_in_bits,
                 mlx5.MLX5_CMD_OP_INIT2RTR_QP: mlx5.struct_mlx5_ifc_init2rtr_qp_in_bits,
                 mlx5.MLX5_CMD_OP_RTR2RTS_QP: mlx5.struct_mlx5_ifc_rtr2rts_qp_in_bits}

  def _qp_transition(self, opcode, qpc_kw=None, ads_kw=None, **extra):
    kw = {f'qpc__{k}': v for k, v in dict(st=0, pm_state=3, pd=self.dev.pd, cqn_snd=self.cqn, cqn_rcv=self.cqn, **(qpc_kw or {})).items()}
    if ads_kw: kw.update({f'qpc__primary_address_path__{k}': v for k, v in ads_kw.items()})
    self.dev.cmd.exec(opcode, in_struct=self._QP_STRUCTS[opcode], qpn=self.qpn, **extra, **kw)

  @staticmethod
  def _ipv4_to_gid(ip): return bytes(10) + b'\xff\xff' + socket.inet_aton(ip)
  @staticmethod
  def _calc_udp_sport(lqpn, rqpn):
    v = (lqpn * rqpn ^ ((lqpn * rqpn) >> 20) ^ ((lqpn * rqpn) >> 40)) & 0xFFFFF
    return ((v & 0x3FFF) ^ ((v & 0xFC000) >> 14)) | 0xC000

  def connect(self, remote_qpn, remote_mac, remote_gid):
    if isinstance(remote_mac, str): remote_mac = bytes.fromhex(remote_mac)
    if isinstance(remote_gid, str): remote_gid = bytes.fromhex(remote_gid)
    self._qp_transition(mlx5.MLX5_CMD_OP_INIT2RTR_QP, opt_param_mask=0x1A,
      qpc_kw=dict(mtu=3, log_msg_max=read_ifc(self.dev.gen_caps, mlx5.struct_mlx5_ifc_cmd_hca_cap_bits, 'log_max_msg'),
                  remote_qpn=remote_qpn, log_ack_req_freq=8, log_rra_max=3, rre=1, rwe=1, min_rnr_nak=12, next_rcv_psn=0),
      ads_kw=dict(pkey_index=0, src_addr_index=0, hop_limit=64, udp_sport=self._calc_udp_sport(self.qpn, remote_qpn), vhca_port_num=1,
                  rmac_47_32=int.from_bytes(remote_mac[0:2], 'big'), rmac_31_0=int.from_bytes(remote_mac[2:6], 'big'),
                  rgid_rip=int.from_bytes(remote_gid, 'big')))
    self._qp_transition(mlx5.MLX5_CMD_OP_RTR2RTS_QP,
      qpc_kw=dict(log_ack_req_freq=8, next_send_psn=0, log_sra_max=3, retry_count=7, rnr_retry=7),
      ads_kw=dict(ack_timeout=14, vhca_port_num=1))
    if MLX_DEBUG >= 1: print(f"mlx5: QP 0x{self.qpn:x} connected (remote=0x{remote_qpn:x})")

  def _post_sq(self, opcode, ds_count, segs):
    wqe = memoryview(self.qp_buf.mv)[(wo:=self.sq_offset + (self.sq_head & 0xF) * 64):wo + 64]
    wqe[:64] = bytes(64)
    struct.pack_into('>II', wqe, 0, (self.sq_head << 8) | opcode, (self.qpn << 8) | ds_count)
    wqe[11] = 0x08
    for off, data in segs: wqe[off:off + len(data)] = data
    self.sq_head += 1
    self._dbr_write(self.qp_dbr_phys + 4, self.sq_head)
    self.dev.uar_page[0x200] = swap32(struct.unpack_from('>I', wqe, 0)[0])
    self.dev.uar_page[0x201] = swap32(struct.unpack_from('>I', wqe, 4)[0])
    self.poll_cq()

  def rdma_write(self, remote_addr, rkey, local_addr, lkey, length):
    self._post_sq(0x08, 3, [(16, struct.pack('>QI4x', remote_addr, rkey)), (32, struct.pack('>IIQ', length, lkey, local_addr))])
  def send(self, addr, lkey, length): self._post_sq(0x0a, 2, [(16, struct.pack('>IIQ', length, lkey, addr))])

  def post_recv(self, addr, lkey, length):
    struct.pack_into('>IIQ', self.qp_buf.mv, (self.rq_head & 0xF) * 16, length, lkey, addr)
    self.rq_head += 1
    self._dbr_write(self.qp_dbr_phys, self.rq_head & 0xFFFF)

  def poll_cq(self, timeout=5.0):
    t = time.monotonic()
    while True:
      cqe = bytes(self.cq_mem.mv[(idx:=self.cq_ci & (self.cq_size - 1)) * 64:(idx + 1) * 64])
      opcode, owner = cqe[63] >> 4, cqe[63] & 1
      if opcode != 0x0F and owner == (1 if (self.cq_ci & self.cq_size) else 0):
        self.cq_ci += 1
        self._dbr_write(self.cq_dbr_phys, self.cq_ci & 0xFFFFFF)
        if opcode in (13, 14): raise RuntimeError(f"CQE error: opcode={opcode} syndrome=0x{cqe[55]:02x} vendor=0x{cqe[54]:02x}")
        return opcode
      if time.monotonic() - t > timeout: raise TimeoutError("CQ poll timeout")
      time.sleep(0.0001)

if __name__ == "__main__":
  dev = MLXDev(PCIDevice("mlx5", getenv("MLX_PCI", "0000:41:00.0")))
  ip = sys.argv[sys.argv.index("--ip") + 1] if "--ip" in sys.argv else None
  qp = MLXQP(dev)

  if "--server" in sys.argv:
    dev.set_roce_address(0, ip)
    print(json.dumps({"qpn": qp.qpn, "mac": dev.mac.hex(), "gid": dev.local_gid.hex()}), flush=True)
    remote = json.loads(input())
    qp.connect(remote["qpn"], remote["mac"], remote["gid"])
    print("connected", flush=True)
    target_mem, target_paddrs = dev.pci_dev.alloc_sysmem(0x1000)
    print(json.dumps({"target_addr": target_paddrs[0], "rkey": dev.mkey}), flush=True)
    input()
    data = bytes(target_mem[i] for i in range(64))
    print(f"RECEIVED: {data.hex(' ')}\nAS TEXT: {data.rstrip(b'\\x00').decode('ascii', errors='replace')}", flush=True)
