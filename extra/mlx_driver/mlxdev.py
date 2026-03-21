import os, struct, time, random
from tinygrad.runtime.support.system import PCIDevice
from tinygrad.runtime.autogen import mlx5

MLX_DEBUG = int(os.getenv("MLX_DEBUG", "0"))

def swap32(v): return ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v >> 8) & 0xFF00) | ((v >> 24) & 0xFF)

def ifc_get(buf, bit_off, width):
  """Read a field from a big-endian IFC struct (MSB-first bit numbering)."""
  byte_off, bit_in = bit_off // 8, bit_off % 8
  n = (bit_in + width + 7) // 8
  val = int.from_bytes(buf[byte_off:byte_off + n], 'big')
  return (val >> (n * 8 - bit_in - width)) & ((1 << width) - 1)

def ifc_set(buf, bit_off, width, value):
  """Write a field to a big-endian IFC struct (MSB-first bit numbering)."""
  byte_off, bit_in = bit_off // 8, bit_off % 8
  n = (bit_in + width + 7) // 8
  val = int.from_bytes(buf[byte_off:byte_off + n], 'big')
  shift = n * 8 - bit_in - width
  val = (val & ~(((1 << width) - 1) << shift)) | ((value & ((1 << width) - 1)) << shift)
  buf[byte_off:byte_off + n] = val.to_bytes(n, 'big')

class MLXDev:
  CMD_IF_REV, MBOX_SZ, MBOX_STRIDE = 5, 512, 1024  # cmd interface rev, data per mailbox block, mailbox alignment

  def __init__(self, pci_dev:PCIDevice):
    self.pci_dev, self.bar, self.token = pci_dev, pci_dev.map_bar(0, fmt='I'), 0

    fw_rev, cmdif_sub = self.rreg(0), self.rreg(4)
    print(f"mlx5: firmware {fw_rev >> 16}.{fw_rev & 0xFFFF}.{cmdif_sub & 0xFFFF}")
    assert (cmdif_sub >> 16) == self.CMD_IF_REV, f"unsupported cmdif rev {cmdif_sub >> 16}"

    self.wait_fw_init()
    self._setup_cmd()
    self.wait_fw_init()

    # Init sequence: matches mlx5_function_enable + mlx5_function_open
    self.cmd_exec(mlx5.MLX5_CMD_OP_ENABLE_HCA)                    # mlx5_core_enable_hca
    self._set_issi()                                                # mlx5_core_set_issi
    self._satisfy_pages(mlx5.MLX5_BOOT_PAGES)                     # mlx5_satisfy_startup_pages(boot)
    self._set_hca_ctrl()                                            # set_hca_ctrl
    self._set_hca_caps()                                            # set_hca_cap
    self._satisfy_pages(mlx5.MLX5_INIT_PAGES)                     # mlx5_satisfy_startup_pages(init)
    self._init_hca()                                                # mlx5_cmd_init_hca
    print("mlx5: INIT_HCA OK")

    # Post-INIT_HCA: re-query caps, allocate resources (mlx5_init_once + mlx5_load)
    self._query_hca_caps_post_init()
    self.uar = self._alloc_uar()
    self.pd = self._alloc_pd()
    self.td = self._alloc_td()
    self.resd_lkey, self.null_mkey = self._query_special_contexts()
    self.mac = self._query_nic_vport_mac()

    # Allocate doorbell record page (for CQ/QP doorbells)
    self.dbr_mem, self.dbr_paddrs = self.pci_dev.alloc_sysmem(0x1000)
    self.dbr_offset = 0  # next free offset in dbr page

    self.eq = self._create_eq()
    self.cq = self._create_cq()
    self.mkey = self._create_mkey()
    self.qpn = self._create_qp()
    print(f"mlx5: MAC={':'.join(f'{b:02x}' for b in self.mac)} UAR={self.uar} PD=0x{self.pd:x} TD=0x{self.td:x}")
    print(f"mlx5: EQ={self.eq} CQ=0x{self.cq:x} MKEY=0x{self.mkey:x} QPN=0x{self.qpn:x}")
    print("mlx5: QP in INIT state, ready for connection")

  # --- Big-endian MMIO (init segment at BAR0) ---
  def rreg(self, off): return swap32(self.bar[off // 4])
  def wreg(self, off, val): self.bar[off // 4] = swap32(val)

  def wait_fw_init(self, timeout=10.0):
    t = time.monotonic()
    while self.rreg(0x1FC) & 0x80000000:  # iseg->initializing bit 31
      if time.monotonic() - t > timeout: raise TimeoutError("FW init timeout")
      time.sleep(0.001)

  # --- DMA helpers ---
  def _dma_phys(self, off): return self.dma_paddrs[off // 0x1000] + (off % 0x1000)

  def _setup_mbox(self, n, base, tok):
    """Chain n mailbox blocks starting at pool index `base`. Returns phys addr of first block."""
    if n == 0: return 0
    for i in range(n):
      off = 0x1000 + (base + i) * self.MBOX_STRIDE
      self.dma.mv[off:off + 576] = bytes(576)
      struct.pack_into('>I', self.dma.mv, off + 568, i)                       # block_num
      self.dma.mv[off + 573] = tok                                             # token
      if i < n - 1:
        struct.pack_into('>Q', self.dma.mv, off + 560, self._dma_phys(0x1000 + (base + i + 1) * self.MBOX_STRIDE))  # next
    return self._dma_phys(0x1000 + base * self.MBOX_STRIDE)

  def _mbox_write(self, data, base, tok):
    """Copy data into mailbox chain data sections."""
    for i in range((len(data) + self.MBOX_SZ - 1) // self.MBOX_SZ):
      off = 0x1000 + (base + i) * self.MBOX_STRIDE
      chunk = data[i * self.MBOX_SZ:(i + 1) * self.MBOX_SZ]
      self.dma.mv[off:off + len(chunk)] = chunk
      self.dma.mv[off + 573] = tok
      struct.pack_into('>I', self.dma.mv, off + 568, i)

  def _mbox_read(self, size, base):
    """Read data from mailbox chain data sections."""
    out = bytearray()
    for i in range((size + self.MBOX_SZ - 1) // self.MBOX_SZ):
      off = 0x1000 + (base + i) * self.MBOX_STRIDE
      out += bytes(self.dma.mv[off:off + min(self.MBOX_SZ, size - i * self.MBOX_SZ)])
    return out

  # --- Command interface (polling mode) ---
  def _setup_cmd(self):
    cmd_l = self.rreg(0x14) & 0xFF  # cmdq_addr_l_sz low byte: log_sz(4) | log_stride(4)
    self.log_sz, self.log_stride = (cmd_l >> 4) & 0xF, cmd_l & 0xF
    self.max_reg_cmds = (1 << self.log_sz) - 1

    # Allocate DMA: 4K cmd page + 128 mailbox blocks (1024-byte aligned each)
    self.dma, self.dma_paddrs = self.pci_dev.alloc_sysmem(0x1000 + 128 * self.MBOX_STRIDE)

    # Write cmd queue DMA address to init segment
    cmd_phys = self.dma_paddrs[0]
    self.wreg(0x10, cmd_phys >> 32)       # cmdq_addr_h
    self.wreg(0x14, (cmd_phys & 0xFFFFFFFF) | cmd_l)  # cmdq_addr_l_sz (preserve log_sz/stride in low byte)

    self.fw_pages = []  # track allocated FW page memory: [(view, paddrs), ...]

  def cmd_exec(self, opcode, op_mod=0, inp=b'', out_sz=0, page_queue=False):
    """Execute a FW command. inp=data after opcode/op_mod header. Returns output data after status/syndrome."""
    self.token = (self.token % 255) + 1
    tok, slot = self.token, self.max_reg_cmds if page_queue else 0
    inlen, outlen = 16 + len(inp), 16 + out_sz

    # Build 16-byte header: DW0(opcode|uid) DW1(rsvd|op_mod) DW2-DW3(first 8 bytes of inp)
    hdr = bytearray(16)
    struct.pack_into('>HH', hdr, 0, opcode, 0)
    struct.pack_into('>HH', hdr, 4, 0, op_mod)
    hdr[8:8 + min(8, len(inp))] = inp[:min(8, len(inp))]

    # Setup mailbox chains for data beyond the 16-byte header
    mbox_in = inp[8:] if len(inp) > 8 else b''
    n_in = (len(mbox_in) + self.MBOX_SZ - 1) // self.MBOX_SZ if mbox_in else 0
    n_out = (out_sz + self.MBOX_SZ - 1) // self.MBOX_SZ if out_sz > 0 else 0
    in_ptr = self._setup_mbox(n_in, 0, tok)
    out_ptr = self._setup_mbox(n_out, n_in, tok)
    if mbox_in: self._mbox_write(mbox_in, 0, tok)

    # Build cmd_layout at slot offset in the cmd queue page
    lay = slot << self.log_stride
    self.dma.mv[lay:lay + 64] = bytes(64)
    self.dma.mv[lay] = mlx5.MLX5_PCI_CMD_XPORT               # type
    struct.pack_into('>I', self.dma.mv, lay + 4, inlen)        # inlen
    struct.pack_into('>Q', self.dma.mv, lay + 8, in_ptr)       # in_ptr
    self.dma.mv[lay + 16:lay + 32] = hdr                       # in[4]
    struct.pack_into('>Q', self.dma.mv, lay + 48, out_ptr)     # out_ptr
    struct.pack_into('>I', self.dma.mv, lay + 56, outlen)      # outlen
    self.dma.mv[lay + 60] = tok                                 # token
    self.dma.mv[lay + 63] = mlx5.CMD_OWNER_HW                 # status_own

    # Signature: sig = ~(XOR of all 64 layout bytes) with sig=0 initially
    xor_val = 0
    for i in range(64): xor_val ^= self.dma.mv[lay + i]
    self.dma.mv[lay + 61] = (~xor_val) & 0xFF

    if MLX_DEBUG >= 2:
      d = bytes(self.dma.mv[lay:lay + 64])
      print(f"  CMD[{slot}] op=0x{opcode:04x} mod=0x{op_mod:04x} inlen={inlen} outlen={outlen} tok={tok}")
      print(f"  LAY: {d[:32].hex(' ')}")
      print(f"  LAY: {d[32:].hex(' ')}")

    # Ring doorbell
    self.wreg(0x18, 1 << slot)

    # Poll for completion: wait for CMD_OWNER_HW bit to clear
    t = time.monotonic()
    while self.dma.mv[lay + 63] & mlx5.CMD_OWNER_HW:
      if time.monotonic() - t > 60.0: raise TimeoutError(f"cmd 0x{opcode:04x} timeout")
      time.sleep(0.0001)

    # Parse output from layout's out[4] (bytes 32-47)
    delivery = self.dma.mv[lay + 63] >> 1
    out_hdr = [struct.unpack_from('>I', self.dma.mv, lay + 32 + i * 4)[0] for i in range(4)]
    status, syndrome = out_hdr[0] >> 24, out_hdr[1]

    if MLX_DEBUG >= 2: print(f"  DONE[{slot}] status=0x{status:02x} syn=0x{syndrome:08x} delivery=0x{delivery:x}")
    assert delivery == 0, f"cmd 0x{opcode:04x} delivery error 0x{delivery:x}"
    assert status == 0, f"cmd 0x{opcode:04x} failed status=0x{status:x} syn=0x{syndrome:08x}"

    result = bytearray(struct.pack('>II', out_hdr[2], out_hdr[3]))
    if out_sz > 0: result += self._mbox_read(out_sz, n_in)
    return result

  # --- Capability helpers ---
  def _query_cap(self, cap_type, mode):
    """QUERY_HCA_CAP. mode: 0=max, 1=current. Returns 4096 bytes of capability data."""
    return bytearray(self.cmd_exec(mlx5.MLX5_CMD_OP_QUERY_HCA_CAP, op_mod=(cap_type << 1) | mode, out_sz=4096)[8:])

  def _set_cap(self, cap_type, data):
    """SET_HCA_CAP with 4096 bytes of capability data."""
    self.cmd_exec(mlx5.MLX5_CMD_OP_SET_HCA_CAP, op_mod=cap_type << 1, inp=bytearray(8) + data)

  # --- HCA Initialization Steps ---
  def _set_issi(self):
    """QUERY_ISSI + SET_ISSI to version 1."""
    out = self.cmd_exec(mlx5.MLX5_CMD_OP_QUERY_ISSI, out_sz=96)
    sup_issi = struct.unpack_from('>I', out, 100)[0]  # supported_issi_dw0 at byte 108 of output struct
    if sup_issi & (1 << 1):
      inp = bytearray(8)
      struct.pack_into('>I', inp, 0, 1)  # current_issi=1 (16-bit field in lower half of DW2)
      self.cmd_exec(mlx5.MLX5_CMD_OP_SET_ISSI, inp=inp)

  def _satisfy_pages(self, mode):
    """QUERY_PAGES + MANAGE_PAGES(GIVE) for boot or init pages."""
    out = self.cmd_exec(mlx5.MLX5_CMD_OP_QUERY_PAGES, op_mod=mode)
    npages = struct.unpack_from('>i', out, 4)[0]  # num_pages at DW3 (signed)
    if MLX_DEBUG >= 1: print(f"mlx5: {'boot' if mode == mlx5.MLX5_BOOT_PAGES else 'init'} pages: {npages}")
    if npages <= 0: return

    mem, paddrs = self.pci_dev.alloc_sysmem(npages * 0x1000)
    self.fw_pages.append((mem, paddrs))

    # Build MANAGE_PAGES input: DW2(func_id=0) DW3(num_entries) + pas[](64-bit phys addrs)
    pas = bytearray(npages * 8)
    for i in range(npages): struct.pack_into('>Q', pas, i * 8, paddrs[i])
    inp = bytearray(8) + pas
    struct.pack_into('>I', inp, 4, npages)  # input_num_entries at DW3
    self.cmd_exec(mlx5.MLX5_CMD_OP_MANAGE_PAGES, op_mod=mlx5.MLX5_PAGES_GIVE, inp=inp, page_queue=True)

  def _access_reg(self, reg_id, data, write=True):
    """ACCESS_REG command. op_mod: 0=write, 1=read."""
    inp = bytearray(8 + len(data))
    struct.pack_into('>HH', inp, 0, 0, reg_id)  # DW2: rsvd(16)|register_id(16)
    inp[8:8 + len(data)] = data
    return self.cmd_exec(mlx5.MLX5_CMD_OP_ACCESS_REG, op_mod=0 if write else 1, inp=inp, out_sz=len(data))[8:]

  def _set_hca_ctrl(self):
    """Set host endianness via ACCESS_REG."""
    self._access_reg(mlx5.MLX5_REG_HOST_ENDIANNESS, bytearray(16), write=True)

  def _set_hca_caps(self):
    """Configure all HCA capability types (matches kernel set_hca_cap)."""
    # --- General capabilities (handle_hca_cap) ---
    gen_max = self._query_cap(mlx5.MLX5_CAP_GENERAL, 0)
    gen_cur = self._query_cap(mlx5.MLX5_CAP_GENERAL, 1)
    self.gen_caps = gen_cur  # save for _init_hca
    cap = bytearray(gen_cur)

    ifc_set(cap, 0x190, 16, 0)                                                    # pkey_table_size: 128 entries (fw encoding=0)
    ifc_set(cap, 0x09B, 5, min(18, ifc_get(gen_max, 0x09B, 5)))                  # log_max_qp
    ifc_set(cap, 0x210, 2, 0)                                                     # cmdif_checksum: disabled
    page_sz = os.sysconf('SC_PAGE_SIZE')
    if ifc_get(gen_max, 0x240, 1) and page_sz > 4096: ifc_set(cap, 0x240, 1, 1)  # uar_4k
    ifc_set(cap, 0x490, 16, page_sz.bit_length() - 1 - 12)                       # log_uar_page_sz = PAGE_SHIFT - 12
    if ifc_get(gen_max, 0x164, 1):                                                 # cache_line_128byte
      try: ifc_set(cap, 0x164, 1, 1 if os.sysconf('SC_LEVEL1_DCACHE_LINESIZE') >= 128 else 0)
      except (ValueError, OSError): ifc_set(cap, 0x164, 1, 0)
    if ifc_get(gen_max, 0x21A, 1): ifc_set(cap, 0x21A, 1, 1)                     # dct
    if ifc_get(gen_max, 0x1F1, 1): ifc_set(cap, 0x1F1, 1, 1)                     # pci_sync_for_fw_update_event
    if ifc_get(gen_max, 0x336, 1): ifc_set(cap, 0x336, 1, 1)                     # pci_sync_for_fw_update_with_driver_unload
    if ifc_get(gen_max, 0x335, 1): ifc_set(cap, 0x335, 1, 1)                     # pcie_reset_using_hotreset_method
    vhca_ports = ifc_get(gen_max, 0x610, 8)
    if vhca_ports: ifc_set(cap, 0x610, 8, vhca_ports)                             # num_vhca_ports
    if ifc_get(gen_max, 0x145, 1): ifc_set(cap, 0x145, 1, 1)                     # release_all_pages
    if ifc_get(gen_max, 0x266, 1): ifc_set(cap, 0x266, 1, 1)                     # mkey_by_name
    if ifc_get(gen_max, 0x3EA, 1):                                                 # vhca_state -> enable events
      for off in [0x023, 0x024, 0x025, 0x026]: ifc_set(cap, off, 1, 1)            # teardown_req, in_use, active, allocated
    max_msix = ifc_get(gen_max, 0x708, 24)
    if max_msix: ifc_set(cap, 0x708, 24, max_msix)                               # num_total_dynamic_vf_msix
    if ifc_get(gen_cur, 0x3A1, 1) and ifc_get(gen_max, 0x21D, 1):                # roce_rw_supported && max(roce)
      ifc_set(cap, 0x21D, 1, 1)                                                   # roce
    if ifc_get(gen_max, 0x007, 1): ifc_set(cap, 0x007, 1, 1)                     # abs_native_port_num
    self._set_cap(mlx5.MLX5_CAP_GENERAL, cap)

    # --- Atomic capabilities (handle_hca_cap_atomic) ---
    if ifc_get(gen_cur, 0x21E, 1):  # atomic supported
      self._query_cap(mlx5.MLX5_CAP_ATOMIC, 0)  # query max (stored by FW)
      atom_cur = self._query_cap(mlx5.MLX5_CAP_ATOMIC, 1)
      if ifc_get(atom_cur, 0x46, 1):  # supported_atomic_req_8B_endianness_mode_1
        cap = bytearray(4096)
        ifc_set(cap, 0x40, 2, 1)  # atomic_req_8B_endianness_mode = host_endianness
        self._set_cap(mlx5.MLX5_CAP_ATOMIC, cap)

    # --- ODP capabilities (handle_hca_cap_odp) ---
    if ifc_get(gen_cur, 0x227, 1):  # pg (ODP) supported
      odp_max = self._query_cap(mlx5.MLX5_CAP_ODP, 0)
      odp_cur = self._query_cap(mlx5.MLX5_CAP_ODP, 1)
      cap, modified = bytearray(odp_cur), False

      if ifc_get(odp_max, 0x600, 1) and ifc_get(odp_max, 0x245, 1):  # mem_page_fault + memory_scheme.page_prefetch
        ifc_set(cap, 0x600, 1, 1)
        modified = True
      else:  # transport page fault scheme: set each supported field from max
        # ud.srq_rx, rc.srq_rx, xrc.{srq_rx,send,rx,wr,rd,atomic}, dc.{srq_rx,send,rx,wr,rd,atomic}
        for off in [0xC5, 0x85, 0xE5, 0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0x105, 0x100, 0x101, 0x102, 0x103, 0x104]:
          if ifc_get(odp_max, off, 1): ifc_set(cap, off, 1, 1); modified = True

      if modified: self._set_cap(mlx5.MLX5_CAP_ODP, cap)

    # --- RoCE capabilities (handle_hca_cap_roce) ---
    roce_max = self._query_cap(mlx5.MLX5_CAP_ROCE, 0)
    roce_cur = self._query_cap(mlx5.MLX5_CAP_ROCE, 1)
    if not ifc_get(roce_cur, 0x04, 1) and ifc_get(roce_max, 0x04, 1):  # sw_r_roce_src_udp_port not yet set
      cap = bytearray(roce_cur)
      ifc_set(cap, 0x04, 1, 1)                                          # sw_r_roce_src_udp_port
      if ifc_get(roce_max, 0x08, 1): ifc_set(cap, 0x08, 1, 1)          # qp_ooo_transmit_default
      self._set_cap(mlx5.MLX5_CAP_ROCE, cap)

  def _init_hca(self):
    """INIT_HCA with random sw_owner_id."""
    inp = bytearray(24)  # DW2(rsvd) DW3(sw_vhca_id) DW4-7(sw_owner_id[4])
    if ifc_get(self.gen_caps, 0x61E, 1):  # sw_owner_id capability
      for i in range(4): struct.pack_into('>I', inp, 8 + i * 4, random.getrandbits(32))
    self.cmd_exec(mlx5.MLX5_CMD_OP_INIT_HCA, inp=inp)

  # --- Post-INIT_HCA resource allocation ---
  def _query_hca_caps_post_init(self):
    """Re-query all HCA caps after INIT_HCA (mlx5_query_hca_caps). Saves current general caps."""
    self.gen_caps = self._query_cap(mlx5.MLX5_CAP_GENERAL, 1)
    # Query additional cap types based on general caps (only current mode needed post-init)
    if ifc_get(self.gen_caps, 0x21E, 1): self._query_cap(mlx5.MLX5_CAP_ATOMIC, 1)    # atomic
    if ifc_get(self.gen_caps, 0x227, 1): self._query_cap(mlx5.MLX5_CAP_ODP, 1)       # odp
    if ifc_get(self.gen_caps, 0x21D, 1): self._query_cap(mlx5.MLX5_CAP_ROCE, 1)      # roce

  def _alloc_uar(self):
    """ALLOC_UAR -> returns UAR number. Also maps the UAR MMIO page."""
    out = self.cmd_exec(mlx5.MLX5_CMD_OP_ALLOC_UAR)
    uar = struct.unpack_from('>I', out, 0)[0] & 0xFFFFFF  # uar[24] at DW2
    # Map UAR MMIO page within BAR0 for doorbells
    page_sz = os.sysconf('SC_PAGE_SIZE')
    uar_offset = uar * page_sz  # byte offset in BAR0
    self.uar_page = self.pci_dev.map_bar(0, off=uar_offset, size=page_sz, fmt='I')
    return uar

  def _alloc_pd(self):
    """ALLOC_PD -> returns 24-bit PD number."""
    return struct.unpack_from('>I', self.cmd_exec(mlx5.MLX5_CMD_OP_ALLOC_PD), 0)[0] & 0xFFFFFF

  def _alloc_td(self):
    """ALLOC_TRANSPORT_DOMAIN -> returns 24-bit TD number."""
    return struct.unpack_from('>I', self.cmd_exec(mlx5.MLX5_CMD_OP_ALLOC_TRANSPORT_DOMAIN), 0)[0] & 0xFFFFFF

  def _query_special_contexts(self):
    """QUERY_SPECIAL_CONTEXTS -> returns (resd_lkey, null_mkey)."""
    out = self.cmd_exec(mlx5.MLX5_CMD_OP_QUERY_SPECIAL_CONTEXTS, out_sz=16)
    return struct.unpack_from('>I', out, 4)[0], struct.unpack_from('>I', out, 8)[0]  # resd_lkey at DW3, null_mkey in mbox

  def _query_nic_vport_mac(self):
    """QUERY_NIC_VPORT_CONTEXT -> returns 6-byte MAC address."""
    out = self.cmd_exec(mlx5.MLX5_CMD_OP_QUERY_NIC_VPORT_CONTEXT, out_sz=256)
    # nic_vport_context starts at out[8]. permanent_address at context bit 0x7A0 = byte 0xF4
    # mac_address_layout: rsvd[2] mac_47_32[2] mac_31_0[4] at context byte 0xF4
    off = 8 + 0xF4  # offset in out_data
    return bytes(out[off + 2:off + 8])  # skip 2-byte reserved, take 6 bytes MAC

  def _alloc_dbr(self):
    """Allocate an 8-byte doorbell record from the DBR page. Returns physical address."""
    assert self.dbr_offset + 8 <= 0x1000, "DBR page full"
    phys = self.dbr_paddrs[0] + self.dbr_offset
    self.dbr_offset += 8
    return phys

  def _create_eq(self, log_eq_size=7):
    """CREATE_EQ for completion events. Returns EQ number."""
    eq_size = 1 << log_eq_size  # number of EQEs
    eq_buf_sz = eq_size * 64    # each EQE is 64 bytes
    n_pages = (eq_buf_sz + 0xFFF) // 0x1000

    # Allocate EQ buffer (DMA memory, zero-initialized)
    self.eq_mem, eq_paddrs = self.pci_dev.alloc_sysmem(n_pages * 0x1000)
    # Set ownership bit on all EQEs: byte 31 bit 0 (owner) = 1 (HW owns initially)
    for i in range(eq_size): self.eq_mem.mv[i * 64 + 31] = 0x01

    # Build CREATE_EQ input
    # Layout from DW2: rsvd[8B] + eqc[64B] + rsvd[8B] + event_bitmask[32B] + rsvd[152B] + pas[]
    inp = bytearray(264 + n_pages * 8)

    # Fill EQ context (eqc) at inp[8:72]
    eqc = memoryview(inp)[8:72]
    ifc_set(eqc, 0x63, 5, log_eq_size)       # log_eq_size
    ifc_set(eqc, 0x68, 24, self.uar)          # uar_page
    ifc_set(eqc, 0xC3, 5, 0)                  # log_page_size = 0 (4K)

    # Event bitmask at inp[80:112] - leave all zeros for completion EQ
    # (completions are routed via CQ's c_eqn field, not the event bitmask)

    # Page addresses at inp[264:]
    for i in range(n_pages): struct.pack_into('>Q', inp, 264 + i * 8, eq_paddrs[i])

    out = self.cmd_exec(mlx5.MLX5_CMD_OP_CREATE_EQ, inp=inp)
    eq_num = struct.unpack_from('>I', out, 0)[0] & 0xFF  # eq_number[8] in low byte of DW2
    if MLX_DEBUG >= 1: print(f"mlx5: created EQ {eq_num}, {eq_size} entries")
    return eq_num

  def _create_cq(self, log_cq_size=7):
    """CREATE_CQ with doorbell record. Returns CQN."""
    cq_size = 1 << log_cq_size
    cq_buf_sz = cq_size * 64  # each CQE is 64 bytes
    n_pages = (cq_buf_sz + 0xFFF) // 0x1000

    # Allocate CQ buffer
    self.cq_mem, cq_paddrs = self.pci_dev.alloc_sysmem(n_pages * 0x1000)
    # Set ownership on all CQEs: byte 63 bit 0 (owner) = 1 (HW owns)
    for i in range(cq_size): self.cq_mem.mv[i * 64 + 63] = 0x01

    # Allocate doorbell record for this CQ
    self.cq_dbr_phys = self._alloc_dbr()

    # Build CREATE_CQ input
    # Layout from DW2: rsvd[8B] + cqc[64B] + rsvd[12B] + cq_umem_valid(1bit) + rsvd[~180B] + pas[]
    inp = bytearray(264 + n_pages * 8)

    # Fill CQ context (cqc) at inp[8:72]
    cqc = memoryview(inp)[8:72]
    ifc_set(cqc, 0x63, 5, log_cq_size)        # log_cq_size
    ifc_set(cqc, 0x68, 24, self.uar)           # uar_page
    ifc_set(cqc, 0xA0, 32, self.eq)            # c_eqn (EQ number for completion events)
    ifc_set(cqc, 0xC3, 5, 0)                   # log_page_size = 0 (4K)
    # dbr_addr at cqc bit 0x1C0 (64 bits)
    ifc_set(cqc, 0x1C0, 64, self.cq_dbr_phys)  # doorbell record address

    # Page addresses at inp[264:]
    for i in range(n_pages): struct.pack_into('>Q', inp, 264 + i * 8, cq_paddrs[i])

    out = self.cmd_exec(mlx5.MLX5_CMD_OP_CREATE_CQ, inp=inp)
    cqn = struct.unpack_from('>I', out, 0)[0] & 0xFFFFFF  # cqn[24] at DW2
    if MLX_DEBUG >= 1: print(f"mlx5: created CQ 0x{cqn:x}, {cq_size} entries")
    return cqn

  def _create_mkey(self):
    """CREATE_MKEY in PA (physical address) mode covering all memory. Returns full mkey value."""
    # create_mkey_in from DW2: rsvd[4B] + flags[4B] + mkc[64B] + rsvd[16B] + translations_sz[4B] + rsvd[172B]
    inp = bytearray(264)

    # MKey context (mkc) at inp[8:72]
    mkc = memoryview(inp)[8:72]
    ifc_set(mkc, 0x03, 3, 0)           # access_mode_4_2 = 0 (PA mode)
    ifc_set(mkc, 0x16, 2, 0)           # access_mode_1_0 = 0 (PA mode)
    ifc_set(mkc, 0x12, 1, 1)           # rw (remote write)
    ifc_set(mkc, 0x13, 1, 1)           # rr (remote read)
    ifc_set(mkc, 0x14, 1, 1)           # lw (local write)
    ifc_set(mkc, 0x15, 1, 1)           # lr (local read)
    ifc_set(mkc, 0x20, 24, 0xFFFFFF)   # qpn = 0xFFFFFF (any QP)
    ifc_set(mkc, 0x28, 8, 0x42)        # mkey_7_0 (low 8 bits of key)
    ifc_set(mkc, 0x68, 24, self.pd)    # pd

    out = self.cmd_exec(mlx5.MLX5_CMD_OP_CREATE_MKEY, inp=inp)
    mkey_index = struct.unpack_from('>I', out, 0)[0] & 0xFFFFFF  # mkey_index[24] at DW2
    mkey = (mkey_index << 8) | 0x42  # full mkey = index << 8 | mkey_7_0
    if MLX_DEBUG >= 1: print(f"mlx5: created MKey 0x{mkey:x} (PA mode, full memory)")
    return mkey

  def _create_qp(self, log_sq_size=4, log_rq_size=4):
    """CREATE_QP (RC) + RST2INIT_QP. Returns QPN.

    WQ layout: [RQ at offset 0 | SQ at offset rq_size]
    - RQ: 2^log_rq_size WQEs, each 16 bytes (1 data segment = 1 SGE)
    - SQ: 2^log_sq_size WQEBBs, each 64 bytes (basic block)
    - Doorbell record: 8 bytes [RCV_DBR(be32) | SND_DBR(be32)]
    """
    # WQ sizing (matches kernel set_rq_size + calc_sq_size)
    log_rq_stride = 0  # stride = 16 << log_rq_stride = 16 bytes (1 SGE per RQ WQE)
    rq_sz = (1 << log_rq_size) << (log_rq_stride + 4)  # wqe_cnt * stride
    sq_sz = (1 << log_sq_size) * 64                      # wqe_cnt * WQEBB(64)
    n_pages = ((rq_sz + sq_sz) + 0xFFF) // 0x1000

    self.qp_buf, qp_paddrs = self.pci_dev.alloc_sysmem(n_pages * 0x1000)
    self.qp_dbr_phys = self._alloc_dbr()
    self.sq_offset = rq_sz  # SQ starts after RQ in the buffer

    # ---- CREATE_QP ----
    # create_qp_in from DW2: input_qpn(4) rsvd(4) opt_param_mask(4) ece(4) qpc(232) umem_off(8) umem_id(4) umem_valid(4) pas[]
    inp = bytearray(264 + n_pages * 8)

    # QPC at inp[16:248] — WQ configuration for the QP (kernel create_kernel_qp)
    qpc = memoryview(inp)[16:248]
    ifc_set(qpc, 0x08, 8, 0)                        # st = RC
    ifc_set(qpc, 0x13, 2, 3)                        # pm_state = MIGRATED
    ifc_set(qpc, 0x28, 24, self.pd)                  # pd
    ifc_set(qpc, 0x43, 5, 30)                        # log_msg_max = 30 (max 1GB messages)
    ifc_set(qpc, 0x49, 4, log_rq_size)               # log_rq_size
    ifc_set(qpc, 0x4D, 3, log_rq_stride)             # log_rq_stride (stride = 16 << val)
    ifc_set(qpc, 0x51, 4, log_sq_size)               # log_sq_size (in WQEBBs)
    ifc_set(qpc, 0x5B, 1, 1)                         # rlky (use reserved lkey)
    ifc_set(qpc, 0x68, 24, self.uar)                  # uar_page
    ifc_set(qpc, 0xA3, 5, 0)                         # log_page_size = 0 (4K)
    ifc_set(qpc, 0x3E8, 24, self.cq)                 # cqn_snd
    ifc_set(qpc, 0x4E8, 24, self.cq)                 # cqn_rcv (same CQ for both)
    ifc_set(qpc, 0x500, 64, self.qp_dbr_phys)        # dbr_addr

    # Page addresses at inp[264:]
    for i in range(n_pages): struct.pack_into('>Q', inp, 264 + i * 8, qp_paddrs[i])

    out = self.cmd_exec(mlx5.MLX5_CMD_OP_CREATE_QP, inp=inp)
    qpn = struct.unpack_from('>I', out, 0)[0] & 0xFFFFFF  # qpn[24] at DW2
    if MLX_DEBUG >= 1: print(f"mlx5: created QP 0x{qpn:x} (RC, sq={1<<log_sq_size} rq={1<<log_rq_size})")

    # ---- RST2INIT_QP ----
    # rst2init_qp_in from DW2: qpn(4) rsvd(4) opt_param_mask(4) ece(4) qpc(232) rsvd(16)
    inp2 = bytearray(264)
    struct.pack_into('>I', inp2, 0, qpn)  # DW2: rsvd[8] | qpn[24]

    # QPC at inp2[16:248] — connection parameters for INIT state
    qpc2 = memoryview(inp2)[16:248]
    ifc_set(qpc2, 0x08, 8, 0)                        # st = RC
    ifc_set(qpc2, 0x13, 2, 3)                        # pm_state = MIGRATED
    ifc_set(qpc2, 0x28, 24, self.pd)                  # pd
    ifc_set(qpc2, 0x3E8, 24, self.cq)                # cqn_snd
    ifc_set(qpc2, 0x4E8, 24, self.cq)                # cqn_rcv
    ifc_set(qpc2, 0x380, 4, 8)                       # log_ack_req_freq = 8
    # Primary address path: pkey_index + port
    ifc_set(qpc2, 0xC0 + 0x10, 16, 0)                # pkey_index = 0
    ifc_set(qpc2, 0xC0 + 0x128, 8, 1)                # vhca_port_num = 1

    self.cmd_exec(mlx5.MLX5_CMD_OP_RST2INIT_QP, inp=inp2)
    if MLX_DEBUG >= 1: print(f"mlx5: QP 0x{qpn:x} RST -> INIT")
    return qpn

if __name__ == "__main__":
  pci_bdf = os.getenv("MLX_PCI", "0000:41:00.0")
  dev = MLXDev(PCIDevice("mlx5", pci_bdf))
