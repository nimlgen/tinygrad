import ctypes, struct
from tinygrad.helpers import ceildiv, getenv, wait_cond, DEBUG
from tinygrad.runtime.autogen import bnxt, pci
from tinygrad.runtime.support.system import PCIDevice, System, ipv4_to_gid

# BCM57608 static WQEs: 32 x 128B per ring, with the SQ MSN table on the next page. Doorbells count WQEs, not 16B slots.

BNXT_DEBUG = getenv("BNXT_DEBUG", 0)
BNXT_ACCESS, BNXT_INIT_MASK, BNXT_RTR_MASK, BNXT_RTS_MASK = 3, 0xd, 0x41515ad, 0xae005
BNXT_CHIMP_COMM, BNXT_CHIMP_COMM_TRIGGER = 0x0, 0x100
BNXT_BACKING_STORE = ((0, 64), (1, 0), (2, 128), (3, 0), (4, 2), (5, 0), (6, 0), (14, 1024), (15, 0))
WQE_SIZE, RING_ENTRIES, CQ_ENTRIES, MTU = 128, 32, 128, 4096
def db_value(xid, typ, index, epoch):
  return (xid & bnxt.DBC_DBC_XID_MASK | bnxt.DBC_DBC_PATH_ROCE | typ | bnxt.BNXT_QPLIB_DBR_VALID) << 32 | \
         index & bnxt.DBC_DBC_INDEX_MASK | epoch << bnxt.BNXT_QPLIB_DBR_EPOCH_SHIFT

# a wqe: type, flags, size in 16 byte units, then the send header (length at dword 1) or the receive header, then the sge (va, key, size)
def send_wqe(va:int, key:int, size:int) -> bytes:
  return struct.pack("<BBBBIIIIIIIQII", 0, bnxt.SQ_SEND_FLAGS_SIGNAL_COMP, 3, 0, 0, size, 0, 0, 0, 0, 0, va, key, size)
def recv_wqe(va:int, key:int, size:int) -> bytes: return struct.pack("<BBBBIII16xQII", 0x80, 0, 3, 0, 0, 0, 0, va, key, size)
def msn_entry(wqe_idx:int, psn:int, size:int) -> tuple[int, int]: # the entry and the psn after this send
  psn &= 0xffffff
  nxt = (psn + max(1, ceildiv(size, MTU))) & 0xffffff
  # Thor2 static-mode retransmission indexes WQEs, not 16B slots.
  return (wqe_idx % RING_ENTRIES) << bnxt.SQ_MSN_SEARCH_START_IDX_SFT | nxt << bnxt.SQ_MSN_SEARCH_NEXT_PSN_SFT | psn, nxt
def cqe_ready(cqe:bytes, cons:int) -> bool: return (cqe[24] & bnxt.CQ_BASE_TOGGLE) != (cons // CQ_ENTRIES & 1)

def _pbl(dev, paddrs, queue=False):
  if len(paddrs) == 1: return 0, paddrs[0]
  values = [p | bnxt.PTU_PTE_VALID for p in paddrs]
  if queue:
    values[-1], values[-2] = values[-1] | bnxt.PTU_PTE_LAST, values[-2] | bnxt.PTU_PTE_NEXT_TO_LAST
  table, table_paddrs = dev.pci_dev.alloc_sysmem(ceildiv(len(values), 512) * 0x1000)
  table[:len(values) * 8] = struct.pack(f"<{len(values)}Q", *values)
  if len(table_paddrs) == 1: return 1, table_paddrs[0]
  top, top_paddrs = dev.pci_dev.alloc_sysmem(0x1000)
  top[:len(table_paddrs) * 8] = struct.pack(f"<{len(table_paddrs)}Q", *(p | bnxt.PTU_PTE_VALID for p in table_paddrs))
  return 2, top_paddrs[0]

def _queue(dev, stride:int=16, aux=False):
  mem, paddrs = dev.pci_dev.alloc_sysmem(0x1000 + aux * 0x1000)
  level, base = _pbl(dev, paddrs, queue=True)
  return {"mem":mem, "paddrs":paddrs, "stride":stride, "prod":0, "cons":0, "level":level, "base":base}

def _qread(q, i):
  off = i % (0x1000 // q["stride"]) * q["stride"]
  return q["mem"][off:off + q["stride"]]

def _qwrite(q, i, data, aux=False):
  off = 0x1000 + i % RING_ENTRIES * 8 if aux else i % (0x1000 // q["stride"]) * q["stride"]
  q["mem"][off:off + len(data)] = data

class BNXTDev:
  def __init__(self, pci_dev:PCIDevice, ip:str=getenv("BNXT_IP", "10.0.0.1")):
    self.pci_dev, self.devfmt = pci_dev, pci_dev.pcibus
    self.bar0, self.db = pci_dev.map_bar(0, fmt='I'), pci_dev.map_bar(2, fmt='Q')
    pci_dev.write_config(pci.PCI_COMMAND, pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER, 2)
    self.resp, self.resp_pa = pci_dev.alloc_sysmem(0x1000)
    self.seq = 0

    ver = self.hwrm("ver_get")
    if DEBUG >= 2: print(f"bnxt {self.devfmt}: firmware {ver.hwrm_fw_maj_8b}.{ver.hwrm_fw_min_8b}.{ver.hwrm_fw_bld_8b}")
    self.hwrm("func_reset", timeout_ms=40000)
    caps = self.hwrm("func_qcaps", fid=0xffff)
    self.mac, self.port_id = int.from_bytes(bytes(caps.mac_address), 'big'), caps.port_id
    self.hwrm("func_drv_rgtr")
    self.db_off = self.hwrm("func_qcfg", fid=0xffff).legacy_l2_db_size_kb * 1024

    self.setup_backing_store()
    self._open_rcfw()
    self._open_l2()
    self.local_gid = ipv4_to_gid(ip)
    gids, mac = (ctypes.c_uint32 * 4)(*(int.from_bytes(self.local_gid[i:i + 4], 'big') for i in (12, 8, 4, 0))), self.mac.to_bytes(6, 'big')
    smac = (ctypes.c_uint16 * 3)(*(int.from_bytes(mac[i:i + 2], 'big') for i in (0, 2, 4)))
    self.gid_id = self.rcfw("add_gid", gid=gids, src_mac=smac).xid

    if DEBUG >= 2: print(f"bnxt {self.devfmt}: booted mac={self.mac.to_bytes(6, 'big').hex(':')} gid={self.local_gid.hex()}")

  def hwrm(self, name, timeout_ms=10000, **fields):
    inp, out = getattr(bnxt, f"struct_hwrm_{name}_input"), getattr(bnxt, f"struct_hwrm_{name}_output")
    opcode = getattr(bnxt, f"HWRM_{name.upper()}")
    self.seq = (self.seq + 1) & 0xffff
    data = bytes(inp(req_type=opcode, cmpl_ring=bnxt.BNXT_HWRM_NO_CMPL_RING, seq_id=self.seq, target_id=bnxt.BNXT_HWRM_TARGET,
                     resp_addr=self.resp_pa[0], **fields))
    self.resp[:] = bytes(len(self.resp))
    System.memory_barrier()
    for i, w in enumerate(memoryview(bytearray(data.ljust(bnxt.HWRM_MAX_REQ_LEN, b'\0'))).cast('I')):
      self.bar0[BNXT_CHIMP_COMM // 4 + i] = w
    self.bar0[BNXT_CHIMP_COMM_TRIGGER // 4] = 1
    def hdr(): return bnxt.struct_hwrm_resp_hdr.from_buffer_copy(bytes(self.resp[:8]))
    wait_cond(lambda: (n := hdr().resp_len) and hdr().seq_id == self.seq and self.resp[n - 1], timeout_ms=timeout_ms, msg=f"HWRM {name}")
    ret = out.from_buffer_copy(bytes(self.resp[:ctypes.sizeof(out)]))
    assert ret.error_code == 0, f"HWRM {name}: {ret.error_code}"
    return ret

  def setup_backing_store(self):
    counts: dict[int, int] = {}
    for typ, extra in BNXT_BACKING_STORE:
      caps = self.hwrm("func_backing_store_qcaps_v2", type=typ)
      size, splits = caps.entry_size, tuple(getattr(caps, f"split_entry_{j}") for j in range(caps.subtype_valid_cnt))
      counts[typ] = n = counts[0] if typ == 15 else max(caps.min_num_entries, sum(splits) + extra)
      # a zero bitmap means the type has a single instance 0
      for instance in [i for i in range(8) if caps.instance_bit_map >> i & 1] or [0]:
        mem, paddrs = self.pci_dev.alloc_sysmem(ceildiv(n * size, 0x1000) * 0x1000)
        if caps.ctx_init_value:
          init = bytearray(len(mem))
          init[caps.ctx_init_offset::size] = bytes([caps.ctx_init_value]) * len(range(caps.ctx_init_offset, len(mem), size))
          mem[:] = init
        lvl, base = _pbl(self, paddrs)
        self.hwrm("func_backing_store_cfg_v2", type=typ, instance=instance, entry_size=size, num_entries=n, page_dir=base,
          page_size_pbl_level=lvl, subtype_valid_cnt=len(splits),
          flags=bnxt.FUNC_BACKING_STORE_CFG_V2_REQ_FLAGS_BS_CFG_ALL_DONE if typ == 15 else 0,
          **{f"split_entry_{j}": v for j, v in enumerate(splits)})

  def _open_rcfw(self):
    self.rcfw_first = True

    self.creq = _queue(self)
    self.creq_id = self.hwrm("ring_alloc", ring_type=bnxt.RING_ALLOC_REQ_RING_TYPE_NQ, page_tbl_addr=self.creq["base"],
      page_size=12, page_tbl_depth=self.creq["level"], length=256, int_mode=bnxt.RING_ALLOC_REQ_INT_MODE_MSIX).ring_id

    self.cmdq = _queue(self)
    self.doorbell(self.creq_id, bnxt.DBC_DBC_TYPE_NQ_ARM, 0, 0)
    init = bnxt.struct_cmdq_init(cmdq_pbl=self.cmdq["base"], creq_ring_id=self.creq_id,
                                 cmdq_size_cmdq_lvl=256 << bnxt.CMDQ_INIT_CMDQ_SIZE_SFT)

    System.memory_barrier()
    for i, w in enumerate(memoryview(bytearray(bytes(init))).cast('I')): self.bar0[bnxt.RCFW_COMM_BASE_OFFSET // 4 + i] = w

    _, p = self.pci_dev.alloc_sysmem(0x1000)
    self.rcfw("initialize_fw", stat_ctx_id=self.hwrm("stat_ctx_alloc", stats_dma_addr=p[0], stats_dma_length=176).stat_ctx_id,
      flags=bnxt.CMDQ_INITIALIZE_FW_FLAGS_HW_REQUESTER_RETX_SUPPORTED)

    # RoCE notification ring: never armed or serviced, but CQ and L2 ring allocation require one
    nq = _queue(self)
    self.nq_id = self.hwrm("ring_alloc", ring_type=bnxt.RING_ALLOC_REQ_RING_TYPE_NQ, page_tbl_addr=nq["base"],
      page_size=12, page_tbl_depth=nq["level"], length=16, logical_id=1, int_mode=bnxt.RING_ALLOC_REQ_INT_MODE_MSIX).ring_id

  def rcfw(self, name, timeout_ms=20000, **fields):
    req_t, resp_t = getattr(bnxt, f"struct_cmdq_{name}"), getattr(bnxt, f"struct_creq_{name}_resp")
    op = getattr(bnxt, f"CMDQ_BASE_OPCODE_{name.upper()}")
    data = bytes(req_t(opcode=op, cmd_size=(slots := ceildiv(ctypes.sizeof(req_t), 16)), **fields)).ljust(slots * 16, b'\0')
    for i in range(slots): _qwrite(self.cmdq, self.cmdq["prod"] + i, data[i * 16:(i + 1) * 16])

    self.cmdq["prod"] += slots
    prod = self.cmdq["prod"] & 0xffff
    if self.rcfw_first: prod, self.rcfw_first = prod | 1 << bnxt.FIRMWARE_FIRST_FLAG, False

    System.memory_barrier()

    self.bar0[(bnxt.RCFW_COMM_BASE_OFFSET + bnxt.RCFW_PF_VF_COMM_PROD_OFFSET) // 4] = prod
    self.bar0[(bnxt.RCFW_COMM_BASE_OFFSET + bnxt.RCFW_COMM_TRIG_OFFSET) // 4] = bnxt.RCFW_CMDQ_TRIG_VAL

    def poll():
      h = bnxt.struct_creq_base.from_buffer_copy(bytes(_qread(self.creq, self.creq["cons"])))
      return bool(h.v & bnxt.CREQ_BASE_V) != bool((self.creq["cons"] // 256) & 1)
    wait_cond(poll, timeout_ms=timeout_ms, msg=f"RCFW {name}")

    ret = resp_t.from_buffer_copy(bytes(_qread(self.creq, self.creq["cons"])))
    self.creq["cons"] += 1

    # NQ_ARM also publishes the CREQ consumer index, which is what frees ring space for the next command
    self.doorbell(self.creq_id, bnxt.DBC_DBC_TYPE_NQ_ARM, self.creq["cons"] & 255, (self.creq["cons"] // 256) & 1)
    assert ret.status == 0, f"RCFW {name}: {ret.status}"

    if BNXT_DEBUG >= 1: print(f"bnxt {self.devfmt}: rcfw {name} xid={getattr(ret, 'xid', 0):#x}")
    return ret

  def fini(self): self.hwrm("func_drv_unrgtr") # the firmware forgets this driver: the next takeover starts clean

  def doorbell(self, xid, typ, index, epoch):
    System.memory_barrier()
    self.db[self.db_off // 8] = db_value(xid, typ, index, epoch)

  # L2 receive path, required for RoCE ingress even though no ethernet receive buffers are posted
  def _open_l2(self):
    cq = _queue(self)
    ci = self.hwrm("ring_alloc", enables=bnxt.RING_ALLOC_REQ_ENABLES_NQ_RING_ID_VALID, ring_type=bnxt.RING_ALLOC_REQ_RING_TYPE_L2_CMPL,
      page_tbl_addr=cq["base"], page_size=12, page_tbl_depth=cq["level"], length=16, nq_ring_id=self.nq_id).ring_id
    rx = _queue(self)
    ri = self.hwrm("ring_alloc", enables=bnxt.RING_ALLOC_REQ_ENABLES_NQ_RING_ID_VALID |
      bnxt.RING_ALLOC_REQ_ENABLES_RX_BUF_SIZE_VALID, ring_type=bnxt.RING_ALLOC_REQ_RING_TYPE_RX, page_tbl_addr=rx["base"],
      page_size=12, page_tbl_depth=rx["level"], length=16, rx_buf_size=640, nq_ring_id=self.nq_id).ring_id
    vi = self.hwrm("vnic_alloc").vnic_id
    self.hwrm("vnic_cfg", enables=bnxt.VNIC_CFG_REQ_ENABLES_MRU | bnxt.VNIC_CFG_REQ_ENABLES_DEFAULT_RX_RING_ID |
      bnxt.VNIC_CFG_REQ_ENABLES_DEFAULT_CMPL_RING_ID, vnic_id=vi, mru=9018,
      default_rx_ring_id=ri, default_cmpl_ring_id=ci)
    self.hwrm("cfa_l2_filter_alloc", flags=bnxt.CFA_L2_FILTER_ALLOC_REQ_FLAGS_PATH_RX,
      enables=bnxt.CFA_L2_FILTER_ALLOC_REQ_ENABLES_L2_ADDR | bnxt.CFA_L2_FILTER_ALLOC_REQ_ENABLES_L2_ADDR_MASK |
      bnxt.CFA_L2_FILTER_ALLOC_REQ_ENABLES_DST_ID, l2_addr=tuple(self.mac.to_bytes(6, 'big')), l2_addr_mask=(0xff,) * 6, dst_id=vi)

  # a memory region over pages: the key addresses [va, va + size) as those pages
  def register_mem(self, paddrs:list[int], size:int, log_page_size:int=12, va:int|None=None) -> int:
    level, base = _pbl(self, paddrs[:ceildiv(size, 1 << log_page_size)])
    return self.rcfw("register_mr", flags=bnxt.CMDQ_REGISTER_MR_FLAGS_ALLOC_MR,
      log2_pg_size_lvl=level << bnxt.CMDQ_REGISTER_MR_LVL_SFT | log_page_size << bnxt.CMDQ_REGISTER_MR_LOG2_PG_SIZE_SFT,
      access=bnxt.CMDQ_REGISTER_MR_ACCESS_LOCAL_WRITE | bnxt.CMDQ_REGISTER_MR_ACCESS_REMOTE_WRITE,
      log2_pbl_pg_size=12, pbl=base, va=paddrs[0] if va is None else va, mr_size=size).xid
  def unregister_mem(self, key:int): self.rcfw("deregister_mr", lkey=key)

class BNXTQP:
  def __init__(self, dev:BNXTDev):
    self.dev, self.sq_psn = dev, 0
    self.scq, self.rcq = _queue(dev, ctypes.sizeof(bnxt.struct_cq_base)), _queue(dev, ctypes.sizeof(bnxt.struct_cq_base))
    self.scq_id, self.rcq_id = (dev.rcfw("create_cq", cq_size=CQ_ENTRIES, pbl=q["base"], pg_size_lvl=q["level"], cq_fco_cnq_id=dev.nq_id).xid
                                for q in (self.scq, self.rcq))
    self.sq, self.rq = _queue(dev, WQE_SIZE, aux=True), _queue(dev, WQE_SIZE)
    self.qpn = dev.rcfw("create_qp", type=bnxt.CMDQ_CREATE_QP_TYPE_RC, scq_cid=self.scq_id, rcq_cid=self.rcq_id,
      sq_size=RING_ENTRIES, sq_fwo_sq_sge=6, sq_pbl=self.sq["base"], sq_pg_size_sq_lvl=self.sq["level"],
      rq_size=RING_ENTRIES, rq_fwo_rq_sge=6, rq_pbl=self.rq["base"], rq_pg_size_rq_lvl=self.rq["level"]).xid
    self.qp_op(1, BNXT_INIT_MASK, access=BNXT_ACCESS, pkey=0xffff)

  def qp_op(self, state, mask, network_type=0, **fields):
    self.dev.rcfw("modify_qp", qp_cid=self.qpn, modify_mask=mask,
      network_type_en_sqd_async_notify_new_state=state | network_type, **fields)

  def connect(self, qpn:int, gid:bytes, mac:int):
    network_type = bnxt.CMDQ_MODIFY_QP_NETWORK_TYPE_ROCEV2_IPV4
    dgid = (ctypes.c_uint32 * 4)(*(int.from_bytes(gid[i:i + 4], 'little') for i in (0, 4, 8, 12)))
    dmac = (ctypes.c_uint16 * 3)(*(int.from_bytes(mac.to_bytes(6, 'big')[i:i + 2], 'little') for i in (0, 2, 4)))

    self.qp_op(2, BNXT_RTR_MASK, network_type=network_type, qp_type=bnxt.CMDQ_MODIFY_QP_QP_TYPE_RC, access=BNXT_ACCESS,
      pkey=0xffff, dgid=dgid, sgid_index=self.dev.gid_id, hop_limit=64, dest_mac=dmac, min_rnr_timer=1,
      path_mtu_pingpong_push_enable=bnxt.CMDQ_MODIFY_QP_PATH_MTU_MTU_4096, max_dest_rd_atomic=4, dest_qp_id=qpn)
    # a send to a not yet posted receive is retried forever
    self.qp_op(3, BNXT_RTS_MASK, network_type=network_type, qp_type=bnxt.CMDQ_MODIFY_QP_QP_TYPE_RC, access=BNXT_ACCESS,
      max_rd_atomic=1, rnr_retry=7, retry_cnt=7, timeout=14)

    if BNXT_DEBUG >= 1: print(f"bnxt: QP {self.qpn:#x} connected (remote={qpn:#x})")

  # cpu driven posting and polling, the hcq2 queues encode the same rings from the runtime program and the gpu
  def poll(self, cq, cq_id, timeout_ms=20000) -> bytes:
    wait_cond(lambda: cqe_ready(bytes(_qread(cq, cq["cons"])), cq["cons"]), timeout_ms=timeout_ms, msg="BNXT CQ")
    raw = bytes(_qread(cq, cq["cons"]))
    cq["cons"] += 1
    self.dev.doorbell(cq_id, bnxt.DBC_DBC_TYPE_CQ, cq["cons"] % CQ_ENTRIES, (cq["cons"] // CQ_ENTRIES) & 1)
    assert raw[25] == 0, f"BNXT CQE status {raw[25]}"
    return raw

  def post_send(self, va:int, key:int, size:int):
    idx = self.sq["prod"]
    _qwrite(self.sq, idx, send_wqe(va, key, size))
    entry, self.sq_psn = msn_entry(idx, self.sq_psn, size)
    _qwrite(self.sq, idx, struct.pack("<Q", entry), aux=True)
    self.sq["prod"] += 1
    self.dev.doorbell(self.qpn, bnxt.DBC_DBC_TYPE_SQ, self.sq["prod"] % RING_ENTRIES, (self.sq["prod"] // RING_ENTRIES) & 1)

  def post_recv(self, va:int, key:int, size:int):
    _qwrite(self.rq, self.rq["prod"], recv_wqe(va, key, size))
    self.rq["prod"] += 1
    self.dev.doorbell(self.qpn, bnxt.DBC_DBC_TYPE_RQ, self.rq["prod"] % RING_ENTRIES, (self.rq["prod"] // RING_ENTRIES) & 1)

