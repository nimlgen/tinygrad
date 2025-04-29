import ctypes, struct, time, os
from tinygrad.runtime.autogen import libc, libusb
from tinygrad.helpers import DEBUG
from hexdump import hexdump

class USBConnector:
  def reset(self):
    if libusb.libusb_kernel_driver_active(self.handle, 0) == 1:
      ret = libusb.libusb_detach_kernel_driver(self.handle, 0)
      print("detach kernel driver")
      if ret != 0: raise Exception(f"Failed to detach kernel driver: {ret}")
      libusb.libusb_reset_device(self.handle)

    ret = libusb.libusb_set_configuration(self.handle, 1)
    if ret != 0: raise Exception(f"Failed to set configuration: {ret}")

    # Claim interface (gives -3 if we reset)
    ret = libusb.libusb_claim_interface(self.handle, 0)
    if ret != 0: raise Exception(f"Failed to claim interface: {ret}")

    # # Set alternate setting to 1 (this is crucial!)
    ret = libusb.libusb_set_interface_alt_setting(self.handle, 0, 1)
    if ret != 0: raise Exception(f"Failed to set alternate setting: {ret}")

    usb_cmd = [
      0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xe4, 0x24, 0x00, 0xb2, 0x1a, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    ]
    self.read_cmd = (ctypes.c_uint8 * len(usb_cmd))(*bytes(usb_cmd))
    self.read_status = (ctypes.c_uint8 * 64)()
    self.read_data = (ctypes.c_uint8 * 256)()

    print("USB device initialized successfully")

    libusb.libusb_clear_halt(self.handle, 0x81)
    libusb.libusb_clear_halt(self.handle, 0x83)
    libusb.libusb_clear_halt(self.handle, 0x04)
    libusb.libusb_clear_halt(self.handle, 0x02)

    self.max_parallel = 16
    endpoints = [0x02, 0x81, 0x83]
    streams = (ctypes.c_uint8*len(endpoints))(*endpoints)
    # print(hex(streams[0]))
    x = libusb.libusb_alloc_streams(self.handle, len(endpoints) * (self.max_parallel + 1), streams, len(endpoints))
    assert x >= 0, f"got {x}"
    # print("hm", x)

    # required to be set, but not a trigger
    # self.write(0xB213, bytes([0x01]))
    # self.write(0xB214, bytes([0, 0]))
    # self.write(0xB216, bytes([0x20]))
    self.res_transfer = libusb.libusb_alloc_transfer(0)
    self.stat_transfer = libusb.libusb_alloc_transfer(0)
    self.cmd_transfer = libusb.libusb_alloc_transfer(0)
    self.in_transfer = libusb.libusb_alloc_transfer(0)

    self.res_transfers = {}
    self.stat_transfers = {}
    self.cmd_transfers = {}
    self.in_transfers = {}
    self.read_cmds = {}
    self.read_statuses = {}
    self.read_datas = {}
    for i in range(self.max_parallel):
      self.res_transfers[i] = libusb.libusb_alloc_transfer(0)
      self.stat_transfers[i] = libusb.libusb_alloc_transfer(0)
      self.cmd_transfers[i] = libusb.libusb_alloc_transfer(0)
      self.in_transfers[i] = libusb.libusb_alloc_transfer(0)
      self.read_cmds[i] = (ctypes.c_uint8 * len(usb_cmd))(*bytes(usb_cmd))
      self.read_statuses[i] = (ctypes.c_uint8 * 64)()
      self.read_datas[i] = (ctypes.c_uint8 * 256)()

    self.cached = {}

  def __init__(self, name):
    self.usb_ctx = ctypes.POINTER(libusb.struct_libusb_context)()
    ret = libusb.libusb_init(ctypes.byref(self.usb_ctx))
    if ret != 0: raise Exception(f"Failed to init libusb: {ret}")

    if DEBUG >= 6: libusb.libusb_set_option(self.usb_ctx, libusb.LIBUSB_OPTION_LOG_LEVEL, 4)
    
    # Open device
    self.handle = libusb.libusb_open_device_with_vid_pid(self.usb_ctx, 0x2d01, 0x3666)
    if not self.handle: raise Exception("Failed to open device")

    # Detach kernel driver if needed
    if libusb.libusb_kernel_driver_active(self.handle, 0) == 1:
      ret = libusb.libusb_detach_kernel_driver(self.handle, 0)
      print("detach kernel driver")
      if ret != 0: raise Exception(f"Failed to detach kernel driver: {ret}")
      libusb.libusb_reset_device(self.handle)

    ret = libusb.libusb_set_configuration(self.handle, 1)
    if ret != 0: raise Exception(f"Failed to set configuration: {ret}")

    # Claim interface (gives -3 if we reset)
    ret = libusb.libusb_claim_interface(self.handle, 0)
    if ret != 0: raise Exception(f"Failed to claim interface: {ret}")

    # # Set alternate setting to 1 (this is crucial!)
    ret = libusb.libusb_set_interface_alt_setting(self.handle, 0, 1)
    if ret != 0: raise Exception(f"Failed to set alternate setting: {ret}")

    usb_cmd = [
      0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xe4, 0x24, 0x00, 0xb2, 0x1a, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    ]
    self.read_cmd = (ctypes.c_uint8 * len(usb_cmd))(*bytes(usb_cmd))
    self.read_status = (ctypes.c_uint8 * 64)()
    self.read_data = (ctypes.c_uint8 * 256)()

    print("USB device initialized successfully")

    libusb.libusb_clear_halt(self.handle, 0x81)
    libusb.libusb_clear_halt(self.handle, 0x83)
    libusb.libusb_clear_halt(self.handle, 0x04)
    libusb.libusb_clear_halt(self.handle, 0x02)

    self.max_parallel = 16
    endpoints = [0x02, 0x81, 0x83]
    streams = (ctypes.c_uint8*len(endpoints))(*endpoints)
    # print(hex(streams[0]))
    x = libusb.libusb_alloc_streams(self.handle, len(endpoints) * (self.max_parallel + 1), streams, len(endpoints))
    assert x >= 0, f"got {x}"
    # print("hm", x)

    # required to be set, but not a trigger
    # self.write(0xB213, bytes([0x01]))
    # self.write(0xB214, bytes([0, 0]))
    # self.write(0xB216, bytes([0x20]))
    self.res_transfer = libusb.libusb_alloc_transfer(0)
    self.stat_transfer = libusb.libusb_alloc_transfer(0)
    self.cmd_transfer = libusb.libusb_alloc_transfer(0)
    self.in_transfer = libusb.libusb_alloc_transfer(0)

    self.res_transfers = {}
    self.stat_transfers = {}
    self.cmd_transfers = {}
    self.in_transfers = {}
    self.read_cmds = {}
    self.read_statuses = {}
    self.read_datas = {}
    for i in range(self.max_parallel):
      self.res_transfers[i] = libusb.libusb_alloc_transfer(0)
      self.stat_transfers[i] = libusb.libusb_alloc_transfer(0)
      self.cmd_transfers[i] = libusb.libusb_alloc_transfer(0)
      self.in_transfers[i] = libusb.libusb_alloc_transfer(0)
      self.read_cmds[i] = (ctypes.c_uint8 * len(usb_cmd))(*bytes(usb_cmd))
      self.read_statuses[i] = (ctypes.c_uint8 * 64)()
      self.read_datas[i] = (ctypes.c_uint8 * 256)()

    self.cached = {}

  def _prep_transfer(self, transfer, endpoint, stream_id, data, length):
    transfer.contents.dev_handle = self.handle
    transfer.contents.status = 0xff
    transfer.contents.flags = 0
    transfer.contents.endpoint = endpoint
    transfer.contents.type = libusb.LIBUSB_TRANSFER_TYPE_BULK if stream_id is None else libusb.LIBUSB_TRANSFER_TYPE_BULK_STREAM
    transfer.contents.timeout = 1000
    transfer.contents.length = length
    transfer.contents.user_data = None
    transfer.contents.buffer = data
    transfer.contents.num_iso_packets = 0
    if stream_id is not None: libusb.libusb_transfer_set_stream_id(transfer, stream_id)
  
  def _submit_and_wait(self, cmds):
    for tr in cmds: libusb.libusb_submit_transfer(tr)

    while True:
      libusb.libusb_handle_events(self.usb_ctx)
      ready = True
      for tr in cmds:
        if tr.contents.status == libusb.LIBUSB_TRANSFER_COMPLETED: continue
        if tr.contents.status != 0xFF: raise RuntimeError(f"EP 0x{tr.contents.endpoint:02X} error: {tr.contents.status}")
        ready = False
      if ready: return

  def _send_batch(self, cdbs, ret_lens=[]):
    ops_sub = []
    # assert len(cdbs) <= 8
    for i in range(len(cdbs)):
      emm = (i % self.max_parallel)
      self.read_cmds[emm][3] = emm + 1
      self.read_cmds[emm][4:6] = len(cdbs[i]).to_bytes(2, 'big')
      self.read_cmds[emm][16:16+len(cdbs[i])] = cdbs[i]
      if ret_lens[i] > 0:
        assert False
        # self._prep_transfer(self.res_transfers[i], 0x81, 1, self.read_cmds[i], ret_lens[i])
        # ops_sub.append(self.res_transfers[i])
      self._prep_transfer(self.stat_transfers[emm], 0x83, emm + 1, self.read_statuses[emm], 64)
      self._prep_transfer(self.cmd_transfers[emm], 0x04, None, self.read_cmds[emm], len(self.read_cmds[emm]))
      ops_sub += [self.stat_transfers[emm], self.cmd_transfers[emm]]
      if i + 1 == len(cdbs) or len(ops_sub) == self.max_parallel * 2:
        self._submit_and_wait(ops_sub)
        ops_sub = []

  def _send(self, cdb, ret_len=0, in_data=None, do_not_send_ack=False, wait=True, ignore_ep=None):
    def __send():
      actual_length = ctypes.c_int(0)
      for i in range(3):
        #assert len(self.read_cmd) == 31
        if ret_len > 0: self._prep_transfer(self.res_transfer, 0x81, 0x1, self.read_data, ret_len)
        self._prep_transfer(self.stat_transfer, 0x83, 0x1, self.read_status, 64)
        self._prep_transfer(self.cmd_transfer, 0x04, None, self.read_cmd, len(self.read_cmd))
        if in_data is not None: self._prep_transfer(self.in_transfer, 0x02, 0x1, in_data, len(in_data))

        # ret = libusb.libusb_bulk_transfer(self.handle, 0x04, self.read_cmd, len(self.read_cmd), ctypes.byref(actual_length), 1000)
        # assert actual_length.value == len(self.read_cmd)

        transfers = []
        if ret_len > 0: transfers.append(self.res_transfer)
        if not do_not_send_ack: transfers += [self.stat_transfer]
        transfers += [self.cmd_transfer]
        if in_data is not None: transfers.append(self.in_transfer)
        self._submit_and_wait(transfers)
        if not wait: return transfers
        # print("stat is ", [x for x in self.read_status])
        return bytes(self.read_data[:])
      return None

    self.read_cmd[3] = 1
    self.read_cmd[4:6] = len(cdb).to_bytes(2, 'big')
    self.read_cmd[16:16+len(cdb)] = cdb
    for j in range(1):
      #print(j)
      read_data = __send()
      if read_data: return read_data
    raise RuntimeError("USB transfer failed")

  def read(self, start_addr, read_len, stride=255):
    if DEBUG >= 4: print("read", hex(start_addr))
    data = bytearray(read_len)

    for i in range(0, read_len, stride):
      remaining = read_len - i
      buf_len = min(stride, remaining)
      current_addr = (start_addr + i)
      assert current_addr >> 17 == 0
      current_addr &= 0x01ffff
      current_addr |= 0x500000
      cdb = struct.pack('>BBBHB', 0xe4, buf_len, current_addr >> 16, current_addr & 0xffff, 0x00)
      data[i:i+buf_len] = self._send(cdb, buf_len)
      for j in range(buf_len): self.cached[current_addr + j] = data[i + j]
    return bytes(data[:read_len])

  def write(self, start_addr, data, ignore_cache=True):
    if DEBUG >= 4: print("write", hex(start_addr))

    cdbs = []
    for offset, value in enumerate(data):
      # if value == 0x18 and not(0xb000 <= start_addr + offset < 0xd000):
        # print(hex(start_addr + offset), "is 0x18")
        # value = 0x17

      current_addr = start_addr + offset
      assert current_addr >> 17 == 0
      current_addr &= 0x01ffff
      current_addr |= 0x500000
      if not ignore_cache and self.cached.get(current_addr, None) == value: continue
      cdbs.append(cdb:=struct.pack('>BBBHB', 0xe5, value, current_addr >> 16, current_addr & 0xffff, 0x00))

      self.cached[current_addr] = value
      # print("dal", offset)
      # self._send(cdb)
    self._send_batch(cdbs, [0] * len(cdbs))

  def write_batch(self, start_addrs, datas, ignores):
    if DEBUG >= 4: print("write", hex(start_addr))
    
    cdbs = []
    for start_addr, data, ignore_cache in zip(start_addrs, datas, ignores):
      for offset, value in enumerate(data):
        current_addr = start_addr + offset
        assert current_addr >> 17 == 0
        current_addr &= 0x01ffff
        current_addr |= 0x500000
        if not ignore_cache and self.cached.get(current_addr, None) == value: continue
        cdbs.append(struct.pack('>BBBHB', 0xe5, value, current_addr >> 16, current_addr & 0xffff, 0x00))

        self.cached[current_addr] = value
        # self._send(cdb)
    self._send_batch(cdbs, [0] * len(cdbs))

  def scsi_write(self, lba, buf):
    # scsi write 0x28 packet
    assert len(buf) % 512 == 0, "buf length must be multiple of 512"

    # scsi write 0x8a packet
    cdb = struct.pack('>BBQIBB',
      0x8A,             # WRITE(16) opcode
      0,                # flags
      lba,              # 64-bit LBA
      len(buf) // 512,  # number of blocks
      0,                # group number
      0                 # control
    )
    ops = self._send(cdb, in_data=buf, wait=True)
    # for o in ops: print(libusb.libusb_cancel_transfer(o))

  def scsi_read(self, lba, num_blocks):
    # scsi read 0x8a packet
    cdb = struct.pack('>BBQIBB',
      0x88,            # READ(16) opcode
      0,               # flags (RDPROTECT, DPO, FUA, etc. all zero here)
      lba,             # 64-bit logical block address
      num_blocks,      # 32-bit transfer length in blocks
      0,               # group number
      0                # control
    )
    return self._send(cdb, ret_len=num_blocks * 512)

  def pcie_write_request_fw(self, address, value):
    cdb = struct.pack('>BII', 0x03, address, value)
    self._send(cdb)    

  def pcie_request(self, fmt_type, address, value=None, size=4, cnt=10):
    assert fmt_type >> 8 == 0
    assert size > 0 and size <= 4
    if DEBUG >= 3: print("pcie_request", hex(fmt_type), hex(address), value, size, cnt)

    # TODO: why is this needed?
    #time.sleep(0.005)

    # TODO: why is this needed? (the write doesn't matter, just that it's using USB)
    #self.write(0xB210, bytes([0]))
    #self.write(0xB210, bytes([0]))
    #self.write(0xB210, bytes([0]))
    #self.write(0xB210, bytes([0]))

    #print(self.read(0xB296, 1)[0])

    masked_address = address & 0xfffffffc
    offset = address & 0x00000003

    assert size + offset <= 4

    byte_enable = ((1 << size) - 1) << offset

    self.addrs, self.datas, self.ignores = [], [], []
    
    if value is not None:
      assert value >> (8 * size) == 0, f"{value}"
      shifted_value = value << (8 * offset)
      # Store write data in PCIE_CONFIG_DATA register (0xB220)
      # self.write(0xB220, struct.pack('>I', value << (8 * offset)), ignore_cache=True)
      self.addrs.append(0xB220)
      self.ignores.append(True)
      self.datas.append(struct.pack('>I', shifted_value))

    # setup address + length
    # self.write(0xB218, struct.pack('>I', masked_address))
    self.addrs.append(0xB218)
    self.ignores.append(False)
    self.datas.append(struct.pack('>I', masked_address))

    assert byte_enable < 0x100
    # self.write(0xB217, bytes([byte_enable]), ignore_cache=False)
    self.addrs.append(0xB217)
    self.ignores.append(False)
    self.datas.append(bytes([byte_enable]))

    # Configure PCIe request by writing to PCIE_REQUEST_CONTROL (0xB210)
    # self.write(0xB210, bytes([fmt_type]), ignore_cache=False)
    self.addrs.append(0xB210)
    self.ignores.append(False)
    self.datas.append(bytes([fmt_type]))

    # Clear any existing PCIe errors before proceeding (PCIE_ERROR_CLEAR: 0xB254)
    # this appears to be the trigger
    # self.write(0xB254, bytes([0x0f]), ignore_cache=True)
    self.addrs.append(0xB254)
    self.ignores.append(True)
    self.datas.append(bytes([0x0f]))
    self.write_batch(self.addrs, self.datas, self.ignores)

    # Wait for PCIe transaction to complete (PCIE_STATUS_REGISTER: 0xB296, bit 2)
    while (stat:=self.read(0xB296, 1)[0]) & 4 == 0:
      print("stat early poll", stat)
      continue
    # assert stat == 6, f"stat was {stat}"
    #print("stat out", stat)

    # Acknowledge completion of PCIe request (PCIE_STATUS_REGISTER: 0xB295)
    self.write(0xB296, bytes([0x04]), ignore_cache=True)

    if ((fmt_type & 0b11011111) == 0b01000000) or ((fmt_type & 0b10111000) == 0b00110000):
      return
      # assert False, "not supported"

    while (stat:=self.read(0xB296, 1)[0]) & 2 == 0:
      print("stat poll", stat)
      if stat & 1:
        self.write(0xB296, bytes([0x01]))
        print("pci redo")
        if cnt > 0: self.pcie_request(fmt_type, address, value, size, cnt=cnt-1)
    assert stat == 2, f"stat read 2 was {stat}"

    # Acknowledge PCIe completion (PCIE_STATUS_REGISTER: 0xB296)
    # self.write(0xB296, bytes([0x02]), ignore_cache=True)

    b284 = self.read(0xB284, 1)[0]
    b284_bit_0 = b284 & 0x01

    # Retrieve completion data from Link Status (0xB22A, 0xB22B)
    completion = struct.unpack('>H', self.read(0xB22A, 2))
    # print(hex(completion[0]))

    # Validate completion status based on PCIe request typ
    if (fmt_type & 0xbe == 0x04):
      # Completion TLPs for configuration requests always have a byte count of 4.
      assert completion[0] & 0xfff == 4
    else:
      assert completion[0] & 0xfff == size

    status_map = {
      0b000: "Successful Completion (SC)",
      0b001: "Unsupported Request (UR)",
      0b010: "Configuration Request Retry Status (CRS)",
      0b100: "Completer Abort (CA)",
    }

    # Extract completion status field
    status = (completion[0] >> 13) & 0x7

    # Handle completion errors or inconsistencies
    if status or ((fmt_type & 0xbe == 0x04) and (((value is None) and (not b284_bit_0)) or ((value is not None) and b284_bit_0))):
      raise RuntimeError("Completion status: {}, 0xB284 bit 0: {}".format(
        status_map.get(status, "Reserved (0b{:03b})".format(status)), b284_bit_0))

    if value is None:
      # Read from PCIE_CONFIG_DATA (0xB220)
      full_value = struct.unpack('>I', self.read(0xB220, 4))[0]
      shifted_value = full_value >> (8 * offset)
      masked_value = shifted_value & ((1 << (8 * size)) - 1)
      return masked_value

  def pcie_cfg_req(self, byte_addr, bus=1, dev=0, fn=0, value=None, size=4):
    assert byte_addr >> 12 == 0

    assert bus >> 8 == 0
    assert dev >> 5 == 0
    assert fn >> 3 == 0

    cfgreq_type = int(bus > 0)
    assert cfgreq_type >> 1 == 0

    fmt_type = 0x04
    if value is not None: fmt_type = 0x44

    fmt_type |= cfgreq_type
    address = (bus << 24) | (dev << 19) | (fn << 16) | (byte_addr & 0xfff)

    return self.pcie_request(fmt_type, address, value, size)

  def pcie_mem_req(self, address, value=None, size=4):
    fmt_type = 0x00
    if value is not None: fmt_type = 0x40

    return self.pcie_request(fmt_type, address, value, size)
