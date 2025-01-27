import ctypes, struct, time
from tinygrad.runtime.autogen import libc, libusb

class Asm236x:
  def __init__(self, name):
    print("Opening", name)
    self.usb_ctx = ctypes.POINTER(libusb.struct_libusb_context)()
    ret = libusb.libusb_init(ctypes.byref(self.usb_ctx))
    if ret != 0:
      raise Exception(f"Failed to init libusb: {ret}")

    # Set debug level
    libusb.libusb_set_option(self.usb_ctx, libusb.LIBUSB_OPTION_LOG_LEVEL, libusb.LIBUSB_LOG_LEVEL_DEBUG)

    # Open device
    self.handle = libusb.libusb_open_device_with_vid_pid(self.usb_ctx, 0x174c, 0x2362)
    if not self.handle:
      raise Exception("Failed to open device")

    ret = libusb.libusb_detach_kernel_driver(self.handle, 0)
    ret = libusb.libusb_detach_kernel_driver(self.handle, 1)
    ret = libusb.libusb_detach_kernel_driver(self.handle, 2)
    ret = libusb.libusb_detach_kernel_driver(self.handle, 3)
    ret = libusb.libusb_detach_kernel_driver(self.handle, 4)
    # libusb.libusb_attach_kernel_driver(self.handle, 0)
    # exit(0)

    # Detach kernel driver if needed
    if libusb.libusb_kernel_driver_active(self.handle, 0) == 1:
      ret = libusb.libusb_detach_kernel_driver(self.handle, 0)
      if ret != 0:
        raise Exception(f"Failed to detach kernel driver: {ret}")

    libusb.libusb_reset_device(self.handle)

    # Set configuration
    ret = libusb.libusb_set_configuration(self.handle, 1)
    if ret != 0:
      raise Exception(f"Failed to set configuration: {ret}")

    # Claim interface
    ret = libusb.libusb_claim_interface(self.handle, 0)
    if ret != 0:
      raise Exception(f"Failed to claim interface: {ret}")

    # Set alternate setting to 1 (this is crucial!)
    ret = libusb.libusb_set_interface_alt_setting(self.handle, 0, 1)
    if ret != 0:
      raise Exception(f"Failed to set alternate setting: {ret}")

    # Clear halts on endpoints
    libusb.libusb_clear_halt(self.handle, 0x02)  # Command OUT endpoint
    libusb.libusb_clear_halt(self.handle, 0x04)  # Command OUT endpoint
    libusb.libusb_clear_halt(self.handle, 0x81)  # Status IN endpoint
    libusb.libusb_clear_halt(self.handle, 0x83)  # Status IN endpoint

    print("USB device initialized successfully")

    self.setup_seq()

  def _send_ops_and_wait(self, *cmds):
    for x in cmds: libusb.libusb_submit_transfer(x)

    while True:
      libusb.libusb_handle_events(self.usb_ctx)

      all_complete = True
      for transfer in cmds:
        print(transfer.contents.status)
        if transfer.contents.status == libusb.LIBUSB_TRANSFER_COMPLETED:
          continue
        elif transfer.contents.status != libusb.LIBUSB_TRANSFER_COMPLETED:
          if transfer.contents.status != 0xff:
            raise Exception(f"Transfer failed with status: {transfer.contents.status}")
          all_complete = False
      if all_complete: return
    
  
  def _send_setup_seq(self, cmd_str):
    cmd = cmd_str.replace(" ", "")
    usb_cmd = bytes([int(cmd[i:i+2], 16) for i in range(0, len(cmd), 2)])

    status_buffer = (ctypes.c_uint8 * 112)()
    status_transfer = libusb.libusb_alloc_transfer(0)
    self.setup_transfer(status_transfer, 0x83, status_buffer, len(status_buffer))

    res_transfer = libusb.libusb_alloc_transfer(0)
    read_data = (ctypes.c_uint8 * 36)()
    # print(libusb.libusb_fill_bulk_transfer.argtypes)
    # function takes exactly 1 argument
    self.setup_transfer(res_transfer, 0x81, read_data, 36)
    # ret = libusb.libusb_fill_bulk_transfer(res_transfer, self.handle, 0x81, read_data, buf_len, None, None, 1000)

    cmd_transfer = libusb.libusb_alloc_transfer(0)
    data = (ctypes.c_uint8 * len(usb_cmd))(*bytes(usb_cmd))
    self.setup_transfer(cmd_transfer, 4, data, len(usb_cmd))

    self._send_ops_and_wait(cmd_transfer)
    self._send_ops_and_wait(res_transfer)

  def setup_seq(self):
    def control_transfer(bmRequestType, bRequest, wValue, wIndex):
      ret = libusb.libusb_control_transfer(
        self.handle,
        bmRequestType,
        bRequest,
        wValue,
        wIndex,
        None,  # data buffer or length
        0,
        1000  # timeout
      )
      if ret < 0: raise Exception(f"Control transfer failed: {ret}")
      return ret

    # Replicate the sequence:
    # s 23 03 0017 0002 0000 0
    # control_transfer(0x23, 0x03, 0x0017, 0x0002)
    
    # s 00 01 0030 0000 0000 0
    control_transfer(0x00, 0x01, 0x0030, 0x0000)
    
    # s 23 03 0018 0002 0000 0
    # control_transfer(0x23, 0x03, 0x0018, 0x0002)
    
    # s 00 01 0031 0000 0000 0
    control_transfer(0x00, 0x01, 0x0031, 0x0000)
    
    # s 01 0b 0001 0000 0000 0
    control_transfer(0x01, 0x0b, 0x0001, 0x0000)
    
    # s 23 03 0017 3202 0000 0
    # control_transfer(0x23, 0x03, 0x0017, 0x3202)
    
    # s 00 03 0030 0000 0000 0
    control_transfer(0x00, 0x03, 0x0030, 0x0000)
    
    # s 23 03 0018 2802 0000 0
    # control_transfer(0x23, 0x03, 0x0018, 0x2802)
    
    # s 00 03 0031 0000 0000 0
    control_transfer(0x00, 0x03, 0x0031, 0x0000)

    self._send_setup_seq("01000001 00000000 00000000 00000000 12000000 24000000 00000000 00000000")

  def write(self, start_addr, bulk_commands):
    # print('w', hex(start_addr), bulk_commands)

    usb_cmd = [
      0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xe5, 0x24, 0x00, 0xb2, 0x1a, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    ]

    # print(bulk_commands)
    for offset, cmd in enumerate(bulk_commands):
      usb_cmd[17] = cmd
      usb_cmd[19] = (start_addr + offset) >> 8
      usb_cmd[20] = (start_addr + offset) & 0xff

      data = (ctypes.c_uint8 * len(usb_cmd))(*bytes(usb_cmd))
      ret = libusb.libusb_bulk_transfer(self.handle, 4, data, len(usb_cmd), ctypes.byref(transferred:=ctypes.c_int()), 1000)

      assert transferred.value == len(usb_cmd)

  def setup_transfer(self, transfer, endpoint, data, length):
    transfer.contents.dev_handle = self.handle
    transfer.contents.status = 0xff
    transfer.contents.flags = 0
    transfer.contents.endpoint = endpoint
    transfer.contents.type = libusb.LIBUSB_TRANSFER_TYPE_BULK
    transfer.contents.timeout = 1000
    transfer.contents.length = length
    # transfer.contents.callback = None
    transfer.contents.user_data = None
    transfer.contents.buffer = data
    transfer.contents.num_iso_packets = 0

  def read(self, start_addr, read_len, stride=255):
    # print('r', hex(start_addr), hex(read_len))
    data = bytearray(read_len)

    usb_cmd = [
      0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xe4, 0x24, 0x00, 0xb2, 0x1a, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    ]

    for i in range(0, read_len, stride):
      remaining = read_len - i
      buf_len = min(stride, remaining)
      usb_cmd[17] = buf_len
      usb_cmd[19] = (start_addr + i) >> 8
      usb_cmd[20] = (start_addr + i) & 0xff

      status_buffer = (ctypes.c_uint8 * 112)()
      status_transfer = libusb.libusb_alloc_transfer(0)
      self.setup_transfer(status_transfer, 0x83, status_buffer, len(status_buffer))

      res_transfer = libusb.libusb_alloc_transfer(0)
      read_data = (ctypes.c_uint8 * buf_len)()
      # print(libusb.libusb_fill_bulk_transfer.argtypes)
      # function takes exactly 1 argument
      self.setup_transfer(res_transfer, 0x81, read_data, buf_len)
      # ret = libusb.libusb_fill_bulk_transfer(res_transfer, self.handle, 0x81, read_data, buf_len, None, None, 1000)

      cmd_transfer = libusb.libusb_alloc_transfer(0)
      data = (ctypes.c_uint8 * len(usb_cmd))(*bytes(usb_cmd))
      self.setup_transfer(cmd_transfer, 4, data, len(usb_cmd))
      # ret = libusb.libusb_fill_bulk_transfer(cmd_transfer, self.handle, 4, data, len(usb_cmd), None, None, 1000)
      # assert transferred.value == len(usb_cmd)

      libusb.libusb_submit_transfer(status_transfer)
      libusb.libusb_submit_transfer(res_transfer)
      libusb.libusb_submit_transfer(cmd_transfer)

      while True:
        ret = libusb.libusb_handle_events(self.usb_ctx)

        all_complete = True
        for transfer in [res_transfer, cmd_transfer, status_transfer]:
          print(transfer.contents.status)
          if transfer.contents.status == libusb.LIBUSB_TRANSFER_COMPLETED:
            continue
          elif transfer.contents.status != libusb.LIBUSB_TRANSFER_COMPLETED:
            if transfer.contents.status != 0xff:
                raise Exception(f"Transfer failed with status: {transfer.contents.status}")
            all_complete = False
        if all_complete:
          print("OK")
          break

      data[i:i+buf_len] = read_data

    return bytes(data)

  def pcie_request(self, fmt_type, address, value=None, size=4, cnt=10):
    st = time.perf_counter_ns()

    print(self.read(0xB296, 1)[0])
    exit(0)

    assert fmt_type >> 8 == 0
    assert size > 0 and size <= 4

    masked_address = address & 0xfffffffc
    offset = address & 0x00000003

    assert size + offset <= 4

    byte_enable = ((1 << size) - 1) << offset

    if value is not None:
      assert value >> (8 * size) == 0, f"{value}"
      shifted_value = value << (8 * offset)
      self.write(0xB220, struct.pack('>I', value << (8 * offset)))

    self.write(0xB210, struct.pack('>III', 0x00000001 | (fmt_type << 24), byte_enable, masked_address))
    self.write(0xB296, bytes([0x01]))
    self.write(0xB254, bytes([0x0f]))

    while self.read(0xB296, 1)[0] & 4 == 0: continue

    self.write(0xB296, bytes([0x04]))

    if ((fmt_type & 0b11011111) == 0b01000000) or ((fmt_type & 0b10111000) == 0b00110000): return

    while self.read(0xB296, 1)[0] & 2 == 0:
      if self.read(0xB296, 1)[0] & 1:
        self.write(0xB296, bytes([0x01]))
        print("pci redo")
        if cnt > 0: self.pcie_request(fmt_type, address, value, size, cnt=cnt-1)

    self.write(0xB296, bytes([0x02]))

    b284 = self.read(0xB284, 1)[0]
    b284_bit_0 = b284 & 0x01

    completion = struct.unpack('>III', self.read(0xB224, 12))

    if (fmt_type & 0xbe == 0x04):
      # Completion TLPs for configuration requests always have a byte count of 4.
      assert completion[1] & 0xfff == 4
    else:
      assert completion[1] & 0xfff == size

    status_map = {
      0b000: "Successful Completion (SC)",
      0b001: "Unsupported Request (UR)",
      0b010: "Configuration Request Retry Status (CRS)",
      0b100: "Completer Abort (CA)",
    }

    status = (completion[1] >> 13) & 0x7
    if status or ((fmt_type & 0xbe == 0x04) and (((value is None) and (not b284_bit_0)) or ((value is not None) and b284_bit_0))):
      raise Exception("Completion status: {}, 0xB284 bit 0: {}".format(
        status_map.get(status, "Reserved (0b{:03b})".format(status)), b284_bit_0))

    en = time.perf_counter_ns()
    print(f"Prep waiting {(prep_st-st)*1e-6:.2}ms, wt {(en-prep_st)*1e-6:.2}ms")

    if value is None:
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
    if value is not None:
        fmt_type = 0x44

    fmt_type |= cfgreq_type
    address = (bus << 24) | (dev << 19) | (fn << 16) | (byte_addr & 0xfff)

    return self.pcie_request(fmt_type, address, value, size)

  def pcie_mem_req(self, address, value=None, size=4):
    fmt_type = 0x00
    if value is not None:
        fmt_type = 0x40

    return self.pcie_request(fmt_type, address, value, size)

    
