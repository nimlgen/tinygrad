import ctypes, collections
import tinygrad.runtime.autogen.hsakmt as hsakmt
import tinygrad.runtime.autogen.hsa as hsa
from tinygrad.device import BufferOptions
from tinygrad.helpers import init_c_var

def check(status):
  if status != 0:
    hsa.hsa_status_string(status, ctypes.byref(status_str := ctypes.POINTER(ctypes.c_char)()))
    raise RuntimeError(f"HSA Error {status}: {ctypes.string_at(status_str).decode()}")

# Precalulated AQL info
AQL_PACKET_SIZE = ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t)
EMPTY_SIGNAL = hsa.hsa_signal_t()

DISPATCH_KERNEL_SETUP = 3 << hsa.HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS
DISPATCH_KERNEL_HEADER  = 1 << hsa.HSA_PACKET_HEADER_BARRIER
DISPATCH_KERNEL_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
DISPATCH_KERNEL_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
DISPATCH_KERNEL_HEADER |= hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH << hsa.HSA_PACKET_HEADER_TYPE

BARRIER_HEADER  = 1 << hsa.HSA_PACKET_HEADER_BARRIER
BARRIER_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
BARRIER_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
BARRIER_HEADER |= hsa.HSA_PACKET_TYPE_BARRIER_AND << hsa.HSA_PACKET_HEADER_TYPE

SDMA_MAX_COPY_SIZE = 0x3fffe0

class AQLQueue:
  def __init__(self, device, sz=-1):
    self.device = device
    self.wait_signals = []

    check(hsa.hsa_agent_get_info(self.device.agent, hsa.HSA_AGENT_INFO_QUEUE_MAX_SIZE, ctypes.byref(max_queue_size := ctypes.c_uint32())))
    queue_size = min(max_queue_size.value, sz) if sz != -1 else max_queue_size.value

    null_func = ctypes.CFUNCTYPE(None, hsa.hsa_status_t, ctypes.POINTER(hsa.struct_hsa_queue_s), ctypes.c_void_p)()
    self.hw_queue = init_c_var(ctypes.POINTER(hsa.hsa_queue_t)(), lambda x: check(
      hsa.hsa_queue_create(self.device.agent, queue_size, hsa.HSA_QUEUE_TYPE_SINGLE, null_func, None, (1<<32)-1, (1<<32)-1, ctypes.byref(x))))

    self.next_doorbell_index = 0
    self.queue_size = self.hw_queue.contents.size
    self.write_addr = self.hw_queue.contents.base_address
    self.write_addr_end = self.hw_queue.contents.base_address + (AQL_PACKET_SIZE * self.queue_size) - 1
    self.available_packet_slots = self.queue_size

    check(hsa.hsa_amd_queue_set_priority(self.hw_queue, hsa.HSA_AMD_QUEUE_PRIORITY_HIGH))
    check(hsa.hsa_amd_profiling_set_profiler_enabled(self.hw_queue, 1))

  def __del__(self):
    if hasattr(self, 'hw_queue'): check(hsa.hsa_queue_destroy(self.hw_queue))

  def submit_kernel(self, prg, global_size, local_size, kernargs, need_signal=False):
    if self.available_packet_slots == 0: self._wait_queue()
    signal = self._alloc_signal(reusable=True) if need_signal else EMPTY_SIGNAL

    packet = hsa.hsa_kernel_dispatch_packet_t.from_address(self.write_addr)
    packet.workgroup_size_x = local_size[0]
    packet.workgroup_size_y = local_size[1]
    packet.workgroup_size_z = local_size[2]
    packet.reserved0 = 0
    packet.grid_size_x = global_size[0] * local_size[0]
    packet.grid_size_y = global_size[1] * local_size[1]
    packet.grid_size_z = global_size[2] * local_size[2]
    packet.private_segment_size = prg.private_segment_size
    packet.group_segment_size = prg.group_segment_size
    packet.kernel_object = prg.handle
    packet.kernarg_address = kernargs
    packet.reserved2 = 0
    packet.completion_signal = signal
    packet.setup = DISPATCH_KERNEL_SETUP
    packet.header = DISPATCH_KERNEL_HEADER
    self._submit_packet()

    return signal

  def submit_barrier(self, wait_signals=None, need_signal=False, completion_signal=None):
    assert wait_signals is None or len(wait_signals) <= 5
    if self.available_packet_slots == 0: self._wait_queue()
    signal = (completion_signal or self._alloc_signal(reusable=True)) if need_signal else EMPTY_SIGNAL

    packet = hsa.hsa_barrier_and_packet_t.from_address(self.write_addr)
    packet.reserved0 = 0
    packet.reserved1 = 0
    for i in range(5):
      packet.dep_signal[i] = wait_signals[i] if wait_signals and len(wait_signals) > i else EMPTY_SIGNAL
    packet.reserved2 = 0
    packet.completion_signal = signal
    packet.header = BARRIER_HEADER
    self._submit_packet()

    return signal

  def blit_packets(self, packet_addr, packet_cnt):
    if self.available_packet_slots < packet_cnt: self._wait_queue(packet_cnt)

    tail_blit_packets = min(((self.write_addr_end + 1) - self.write_addr) // 64, packet_cnt)
    rem_packet_cnt = packet_cnt - tail_blit_packets
    ctypes.memmove(self.write_addr, packet_addr, AQL_PACKET_SIZE * tail_blit_packets)
    self.write_addr += AQL_PACKET_SIZE * tail_blit_packets
    if self.write_addr > self.write_addr_end: self.write_addr = self.hw_queue.contents.base_address
    if tail_blit_packets > 0:
      ctypes.memmove(self.write_addr, packet_addr + AQL_PACKET_SIZE * tail_blit_packets, AQL_PACKET_SIZE * rem_packet_cnt)
      self.write_addr += AQL_PACKET_SIZE * rem_packet_cnt

    self.next_doorbell_index += packet_cnt
    hsa.hsa_queue_store_write_index_screlease(self.hw_queue, self.next_doorbell_index + 1)
    hsa.hsa_signal_store_screlease(self.hw_queue.contents.doorbell_signal, self.next_doorbell_index)

  def wait(self):
    signal = self.submit_barrier(need_signal=True)
    hsa.hsa_signal_wait_scacquire(signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
    self.available_packet_slots = self.queue_size

  def _wait_queue(self, need_packets=1):
    while self.available_packet_slots < need_packets:
      rindex = hsa.hsa_queue_load_read_index_relaxed(self.hw_queue)
      self.available_packet_slots = self.queue_size - (self.next_doorbell_index - rindex)

  def _submit_packet(self):
    hsa.hsa_queue_store_write_index_relaxed(self.hw_queue, self.next_doorbell_index + 1)
    hsa.hsa_signal_store_screlease(self.hw_queue.contents.doorbell_signal, self.next_doorbell_index)

    self.write_addr += AQL_PACKET_SIZE
    if self.write_addr > self.write_addr_end: self.write_addr = self.hw_queue.contents.base_address
    self.next_doorbell_index += 1
    self.available_packet_slots -= 1

  def _alloc_signal(self, reusable=False): return self.device.alloc_signal(reusable=reusable)

class SDMAQueue:
  def __init__(self, device):
    self.queue_size = 1 << 20
    self.queue_start = device.allocator._alloc_with_options(self.queue_size, BufferOptions(host=True))
    self.write_addr = self.queue_start
    self.write_addr_end = self.write_addr + self.queue_size - 1

    hsakmt.hsaKmtCreateQueue(device.node_id, hsakmt.HSA_QUEUE_SDMA, 100, hsakmt.HSA_QUEUE_PRIORITY_MAXIMUM, self.queue_start, self.queue_size, None,
                             ctypes.byref(queue_desc := hsakmt.HsaQueueResource()))
    self.queue_desc = queue_desc
    self.next_doorbell_index = 0

  def __del__(self):
    pass

  def _build_poll_cmd(self, addr, value):
    cmd = hsakmt.SDMA_PKT_POLL_REGMEM.from_address(self.write_addr)
    ctypes.memset(self.write_addr, 0, ctypes.sizeof(hsakmt.SDMA_PKT_POLL_REGMEM))

    cmd.HEADER_UNION.op = hsakmt.SDMA_OP_POLL_REGMEM
    cmd.HEADER_UNION.mem_poll = 1
    cmd.HEADER_UNION.func = 0x3 # is equal
    cmd.ADDR_LO_UNION.addr_31_0 = addr & 0xffffffff
    cmd.ADDR_HI_UNION.addr_63_32 = (addr >> 32) & 0xffffffff

    cmd.VALUE_UNION.value = value

    cmd.MASK_UNION.mask = 0xffffffff; # the whole content.

    cmd.DW5_UNION.interval = 0x04
    cmd.DW5_UNION.retry_count = 0xfff # retry forever.

    self._submit_cmd(ctypes.sizeof(hsakmt.SDMA_PKT_POLL_REGMEM))
    return cmd

  def _build_atomic_dec_cmd(self, addr):
    cmd = hsakmt.SDMA_PKT_ATOMIC.from_address(self.write_addr)
    ctypes.memset(self.write_addr, 0, ctypes.sizeof(hsakmt.SDMA_PKT_ATOMIC))

    cmd.HEADER_UNION.op = hsakmt.SDMA_OP_ATOMIC
    cmd.HEADER_UNION.operation = hsakmt.SDMA_ATOMIC_ADD64
    cmd.ADDR_LO_UNION.addr_31_0 = addr & 0xffffffff
    cmd.ADDR_HI_UNION.addr_63_32 = (addr >> 32) & 0xffffffff

    cmd.SRC_DATA_LO_UNION.src_data_31_0 = 0xffffffff
    cmd.SRC_DATA_HI_UNION.src_data_63_32 = 0xffffffff

    self._submit_cmd(ctypes.sizeof(hsakmt.SDMA_PKT_ATOMIC))
    return cmd

  def _build_cache_cmd(self, invalidate=False):
    cmd = hsakmt.SDMA_PKT_GCR.from_address(self.write_addr)
    ctypes.memset(self.write_addr, 0, ctypes.sizeof(hsakmt.SDMA_PKT_GCR))

    cmd.HEADER_UNION.op = hsakmt.SDMA_OP_GCR
    cmd.HEADER_UNION.sub_op = hsakmt.SDMA_SUBOP_USER_GCR
    cmd.WORD2_UNION.GCR_CONTROL_GL2_WB = 1
    cmd.WORD2_UNION.GCR_CONTROL_GLK_WB = 1

    if invalidate:
      cmd.WORD2_UNION.GCR_CONTROL_GL2_INV = 1
      cmd.WORD2_UNION.GCR_CONTROL_GL1_INV = 1
      cmd.WORD2_UNION.GCR_CONTROL_GLV_INV = 1
      cmd.WORD2_UNION.GCR_CONTROL_GLK_INV = 1

    # TODO: They inv the whole cache, try the required part only?
    cmd.WORD2_UNION.GCR_CONTROL_GL2_RANGE = 0

    self._submit_cmd(ctypes.sizeof(hsakmt.SDMA_PKT_GCR))
    return cmd

  def _build_hdp_cmd(self):
    cmd = hsakmt.SDMA_PKT_HDP_FLUSH.from_address(self.write_addr)
    ctypes.memset(self.write_addr, 0, ctypes.sizeof(hsakmt.SDMA_PKT_HDP_FLUSH))
    cmd.DW_0_DATA = 0x8
    cmd.DW_1_DATA = 0x0
    cmd.DW_2_DATA = 0x80000000
    cmd.DW_3_DATA = 0x0
    cmd.DW_4_DATA = 0x0
    cmd.DW_5_DATA = 0x0

    self._submit_cmd(ctypes.sizeof(hsakmt.SDMA_PKT_HDP_FLUSH))
    return cmd
  
  def _build_fence_cmd(self, fence_addr, value):
    cmd = hsakmt.SDMA_PKT_FENCE.from_address(self.write_addr)
    ctypes.memset(self.write_addr, 0, ctypes.sizeof(hsakmt.SDMA_PKT_FENCE))

    cmd.HEADER_UNION.op = hsakmt.SDMA_OP_FENCE
    cmd.ADDR_LO_UNION.addr_31_0 = fence_addr & 0xffffffff
    cmd.ADDR_HI_UNION.addr_63_32 = (fence_addr >> 32) & 0xffffffff
    cmd.DATA_UNION.data = value
    self._submit_cmd(ctypes.sizeof(hsakmt.SDMA_PKT_FENCE))
    return cmd

  def _build_trap_cmd(self, event_id):
    cmd = hsakmt.SDMA_PKT_TRAP.from_address(self.write_addr)
    ctypes.memset(self.write_addr, 0, ctypes.sizeof(hsakmt.SDMA_PKT_TRAP))

    cmd.HEADER_UNION.op = hsakmt.SDMA_OP_TRAP
    cmd.INT_CONTEXT_UNION.int_ctx = event_id
    self._submit_cmd(ctypes.sizeof(hsakmt.SDMA_PKT_TRAP))
    return cmd

  def _build_cp_cmd(self, dest, src, sz):
    copies_commands = (sz + SDMA_MAX_COPY_SIZE - 1) // SDMA_MAX_COPY_SIZE
    copied = 0

    for _ in range(copies_commands):
      copy_size = min(sz - copied, SDMA_MAX_COPY_SIZE)
      src_off = src + copied
      dest_off = dest + copied

      cmd = hsakmt.SDMA_PKT_COPY_LINEAR.from_address(self.write_addr)
      ctypes.memset(self.write_addr, 0, ctypes.sizeof(hsakmt.SDMA_PKT_COPY_LINEAR))

      cmd.HEADER_UNION.op = hsakmt.SDMA_OP_COPY
      cmd.HEADER_UNION.sub_op = hsakmt.SDMA_SUBOP_COPY_LINEAR
      cmd.COUNT_UNION.count = copy_size - 1

      cmd.SRC_ADDR_LO_UNION.src_addr_31_0 = src_off & 0xffffffff
      cmd.SRC_ADDR_HI_UNION.src_addr_63_32 = (src_off >> 32) & 0xffffffff

      cmd.DST_ADDR_LO_UNION.dst_addr_31_0 = dest_off & 0xffffffff
      cmd.DST_ADDR_HI_UNION.dst_addr_63_32 = (dest_off >> 32) & 0xffffffff

      copied += copy_size
      self._submit_cmd(ctypes.sizeof(hsakmt.SDMA_PKT_COPY_LINEAR))

  def _ring_doorbell(self):
    self.queue_desc.Queue_write_ptr[0] = self.next_doorbell_index
    self.queue_desc.Queue_DoorBell[0] = self.next_doorbell_index

  def submit_copy(self, dest, src, nbytes, wait_signals=None, completion_signal=None):
    if wait_signals is not None:
      for sig in wait_signals:
        check(hsa.hsa_amd_signal_value_pointer(sig, ctypes.byref(val_ptr := ctypes.POINTER(ctypes.c_int64)())))
        self._build_poll_cmd(ctypes.addressof(val_ptr.contents), 0)
        self._build_poll_cmd(ctypes.addressof(val_ptr.contents) + 4, 0)

    self._build_hdp_cmd()
    self._build_cache_cmd(invalidate=True)
    self._build_cp_cmd(dest, src, nbytes)
    self._build_cache_cmd()

    # Signal that we have finished.
    if completion_signal is not None:
      check(hsa.hsa_amd_signal_value_pointer(completion_signal, ctypes.byref(val_ptr := ctypes.POINTER(ctypes.c_int64)())))
      self._build_atomic_dec_cmd(ctypes.addressof(val_ptr.contents))

      mailbox_ptr = hsa_signal_event_mailbox_ptr(completion_signal)
      event_id = hsa_signal_event_id(completion_signal)
      if mailbox_ptr != 0:
        self._build_fence_cmd(mailbox_ptr, event_id)
        self._build_trap_cmd(event_id)

    self._ring_doorbell()

  def _submit_cmd(self, size):
    self.write_addr += size
    self.next_doorbell_index += size
    # TODO: this is kindof not the best, but 256 bytes is on tail every time.
    if self.write_addr + 256 > self.write_addr_end:
      ctypes.memset(self.write_addr, 0, self.write_addr_end - self.write_addr + 1) # NOP the ending
      self.next_doorbell_index += self.write_addr_end - self.write_addr + 1
      self.write_addr = self.queue_start

def scan_agents():
  agents = collections.defaultdict(list)

  @ctypes.CFUNCTYPE(hsa.hsa_status_t, hsa.hsa_agent_t, ctypes.c_void_p)
  def __scan_agents(agent, data):
    status = hsa.hsa_agent_get_info(agent, hsa.HSA_AGENT_INFO_DEVICE, ctypes.byref(device_type := hsa.hsa_device_type_t()))
    if status == 0: agents[device_type.value].append(agent)
    return hsa.HSA_STATUS_SUCCESS

  hsa.hsa_iterate_agents(__scan_agents, None)
  return agents

def find_memory_pool(agent, segtyp=-1, location=-1):
  @ctypes.CFUNCTYPE(hsa.hsa_status_t, hsa.hsa_amd_memory_pool_t, ctypes.c_void_p)
  def __filter_amd_memory_pools(mem_pool, data):
    check(hsa.hsa_amd_memory_pool_get_info(mem_pool, hsa.HSA_AMD_MEMORY_POOL_INFO_SEGMENT, ctypes.byref(segment := hsa.hsa_amd_segment_t())))
    if segtyp >= 0 and segment.value != segtyp: return hsa.HSA_STATUS_SUCCESS

    check(hsa.hsa_amd_memory_pool_get_info(mem_pool, hsa.HSA_AMD_MEMORY_POOL_INFO_LOCATION, ctypes.byref(loc:=hsa.hsa_amd_memory_pool_location_t())))
    if location >= 0 and loc.value != location: return hsa.HSA_STATUS_SUCCESS

    check(hsa.hsa_amd_memory_pool_get_info(mem_pool, hsa.HSA_AMD_MEMORY_POOL_INFO_SIZE, ctypes.byref(sz := ctypes.c_size_t())))
    if sz.value == 0: return hsa.HSA_STATUS_SUCCESS

    ret = ctypes.cast(data, ctypes.POINTER(hsa.hsa_amd_memory_pool_t))
    ret[0] = mem_pool
    return hsa.HSA_STATUS_INFO_BREAK

  hsa.hsa_amd_agent_iterate_memory_pools(agent, __filter_amd_memory_pools, ctypes.byref(region := hsa.hsa_amd_memory_pool_t()))
  return region

def hsa_signal_event_mailbox_ptr(signal: hsa.hsa_signal_t):
  # HACK: depends on ABI
  check(hsa.hsa_amd_signal_value_pointer(signal, ctypes.byref(ptr := ctypes.POINTER(ctypes.c_int64)())))
  return ptr[1]

def hsa_signal_event_id(signal: hsa.hsa_signal_t):
  # HACK: depends on ABI
  check(hsa.hsa_amd_signal_value_pointer(signal, ctypes.byref(ptr := ctypes.POINTER(ctypes.c_int64)())))
  return ptr[2]