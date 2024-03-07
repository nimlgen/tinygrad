import ctypes, collections
import tinygrad.runtime.autogen.hsa as hsa
from tinygrad.helpers import init_c_var, from_mv

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

COPY_BARRIER_HEADER  = hsa.HSA_FENCE_SCOPE_NONE << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
COPY_BARRIER_HEADER |= hsa.HSA_FENCE_SCOPE_NONE << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
COPY_BARRIER_HEADER |= hsa.HSA_PACKET_TYPE_BARRIER_AND << hsa.HSA_PACKET_HEADER_TYPE

kCopyAlignedVecWidth = 4
kCopyAlignedUnroll = 1
class CopyAligned(ctypes.Structure):
  _fields_ = [
      ("phase1_src_start", ctypes.c_uint64),
      ("phase1_dst_start", ctypes.c_uint64),
      ("phase2_src_start", ctypes.c_uint64),
      ("phase2_dst_start", ctypes.c_uint64),
      ("phase3_src_start", ctypes.c_uint64),
      ("phase3_dst_start", ctypes.c_uint64),
      ("phase4_src_start", ctypes.c_uint64),
      ("phase4_dst_start", ctypes.c_uint64),
      ("phase4_src_end", ctypes.c_uint64),
      ("phase4_dst_end", ctypes.c_uint64),
      ("num_workitems", ctypes.c_uint32),
  ]
  _align_ = 16

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

  def submit_kernel(self, prg, global_size, local_size, kernargs, completion_signal=None, need_signal=False):
    if self.available_packet_slots == 0: self._wait_queue()
    signal = (completion_signal or self._alloc_signal(reusable=True)) if need_signal else EMPTY_SIGNAL

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

  def submit_barrier(self, wait_signals=None, need_signal=False, completion_signal=None, header=BARRIER_HEADER):
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
    packet.header = header
    self._submit_packet()

    return signal
  
  # def submit_copy(self, dest, src, size, wait_signals=None, need_signal=False, completion_signal=None, kernargs=None):
  #   assert src & 0x3 == dest & 0x3, "only aligned supported"
  #   assert wait_signals is None or len(wait_signals) < 5 # TODO: remove this
  #   if self.available_packet_slots < 2: self._wait_queue()

  #   if self.device.copy_kern_object is None: 
  #     self.device.copy_kern_object = int(input(f"dai handle {self.device.device_id}: "), base=10)

  #   num_cus_ = 48
  #   num_workitems = 64 * 4 * num_cus_
  #   phase1_size = min(size, (0x100 - dest & 0xFF) & 0xFF)
  #   phase2_block = num_workitems * 4 * kCopyAlignedUnroll * kCopyAlignedVecWidth
  #   phase2_size = ((size - phase1_size) // phase2_block) * phase2_block
  #   phase3_size = ((size - phase1_size - phase2_size) // 4) * 4

  #   if kernargs is None: kernargs = self.device.alloc_kernargs(ctypes.sizeof(CopyAligned))
  #   copy_st = CopyAligned.from_address(kernargs)
  #   copy_st.phase1_src_start = src
  #   copy_st.phase1_dst_start = dest
  #   copy_st.phase2_src_start = src + phase1_size
  #   copy_st.phase2_dst_start = dest + phase1_size
  #   copy_st.phase3_src_start = src + phase1_size + phase2_size
  #   copy_st.phase3_dst_start = dest + phase1_size + phase2_size
  #   copy_st.phase4_src_start = src + phase1_size + phase2_size + phase3_size
  #   copy_st.phase4_dst_start = dest + phase1_size + phase2_size + phase3_size
  #   copy_st.phase4_src_end = src + size
  #   copy_st.phase4_dst_end = dest + size
  #   copy_st.num_workitems = num_workitems

  #   global_size, local_size = (num_workitems, 1, 1), (64, 1, 1)

  #   self.submit_barrier(wait_signals, header=COPY_BARRIER_HEADER)

  #   signal = (completion_signal or self._alloc_signal(reusable=True)) if need_signal else EMPTY_SIGNAL

  #   packet = hsa.hsa_kernel_dispatch_packet_t.from_address(self.write_addr)
  #   packet.workgroup_size_x = local_size[0]
  #   packet.workgroup_size_y = local_size[1]
  #   packet.workgroup_size_z = local_size[2]
  #   packet.reserved0 = 0
  #   packet.grid_size_x = global_size[0]
  #   packet.grid_size_y = global_size[1]
  #   packet.grid_size_z = global_size[2]
  #   packet.private_segment_size = 0
  #   packet.group_segment_size = 0
  #   packet.kernel_object = self.device.copy_kern_object
  #   packet.kernarg_address = kernargs
  #   packet.reserved2 = 0
  #   packet.completion_signal = signal
  #   packet.setup = DISPATCH_KERNEL_SETUP
  #   packet.header = DISPATCH_KERNEL_HEADER

  #   self._submit_packet()

  def blit_packets(self, packet_addr, packet_cnt):
    if packet_cnt == 0: return
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

kExecHeader = bytearray([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0xc2, 0x00, 0x8c, 0x00, 0x84, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

kCodeCopyAligned11 = bytearray([0x00, 0x01, 0x08, 0xf4, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x08, 0xf4,
  0x10, 0x00, 0x00, 0xf8, 0x00, 0x03, 0x08, 0xf4, 0x20, 0x00, 0x00, 0xf8,
  0x00, 0x04, 0x08, 0xf4, 0x30, 0x00, 0x00, 0xf8, 0x00, 0x05, 0x08, 0xf4,
  0x40, 0x00, 0x00, 0xf8, 0x00, 0x06, 0x00, 0xf4, 0x50, 0x00, 0x00, 0xf8,
  0x07, 0xfc, 0x89, 0xbf, 0x02, 0x86, 0x02, 0x84, 0x00, 0x6a, 0x00, 0xd7,
  0x02, 0x00, 0x02, 0x00, 0x05, 0x02, 0x06, 0x7e, 0x02, 0x6a, 0x00, 0xd7,
  0x00, 0x09, 0x00, 0x00, 0x03, 0x6a, 0x20, 0xd5, 0x03, 0x01, 0xa9, 0x01,
  0x07, 0x02, 0x0a, 0x7e, 0x04, 0x6a, 0x00, 0xd7, 0x00, 0x0d, 0x00, 0x00,
  0x05, 0x6a, 0x20, 0xd5, 0x05, 0x01, 0xa9, 0x01, 0x6a, 0x00, 0x59, 0xd4,
  0x02, 0x11, 0x00, 0x00, 0x0f, 0x00, 0xa3, 0xbf, 0x7e, 0x6a, 0xfe, 0x8b,
  0x00, 0x00, 0x40, 0xdc, 0x02, 0x00, 0x7c, 0x01, 0xf7, 0x03, 0x89, 0xbf,
  0x02, 0x6a, 0x00, 0xd7, 0x02, 0x31, 0x00, 0x00, 0x03, 0x6a, 0x20, 0xd5,
  0x03, 0x01, 0xa9, 0x01, 0x00, 0x00, 0x60, 0xdc, 0x04, 0x01, 0x7c, 0x00,
  0x04, 0x6a, 0x00, 0xd7, 0x04, 0x31, 0x00, 0x00, 0x05, 0x6a, 0x20, 0xd5,
  0x05, 0x01, 0xa9, 0x01, 0xee, 0xff, 0xa0, 0xbf, 0xc1, 0x01, 0xfe, 0xbe,
  0x18, 0x84, 0x19, 0x84, 0x84, 0x00, 0x02, 0x30, 0x09, 0x02, 0x06, 0x7e,
  0x02, 0x6a, 0x00, 0xd7, 0x01, 0x11, 0x00, 0x00, 0x03, 0x6a, 0x20, 0xd5,
  0x03, 0x01, 0xa9, 0x01, 0x0b, 0x02, 0x0a, 0x7e, 0x04, 0x6a, 0x00, 0xd7,
  0x01, 0x15, 0x00, 0x00, 0x05, 0x6a, 0x20, 0xd5, 0x05, 0x01, 0xa9, 0x01,
  0x6a, 0x00, 0x59, 0xd4, 0x02, 0x19, 0x00, 0x00, 0x0e, 0x00, 0xa3, 0xbf,
  0x00, 0x00, 0x5c, 0xdc, 0x02, 0x00, 0x7c, 0x08, 0x02, 0x6a, 0x00, 0xd7,
  0x02, 0x33, 0x00, 0x00, 0x03, 0x6a, 0x20, 0xd5, 0x03, 0x01, 0xa9, 0x01,
  0xf7, 0x03, 0x89, 0xbf, 0x00, 0x00, 0x74, 0xdc, 0x04, 0x08, 0x7c, 0x00,
  0x04, 0x6a, 0x00, 0xd7, 0x04, 0x33, 0x00, 0x00, 0x05, 0x6a, 0x20, 0xd5,
  0x05, 0x01, 0xa9, 0x01, 0xef, 0xff, 0xa0, 0xbf, 0x18, 0x82, 0x19, 0x84,
  0x82, 0x00, 0x02, 0x30, 0x0d, 0x02, 0x06, 0x7e, 0x02, 0x6a, 0x00, 0xd7,
  0x01, 0x19, 0x00, 0x00, 0x03, 0x6a, 0x20, 0xd5, 0x03, 0x01, 0xa9, 0x01,
  0x0f, 0x02, 0x0a, 0x7e, 0x04, 0x6a, 0x00, 0xd7, 0x01, 0x1d, 0x00, 0x00,
  0x05, 0x6a, 0x20, 0xd5, 0x05, 0x01, 0xa9, 0x01, 0x6a, 0x00, 0x59, 0xd4,
  0x02, 0x21, 0x00, 0x00, 0x0f, 0x00, 0xa3, 0xbf, 0x7e, 0x6a, 0xfe, 0x8b,
  0x00, 0x00, 0x50, 0xdc, 0x02, 0x00, 0x7c, 0x01, 0x02, 0x6a, 0x00, 0xd7,
  0x02, 0x33, 0x00, 0x00, 0x03, 0x6a, 0x20, 0xd5, 0x03, 0x01, 0xa9, 0x01,
  0xf7, 0x03, 0x89, 0xbf, 0x00, 0x00, 0x68, 0xdc, 0x04, 0x01, 0x7c, 0x00,
  0x04, 0x6a, 0x00, 0xd7, 0x04, 0x33, 0x00, 0x00, 0x05, 0x6a, 0x20, 0xd5,
  0x05, 0x01, 0xa9, 0x01, 0xee, 0xff, 0xa0, 0xbf, 0xc1, 0x01, 0xfe, 0xbe,
  0x11, 0x02, 0x06, 0x7e, 0x02, 0x6a, 0x00, 0xd7, 0x00, 0x21, 0x00, 0x00,
  0x03, 0x6a, 0x20, 0xd5, 0x03, 0x01, 0xa9, 0x01, 0x13, 0x02, 0x0a, 0x7e,
  0x04, 0x6a, 0x00, 0xd7, 0x00, 0x25, 0x00, 0x00, 0x05, 0x6a, 0x20, 0xd5,
  0x05, 0x01, 0xa9, 0x01, 0x6a, 0x00, 0x59, 0xd4, 0x02, 0x29, 0x00, 0x00,
  0x06, 0x00, 0xa3, 0xbf, 0x7e, 0x6a, 0xfe, 0x8b, 0x00, 0x00, 0x40, 0xdc,
  0x02, 0x00, 0x7c, 0x01, 0xf7, 0x03, 0x89, 0xbf, 0x00, 0x00, 0x60, 0xdc,
  0x04, 0x01, 0x7c, 0x00, 0x00, 0x00, 0xb0, 0xbf])

def load_copy_code(device, sgrps, vgrps):
  from tinygrad.runtime.ops_hsa import HSADevice
  # gran_sgprs = max(0, (sgrps - 1) // 8)
  # gran_vgprs = max(0, (vgrps - 1) // 4)

  src = memoryview(kExecHeader + kCodeCopyAligned11)
  check(hsa.hsa_amd_memory_pool_allocate(device.gpu_mempool, src.nbytes, 0, ctypes.byref(dest := ctypes.c_void_p())))
  dest = dest.value

  copy_signal = device.alloc_signal(reusable=True)
  sync_signal = device.hw_queue.submit_barrier(need_signal=True)
  c_agents = (hsa.hsa_agent_t*2)(*[HSADevice.cpu_agent, device.agent])
  check(hsa.hsa_amd_memory_lock_to_pool(from_mv(src), src.nbytes, c_agents, 2, HSADevice.cpu_mempool, 0, ctypes.byref(addr:=ctypes.c_void_p())))
  check(hsa.hsa_amd_memory_async_copy_on_engine(dest, device.agent, addr, HSADevice.cpu_agent, src.nbytes,
                                                1, ctypes.byref(sync_signal), copy_signal, hsa.HSA_AMD_SDMA_ENGINE_0, True))
  hsa.hsa_signal_wait_scacquire(copy_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
  return dest
