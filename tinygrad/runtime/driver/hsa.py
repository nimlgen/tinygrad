import ctypes, functools
import gpuctypes.hsa as hsa
from tinygrad.helpers import init_c_var

def check(status):
  if status != 0: raise RuntimeError(f"HSA Error {status}")

# Precalulated AQL info
AQL_PACKET_SIZE = ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t)
EMPTY_SIGNAL = hsa.hsa_signal_t()

DISPATCH_KERNEL_SETUP = 3 << hsa.HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS
DISPATCH_KERNEL_HEADER = 0
DISPATCH_KERNEL_HEADER |= 1 << hsa.HSA_PACKET_HEADER_BARRIER
DISPATCH_KERNEL_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
DISPATCH_KERNEL_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
DISPATCH_KERNEL_HEADER |= hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH << hsa.HSA_PACKET_HEADER_TYPE

BARRIER_HEADER = 0
BARRIER_HEADER |= 1 << hsa.HSA_PACKET_HEADER_BARRIER
BARRIER_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
BARRIER_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
BARRIER_HEADER |= hsa.HSA_PACKET_TYPE_BARRIER_AND << hsa.HSA_PACKET_HEADER_TYPE

class HWQueue:
  def __init__(self, agent, sz=-1):
    self.agent = agent
    self.signals = []

    check(hsa.hsa_agent_get_info(self.agent, hsa.HSA_AGENT_INFO_QUEUE_MAX_SIZE, ctypes.byref(max_queue_size := ctypes.c_uint32())))
    queue_size = min(max_queue_size, sz) if sz != -1 else max_queue_size

    null_func = ctypes.CFUNCTYPE(None, hsa.hsa_status_t, ctypes.POINTER(hsa.struct_hsa_queue_s), ctypes.POINTER(None))()
    self.hw_queue = init_c_var(ctypes.POINTER(hsa.hsa_queue_t)(), lambda x: check(hsa.hsa_queue_create(self.agent, queue_size, hsa.HSA_QUEUE_TYPE_SINGLE, null_func, None, (1<<32)-1, (1<<32)-1, ctypes.byref(x))))
    self.write_addr = self.hw_queue.contents.base_address
    self.next_doorbell_index = -1

    check(hsa.hsa_amd_profiling_set_profiler_enabled(self.hw_queue, 1))
    check(hsa.hsa_system_get_info(hsa.HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, ctypes.byref(gpu_freq := ctypes.c_uint64())))
    self.clocks_to_time = 1 / gpu_freq.value # TODO: double check

  def submit_kernel(self, prg, global_size, local_size, kernargs, profile):
    if profile: check(hsa.hsa_signal_create(1, 0, None, ctypes.byref(signal := hsa.hsa_signal_t())))

    packet = hsa.hsa_kernel_dispatch_packet_t.from_address(self.write_addr)
    packet.workgroup_size_x = local_size[0]
    packet.workgroup_size_y = local_size[1]
    packet.workgroup_size_z = local_size[2]
    packet.grid_size_x = global_size[0] * local_size[0]
    packet.grid_size_y = global_size[1] * local_size[1]
    packet.grid_size_z = global_size[2] * local_size[2]
    packet.kernel_object = prg.handle
    packet.kernarg_address = kernargs
    packet.group_segment_size = prg.group_segment_size
    packet.private_segment_size = prg.private_segment_size
    packet.setup = DISPATCH_KERNEL_SETUP
    packet.header = DISPATCH_KERNEL_HEADER
    if profile: packet.completion_signal = signal

    self.write_addr += AQL_PACKET_SIZE
    self.next_doorbell_index += 1
    hsa.hsa_signal_store_screlease(self.hw_queue.contents.doorbell_signal, self.next_doorbell_index)

    if profile:
      hsa.hsa_signal_wait_scacquire(signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
      check(hsa.hsa_amd_profiling_get_dispatch_time(self.agent, signal, ctypes.byref(timings := hsa.hsa_amd_profiling_dispatch_time_t())))
      return (timings.end - timings.start) * self.clocks_to_time

  def submit_barrier(self):
    check(hsa.hsa_signal_create(1, 0, None, ctypes.byref(signal := hsa.hsa_signal_t())))

    packet = hsa.hsa_barrier_and_packet_t.from_address(self.write_addr)
    packet.dep_signal[0] = EMPTY_SIGNAL
    packet.dep_signal[1] = EMPTY_SIGNAL
    packet.dep_signal[2] = EMPTY_SIGNAL
    packet.dep_signal[3] = EMPTY_SIGNAL
    packet.dep_signal[4] = EMPTY_SIGNAL
    packet.completion_signal = signal
    packet.header = BARRIER_HEADER

    self.signals.append(signal)
    self.write_addr += AQL_PACKET_SIZE
    self.next_doorbell_index += 1
    hsa.hsa_signal_store_screlease(self.hw_queue.contents.doorbell_signal, self.next_doorbell_index)

  def wait(self):
    self.submit_barrier()
    for sig in self.signals:
      hsa.hsa_signal_wait_scacquire(sig, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
    self.signals.clear()

@functools.lru_cache(None)
def find_hsa_agent(typ, device_id):
  @ctypes.CFUNCTYPE(hsa.hsa_status_t, hsa.hsa_agent_t, ctypes.c_void_p)
  def __filter_agent(agent, data):
    status = hsa.hsa_agent_get_info(agent, hsa.HSA_AGENT_INFO_DEVICE, ctypes.byref(device_type := hsa.hsa_device_type_t()))
    if status == 0 and device_type.value == typ:
      ret = ctypes.cast(data, ctypes.POINTER(hsa.hsa_agent_t))
      if ret[0].handle < device_id:
        ret[0].handle = ret[0].handle + 1
        return hsa.HSA_STATUS_SUCCESS

      ret = ctypes.cast(data, ctypes.POINTER(hsa.hsa_agent_t))
      ret[0] = agent
      return hsa.HSA_STATUS_INFO_BREAK
    return hsa.HSA_STATUS_SUCCESS

  agent = hsa.hsa_agent_t()
  agent.handle = 0
  hsa.hsa_iterate_agents(__filter_agent, ctypes.byref(agent))
  return agent

def find_memory_pool(agent, segtyp=-1, flags=-1, location=-1):
  @ctypes.CFUNCTYPE(hsa.hsa_status_t, hsa.hsa_amd_memory_pool_t, ctypes.c_void_p)
  def __filter_amd_memory_pools(mem_pool, data):
    if segtyp != -1:
      check(hsa.hsa_amd_memory_pool_get_info(mem_pool, hsa.HSA_AMD_MEMORY_POOL_INFO_SEGMENT, ctypes.byref(segment := hsa.hsa_amd_segment_t())))
      if segment.value != segtyp: return hsa.HSA_STATUS_SUCCESS

    if flags != -1:
      check(hsa.hsa_amd_memory_pool_get_info(mem_pool, hsa.HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, ctypes.byref(fgs := hsa.hsa_amd_memory_pool_global_flag_t())))
      if fgs.value != flags: return hsa.HSA_STATUS_SUCCESS

    if location != -1:
      check(hsa.hsa_amd_memory_pool_get_info(mem_pool, hsa.HSA_AMD_MEMORY_POOL_INFO_LOCATION, ctypes.byref(loc := hsa.hsa_amd_memory_pool_location_t())))
      if loc.value != location: return hsa.HSA_STATUS_SUCCESS

    ret = ctypes.cast(data, ctypes.POINTER(hsa.hsa_amd_memory_pool_t))
    ret[0] = mem_pool
    return hsa.HSA_STATUS_INFO_BREAK

  region = hsa.hsa_amd_memory_pool_t()
  region.handle = 0
  hsa.hsa_amd_agent_iterate_memory_pools(agent, __filter_amd_memory_pools, ctypes.byref(region))
  return region

def find_old_mem_zone(agent, typ, req_flags):
  @ctypes.CFUNCTYPE(hsa.hsa_status_t, hsa.hsa_region_t, ctypes.c_void_p)
  def filter_shared_memtype(region, data):
    check(hsa.hsa_region_get_info(region, hsa.HSA_REGION_INFO_SEGMENT, ctypes.byref(segment := hsa.hsa_region_segment_t())))
    if segment.value != typ:
      return hsa.HSA_STATUS_SUCCESS
    
    check(hsa.hsa_region_get_info(region, hsa.HSA_REGION_INFO_GLOBAL_FLAGS, ctypes.byref(flags := hsa.hsa_region_global_flag_t())))
    if flags.value & req_flags:
      ret = ctypes.cast(data, ctypes.POINTER(hsa.hsa_region_t))
      ret[0] = region
      return hsa.HSA_STATUS_INFO_BREAK
    return hsa.HSA_STATUS_SUCCESS

  region = hsa.hsa_region_t()
  region.handle = -1
  hsa.hsa_agent_iterate_regions(agent, filter_shared_memtype, ctypes.byref(region))
  return region