import ctypes, functools, time
import gpuctypes.hsa as hsa
from tinygrad.helpers import init_c_var

def check(status: hsa.hsa_status_t):
  assert status == 0, f"has status is {status}"

@functools.lru_cache(None)
def find_gpu_agent(device_id):
  assert device_id == 0, "FIXME"

  @ctypes.CFUNCTYPE(hsa.hsa_status_t, hsa.hsa_agent_t, ctypes.c_void_p)
  def __filter_amdgpu_agent(agent, data):
    status = hsa.hsa_agent_get_info(agent, hsa.HSA_AGENT_INFO_DEVICE, ctypes.byref(device_type := hsa.hsa_device_type_t()))
    if status == 0 and device_type.value == hsa.HSA_DEVICE_TYPE_GPU:
      ret = ctypes.cast(data, ctypes.POINTER(hsa.hsa_agent_t))
      ret[0] = agent
      return hsa.HSA_STATUS_INFO_BREAK
    return hsa.HSA_STATUS_SUCCESS

  hsa.hsa_iterate_agents(__filter_amdgpu_agent, ctypes.byref(agent := hsa.hsa_agent_t()))
  return agent

@functools.lru_cache(None)
def find_mem_zone(device_id, typ, req_flags):
  agent = find_gpu_agent(device_id)

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

class Kernel:
  def __init__(self, device_id, binary, kernel_name):
    self.device_id = device_id
    agent = find_gpu_agent(self.device_id)
    bin_size = len(binary)

    self.exec = init_c_var(hsa.hsa_executable_t(), lambda x: check(hsa.hsa_executable_create_alt(hsa.HSA_PROFILE_FULL, hsa.HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, None, ctypes.byref(x))))
    check(hsa.hsa_code_object_reader_create_from_memory(binary, bin_size, ctypes.byref(code_reader := hsa.hsa_code_object_reader_t())))
    check(hsa.hsa_executable_load_agent_code_object(self.exec, agent, code_reader, None, None))
    check(hsa.hsa_executable_freeze(self.exec, None))

    sym = kernel_name + ".kd"
    self.kernel = init_c_var(hsa.hsa_executable_symbol_t(), lambda x: check(hsa.hsa_executable_get_symbol_by_name(self.exec, sym.encode("utf-8"), ctypes.byref(agent), ctypes.byref(x))))
    self.handle = init_c_var(ctypes.c_uint64(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, ctypes.byref(x))))
    self.kernargs_segment_size = init_c_var(ctypes.c_uint32(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, ctypes.byref(x))))
    self.group_segment_size = init_c_var(ctypes.c_uint32(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, ctypes.byref(x))))
    self.private_segment_size = init_c_var(ctypes.c_uint32(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, ctypes.byref(x))))

class Queue:
  def __init__(self, device_id, sz=-1):
    check(hsa.hsa_init())
    self.agent = find_gpu_agent(device_id)
    
    check(hsa.hsa_agent_get_info(self.agent, hsa.HSA_AGENT_INFO_QUEUE_MAX_SIZE, ctypes.byref(max_queue_size := ctypes.c_uint32())))
    queue_size = min(max_queue_size, sz) if sz != -1 else max_queue_size

    null_func = ctypes.CFUNCTYPE(None, hsa.hsa_status_t, ctypes.POINTER(hsa.struct_hsa_queue_s), ctypes.POINTER(None))()
    self.hw_queue = init_c_var(ctypes.POINTER(hsa.hsa_queue_t)(), lambda x: check(hsa.hsa_queue_create(self.agent, queue_size, hsa.HSA_QUEUE_TYPE_SINGLE, null_func, None, (1<<32)-1, (1<<32)-1, ctypes.byref(x))))
    self.last_signal = None

  def submit(self, cmds):
    index = None
    for cmd in cmds:
      check(hsa.hsa_signal_create(1, 0, None, ctypes.byref(signal := hsa.hsa_signal_t()))) # TODO: Better sync

      index = hsa.hsa_queue_add_write_index_screlease(self.hw_queue, 1)
      base_address = ctypes.cast(self.hw_queue.contents.base_address, ctypes.POINTER(hsa.hsa_kernel_dispatch_packet_t))
      dispatch_packet_ptr = ctypes.pointer(base_address[index & (self.hw_queue.contents.size - 1)])

      # Fill the packet for the given command.
      dispatch_packet_ptr.contents.completion_signal = signal
      cmd.fill_aql_packet(dispatch_packet_ptr)

      self.last_signal = signal
      
    hsa.hsa_signal_store_relaxed(self.hw_queue.contents.doorbell_signal, index)
    # st = time.perf_counter()
    self.wait() # FIXME
    # print(time.perf_counter()-st)

  # TODO: Better sync
  def wait(self):
    if self.last_signal is None: return
    hsa.hsa_signal_wait_scacquire(self.last_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (2 << 64) - 1, hsa.HSA_WAIT_STATE_BLOCKED)
    self.last_signal = None


class Command:
  def __init__(self): pass
  def fill_aql_packet(self, packet_ptr): pass

class ExecCommand(Command):
  def __init__(self, prg, global_size, local_size, kernargs):
    self.prg, self.global_size, self.local_size, self.kernargs = prg, global_size, local_size, kernargs

  def fill_aql_packet(self, packet_ptr):
    grid_size = tuple(int(g*l) for g,l in zip(self.global_size, self.local_size))

    packet_ptr.contents.setup |= 1 << hsa.HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS

    packet_ptr.contents.workgroup_size_x = self.local_size[0]
    packet_ptr.contents.workgroup_size_y = self.local_size[1]
    packet_ptr.contents.workgroup_size_z = self.local_size[2]
    packet_ptr.contents.grid_size_x = grid_size[0]
    packet_ptr.contents.grid_size_y = grid_size[1]
    packet_ptr.contents.grid_size_z = grid_size[2]
    packet_ptr.contents.kernel_object = self.prg.handle
    packet_ptr.contents.kernarg_address = self.kernargs
    packet_ptr.contents.group_segment_size = self.prg.group_segment_size
    packet_ptr.contents.private_segment_size = 16 << 10 # self.prg.private_segment_size

    header = 0
    header |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE
    header |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE
    header |= hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH << hsa.HSA_PACKET_HEADER_TYPE

    packet_ptr.contents.header = header

class CopyCommand(Command):
  def __init__(self): pass

def launch_kernel(dev, prg, global_size, local_size, extra):
  args_sz = ctypes.cast(extra[3], ctypes.POINTER(ctypes.c_int)).contents.value
  assert prg.kernargs_segment_size.value == args_sz

  # dev.synchronize() # for debug

  # Temprory, think this could be better
  kernarg_region = find_mem_zone(dev.device, hsa.HSA_REGION_SEGMENT_GLOBAL, hsa.HSA_REGION_GLOBAL_FLAG_KERNARG)
  kernargs = init_c_var(ctypes.c_void_p(), lambda x: check(hsa.hsa_memory_allocate(kernarg_region, args_sz, ctypes.byref(x))))
  ctypes.memmove(kernargs, extra[1], args_sz)

  cmd = ExecCommand(prg, global_size, local_size, kernargs)
  dev.hsa_queue.submit([cmd])

  # dev.synchronize() # for debug
