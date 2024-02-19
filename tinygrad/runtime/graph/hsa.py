from typing import List, Any, Dict, cast, Optional, Set
import ctypes, collections, time, heapq
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv, GraphException
from tinygrad.device import Buffer, CompiledASTRunner, BufferXfer, update_stats
from tinygrad.shape.symbolic import Variable
from tinygrad.runtime.ops_hsa import HSADevice
from tinygrad.features.jit import JitItem, get_input_replace, get_jit_stats, \
                                  get_jc_idxs_with_updatable_launch_dims, get_jc_idxs_with_updatable_var_vals

import tinygrad.runtime.autogen.hsa as hsa
from tinygrad.runtime.driver.hsa import *

class HSAGraph:
  def __init__(self, jit_cache: List[JitItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    self.jit_cache = jit_cache

    # Optimize jit cache for more optimial kernel execution
    if getenv("HSA_OPT_JC", 0): self.optimize_jc()

    self.input_replace = get_input_replace(self.jit_cache, input_rawbuffers)
    self.op_estimate, self.mem_estimate = get_jit_stats(self.jit_cache)
    self.jc_idxs_with_updatable_launch_dims = get_jc_idxs_with_updatable_launch_dims(self.jit_cache)
    self.jc_idxs_with_updatable_var_vals = get_jc_idxs_with_updatable_var_vals(self.jit_cache)
    self.devices: Set[HSADevice] = set()

    # Check all jit items are compatible.
    kernargs_size: Dict[HSADevice, int] = collections.defaultdict(int)
    for ji in self.jit_cache:
      if isinstance(ji.prg, CompiledASTRunner):
        self.devices.add(ji.prg.device)
        kernargs_size[ji.prg.device] += (ctypes.sizeof(ji.prg.clprg.args_struct_t) + 15) & ~15
      elif isinstance(ji.prg, BufferXfer):
        for x in ji.rawbufs[0:2]: self.devices.add(x.d)
      else: raise GraphException

    # Check all devices are HSA.
    if any(not isinstance(d, HSADevice) for d in self.devices): raise GraphException

    # Allocate queues
    self.packets_count = {}
    self.c_aql_packets = {}
    self.c_aql_packets_addr = {}
    self.packet_off = {}
    for dev in self.devices:
      self.packets_count[dev] = len(self.jit_cache) + 300 # TODO: fixme
      self.c_aql_packets[dev] = (hsa.hsa_kernel_dispatch_packet_t * self.packets_count[dev])()
      self.c_aql_packets_addr[dev] = ctypes.addressof(self.c_aql_packets[dev])
      self.packet_off[dev] = ctypes.addressof(self.c_aql_packets[dev])

    # Allocate kernel args.
    kernargs_ptrs: Dict[HSADevice, int] = dict()
    for dev,sz in kernargs_size.items():
      kernargs_ptrs[dev] = init_c_var(ctypes.c_void_p(), lambda x: check(hsa.hsa_amd_memory_pool_allocate(dev.kernargs_pool, sz, 0, ctypes.byref(x)))).value
      check(hsa.hsa_amd_agents_allow_access(1, ctypes.byref(dev.agent), None, kernargs_ptrs[dev]))

    # Fill initial arguments.
    self.ji_kernelargs_addr: Dict[HSADevice, Dict[int, int]] = collections.defaultdict(dict) # TODO: do not need this really...
    self.ji_kernelargs_structs: Dict[int, int] = {}
    for j,ji in enumerate(self.jit_cache):
      if not isinstance(ji.prg, CompiledASTRunner): continue
      dev = ji.prg.device
      self.ji_kernelargs_addr[j] = kernargs_ptrs[dev]
      self.ji_kernelargs_structs[j] = ji.prg.clprg.args_struct_t.from_address(self.ji_kernelargs_addr[j])
      kernargs_ptrs[dev] += (ctypes.sizeof(ji.prg.clprg.args_struct_t) + 15) & ~15
      for i in range(len(ji.rawbufs)): self.ji_kernelargs_structs[j].__setattr__(f'f{i}', ji.rawbufs[i]._buf)
      for i in range(len(ji.prg.vars)): self.ji_kernelargs_structs[j].__setattr__(f'v{i}', var_vals[ji.prg.vars[i]])

    # Build packets for hwqueues.
    self.packets = []
    self.transfers = []
    self.kickoff_signals = {dev:self.alloc_signal() for dev in self.devices}
    self.finish_signal = self.alloc_signal() # This is a special signal, we cannot run this graph instance while it's running.
    self.signals_to_reset = []
    r_resourse_to_signal = {}
    w_resourse_to_signal = {}
    signal_to_devices = {}

    # Special packet to wait for the world.
    self.signals_to_reset += list(self.kickoff_signals.values())
    for dev in self.devices: self.add_barrier_packet(dev, [], self.kickoff_signals[dev])

    copies = 0
    for j,ji in enumerate(self.jit_cache):
      if isinstance(ji.prg, CompiledASTRunner):
        # continue # ignor
        wait_signals = []
        for i,buf in enumerate(ji.rawbufs):
          sigs = []
          if i == 0 and buf._buf in r_resourse_to_signal: sigs.append(r_resourse_to_signal.pop(buf._buf))
          if buf._buf in w_resourse_to_signal: sigs.append(w_resourse_to_signal[buf._buf])
          for sig in sigs:
            if isinstance(sig, hsa.hsa_signal_t):
              wait_signals.append(sig)
            else:
              assert sig[1] == ji.prg.device, "input used on another device (not supported)"

        for i in range(0, len(wait_signals), 5):
          self.add_barrier_packet(ji.prg.device, wait_signals[i:i+5], EMPTY_SIGNAL)

        self.packets.append(self.add_exec_packet(ji.prg.device, ji, self.ji_kernelargs_addr[j], var_vals))
        for i,buf in enumerate(ji.rawbufs):
          if i == 0: w_resourse_to_signal[buf._buf] = (self.packets[-1], ji.prg.device)
          else: r_resourse_to_signal[buf._buf] = (self.packets[-1], ji.prg.device)
      elif isinstance(ji.prg, BufferXfer):
        dest, src = ji.rawbufs[0:2]
        copies += dest.nbytes
        self.packets.append(None)
        # continue
        dest_h, src_h = False, False
        wait_signals = []
        # wait_signals = [self.kickoff_signals[dest.d], self.kickoff_signals[src.d]]
        if dest._buf in w_resourse_to_signal:
          wait_signals.append(self.dependency_as_signal(w_resourse_to_signal, dest._buf, dest.d))
          w_resourse_to_signal.pop(dest._buf)
          dest_h = True
        if dest._buf in r_resourse_to_signal:
          wait_signals.append(self.dependency_as_signal(r_resourse_to_signal, dest._buf, dest.d))
          r_resourse_to_signal.pop(dest._buf)
          dest_h = True
        
        if src._buf in w_resourse_to_signal:
          wait_signals.append(self.dependency_as_signal(w_resourse_to_signal, src._buf, src.d))
          src_h = True

        if src_h == False: wait_signals.append(self.kickoff_signals[src.d])
        if dest_h == False: wait_signals.append(self.kickoff_signals[dest.d])

        # print(f"transfer {hex(src._buf)} -> {hex(dest._buf)}")

        sync_signal = self.alloc_signal()
        self.signals_to_reset.append(sync_signal)

        c_wait_signal = (hsa.hsa_signal_t * len(wait_signals))(*wait_signals)
        self.transfers.append((dest._buf, dest.d.agent, src._buf, src.d.agent, dest.nbytes, len(wait_signals),
                               c_wait_signal, sync_signal, hsa.HSA_AMD_SDMA_ENGINE_0, True))

        r_resourse_to_signal[src._buf] = sync_signal
        w_resourse_to_signal[dest._buf] = sync_signal
        signal_to_devices[sync_signal.handle] = [dest.d, src.d]
      else: assert False

    # print("COPY", copies / 1e6, "mb")

    # Signaling we have finished
    wait_signals_to_finish = collections.defaultdict(list)
    for v in list(w_resourse_to_signal.values()) + list(r_resourse_to_signal.values()):
      if not isinstance(v, hsa.hsa_signal_t): continue
      for dev in signal_to_devices[v.handle]:
        wait_signals_to_finish[dev].append(v)

    for dev in self.devices:
      wait_signals = wait_signals_to_finish[dev]
      if len(wait_signals): # TODO: remove if here
        for i in range(0, len(wait_signals), 5):
          self.add_barrier_packet(dev, wait_signals[i:i+5], self.finish_signal if i+5 >= len(wait_signals) else EMPTY_SIGNAL)
      else:
        self.add_barrier_packet(dev, [], self.finish_signal)

    for dev in self.devices: self.packets_count[dev] = (self.packet_off[dev] - self.c_aql_packets_addr[dev]) // AQL_PACKET_SIZE
    for sig in self.signals_to_reset: hsa.hsa_signal_store_relaxed(sig, 0)
    hsa.hsa_signal_store_relaxed(self.finish_signal, 0)

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False, jit=False) -> Optional[float]:    
    # Wait and restore signals
    hsa.hsa_signal_wait_scacquire(self.finish_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
    for sig in self.signals_to_reset: hsa.hsa_signal_silent_store_relaxed(sig, 1)
    hsa.hsa_signal_silent_store_relaxed(self.finish_signal, len(self.devices))

    # Update rawbuffers
    for (j,i),input_idx in self.input_replace.items():
      self.ji_kernelargs_structs[j].__setattr__(f'f{i}', input_rawbuffers[input_idx]._buf)

    # Update var_vals
    for j in self.jc_idxs_with_updatable_var_vals:
      for i,v in enumerate(cast(CompiledASTRunner, self.jit_cache[j].prg).vars):
        self.ji_kernelargs_structs[j].__setattr__(f'v{i}', var_vals[v])

    # Update launch dims
    for j in self.jc_idxs_with_updatable_launch_dims:
      gl, lc = cast(CompiledASTRunner, self.jit_cache[j].prg).launch_dims(var_vals)
      self.packets[j].workgroup_size_x = lc[0]
      self.packets[j].workgroup_size_y = lc[1]
      self.packets[j].workgroup_size_z = lc[2]
      self.packets[j].grid_size_x = gl[0] * lc[0]
      self.packets[j].grid_size_y = gl[1] * lc[1]
      self.packets[j].grid_size_z = gl[2] * lc[2]

    for dev in self.devices:
      dev.hw_queue.blit_packets(self.c_aql_packets_addr[dev], self.packets_count[dev])

    for transfer_data in self.transfers:
      check(hsa.hsa_amd_memory_async_copy_on_engine(*transfer_data)) # check(hsa.hsa_amd_memory_async_copy(*transfer_data[:-2]))

    et = None
    if wait:
      hsa.hsa_signal_wait_scacquire(self.finish_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
      check(hsa.hsa_amd_profiling_get_dispatch_time(dev.agent, self.finish_signal, ctypes.byref(timings := hsa.hsa_amd_profiling_dispatch_time_t())))
      et = (timings.end - timings.start) / 1e9

    update_stats(f"<batched {len(self.jit_cache)}>", self.op_estimate, self.mem_estimate, var_vals, et, buf_count=len(input_rawbuffers),
                 jit=jit, num_kernels=len(self.jit_cache), device="HSA")

  def add_exec_packet(self, dev, ji, args, var_vals):
    global_size, local_size = ji.prg.launch_dims(var_vals)
    packet = hsa.hsa_kernel_dispatch_packet_t.from_address(self.packet_off[dev])
    packet.workgroup_size_x = local_size[0]
    packet.workgroup_size_y = local_size[1]
    packet.workgroup_size_z = local_size[2]
    packet.reserved0 = 0
    packet.grid_size_x = global_size[0] * local_size[0]
    packet.grid_size_y = global_size[1] * local_size[1]
    packet.grid_size_z = global_size[2] * local_size[2]
    packet.private_segment_size = ji.prg.clprg.private_segment_size
    packet.group_segment_size = ji.prg.clprg.group_segment_size
    packet.kernel_object = ji.prg.clprg.handle
    packet.kernarg_address = args
    packet.reserved2 = 0
    packet.completion_signal = EMPTY_SIGNAL
    packet.setup = DISPATCH_KERNEL_SETUP
    packet.header = DISPATCH_KERNEL_HEADER
    self.packet_off[dev] += AQL_PACKET_SIZE
    return packet

  def add_barrier_packet(self, dev, wait_signals, completion_signal):
    barrier_packet = hsa.hsa_barrier_and_packet_t.from_address(self.packet_off[dev])
    barrier_packet.reserved0 = 0
    barrier_packet.reserved1 = 0
    for i in range(5):
      barrier_packet.dep_signal[i] = wait_signals[i] if wait_signals and len(wait_signals) > i else EMPTY_SIGNAL
    barrier_packet.reserved2 = 0
    barrier_packet.completion_signal = completion_signal
    barrier_packet.header = BARRIER_HEADER
    self.packet_off[dev] += AQL_PACKET_SIZE
    return barrier_packet

  def alloc_signal(self):
    check(hsa.hsa_amd_signal_create(1, 0, None, 0, ctypes.byref(signal := hsa.hsa_signal_t())))
    return signal

  def dependency_as_signal(self, rs, dep, dep_dev):
    if isinstance(rs[dep], hsa.hsa_signal_t):
      return rs[dep]
    else:
      packet, packet_dev = rs[dep]
      assert packet_dev == dep_dev, "transfer packet used on another device last time"
      if packet.completion_signal.handle == EMPTY_SIGNAL.handle:
        packet.completion_signal = self.alloc_signal()
        self.signals_to_reset.append(packet.completion_signal)
      return packet.completion_signal

  def optimize_jc(self):
    self.ji_prio = {}
    self.ji_deps = {}

    for j,ji in enumerate(self.jit_cache):
      self.ji_prio[j] = -1 if isinstance(ji.prg, BufferXfer) else 0
      self.ji_deps[j] = set()

    self.last_w_access = {}
    self.last_r_access = {}
    for j,ji in enumerate(self.jit_cache):
      if isinstance(ji.prg, CompiledASTRunner):
        self.ji_prio[j] = -1
        for buf in ji.rawbufs:
          if buf._buf in self.last_w_access:
            dep_ji, dep_idx = self.jit_cache[self.last_w_access[buf._buf]], self.last_w_access[buf._buf]
            if isinstance(dep_ji.prg, BufferXfer): self.ji_prio[j] += 1 # bigger prio, should be at the end
            self.ji_deps[j].add(dep_idx)
        if ji.rawbufs[0]._buf in self.last_r_access:
          dep_ji, dep_idx = self.jit_cache[self.last_r_access[ji.rawbufs[0]._buf]], self.last_r_access[ji.rawbufs[0]._buf]
          if isinstance(dep_ji.prg, BufferXfer): self.ji_prio[j] += 1 # bigger prio, should be at the end
          self.ji_deps[j].add(dep_idx)

        self.last_w_access[ji.rawbufs[0]._buf] = j
        for buf in ji.rawbufs[1:]: self.last_r_access[buf._buf] = j
      elif isinstance(ji.prg, BufferXfer):
        dest, src = ji.rawbufs[0:2]
        if src._buf in self.last_w_access:
          dep_ji, dep_idx = self.jit_cache[self.last_w_access[src._buf]], self.last_w_access[src._buf]
          self.ji_prio[dep_idx] -= ji.rawbufs[0].nbytes # lower prio, should be at the start to kickoff it earlier
          self.ji_deps[j].add(dep_idx)

        if dest._buf in self.last_w_access:
          dep_ji, dep_idx = self.jit_cache[self.last_w_access[dest._buf]], self.last_w_access[dest._buf]
          self.ji_prio[dep_idx] -= ji.rawbufs[0].nbytes # lower prio, should be at the start to kickoff it earlier
          self.ji_deps[j].add(dep_idx)

        if dest._buf in self.last_r_access:
          dep_ji, dep_idx = self.jit_cache[self.last_r_access[dest._buf]], self.last_r_access[dest._buf]
          self.ji_prio[dep_idx] -= ji.rawbufs[0].nbytes # lower prio, should be at the start to kickoff it earlier
          self.ji_deps[j].add(dep_idx)

        self.last_r_access[src._buf] = j
        self.last_w_access[dest._buf] = j
      else: assert False
  
    def toposort(graph, key=None):
      # init the indegree for each node
      nodes = graph.keys() | set([node for adjacents in graph.values() for node in adjacents])
      in_degree = {node: 0 for node in nodes}

      # compute the indegree
      for k, adjacents in graph.items():
        for node in adjacents:
          in_degree[node] += 1

      # init the heap with the nodes with indegree 0 and priority given by key
      heap = [(key(node), node) for node, degree in in_degree.items() if degree == 0]
      heapq.heapify(heap)

      top_order = []
      while heap:  # heap is not empty
        _, node = heapq.heappop(heap)  # get the element with highest priority and remove from heap
        top_order.append(node)  # add to topological order
        for adjacent in graph.get(node, []):  # iter over the neighbors of the node
          in_degree[adjacent] -= 1
          if in_degree[adjacent] == 0:  # if the node has in_degree 0 add to the heap with priority given by key
            heapq.heappush(heap, (key(adjacent), adjacent))

      return top_order
    
    ji_deps_rev = {}
    for j in range(len(self.jit_cache)): ji_deps_rev[j] = []
    for k, v in self.ji_deps.items():
      for x in v: ji_deps_rev[x].append(k)
    order = toposort(ji_deps_rev, self.ji_prio.get)
    assert len(order) == len(self.jit_cache)

    new_jit_cache = []
    for i in range(len(self.jit_cache)): new_jit_cache.append(self.jit_cache[order[i]])
    self.jit_cache = new_jit_cache
