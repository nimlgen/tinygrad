from __future__ import annotations
from typing import Callable, List, Tuple, Dict, cast, Union, Optional, TypeVar, Generic
import functools, itertools, operator
from tinygrad.nn.state import get_parameters
from tinygrad.dtype import DType
from tinygrad.helpers import DEBUG, merge_dicts, getenv, all_int, Context, GRAPH, flatten, GraphException
from tinygrad.device import Compiled, JITRunner, CompiledASTRunner, Buffer, BufferXfer, MultiDeviceJITGraph
from tinygrad.tensor import Tensor
from tinygrad.lazy import LazyBuffer
from tinygrad.features.multi import MultiLazyBuffer
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable, sint
from weakref import ref, WeakKeyDictionary
from dataclasses import dataclass

@dataclass(frozen=True)
class JitItem:
  prg: JITRunner  # or a graph executor like MetalGraph
  rawbufs: List[Optional[Buffer]]

def get_jit_stats(jit_cache: List[JitItem]) -> Tuple[sint, int]:
  return functools.reduce(operator.add, [ji.prg.op_estimate for ji in jit_cache if isinstance(ji.prg, CompiledASTRunner)], 0), \
         functools.reduce(operator.add, [ji.prg.mem_estimate for ji in jit_cache if isinstance(ji.prg, CompiledASTRunner)], 0)
def get_input_replace(jit_cache: List[JitItem], input_rawbuffers:List[Buffer]) -> Dict[Tuple[int, int], int]:
  input_replace: Dict[Tuple[int, int], int] = {}
  for j,ji in enumerate(jit_cache):
    for i,a in enumerate(ji.rawbufs):
      if a in input_rawbuffers:
        input_replace[(j,i)] = input_rawbuffers.index(a)
  return input_replace
def get_jc_idxs_with_updatable_launch_dims(jit_cache: List[JitItem]) -> List[int]:
  return [j for j,ji in enumerate(jit_cache) if isinstance(ji.prg, CompiledASTRunner) and ((ji.prg.global_size and not all_int(ji.prg.global_size)) or (ji.prg.local_size and not all_int(ji.prg.local_size)))]  # noqa: E501
def get_jc_idxs_with_updatable_var_vals(jit_cache: List[JitItem]) -> List[int]:
  return [j for j,ji in enumerate(jit_cache) if isinstance(ji.prg, CompiledASTRunner) and ji.prg.vars]

import collections, heapq

def optimize_jc(jit_cache):
  ji_prio = {}
  ji_deps = {}

  for j,ji in enumerate(jit_cache):
    ji_prio[j] = -1 if isinstance(ji.prg, BufferXfer) else 0
    ji_deps[j] = set()

  last_w_access = {}
  last_r_access = collections.defaultdict(list)
  for j,ji in enumerate(jit_cache):
    if isinstance(ji.prg, CompiledASTRunner):
      ji_prio[j] = -1

      for buf in ji.rawbufs:
        if buf._buf in last_w_access:
          dep_ji, dep_idx = jit_cache[last_w_access[buf._buf]], last_w_access[buf._buf]
          if isinstance(dep_ji.prg, BufferXfer): ji_prio[j] += 1 # bigger prio, should be at the end
          ji_deps[j].add(dep_idx)
      if ji.rawbufs[0]._buf in last_r_access:
        dep_ids = last_r_access.pop(ji.rawbufs[0]._buf)
        for dep_idx in dep_ids:
          dep_ji = jit_cache[dep_idx]
          if isinstance(dep_ji.prg, BufferXfer): ji_prio[j] += 1 # bigger prio, should be at the end
          ji_deps[j].add(dep_idx)
      last_w_access[ji.rawbufs[0]._buf] = j
      for buf in ji.rawbufs[1:]: last_r_access[buf._buf].append(j)
    elif isinstance(ji.prg, BufferXfer):
      dest, src = ji.rawbufs[0:2]
      if src._buf in last_w_access:
        dep_ji, dep_idx = jit_cache[last_w_access[src._buf]], last_w_access[src._buf]
        ji_prio[dep_idx] -= dest.nbytes # lower prio, should be at the start to kickoff it earlier
        ji_deps[j].add(dep_idx)

      if dest._buf in last_w_access:
        dep_ji, dep_idx = jit_cache[last_w_access[dest._buf]], last_w_access[dest._buf]
        ji_prio[dep_idx] -= dest.nbytes # lower prio, should be at the start to kickoff it earlier
        ji_deps[j].add(dep_idx)

      if dest._buf in last_r_access:
        dep_ids = last_r_access.pop(dest._buf)
        for dep_idx in dep_ids:
          dep_ji = jit_cache[dep_idx]
          ji_prio[dep_idx] -= dest.nbytes # lower prio, should be at the start to kickoff it earlier
          ji_deps[j].add(dep_idx)
      last_r_access[src._buf].append(j)
      last_w_access[dest._buf] = j
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
  for j in range(len(jit_cache)): ji_deps_rev[j] = []
  for k, v in ji_deps.items():
    for x in v: ji_deps_rev[x].append(k)
  order = toposort(ji_deps_rev, ji_prio.get)
  assert len(order) == len(jit_cache)

  new_jit_cache = []
  for i in range(len(jit_cache)): new_jit_cache.append(jit_cache[order[i]])
  return new_jit_cache

def apply_graph_to_jit(jit_cache: List[JitItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]) -> List[JitItem]:
  # Split JIT cache into batches for faster graph execution.
  # This allows the accelerator to run some batches while subsequent graphs are still being updated.
  jit_cache = optimize_jc(jit_cache)

  max_batch_size = getenv("JIT_BATCH_SIZE", 16)
  graphed_jit_cache: List[JitItem] = []
  current_batch: List[JitItem] = []
  current_device: Optional[Compiled] = None

  def flush_batch():
    nonlocal current_batch, current_device, max_batch_size
    try:
      if len(current_batch) <= 1 or current_device is None: raise GraphException("only one kernel doesn't graph")
      graphed_jit_cache.append(JitItem(current_device.graph(current_batch, input_rawbuffers, var_vals), cast(List[Optional[Buffer]], input_rawbuffers))) # noqa: E501
      max_batch_size = max_batch_size * 2
      if DEBUG >= 2: print(f"\tJIT GRAPHing batch with {len(current_batch)} kernels on device {current_device}")
    except GraphException as e:
      graphed_jit_cache.extend(current_batch)
      if DEBUG >= 2: print(f"\tJIT GRAPHing failed batch with {len(current_batch)} kernels on device {current_device}: {e}")
    current_batch = []
    current_device = None

  for ji in jit_cache:
    ji_graph_dev: Optional[Compiled] = None # device on which the ji will be graphed. Not graphed if None.
    if isinstance(ji.prg, CompiledASTRunner): ji_graph_dev = ji.prg.device
    elif isinstance(ji.prg, BufferXfer) and ji.rawbufs[0] and ji.rawbufs[0].d.dname.startswith("HSA"): ji_graph_dev = ji.rawbufs[0].d

    can_be_graphed = ji_graph_dev and ji_graph_dev.graph
    can_extend_graph_batch = can_be_graphed and len(current_batch) < max_batch_size and (ji_graph_dev == current_device or
      (isinstance(ji_graph_dev.graph, type) and issubclass(ji_graph_dev.graph, MultiDeviceJITGraph) and type(ji_graph_dev) == type(current_device))) #type:ignore
    if not can_extend_graph_batch and len(current_batch) > 0: flush_batch()

    if can_be_graphed: current_batch.append(ji)
    else: graphed_jit_cache.append(ji)

    current_device = ji_graph_dev

  if len(current_batch) > 0: flush_batch()
  return graphed_jit_cache

# *** JIT ***

ReturnType = TypeVar('ReturnType')
class TinyJit(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]):
    self.fxn = fxn
    self.reset()

  def reset(self):
    self.jit_cache: List[JitItem] = []
    self.input_replace: Dict[Tuple[int, int], int] = {}
    self.cnt: int = 0
    self.ret: Optional[ReturnType] = None
    self.expected_vals: Optional[Tuple[Variable, ...]] = None
    self.expected_name_sts_dtype_device: Optional[Tuple[Tuple[Union[int, str], ShapeTracker, DType, Union[str, Tuple[str, ...]]], ...]] = None

  # add support for instance methods
  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj)

  def __call__(self, *args, **kwargs) -> ReturnType:
    # all inputs (except const) are realized
    input_tensors: Dict[Union[int, str], Union[LazyBuffer, MultiLazyBuffer]] = { cast(Union[int, str], k):v.realize().lazydata for k,v in itertools.chain(enumerate(args), kwargs.items()) if v.__class__ is Tensor }  # noqa: E501
    expected_name_sts_dtype_device = tuple([(k, v.st.unbind()[0] if isinstance(v, LazyBuffer) else ShapeTracker.from_shape(v.shape), v.dtype, v.device) for k,v in input_tensors.items()]) #noqa: E501

    # get rawbuffers
    lbs: List[LazyBuffer] = [v for v in input_tensors.values() if isinstance(v, LazyBuffer)] + flatten([mlb.lbs for mlb in input_tensors.values() if isinstance(mlb, MultiLazyBuffer)]) #noqa: E501
    input_rawbuffers: List[Buffer] = [v.base.realized for v in lbs if v.base.realized is not None]
    assert len(set(input_rawbuffers)) == len(input_rawbuffers), "duplicate inputs to JIT"

    # get variables: they can either be in Tensors or passed in as arguments, and all must be bound. these are all global
    var_vals: Dict[Variable, int] = merge_dicts([arg.st.var_vals for arg in lbs] + [dict(x.unbind() for x in itertools.chain(args, kwargs.values()) if isinstance(x, Variable))])  # noqa: E501
    expected_vals = tuple(var_vals.keys())

    if self.cnt >= 2:
      # jit exec
      assert self.expected_vals == expected_vals and self.expected_name_sts_dtype_device is not None, "missing/mismatch of var_vals"
      assert all(x[0] == y[0] and x[1].views == y[1].views and x[2] == y[2] and x[3] == y[3]
                 for x,y in zip(self.expected_name_sts_dtype_device, expected_name_sts_dtype_device)), \
        f"mismatch of input tensors, expected {self.expected_name_sts_dtype_device} got {expected_name_sts_dtype_device}"
      for (j,i),input_idx in self.input_replace.items(): self.jit_cache[j].rawbufs[i] = input_rawbuffers[input_idx]
      for ji in self.jit_cache: ji.prg(cast(List[Buffer], ji.rawbufs), var_vals, wait=DEBUG>=2, jit=True)
    elif self.cnt == 1:
      # jit capture
      self.expected_vals, self.expected_name_sts_dtype_device = expected_vals, expected_name_sts_dtype_device
      CacheCollector.start(var_vals)
      with Context(GRAPH=getenv("JITGRAPH", GRAPH.value)):
        self.ret = self.fxn(*args, **kwargs)
        for p in get_parameters(self.ret): p.realize()
      self.jit_cache = CacheCollector.finish()
      assert len(self.jit_cache) != 0, "didn't JIT anything!"
      if DEBUG >= 1 and len(set(get_input_replace(self.jit_cache, input_rawbuffers).values())) != len(input_rawbuffers):
        print("WARNING: some input tensors not found")
      if DEBUG >= 1: print(f"JIT captured {len(self.jit_cache)} kernels with {len(input_rawbuffers)} inputs")

      # Condense the items into a graph executor.
      if getenv("JIT") != 2: self.jit_cache = apply_graph_to_jit(self.jit_cache, input_rawbuffers, var_vals)

      self.input_replace = get_input_replace(self.jit_cache, input_rawbuffers)
    elif self.cnt == 0:
      # jit ignore
      self.ret = self.fxn(*args, **kwargs)
      for p in get_parameters(self.ret): p.realize()

    # clear jit inputs
    for (j,i) in self.input_replace.keys(): self.jit_cache[j].rawbufs[i] = None

    self.cnt += 1
    return cast(ReturnType, self.ret)

class PlaceHolder:
  def __init__(self, buf:Buffer):
    self.size, self.dtype, self.device, self.ref, self.bufid, self.options = buf.size, buf.dtype, buf.device, ref(buf), id(buf._buf), buf.options
  def to_tuple(self): return (self.size, self.dtype, self.device, self.bufid, self.options)
  def __hash__(self): return hash(self.to_tuple())
  def __eq__(self, x): return isinstance(x, PlaceHolder) and self.to_tuple() == x.to_tuple()
  def alloc_if_needed(self, buffer_cache: Dict[PlaceHolder, Buffer]) -> Buffer:
    ret = self.ref()
    if ret: return ret
    if self not in buffer_cache: buffer_cache[self] = Buffer(self.device, self.size, self.dtype, options=self.options)
    return buffer_cache[self]

class _CacheCollector:
  def __init__(self):
    self.cache: Optional[List[Tuple[JITRunner, List[Union[Buffer, PlaceHolder]]]]] = None

  def start(self, var_vals:Optional[Dict[Variable, int]]=None):
    self.cache = []
    self.placeholders: WeakKeyDictionary[Buffer, PlaceHolder] = WeakKeyDictionary()
    self.var_vals = var_vals if var_vals is not None else {}

  def add(self, prg, rawbufs, var_vals):
    if self.cache is None: return
    for k,v in var_vals.items(): assert k in self.var_vals and self.var_vals[k] == v, f"var_vals {k} mismatch {v} != {self.var_vals.get(k)}"

    # Buffer optimization is allowed only for kernel operations. Avoids for copies (prevents parallelism) and syncs (incorrect buffer reuse).
    allow_buffer_optimization = isinstance(prg, CompiledASTRunner)

    # NOTE: this is making an assumption that 0 is special
    if len(rawbufs): self.placeholders[rawbufs[0]] = PlaceHolder(rawbufs[0])
    self.cache.append((prg, [self.placeholders.get(x, x) if isinstance(x, Buffer) and allow_buffer_optimization else x for x in rawbufs]))

  def finish(self) -> List[JitItem]:
    if self.cache is None: return []
    buffer_cache: Dict[PlaceHolder, Buffer] = {}
    saved_cache, self.cache = self.cache, None
    return [JitItem(prg, [x.alloc_if_needed(buffer_cache) if isinstance(x, PlaceHolder) else x for x in pl]) for prg, pl in saved_cache]
CacheCollector = _CacheCollector()
