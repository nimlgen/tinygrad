#!/usr/bin/env python
import unittest, gc
import numpy as np
from weakref import ref
from tinygrad.tensor import Tensor
from tinygrad.state import get_parameters, get_state_dict
from tinygrad.ops import GlobalCounters, LazyOp, LoadOps
from tinygrad.runtime.lib import RawBuffer, LRUAllocator
from tinygrad.helpers import CI, dtypes, prod
from tinygrad.lazy import Device

from examples.llama import Transformer

ALLOCATED_DEV_BUFS = 0
ALIVE_DEV_BUFS = 0

class FakeDeviceBuffer():
  def __init__(self, sz, dt, device):
    self.id = 1
    self.size = sz
    self.dtype = dt
    self.device = device

    global ALIVE_DEV_BUFS, ALLOCATED_DEV_BUFS
    ALIVE_DEV_BUFS += 1
    ALLOCATED_DEV_BUFS += 1
  def __del__(self):
    global ALIVE_DEV_BUFS
    ALIVE_DEV_BUFS -= 1

class FakeAllocator(LRUAllocator):
  def _do_alloc(self, size, dtype, device, **kwargs): return FakeDeviceBuffer(size, dtype, device)
  def _do_free(self, buf):
    buf.id -= 1
    assert buf.id == 0, f"Free should be called once, but {buf.id}"

FAKE_GLOBAL_ALLOCATOR = None
class FakeBuffer(RawBuffer):
  def __init__(self, size, dtype, device='0'):
    global FAKE_GLOBAL_ALLOCATOR
    super().__init__(size, dtype, allocator=FAKE_GLOBAL_ALLOCATOR, **{'device': device})
    assert self._buf.size == size and self._buf.dtype == dtype and self._buf.device == device, "This allocator requires 100% match of dtype and size."
  @classmethod
  def fromCPU(cls, x:np.ndarray, **kwargs): return cls(prod(x.shape), dtypes.from_np(x.dtype), **kwargs)
  def toCPU(self): return np.empty(self.size, dtype=self.dtype.np)
class FakeProgram:
  def __init__(self, name:str, prg:str): pass
  def __call__(self, global_size, local_size, *bufs, wait=False): pass

def alloc(allocator, size, dtype, **kwargs):
  global FAKE_GLOBAL_ALLOCATOR
  FAKE_GLOBAL_ALLOCATOR = allocator
  buf = FakeBuffer(size, dtype, **kwargs)
  assert buf.dtype == dtype and buf.size == size
  FAKE_GLOBAL_ALLOCATOR = None
  return buf

def alloc_free_trace(allocator, size, dtype, **kwargs):
  buf = alloc(allocator, size, dtype, **kwargs)
  return ref(buf._buf)

def cmp_trace_and_buf(buf, trace_ref): return trace_ref and trace_ref() == buf._buf

def helper_test_correctness(gen, train):
  from tinygrad.runtime.ops_gpu import CL, CLAllocator
  old_alloc = CL.cl_allocator
  CL.cl_allocator = CLAllocator(0)
  no_alloc_result = train(*gen()).numpy()
  Device[Device.DEFAULT].synchronize()
  CL.cl_allocator = CLAllocator(512<<30) # Test cache correctness, so cache as much as possible, 512gb
  for _ in range(4):
    GlobalCounters.reset()
    np.testing.assert_allclose(train(*gen()).numpy(), no_alloc_result, rtol=1e-3, atol=1e-5)
    Device[Device.DEFAULT].synchronize()
  assert len(CL.cl_allocator.cached_buffers) != 0, "Cache must be used"
  CL.cl_allocator = old_alloc

def __helper_test_alloc_count(gen, train):
  was_alloc = ALLOCATED_DEV_BUFS
  for _ in range(2):
    train(*gen())
  return ALLOCATED_DEV_BUFS - was_alloc

def helper_test_alloc_count(mm, gen, train):
  global FAKE_GLOBAL_ALLOCATOR
  backup_program = Device[Device.DEFAULT].runtime
  backup_buffer = Device[Device.DEFAULT].buffer
  Device[Device.DEFAULT].runtime = FakeProgram
  Device[Device.DEFAULT].buffer = FakeBuffer
  Device[Device.DEFAULT].method_cache.clear()
  FAKE_GLOBAL_ALLOCATOR = FakeAllocator(16<<30)
  new_allocs = __helper_test_alloc_count(gen, train)
  Device[Device.DEFAULT].method_cache.clear()
  FAKE_GLOBAL_ALLOCATOR = FakeAllocator(0)
  old_allocs = __helper_test_alloc_count(gen, train)
  print(f"{mm}: llama: old allocs count {old_allocs}, new allocs count {new_allocs}")
  assert new_allocs < old_allocs, f"Hmm, doesn't cache work any more?"
  Device[Device.DEFAULT].runtime = backup_program
  Device[Device.DEFAULT].buffer = backup_buffer
  FAKE_GLOBAL_ALLOCATOR = None

def check_gc():
  if Device.DEFAULT == "GPU":
    from extra.introspection import print_objects
    assert print_objects() == 0

# for speed
def derandomize(x):
  if isinstance(x, LazyOp):
    if x.op == LoadOps.RAND: x.op = LoadOps.EMPTY
    x.src = [derandomize(s) for s in x.src]
  else:
    x.op = derandomize(x.op)
  return x

def derandomize_model(model):
  for p in get_parameters(model):
    p.lazydata = derandomize(p.lazydata)
    p.realize()

class TestAllocators(unittest.TestCase):
  def test_lru_allocator_reusage(self):
    assert ALIVE_DEV_BUFS == 0
    lru_allocator = FakeAllocator(2048)
    traced_buf = alloc_free_trace(lru_allocator, 16, dtypes.float32)
    assert GlobalCounters.mem_cached == 16*dtypes.float32.itemsize, "Buffer should be cached"
    for _ in range(32):
      def __test():
        buf = alloc(lru_allocator, 16, dtypes.float32)
        assert cmp_trace_and_buf(buf, traced_buf), "Buffer should be reused"
      __test()

    usedbuf = alloc(lru_allocator, 16, dtypes.float32)
    for _ in range(32):
      def __test():
        buf = alloc(lru_allocator, 16, dtypes.float32)
        assert usedbuf != buf, "Nobody should get used buffer"
      __test()
    assert GlobalCounters.mem_used == 16*dtypes.float32.itemsize, "Only usedbuf is still allocated."

  def test_lru_allocator_cache_free(self):
    assert ALIVE_DEV_BUFS == 0
    lru_allocator = FakeAllocator(128)
    refs = []
    for _ in range(32):
      refs.append(alloc_free_trace(lru_allocator, 16, dtypes.float32))
    for sz in range(32):
      alloc_free_trace(lru_allocator, sz, dtypes.float32)
      assert GlobalCounters.mem_used + GlobalCounters.mem_cached <= 128, "Should not allocate on device more than allowed (128)"
    for r in refs: assert r() is None, "All refs should be dead, since buffers were cleared from cache"

  def test_lru_allocator_multidevice(self):
    assert ALIVE_DEV_BUFS == 0
    lru_allocator = FakeAllocator(256)
    refs=[]
    for i in range(8):
      refs.append(alloc_free_trace(lru_allocator, 16, dtypes.float32, device=str(i)))
    for i in range(64):
      def __test():
        dev = str(i % 8)
        buf = alloc(lru_allocator, 16, dtypes.float32, device=dev)
        assert cmp_trace_and_buf(buf, refs[i%8]), "Buffer should be reused"
      __test()
    for r in refs: assert r() is not None, "All refs should be cached"

  @unittest.skip("huge for CI")
  def test_lru_allocator_tiny_llama(self):
    old_type = Tensor.default_type
    Tensor.default_type = dtypes.float16

    args_tiny = {"dim": 1024, "multiple_of": 256, "n_heads": 8, "n_layers": 8, "norm_eps": 1e-05, "vocab_size": 1000}
    def __test():
      model = Transformer(**args_tiny)
      derandomize_model(model)
      def test(t): return model(t, 0).realize()
      helper_test_correctness(lambda: (Tensor([[1,]]),), test)
    __test()
    Tensor.default_type = old_type
    gc.collect() # Need to collect Tensors.
    check_gc()

  @unittest.skip("huge for CI")
  def test_lru_allocator_tiny_llama_alloc_counts(self):
    args_tiny = {"dim": 1024, "multiple_of": 256, "n_heads": 8, "n_layers": 8, "norm_eps": 1e-05, "vocab_size": 1000}
    def test_alloc_count(t):
      model = Transformer(**args_tiny)
      for v in get_state_dict(model).values(): v.assign(Tensor.empty(*v.shape, dtype=v.dtype))
      return model(t, 0).realize()
    helper_test_alloc_count("llama", lambda: (Tensor([[2,]]),), test_alloc_count)
    gc.collect() # Need to collect Tensors.
    check_gc()

  @unittest.skip("huge for CI")
  def test_stable_diffusion(self):
    from examples.stable_diffusion import UNetModel
    model = UNetModel()
    derandomize_model(model)
    def test(t, t2): return model(t, 801, t2).realize()
    helper_test_correctness(lambda: (Tensor.randn(1, 4, 16, 16),Tensor.randn(1, 77, 768)), test)

if __name__ == "__main__":
  unittest.main()
