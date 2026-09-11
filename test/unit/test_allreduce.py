import unittest
from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops
from test.helpers import KernelCountException
from tinygrad.engine.realize import run_linear

class TestRingAllReduce(unittest.TestCase):
  def test_schedule_ring(self):
    with Context(RING=2):
      N = 4
      ds = tuple(f"CPU:{i}" for i in range(N))
      t = Tensor.empty(N, N*100).shard(ds, axis=0).realize()
      linear = t.sum(0).linear_with_vars()[0]
      copies = [si for si in linear.src if si.src[0].op is Ops.COPY]
      pairs = [(c.src[1].buffer.device, c.src[2].buffer.device) for c in copies]
      # N*(N-1) scatter reduce, and N*(N-1) allgather
      if len(pairs) != N*(N-1)*2: raise KernelCountException(N*(N-1)*2, len(pairs))
      # copy topology forms a ring
      self.assertEqual(len(set(pairs)), N)

  def test_schedule_two_nodes(self): # reduce-scatter in each node, one exchange per counterpart pair, all-gather in each node
    from unittest.mock import patch
    from tinygrad.device import Device
    N, M = 4, 1007 # a shape no other test schedules on these devices: the rewrite of the same allreduce uop is cached
    ds = tuple(f"CPU:{i}" for i in range(N))
    node = property(lambda self: "a" if int(self.device.split(":")[1] if ":" in self.device else 0) < N // 2 else "b")
    with Context(ALL2ALL=2), patch.object(type(Device["CPU"]), "peer_group", node):
      x = Tensor.arange(N*M, dtype=dtypes.float).reshape(N, M)
      t = (x*x).clone().shard(ds, axis=0).realize()
      linear, var_vals = t.sum(0).contiguous().linear_with_vars()
      pairs = [(c.src[1].buffer.device, c.src[2].buffer.device) for c in linear.src if c.src[0].op is Ops.COPY]
      across = [p for p in pairs if Device[p[0]].peer_group != Device[p[1]].peer_group]
      expect = (("CPU:0", "CPU:2"), ("CPU:2", "CPU:0"), ("CPU:1", "CPU:3"), ("CPU:3", "CPU:1"))
      self.assertEqual(sorted(across), sorted((Device.canonicalize(a), Device.canonicalize(b)) for a, b in expect))
      run_linear(linear, var_vals)
    import numpy as np
    np.testing.assert_allclose(t.sum(0).numpy(), (x*x).sum(0).numpy())

  def test_schedule_all2all(self):
    with Context(ALL2ALL=2):
      N = 4
      M = N*100
      ds = tuple(f"CPU:{i}" for i in range(N))
      x = Tensor.arange(N*M, dtype=dtypes.float).reshape(N, M)
      t = (x*x).clone().shard(ds, axis=0).realize()
      out = t.sum(0).mul(2.).contiguous()
      linear, var_vals = out.linear_with_vars()
      copies = [si for si in linear.src if si.src[0].op is Ops.COPY]
      sinks = [si for si in linear.src if si.src[0].op is Ops.SINK]
      # N*(N-1) copies for input and output
      copy_count = N*(N-1)*2
      if len(copies) != copy_count: raise KernelCountException(copy_count, len(copies))
      # N*(N-1) shrinks from other devices becoming contigs, N ALU, N extra contig, reassembly (cat), and mul
      sink_count = (N*(N-1))+(N)+(N)+(1)+(1)
      if len(sinks) != sink_count: raise KernelCountException(sink_count, len(sinks))
      # correctness
      run_linear(linear, var_vals)
      expected = [2*sum((d*M+i)**2 for d in range(N)) for i in range(M)]
      dev_nums = Tensor.arange(1, N+1, dtype=dtypes.float).reshape(N, 1).expand(N, M).shard(ds, axis=0)
      shards = out.reshape(1, M).expand(N, M)+dev_nums
      self.assertListEqual(shards.tolist(), [[x+d+1 for x in expected] for d in range(N)])

  @Context(RING=0, ALL2ALL=0)
  def test_schedule_naive(self):
    N = 4
    ds = tuple(f"NULL:{i}" for i in range(N))
    t = Tensor.empty(N, 4096).shard(ds, axis=0).realize()
    linear = t.sum(0).linear_with_vars()[0]

    copies = [si for si in linear.src if si.src[0].op is Ops.COPY]
    sinks = [si for si in linear.src if si.src[0].op is Ops.SINK]
    pairs = [(c.src[1].buffer.device, c.src[2].buffer.device) for c in copies]

    if len(pairs) != N*(N-1): raise KernelCountException(N*(N-1), len(pairs))
    if len(sinks) != 2: raise KernelCountException(2, len(sinks))
    self.assertTrue(all(dst != src for dst, src in pairs))

  def test_symbolic_shape(self):
    rows = UOp.variable("rows", 1, 4).bind(3)
    t = Tensor.ones(4, 4).shard(("CPU:0", "CPU:1"), axis=1).realize()
    out = t[:rows].sum(1).realize()
    self.assertEqual(out.shape, (rows,))
    self.assertTrue((out == 4).all().item())

  def test_correct_ring(self):
    with Context(RING=2):
      N = 4
      ds = tuple(f"CPU:{i}" for i in range(N))
      t = Tensor.ones(N, N*100).contiguous().shard(ds, axis=0).realize()
      out = t.sum(0)
      self.assertListEqual(out.tolist(), [4]*N*100)

class TestAllreduceCast(unittest.TestCase):
  def _get_copy_dtypes(self, dtype, allreduce_cast):
    ds = tuple(f"CPU:{i}" for i in range(2))
    with Context(ALLREDUCE_CAST=allreduce_cast, RING=0, SCACHE=0):
      t = Tensor.empty(4, 4, dtype=dtype).shard(ds, axis=0)
      linear = t.sum(0).linear_with_vars()[0]
      return {si.src[1].buffer.dtype for si in linear.src if si.src[0].op is Ops.COPY}

  def test_allreduce_cast_bf16(self):
    # with ALLREDUCE_CAST, allreduce copies stay in bfloat16 instead of promoting to float32
    self.assertNotIn(dtypes.float, self._get_copy_dtypes(dtypes.bfloat16, allreduce_cast=1))
    self.assertIn(dtypes.float, self._get_copy_dtypes(dtypes.bfloat16, allreduce_cast=0))

  def test_allreduce_cast_half(self):
    self.assertNotIn(dtypes.float, self._get_copy_dtypes(dtypes.half, allreduce_cast=1))
    self.assertIn(dtypes.float, self._get_copy_dtypes(dtypes.half, allreduce_cast=0))

  def test_allreduce_cast_float32_noop(self):
    # float32 should not be affected by ALLREDUCE_CAST (no promotion happens)
    dtypes_on = self._get_copy_dtypes(dtypes.float, allreduce_cast=1)
    dtypes_off = self._get_copy_dtypes(dtypes.float, allreduce_cast=0)
    self.assertEqual(dtypes_on, dtypes_off)

if __name__ == '__main__':
  unittest.main()
