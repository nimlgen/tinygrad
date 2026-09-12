import unittest
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.helpers import getenv
from examples.mlperf.models.flat_llama import apply_grad
from examples.mlperf.optim import clip_grads

@unittest.skipUnless(getenv("RDMA"), "requires two eight-GPU AMD nodes")
class TestRDMATrainingScale(unittest.TestCase):
  def test_fixed_global_batch_gradients(self):
    devices = tuple(f"AMD:{i}" for i in range(16))
    self.assertNotEqual(Device[devices[0]].peer_group, Device[devices[8]].peer_group)
    initial = np.random.default_rng(123).normal(0, 0.1, (32, 16)).astype(np.float32)
    # Same global batch: eight GPUs accumulate two minibatches; sixteen GPUs use one.
    for devs, accumulation in ((devices[8:], 2), (devices, 1)):
      weight = Tensor(initial.copy(), dtype=dtypes.float32).shard(devs).realize()
      grad = Tensor.zeros(*initial.shape, dtype=dtypes.float32).shard(devs).contiguous().realize()
      coeff = Tensor.empty(1, dtype=dtypes.float32, device=devs).realize()

      @TinyJit
      def step(xs):
        for i, x in enumerate(xs):
          loss = (x @ weight).square().mean()
          apply_grad(grad, loss.gradient(weight)[0].uop, accumulate=i != 0)
          grad.realize()
        norm, scale = clip_grads([grad], accumulation, 1.0, coeff)
        return norm.realize(scale)

      for iteration in range(5):
        x = np.random.default_rng(1000+iteration).normal(size=(32, 32)).astype(np.float32)
        xs = [Tensor(a.copy()).shard(devs, axis=0).realize() for a in np.split(x, accumulation)]
        norm = step(xs)
        expected = 2 * x.T @ (x @ initial) / (32*16)
        for rank in range(len(devs)):
          np.testing.assert_allclose(Tensor(grad.uop.mselect(rank)).numpy()/accumulation, expected, rtol=3e-4, atol=3e-5)
        np.testing.assert_allclose(norm.numpy(), np.linalg.norm(expected), rtol=3e-4, atol=3e-5)
      print(f"verified DP={len(devs)} accumulation={accumulation} GBS=32: all replicas and global gradient norm")

if __name__ == "__main__": unittest.main()
