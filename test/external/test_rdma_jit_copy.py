import os, time, unittest
import numpy as np
from unittest.mock import patch
from tinygrad.runtime.ops_cpu import CPUProgram
from tinygrad import Device, Tensor, TinyJit
from tinygrad.helpers import getenv

@unittest.skipUnless(getenv("RDMA"), "requires two AMD nodes with BNXT NICs and RDMA=1")
class TestRDMACopy(unittest.TestCase):
  def setUp(self):
    self.devs = os.environ.get("RDMA_DEVS", "AMD,AMD:6").split(",")
    assert Device[self.devs[0]].peer_group != Device[self.devs[1]].peer_group

  def test_copy_replay(self):
    expected = np.arange(16 << 20, dtype=np.uint8)
    x = Tensor(expected, device=self.devs[0]).realize()
    def copy(x): return x.to(self.devs[1]).contiguous().realize()
    np.testing.assert_equal(copy(x).numpy(), expected)
    f = TinyJit(copy)
    for _ in range(2): f(x)
    Device[self.devs[1]].synchronize()
    original = CPUProgram.remote_exec
    def delayed(prg, peer, *args, **kwargs):
      if peer.peer_group == Device[self.devs[1]].peer_group: time.sleep(0.05)
      return original(prg, peer, *args, **kwargs)
    # Force RNR on nonzero MSN indices: a posted receive must not be required before sending.
    with patch.object(CPUProgram, "remote_exec", delayed):
      for _ in range(3):
        f(x)
        Device[self.devs[1]].synchronize()
    start = time.perf_counter()
    for _ in range(50):
      y = f(x)
      Device[self.devs[1]].synchronize()
    elapsed = time.perf_counter() - start
    np.testing.assert_equal(y.numpy(), expected)
    print(f"RDMA copy: {50 * expected.nbytes / elapsed / 1e9:.2f} GB/s, 50 x 16 MiB")

  def test_sharded_reduce(self):
    def reduce(x): return (x + 1).sum().realize()
    x = Tensor(np.arange(1024, dtype=np.float32)).shard(self.devs, axis=0).clone().realize()
    expected = reduce(x).item()
    self.assertEqual(expected, 524800)
    f = TinyJit(reduce)
    for _ in range(52): self.assertEqual(f(x).item(), expected)

if __name__ == "__main__": unittest.main()
