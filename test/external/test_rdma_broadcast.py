import unittest
import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.helpers import getenv
from tinygrad.runtime.support.hcq2 import STAGING_SIZE

@unittest.skipUnless(getenv("RDMA"), "requires two eight-GPU AMD nodes")
class TestRDMABroadcast(unittest.TestCase):
  def test_parameter_broadcast_replay(self):
    devices = tuple(f"AMD:{i}" for i in range(16))
    for source in (devices[0], devices[8]):
      f = TinyJit(lambda x: x.shard(devices).contiguous().realize())
      for iteration in range(5):
        expected = np.arange(16384, dtype=np.float32) + iteration
        result = f(Tensor(expected, device=source).realize())
        for rank in range(16):
          np.testing.assert_array_equal(Tensor(result.uop.mselect(rank)).numpy(), expected,
                                        err_msg=f"source={source} iteration={iteration} rank={rank}")

  def test_unlinked_copy_reuses_staging_slots(self):
    n = STAGING_SIZE + 17 # three chunks: the third reuses the first staging slot
    for source, dest in (("AMD", "AMD:9"), ("AMD:8", "AMD:1")):
      f = TinyJit(lambda x: x[16:].to(dest).contiguous().realize())
      for iteration in range(5):
        x = ((Tensor.arange(n+16).clone(device=source) + iteration) % 251).cast(dtypes.uint8).contiguous().realize()
        result = f(x)
        expected = ((Tensor.arange(16, n+16).clone(device=dest) + iteration) % 251).cast(dtypes.uint8)
        self.assertTrue((result == expected).all().item(), f"source={source} iteration={iteration}")

if __name__ == "__main__": unittest.main()
