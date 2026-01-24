#!/usr/bin/env python3
"""Test AMD compute queue stop and recreation functionality."""
import unittest
from tinygrad import Tensor, Device
from tinygrad.helpers import getenv, DEBUG, Timing

@unittest.skipUnless(Device.DEFAULT == "AMD", "AMD device required")
class TestAMDQueueReset(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.dev = Device["AMD"]
    # Check if this is an AM device (supports queue reset)
    if not cls.dev.is_am():
      raise unittest.SkipTest("Queue reset only supported on AM devices")

  def test_basic_compute_before_reset(self):
    """Verify basic compute works before any reset."""
    a = Tensor([1, 2, 3, 4]).realize()
    b = Tensor([5, 6, 7, 8]).realize()
    c = (a + b).realize()
    assert c.tolist() == [6, 8, 10, 12], f"Expected [6, 8, 10, 12], got {c.tolist()}"

  def test_queue_reset(self):
    """Test that queue can be stopped and recreated."""
    # Run a kernel first
    a = Tensor([1, 2, 3, 4]).realize()
    b = Tensor([5, 6, 7, 8]).realize()
    c = (a + b).realize()
    self.dev.synchronize()

    if DEBUG >= 2: print("Resetting compute queue...")

    # Reset the compute queue
    self.dev.reset_compute_queue()

    if DEBUG >= 2: print("Queue reset complete, running new kernel...")

    # Run another kernel after reset
    d = Tensor([10, 20, 30, 40]).realize()
    e = Tensor([1, 2, 3, 4]).realize()
    f = (d - e).realize()
    assert f.tolist() == [9, 18, 27, 36], f"Expected [9, 18, 27, 36], got {f.tolist()}"

  def test_multiple_resets(self):
    """Test that queue can be reset multiple times."""
    for i in range(3):
      if DEBUG >= 2: print(f"Reset iteration {i+1}")

      # Run kernel
      a = Tensor.randn(20000, 20000).realize()
      b = Tensor.randn(20000, 20000).realize()
      c = (a @ b).realize()

      # Reset queue
      with Timing("Resetting compute queue iteration"):
        self.dev.reset_compute_queue()

      self.dev.synchronize()

      # Verify compute still works after reset
      d = Tensor([i, i+1, i+2]).realize()
      e = (d * 2).realize()
      assert e.tolist() == [i*2, (i+1)*2, (i+2)*2], f"Iteration {i}: compute failed after reset"

  def test_synchronize_timeout_success(self):
    """Test synchronize_timeout returns True for fast kernel."""
    a = Tensor([1, 2, 3, 4]).realize()
    b = (a + 1).realize()
    # The kernel should complete well within 5 seconds
    success = self.dev.synchronize_timeout(5000)
    assert success, "synchronize_timeout should return True for completed kernel"
    assert b.tolist() == [2, 3, 4, 5]

if __name__ == "__main__":
  unittest.main()
