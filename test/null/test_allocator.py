import unittest
from types import SimpleNamespace
from tinygrad.device import Allocator, BufferStorage

class FakeDevice:
  def __init__(self): self.timeline, self.synced = SimpleNamespace(host=SimpleNamespace(view=lambda fmt: [3, 5])), 0
  def synchronize(self): self.synced += 1

class FakeAllocator(Allocator):
  def _alloc(self, size, options): return BufferStorage(object(), host=memoryview(bytearray(size)))  # type: ignore[arg-type]
  def _free(self, storage, options): assert self.dev.synced, "freed under queued work"

class TestAllocator(unittest.TestCase):
  def test_free_cache_synchronizes(self):
    a = FakeAllocator(FakeDevice())
    a.free(a.alloc(64), 64)
    a.free_cache()
    self.assertEqual(a.dev.synced, 1)

if __name__ == "__main__":
  unittest.main()
