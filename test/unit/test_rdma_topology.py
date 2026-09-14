import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from tinygrad.runtime import ops_rdma as rdma

class TestRDMATopology(unittest.TestCase):
  def test_all_links_and_cross_rank_broadcast(self):
    buses = (5, 21, 101, 117, 133, 149, 229, 245)
    def dev(name, group, bus): return NS(device=name, peer_group=group, iface=NS(pci_dev=NS(pcibus=f"remote:host:6667:0000:{bus:02x}:00.0:0")))
    nics = {g: tuple(dev(f"RDMA:{g*8+i}", g, b+1) for i, b in enumerate(buses)) for g in range(2)}
    gpus = [dev(f"AMD:{g*8+i}", g, b) for g in range(2) for i, b in enumerate(buses)]
    with patch.object(rdma, "node_nics", side_effect=nics.__getitem__):
      for a in range(8):
        for b in range(8):
          left, right = rdma.rdma_nic_for(gpus[a], gpus[8+b]), rdma.rdma_nic_for(gpus[8+b], gpus[a])
          self.assertEqual(nics[0].index(left), nics[1].index(right))
          if a == b: self.assertEqual(left, nics[0][a])

if __name__ == "__main__": unittest.main()
