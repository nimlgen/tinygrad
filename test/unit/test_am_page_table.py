import struct, unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
from tinygrad.runtime.support.am.amdev import AMPageTableEntry
from tinygrad.runtime.support.system import RemoteCmd, RemoteMMIOInterface

class TestRemotePageTableReads(unittest.TestCase):
  def test_local_scan_stops_at_first_valid_entry(self):
    pt = AMPageTableEntry.__new__(AMPageTableEntry)
    pt.entries = MagicMock()
    pt.entries.__getitem__.return_value = 1
    self.assertTrue(any(pt.valid_entries(0, 512)))
    pt.entries.__getitem__.assert_called_once_with(0)

  def test_valid_entries_one_read_and_no_stale_cache(self):
    words = [0] * 512
    words[3], words[255], words[511] = 0x600000000000076, 0x12345001, 1
    def read(cmd, addr, size, *, bar, readout_size):
      self.assertEqual((cmd, bar, readout_size), (RemoteCmd.MMIO_READ, 0, size))
      return 0, 0, struct.pack(f'<{size//8}Q', *words[addr//8:(addr+size)//8])
    peer = SimpleNamespace(rpc=Mock(side_effect=read))
    pt = AMPageTableEntry(SimpleNamespace(vram=RemoteMMIOInterface(peer, 0, 0, 4096, 'Q')), 0, 3)
    self.assertEqual([i for i, v in enumerate(pt.valid_entries(0, 512)) if v], [255, 511])
    peer.rpc.assert_called_once_with(RemoteCmd.MMIO_READ, 0, 4096, bar=0, readout_size=4096)
    words[255], words[256] = 0, 1
    peer.rpc.reset_mock()
    self.assertEqual(list(pt.valid_entries(254, 3)), [False, False, True])
    peer.rpc.assert_called_once_with(RemoteCmd.MMIO_READ, 254*8, 3*8, bar=0, readout_size=3*8)

if __name__ == '__main__': unittest.main()
