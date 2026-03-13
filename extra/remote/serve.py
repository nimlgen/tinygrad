#!/usr/bin/env python3
import socket, struct, sys
from tinygrad.runtime.support.system import PCIDevice, RemoteCmd
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.helpers import DEBUG

_resp_ok = struct.pack('<BQQ', 0, 0, 0)

def serve(pci_dev:PCIDevice, conn:socket.socket):
  bar_mmio:dict[int, MMIOInterface] = {b: pci_dev.map_bar(b) for b in pci_dev.bar_fds}
  sysmem:list[tuple[MMIOInterface, list[int]]] = []
  recv, send, sendmsg = conn.recv, conn.sendall, conn.sendmsg
  _pack, _unpack = struct.pack, struct.unpack

  while True:
    hdr = recv(26, socket.MSG_WAITALL)
    if len(hdr) < 26: raise ConnectionError("client disconnected")
    cmd, bar, offset, size, value = _unpack('<BBQQQ', hdr)
    if DEBUG >= 2: print(f"cmd={RemoteCmd(cmd).name} bar={bar} offset={offset:#x} size={size:#x} value={value:#x}")

    if cmd == RemoteCmd.MAP_BAR:
      if DEBUG >= 1: print(f"  MAP_BAR {bar}: size={pci_dev.bar_info[bar].size:#x}")
      send(_pack('<BQQ', 0, pci_dev.bar_info[bar].size, 0))

    elif cmd == RemoteCmd.CFG_READ:
      val = pci_dev.read_config(offset, size)
      if DEBUG >= 2: print(f"  CFG_READ offset={offset:#x} size={size} -> {val:#x}")
      send(_pack('<BQQ', 0, val, 0))

    elif cmd == RemoteCmd.CFG_WRITE:
      if DEBUG >= 2: print(f"  CFG_WRITE offset={offset:#x} value={value:#x} size={size}")
      pci_dev.write_config(offset, value, size)
      send(_resp_ok)

    elif cmd == RemoteCmd.RESET:
      if DEBUG >= 1: print("  RESET")
      pci_dev.reset()
      send(_resp_ok)

    elif cmd == RemoteCmd.MMIO_READ:
      if DEBUG >= 3: print(f"  MMIO_READ bar={bar} offset={offset:#x} size={size:#x}")
      sendmsg([_pack('<BQQ', 0, size, 0), bar_mmio[bar].mv[offset:offset + size]])

    elif cmd == RemoteCmd.MMIO_WRITE:
      bar_mmio[bar].mv[offset:offset + size] = recv(size, socket.MSG_WAITALL)
      if DEBUG >= 3: print(f"  MMIO_WRITE bar={bar} offset={offset:#x} size={size:#x}")

    elif cmd == RemoteCmd.SYSMEM_READ:
      if DEBUG >= 3: print(f"  SYSMEM_READ handle={bar} offset={offset:#x} size={size:#x}")
      sendmsg([_pack('<BQQ', 0, size, 0), sysmem[bar][0].mv[offset:offset + size]])

    elif cmd == RemoteCmd.SYSMEM_WRITE:
      sysmem[bar][0].mv[offset:offset + size] = recv(size, socket.MSG_WAITALL)
      if DEBUG >= 3: print(f"  SYSMEM_WRITE handle={bar} offset={offset:#x} size={size:#x}")

    elif cmd == RemoteCmd.MAP_SYSMEM:
      memview, paddrs = pci_dev.alloc_sysmem(size)
      handle = len(sysmem)
      sysmem.append((memview, paddrs))
      paddrs_bytes = _pack(f'<{len(paddrs)}Q', *paddrs)
      if DEBUG >= 1: print(f"  MAP_SYSMEM size={size:#x} handle={handle} paddrs={len(paddrs)}")
      send(_pack('<BQQ', 0, len(paddrs_bytes), handle) + paddrs_bytes)

if __name__ == "__main__":
  pcibus, bars = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
  resize_bars = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
  port = int(sys.argv[4]) if len(sys.argv) > 4 else 6667

  pci_dev = PCIDevice("SV", pcibus, bars=bars, resize_bars=resize_bars)
  server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  server.bind(("0.0.0.0", port))
  server.listen(1)
  print(f"listening on 0.0.0.0:{port} for {pcibus}")
  while True:
    conn, addr = server.accept()
    print(f"connected: {addr}")
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for bt in [socket.SO_SNDBUF, socket.SO_RCVBUF]: conn.setsockopt(socket.SOL_SOCKET, bt, 64 << 20)
    try: serve(pci_dev, conn)
    except ConnectionError: print("disconnected")
