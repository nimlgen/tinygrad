#!/usr/bin/env python3
# a node behind a socket (tinygrad.runtime.support.system.RemotePCIDevice): the client drives this node's PCI devices by address and runs the
# programs it linked for them here. nothing here knows about GPUs.
#   every node, the local one too:  PYTHONPATH=. DEV=PCI+AMD python extra/remote/serve.py 6667
#   the driver:  REMOTE="localhost:6667,192.168.52.213:6667" DEV=PCI+AMD RDMA=1 python ...   (AMD:n counts through the nodes in REMOTE order,
#   BNXT:n is node n's nic; RDMA=1 copies between nodes over the nics, without it they stage through the nodes' host memory)
import socket, struct, sys, pickle, array, traceback
from tinygrad.device import Device
from tinygrad.helpers import DEBUG, DEV
from tinygrad.runtime.support.system import PCIDevice, RemoteCmd, REMOTE_REQ, REMOTE_RESP, System
from tinygrad.runtime.support.am.amdev import AMMemoryManager

devices:list[tuple[type, str]] = [] # probe order: the dev id is the index
opened:dict[int, PCIDevice] = {}
maps:list = [] # bar and sysmem views, looked up by address
progs:dict = {}
FMT = {1: 'B', 4: 'I', 8: 'Q'} # single elements are real 32/64-bit accesses (registers, doorbells), the rest is a copy

def resp(r0:int=0, payload:bytes=b'', status:int=0) -> bytes: return struct.pack(REMOTE_RESP, status, r0, len(payload)) + payload

def device(dev_id:int) -> PCIDevice:
  if dev_id not in opened:
    cl, pcibus = devices[dev_id]
    opened[dev_id] = cl("SV", pcibus)
  return opened[dev_id]

def view(addr:int, size:int, fmt:str):
  # the whole mapping and the element index in it: the mock emulates registers by their index in the bar
  if (m:=next((m for m in maps if m.addr <= addr and addr + size <= m.addr + m.nbytes), None)) is None:
    raise RuntimeError(f"{addr:#x}+{size:#x} is not mapped on this node")
  return m.view(fmt=fmt), (addr - m.addr) // struct.calcsize(fmt)

def kick_mock():
  # native programs bypass the mock's memoryview hooks: run the emulated queues after every program
  if DEV.interface.startswith("MOCK"):
    from test.mockgpu.mockgpu import drivers
    for d in drivers: d._emulate_execute()

def handle(cmd:RemoteCmd, dev_id:int, bar:int, a0:int, a1:int, a2:int, payload:bytes) -> bytes|None:
  if cmd == RemoteCmd.PROBE:
    filters:dict[int, list[int]] = {}
    for mask, dev in struct.iter_unpack('<II', payload): filters.setdefault(mask, []).append(dev)
    devs = System.list_devices(a1, tuple((m, tuple(d)) for m, d in filters.items()), a0 or None)
    for d in devs:
      if d not in devices: devices.append(d)
    return resp(payload="\n".join(f"{d[1]}:{devices.index(d)}" for d in devs).encode())
  if cmd == RemoteCmd.MEM_READ:
    v, i = view(a0, a1, FMT[a2])
    return resp(payload=bytes(v[i:i + a1]) if a2 == 1 else struct.pack(f'<{a1 // a2}{FMT[a2]}', *v[i:i + a1 // a2]))
  if cmd == RemoteCmd.MEM_WRITE:
    v, i = view(a0, len(payload), FMT[a2])
    if a2 == 1: v[i:i + len(payload)] = payload
    elif len(payload) == a2: v[i] = struct.unpack(f'<{FMT[a2]}', payload)[0]
    else: v[i:i + len(payload) // a2] = array.array(FMT[a2], payload)
    return None
  if cmd == RemoteCmd.LOAD_PROG:
    progs[h:=len(progs) + 1] = Device["CPU"].runtime(pickle.loads(payload))
    return resp(h)
  if cmd == RemoteCmd.EXEC_PROG:
    et = progs[a0](*struct.unpack(f'<{a1}Q', payload), wait=bool(a2))
    kick_mock()
    return resp(int(et * 1e9)) if a2 else None
  # device commands
  pci_dev = device(dev_id)
  if cmd == RemoteCmd.MAP_BAR:
    if (v:=next((m for m in maps if getattr(m, 'bar', None) == (dev_id, bar)), None)) is None:
      maps.append(v:=pci_dev.map_bar(bar))
      v.bar = (dev_id, bar) # type: ignore[attr-defined]
    return resp(pci_dev.bar_info(bar)[0], struct.pack('<QQ', v.nbytes, v.addr))
  if cmd == RemoteCmd.MAP_SYSMEM:
    v, paddrs = pci_dev.alloc_sysmem(a0, vaddr=a2, contiguous=bool(a1))
    maps.append(v)
    return resp(v.addr, struct.pack(f'<{len(paddrs)}Q', *paddrs))
  if cmd == RemoteCmd.CFG_READ: return resp(pci_dev.read_config(a0, a1))
  if cmd == RemoteCmd.CFG_WRITE:
    pci_dev.write_config(a0, a2, a1)
    return resp()
  if cmd == RemoteCmd.RESIZE_BAR:
    pci_dev.resize_bar(bar)
    return resp()
  if cmd == RemoteCmd.RESET:
    pci_dev.reset()
    return resp()
  raise RuntimeError(f"unknown command {cmd}")

def serve(conn:socket.socket):
  while True:
    if len(hdr:=conn.recv(struct.calcsize(REMOTE_REQ), socket.MSG_WAITALL)) < struct.calcsize(REMOTE_REQ): raise ConnectionError("client gone")
    cmd, dev_id, bar, a0, a1, a2 = struct.unpack(REMOTE_REQ, hdr)
    n = {RemoteCmd.PROBE: a2, RemoteCmd.MEM_WRITE: a1, RemoteCmd.LOAD_PROG: a0, RemoteCmd.EXEC_PROG: a1 * 8}.get(cmd, 0)
    payload = conn.recv(n, socket.MSG_WAITALL) if n else b''
    if DEBUG >= 4: print(f"cmd={RemoteCmd(cmd).name} dev={dev_id} bar={bar} a0={a0:#x} a1={a1:#x} a2={a2:#x}")
    try:
      if (r:=handle(RemoteCmd(cmd), dev_id, bar, a0, a1, a2, payload)) is not None: conn.sendall(r)
    except ConnectionError: raise
    except Exception as e:
      # posted commands have no reply to carry the error, so the connection is the error
      if cmd == RemoteCmd.MEM_WRITE or (cmd == RemoteCmd.EXEC_PROG and not a2): raise ConnectionError(f"{RemoteCmd(cmd).name} failed: {e}") from e
      traceback.print_exc()
      conn.sendall(resp(payload=str(e).encode(), status=1))

if __name__ == "__main__":
  port = int(sys.argv[1]) if len(sys.argv) > 1 else 6667
  System.reserve_va(AMMemoryManager.va_allocator.base, AMMemoryManager.va_allocator.size) # sysmem is mapped at the GPU VA the client plans
  server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  server.bind(("0.0.0.0", port))
  server.listen(1)
  print(f"listening on {port}")
  while True:
    conn, addr = server.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF): conn.setsockopt(socket.SOL_SOCKET, opt, 64 << 20)
    try: serve(conn)
    except ConnectionError as e: print(f"disconnected: {e}")
    finally: conn.close()
