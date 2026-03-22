#!/usr/bin/env python3
import subprocess, json, sys, os

REMOTE_HOST = os.getenv("REMOTE_HOST", "192.168.52.154")
LOCAL_PCI   = os.getenv("MLX_PCI", "0000:41:00.0")
REMOTE_PCI  = os.getenv("REMOTE_PCI", "0000:41:00.0")
LOCAL_IP    = os.getenv("LOCAL_IP", "10.0.0.1")
REMOTE_IP   = os.getenv("REMOTE_IP", "10.0.0.2")
SSH         = ["ssh", "-o", "StrictHostKeyChecking=no", REMOTE_HOST]
TINYGRAD    = os.path.dirname(os.path.abspath(__file__)) + "/../.."

print("syncing code to remote")
subprocess.run(["rsync", "-az", "--exclude=.git", "--exclude=__pycache__", "--exclude=*.pyc",
                TINYGRAD + "/", f"{REMOTE_HOST}:~/tinygrad/"], check=True)

print("booting remote")
remote = subprocess.Popen(
  SSH + [f"cd ~/tinygrad && sudo PYTHONPATH=. MLX_DEBUG=1 MLX_PCI={REMOTE_PCI} python3 extra/mlx_driver/mlxdev.py --server --ip {REMOTE_IP}"],
  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True)

remote_info = None
for line in iter(remote.stdout.readline, ''):
  print(f"  [remote] {line}", end='')
  try: remote_info = json.loads(line.strip()); break
  except json.JSONDecodeError: pass
assert remote_info, "failed to get remote connection info"
print(f"remote info: QPN=0x{remote_info['qpn']:x} MAC={remote_info['mac']}")

print("booting local")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from extra.mlx_driver.mlxdev import MLXDev
from tinygrad.runtime.support.system import PCIDevice

local_dev = MLXDev(PCIDevice("mlx5", LOCAL_PCI))
local_dev.set_roce_address(0, LOCAL_IP)
local_info = local_dev.connection_info()
print(f"local info: QPN=0x{local_info['qpn']:x} MAC={local_info['mac']}")

remote.stdin.write(json.dumps(local_info) + "\n")
remote.stdin.flush()
for line in iter(remote.stdout.readline, ''):
  print(f"  [remote] {line}", end='')
  if "connected" in line: break

local_dev.init2rtr(remote_info["qpn"], remote_info["mac"], remote_info["gid"])
local_dev.rtr2rts()
print("both QPs in RTS - connection established")

remote_target = None
for line in iter(remote.stdout.readline, ''):
  print(f"  [remote] {line}", end='')
  try: remote_target = json.loads(line.strip()); break
  except json.JSONDecodeError: pass
assert remote_target, "failed to get remote target info"

test_msg = b"Test message, rdma works!"
src_mem, src_paddrs = local_dev.pci_dev.alloc_sysmem(0x1000)
for i, b in enumerate(test_msg): src_mem[i] = b

print(f"RDMA WRITE {len(test_msg)}B to remote phys 0x{remote_target['target_addr']:x}")
local_dev.rdma_write(remote_target["target_addr"], remote_target["rkey"], src_paddrs[0], local_dev.mkey, len(test_msg))

remote.stdin.write("done\n")
remote.stdin.flush()
for line in iter(remote.stdout.readline, ''):
  print(f"  [remote] {line}", end='')
  if "AS TEXT" in line: break

remote.stdin.close()
remote.wait()
print("RDMA WRITE test complete")
