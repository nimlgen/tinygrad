#!/usr/bin/env python3
"""Connect two MLX5 devices over RoCE v2. Run on the local machine."""
import subprocess, json, sys, os

REMOTE_HOST = os.getenv("REMOTE_HOST", "192.168.52.154")
LOCAL_PCI   = os.getenv("MLX_PCI", "0000:41:00.0")
REMOTE_PCI  = os.getenv("REMOTE_PCI", "0000:41:00.0")
LOCAL_IP    = os.getenv("LOCAL_IP", "10.0.0.1")
REMOTE_IP   = os.getenv("REMOTE_IP", "10.0.0.2")
SSH         = ["ssh", "-o", "StrictHostKeyChecking=no", REMOTE_HOST]
TINYGRAD    = os.path.dirname(os.path.abspath(__file__)) + "/../.."

# 1. Sync code to remote
print("=== syncing code to remote ===")
subprocess.run(["rsync", "-az", "--exclude=.git", "--exclude=__pycache__", "--exclude=*.pyc",
                TINYGRAD + "/", f"{REMOTE_HOST}:~/tinygrad/"], check=True)

# 2. Start remote in server mode (boots device, prints JSON info, waits for our info on stdin)
print("=== booting remote ===")
remote = subprocess.Popen(
  SSH + [f"cd ~/tinygrad && sudo PYTHONPATH=. MLX_DEBUG=1 MLX_PCI={REMOTE_PCI} python3 extra/mlx_driver/mlxdev.py --server --ip {REMOTE_IP}"],
  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True)

# Read remote output until we get the JSON connection info line
remote_info = None
for line in iter(remote.stdout.readline, ''):
  print(f"  [remote] {line}", end='')
  try: remote_info = json.loads(line.strip()); break
  except json.JSONDecodeError: pass
assert remote_info, "failed to get remote connection info"
print(f"=== remote info: QPN=0x{remote_info['qpn']:x} MAC={remote_info['mac']} ===")

# 3. Boot local device
print("=== booting local ===")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from extra.mlx_driver.mlxdev import MLXDev
from tinygrad.runtime.support.system import PCIDevice

local_dev = MLXDev(PCIDevice("mlx5", LOCAL_PCI))
local_dev.set_roce_address(0, LOCAL_IP)
local_info = local_dev.connection_info()
print(f"=== local info: QPN=0x{local_info['qpn']:x} MAC={local_info['mac']} ===")

# 4. Send local info to remote -> it does INIT2RTR + RTR2RTS
remote.stdin.write(json.dumps(local_info) + "\n")
remote.stdin.flush()

# Wait for remote "connected" ack
for line in iter(remote.stdout.readline, ''):
  print(f"  [remote] {line}", end='')
  if "connected" in line: break

# 5. Local INIT2RTR + RTR2RTS
local_dev.init2rtr(remote_info["qpn"], remote_info["mac"], remote_info["gid"])
local_dev.rtr2rts()

print("=== both QPs in RTS - connection established ===")
print(f"  local:  QPN=0x{local_info['qpn']:x}  IP={LOCAL_IP}  MAC={local_info['mac']}")
print(f"  remote: QPN=0x{remote_info['qpn']:x} IP={REMOTE_IP} MAC={remote_info['mac']}")

# 6. Wait for remote to post receive WQE
for line in iter(remote.stdout.readline, ''):
  print(f"  [remote] {line}", end='')
  if "recv_posted" in line: break

# 7. SEND test data to remote
local_dev.cq_ci = 0
local_dev.sq_head = 0
local_dev.rq_head = 0
test_msg = b"Hello from tinygrad MLX5 driver! SEND works!"
src_mem, src_paddrs = local_dev.pci_dev.alloc_sysmem(0x1000)
for i, b in enumerate(test_msg): src_mem[i] = b

print(f"=== SEND {len(test_msg)}B to remote ===")
local_dev.send(src_paddrs[0], local_dev.resd_lkey, len(test_msg))

# 8. Tell remote to poll CQ and read received data
remote.stdin.write("done\n")
remote.stdin.flush()

# 9. Read remote's verification output
for line in iter(remote.stdout.readline, ''):
  print(f"  [remote] {line}", end='')
  if "AS TEXT" in line: break

remote.stdin.close()
remote.wait()
print("=== SEND/RECV test complete ===")
