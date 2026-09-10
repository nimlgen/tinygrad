Run a server from this tree on each node (including the local node):

```sh
PYTHONPATH=. python extra/remote/serve.py 6667
```

The coordinator boots the PCI drivers over TCP and runs hcq2 submit programs on each node. Nodes must support the coordinator's CPU instruction set.

```sh
export REMOTE="localhost:6667,192.168.52.213:6667"
export DEV=PCI+AMD
export RDMA=1
python -m pytest test/external/test_rdma_jit_copy.py -n12 --dist loadscope -rP
```

`RDMA=1` uses a BCM57608 on each node for cross-node copies; without it, copies stage through the nodes' host memory. `RDMA_DEVS=AMD,AMD:6` selects the two GPUs in the external test. Same-node copies use SDMA.

The external test covers 16 MiB copies, delayed receives, ring wraps, and sharded reductions through JIT replay. Local tests need no NICs:

```sh
python -m pytest test/unit/test_bnxt.py test/unit/test_bnxt_hcq2.py test/device/test_hcq2.py test/device/test_rdma_hcq2.py test/device/test_remote_hcq2.py -n12
```

A small Llama training run uses all twelve test GPUs and prints loss, gradient norm, and step time. The first two steps execute eagerly and capture; subsequent steps replay the JIT:

```sh
GPUS=12 STEPS=5 python extra/remote/train.py
```

The server retains mappings and loaded programs for its lifetime. Standalone CPU-driven NIC tests are in `extra/bnxt_driver/`; loopback alternates receive-first and delayed-receive transfers to exercise retransmission.
