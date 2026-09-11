Review of `hcq2_rdma` for Claude, 2026-09-11. Reviewed HEAD: `32c40ddaf64771bca0820caaa6ed2a36af9f4449`.

**Keep the current single-controller, sharded-Tensor training model. Simplify its state and supported operating mode.** The essential runtime is already reasonably small. Replacing it with another distributed training framework, a new collective implementation, or a separate Python training process on every GPU would introduce substantial new work. The useful reductions here are fewer counters, one server lifetime per job, a smaller training example, and explicit device assumptions.

I reviewed the branch-specific changes against `4c5ec4602`, the commit immediately before this series: 25 files, 1,142 insertions and 392 deletions, including the BNXT driver move, generated definitions, and tests. Comparing with the current local `master` also shows unrelated MLPerf, compiler, profiler, and other changes. Those should not be attributed to the RDMA implementation. This review changes no implementation code.

The execution path worth preserving is:

```text
one Python model + optimizer + TinyJit
  -> replicated parameters, batch sharded across AMD devices
  -> existing allreduce lowering produces ordinary tensor copies
  -> cross-node copies split into SEND on source / RECV on destination
  -> one HCQ submit batch per participating node, within each contiguous HCQ region
  -> CPU submit program runs on that node
  -> GPU writes NIC descriptors, rings doorbells, waits for completion
```

`tinygrad/schedule/multi.py` and `tinygrad/schedule/allreduce.py` already supply the collective semantics. The model does not need to know about QPs, NICs, ranks, or remote addresses. TCP carries setup, submission, and host transfers; RDMA carries the cross-node GPU copies when enabled.

| Part | What training needs | Recommendation |
|---|---|---|
| Sharding, autograd, optimizer, TinyJit | Existing training semantics | Keep unchanged. |
| `peer_group`, batch partitioning, node-local scratch | Memory addresses and queue dependencies must belong to the right node | Keep. |
| Remote PCI access and `LOAD_PROG` / `EXEC_PROG` | Reuse the client-side drivers and execute a whole submit program near the GPU | Keep the small protocol; narrow its lifetime. |
| `RDMADevice` / allocator | Buffer registration, cached keys, mapping ownership and synchronization | Keep the existing device integration. |
| Paired SEND/RECV on the GPU compute queues | Ordered copies without CPU intervention for every tensor | Keep the current conservative path. |
| CPU-driven BNXT send/receive helpers | Hardware diagnosis and transport bring-up | Keep as diagnostic utilities; they are not steps required to launch training. |
| Llama smoke example | A recognizable end-to-end exercise | Keep as an example; add a much smaller numerical correctness test. |

**1. Collapse the five replay counters to three, and allocate their storage together.**

At `tinygrad/runtime/ops_rdma.py:78`, each QP endpoint allocates `sq_prod`, `sq_psn`, `rq_prod`, `scq_cons`, and `rcq_cons`. At lines 93–110, every send or receive advances its producer and completion counter exactly once. Every send requests a completion, each QP has its own CQs, and each copy waits for its completion before proceeding. In this path, `sq_prod == scq_cons` and `rq_prod == rcq_cons` throughout successful execution. These are host-maintained sequence numbers for encoded work, not independently sampled hardware progress.

Use one send sequence, one receive sequence, and the send packet sequence number. Derive both the WQE index/epoch and CQ index/toggle from the corresponding operation sequence, with their respective ring sizes. Keep the packet sequence separate: message sizes change how many packets a send consumes.

`BNXTIface.counter()` currently allocates a separate 4 KiB page for each eight-byte counter. One zeroed page with three uint64 views would replace five pages and simplify the buffer/tag state. This is a concrete reduction from five allocations and state variables to one allocation and three state variables per endpoint.

Preserve `HWQueue.bump()` and the ordering in `encode_submit()`: encode this execution's command words, then save the next counter values, then publish the submission. The current AFTER dependencies are doing real work. Revalidate this simplification if unsignaled sends, shared CQs, batched outstanding work, or mixing CPU and HCQ posting on the same QP is introduced.

**2. Make the server a process for one training job.**

`extra/remote/serve.py:12` keeps `devices`, `opened`, `maps`, and `progs` globally. The accept loop at line 111 accepts another client after disconnect without clearing them. Every `LOAD_PROG` adds a program and every `MAP_SYSMEM` adds a retained mapping. There is no unload/free command; `PCIIfaceBase.free()` at `tinygrad/runtime/support/system.py:287` deliberately skips CPU unmapping for remote allocations.

That is an incomplete persistent-server lifecycle. The simplest supported mode is one client/job per server process, followed by process exit and a fresh process for the next job. Remove the multi-client accept loop instead of introducing reconnect, handle generations, session tables, and a general remote resource manager now. Normal controller shutdown should retain device finalization, including BNXT driver unregister. Abrupt disconnect still needs hardware validation of the next takeover; process exit alone does not prove the NIC has been reset safely.

This choice addresses resources retained across jobs. It does not reclaim allocations continually created within one job. Check that a fixed-shape JIT training run reaches a stable allocation/program count after warmup, including when fresh input tensors are supplied. If that grows, implement the specific missing free/reuse operation. Do not describe process-per-job as solving arbitrary long-running allocation churn.

**3. Shrink the training step and prove its updates.**

`extra/remote/train.py:18` computes a global gradient norm on every step, realizes loss/norm separately, then calls `optim.step()`. The norm is useful for diagnosis but is unnecessary for SGD. Use the standard existing pattern, also used in `examples/hlb_cifar10.py`:

```python
@TinyJit
def step(x, y):
  optim.zero_grad()
  loss = model(x, 0, temperature=float('nan')).sparse_categorical_crossentropy(y)
  loss.backward()
  return loss.realize(*optim.schedule_step())
```

This removes the norm computation and gives the eager path one realization containing loss and updates. JIT can already combine captured work, so do not assume the current two realization calls necessarily mean two permanent JIT batches. Keep loss evaluation ordered with the updates: moving `loss.realize()` after a separate `optim.step()` can recompute a lazy loss using updated weights.

The existing example's only assertion is finite loss and finite positive norm (`train.py:33`). Frozen weights, an incorrect reduction scale, or replicas that diverge can pass that assertion. A smaller linear regression + SGD check is much more decisive:

- Use different samples on the two nodes and known initial weights.
- Compare loss and every replica's updated weights with a full-batch reference after each step.
- Change the input data and allocations across JIT replay.
- Run the same check through staging and RDMA. For momentum/Adam training, extend it to optimizer state when that optimizer is actually needed.

I verified this approach using the existing remote mock harness: six steps, two nodes, changing inputs, and both weight replicas matched NumPy with `rtol=1e-4`, `atol=1e-6`. The tested step was `return loss.realize(*optim.schedule_step())`. The reference used `loss = mean((X @ W - Y)**2)` and `W -= 0.01 * (2 / batch_size) * X.T @ (X @ W - Y)`. This was `RDMA=0`; it establishes the training semantics and staging path, not hardware RDMA correctness.

**4. Support one explicit hardware arrangement first.**

For the first reliable training path, use AMD PCI devices, one supported BNXT NIC per node, one controller, and CPU-compatible servers running the same checkout. Running a server on the controller's GPU node too makes all GPU nodes follow the same path. Keep local execution support in the shared runtime; simply avoid requiring a mixed local/remote deployment to get started.

There are three places where current generality exceeds the implemented path:

- `split_rdma()` at `hcq2.py:155` accepts all `HCQ_DEVS`, which includes NV and QCOM. The actual encoder needs AMD-style `write`, full-width `signal`, and `wait(..., eq=True)` methods. NV/QCOM do not currently supply that same interface. Reject unsupported RDMA endpoints early with a small explicit check; do not build a general transport capability framework just for this branch.
- `nic_for()` at `hcq2.py:37` scans `Device['RDMA:n']` from zero with `itertools.count()`. Looking up a device initializes its driver, including a NIC reset. Discover a bounded mapping from node to NIC once, validate the one-NIC assumption, and cache selection. A missing NIC should name the node in its error. Caching alone avoids repeated traversal but does not fix first-use probing of unrelated NICs.
- `BNXT_IP` at `ops_rdma.py:21` is read by the controller for every NIC. A single override assigns the same configured IP/GID to all of them. The current defaults are distinct by global NIC index. Document those defaults and avoid suggesting per-server environment overrides configure client-driven NIC initialization. Add a small per-NIC mapping only if the deployment needs explicit addresses.

The trainer defaults to 12 GPUs, and the RDMA test defaults to `AMD,AMD:6`. Those fit a particular machine arrangement. Allow an explicit device list for the example, or document the enumeration and validate the selected nodes. `GPUS=2` selects the first two devices; on a six-GPU first node it does not exercise cross-node training.

`CPUProgram.remote_exec()` sends an already compiled object to the server. The controller's CPU target is `native`; matching CPU ISA/features and checkout are real requirements today. Document a homogeneous setup. Cross-architecture compilation and a compatibility protocol can wait.

**5. Add a bounded failure path before calling this a dependable trainer.**

`RemotePCIDevice.connect()` at `system.py:343` uses `REMOTE_TIMEOUT` only while connecting, then calls `sock.settimeout(None)`. `_recvall()` can subsequently block indefinitely. The server executes a submit program inline (`serve.py:59`), and that program can spin in `hcq_fence()` waiting for prior GPU work. If that GPU work never finishes, the server cannot service later reads on the same socket.

The GPU CQ wait at `ops_rdma.py:109` matches the expected toggle/type plus status zero. An error CQE will not satisfy it. The surrounding Python device timeout cannot interrupt a socket read already blocked inside the polling operation. Thus there is a concrete path from a device/peer failure to an unbounded control-plane wait, despite existing device timeout code.

Give remote RPCs a finite, configurable deadline and fail the whole job on timeout. Pair that with fresh server processes for the next attempt. A slow firmware operation needs an appropriately generous deadline. This is substantially simpler than adding automatic reconnect, QP recovery, or replay after partial submission. I established this failure path by code inspection; I did not inject NIC failures on hardware.

**6. Keep the ordering and ownership that make this small design possible.**

These are poor targets for deletion:

- Per-GPU-pair QPs (`ops_rdma.py:71`). Independent GPU queues must not race to publish descriptors or disagree on SEND/RECV matching. Sharing one QP for a whole node would require a new sequencer or serialization scheme. Keep the current lazy allocation.
- Splitting copies before batching, and tracking SEND as a source read / RECV as a destination write (`hcq2.py:155`, `:203`). This preserves source lifetime and destination readiness without cross-node GPU semaphore pointers.
- Posting peer batches before waiting (`realize.py:204`). Waiting immediately for one RDMA batch can prevent its receive/send peer from ever being submitted. Preserve this under DEBUG/profiling too.
- Node-local host scratch, staging, and the memory-owner dependency bookkeeping (`hcq2.py:130`, `:264`, `:294`; `device.py:414`). CPU addressability and completion tracking are necessary even when GPU payload copies use RDMA.
- Memory registration through the buffer allocator (`ops_rdma.py:42`). Base-buffer registration and view key reuse are useful; a parallel tensor-address/key cache would duplicate existing ownership machinery.
- Confirmed descriptor writes, 64-bit end-of-pipe doorbells, the receive-side cache barrier, volatile accesses through views, and the uncached RDMA command buffer (`ops_amd.py:401`, `coalesce.py:116`, `hcq2.py:424`). They address visibility between the submit CPU, GPU command processor, GPU caches, and NIC. Unit word-generation tests cannot justify removing them.
- Chunking and packet-sequence/MSN handling. Small-model success is not evidence that large transfers or retransmission state are unnecessary.

The NIC-as-`Compiled` design looks broader than the device's purpose, but `ops_rdma.py` is only 111 lines and reuses buffer mapping, linking and finalization. Removing this abstraction is unlikely to make the whole implementation smaller.

The CPU-driven BNXT diagnostic helpers duplicate some descriptor/sequence logic, but they remain useful for separating NIC issues from HCQ issues. They can stay in a diagnostic layer if runtime tidiness matters. Avoid a generic posting framework to unify the CPU and GPU execution paths. One small cleanup is the unused `Ops.INS('write')` rewrite in `hcq2.py:304`: repository search found no producer, and RDMA calls `hq.write()` directly. Keep the `wait_eq` rewrite: `support/usb.py` uses it.

**7. Strengthen the hardware check before optimizing communication.**

`test/external/test_rdma_jit_copy.py:14` replays the same input allocation with the same bytes. Its copy test does 56 transfers in total, enough to wrap the 32-entry WQ ring but insufficient to wrap its 128-entry CQ. Reusing identical contents also weakens detection of stale data. The unit encoder test runs 130 iterations and checks words/counters, but it does not exercise NIC completion or GPU cache behavior.

Extend the hardware test to at least 300 transfers per direction with changing contents, alternating input allocations, and delayed receiver submission after wrap. Include an offset view and a multi-chunk copy. The split can be exercised with a temporarily smaller `RDMA_CHUNK` rather than requiring multi-gigabyte test buffers. Then run the numerical SGD check on two real nodes, followed by the Llama smoke run.

Do not add a new RDMA allreduce, RDMA WRITE protocol, communication thread, or overlap scheme until this passes. The current SEND/RECV protocol avoids distributing remote destination addresses/keys to the sender. Waiting for every completion on the compute queue gives up overlap, but buys a simple ordering argument. Existing flat allreduce may become expensive across many GPUs; profile the real training step before adding a hierarchical collective or changing algorithms. Tiny Llama and a one-way copy benchmark do not establish training throughput.

For timing, warm up through capture and synchronize every participating GPU at the measurement boundary. The example reads scalar results and mixes eager, capture, and replay times. Use `tinygrad/viz/README.md` to inspect the training interval, separating kernels, copies, and submission overhead.

The current launch shape, assuming two nodes with six visible GPUs each, is:

```bash
# On each GPU node, from the same checkout; restart for each training job.
PYTHONPATH=. REMOTE= DEV=PCI+AMD python extra/remote/serve.py 6667

# On the controller. HOST_A and HOST_B are the two server addresses.
PYTHONPATH=. REMOTE="HOST_A:6667,HOST_B:6667" DEV=PCI+AMD RDMA=1 GPUS=12 STEPS=5 python extra/remote/train.py
```

For two nodes with one visible GPU each, use `GPUS=2`. No preliminary `extra/bnxt_driver/connect.py` run is required; `RDMADevice.qp()` connects peers during setup. `RDMA=0` selects the existing staging fallback for comparison. These are commands derived from the implementation, not a hardware deployment I ran during this review.

My suggested implementation order is: simplify the training step and add the numerical test; reduce the counters; narrow the server lifecycle and bound RPC waits; validate supported device/NIC selection; run the hardware replay and training checks. Leave collective algorithms and queue overlap for a measured bottleneck. The largest simplification is a smaller supported operating mode, not deleting synchronization or replacing HCQ.

Validation performed in this worktree:

```text
python -m pytest test/unit/test_bnxt.py test/unit/test_bnxt_transport.py \
  test/unit/test_bnxt_hcq2.py test/device/test_remote_hcq2.py \
  test/null/test_uops.py -x -q -n12
87 passed, 2 skipped, 1 xfailed

DEV=MOCKPCI+AMD python -m pytest test/device/test_hcq2.py \
  -x -q -n12 --dist=loadfile --tb=short
21 passed

python -m mypy tinygrad/
Success: no issues found in 217 source files

python -m ruff check .
All checks passed
```

The first attempt at the local mock HCQ2 suite used default xdist distribution and failed because workers contended for `am_mock:am:0.lock`; assigning the file to one worker with `--dist=loadfile` resolved it while retaining `-n12`. The additional six-step numerical training experiment described above passed using `TestRemoteHCQ2.run_remote(..., nodes=2)`. No real NIC/GPU RDMA tests, hardware Llama training, or throughput measurements were run. The review's hardware-sensitive recommendations still need those checks.
