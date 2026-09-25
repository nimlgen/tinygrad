# Handoff: gpt-oss 20B, 2x8 MI350X (tinyamd3 + tinyamd4), tinygrad AM driver, hcq2, AQL only

> **Codex follow-up, 2026-09-23 09:24 PDT:** See `/home/nimlgen/tinygrad/HANDOFF_CLAUDE.md` for the latest measurements and deployment state. The two-node embedding below was found deployed on both machines and failed a fresh control; remote embedding and scheduler are now restored to HEAD, while the local experimental files remain untouched. A model-only deferred expert-reduction prototype showed no clear speedup and was removed. Both machines are idle; reset/preflight before the next run.

Written 2026-09-23 ~09:00 PDT by Claude. Goal: the MLPerf gpt-oss run (24 layers, BS=32, DP=16 over two boxes) converges in < 71 min.
Rules from the user: `AMD_AQL=1` only, plain O2 for the host compiler, keep it simple, do not hack tinygrad core more than needed.

## 1. State in one paragraph

Steady state is **0.905 s/update** (24L, BS=32, O2, JITBEAM=3 warm cache), down from 1.25 at the start of the day; startup is
**117 s init + 729 s two-update JIT capture (~14 min)**, down from 27 min. Losses match the single-box run. One box does the same
per-GPU work in **0.61 s**, so ~0.3 s is unhidden two-node communication. The 71-min target (196608 sequences = 6144 steps at BS=32,
minus ~14 min startup and ~6 min of 16 evals) needs **~0.45-0.5 s/step**: hiding all the communication gets to ~0.65-0.7, the rest
is per-GPU compute (kernels / batch shape). Everything codex had written earlier today was reverted by the user's request; the tree
now carries only the small changes in section 3.

## 2. Boxes, trees, helpers

| | tinyamd3 (ta3) | tinyamd4 (ta4) |
|---|---|---|
| ssh from tinyr4 | `ssh tinyamd3` (192.168.52.209) | `ssh tinyamd4` (192.168.52.153, the coordinator) |
| IPMI (last resort) | `ipmitool -I lanplus -H 192.168.52.137 -U ADMIN -P RZLWLOKHKI power reset` | `ipmitool -I lanplus -H 192.168.52.167 -U ADMIN -P WHHUMUXTKS power reset` |
| venv | `~/rdma16/aql-test-venv` | `~/rdma16/wandb-venv` (py-spy, wandb) |
| tree | `~/tg-gw2` | `~/tg-gw2` |

- Local source of truth: `/home/nimlgen/tg-gptoss` (branch `gptoss_work2_master`, HEAD bfce4848d + uncommitted section 3). Deploy with
  `rsync -rc --exclude __pycache__ --exclude '*.pyc' --exclude .git ./tinygrad ./extra ./examples ./test <box>:tg-gw2/` (both boxes are
  currently identical to it except the two files marked "not deployed" in section 3).
- No kfd on either box: the AM userspace driver runs the GPUs (`DEV=PCI+AMD`), no amdgpu module may be loaded (`lsmod | grep amdgpu`).
- **Before every run:** `~/gw2_serve.sh` on both boxes (refuses if a coordinator is connected or a training job runs; rebinds bnxt_en
  to revive NICs; hive_reset of all 8 GPUs; starts `extra/remote/serve.py 6667`). `/home/nimlgen/gptoss-handoff/launch.sh <script> <log> ENV..`
  does both resets in parallel, opens all 16 NICs once (catches a wedged NIC before the run), then launches detached on ta4.
- Launch: `launch.sh '~/gw2_run2x8.sh x' l24-xyz.log AMD_AQL=1 BENCHMARK=12 JITBEAM=3 GPTOSS_JIT_NO_WARMUP=1 HCQ2_STATS=1 PYTHONUNBUFFERED=1 SEED=42`
  (`~/gw2_run2x8.sh` = the plain `tinybox_2x8xMI350X/dev_run.sh`, REMOTE far box first, DP=16 BS=32 ALLREDUCE_NODE_NDEVS=8). Add
  `LAYERS=4 BENCHMARK=11 JITBEAM=0` for ~8-min iterations (4L baseline: 0.229-0.231 s/update). Logs land in ta4 `~/gptoss-perf/`.
- Progress: `grep -v HCQ2STAT log | grep amortized`. Kill with TERM (finalize releases the NICs). NIC wedge = `HWRM ring_alloc: 4` at the
  first RDMA open: find the NIC (`REMOTE=... DEV=BNXT+RDMA python -c 'from tinygrad import Device; [Device[f"RDMA:{i}"] for i in range(16)]'`,
  0-7 = ta3, 8-15 = ta4 in PCI order 06,16,66,76,86,96,e6,f6), then slot power cycle: `cat /sys/bus/pci/slots/*/address` to find the slot,
  `echo 0 > /sys/bus/pci/slots/N/power; sleep 5; echo 1 > ...`, then gw2_serve.sh again. Foreign users: `pgrep -af model_train`, `who`.
- Analysis scripts (PROFILE=1 pickles): `/home/nimlgen/gptoss-handoff/{timeline,zoom,copyhist,bytes,conc}.py <pkl> "train @ 2" "train @ 4" AMD:1 ...`
  (run with `cd /home/nimlgen/tg-gptoss && PYTHONPATH=.`). Reference traces there: `l4-1x8-prof.pkl` (one box) vs `l4-rdmaq-prof.pkl` (2x8).
  PROFILE=1 distorts the 2x8 wall time (host RPC per timestamp) but the GPU timeline is real.

## 3. Changes in the tree (uncommitted, all small)

| file | what | measured |
|---|---|---|
| `tinygrad/runtime/support/hcq2.py` sched_batches | RDMA sends on `COPY:{num_queues}`, receives on `COPY:{num_queues+1}` (own SDMA engines; separate rings/CQs so the two directions are independent) instead of sharing `COPY:0` with bulk xgmi copies | 4L 0.284 -> 0.241 (queue) -> 0.230 (duplex); 24L 1.25 -> 0.97 |
| `examples/mlperf/model_train.py` | `GPTOSS_JIT_NO_WARMUP=1`: realize the lazy optimizer/scheduler/grad state, then TinyJit captures on the first call (no eager update). Without the realize the LR schedule bakes into the capture. | 24L startup: eager 512 s gone; capture 842 -> 729 s total |
| `extra/gemm/moe_gemm.py` reduce_scatter_devaxis | with ALLREDUCE_NODE_NDEVS: each node sums its 8 contributions at the owner's rank peer in parallel, one NIC hop to the owner (was a chain node0 -> node1 -> owner) | 4L 0.241 -> 0.237 |
| `tinygrad/schedule/multi.py` lower_broadcast_copy | broadcast = one NIC hop to each node's rank peer, then xgmi fan-out. **Required**: the flat broadcast makes cross-rank NIC copies that master stages through the linked GPU -> OOM on AMD:0 at 24L | needed to run at all |
| `tinygrad/runtime/support/am/ip.py` interrupt_handler | read the pending IH ring entries in one slice (remote devices did one RPC per word) | (codex measured 2x on 2L with the same idea) |
| `tinygrad/schedule/__init__.py` | `SCHED_RDMA_DELAY=N` (default 0 = off): a kernel made ready by a NIC receive is emitted N kernels later. **Untested**; the generic variant (delay every copy consumer, `SCHED_COPY_DELAY`) was measured and HURT: 4L 0.241 -> 0.256, 24L 0.905 -> 1.05. Delete if the RDMA-only variant does not help either. **Not deployed.** |
| `extra/gptoss_kernels/embedding/__init__.py` | two-node embedding backward: node-local token gather, owner-reduce for own rows and the rank peer's rows, one 46 MB hop (today: pad each 94 MB token-grad shard to the full 1.5 GB and ALLREDUCE it, i.e. 2x the bytes of one box). **Untested, not deployed.** Test: `launch.sh '~/gptoss-perf/emb_test.sh' emb.log` (compares to the old path and to numpy, prints timings). |

Codex's earlier uncommitted work (bulk PTE checks, fold_words runs, mmap dataloader, profile bulk reads, startup logging, tests) was reverted
on the user's request; the full diff is `/home/nimlgen/gptoss-handoff/tree_full_before_revert.diff` if a piece is wanted back.

## 4. Where the time goes (per update, from the 1x8 vs 2x8 4-layer traces; scale backward items by 24/4)

- Forward: same as one box (~40 ms at 4L, kernels 100% busy).
- Backward: 2x8 kernels ~60% busy vs 80-90% on one box. The compute stream is in order; every per-tensor reduce-scatter ends in an ADD
  of the other node's partial, and that ADD waits for the RDMA receive (transfer 69+36 MB per layer ~2 ms at 48 GB/s, plus ~12 tiny
  bias/norm grads per layer with their own latencies). ~6 ms/layer profiled, ~4 ms unprofiled -> ~0.1 s at 24L.
- Extra reduce kernels: 16-way sums are 2-3 kernels per tensor vs 1 fused 8-input add on one box (+20 ms at 4L).
- Tail (grad-norm barrier -> Adam -> weight all-gather -> next forward): +38 ms vs one box. Vocab/embedding grad chain 26 vs 7 ms
  (section 3 last row addresses it), then the weight all-gather runs ~12 ms with compute idle, reassembly kernels after it.
- RDMA is never bandwidth-bound (1.5 GB send + 1 GB recv per GPU per update); it is latency on the critical path.
- Startup: capture 729 s = python scheduling/lowering of a 2x bigger graph (hcq lower ~70 s per node, sequential), clang -O2 of the
  ~12 MB host C program per node (~130 s, parallel), link over RPC. It scales with the number of collective copies (~2400 per GPU per update).

## 5. What the MLPerf v6.0 submissions do (mlcommons/training_results_v6.0, `*/benchmarks/gpt_oss_20b`)

- All configs at <=16 GPUs are pure DP with a distributed (ZeRO) optimizer, like us. NVIDIA GB200 8 GPU: 0.43 s/step at 2 seq/GPU.
- Reference convergence (rcps_gpt_oss_20b.json): GBS32 -> 234.7k samples (~7.3k steps), LR 8e-4, warmup 4096 samples; GBS16 -> 195k;
  GBS64 -> 302k. The user counts 196608 sequences (16 evals x 12288) -> 6144 steps at BS=32.
- Their tricks: **gradient bucketing** (NVIDIA 768M-element buckets; Primus on MI350X 2x8: `ddp_num_buckets 8`) so there are ~8-30
  collectives per phase instead of ~500; reduce-scatter of each bucket overlapped with backward; **param all-gather overlapped with
  the next forward, bucket by bucket** (`overlap_param_gather`); bf16 grads averaged in the collective; grad clip from local shard
  norms + one scalar all-reduce; no aux loss; EP1 with grouped GEMM. Cisco: `Cisco/benchmarks/gpt_oss_20b/primus/config_MI350X_2x8x1_tp1pp1ep1_gbs64.sh`.

## 6. Suggested order

1. Run the embedding test (section 3), then 24L with it; expect ~-15 ms.
2. Try `SCHED_RDMA_DELAY=16` at 4L then 24L; delete the knob if it does not win clearly.
3. The real fix for the backward stalls is structural: either bucket a layer's gradients into one buffer (one exchange per layer,
   MLPerf style) or move the cross-node ADD out of backward into the optimizer stage (grads stay node-partial during backward; the
   NIC copy is issued during backward but consumed only before clipping). Both are model/optimizer-level (`examples/mlperf/optim.py`,
   `extra/gemm/moe_gemm.py`, the layer backward), not core.
4. Order the post-Adam all-gather layer-0 first and gate the forward per layer (the tail's 12 ms + reassembly).
5. Below ~0.65 s the remaining gap is per-GPU compute (same on one box): kernel work or a different per-GPU batch, which changes the recipe.

Memory notes with more history: `~/.claude/projects/-home-nimlgen-tinygrad/memory/gptoss-2x8-step-anatomy.md`, `project_mi350_multimachine_state.md`.
