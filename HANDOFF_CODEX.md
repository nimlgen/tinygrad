# Handoff: gpt-oss 2x8 on tinyamd3 + tinyamd4 (MI350X, tinygrad AM driver, hcq2, AQL)

Written 2026-09-22 22:30 by the previous agent. Goal: gpt-oss training on two boxes (2x8) fast enough to finish the run in < 71 min.
Rule from the user: AMD_AQL=1 only (PM4 is not an acceptable workaround). Everything below runs the AM userspace driver (no kfd/amdgpu).

## 1. State in one paragraph

The multi-node MM fault that blocked every 2x8 run is root-caused and fixed (section 4). The 2x8 trains correctly (AQL, RDMA on,
BENCHMARK=40: 37 jitted updates, loss 12.3 -> 7.8, no faults). It is slow: **6.5-6.9 s/update** vs **0.66 s/update on one box**,
the eager first update takes ~16 min and the two-update JIT compile ~33 min (both dominated by the coordinator's python and one giant
clang compile). The steady-state step is device-side (GPU/NIC), the coordinator only waits. A PROFILE=1 run was in progress when this was
written; its trace tells which device op fills the 6.5 s (section 6). The 71-min target needs roughly **1.2 s/update** on 2x8 and a
JIT compile **< 10 min**.

## 2. Boxes, trees, helpers

| | tinyamd3 (ta3) | tinyamd4 (ta4) |
|---|---|---|
| ssh | `ssh tinyamd3` (192.168.52.209, ProxyJump tinygateway) | `ssh tinyamd4` (192.168.52.153) |
| IPMI | `ipmitool -I lanplus -H 192.168.52.137 -U ADMIN -P RZLWLOKHKI power reset` | `ipmitool -I lanplus -H 192.168.52.167 -U ADMIN -P WHHUMUXTKS power reset` |
| venv | `~/rdma16/aql-test-venv` (py-spy binary at `~/py-spy`) | `~/rdma16/wandb-venv` (has py-spy, wandb creds) |
| tree | `~/tg-gw2` | `~/tg-gw2` (the coordinator runs here) |

- Tree = branch `gptoss_work2_master` (wozeparrot/gptoss_work2 rebased on master 98807603d) + the fixes and debug knobs below.
  Deployed by scp of single files (`git archive` tarball + `~/gw2_swap.sh` for whole-tree swaps). Both boxes have the same files.
- `~/gw2_serve.sh` (both boxes): refuses if a coordinator is connected to :6667 or a model_train runs; unbinds+rebinds every BCM57608
  NIC (revives a NIC left wedged by a killed job), hive reset of all GPUs, starts `extra/remote/serve.py 6667` (log `~/serve_gw2.log`).
  **Run it on both boxes before every 2x8 run.**
- `~/gw2_local.sh` (both): for a *local* run on one box: refuses while a coordinator uses the server, else resets and stops serve.py.
  Never `pkill serve.py` by hand: killing a server under a running coordinator kills the run (happened once).
- `~/gw2_run2x8.sh <log> ENV=..` (ta4): launches the 2x8 via `examples/mlperf/.../gpt_oss/implementations/tinybox_2x8xMI350X/dev_run.sh`
  (REMOTE=tinyamd3:6667,127.0.0.1:6667, DP=16 BS=32, ALLREDUCE_NODE_NDEVS=8). Far box first in REMOTE is load-bearing.
- `~/gw2_run1x8r.sh` (ta4): 1x8 through the local server (REMOTE=127.0.0.1:6667). `~/gw2_run.sh` (ta3): 1x8 local.
- `~/clang_wrap.sh` (ta4): `CC=~/clang_wrap.sh` saves every C source > 1 MB to `/tmp/cc_src_<pid>.c` and logs compile times to
  `~/clang_wrap.log`. Not used yet: use it on the next run to get the giant per-node program sources.
- Launch detached: `ssh -f tinyamd4 'sleep 15; cd ~ && setsid nohup ~/gw2_run2x8.sh x AMD_AQL=1 BENCHMARK=6 JITBEAM=0 AMD_ERR_DUMP=1 > ~/log 2>&1 < /dev/null &'`
- Progress: the per-step lines are tqdm-buffered; grep the `GPTOSS_CAPTURE` json lines (flushed) or `amortized/update` later.
- Kill with TERM (finalize releases the NICs). `pkill -f` patterns must not appear literally in the ssh command line (use `[m]odel_train`).
- NIC wedge symptom: `AssertionError: HWRM ring_alloc: 4` at the first RDMA open. Find the NIC with
  `REMOTE=tinyamd3:6667,127.0.0.1:6667 DEV=BNXT+RDMA python -c 'from tinygrad import Device; [Device[f"RDMA:{i}"] for i in range(16)]'`
  (RDMA:0-7 = ta3 NICs, 8-15 = ta4). Ladder: rebind bnxt_en -> PCI bus reset -> **slot power cycle** (`/sys/bus/pci/slots/N/power`,
  N from `cat /sys/bus/pci/slots/*/address`), which is what fixed ta4 76:00.0 today. Last resort: IPMI power cycle (then `rmmod amdgpu`).
- Foreign users: check `pgrep -af model_train` and `sudo -n lsof /tmp/am_*.lock` before touching a box.

## 3. Measured numbers (AMD_AQL=1 JITBEAM=0, 24-layer gpt-oss, c4 data)

| config | update 1 (eager) | updates 2-3 (2-update JIT capture + compile) | steady s/update |
|---|---|---|---|
| 1x8 local ta3 | 92 s | 108 s | 0.660 |
| 1x8 via REMOTE=127.0.0.1 on ta4 | 105 s | 130 s | 0.659 |
| 2x8 (DP=16 BS=32) before the tracker fix | 1091 s | 3941 s | 6.93 |
| 2x8 after the tracker fix | 984 s | 1944-2247 s | 6.4-6.9 |
| 2x8 with PROFILE=1 | 1106 s | ~4200 s | (trace pending) |

The remote host path costs nothing. Eval on 2x8 runs at ~7 s/batch. Loss curve is healthy.

## 4. The MM fault (fixed) and the other real fixes

- Symptom: deterministic MMHUB fault at VA 0x389800003000 on the replication source GPU ~90 s into `model.shard`, AQL only, 2x8 only.
- Cause: the stuck packet was always a `POLL_REGMEM` on a peer's host runtime-pool slot 0 whose VA was an exact 4 GB boundary
  (0x200500000000; with AM_SYSMEM_VA_ALIGN=1GB it moved to 0x200600000000; polls on non-boundary pools passed; fences and polls at
  +0x100/+0x110 of a boundary pool were fine). The fault VA is firmware garbage. Only the 16-device layout put a pool there.
- Fix: `tinygrad/runtime/support/system.py` sysmem path: after `alloc_vaddr`, while `vaddr & 0xffffffff == 0` allocate again and free
  the skipped ones. Unconditional now (the debug knob AM_SYSMEM_VA_NO4G is gone).
- Other real bugs fixed on the way: AQL scratch growth rewrote the live `amd_queue_t` dispatch ids (write only the scratch fields);
  old AQL scratch buffers freed while in use (keep-alive); bnxt `rcfw()` took async CREQ QP-error events as command responses (skip them);
  PROFILE=1 crashed on MultiBuffer inputs in `track_stats` (getattr).
- Clean branch for upstream: `mi350_sdma_poll_fix` (worktree `/tmp/claude-30036/-home-nimlgen-tinygrad/d3e7b5b8-4c43-4ecc-84b8-ebf22c936c97/scratchpad/wt_fixes`,
  5 commits on origin/master, ruff+mypy clean, `test/backend/test_hcq2.py` on mockgpu passes). Not pushed.
- Main work tree with everything (fixes + debug knobs + HCQ2_STATS + DepsTracker rewrite):
  `/tmp/claude-30036/-home-nimlgen-tinygrad/feb482fc-5b12-4a5f-a4de-815395a61c00/scratchpad/wt_gw2` (branch gptoss_work2_master).

## 5. Where the 2x8 time goes (findings so far)

py-spy profiles and scripts live in `/tmp/claude-30036/-home-nimlgen-tinygrad/d3e7b5b8-4c43-4ecc-84b8-ebf22c936c97/scratchpad/`
(`pyspy_agg.py` aggregates `py-spy record -f raw` files; `pyspy_2x8_*.txt` are the recordings; `tracker_equiv.py`; `prof_analyze.py`).

**Steady state (6.5 s/update):** in the jitted step the coordinator only waits on the devices (`exec_copy` of the metrics ->
`synchronize` -> `_wait_signal`/`on_sleep`/`_collect_interrupts` and timeline reads over RPC). So it is GPU/NIC time. Not yet known
whether it is RDMA transfers, SDMA copies or kernel serialization. Rough bandwidth math says RDMA volume alone should be ~0.1 s.
Watch out: `HCQ_RDMA_NOP=1` + `GPTOSS_ALLOW_NONFINITE=1` gives the same schedule without NIC ops (metrics go non-finite): if the step
drops to ~0.7 s the NIC path is the cost, if not it is the schedule.

**Eager update 1 (16 min):** coordinator python: scheduler rewrites (~30%), hcq2 encode/lower_call (~30%), hcq_link (~7%, of which
a good part is RPC round trips to the far box), remote RPC ~5%. The 1x8 profile has the same shape at 1/10 the time: the 16-device
graph makes every rewrite ~10x slower, not one pathological spot. Largest single lowerings: 1.45M words, 52 s each (HCQ2_STATS).

**JIT compile (33 min without profiling):**
- `DepsTracker.access_resources` was 38% (per-key range lists scanned/pruned per call, quadratic). Rewritten as an interval map in
  `tinygrad/device.py` (commit d48c36465 on gptoss_work2_master, randomized equivalence test in the scratchpad): compile 65 -> 33 min.
- Remaining: hcq2 `lower_call` per node (one call per 8-device node, 3.8M words, ~180 s each; `encode_submit` of the SDMA copy streams
  and the address `substitute` passes at hcq2.py:488/499/510 dominate), then **one `clang -c -O2` per node of the rendered C program,
  35+ min at 100% of one core** (`tinygrad/runtime/support/compiler_cpu.py:21`). This clang is the biggest single item. Candidates:
  -O1/-O0 for huge programs, splitting the per-node program into several functions/objects compiled in parallel, or shrinking it
  (loops instead of unrolled per-device/per-update code). Use `CC=~/clang_wrap.sh` on ta4 to capture the sources first.
- `_is_link_patch` recursion was ~8% (memoized with a WeakKeyDictionary, commit on gptoss_work2_master).
- `HCQ2_STATS=1` prints `HCQ2STAT encode <queue> <dev>: ins= blob= patches= t=` and `HCQ2STAT lower <devices>: words= t=` per call.

## 6. The profiled run (in flight at handoff)

`~/gw2_2x8_prof2.log` on ta4, `AMD_AQL=1 BENCHMARK=6 JITBEAM=0 AMD_ERR_DUMP=1 PROFILE=1 HCQ2_STATS=1`, launched 20:45. Updates 1-6
done by 22:18; it was in the eval phase at 22:25. At process exit tinygrad writes `/tmp/profile.pkl.nimlgen` on ta4 (must be MBs and
newer than 21:00; a 1373-byte file there is from a helper process). Analyze with
`cd wt_gw2 && PYTHONPATH=. python <scratchpad>/prof_analyze.py /path/profile.pkl.nimlgen` (lists the `train @ i` markers), then
`... <pkl> <marker_lo> <marker_hi>` bracketing updates 4-6: per device busy time by op name, kernels vs copies vs rdma, and the busy/span
ratio (idle gaps = waiting on peers/NIC). `python -m tinygrad.viz.cli` can also open it. A background waiter of mine may still be
polling for the exit; ignore it.

## 7. Debug knobs on gptoss_work2_master (all env, default off)

AMD_ERR_DUMP=1 (hw regs, queue params, rings around rptr, page walks, per-hub fault regs, pool dump on error/hang; AMD_ERR_DUMP_FULL=1
whole rings; AMD_ERR_FLUSH; AMD_ERR_PTSCAN), AM_RESERVE_PTABLE, AM_RESERVED_VRAM_MB, AM_SDMA_NO_CTXSW, HCQ_QUEUE_SHIFT,
AM_SYSMEM_VA_ALIGN, AM_SYSMEM_VA_FORCE=va,va (big host pools at given VAs in allocation order), HCQ_RDMA_NOP=1, HCQ_RDMA_OPS=wqe,db,wait,fakewait,
HCQ_RDMA_NIC_BY_RANK=1, BNXT_MR_LOG_PAGE, HCQ2_ADDRSCAN=lo-hi (prints link-time addresses in a range), HCQ2_STATS=1, GPTOSS_ALLOW_NONFINITE=1.
mypy has attr-defined complaints in the debug code only (ops_rdma.py:74, ops_amd.py errdump); the clean branch has none.

## 8. Suggested order for the profiling round

1. Read the PROFILE trace (section 6). If RDMA waits dominate: look at `ops_rdma.rdma_copies` (a WQE per 1 GB chunk, doorbell, CQE wait,
   CQ ack per copy; pairs cabled NIC k <-> NIC k, anchor rule `rdma_nic_for`) and the hierarchical allreduce placement. If SDMA copies
   dominate: copy queue assignment `COPY:{(dst-src-1+shift) % peers % HCQ_NUM_SDMA}` in `hcq2.sched_batches`. If gaps dominate: cross-node
   dependency chains (`_wait_ins`/`_start_ins` in hcq2.py).
2. Confirm with HCQ_RDMA_NOP=1 GPTOSS_ALLOW_NONFINITE=1 BENCHMARK=8 (same schedule, no NIC ops).
3. Compile: capture the per-node C sources with CC=~/clang_wrap.sh, time -O1/-O0 offline, then decide between opt level, splitting and
   shrinking the program. Then the hcq2 lowering passes (HCQ2_STATS lines give sizes/times per call).
4. Eager update 1 (16 min) is the last item; it is the same rewrites as 1x8, just on a 2x bigger graph with superlinear cost.

Memory notes of the previous agent (more detail, chronological): `~/.claude/projects/-home-nimlgen-tinygrad/memory/project_mi350_multimachine_state.md`
and `hcq2-aql-scratch-freed.md` (the fault RCA).
