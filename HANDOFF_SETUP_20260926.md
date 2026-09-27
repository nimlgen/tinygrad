# Handoff 2026-09-26: gpt-oss 20B MLPerf on 2x8 MI350X, timed run + rules-compliant setup

Tree: `/home/nimlgen/tg-gptoss` (branch base `80954ba0b 0.617 s/update`), everything below is **uncommitted** and deployed
(rsync, identical) to `~/tg-gw2` on tinyamd3 and tinyamd4. Launchers/logs live in `ta4:~/gptoss-perf/`, my scratch scripts in
`/tmp/claude-30036/-home-nimlgen-tinygrad/ca63ea50-9c82-4357-b725-8103a1896b4b/scratchpad/`.

## Goals and where they stand

| goal | status |
|---|---|
| timed training (run_start -> run_stop) <= 69 min | **69.77 min** best (2026-09-25, seed 1337, G2, eval BS32): converged at 221184 samples (3.3379), 6912 updates x 0.5886 s + 18 evals x 5.1 s. Needs -7 ms/update or a lucky seed |
| full init (setup + everything before run_start) <= 28 min | **not met**. Last complete setup 28.5 min (23:45 run) + the old reload/relink ~3.5 min = ~32. The new single-process flow should land ~27.5-28 min, **never measured at 2x8** |
| setup from an empty cache, fake data only, random SEED and DATA_SEED | implemented in `run_and_time.sh` |

MLPerf accounting facts: HPE's 72.9 min excludes their 5.35 min init (their log: run_start = init_stop). Init = init_start ..
run_start, must be <= 30 min, dataset untouched until run_start. v6.1 compliance keys are all logged (see below).

## How to run (the intended flow)

On ta4 (the "local" box), repo root `~/tg-gw2`, venv `export PATH=$HOME/rdma16/wandb-venv/bin:$PATH`:

1. ta3 must run `setup.sh up` (hive reset + `extra/remote/serve.py 6667`). ta4 needs **no** serve (its gpus are opened directly).
   In practice I used scratchpad `launch.sh rat` (resets both, serves both, 16-NIC preflight, runs `~/gptoss-perf/run-rat.sh`);
   run_and_time then stops ta4's serve itself (`setup.sh down`).
2. `bash examples/mlperf/training_submission_v6.1/tinycorp/benchmarks/gpt_oss/implementations/tinybox_2x8xMI350X/run_and_time.sh`
   = fresh CACHEDB, `setup.sh down $(hostname)`, then **one process**: `INITMLPERF=1 RUNMLPERF=1 GPTOSS_CAPTURE_FIRST=1
   BEAM_STOP_US=12`. It builds the model, BEAMs + captures the training and eval graphs on fake batches, restores the initial
   state, logs init_stop/run_start, then trains on the dataset.
3. Logs: `ta4:~/gptoss-perf/rat.log` (stdout), `~/tg-gw2/gpt_oss_2x8xMI350X_<date>_<seed>.log`, mllog `~/tg-gw2/result_gptoss_<seed>.log`.
   Check with `python -m mlperf_logging.compliance_checker --ruleset 6.1.0 <result log>` (installed only in ta4's wandb-venv).

**Not yet run end to end**: the single-process flow was verified only on ta3 alone at 2 layers (loss restarts at 12.3588 with
warmup LR 7.8e-6 after restore, 12 real updates + eval OK). A 24-layer 1x8 run on ta3 OOMs at model init (ZeRO shards 1/8), so
the full test needs both boxes.

## What changed (files)

Submission scripts `examples/mlperf/training_submission_v6.1/.../tinybox_2x8xMI350X/`:
- `common.sh` (new): every production flag in one place. `REMOTE=local,<far>:6667` (ta4 gpus AMD:0-7 opened directly, ta3
  AMD:8-15 via serve), DP16 BS32 EVAL_BS32, AQL, fixer + `GPTOSS_FWD_INTERLEAVE=1 GPTOSS_FIRST_NEED=1`, ref WD, LR 5e-4 /
  END_LR 5e-5 / warmup 128, `SAMPLES=300000` (index for a fresh data seed builds in 4.7 s instead of 262 s for 38.4M),
  `GPTOSS_OWNED_EXPERT_GATHER=1`, `DEBUG=0`, `LOGMLPERF=1`.
- `run_and_time.sh` (new): the single-process flow above; `SEED=$RANDOM`, `DATA_SEED=$SEED` (rules: rng seeds must come from
  clock/urandom; llama's run_and_time does the same).
- `setup.sh` (new): `setup.sh up|down [node]` = stop serve, hive reset, (start serve). Runs with REMOTE unset (a serve started
  with REMOTE set probes itself and times out: that broke one run). ta4 cannot ssh to ta3 (no key), so each node runs its own.
- `dev_beam.sh` / `dev_run.sh`: the older two-process flow (pickle save/load via `GPTOSS_JIT_SAVE/LOAD`); still works, but the
  gap between the processes lets other jobs take ta4 (see pitfalls).

`examples/mlperf/model_train.py` (train_gptoss):
- mllog: INITMLPERF -> submission events, `diskcache_clear` + cache_clear, init_start (skipped with `INIT_RESUME`); RUNMLPERF ->
  init_stop immediately before run_start, all hyperparameters, v6.1 system keys (lowest precision fp8/bf16/fp8, TP/PP/CP/EP=1,
  micro batch 2, config filename), block/eval/epoch start-stop, eval_accuracy, run_stop. `run_start` sits right before the first
  dataset access (eval dataset open).
- `GPTOSS_CAPTURE_FIRST=1`: snapshot `jit_state()` (clone), capture `train_group` on fake batches, drain deferred, eval warmup +
  capture, restore (one realize of assigns), sync; `group_calls=1` so the loop replays.
- eval capture is **warmup + capture**: a first-call capture (`eval_step.cnt=1`) hung its first replay ("Device hang detected")
  on 2x8. `GPTOSS_JIT_SAVE_EVAL` / `GPTOSS_JIT_LOAD_EVAL` for eval-only pickles (eval BS32: 5.1 s/eval vs 6.3, loss identical).
- training data: `batch_load_llama3_pool` (dataloader.py) with a spawn Pool of 8 readers created at the top of train_gptoss
  (before any device opens; fork would share DMA memory). Cold page cache after a reboot made synchronous reads 0.14-0.17 s per
  update; pool tested on ta3: 0.14 ms wait, 99% main-thread python throughput. **Not yet exercised in a 2x8 timed run.**

`tinygrad/runtime/support/system.py`: a `local` entry in REMOTE = this host's own PCI devices opened directly, in that position.

`tinygrad/codegen/opt/search.py`: `BEAM_STOP_US` (first candidate under it is taken and the search ends; ~half the searched
kernels run at the ~10 us launch floor; measured: BEAM ends at 11.8 min instead of ~17 in the 24L capture),
`BEAM_CACHE_ONLY` (cache hit or hand-coded, never search; used by dev_run), `BEAM_LOG_MISSES`.

`examples/mlperf/gptoss_sched_fixer.py`: `GPTOSS_FWD_INTERLEAVE=1` (next forward list-scheduled right after the gather region
starts). Emulator -9 ms/update, hardware fake-data A/B -1.5..-1.9 ms/update. Emulator handoff: `~/gptoss-perf/emu/HANDOFF_CODEX.md`.

Folded in from out-of-tree wrappers (ta4 `~/gptoss-perf/run_deferred_*.py`): two-node trimmed expert-grad reduce-scatter as the
first case of `moe_gemm.reduce_scatter_devaxis`, owned expert gather in `optim._gptoss_gather_owned`, the two-node embedding
backward in `extra/gptoss_kernels/embedding/__init__.py`. Earlier in the week (see memory files): node-paired wgrad shards,
pre-padded attention weights, HIP routing metadata, rdma rings 4096, vocab Adam WD, ref WD, jit pickle cache.

## Measured setup timeline (single-pass 2x8, 23:45 run, BEAM_STOP_US=12, fresh cache)

| phase | time |
|---|---|
| model init (both boxes) | 2.2 min |
| BEAM of the capture (ends when `gptoss_sched_fixer:` prints) | ~9.6 min |
| capture lowering + link + first run of the 143720-call graph | **~13.5-14 min (largest, not profiled)** |
| training pickle | 29 s (gone in the single-process flow) |
| eval warmup + capture | 73 s |
| old flow only: timed-run hive reset + init + load 19 s + link 93 s + eval link 4 s | ~3.5 min (gone in the single-process flow) |

BEAM profile (py-spy, ta3): 76% of BEAM is `time_call` doing a full hcq2 compile (41%) + link (30%) of the one-kernel launch per
candidate; the GPU wait is 1.2%. Caching it does not hit (this tree has no program templating: each candidate's kernel binary
is part of the key). Threads per GPU failed on driver thread-safety (shared VA allocator). Both reverted.

## Next steps (in order)

1. Measure the single-process flow at 2x8 (one launch; expect ~27.5-28 min init). Profile its capture phase: scratchpad-free
   helper `ta4:~/gptoss-perf/prof-setup.sh` (py-spy until "captured and restored"). Run it right after launching.
2. Cut the ~14 min capture lowering. The user's pointer: hcq2 already supports ranges (`END(LINEAR(body), RANGE(n))`, commit
   a98983422 "hcq2 range"): a block is encoded/lowered once and repeated with per-trip patched words. The 24 layers unroll into
   24 identical call blocks differing only by weight offsets (per-layer weights are stacked `(n_layers, ...)`), so rolling them
   into one ranged block would cut lowering ~24x. Needs either tracing one layer with a symbolic layer index or a pass that rolls
   repeated call blocks. Confirm with the profile first.
3. BEAM_STOP_US 12 -> 15 (probably -1 min, costs at most a few us per floor-level kernel).
4. Eval: find why a first-call eval capture hangs its first replay (would save ~1 min of eager warmup).
5. Step time for 69 min: the timed run needs ~0.581 s/update at 221184 convergence (all GBS32 submissions converge at 233472
   except 1 of 12 at 221184); interleave (-1.5 ms) is on; next candidates are in `~/gptoss-perf/emu/HANDOFF_CODEX.md`.

## Pitfalls (each cost hours today)

- **woze has priority on ta4**. His codex watchers (`~woze/tinygrad17/.codex_artifacts/*/run.py watch`, `--interval 10`) take
  the gpus in any gap. Two of our runs died with `Failed to acquire lock file am_0000:05:00.0.lock` in the ~30 s between our
  setup and timed-run processes -> the single-process flow. Never restart our run if he took the gpus; launch by hand when the
  locks are free (watch with a Monitor on `sudo lsof /tmp/am_*.lock`). As of 00:56 he started a new one: `hcq_measured_replay_20260926`.
- Killing a process that drives gpus directly left all 8 ta4 gpus failing `discovery signatures mismatch` (hive_reset cannot
  open them) -> BMC power cycle of both boxes (`ipmitool -I lanplus -H 192.168.52.137|.167 ... power reset`). Stop runs with
  SIGINT (python finalizes), not SIGKILL. After a reboot ta3 comes up with amdgpu bound: `sudo rmmod amdgpu` (kimi container is
  stopped, restart policy `no`); ta4's clock is ~7 h off after the reboot (no NTP).
- Killed runs can leave ta3's python crashing with corrupted memory right after start; a second hive reset fixed it each time.
- `BEAM` "HANGDUMP ... waiting for X" lines are candidate timeouts, not hangs.
- With DEBUG=2 the 2-layer local run reports ~0.5 s/update; with DEBUG=0 it is 0.09 s.
- Deploy with rsync (atomic rename), never scp over a bash script that is running.
