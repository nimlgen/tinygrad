#!/usr/bin/env bash
# MLPerf llama 3.1 8B on 2x8 MI350X, run from the repo root on one node of NODES after `setup.sh up <other node>` there.
set -e
set -o pipefail
export PYTHONPATH="."
here=$(dirname "$0")
source "$here/common.sh"
[ -n "$VENV" ] && export PATH="$VENV/bin:$PATH" # a python with mlperf_logging

export SEED=${SEED:-$RANDOM}
export DATA_SEED=${DATA_SEED:-$SEED}
export LOGMLPERF=1 SUBMISSION_PLATFORM=${SUBMISSION_PLATFORM:-tinybox_2x8xMI350X} PYTHONUNBUFFERED=1
LOGFILE="llama31_8b_2x8xMI350X_$(date "+%m%d%H%M")_${SEED}.log"
# empty caches: every kernel is BEAM-searched and compiled from scratch in the init (not the machine's shared cache)
export CACHEDB=${CACHEDB:-$(mktemp -d)/cache.db}
export AMD_COMGR_CACHE_DIR=$(dirname "$CACHEDB")/comgr
echo "cache $CACHEDB, seed $SEED, log $LOGFILE" | tee "$LOGFILE"

bash "$here/setup.sh" down
# one process: init (empty caches: BEAM, compile, the training and eval graphs captured on fake batches, the initial state restored, <= 30 min)
# then run_start, the dataset, training to the target
INITMLPERF=1 RUNMLPERF=1 PRE_RUN_CAPTURE=1 LLAMA_JIT_NO_WARMUP=${LLAMA_JIT_NO_WARMUP:-1} \
  bash "$here/../tinybox_8xMI350X/dev_run.sh" 2>&1 | tee -a "$LOGFILE"
