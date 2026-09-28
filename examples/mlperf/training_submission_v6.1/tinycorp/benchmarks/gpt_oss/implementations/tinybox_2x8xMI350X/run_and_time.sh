#!/usr/bin/env bash
# MLPerf gpt-oss 20B on 2x8 MI350X from empty caches, init in two processes: dev_beam.sh (init_start, fake data only: BEAM, capture and
# save the training and eval graphs) then dev_run.sh (load and link them, init_stop/run_start, train on the dataset to the target).
# Run from the repo root on one of the NODES (common.sh) after `setup.sh up` on the other node (its serve.py); this node needs none.
set -e
set -o pipefail
export PYTHONPATH="."
here=$(dirname "$0")
source "$here/common.sh"

export SEED=${SEED:-$RANDOM}
export DATA_SEED=${DATA_SEED:-$SEED}
DATETIME=$(date "+%m%d%H%M")
LOGFILE="gpt_oss_2x8xMI350X_${DATETIME}_${SEED}.log"

# empty caches: every kernel is BEAM-searched and compiled from scratch inside the measured init
export CACHEDB=$(mktemp -d)/cache.db
export AMD_COMGR_CACHE_DIR=$(dirname "$CACHEDB")/comgr GPTOSS_JIT_PKL=$(dirname "$CACHEDB")/gptoss_jit.pkl
echo "cache $CACHEDB, seed $SEED, log $LOGFILE" | tee "$LOGFILE"

bash "$here/dev_beam.sh" 2>&1 | tee -a "$LOGFILE"
bash "$here/dev_run.sh" 2>&1 | tee -a "$LOGFILE"
