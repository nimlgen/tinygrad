#!/usr/bin/env bash
# untimed setup (MLPerf model initialization, <= 30 min), fake data only, the dataset is never touched: one pass that BEAMs, captures
# and saves the full 24-layer 2x8 training and eval graphs to GPTOSS_JIT_PKL(.eval). BEAM times candidates on AMD:0, this box's first gpu,
# opened directly (never through the far box's serve)
set -e
here=$(dirname "$0")
source "$here/common.sh"
export INITMLPERF=1 FAKEDATA=1 SEED=${SEED:-5760} WANDB=${WANDB:-0} BENCHMARK=0 FULL_LAYERS=1 LAYERS=24
# ~half the searched kernels run at the ~10 us launch floor: the first candidate under 12 us ends their search
export BEAM_STOP_US=${BEAM_STOP_US:-12}
export GPTOSS_JIT_SAVE=${GPTOSS_JIT_SAVE:-$GPTOSS_JIT_PKL} GPTOSS_JIT_SAVE_EXIT=${GPTOSS_JIT_SAVE_EXIT:-1}
echo "setup start $(date -Is)"
bash "$here/setup.sh" down "$(hostname)" # this box's gpus are opened directly: no serve here, fresh hive
bash "$here/../tinybox_8xMI350X/dev_beam.sh"
echo "setup end $(date -Is)"
