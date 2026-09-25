#!/usr/bin/env bash
# two boxes, a serve.py on each: PYTHONPATH=. DEV=PCI+AMD python extra/remote/serve.py 6667. run this on either, the far one goes first
export NODES=${NODES:-"tinyamd3 tinyamd4"}
export REMOTE=${REMOTE:-"$(echo $NODES | tr ' ' '\n' | grep -vx "$(hostname)" | head -1):6667,127.0.0.1:6667"}
export DEV=PCI+AMD REMOTE_TIMEOUT=600 ALLREDUCE_NODE_NDEVS=8 DP=16 BS=32 EVAL_BS=${EVAL_BS:-16}
# Preserve the small-message collective used by the earlier fast DP16 runs.
export ALLREDUCE_NODE_LEADER=${ALLREDUCE_NODE_LEADER:-1}
# HPE peak LR; retain the original warmup/end LR and required 12,288-sequence evaluation cadence.
export LR=${LR:-0.0005} END_LR=${END_LR:-0.00004} WARMUP_STEPS=${WARMUP_STEPS:-128}
export EVAL_FREQ=${EVAL_FREQ:-12288} CKPT=${CKPT:-0}
exec bash "$(dirname "$0")/../tinybox_8xMI350X/dev_run.sh"
