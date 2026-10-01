# production 1x8 MI350X configuration: the 2x8 flags on one box (its 8 gpus opened directly), GBS 16 = 2 sequences per gpu
# GBS 16: the v6.0 reference and NeMo recipe (peak LR 4e-4, 128-update warmup), end LR 0.1x per the rules
export LR=${LR:-0.0004} END_LR=${END_LR:-0.00004} WARMUP_STEPS=${WARMUP_STEPS:-128} EVAL_BS=${EVAL_BS:-16}
source "$(dirname "${BASH_SOURCE[0]}")/../tinybox_2x8xMI350X/common.sh"
unset REMOTE
export NODES=$(hostname) DP=8 BS=16
# the lazy grads/moments realize in 16 GB batches: at DP=8 their ZeRO shards' full replicated sources (~270 GB/gpu together) OOM
export GPTOSS_LAZY_REALIZE_GB=${GPTOSS_LAZY_REALIZE_GB:-16}
