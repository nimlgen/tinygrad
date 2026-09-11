#!/usr/bin/env bash

export PYTHONPATH="."
export PATH="/opt/rocm-7.1.1/bin:$PATH"
export ROCM_PATH="/opt/rocm-7.1.1"
# two boxes: a serve.py on each (PYTHONPATH=. DEV=PCI+AMD python extra/remote/serve.py 6667), this runs on the first; the gpus of
# both are AMD:0-15 in REMOTE order, copies between the nodes go over the nics, the gradient allreduce exchanges between linked gpus
export REMOTE=${REMOTE:-"127.0.0.1:6667,192.168.52.153:6667"} RDMA=${RDMA:-1} REMOTE_TIMEOUT=${REMOTE_TIMEOUT:-600}
export DEV=${DEV:-PCI+AMD}
export CHECK_OOB=0
export REWRITE_STACK_LIMIT=5000000 HCQDEV_WAIT_TIMEOUT_MS=240000
export DEVICE_IN_FUNCTION_BUG=1

export DEBUG=${DEBUG:-0}
export HK_FLASH_ATTENTION=${HK_FLASH_ATTENTION:-1}
export ALL2ALL=${ALL2ALL:-1}
export LATE_ALLREDUCE=${LATE_ALLREDUCE:-0}
export USE_ATOMICS=${USE_ATOMICS:-1}
export ASM_GEMM=${ASM_GEMM:-1}
export WQKV=${WQKV:-1}
export MASTER_WEIGHTS=${MASTER_WEIGHTS:-1}
export FP8=${FP8:-1}
export ALLREDUCE_CAST=${ALLREDUCE_CAST:-1}
export FAST_CE=${FAST_CE:-1}
export FUSED_INPUT_QUANTIZE=${FUSED_INPUT_QUANTIZE:-1}
export FUSED_GRAD_QUANTIZE=${FUSED_GRAD_QUANTIZE:-1}
export FUSED_ADD_NORM_MUL_QUANTIZE=${FUSED_ADD_NORM_MUL_QUANTIZE:-1}
export FUSED_SILU_W13=${FUSED_SILU_W13:-1}
export SPLIT_W13=${SPLIT_W13:-0}
export OFFLOAD_OPTIM=${OFFLOAD_OPTIM:-0}

export DEFAULT_FLOAT="bfloat16" OPTIM_DTYPE="bfloat16"
export DP=${DP:-16} MP=${MP:-1} BS=${BS:-32} EVAL_BS=${EVAL_BS:-16} GRADIENT_ACC_STEPS=${GRADIENT_ACC_STEPS:-2}
export GBS=$((BS * GRADIENT_ACC_STEPS))

export MODEL="llama3"
export BASEDIR="/raid/datasets/c4-8b/"
export SMALL=1
export LLAMA3_SIZE=${LLAMA3_SIZE:-"8B"}
export EVAL_TARGET=3.3 EVAL_FREQ=12288
export LR="1e-3" END_LR="1e-4" WARMUP_SAMPLES=4096 MAX_STEPS=1200000
export WARMUP_STEPS=$((WARMUP_SAMPLES / GBS))
export SAMPLES=$((MAX_STEPS * GBS))
export SEQLEN=${SEQLEN:-8192}

export SEED=${SEED:-$RANDOM}
export DATA_SEED=${DATA_SEED:-5760}

export JITBEAM=${JITBEAM:-3}
export BEAM_UOPS_MAX=6000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5 BEAM_PADTO=1

python3 examples/mlperf/model_train.py
