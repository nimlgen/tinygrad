#!/usr/bin/env bash
set -e  # Exit on any error
set -o pipefail  # Make pipeline fail if any command fails

export PYTHONPATH="."
export PATH="/opt/rocm-7.1.1/bin:$PATH"
export ROCM_PATH="/opt/rocm-7.1.1"
# two boxes: a serve.py on each (PYTHONPATH=. DEV=PCI+AMD python extra/remote/serve.py 6667), this runs on the first; the gpus of
# both are AMD:0-15 in REMOTE order, copies between the nodes go over the nics, the gradient allreduce exchanges between linked gpus
export REMOTE="127.0.0.1:6667,192.168.52.153:6667" RDMA=1 REMOTE_TIMEOUT=600
export DEV=PCI+AMD
export CHECK_OOB=0
export REWRITE_STACK_LIMIT=5000000 HCQDEV_WAIT_TIMEOUT_MS=240000
export DEVICE_IN_FUNCTION_BUG=1

export HK_FLASH_ATTENTION=1
export ALL2ALL=1
export LATE_ALLREDUCE=0
export USE_ATOMICS=1
export ASM_GEMM=1
export WQKV=1
export MASTER_WEIGHTS=1
export FP8=1
export ALLREDUCE_CAST=1
export FAST_CE=1
export FUSED_INPUT_QUANTIZE=1
export FUSED_GRAD_QUANTIZE=1
export FUSED_ADD_NORM_MUL_QUANTIZE=1
export FUSED_SILU_W13=1
export SPLIT_W13=0

export DEFAULT_FLOAT="bfloat16" OPTIM_DTYPE="bfloat16"
export DP=16 MP=1 BS=16 EVAL_BS=16 GRADIENT_ACC_STEPS=2
export GBS=$((BS * GRADIENT_ACC_STEPS))

export MODEL="llama3"
export BASEDIR="/raid/datasets/c4-8b/"
export SMALL=1
export LLAMA3_SIZE=8B
export EVAL_TARGET=3.3 EVAL_FREQ=12288
export LR="1e-3" END_LR="1e-4" WARMUP_SAMPLES=4096 MAX_STEPS=1200000
export WARMUP_STEPS=$((WARMUP_SAMPLES / GBS))
export SAMPLES=$((MAX_STEPS * GBS))
export SEQLEN=8192

export SEED=$RANDOM
export DATA_SEED=$SEED

export JITBEAM=3
export BEAM_UOPS_MAX=6000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5 BEAM_PADTO=1

export LOGMLPERF=1

DATETIME=$(date "+%m%d%H%M")
LOGFILE="llama31_8b_2x8xMI350x_${DATETIME}_${SEED}.log"

# beam
FAKEDATA=1 BENCHMARK=10 INITMLPERF=1 LLAMA_LAYERS=2 python3 examples/mlperf/model_train.py | tee "$LOGFILE"

# run
RUNMLPERF=1 python3 examples/mlperf/model_train.py | tee -a "$LOGFILE"
