#!/usr/bin/env bash
set -eo pipefail

# Tune matching per-GPU shapes on eight local GPUs, using the same real-data recipe.
unset REMOTE
export RDMA=0 DEV=${DEV:-PCI+AMD} HCQ2=1
export DP=${DP:-8} MP=${MP:-1} BS=${BS:-16} GRADIENT_ACC_STEPS=${GRADIENT_ACC_STEPS:-2}
export FAKEDATA=${FAKEDATA:-0} BENCHMARK=${BENCHMARK:-3} WANDB=0
exec bash examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/llama31_8b/implementations/tinybox_8xMI350X/dev_beam.sh
