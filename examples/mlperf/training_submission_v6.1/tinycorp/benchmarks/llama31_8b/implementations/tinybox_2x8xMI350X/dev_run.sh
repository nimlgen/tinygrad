#!/usr/bin/env bash
set -euo pipefail

# Preserve the v6.0 MXFP4 recipe's global batch and optimizer schedule on two nodes.
export REMOTE=${REMOTE:-"127.0.0.1:6667,192.168.52.153:6667"} RDMA=${RDMA:-1} REMOTE_TIMEOUT=${REMOTE_TIMEOUT:-600}
export DEV=${DEV:-PCI+AMD} HCQ2=1
export DP=${DP:-16} MP=${MP:-1} BS=${BS:-32} EVAL_BS=${EVAL_BS:-16} GRADIENT_ACC_STEPS=${GRADIENT_ACC_STEPS:-1}
exec bash examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/llama31_8b/implementations/tinybox_8xMI350X/dev_run.sh
