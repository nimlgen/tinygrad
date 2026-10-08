# 2x8 MI350X: this box's gpus opened directly ("local", AMD:0-7), the other box behind its serve.py (`setup.sh up` there, AMD:8-15)
export NODES=${NODES:-"tinyamd3 tinyamd4"}
export REMOTE=${REMOTE:-"local,$(echo $NODES | tr ' ' '\n' | grep -vx "$(hostname)" | head -1):6667"}
export DEV=PCI+AMD REMOTE_TIMEOUT=600 ALLREDUCE_NODE_NDEVS=8 DP=16 BS=${BS:-32} EVAL_BS=${EVAL_BS:-16}
# GBS 32: the batch the 2x8 runs converged with (184320 samples)
export GRADIENT_ACC_STEPS=${GRADIENT_ACC_STEPS:-1}
# the 1x8 dev_run recipe (same GBS 32): peak LR 1e-3 with a 2048-sample warmup, end LR 0.1x per the rules (our runs: 3.3 at 184k)
export LR=${LR:-1e-3} END_LR=${END_LR:-1e-4} WARMUP_SAMPLES=${WARMUP_SAMPLES:-2048}
# updates per captured graph (2 was no faster on 2x8 and doubles the capture in init)
export LLAMA_STEP_GROUP=${LLAMA_STEP_GROUP:-1}
# BEAM: a candidate at the ~10 us launch floor ends its search
export BEAM_STOP_US=${BEAM_STOP_US:-12}
# collectives: node-leader small-message allreduce, duplex rdma with receives posted ahead, copy destinations held ahead of reuse
export ALLREDUCE_NODE_LEADER=${ALLREDUCE_NODE_LEADER:-1} RDMA_DUPLEX=${RDMA_DUPLEX:-1} RDMA_POST_AHEAD=${RDMA_POST_AHEAD:-1} COPY_PREHOLD=${COPY_PREHOLD:-1000}
# the shuffled training index covers 300k samples (well past convergence): builds in seconds for a fresh data seed
export SAMPLES=${SAMPLES:-300000}
