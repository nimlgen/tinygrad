#!/usr/bin/env bash
# usage: setup.sh up|down [node ...]   (default: this node; run `setup.sh up` on every node in NODES before run_and_time.sh)
# every node's gpus sit behind extra/remote/serve.py 6667 for the 2x8 run; a hive reset separates owners (AM warm starts fault)
# up: stop the serve, hive reset, start the serve. down: stop the serve, hive reset (the gpus are then free for a local process)
set -e
unset REMOTE # the serve and the reset own this box's gpus directly; with REMOTE set they would probe through the other serves
action=$1; shift
nodes=${*:-$(hostname)}
repo=$(pwd)
node_cmd() { cat <<CMD
cd $repo
unset REMOTE
pkill -TERM -f "extra/remote/serve.py 6667" || true
for _ in {1..100}; do pgrep -f "extra/remote/serve.py 6667" > /dev/null || break; sleep 0.2; done
pkill -KILL -f "extra/remote/serve.py 6667" || true; sleep 1
sync; sudo -n sysctl -q vm.drop_caches=3 || echo "\$(hostname): page cache not flushed" # the rules: flush the cache before benchmarking
# a stopped serve can hold the gpu locks for a few more seconds
for _ in {1..5}; do PYTHONPATH=. DEV=PCI+AMD timeout 600 \${SERVE_PYTHON:-python3} extra/amdpci/hive_reset.py > /tmp/hive_reset.log 2>&1 && break; sleep 10; done
tail -2 /tmp/hive_reset.log; sleep 5
[ "$action" == up ] || { echo "\$(hostname) down"; exit 0; }
PYTHONPATH=. DEV=PCI+AMD nohup \${SERVE_PYTHON:-python3} -u extra/remote/serve.py 6667 > \$HOME/serve_gw2.log 2>&1 < /dev/null &
for _ in {1..100}; do grep -q "listening on" \$HOME/serve_gw2.log && { echo "\$(hostname) serve up"; exit 0; }; sleep 0.2; done
echo "\$(hostname) serve did not start" >&2; exit 1
CMD
}
for n in $nodes; do
  if [ "$n" == "$(hostname)" ]; then node_cmd | bash & else node_cmd | ssh "$n" bash & fi
done
wait
