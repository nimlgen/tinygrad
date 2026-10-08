#!/usr/bin/env bash
# usage: setup.sh up|down [node ...] (default: this node). run `setup.sh up` on the far node before a 2x8 run, `setup.sh down` here
# up: stop the serve, hive reset, start extra/remote/serve.py 6667. down: stop the serve, hive reset (gpus free for a local process)
set -e
unset REMOTE # the serve and the reset own this box's gpus directly
action=$1; shift
nodes=${*:-$(hostname)}
repo=$(pwd)
node_cmd() { cat <<CMD
cd $repo
unset REMOTE
# gpu<->nic p2p stays on the switch: clear ACS request/completion redirect on every bridge
for p in \$(lspci -D -d ::0604 | cut -d" " -f1); do sudo -n setpci -s \$p ECAP_ACS+0x6.w=0000:000c 2>/dev/null || true; done
pkill -TERM -f "extra/remote/serve.py 6667" || true
for _ in {1..100}; do pgrep -f "extra/remote/serve.py 6667" > /dev/null || break; sleep 0.2; done
pkill -KILL -f "extra/remote/serve.py 6667" || true; sleep 1
sync; sudo -n sysctl -q vm.drop_caches=3 || echo "\$(hostname): page cache not flushed" # the rules: flush the cache before benchmarking
# a stopped serve can hold the gpu locks for a few more seconds
ok=0; for _ in {1..5}; do PYTHONPATH=. timeout 600 python3 extra/amdpci/hive_reset.py > /tmp/hive_reset.log 2>&1 && { ok=1; break; }; sleep 10; done
tail -1 /tmp/hive_reset.log; [ \$ok == 1 ] || { echo "\$(hostname) hive reset failed" >&2; exit 1; }
[ "$action" == up ] || { echo "\$(hostname) down"; exit 0; }
PYTHONPATH=. nohup python3 -u extra/remote/serve.py 6667 > \$HOME/serve_6667.log 2>&1 < /dev/null &
for _ in {1..100}; do grep -q "listening on" \$HOME/serve_6667.log && { echo "\$(hostname) serve up"; exit 0; }; sleep 0.2; done
echo "\$(hostname) serve did not start" >&2; exit 1
CMD
}
pids=()
for n in $nodes; do
  if [ "$n" == "$(hostname)" ]; then node_cmd | bash & else node_cmd | ssh "$n" bash & fi
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done # set -e: any node's failure fails the setup
