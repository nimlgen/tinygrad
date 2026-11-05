#!/bin/bash
# the engine serving openai chat on :8100 for terminal-bench (q3 experts, fast + TAUC=0, 32k context), then a long prompt to capture the chunk jits
lsmod | grep -q "^amdgpu " && { echo "SERVER FAILED: amdgpu is loaded (sudo rmmod amdgpu)"; exit 1; }
cd ~/k3 && TG=$HOME/k3/tg-clean TMO=0 bash m3run.sh tb_serve - $EXTRA SLOTS=${SLOTS:-4800} CHUNKS=${CHUNKS:-256} DRAFT=$HOME/k3/eagle TAUC=0 THINK=off CTX=32768 MAXTOK=4096 SERVE=8100 &
until curl -sf http://127.0.0.1:8100/v1/models > /dev/null; do sleep 10; grep -qE "Traceback|Error" ~/k3/m3_tb_serve.log 2>/dev/null && { echo SERVER FAILED; exit 1; }; done
python3 -c "
import json; t=open(\"/home/nimlgen/k3/prompt_1k.txt\").read()
print(json.dumps({\"model\":\"k3\",\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":t+t[:2000]+\"\n\nSummarize the above in one sentence.\"}]}))" > /tmp/tb_warm.json
curl -s http://127.0.0.1:8100/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/tb_warm.json | cut -c1-300; echo
echo SERVER READY
