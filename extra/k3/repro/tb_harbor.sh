#!/bin/bash
# tb_harbor.sh job_name [harbor args]: terminus-2 on the engine serving on :8100
export PATH=$HOME/.local/bin:$PATH OPENAI_API_KEY=x
cd ~/k3/tb && name=$1 && shift
harbor run -y -q -p terminal-bench -a terminus-2 -m openai/k3 -n 1 --ak api_base=http://127.0.0.1:8100/v1 --ak temperature=0 --ak "llm_call_kwargs={\"timeout\":3600}" \
  --ak "model_info={\"max_input_tokens\":32768,\"max_output_tokens\":4096,\"input_cost_per_token\":0,\"output_cost_per_token\":0}" \
  --job-name $name -o jobs "$@" > $name.out 2>&1
echo "$name DONE"
