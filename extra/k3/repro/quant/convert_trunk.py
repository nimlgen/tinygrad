# run the fork's hf->gguf converter for everything except the routed experts (those are streamed by quant_experts.py)
import sys, runpy
FORK = "/home/nimlgen/k3q/atomic-llama"
sys.path[:0] = [FORK, f"{FORK}/gguf-py"]
import conversion.kimi_k3 as k3m
orig = k3m.KimiK3Model.modify_tensors
def modify_tensors(self, data_torch, name, bid):
  if ".block_sparse_moe.experts." in name: return
  yield from orig(self, data_torch, name, bid)
k3m.KimiK3Model.modify_tensors = modify_tensors
hf, out, outtype = sys.argv[1:4]
sys.argv = ["convert_hf_to_gguf.py", hf, "--outtype", outtype, "--outfile", out]
runpy.run_path(f"{FORK}/convert_hf_to_gguf.py", run_name="__main__")
