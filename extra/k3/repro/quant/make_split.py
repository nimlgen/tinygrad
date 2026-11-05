# turn a trunk-only gguf into shard 1 of a 2-shard split model whose shard 2 is a shared expert file
import sys, os
FORK = "/home/nimlgen/k3q/atomic-llama"
sys.path.insert(0, f"{FORK}/gguf-py")
import gguf
from gguf.scripts.gguf_new_metadata import copy_with_new_metadata, MetadataDetails
trunk, experts, outdir = sys.argv[1:4]
n_experts = len(gguf.GGUFReader(experts).tensors)
reader = gguf.GGUFReader(trunk)
n_total = len(reader.tensors) + n_experts
os.makedirs(outdir, exist_ok=True)
S = gguf.Keys.Split
new = {S.LLM_KV_SPLIT_NO: MetadataDetails(gguf.GGUFValueType.UINT16, 0), S.LLM_KV_SPLIT_COUNT: MetadataDetails(gguf.GGUFValueType.UINT16, 2),
       S.LLM_KV_SPLIT_TENSORS_COUNT: MetadataDetails(gguf.GGUFValueType.INT32, n_total)}
writer = gguf.GGUFWriter(f"{outdir}/K3-00001-of-00002.gguf", arch=reader.fields["general.architecture"].contents(), endianess=reader.endianess)
copy_with_new_metadata(reader, writer, new, [])
dst = f"{outdir}/K3-00002-of-00002.gguf"
if not os.path.exists(dst): os.symlink(os.path.abspath(experts), dst)
print(f"{outdir}: {len(reader.tensors)} trunk + {n_experts} expert tensors = {n_total}")
