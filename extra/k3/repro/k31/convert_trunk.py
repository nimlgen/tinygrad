"""Preserve K3.1's 120 DSA indexer tensors and emit metadata k3_rebased understands."""
import json, pathlib, runpy, sys, time

STATE = pathlib.Path('/home/nimlgen/k31-update')
HF = pathlib.Path('/raid/downloads/nimlgen/Kimi-K3.1-Pro')
RAM = pathlib.Path('/home/nimlgen/kimi-prefetch-study/model-ram/k31-update')
FORK = pathlib.Path('/home/nimlgen/k3q/atomic-llama')
SMOKE = '--smoke' in sys.argv
while not SMOKE and not (STATE / 'download-complete.json').exists(): time.sleep(30)
source = json.loads((STATE / 'source-manifest.json').read_text())
config = json.loads((HF / 'config.json').read_text())
dsa = config['text_config']['dsa_config']
assert config['architectures'] == ['KimiK31ForConditionalGeneration']
assert dsa['dsa_layers'] == config['text_config']['linear_attn_config']['full_attn_layers']
assert dsa['index_topk_freq'] == 1 and dsa['index_rope_dim'] == 0 and dsa['index_k_norm_type'] == 'layernorm'

# The old converter recognizes the K3 architecture name. Only this private view
# aliases the name; every source file and the original config are kept intact.
view = STATE / ('converter-smoke' if SMOKE else 'converter-input')
view.mkdir(exist_ok=True)
for entry in source['siblings']:
  name = entry['rfilename']
  if name == 'config.json': continue
  if SMOKE and (name.endswith('.safetensors') or name == 'model.safetensors.index.json'): continue
  p = view / name
  p.parent.mkdir(parents=True, exist_ok=True)
  if not p.exists(): p.symlink_to(HF / name)
config['architectures'] = ['KimiK3ForConditionalGeneration']
config['model_type'] = 'kimi_k3'
(view / 'config.json').write_text(json.dumps(config, indent=2))
if SMOKE:
  import numpy as np
  from safetensors.numpy import save_file
  prefix = 'language_model.model.layers.3.self_attn.indexer.'
  shapes = {'wq_b.weight': (4096, 1536), 'wk.weight': (128, 7168), 'weights_proj.weight': (32, 7168),
            'k_norm.weight': (128,), 'k_norm.bias': (128,)}
  save_file({prefix + name: np.zeros(shape, np.float32) for name, shape in shapes.items()}, str(view / 'model.safetensors'))

sys.path[:0] = [str(FORK), str(FORK / 'gguf-py')]
import gguf
from conversion.kimi_k3 import KimiK3Model

for name in ('INDEXER_K_NORM', 'INDEXER_PROJ', 'INDEXER_ATTN_K', 'INDEXER_ATTN_Q_B'):
  tensor = getattr(gguf.MODEL_TENSOR, name)
  if tensor not in gguf.MODEL_TENSORS[gguf.MODEL_ARCH.KIMI_K3]:
    gguf.MODEL_TENSORS[gguf.MODEL_ARCH.KIMI_K3].append(tensor)

original_filter = KimiK3Model.filter_tensors
def filter_tensors(self, item):
  if '.block_sparse_moe.experts.' in item[0] or item[0].startswith(('vision_tower.', 'mm_projector.')): return None
  return original_filter(item)
KimiK3Model.filter_tensors = filter_tensors

original_params = KimiK3Model.set_gguf_parameters
def set_gguf_parameters(self):
  original_params(self)
  w, prefix = self.gguf_writer, 'kimi-k3.attention.'
  w.add_uint32(prefix + 'indexer.head_count', dsa['index_num_heads'])
  w.add_uint32(prefix + 'indexer.key_length', dsa['index_head_dim'])
  w.add_uint32(prefix + 'indexer.top_k', dsa['index_topk'])
  w.add_uint32(prefix + 'indexer.kpool', 1)
  w.add_float32(prefix + 'layer_norm_epsilon', config['text_config']['rms_norm_eps'])
  w.add_string('source.huggingface.repository', source['id'])
  w.add_string('source.huggingface.revision', source['sha'])
KimiK3Model.set_gguf_parameters = set_gguf_parameters

RAM.mkdir(parents=True, exist_ok=True)
sys.argv = ['convert_hf_to_gguf.py', str(view), '--outtype', 'bf16', '--outfile', str(RAM / ('smoke-trunk-bf16.gguf' if SMOKE else 'trunk-bf16.gguf'))]
runpy.run_path(str(FORK / 'convert_hf_to_gguf.py'), run_name='__main__')
if not SMOKE: (STATE / 'trunk-convert-complete.json').write_text(json.dumps({'revision': source['sha']}))
