import pathlib, tempfile, unittest
import numpy as np
from gguf import GGUFWriter
from tinygrad import Tensor, nn
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig, IndexerConfig, MLATransformerBlock, GatedDeltaNetBlock, ExpertGating

def write_kimi_gguf(path:pathlib.Path, arch:str, indexer:bool=False, experts:bool=True):
  # two blocks: a dense kda block, then an mla block with routed experts
  config = TransformerConfig(num_blocks=2, dim=16, hidden_dim=8, n_heads=2, n_kv_heads=1, norm_eps=1e-5, vocab_size=16, head_dim=12,
    rope_theta=10000.0, rope_dim=4, v_head_dim=8, max_context=8, num_experts=4, num_experts_per_tok=2, expert_gating_func=ExpertGating.SIGMOID,
    q_lora_rank=8, kv_lora_rank=8, shared_expert_dim=8, ssm_layers=(True, False), attn_output_gate=True, leading_dense_blocks=1,
    dense_hidden_dim=16, expert_bias=True, shared_expert_gate=False, expert_latent_dim=8, attn_res_block_size=1,
    ssm=SSMConfig(conv_kernel=2, state_size=8, group_count=2, time_step_rank=2, inner_size=16, kda=True, gate_lower_bound=-5.0,
                  full_rank=(True, False)),
    indexer=IndexerConfig(n_heads=2, head_dim=8, top_k=4) if indexer else None)
  writer = GGUFWriter(path, arch)
  for key, value in {'context_length': 8, 'embedding_length': 16, 'feed_forward_length': 16, 'expert_feed_forward_length': 8, 'block_count': 2,
                     'attention.head_count': 2, 'ssm.conv_kernel': 2, 'kda.head_dim': 8, 'attention.q_lora_rank': 8, 'attention.kv_lora_rank': 8,
                     'rope.dimension_count': 4, 'attention.key_length_mla': 12, 'attention.value_length_mla': 8, 'expert_count': 4,
                     'expert_used_count': 2, 'expert_shared_count': 1, 'leading_dense_block_count': 1, 'expert_gating_func': 2,
                     'expert_latent_length': 8, 'attn_res.block_size': 1, **({'attention.indexer.head_count': 2, 'attention.indexer.key_length': 8,
                     'attention.indexer.top_k': 4} if indexer else {})}.items():
    writer.add_uint32(f'{arch}.{key}', value)
  writer.add_array(f'{arch}.attention.head_count_kv', [0, 1])
  writer.add_float32(f'{arch}.attention.layer_norm_rms_epsilon', 1e-5)
  writer.add_float32(f'{arch}.kda.gate_lower_bound', -5.0)
  writer.add_array('tokenizer.ggml.tokens', [str(i) for i in range(16)])
  for name, weight in nn.state.get_state_dict(Transformer(config)).items():
    if not experts and '_exps.' in name: continue
    writer.add_tensor(name.replace('blk.0.ssm_out', 'blk.0.attn_output'), np.zeros(weight.shape, dtype=np.float32))
  writer.write_header_to_file()
  writer.write_kv_data_to_file()
  writer.write_tensors_to_file()
  writer.close()

class TestMLA(unittest.TestCase):
  def _make_config(self, **kwargs):
    return TransformerConfig(**{
      "num_blocks": 1, "dim": 64, "hidden_dim": 128, "n_heads": 4, "n_kv_heads": 1,
      "norm_eps": 1e-5, "vocab_size": 100, "head_dim": 16, "rope_theta": 10000.0, "rope_dim": 8, "max_context": 32,
      "kv_lora_rank": 16, "v_head_dim": 8,
    } | kwargs)

  def test_shared_expert_gate_optional(self):
    model = Transformer(self._make_config(num_experts=4, num_experts_per_tok=2, shared_expert_dim=32, shared_expert_gate=False))
    self.assertNotIn('blk.0.ffn_gate_inp_shexp.weight', nn.state.get_state_dict(model))
    out = model.blk[0]._feed_forward(Tensor.randn(1, 4, model.blk[0].config.dim))
    self.assertEqual(out.shape, (1, 4, model.blk[0].config.dim))

  def test_kimi_k31_indexers(self):
    with tempfile.TemporaryDirectory() as folder:
      write_kimi_gguf(path:=pathlib.Path(folder)/'model.gguf', 'kimi-k31', indexer=True)
      model = Transformer.from_gguf(path)[0]
    self.assertIsInstance(model.blk[0], GatedDeltaNetBlock)
    self.assertIsInstance(model.blk[1], MLATransformerBlock)
    self.assertFalse(hasattr(model.blk[0], 'indexer'))
    self.assertIsNotNone(model.blk[1].indexer)
    self.assertFalse(model.config.streamed_experts)

  def test_kimi_without_routed_experts(self):
    with tempfile.TemporaryDirectory() as folder:
      write_kimi_gguf(path:=pathlib.Path(folder)/'model.gguf', 'kimi-k3', experts=False)
      model = Transformer.from_gguf(path)[0]
    self.assertTrue(model.config.streamed_experts)
    self.assertFalse(any('_exps.' in name for name in nn.state.get_state_dict(model)))
    with self.assertRaises(AssertionError): model.blk[1]._feed_forward(Tensor.empty(1, 3, 16))  # (an engine streams them in)

if __name__ == '__main__':
  unittest.main()
