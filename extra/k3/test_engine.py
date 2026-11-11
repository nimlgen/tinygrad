#!/usr/bin/env python3
# the engine's forward against master's model with the routed experts on a tiny random kimi-k3 on two CPU "gpus": every expert in the pools
# (a slot a code, each gpu its units), the host's part stubbed by exact routing (TAU=0: a token's top k in the store's two parts)
# DEV=CPU PYTHONPATH=. python3 extra/k3/test_engine.py (DEV=AMD: the jitted steps with the host's step_io too)
import pathlib, tempfile, unittest
from types import SimpleNamespace
import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType, quants
from tinygrad import Tensor, Device, nn, dtypes
from tinygrad.llm.kernels.amd import amd_custom_kernels_supported
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig, ExpertGating
from extra.k3.engine import Engine

DIM, LAT, FF, E, K, G, VOCAB = 64, 32, 64, 8, 4, 2, 32
CONFIG = TransformerConfig(num_blocks=3, dim=DIM, hidden_dim=FF, n_heads=2, n_kv_heads=1, norm_eps=1e-5, vocab_size=VOCAB, head_dim=24,
  rope_theta=10000.0, rope_dim=8, v_head_dim=16, max_context=16, num_experts=E, num_experts_per_tok=K, expert_gating_func=ExpertGating.SIGMOID,
  norm_topk_prob=True, q_lora_rank=16, kv_lora_rank=16, shared_expert_dim=FF, ssm_layers=(True, True, False), attn_output_gate=True,
  leading_dense_blocks=1, dense_hidden_dim=64, expert_bias=True, shared_expert_gate=False, expert_latent_dim=LAT, attn_res_block_size=1,
  situ=(4.0, 25.0), ssm=SSMConfig(conv_kernel=2, state_size=16, group_count=2, time_step_rank=2, inner_size=32, kda=True, gate_lower_bound=-5.0,
                                    full_rank_gate=True))

def write(path:pathlib.Path, weights:dict[str, np.ndarray], experts:bool):
  w = GGUFWriter(path, "kimi-k3")
  for key, value in {'context_length': 16, 'embedding_length': DIM, 'feed_forward_length': 64, 'expert_feed_forward_length': FF, 'block_count': 3,
                     'attention.head_count': 2, 'ssm.conv_kernel': 2, 'kda.head_dim': 16, 'attention.q_lora_rank': 16, 'attention.kv_lora_rank': 16,
                     'rope.dimension_count': 8, 'attention.key_length_mla': 24, 'attention.value_length_mla': 16, 'expert_count': E,
                     'expert_used_count': K, 'expert_shared_count': 1, 'leading_dense_block_count': 1, 'expert_gating_func': 2,
                     'expert_latent_length': LAT, 'attn_res.block_size': 1}.items():
    w.add_uint32(f'kimi-k3.{key}', value)
  w.add_array('kimi-k3.attention.head_count_kv', [0, 0, 1])
  w.add_float32('kimi-k3.attention.layer_norm_rms_epsilon', 1e-5)
  w.add_float32('kimi-k3.kda.gate_lower_bound', -5.0)
  w.add_float32('kimi-k3.activation.situ_beta', 4.0)
  w.add_float32('kimi-k3.activation.situ_linear_beta', 25.0)
  w.add_bool('kimi-k3.expert_weights_norm', True)
  w.add_array('tokenizer.ggml.tokens', [str(i) for i in range(VOCAB)])
  for name, x in weights.items():
    if '_exps.' not in name: w.add_tensor(name.replace('ssm_out', 'attn_output'), x)
    elif experts: w.add_tensor(name, quants.quantize(x, GGMLQuantizationType.Q8_0), raw_dtype=GGMLQuantizationType.Q8_0)
  w.write_header_to_file()
  w.write_kv_data_to_file()
  w.write_tensors_to_file()
  w.close()

def parts(weights:dict[str, np.ndarray], li:int, e:int, part:int) -> list[bytes]:
  """each gpu's part of expert e of moe block li as the store has it: its units' gate rows, up rows, then down rows (q8_0), padded"""
  gate, up, down = (quants.quantize(weights[f'blk.{li}.ffn_{p}_exps.weight'][e], GGMLQuantizationType.Q8_0) for p in ('gate', 'up', 'down'))
  u = FF // G
  return [(gate[g*u:(g+1)*u].tobytes() + up[g*u:(g+1)*u].tobytes() + down[:, g*u//32*34:(g+1)*u//32*34].tobytes()).ljust(part, b'\0')
          for g in range(G)]

class TestEngine(unittest.TestCase):
  def setUp(self):
    rng = np.random.default_rng(0)
    weights = {n: (1 + rng.normal(0, 0.1, t.shape) if 'norm' in n or 'res_score' in n else rng.normal(0, 0.3, t.shape)).astype(np.float32)
               for n, t in nn.state.get_state_dict(Transformer(CONFIG)).items()}
    with tempfile.TemporaryDirectory() as folder:
      write(full:=pathlib.Path(folder)/'full.gguf', weights, True)
      write(trunk:=pathlib.Path(folder)/'trunk.gguf', weights, False)
      self.ref, model = Transformer.from_gguf(full)[0], Transformer.from_gguf(trunk, shard=G)[0]
    # every expert in its slot, a code (layer * E + expert) a slot
    part, layers = 4096, [1, 2]
    m = {"n_experts": E, "lat": LAT, "units": [FF // G] * G, "part_bytes": [part] * G, "expert_type": "Q8_0", "block": [32, 34], "layers": layers}
    self.eng = eng = Engine(model, m, slots=len(layers) * E, tokens=2)
    pools = [b''.join(parts(weights, li, e, part)[g] for li in layers for e in range(E)) for g in range(G)]
    eng.pool.assign(Tensor(np.frombuffer(b''.join(pools), np.dtype(eng.word.fmt))).shard(eng.devs, 0)).realize()
    eng.state["slot_of"][:] = eng.state["expert_in"][:] = np.arange(len(layers) * E)

  def want(self, tok:int, pos:int) -> np.ndarray:  # master's model's logits (the engine's forward of it, with the routed experts)
    return Engine.forward(SimpleNamespace(model=self.ref, draft=None), Tensor([[tok]], dtype=dtypes.int32), pos)[0].numpy()[0]

  def stub(self):  # the host's part stubbed: a token's top k by sigmoid + bias, the weights the sigmoids normalized, in the two parts
    eng = self.eng
    tables:dict[int, list[Tensor]] = {}
    def host(li:int, i:int, dep:Tensor) -> Tensor:  # (part 1's dep: the scores)
      if i == 0:
        sig = dep[:, 0].to("CPU").sigmoid()
        sel = (sig + Tensor(eng.bias[li], device="CPU")).topk(K)[1]
        w = sig.gather(-1, sel) / sig.gather(-1, sel).sum(-1, keepdim=True)
        tables[li] = [Tensor([1], dtype=dtypes.uint32, device="CPU").cat((sel[:, ks].T.flatten() + li * E).cast(dtypes.int32).bitcast(dtypes.uint32),
                                                           w[:, ks].T.flatten().cast(dtypes.float32).bitcast(dtypes.uint32)).shard(eng.devs)
                      for ks in (slice(0, eng.ks[0]), slice(eng.ks[0], K))]
      return tables[li][i]
    eng.host = host  # type: ignore[method-assign]

  def test_forward(self):
    self.stub()
    tok = 3
    for pos in range(6):
      ours, want = self.eng.forward(Tensor([[tok]], dtype=dtypes.int32), pos)[0].numpy()[0], self.want(tok, pos)
      np.testing.assert_allclose(ours, want, rtol=1e-2, atol=1e-2 * np.abs(want).max(), err_msg=f"position {pos}")
      self.assertEqual(ours.argmax(), want.argmax())
      tok = int(want.argmax())

  def test_verify(self):  # a step of [the next, a wrong guess], rollback(1): the step after as after the next alone
    self.stub()
    def logits(toks:list[int], pos:int) -> np.ndarray: return self.eng.forward(Tensor([toks], dtype=dtypes.int32), pos)[0].numpy()
    want = [logits([t], pos)[0] for pos, t in enumerate([3, 5, 7])]
    logits([3], 0)
    np.testing.assert_allclose(logits([5, 1], 1)[0], want[1], rtol=1e-4, atol=1e-4)
    self.eng.model.rollback(1)
    np.testing.assert_allclose(logits([7], 2)[0], want[2], rtol=1e-4, atol=1e-4)

  @unittest.skipUnless(amd_custom_kernels_supported(Device.DEFAULT), "the graph's host syncs and the expert kernel: AMD")
  def test_generate(self):  # the jitted steps with the host's step_io (every expert cached: no reads), captured on dry steps
    self.eng.warm()
    want, tok = [], 3
    for pos in range(6): want.append(tok:=int(self.want(tok, pos).argmax()))
    tokens = [3]
    self.assertEqual([t for _, t in zip(range(3), self.eng.generate(tokens))], want[:3])
    self.assertEqual(self.eng.get_start_pos(tokens + [want[3]]), 3)  # (a prompt that extends the tokens fed: only the rest)
    self.assertEqual([t for _, t in zip(range(2), self.eng.generate(tokens + [want[3]]))], want[4:])
    self.assertEqual((self.eng.io.refs, self.eng.io.misses), (self.eng.io.hits, 0))

if __name__ == "__main__":
  unittest.main()
