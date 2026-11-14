#!/usr/bin/env python3
# the engine's forward against master's model with the routed experts on a tiny random kimi-k3 on two CPU "gpus": every expert in the pools
# (a slot a code, each gpu its units), the host's part stubbed by exact routing (TAU=0: a token's top k in the store's two parts)
# DEV=CPU PYTHONPATH=. python3 extra/k3/test_engine.py (DEV=AMD: the jitted steps with the host's step_io too)
import json, pathlib, tempfile, unittest
from types import SimpleNamespace
import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType, quants
from tinygrad import Tensor, Device, UOp, nn, dtypes
from tinygrad.llm.kernels.amd import amd_custom_kernels_supported
from tinygrad.llm.model import Transformer, TransformerConfig, SSMConfig, ExpertGating, MLATransformerBlock
from extra.k3.engine import Engine
from extra.k3.draft import Eagle3

DIM, LAT, FF, E, K, G, VOCAB = 64, 32, 64, 8, 4, 2, 32
CONFIG = TransformerConfig(num_blocks=3, dim=DIM, hidden_dim=FF, n_heads=2, n_kv_heads=1, norm_eps=1e-5, vocab_size=VOCAB, head_dim=24,
  rope_theta=10000.0, rope_dim=8, v_head_dim=16, max_context=16, num_experts=E, num_experts_per_tok=K, expert_gating_func=ExpertGating.SIGMOID,
  norm_topk_prob=True, q_lora_rank=16, kv_lora_rank=16, shared_expert_dim=FF, ssm_layers=(True, True, False), attn_output_gate=True,
  leading_dense_blocks=1, dense_hidden_dim=64, expert_bias=True, shared_expert_gate=False, expert_latent_dim=LAT, attn_res_block_size=1,
  situ=(4.0, 25.0), ssm=SSMConfig(conv_kernel=2, state_size=16, group_count=2, time_step_rank=2, inner_size=32, kda=True, gate_lower_bound=-5.0,
                                    full_rank=(True, False)))

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

def eagle(folder:str, rng:np.random.Generator) -> str:
  """a tiny random EAGLE-3 head (kimi-k3-eagle3.1-mla's layout) on the hidden states after blocks 0, 1, 2"""
  H, nope, rope, vd, lora, ql, ff, p, a = 2, 8, 8, 8, 16, 16, 32, "layers.0.", "layers.0.self_attn."
  json.dump({"num_attention_heads": H, "qk_nope_head_dim": nope, "qk_rope_head_dim": rope, "v_head_dim": vd, "kv_lora_rank": lora,
             "rms_norm_eps": 1e-5, "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3]}, "rope_theta": 10000.0, "rope_scaling":
             {"factor": 4.0, "original_max_position_embeddings": 16, "beta_fast": 32, "beta_slow": 1, "mscale_all_dim": 1.0}},
            open(f"{folder}/config.json", "w"))
  shapes = {**{f"fc_norm.{i}.weight": (DIM,) for i in range(3)}, "fc.weight": (DIM, 3*DIM), p+"input_layernorm.weight": (DIM,),
            p+"hidden_norm.weight": (DIM,), a+"q_a_proj.weight": (ql, 2*DIM), a+"q_a_layernorm.weight": (ql,),
            a+"q_b_proj.weight": (H*(nope+rope), ql),
            a+"kv_a_proj_with_mqa.weight": (lora+rope, 2*DIM), a+"kv_a_layernorm.weight": (lora,), a+"kv_b_proj.weight": (H*(nope+vd), lora),
            a+"o_proj.weight": (DIM, H*vd), p+"post_attention_layernorm.weight": (DIM,), p+"mlp.gate_proj.weight": (ff, DIM),
            p+"mlp.up_proj.weight": (ff, DIM), p+"mlp.down_proj.weight": (DIM, ff), "norm.weight": (DIM,), "lm_head.weight": (VOCAB, DIM)}
  nn.state.safe_save({k: Tensor(rng.normal(0, 0.3, s).astype(np.float32)) for k, s in shapes.items()}, f"{folder}/model.safetensors")
  return folder

class TestEngine(unittest.TestCase):
  def setUp(self):
    rng = np.random.default_rng(0)
    weights = {n: (1 + rng.normal(0, 0.1, t.shape) if 'norm' in n or 'res_score' in n else rng.normal(0, 0.3, t.shape)).astype(np.float32)
               for n, t in nn.state.get_state_dict(Transformer(CONFIG)).items()}
    folder = tempfile.mkdtemp()
    write(full:=pathlib.Path(folder)/'full.gguf', weights, True)
    write(trunk:=pathlib.Path(folder)/'trunk.gguf', weights, False)
    self.ref, model, self.trunk = Transformer.from_gguf(full)[0], Transformer.from_gguf(trunk, shard=G)[0], trunk
    # every expert in its slot, a code (layer * E + expert) a slot
    part, layers = 4096, [1, 2]
    m = {"n_experts": E, "lat": LAT, "units": [FF // G] * G, "part_bytes": [part] * G, "expert_type": "Q8_0", "block": [32, 34], "layers": layers}
    self.eng = eng = Engine(model, m, slots=len(layers) * E, tokens=2, chunk=4)
    pools = [b''.join(parts(weights, li, e, part)[g] for li in layers for e in range(E)) for g in range(G)]
    eng.pool.assign(Tensor(np.frombuffer(b''.join(pools), np.dtype(eng.word.fmt))).shard(eng.devs, 0)).realize()
    eng.state["slot_of"][:] = eng.state["expert_in"][:] = np.arange(len(layers) * E)

  def want(self, tok:int, pos:int) -> np.ndarray:  # master's model's logits (the engine's forward of it, with the routed experts)
    return Engine.forward(SimpleNamespace(model=self.ref, draft=None), Tensor([[tok]], dtype=dtypes.int32), pos)[0].numpy()[0]

  def stub(self):  # the host's part stubbed: a token's top k by sigmoid + bias, the weights the sigmoids normalized, in the two parts (an
    # entry a token's: a union of the tokens' with an expert more than once), a chunk's in one (entry j*T + t token t's j-th)
    eng = self.eng
    tables:dict[int, list[Tensor]] = {}
    def host(li:int, i:int, dep:Tensor) -> Tensor:  # (part 1's dep: the scores)
      if i == 0:
        sig = dep[:, 0].to("CPU").sigmoid()
        sel = (sig + Tensor(eng.bias[li], device="CPU")).topk(K)[1]
        w = sig.gather(-1, sel) / sig.gather(-1, sel).sum(-1, keepdim=True)
        T = sel.shape[0]
        def table(ks:slice) -> Tensor:  # (an entry a token's expert, j*T + t: its weight the token's, the others' 0)
          n, tok = (ks.stop - ks.start) * T, Tensor.arange((ks.stop - ks.start) * T).to("CPU").reshape(-1, 1) % T
          wt = (tok == Tensor.arange(T).to("CPU")).where(w[:, ks].T.flatten().reshape(n, 1), 0).flatten()
          return Tensor([1], dtype=dtypes.uint32, device="CPU").cat((sel[:, ks].T.flatten() + li * E).cast(dtypes.int32).bitcast(dtypes.uint32),
                                                                    wt.cast(dtypes.float32).bitcast(dtypes.uint32)).shard(eng.devs)
        def chunk() -> Tensor:  # (the rows in entry order: [1 | their slots, tokens | the entries' slots, rows | weights])
          slots, k = (sel.T.flatten() + li * E).cast(dtypes.int32), Tensor.arange(K * T, dtype=dtypes.int32).to("CPU")
          return Tensor([1], dtype=dtypes.int32, device="CPU").cat(slots, k % T, slots, k).bitcast(dtypes.uint32) \
            .cat(w.T.flatten().cast(dtypes.float32).bitcast(dtypes.uint32)).shard(eng.devs)
        tables[li] = [table(slice(0, eng.ks[0])), table(slice(eng.ks[0], K))] if T <= eng.tokens else [chunk()]
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

  def test_chunk(self):  # a prompt of 9 in two chunks of 4 then a token: the logits after it as a token a step's
    self.stub()
    toks = [3, 5, 7, 1, 2, 9, 4, 6, 8]
    for pos, t in enumerate(toks): want = self.eng.forward(Tensor([[t]], dtype=dtypes.int32), pos)[0].numpy()[0]
    for pos in (0, 4): self.eng.forward(Tensor([toks[pos:pos + 4]], dtype=dtypes.int32), pos)[0].realize()
    ours = self.eng.forward(Tensor([toks[8:]], dtype=dtypes.int32), 8)[0].numpy()[0]
    np.testing.assert_allclose(ours, want, rtol=1e-4, atol=1e-4 * np.abs(want).max())

  def test_prefill(self):  # the chunks of a prompt but its last token (each with the tokens after it given), none without a chunk
    steps:list = []
    self.eng.step = lambda toks, pos, given: steps.append((pos, toks, given))  # type: ignore[method-assign,assignment,misc]
    self.assertEqual(self.eng.prefill(list(range(10)), 2), 6)
    self.assertEqual(steps, [(2, [2, 3, 4, 5], [3, 4, 5, 6])])
    self.assertEqual(self.eng.prefill(list(range(9)), 0), 8)
    self.assertEqual([s[0] for s in steps[1:]], [0, 4])
    self.eng.chunk = 0
    self.assertEqual(self.eng.prefill(list(range(9)), 3), 3)
    self.assertEqual(len(steps), 3)

  def test_states_allocated(self):  # the blocks' caches and states and the draft's allocated by the engine, before its pool takes what's free
    model, folder = Transformer.from_gguf(self.trunk, shard=G)[0], tempfile.mkdtemp()
    draft = Eagle3(eagle(folder, np.random.default_rng(1)), lambda t: model.token_embd(t).float(), self.eng.devs, model.max_context)
    Engine(model, self.eng.m, slots=1, tokens=2, draft=draft)
    for t in (*(b.cache_k for b in model.blk if isinstance(b, MLATransformerBlock)), draft.cache, model.blk[0].kept[0][0]):
      self.assertTrue(all(b.is_allocated() for b in t.uop.base.buffer.bufs))

  def test_draft_step(self):  # a jitted verify step with the draft: its tokens and guesses as an unjitted one's
    self.stub()
    eng, folder = self.eng, tempfile.mkdtemp()
    eng.draft = Eagle3(eagle(folder, np.random.default_rng(1)), lambda t: eng.model.token_embd(t).float(), eng.devs, eng.model.max_context)
    eng.tok_np[:2], eng.given_np[:2] = [3, 5], [-1, -1]
    eng.run(2, UOp.variable("start_pos", 0, 15).bind(0))
    want = eng.res_np.copy()
    for _ in range(3): eng.jits[2](eng.v_pos.bind(0))
    np.testing.assert_equal(eng.res_np, want)

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

  @unittest.skipUnless(amd_custom_kernels_supported(Device.DEFAULT), "the graph's host syncs and the expert kernel: AMD")
  def test_generate_fork(self):  # prompts sharing X: the third fed from where the second diverged from the first (a chunk's end, its states saved)
    self.eng.warm()
    X = [3, 5, 7, 9]
    def take(prompt:list[int]) -> list[int]: return [t for _, t in zip(range(3), self.eng.generate(prompt))]
    want = take(X + [2, 4])
    take(X + [6, 8])
    self.assertEqual(self.eng.get_start_pos(X + [2, 4]), len(X))
    self.assertEqual(take(X + [2, 4]), want)

if __name__ == "__main__":
  unittest.main()
