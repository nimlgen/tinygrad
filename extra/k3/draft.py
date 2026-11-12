# the EAGLE-3 draft of K3 (kimi-k3-eagle3.1-mla): one MLA decoder layer on the target's hidden states after three of its blocks. a position
# j takes the next token t[j+1] and the target's hidden states after t[j], and guesses t[j+2]. its MLA keys (the normed latent, the roped key)
# in a cache. its linears split over the target's gpus like the trunk's (outputs or inputs), in q8_0 (in f16 where there's no kernel)
import json, math
from typing import Callable
from tinygrad import Tensor, UOp, dtypes, nn
from tinygrad.llm.model import precompute_freqs_cis, apply_rope, YaRNConfig
from tinygrad.llm.kernels.amd import Linear, amd_custom_kernels_supported
from tinygrad.llm.kernels.amd_gguf import q8_0_linear

class Eagle3:
  def __init__(self, path:str, embed:Callable[[Tensor], Tensor], devs:tuple[str, ...], max_context:int):
    cfg = json.load(open(f"{path}/config.json"))
    self.H, self.nope, self.rope, self.vd, self.lora = (cfg[k] for k in ("num_attention_heads", "qk_nope_head_dim", "qk_rope_head_dim",
                                                                          "v_head_dim", "kv_lora_rank"))
    self.eps, self.embed = cfg["rms_norm_eps"], embed
    self.layers = [i - 1 for i in cfg["eagle_config"]["eagle_aux_hidden_state_layer_ids"]]  # (the target's blocks it reads after)
    rs = cfg["rope_scaling"]
    yarn = YaRNConfig(factor=rs["factor"], orig_ctx_len=rs["original_max_position_embeddings"], beta_fast=rs["beta_fast"], beta_slow=rs["beta_slow"])
    # deepseek's yarn: plain cos/sin (precompute_freqs_cis scales them by the concentration), the scores scaled by mscale^2 instead
    self.mscale = 0.1 * rs["mscale_all_dim"] * math.log(rs["factor"]) + 1.0
    self.freqs_cis = (precompute_freqs_cis(self.rope, max_context, cfg["rope_theta"], yarn=yarn) / self.mscale).to(devs).realize()
    st = nn.state.safe_load(f"{path}/model.safetensors")
    def w(name:str, axis:int|None=None) -> Tensor|Linear:  # (read once)
      if len(st[name].shape) == 1 or not amd_custom_kernels_supported(devs): return st[name].to(devs[0]).half().shard(devs, axis).realize()
      return q8_0_linear(st[name].to(devs[0]), devs, axis)
    self.w = {k[:-len(".weight")]: w(k, axis) for k, axis in (
      *((k, None) for k in st if k.endswith("norm.weight") or k.startswith("fc_norm") or "_a_proj" in k or k == "fc.weight"),
      ("layers.0.self_attn.o_proj.weight", 1), ("layers.0.mlp.down_proj.weight", 1), ("layers.0.self_attn.q_b_proj.weight", 0),
      ("layers.0.mlp.gate_proj.weight", 0), ("layers.0.mlp.up_proj.weight", 0), ("lm_head.weight", 0))}
    # kv_b by head: the up projections of the latent to the nope keys (absorbed into the queries) and to the values
    kv_b = st["layers.0.self_attn.kv_b_proj.weight"].to(devs[0]).half().reshape(self.H, self.nope + self.vd, self.lora)
    self.k_b, self.v_b = (x.contiguous().shard(devs, 0).realize() for x in (kv_b[:, :self.nope], kv_b[:, self.nope:]))
    self.cache = Tensor.empty(1, 1, max_context, self.lora + self.rope, dtype=dtypes.half, device=devs).realize()
    self.positions = Tensor.arange(max_context).to(devs).realize()  # (the causal mask's, on the gpus: a symbolic arange isn't)

  def norm(self, x:Tensor, name:str) -> Tensor: return x.float().rmsnorm(eps=self.eps) * self.w[name]
  def lin(self, x:Tensor, name:str) -> Tensor: return w(x) if isinstance(w:=self.w[name], Linear) else x.half() @ w.T

  def __call__(self, tokens:Tensor, hidden:Tensor|None, pos:int|UOp, hid:Tensor|None=None, cache:UOp|None=None) -> tuple[Tensor, Tensor, UOp]:
    """(1, T) tokens at positions pos.., (1, T, 3, dim) the target's hidden states (or hid, its own of the step before: a chain of guesses)
    -> (1, T, vocab) logits, (1, T, dim) its hidden states, its cache after their keys (a chain's next step's cache)"""
    T, p, a = tokens.shape[1], "layers.0.", "layers.0.self_attn."
    if hid is None:
      assert hidden is not None
      hid = self.lin(Tensor.cat(*[self.norm(hidden[:, :, i], f"fc_norm.{i}") for i in range(hidden.shape[2])], dim=-1), "fc")
    h = self.norm(self.embed(tokens), p + "input_layernorm").cat(self.norm(hid, p + "hidden_norm"), dim=-1)
    q = self.lin(self.norm(self.lin(h, a + "q_a_proj"), a + "q_a_layernorm"), a + "q_b_proj").reshape(1, T, self.H, -1).transpose(1, 2)
    kv, fc = self.lin(h, a + "kv_a_proj_with_mqa"), self.freqs_cis[pos:pos + T]
    q_pe = apply_rope(q[..., self.nope:], fc, interleaved=True)
    k_pe = apply_rope(kv[..., self.lora:].reshape(1, 1, T, self.rope), fc, interleaved=True)
    store = self.norm(kv[..., :self.lora], a + "kv_a_layernorm").reshape(1, 1, T, self.lora).cat(k_pe, dim=-1)
    cache = (c:=self.cache.uop if cache is None else cache).after(c[:, :, pos:pos + T].store(store.half().uop))
    k = Tensor(cache)[:, :, :pos + T].float()
    # absorbed: the nope scores (q_nope k_b) . latent, causal over the cache, the values v_b (attention . latent)
    qa = (q[..., :self.nope].half() @ self.k_b).float().cat(q_pe.float(), dim=-1)
    mask = (self.positions[:pos + T].reshape(1, 1, 1, -1) > self.positions[pos:pos + T].reshape(1, 1, T, 1)).where(float("-inf"), 0)
    attn = ((qa @ k.transpose(-1, -2)) * (self.mscale ** 2 / math.sqrt(self.nope + self.rope)) + mask).softmax(-1) @ k[..., :self.lora]
    x = self.lin((attn.half() @ self.v_b.transpose(-1, -2)).transpose(1, 2).reshape(1, T, -1), a + "o_proj") + hid
    m = self.norm(x, p + "post_attention_layernorm")
    x = x + self.lin(self.lin(m, p + "mlp.gate_proj").silu() * self.lin(m, p + "mlp.up_proj"), p + "mlp.down_proj")
    return self.lin(self.norm(x, "norm"), "lm_head"), x, cache
