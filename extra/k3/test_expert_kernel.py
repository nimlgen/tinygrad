#!/usr/bin/env python3
# DEV=AMD:HIP python3 extra/k3/test_expert_kernel.py: the grouped expert gemv over cache slots matches dequantize + matmul
import unittest
import numpy as np
from tinygrad import Tensor, dtypes, nn
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import expert_linear, q8_act, q8_rms_norm, _q8_quantize, _q8_cache, QUANT_SIZES, IQ3_XXS

LAT, UNITS, NSLOTS = 3584, 512, 20
ROW_UP, ROW_DOWN = LAT // 256 * QUANT_SIZES[IQ3_XXS], UNITS // 256 * QUANT_SIZES[IQ3_XXS]
PART = 2 * UNITS * ROW_UP + LAT * ROW_DOWN  # gate+up rows, then down columns: the layout extra/k3/pack.py writes

def random_iq3xxs(nbytes:int, rng) -> np.ndarray:
  raw = rng.integers(0, 256, nbytes, dtype=np.uint8)
  # sane fp16 block scales (the first 2 bytes of every 98 byte block)
  d = raw.reshape(-1, QUANT_SIZES[IQ3_XXS])
  d[:, :2] = np.frombuffer(np.full(len(d), 0.01, np.float16).tobytes(), np.uint8).reshape(-1, 2)
  return raw

class TestExpertLinear(unittest.TestCase):
  def setUp(self):
    rng = np.random.default_rng(0)
    self.raw = random_iq3xxs(NSLOTS * PART, rng)
    self.pool = Tensor(self.raw).bitcast(dtypes.uint16).realize()
    self.sel = [5, 0, 19, 7, 3, 11, 2, 18, 9, 1, 14, 6, 17, 4, 12, 8]
    self.slots = Tensor(np.array(self.sel, np.int32)).realize()
    self.rng = rng

  def weight(self, slot:int, off:int, rows:int, cols:int) -> np.ndarray:
    blob = Tensor(self.raw[slot * PART + off:slot * PART + off + rows * cols // 256 * QUANT_SIZES[IQ3_XXS]])
    return ggml_data_to_tensor(blob, rows * cols, IQ3_XXS).reshape(rows, cols).numpy()

  def test_gate_up_shared_input(self):
    x = self.rng.standard_normal((1, LAT)).astype(np.float32)
    out = expert_linear(self.pool, self.slots, Tensor(x), 2 * UNITS, LAT, IQ3_XXS, PART // 2).numpy()
    for k, s in enumerate(self.sel):
      ref = self.weight(s, 0, 2 * UNITS, LAT) @ x[0]
      np.testing.assert_allclose(out[k], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())

  def test_down_per_expert_input(self):
    h = self.rng.standard_normal((len(self.sel), UNITS)).astype(np.float32)
    off = 2 * UNITS * ROW_UP
    out = expert_linear(self.pool, self.slots, Tensor(h), LAT, UNITS, IQ3_XXS, PART // 2, off // 2).numpy()
    for k, s in enumerate(self.sel):
      ref = self.weight(s, off, LAT, UNITS) @ h[k]
      np.testing.assert_allclose(out[k], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())

  def test_down_weighted_sum(self):  # with weights: the weighted sum of the slots' outputs, plus a bias
    h, w = self.rng.standard_normal((len(self.sel), UNITS)).astype(np.float32), self.rng.uniform(0, 1, len(self.sel)).astype(np.float32)
    off = 2 * UNITS * ROW_UP
    b = self.rng.standard_normal(LAT).astype(np.float32)
    out = expert_linear(self.pool, self.slots, Tensor(h), LAT, UNITS, IQ3_XXS, PART // 2, off // 2, w=Tensor(w), bias=Tensor(b)).numpy()
    ref = sum(w[k] * self.weight(s, off, LAT, UNITS) @ h[k] for k, s in enumerate(self.sel)) + b
    np.testing.assert_allclose(out[0], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())

  def test_tokens(self):  # every token its own slots (entry k of token k % T): gate/up on its token's input, each token's weighted sum plus its bias
    T, off, sel = 2, 2 * UNITS * ROW_UP, self.sel[:12]
    slots = Tensor(np.array(sel, np.int32)).realize()
    x = self.rng.standard_normal((T, LAT)).astype(np.float32)
    gu = expert_linear(self.pool, slots, Tensor(x), 2 * UNITS, LAT, IQ3_XXS, PART // 2).numpy()
    h, w = self.rng.standard_normal((len(sel), UNITS)).astype(np.float32), self.rng.uniform(0, 1, len(sel)).astype(np.float32)
    b = self.rng.standard_normal((T, LAT)).astype(np.float32)
    out = expert_linear(self.pool, slots, Tensor(h), LAT, UNITS, IQ3_XXS, PART // 2, off // 2, tokens=T, w=Tensor(w), bias=Tensor(b)).numpy()
    for k, s in enumerate(sel):
      ref = self.weight(s, 0, 2 * UNITS, LAT) @ x[k % T]
      np.testing.assert_allclose(gu[k], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())
    for t in range(T):
      ref = sum(w[k] * self.weight(s, off, LAT, UNITS) @ h[k] for k, s in enumerate(sel) if k % T == t) + b[t]
      np.testing.assert_allclose(out[t], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())

  def test_padding(self):  # entries of weight 0, each token's last: gate/up skips them (rows not written), the sums are of the others
    T, off, sel = 2, 2 * UNITS * ROW_UP, self.sel[:12]
    w = self.rng.uniform(0.1, 1, len(sel)).astype(np.float32)
    w[6::2], w[9::2] = 0, 0  # (token 0 keeps its first 3, token 1 its first 4 of 6)
    slots, x = Tensor(np.array(sel, np.int32)).realize(), self.rng.standard_normal((T, LAT)).astype(np.float32)
    gu = expert_linear(self.pool, slots, Tensor(x), 2 * UNITS, LAT, IQ3_XXS, PART // 2, w=Tensor(w), summed=False).numpy()
    h = self.rng.standard_normal((len(sel), UNITS)).astype(np.float32)
    out = expert_linear(self.pool, slots, Tensor(h), LAT, UNITS, IQ3_XXS, PART // 2, off // 2, tokens=T, w=Tensor(w)).numpy()
    for k, s in enumerate(sel):
      if w[k] == 0: continue
      ref = self.weight(s, 0, 2 * UNITS, LAT) @ x[k % T]
      np.testing.assert_allclose(gu[k], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())
    for t in range(T):
      ref = sum(w[k] * self.weight(s, off, LAT, UNITS) @ h[k] for k, s in enumerate(sel) if k % T == t)
      np.testing.assert_allclose(out[t], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())

  def test_q8_act(self):  # the act of a gate/up output's halves in its q8 kernel: the q8 of the act
    gu = Tensor(self.rng.standard_normal((len(self.sel), 2 * UNITS)).astype(np.float32) * 3)
    def act(g, u): return (g / 2).tanh() * 2 * g.sigmoid() * u
    y = q8_act(act, gu, parts=2)
    got, want = _q8_cache[y.uop], _q8_quantize(Tensor(y.numpy()), tokens=len(self.sel), in_features=UNITS)
    np.testing.assert_equal(got[0].numpy(), want[0].numpy())
    for g, w in zip(got[1:], want[1:]): np.testing.assert_allclose(g.numpy(), w.numpy(), rtol=1e-6)

  def test_q8_act_inputs(self):  # gate and up apart (the shared expert's): the same q8
    g, u = (Tensor(self.rng.standard_normal((2, UNITS)).astype(np.float32) * 3) for _ in range(2))
    def act(g, u): return (g / 2).tanh() * 2 * g.sigmoid() * u
    y = q8_act(act, g, u)
    got, want = _q8_cache[y.uop], _q8_quantize(Tensor(y.numpy()), tokens=2, in_features=UNITS)
    np.testing.assert_equal(got[0].numpy(), want[0].numpy())
    for a, b in zip(got[1:], want[1:]): np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-6)

  def test_q8_rms_norm(self):  # the q8 of a norm from its input: as the norm's within a step of the q8 (the sums in another order)
    norm, x = nn.RMSNorm(LAT, 1e-6), Tensor(self.rng.standard_normal((1, 2, LAT)).astype(np.float32) * 3)
    norm.weight = Tensor(self.rng.uniform(0.5, 1.5, LAT).astype(np.float32))
    y = q8_rms_norm(norm, x)
    q, scale, _ = _q8_cache[y.uop]
    np.testing.assert_allclose(y.numpy(), norm(x).numpy())
    deq = q.numpy().view(np.int8).reshape(2, -1, 32) * scale.numpy()[..., None]
    np.testing.assert_array_less(np.abs(deq.reshape(2, LAT) - y.numpy()[0]), scale.numpy().repeat(32, -1) * 0.51)

  def test_pool_past_2g_elements(self):  # a slot past 2**31 words: 64 bit offsets
    far = 2 ** 31 // (PART // 2) + 2
    pool = Tensor.empty((far + 1) * PART // 2, dtype=dtypes.uint16).contiguous().realize()
    pool[far * PART // 2:].assign(Tensor(self.raw[5 * PART:6 * PART]).bitcast(dtypes.uint16)).realize()
    x = self.rng.standard_normal((1, LAT)).astype(np.float32)
    out = expert_linear(pool, Tensor(np.full(16, far, np.int32)), Tensor(x), 2 * UNITS, LAT, IQ3_XXS, PART // 2).numpy()
    ref = self.weight(5, 0, 2 * UNITS, LAT) @ x[0]
    for k in range(16): np.testing.assert_allclose(out[k], ref, rtol=0.05, atol=0.05 * np.abs(ref).max())

if __name__ == "__main__":
  unittest.main()
