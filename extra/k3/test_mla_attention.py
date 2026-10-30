#!/usr/bin/env python3
# DEV=AMD python3 extra/k3/test_mla_attention.py: amd.mla_decode_attention (flash decoding on the mla latent cache) against numpy
import numpy as np
from tinygrad import Tensor, UOp
from tinygrad.llm.kernels.amd import mla_decode_attention

rng = np.random.default_rng(0)
H, D, DV, N, SCALE = 24, 576, 512, 131072, 192 ** -0.5
cache = rng.standard_normal((1, 1, N, D)).astype(np.float16)  # (the model's cache is half)
t_cache = Tensor(cache).realize()
cache = cache.astype(np.float32)
for S, T, masked in ((200, 1, False), (201, 2, False), (3000, 2, True), (3999, 4, False), (4096, 1, True), (24000, 2, False), (24577, 1, True),
                     (131072, 2, False), (131000, 4, True)):
  q = rng.standard_normal((1, H, T, D)).astype(np.float32)
  s = (q @ cache[0, 0, :S].T) * SCALE  # (1, H, T, S)
  if masked: mask = np.where(np.arange(S) % 5 == 1, -np.inf, 0).astype(np.float32) * np.ones((1, 1, T, 1), np.float32)  # (keys dropped)
  else: mask = np.triu(np.full((T, S), -np.inf, np.float32), S - T + 1)[None, None]
  p = np.exp(s + mask - (s + mask).max(-1, keepdims=True))
  want = (p / p.sum(-1, keepdims=True)) @ cache[0, 0, :S, :DV]
  valid = UOp.variable("valid", 1, N).bind(S)
  got = mla_decode_attention(Tensor(q), t_cache, valid, DV, SCALE, Tensor(mask) if masked else None).numpy()
  np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)
  print(f"{S} keys, {T} tokens{', masked' if masked else ''}: ok, max |d| {np.abs(got - want).max():.2e}")
