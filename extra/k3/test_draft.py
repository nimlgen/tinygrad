import tempfile, unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Device, Tensor, TinyJit, UOp, dtypes
from tinygrad.llm.kernels.amd import amd_custom_kernels_supported
from extra.k3.draft import Eagle3
from extra.k3.test_engine import eagle, DIM, VOCAB

@unittest.skipUnless(amd_custom_kernels_supported(Device.DEFAULT), "AMD attention kernel")
class TestDraftAttention(unittest.TestCase):
  def test_attention(self):
    rng, N = np.random.default_rng(7), 64
    devs = (Device.DEFAULT,)
    embed = Tensor(rng.normal(0, 0.3, (VOCAB, DIM)).astype(np.float16)).shard(devs).realize()
    with tempfile.TemporaryDirectory() as folder, patch("extra.k3.draft.amd_custom_kernels_supported", return_value=False):
      draft = Eagle3(eagle(folder, rng, lora=32), lambda t: embed[t], devs, N)
    for T, start in ((1, 0), (2, 31), (5, 50)):
      tokens = Tensor(rng.integers(0, VOCAB, (1, T)), dtype=dtypes.int32).shard(devs).realize()
      hidden = Tensor(rng.normal(0, 0.3, (1, T, 3, DIM)).astype(np.float32)).shard(devs).realize()
      cache = rng.standard_normal(draft.cache.shape).astype(np.float16)
      cache[..., start + T:, :] = np.nan  # unwritten cache rows must never affect a guess
      def reset(): draft.cache.assign(Tensor(cache).shard(devs)).realize()
      reset()
      with patch("extra.k3.draft.amd_custom_kernels_supported", return_value=False):
        logits, h, c = draft(tokens, hidden, start)
        want = logits.numpy(), h.numpy(), Tensor(c).numpy()[..., :start + T, :]
      reset()
      @TinyJit
      def run(pos):
        logits, h, c = draft(tokens, hidden, pos)
        out = logits, h, Tensor(c)
        Tensor.realize(*out)
        return out
      for _ in range(3):
        logits, h, c = run(UOp.variable("start_pos", 0, N - T).bind(start))
        for got, ref in zip((logits.numpy(), h.numpy(), c.numpy()[..., :start + T, :]), want):
          np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3)

if __name__ == "__main__": unittest.main()
