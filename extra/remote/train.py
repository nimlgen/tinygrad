# Small data-parallel Llama training run, including captured JIT replay.
import time, numpy as np
from tinygrad import Tensor, TinyJit, nn, Context
from tinygrad.helpers import getenv
from tinygrad.nn.state import get_state_dict, get_parameters
from extra.models.llama import Transformer

if __name__ == "__main__":
  Tensor.manual_seed(5760)
  devs = tuple(f"AMD:{i}" for i in range(getenv("GPUS", 12)))
  model = Transformer(dim=96, hidden_dim=256, n_heads=6, n_kv_heads=6, n_layers=1, norm_eps=1e-5, vocab_size=192,
                      max_context=16, disable_kv_cache=True, jit=False)
  for t in get_state_dict(model).values(): t.shard_(devs, axis=None).realize()
  optim = nn.optim.SGD(get_parameters(model), lr=0.01)
  tokens = np.arange(len(devs) * 17, dtype=np.int32).reshape(len(devs), 17) % 192
  x, y = [Tensor(t.copy()).shard(devs, axis=0).contiguous().realize() for t in (tokens[:, :-1], tokens[:, 1:])]

  @TinyJit
  def step(x, y):
    optim.zero_grad()
    loss = model(x, 0, temperature=float('nan')).sparse_categorical_crossentropy(y)
    loss.backward()
    norm = sum(p.grad.float().square().sum() for p in optim.params).sqrt()
    loss.realize(norm)
    optim.step()
    return loss, norm

  with Context(TRAINING=1):
    for i in range(getenv("STEPS", 5)):
      start = time.perf_counter()
      loss, norm = (t.item() for t in step(x, y))
      print(f"step {i + 1}: loss {loss:.6f}, grad norm {norm:.6f}, {time.perf_counter() - start:.3f}s", flush=True)
      assert np.isfinite(loss) and np.isfinite(norm) and norm > 0, (loss, norm)
