#!/usr/bin/env python3
# K3 decode with the routed experts streamed from the nvme drives: master's model (a trunk gguf without the routed experts, sharded over the
# gpus) in a jitted step graph of the engine's own forward. at a moe block every gpu writes its router scores to the host (store), the host
# routes, reads the missing experts from the drives into each gpu's pool of slots and gives the gpus the slots and weights in two parts
# (host.py's step_io, in c alongside the graph), the gpus wait for each (wait) and run their units of the experts over the pool
# DEV=AMD:HIP PYTHONPATH=. python3 extra/k3/engine.py trunk.gguf "prompt"
import ctypes, functools, json, os, sys, time
import numpy as np
from tinygrad import Tensor, Device, TinyJit, UOp, dtypes
from tinygrad.device import Buffer, BufferSpec, MultiBuffer
from tinygrad.dtype import DType
from tinygrad.helpers import getenv, prod
from tinygrad.llm.model import Transformer, GatedDeltaNetBlock, gated_activation
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import amd_custom_kernels_supported
from tinygrad.llm.kernels.amd_gguf import expert_linear, QUANT_NAMES, HALFWORD_QUANTS
from extra.k3.host import IoT, host_c, store, wait, reads
from extra.k3.draft import Eagle3

# TAU: a token's experts under this weight left out unless cached, TAUC: those under it left out cached or not, BETA: a cached expert's
# routing bonus, FILL: the experts tau left out read anyway (a layer), PREFETCH: the next layer's experts guessed a layer. SLOTS: on a gpu
# (default: what's free but SPARE bytes)
TAU, TAUC, BETA, FILL, PREFETCH = getenv("TAU", 0.065), getenv("TAUC", 0.0), getenv("BETA", 0.07), getenv("FILL", 2), getenv("PREFETCH", 5)
SLOTS, SPARE = getenv("SLOTS", 0), getenv("SPARE", 2 << 30)

def free_bytes(dev:str, chunk:int=256 << 20) -> int:
  """the memory dev can still allocate: chunks of it allocated till one fails, then freed"""
  bufs:list[Buffer] = []
  try:
    while True: bufs.append(Buffer(dev, chunk).allocate())
  except MemoryError: pass
  n = len(bufs)
  bufs.clear()
  Device[dev].allocator.free_cache()
  return n * chunk

def host_tensor(dev:str|tuple[str, ...], n:int, dtype:DType) -> tuple[Tensor, list[np.ndarray]]:
  """n zeroed words in uncached host memory of the device (or each of the devices: they read and write it over pcie), numpy views of each"""
  spec = BufferSpec(host=True, uncached=True, cpu_access=True)
  buf = MultiBuffer(dev, n * dtype.itemsize, spec) if isinstance(dev, tuple) else Buffer(dev, n * dtype.itemsize, options=spec)
  bufs = buf.bufs if isinstance(buf, MultiBuffer) else [buf]
  views = [np.frombuffer(b.ensure_allocated().host.view(fmt='B').mv, np.dtype(dtype.fmt)) for b in bufs]
  for v in views: v[:] = 0
  return Tensor(UOp.from_buffer(buf, dtype)), views

class Engine:
  def __init__(self, model:Transformer, m:dict, slots:int=SLOTS, tokens:int=1, guess:str="", draft:Eagle3|None=None, logits:bool=False):
    """the moe blocks of model with their routed experts streamed from the store of manifest m into a pool of slots on each gpu, a part of
    every expert each (its units), in steps of up to tokens (a draft's verify steps: 2), keeping their logits or not"""
    self.model, self.m, self.devs, self.draft, self.keep_logits = model, m, model.token_embd.weight.device, draft, logits
    blocks = [model.blk[b] for b in m["layers"] if b < len(model.blk)]
    L, E, K, G, self.lat, self.units, self.part = len(blocks), m["n_experts"], model.config.num_experts_per_tok, len(self.devs), m["lat"], \
      m["units"][0], m["part_bytes"][0]
    assert len(set(m["part_bytes"])) == 1 and len(set(m["units"])) == 1 and len(m["units"]) == G, "a part of every expert on each gpu"
    self.qtype, self.block = {n.upper(): t for t, n in QUANT_NAMES.items()}[m["expert_type"]], m["block"]
    # each layer's host buffers on every gpu: its tag, router scores (a token's rows: the router's, the next layer's guess; a gpu its experts')
    # and the two parts of a step's slots and weights, [ready | slots | weights]: a token's first k1 experts, then the rest (j*T + t: token t's j-th)
    self.ks = (K // 2, K - K // 2)
    bufs = [[host_tensor(self.devs, n, dt) for n, dt in ((1, dtypes.uint32), (tokens * 2 * E // G, dtypes.float32),
                                                          *((1 + 2 * k * tokens, dtypes.uint32) for k in self.ks))] for _ in blocks]
    self.tags, self.parts = [b[0][0] for b in bufs], [[b[2 + i][0] for b in bufs] for i in range(2)]
    self.scores = [b[1][0].uop.reshape(tokens, 2, E // G).unshard(2) for b in bufs]
    # the scorers: a layer's router and the guess of the next layer's experts on its input (GUESS=dir: guess-<block>.npy maps fitted on traces,
    # else the next layer's router), a gpu its experts' rows. the routers' memory freed: the scorers route
    routers = [b.ffn_gate_inp.weight.float().numpy() for b in blocks]
    def guessed(li:int) -> np.ndarray: return np.load(f) if os.path.exists(f:=f"{guess}/guess-{m['layers'][li]}.npy") else routers[li]
    self.scorers = [Tensor(np.stack([r, guessed(min(li + 1, L - 1))]).astype(np.float16)).shard(self.devs, 1).realize()
                    for li, r in enumerate(routers)]
    self.bias = [b.exp_probs_b["bias"].float().numpy() for b in blocks]
    for b in model.blk:  # (a verify step's recurrent states after its first token kept: a rollback)
      if isinstance(b, GatedDeltaNetBlock): b.keep = tokens - 1
    for li, b in enumerate(blocks):
      b._routed = functools.partial(self.routed, li)  # type: ignore[method-assign]
      b.ffn_gate_inp.weight = Tensor.empty(0)
    # each gpu's pool in the experts' words (a bitcast of it is a copy), of what's free
    self.slots = slots = slots or (min(free_bytes(d) for d in self.devs) - SPARE) // self.part
    self.word = dtypes.uint16 if self.qtype in HALFWORD_QUANTS else dtypes.uint32
    self.pool = Tensor.zeros(G * slots * self.part // self.word.itemsize, dtype=self.word, device=self.devs, axis=0).realize()
    # the host's state, kept by step_io: each code's (layer * E + expert) slot (-1: not cached) and each slot's code (-1: free), the decayed
    # use counts, the guesses' reads in flight, the guesses their layer hasn't routed, pinned slots, the victims of the next reads
    self.state = {"slot_of": np.full(L * E, -1, np.int32), "expert_in": np.full(slots, -1, np.int64), "freq": np.zeros(L * E, np.float32),
                  "flying": np.zeros(L * E, np.int32), "guess": np.zeros(L * E, np.uint8), "mask": np.zeros(slots, np.uint8),
                  "victims": np.zeros(32 * tokens, np.int64), "bias": np.array([b.ctypes.data for b in self.bias], np.uint64),
                  "bufs": np.array([[[v.ctypes.data for v in views] for _, views in b] for b in bufs], np.uint64)}
    self.io = IoT(layers=L, E=E, ngpu=G, k=K, kmin=K // 4, k1=self.ks[0], rows=2, slots=slots, tau=TAU, tauc=TAUC, beta=BETA, pf=PREFETCH,
                  fill=FILL, decay=0.5 ** (1 / 8))
    for name, a in self.state.items(): setattr(self.io, name, a.ctypes.data)
    # a step's graph for each count of tokens, their tokens and the draft's given tokens in host memory
    self.v_pos = UOp.variable("start_pos", 0, model.max_context - 1)
    (self.tok, (self.tok_np,)), (self.given, (self.given_np,)) = (host_tensor(self.devs[0], tokens, dtypes.int32) for _ in range(2))
    self.jits = {T: TinyJit(functools.partial(self.run, T)) for T in range(1, tokens + 1)}
    self.times, self.verified = np.zeros(2), np.zeros(2)  # (launching the graphs, after the last layer), (verify steps, right guesses)

  def drives(self, disks:list[list[str]]):
    """step_io's reads from the store's drives (each gpu's disk) into the pools"""
    self.tables, scalars = reads(self.m, disks, self.pool.uop.buffer.bufs, len(self.bias) * self.m["n_experts"])
    for name, v in {**scalars, **{k: a.ctypes.data for k, a in self.tables.items()}}.items(): setattr(self.io, name, v)

  # *** the step's graph

  def forward(self, tokens:Tensor, pos:UOp) -> tuple[Tensor, Tensor|None]:
    """the logits (T, vocab) after each of the tokens (1, T) at positions pos.., the hidden states (1, T, 3, dim) after the draft's blocks"""
    m, hidden = self.model, []
    x = m.token_embd(tokens.to(m.token_embd.weight.device)).float().unsqueeze(2)  # (the attention residuals' streams: (1, T, 1, dim))
    mask = None
    for i, b in enumerate(m.blk):
      x, mask = b(x, (pos,), mask)
      if self.draft is not None and i in self.draft.layers: hidden.append(x[:, :, -1])
    return m.output(m.output_norm(m.output_res_score.mix(x)))[0].float(), Tensor.stack(*hidden, dim=2) if hidden else None

  def run(self, T:int, pos:UOp) -> tuple[Tensor, Tensor, Tensor]:
    """the greedy token after each of the step's tokens, their logits (kept, else the tokens) and (a draft) its guess of the token after
    that (after the given one where there's one: a prompt's next)"""
    logits, hidden = self.forward(self.tok[:T].reshape(1, T), pos)
    toks = guesses = logits.argmax(-1)
    if self.draft is not None:
      given = self.given[:T].to(toks.device)
      guesses = self.draft((given >= 0).where(given, toks).reshape(1, T), hidden, pos)[0].argmax(-1)
    if not self.keep_logits: logits = toks  # (realized, the logits are a buffer of their own: only for an eval)
    Tensor.realize(toks, logits, guesses)
    return toks, logits, guesses

  def routed(self, li:int, x:Tensor, h:Tensor) -> Tensor:
    """layer li's routed experts on the tokens x (1, T, dim), h their expert inputs (1, T, lat): the gpus' scores to the host, each gpu's units
    of the experts it gives in two parts, summed over the gpus"""
    scores = Tensor(self.scores[li])[:x.shape[1]].assign((self.scorers[li] @ x[0].T).float().permute(2, 0, 1))
    out = self.experts(0, h[0], self.host(li, 0, scores)).contiguous()  # (a buffer: part 2 waited for after it)
    return (out + self.experts(1, h[0], self.host(li, 1, out))).sum(0).unsqueeze(0)

  def host(self, li:int, i:int, dep:Tensor) -> Tensor:
    """part i of layer li's slots and weights once the host releases it, waited for after dep (part 1: the scores, the host told they're in)"""
    t = self.parts[i][li].uop
    return Tensor(t.after(wait(t.after(store(self.tags[li].uop, dep.uop))) if i == 0 else wait(t, dep.uop)))

  def experts(self, i:int, h:Tensor, table:Tensor) -> Tensor:
    """each gpu's units of the experts of part i's table (j*T + t: token t's j-th, padded with slot 0 weight 0) on h (T, lat): (gpus, T, lat)"""
    T, u, n = h.shape[0], self.units, self.ks[i] * h.shape[0]
    slots, w = table[1:1 + n].bitcast(dtypes.int32), table[1 + n:1 + 2 * n].bitcast(dtypes.float32)
    gu = self.gemv(slots, h, 2 * u, self.lat, 0)
    act = gated_activation(gu[..., :u], gu[..., u:], self.model.config)
    down = self.gemv(slots, act, self.lat, u, 2 * u * self.lat // self.block[0] * self.block[1])
    return (down * w.reshape(n, 1)).reshape(-1, self.ks[i], T, self.lat).sum(1)

  def gemv(self, slots:Tensor, x:Tensor, out_f:int, in_f:int, off:int) -> Tensor:
    """(gpus, n, out_f): W @ x[k % rows] (or a gpu's x[k]) of the W at off bytes in slots[k] of each gpu's pool (decoded where there's no kernel)"""
    if amd_custom_kernels_supported(x.device): return expert_linear(self.pool, slots, x, out_f, in_f, self.qtype, self.part, off)
    n, size, isz = slots.shape[0], out_f * in_f // self.block[0] * self.block[1], self.word.itemsize
    w = self.pool.reshape(-1, self.slots, self.part // isz)[:, slots][..., off // isz:(off + size) // isz].bitcast(dtypes.uint8)
    w = ggml_data_to_tensor(w.flatten(), prod(w.shape) // self.block[1] * self.block[0], self.qtype).reshape(-1, n, out_f, in_f)
    return (w * (x if x.ndim == 3 else x[Tensor.arange(n) % x.shape[0]]).unsqueeze(-2)).sum(-1)

  # *** the host's side

  def warm(self):
    """every step's graph captured on dry steps: the host's flags preset (its tables still zero: no experts)"""
    assert all(Device[d].has_copy_queue for d in self.devs), "the host syncs run on the gpus' copy queues"
    host_c().flags(ctypes.byref(self.io), 1)
    for jit in self.jits.values():
      for _ in range(3): jit(self.v_pos.bind(0))[0].tolist()

  def step(self, toks:list[int], pos:int, given:list[int]|None=None) -> tuple[list[int], list[int]]:
    """a step of the tokens at positions pos..: the gpus run its graph, the host every layer's routing and reads. the greedy token after each
    and the draft's guess after that (their logits kept)"""
    io, T, t0 = self.io, len(toks), time.perf_counter()
    io.tokens, io.victims_n, self.tok_np[:T], self.given_np[:T] = T, 32 * T, toks, ((given or []) + [-1] * T)[:T]
    host_c().flags(ctypes.byref(io), 0)
    out, self.logits, guesses = self.jits[T](self.v_pos.bind(pos))
    t1 = time.perf_counter()
    if (err:=host_c().step_io(ctypes.byref(io))):  # (the gpus run the graph out: one killed waiting on a layer wedges)
      host_c().flags(ctypes.byref(io), 1)
      raise RuntimeError({-1: "a read failed", -2: "timeout", -3: "no victims left"}[err])
    t2 = time.perf_counter()
    ret = out.tolist(), guesses.tolist()
    self.times += (t1 - t0, time.perf_counter() - t2)
    return ret

  # *** master's model's generate (LLMServer's): the caches only go forward, a prompt that extends the tokens fed feeds the rest

  @property
  def max_context(self) -> int: return self.model.max_context
  def get_start_pos(self, tokens:list[int]) -> int: return self.model.get_start_pos(tokens)

  def generate(self, tokens:list[int], temperature:float=0.0):
    """the greedy tokens after tokens (any temperature), appended as they come: fed from get_start_pos a token a step, then a token a step or
    (a draft) a verify step of [the next, the draft's guess after it]: both kept when the guess is right, else the recurrent states rolled back
    to after the next"""
    for pos in range(self.get_start_pos(tokens), len(tokens)): (t,), (g,) = self.step(tokens[pos:pos + 1], pos, tokens[pos + 1:pos + 2])
    while True:
      tokens.append(t)
      self.model._cached_tokens = tokens[:-1]
      yield t
      if self.draft is None:
        (t,), (g,) = self.step([t], len(tokens) - 1)
        continue
      toks, guesses = self.step([t, g], len(tokens) - 1)
      self.verified += (1, toks[0] == g)
      if toks[0] == g:
        tokens.append(g)
        self.model._cached_tokens = tokens[:]
        yield g
        t, g = toks[1], guesses[1]
      else:
        self.model.rollback(1)
        t, g = toks[0], guesses[0]

  def counters(self) -> np.ndarray:
    io = self.io
    return np.array([time.perf_counter(), io.misses + io.guesses + io.fills, io.refs, io.hits, io.late, io.misses, io.guesses, io.used, io.fills,
                     io.t_gpu, io.t_route, io.t_reads, io.t_after, *self.times])

  def report(self, c0:np.ndarray, n:int):
    """a step's time, reads and their split over the n steps since the counters c0"""
    dt, reads, refs, hits, late, misses, guesses, used, fills, *ms = (self.counters() - c0) / n
    print(f"{dt * 1e3:.1f} ms/token = {1 / dt:.2f} tok/s, {reads * sum(self.m['part_bytes']) / 1e9:.2f} GB/token read: {misses:.0f} misses, "
          f"{guesses:.0f} guesses ({used:.0f} used), {fills:.0f} fills; of {refs:.0f} experts {hits / refs:.1%} cached, {late / refs:.1%} in flight")
    print("  ms/token: " + ", ".join(f"{t * 1e3:.1f} {k}" for k, t in zip(("waiting on the gpus", "routing", "waiting on reads",
                                                                           "after the reads", "launching", "after the last layer"), ms)), flush=True)

if __name__ == "__main__":
  from tinygrad.llm.cli import SimpleTokenizer, chat_template
  from tinygrad.llm.serve import LLMServer
  from extra.k3.store import open_store
  # DEADLINE=s: the run stops between steps once it's s seconds in (killed mid-step, the gpus are left waiting on the host)
  st = time.perf_counter()
  def late() -> bool: return time.perf_counter() - st > getenv("DEADLINE", 540)
  model, kv = Transformer.from_gguf(sys.argv[1], max_context=getenv("CTX", 4096), shard=getenv("SHARD", 4))
  tok = SimpleTokenizer.from_gguf_kv(kv)
  print(f"model in {time.perf_counter() - st:.0f} s", flush=True)
  disks, m = open_store(list(model.token_embd.weight.device), getenv("STORE", 512 << 30))  # (k3's iq3_xxs store)
  # DRAFT=dir: the eagle head, speculative decoding with T=2 verify steps (memory: fewer SLOTS)
  draft = Eagle3(d, lambda t: model.token_embd(t).float(), model.token_embd.weight.device, model.max_context) if (d:=getenv("DRAFT", "")) \
    else None
  eng = Engine(model, m, tokens=2 if draft else 1, guess=getenv("GUESS", ""), draft=draft, logits=bool(getenv("EVAL", "")))
  eng.drives(disks)
  eng.warm()
  print(f"warm in {time.perf_counter() - st:.0f} s, {eng.slots} slots a gpu", flush=True)
  if (spec:=getenv("EVAL", "")):  # EVAL=corpus.jsonl:nseq:ntok:first: production completions teacher forced, each from position 0
    path, *nums = spec.split(":")
    nseq, ntok, first = [int(x) for x in nums] + [16, 256, 0][len(nums):]
    stats:list[tuple[float, float]] = []
    for r in [json.loads(line) for line in open(path)][first:first + nseq]:
      ids, outs, n0 = r["prompt_ids"], r["output_ids"][:ntok], len(stats)
      for pos, t in enumerate(ids + outs[:-1]):
        if late(): sys.exit(f"deadline: {len(stats)} tokens, top-1 agree and KL {np.mean(stats, 0) if stats else '-'}")
        eng.step([t], pos)
        if pos == 1: c0 = eng.counters()
        if pos < len(ids) - 1: continue
        # output token j's: top-1 agreement with production and KL(production's top 5 and the rest || ours)
        lg = eng.logits.numpy()[0].astype(np.float64)
        lp, top = lg - lg.max() - np.log(np.exp(lg - lg.max()).sum()), r["output_top"][pos - len(ids) + 1]
        p, q = np.exp([x[0] for x in top]), lp[[x[1] for x in top]]
        rest_p, rest_q = max(1 - p.sum(), 1e-12), max(1 - np.exp(q).sum(), 1e-12)
        stats.append((float(lp.argmax() == top[0][1]), float((p * (np.log(p) - q)).sum() + rest_p * (np.log(rest_p) - np.log(rest_q)))))
      (a1, k1), (a, k) = np.mean(stats[n0:], 0), np.mean(stats, 0)
      print(f"{r['task_id']}: top-1 agree {a1:.1%}, KL {k1:.4f}; all {len(stats)} tokens: top-1 agree {a:.1%}, KL {k:.4f}")
      eng.report(c0, len(ids) + len(outs) - 2)
    sys.exit(0)
  # SERVE=port: master's openai server (and the model's chat template), else N tokens after the prompt
  if (port:=getenv("SERVE", 0)): LLMServer(('', port), eng, kv.get('general.name', 'k3'), tok, chat_template(tok)).serve_forever()
  ids = tok.encode(sys.argv[2] if len(sys.argv) > 2 else "The capital of France is")
  if tok.bos_id is not None and ids[:1] != [tok.bos_id]: ids = [tok.bos_id] + ids
  t0, out = time.perf_counter(), []
  for t in eng.generate(list(ids)):
    if not out:
      print(f"prompt of {len(ids)} tokens: {len(ids) / (time.perf_counter() - t0):.2f} tok/s", flush=True)
      c0 = eng.counters()
    out.append(t)
    if len(out) == getenv("N", 32) or tok.is_end(t) or late(): break
  print(repr(tok.decode(out)), out, flush=True)
  if draft: print(f"{eng.verified[1] / eng.verified[0]:.1%} of {eng.verified[0]:.0f} guesses right")
  eng.report(c0, len(out) - 1)
