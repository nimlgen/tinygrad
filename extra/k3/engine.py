#!/usr/bin/env python3
# K3 decode with the routed experts streamed from the nvme drives: master's model (a trunk gguf without the routed experts, sharded over the
# gpus) in a jitted step graph of the engine's own forward. at a moe block every gpu writes its router scores and a tag to host memory, the
# host routes, reads the missing experts from the drives into each gpu's pool of slots and gives the gpus the slots and weights in two parts
# (host.py's step_io, in c alongside the graph), the gpus wait for each (host_wait) and run their units of the experts over the pool
# DEV=AMD:HIP PYTHONPATH=. python3 extra/k3/engine.py trunk.gguf "prompt"
import ctypes, functools, itertools, os, signal, sys, time
import numpy as np
from tinygrad import Tensor, Device, TinyJit, UOp, dtypes, nn
from tinygrad.helpers import DEBUG, getenv, prod
from tinygrad.uop.ops import Ops
from tinygrad.llm.model import Transformer, GatedDeltaNetBlock, gated_activation
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import Linear, amd_custom_kernels_supported, partial_sum
from tinygrad.llm.kernels.amd_gguf import expert_linear, q8_0_linear, QUANT_NAMES, HALFWORD_QUANTS
from extra.k3.host import IoT, host_c, wait, reads, free_bytes, host_tensor
from extra.k3.draft import Eagle3

# TAU: a token's experts under this weight left out unless cached, TAUC: those under it left out cached or not, BETA: a cached expert's
# routing bonus, FILL: the experts tau left out read anyway (a layer), PREFETCH: the next layer's experts guessed a layer. SLOTS: on a gpu
# (default: what's free but SPARE bytes)
TAU, TAUC, BETA, FILL, PREFETCH = getenv("TAU", 0.065), getenv("TAUC", 0.0), getenv("BETA", 0.07), getenv("FILL", 2), getenv("PREFETCH", 5)
SLOTS, SPARE = getenv("SLOTS", 0), getenv("SPARE", 2 << 30)

class Engine:
  def __init__(self, model:Transformer, m:dict, slots:int=SLOTS, tokens:int=1, guess:str="", draft:Eagle3|None=None, logits:bool=False,
               chunk:int=0):
    """the moe blocks of model with their routed experts streamed from the store of manifest m into a pool of slots on each gpu, a part of
    every expert each (its units), in steps of a token or (a draft's verify steps) of tokens or (a prompt's) chunk tokens, keeping their
    logits or not"""
    self.model, self.m, self.devs, self.draft, self.keep_logits = model, m, model.token_embd.weight.device, draft, logits
    blocks = [model.blk[b] for b in m["layers"] if b < len(model.blk)]
    L, E, K, G, self.lat, self.units, self.part = len(blocks), m["n_experts"], model.config.num_experts_per_tok, len(self.devs), m["lat"], \
      m["units"][0], m["part_bytes"][0]
    assert len(set(m["part_bytes"])) == 1 and len(set(m["units"])) == 1 and len(m["units"]) == G, "a part of every expert on each gpu"
    assert E <= 4096 and K <= 64 and tokens <= 8 and tokens * K <= 512 and 32 * tokens <= 1024 and (not chunk or E <= 1024), \
    "past the host's fixed arrays"
    self.tokens, self.chunk, n = tokens, chunk, max(tokens, chunk)  # (n: a step's most tokens)
    self.qtype, self.block = {n.upper(): t for t, n in QUANT_NAMES.items()}[m["expert_type"]], m["block"]
    # each layer's host buffers on every gpu: its tag, router scores (a token's rows: the router's, the next layer's guess; a gpu its experts')
    # and the two parts of a step's slots and weights, [ready | slots | weights]: the union of the tokens' first k1 experts, then of the rest, their
    # weights a token's (entry j's at j*T + t); a chunk's in part 1, an entry a token's expert (j*T + t, token t's j-th)
    self.ks, self.act = (K // 2, K - K // 2), functools.partial(gated_activation, config=model.config)
    bufs = [[host_tensor(self.devs, sz, dt) for sz, dt in ((1, dtypes.uint32), (n * 2 * E // G, dtypes.float32),
                                                           *((1 + max(k * tokens * (1 + tokens), 2 * K * chunk), dtypes.uint32) for k in self.ks))]
            for _ in blocks]
    self.tags, self.parts = [b[0][0].uop.reshape(1, 1).unshard(0) for b in bufs], [[b[2 + i][0] for b in bufs] for i in range(2)]
    self.scores = [b[1][0].uop.reshape(n, 2, E // G).unshard(2) for b in bufs]
    # the scorers: a layer's router and the guess of the next layer's experts on its input (GUESS=dir: guess-<block>.npy maps fitted on traces,
    # else the next layer's router), a gpu its experts' rows of each (in q8_0 where there's a kernel). the routers' memory freed: the scorers route
    routers = [b.ffn_gate_inp.weight.float().numpy() for b in blocks]
    def guessed(li:int) -> np.ndarray: return np.load(f) if os.path.exists(f:=f"{guess}/guess-{m['layers'][li]}.npy") else routers[li]
    def scorer(w:np.ndarray) -> Tensor|Linear:  # (rows a gpu's: its experts' of the router, then of the guess)
      w = Tensor(w.reshape(2, G, E // G, -1).transpose(1, 0, 2, 3).reshape(2 * E, -1).astype(np.float16))
      return q8_0_linear(w.to(self.devs[0]), self.devs, 0) if amd_custom_kernels_supported(self.devs) else w.shard(self.devs, 0).realize()
    self.scorers = [scorer(np.stack([r, guessed(min(li + 1, L - 1))])) for li, r in enumerate(routers)]
    self.bias = [b.exp_probs_b["bias"].float().numpy() for b in blocks]
    for b in model.blk:  # (a verify step's recurrent states after its first token kept: a rollback)
      if isinstance(b, GatedDeltaNetBlock): b.keep = tokens - 1
    for li, b in enumerate(blocks):
      b._routed = functools.partial(self.routed, li)  # type: ignore[method-assign]
      b.ffn_gate_inp.weight = Tensor.empty(0)
    # each gpu's pool in the experts' words (a bitcast of it is a copy), of what's free after the blocks' caches and states
    weights = {id(t) for t in nn.state.get_parameters(model.blk)}  # (not realized here: some are decoded in the step's graph)
    for b in model.blk: b._init_state(Tensor.empty(1, 1, model.config.dim, device=self.devs))
    Tensor.realize(*(states:=[t for t in nn.state.get_parameters([model.blk, draft]) if id(t) not in weights and t.numel()]))
    # (the empty ones allocated too, else at their first use: the ones with a buffer, not the folded constants)
    for buf in (t.uop.base.buffer for t in states if t.uop.base.op in (Ops.BUFFER, Ops.UNSHARD)):
      for x in getattr(buf, "bufs", [buf]): x.ensure_allocated()
    model.save([])  # (and the two sets of saved recurrent states)
    self.slots = slots = slots or (min(free_bytes(d) for d in self.devs) - SPARE) // self.part
    self.word = dtypes.uint16 if self.qtype in HALFWORD_QUANTS else dtypes.uint32
    self.pool = Tensor.zeros(G * slots * self.part // self.word.itemsize, dtype=self.word, device=self.devs, axis=0).realize()
    # the host's state, kept by step_io: each code's (layer * E + expert) slot (-1: not cached) and each slot's code (-1: free), the decayed
    # use counts, the guesses' reads in flight, the guesses their layer hasn't routed, pinned slots, the victims of the next reads
    self.state = {"slot_of": np.full(L * E, -1, np.int32), "expert_in": np.full(slots, -1, np.int64), "freq": np.zeros(L * E, np.float32),
                  "flying": np.zeros(L * E, np.int32), "guess": np.zeros(L * E, np.uint8), "mask": np.zeros(slots, np.uint8),
                  "victims": np.zeros(max(32 * tokens, E), np.int64), "bias": np.array([b.ctypes.data for b in self.bias], np.uint64),
                  "bufs": np.array([[[v.ctypes.data for v in views] for _, views in b] for b in bufs], np.uint64)}
    self.io = IoT(layers=L, E=E, ngpu=G, k=K, kmin=K // 4, k1=self.ks[0], rows=2, slots=slots, tau=TAU, tauc=TAUC, beta=BETA, pf=PREFETCH,
                  fill=FILL, decay=0.5 ** (1 / 8))
    for name, a in self.state.items(): setattr(self.io, name, a.ctypes.data)
    # a step's graph for each count of tokens, their tokens and the draft's given tokens in host memory
    self.v_pos = UOp.variable("start_pos", 0, model.max_context - 1)
    (self.tok, (self.tok_np,)), (self.given, (self.given_np,)) = (host_tensor(self.devs[0], n, dtypes.int32) for _ in range(2))
    self.res, (self.res_np,) = host_tensor(self.devs[0], 1 + 2 * n, dtypes.int32)  # (a step's [done | tokens | guesses])
    self.jits = {T: TinyJit(functools.partial(self.run, T)) for T in sorted({1, tokens, chunk} - {0})}
    # a draft's chain of guesses after its first (a verify step's other tokens): its steps on its own hidden states (dx: a step's), from a row
    self.cres, (self.cres_np,) = host_tensor(self.devs[0], max(tokens - 1, 1), dtypes.int32)  # ([done | guesses])
    self.dx, self.v_row = Tensor.empty(n, model.config.dim, device=self.devs).realize(), UOp.variable("row", 0, tokens - 1)
    self.chain_jit = TinyJit(self.run_chain) if draft is not None and tokens > 2 else None
    self.times, self.verified = np.zeros(2), np.zeros(3)  # (launching the graphs, after the last layer), (verify steps, guesses, right ones)

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

  def run(self, T:int, pos:UOp) -> tuple[Tensor, Tensor]:
    """the greedy token after each of the step's tokens and (a draft) its guess of the token after that (after the given one where there's
    one: a prompt's next) into host memory, then its done word (the host reads them without a sync), their logits (kept, else done)"""
    logits, hidden = self.forward(self.tok[:T].reshape(1, T), pos)
    out, dx = (toks:=logits.argmax(-1)), []
    if self.draft is not None:
      given = self.given[:T].to(toks.device)
      guess, x, _ = self.draft((given >= 0).where(given, toks).reshape(1, T), hidden, pos)
      out = toks.cat(guess[0].argmax(-1))
      dx = [Tensor(self.dx.uop)[:T].assign(x[0].float())]
    outs = Tensor(self.res.uop)[1:1 + out.shape[0]].assign(out.to(self.devs[0]).cast(dtypes.int32))
    done = Tensor(self.res.uop.after(outs.uop))[:1].assign(Tensor.ones(1, dtype=dtypes.int32, device=self.devs[0]))
    if not self.keep_logits: logits = done  # (realized, the logits are a buffer of their own: only for an eval)
    Tensor.realize(done, logits, *dx)
    return done, logits

  def run_chain(self, pos:UOp, row:UOp) -> Tensor:
    """the draft's next guesses after the given one (a token at pos+1), a step each on its own hidden state of the step before (from its
    row of the verify step's) into host memory, then the done word"""
    tok, hid, cache, out = self.given[:1].to(self.devs), Tensor(self.dx.uop)[row:row + 1], None, []
    for k in range(self.cres_np.size - 1):
      logits, hid, cache = self.draft(tok.reshape(1, 1), None, pos + k, hid=hid.reshape(1, 1, -1), cache=cache)
      out.append(tok:=logits[0].argmax(-1))
    outs = Tensor(self.cres.uop)[1:].assign(Tensor.cat(*[t.to(self.devs[0]) for t in out]).cast(dtypes.int32))
    return Tensor(self.cres.uop.after(outs.uop))[:1].assign(Tensor.ones(1, dtype=dtypes.int32, device=self.devs[0])).realize()

  def routed(self, li:int, x:Tensor, h:Tensor) -> Tensor:
    """layer li's routed experts on the tokens x (1, T, dim), h their expert inputs (1, T, lat): the gpus' scores to the host, each gpu's units
    of the experts it gives in two parts (a chunk: in one, an entry a token's expert), summed over the gpus"""
    T, s = x.shape[1], w(x[0]) if isinstance(w:=self.scorers[li], Linear) else x[0] @ w.T
    scores = Tensor(self.scores[li])[:T].assign(s.float().reshape(-1, len(self.devs), 2, self.m["n_experts"] // len(self.devs))
                                                .permute(0, 2, 1, 3).reshape(-1, 2, self.m["n_experts"]))
    if T > self.tokens: return self.experts(h[0], self.host(li, 0, scores), sum(self.ks) * T, 1).sum(0).unsqueeze(0)
    out = self.experts(h[0], self.host(li, 0, scores), self.ks[0] * T, T).contiguous()  # (a buffer: part 2 waited for after it)
    return partial_sum(out + self.experts(h[0], self.host(li, 1, out), self.ks[1] * T, T)).unsqueeze(0)

  def host(self, li:int, i:int, dep:Tensor) -> Tensor:
    """part i of layer li's slots and weights once the host releases it, waited for after dep (part 1: the scores, the host told they're in:
    a gpu's tag set 1 after its scores, 0 on a NaN)"""
    if i == 0: dep = Tensor(self.tags[li]).assign(((s:=dep[0, 0].reshape(len(self.devs), -1)[:, :1]) == s).cast(dtypes.uint32))
    return Tensor((t:=self.parts[i][li].uop).after(wait(t, dep.uop)))

  def experts(self, h:Tensor, table:Tensor, n:int, per:int) -> Tensor:
    """each gpu's units of the n experts of a table (n slots, padded with slot 0, then their rows' weights) on h (T, lat): (gpus, T, lat). an
    expert on per tokens: T (every token, its weights loaded once, a row's weight at j*T + t) or 1 (entry j*T + t token t's, a chunk's). the
    act in the q8 of the down gemv's input, the down gemv a token's sum"""
    T, u, off = h.shape[0], self.units, 2 * self.units * self.lat // self.block[0] * self.block[1]
    slots, w = table[1:1 + n].bitcast(dtypes.int32), table[1 + n:1 + n + n * per].bitcast(dtypes.float32)
    if amd_custom_kernels_supported(h.device):
      gu = expert_linear(self.pool, slots, h, 2 * u, self.lat, self.qtype, self.part, split=True, per=per)
      return expert_linear(self.pool, slots, gu, self.lat, u, self.qtype, self.part, off, act=self.act, w=w, rows=T, per=per)
    slots = slots.reshape(n, 1).expand(n, per).flatten()  # (a row a slot and token)
    down = self.gemv(slots, self.act(*self.gemv(slots, h, 2 * u, self.lat, 0).split(u, -1)), self.lat, u, off)
    return (down * w.reshape(n * per, 1)).reshape(-1, n * per // T, T, self.lat).sum(1)

  def gemv(self, slots:Tensor, x:Tensor, out_f:int, in_f:int, off:int) -> Tensor:
    """(gpus, n, out_f): W @ x[k % rows] (or a gpu's x[k]) of the W at off bytes in slots[k] of each gpu's pool, decoded (no kernel)"""
    n, size, isz = slots.shape[0], out_f * in_f // self.block[0] * self.block[1], self.word.itemsize
    w = self.pool.reshape(-1, self.slots, self.part // isz)[:, slots][..., off // isz:(off + size) // isz].bitcast(dtypes.uint8)
    w = ggml_data_to_tensor(w.flatten(), prod(w.shape) // self.block[1] * self.block[0], self.qtype).reshape(-1, n, out_f, in_f)
    return (w * (x if x.ndim == 3 else x[Tensor.arange(n) % x.shape[0]]).unsqueeze(-2)).sum(-1)

  # *** the host's side

  def warm(self):
    """every step's graph (and the rollback's) captured on dry steps: the host's flags preset (its tables still zero: no experts)"""
    assert all(Device[d].has_copy_queue for d in self.devs), "the host syncs run on the gpus' copy queues"
    host_c().flags(ctypes.byref(self.io), 1)
    for jit in self.jits.values():
      for _ in range(3): jit(self.v_pos.bind(0))[0].tolist()
    for n, _ in itertools.product(range(1, self.tokens), range(3)): self.model.rollback(n)  # (their jits captured too, after a verify step)
    for _ in range(3 if self.chain_jit else 0): self.chain_jit(self.v_pos.bind(0), self.v_row.bind(0)).tolist()

  def step(self, toks:list[int], pos:int, given:list[int]|None=None) -> tuple[list[int], list[int]]:
    """a step of the tokens at positions pos..: the gpus run its graph, the host every layer's routing and reads. the greedy token after each
    and the draft's guess after that (their logits kept)"""
    assert DEBUG < 2, "DEBUG>=2 syncs in the step's launch, before the host's part runs: a hang"
    io, T, t0 = self.io, len(toks), time.perf_counter()
    io.tokens, io.chunk, io.victims_n, self.tok_np[:T], self.given_np[:T], self.res_np[0] = T, T > self.tokens, 32 * T, toks, \
      ((given or []) + [-1] * T)[:T], 0
    host_c().flags(ctypes.byref(io), 0)
    _, self.logits = self.jits[T](self.v_pos.bind(pos))
    t1 = time.perf_counter()
    err = host_c().step_io(ctypes.byref(io))
    if err: host_c().flags(ctypes.byref(io), 1)  # (the gpus run the graph out: one killed waiting on a layer wedges)
    t2 = time.perf_counter()
    while self.res_np[0] != 1:
      if time.perf_counter() - t2 > 60: raise RuntimeError("the step's graph didn't finish")
    if err: raise RuntimeError({-1: "a read failed", -2: "timeout", -3: "no victims left"}[err])
    ret = self.res_np[1:1 + (2 * T if self.draft is not None else T)].tolist()
    self.times += (t1 - t0, time.perf_counter() - t2)
    return ret[:T], ret[T:] or ret

  def chain(self, g:int, pos:int, row:int) -> list[int]:
    """the draft's guesses after g (at pos+1, guessed at the step's row) for a verify step"""
    if self.chain_jit is None: return []
    self.given_np[0], self.cres_np[0] = g, 0
    self.chain_jit(self.v_pos.bind(pos), self.v_row.bind(row))
    st = time.perf_counter()
    while self.cres_np[0] != 1:
      if time.perf_counter() - st > 60: raise RuntimeError("the draft's chain didn't finish")
    return self.cres_np[1:].tolist()

  # *** master's model's generate (LLMServer's): the caches only go forward, a prompt that extends the tokens fed (or the last prompt) feeds
  # the rest

  @property
  def max_context(self) -> int: return self.model.max_context
  def get_start_pos(self, tokens:list[int]) -> int: return self.model.get_start_pos(tokens)

  def prefill(self, tokens:list[int], pos:int) -> int:
    """tokens from pos on but the last fed in steps of a chunk (exact routing, every expert of a layer read in), the position after them"""
    if not self.chunk: return pos
    end = pos + (len(tokens) - 1 - pos) // self.chunk * self.chunk
    for p in range(pos, end, self.chunk): self.step(tokens[p:p + self.chunk], p, tokens[p + 1:p + 1 + self.chunk])
    return end

  def generate(self, tokens:list[int], temperature:float=0.0):
    """the greedy tokens after tokens (any temperature), appended as they come: fed from get_start_pos in chunks (prefill) and the rest a token
    a step, then a token a step or (a draft) a verify step of [the next, the draft's chain of guesses after it]: the guesses kept up to the first
    wrong one, the target's token after them next, the recurrent states rolled back to after the kept ones"""
    start = self.get_start_pos(tokens)
    if (fork:=self.model.fork(tokens, start, self.chunk or 1)) > start:  # (where tokens diverge from the tokens fed before: its states saved)
      for pos in range(self.prefill(tokens[:fork + 1], start), fork): self.step(tokens[pos:pos + 1], pos, tokens[pos + 1:pos + 2])
      self.model.save(tokens[:fork])
    for pos in range(self.prefill(tokens, fork), len(tokens)):
      (t,), (g,) = self.step(tokens[pos:pos + 1], pos, tokens[pos + 1:pos + 2])
    self.model.save(tokens)
    row = 0
    while len(tokens) + self.tokens <= self.max_context:  # (a verify step's last token in the context)
      tokens.append(t)
      self.model._cached_tokens = tokens[:-1]
      yield t
      if self.draft is None:
        (t,), (g,) = self.step([t], len(tokens) - 1)
        continue
      gs = [g, *self.chain(g, len(tokens) - 1, row)]
      toks, guesses = self.step([t, *gs], len(tokens) - 1)
      row = next((j for j, g in enumerate(gs) if toks[j] != g), len(gs))
      self.verified += (1, len(gs), row)
      for g in gs[:row]:
        tokens.append(g)
        self.model._cached_tokens = tokens[:]
        yield g
      if row < len(gs): self.model.rollback(row + 1)
      t, g = toks[row], guesses[row]

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
  # DEADLINE=s: the run stops between steps once it's s seconds in (killed mid-step, the gpus are left waiting on the host). SIGTERM as SIGINT:
  # a clean exit (the devices finalized)
  signal.signal(signal.SIGTERM, signal.default_int_handler)
  st = time.perf_counter()
  def late() -> bool: return time.perf_counter() - st > getenv("DEADLINE", 540)
  model, kv = Transformer.from_gguf(sys.argv[1], max_context=getenv("CTX", 4096), shard=getenv("SHARD", 4))
  tok = SimpleTokenizer.from_gguf_kv(kv)
  print(f"model in {time.perf_counter() - st:.0f} s", flush=True)
  disks, m = open_store(list(model.token_embd.weight.device), getenv("STORE", 512 << 30))  # (k3's iq3_xxs store)
  # DRAFT=dir: the eagle head, speculative decoding with T=2 verify steps (memory: fewer SLOTS)
  draft = Eagle3(d, lambda t: model.token_embd(t).float(), model.token_embd.weight.device, model.max_context) if (d:=getenv("DRAFT", "")) \
    else None
  # VERIFY=n: a verify step's tokens, the next and the draft's chain of n-1 guesses (an EVAL: its steps of n tokens)
  # CHUNK=n: a prompt fed in steps of n tokens (0: a token a step)
  eng = Engine(model, m, tokens=getenv("VERIFY", 2 if draft else 1), guess=getenv("GUESS", ""), draft=draft, logits=bool(getenv("EVAL", "")),
               chunk=getenv("CHUNK", 256))
  eng.drives(disks)
  eng.warm()
  print(f"warm in {time.perf_counter() - st:.0f} s, {eng.slots} slots a gpu", flush=True)
  if (spec:=getenv("EVAL", "") or getenv("BENCH", "")):  # EVAL=/BENCH=corpus.jsonl:nseq:ntok:first (eval.py)
    from extra.k3.eval import evaluate, bench
    (evaluate if getenv("EVAL", "") else bench)(eng, spec, late)
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
  steps, guesses, right = eng.verified
  if draft: print(f"{right / guesses:.1%} of {guesses:.0f} guesses right, {(steps + right) / steps:.2f} tokens a step")
  eng.report(c0, len(out) - 1)
