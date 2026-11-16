#!/usr/bin/env python3
# K3 decode with the routed experts streamed from the nvme drives: master's model (a trunk gguf without the routed experts, sharded over the
# gpus) in a jitted step graph of the engine's own forward. at a moe block every gpu writes its router scores and a tag to host memory, the
# host routes, reads the missing experts from the drives into each gpu's pool of slots and gives the gpus the slots and weights in two parts
# (host.py's step_io, in c alongside the graph), the gpus wait for each (host_wait) and run their units of the experts over the pool
# DEV=AMD:HIP PYTHONPATH=. python3 extra/k3/engine.py trunk.gguf "prompt"
import collections, ctypes, functools, itertools, math, os, signal, sys, time
import numpy as np
from typing import Callable, cast
from tinygrad import Tensor, TinyJit, UOp, Device, dtypes, nn
from tinygrad.dtype import DType
from tinygrad.helpers import DEBUG, getenv, prod, round_up
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.uop.ops import Ops
from tinygrad.llm.model import Transformer, FFNBlock, GatedDeltaNetBlock, AttentionResidual, gated_activation
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernels.amd import Linear, amd_custom_kernels_supported, partial_sum, allgather_last
from tinygrad.llm.kernels.amd_gguf import expert_linear, expert_tiles, moe_route, q8_0_linear, QUANT_NAMES, HALFWORD_QUANTS, IQ3_XXS
from extra.k3.host import IoT, host_c, wait, reads, free_bytes, host_tensor
from extra.k3.draft import Eagle3, DSpark, load

# TAU: a token's experts under this weight left out unless cached, TAUC: those under it left out cached or not, BETA: a cached expert's
# routing bonus, FILL: the experts tau left out read anyway (a layer), PREFETCH: the next layer's experts guessed a layer. SLOTS: on a gpu
# (default: what's free but SPARE bytes)
TAU, TAUC, BETA, FILL, PREFETCH = getenv("TAU", 0.065), getenv("TAUC", 0.0), getenv("BETA", 0.07), getenv("FILL", 2), getenv("PREFETCH", 5)
SLOTS, SPARE = getenv("SLOTS", 0), getenv("SPARE", 2 << 30)
# (a window's seconds: a fixed part and a token's, a step's (of a token) before any are measured: tinys3's, 4.8 s and 1.26 ms, 50 ms)
WINDOW_S, WINDOW_TOKEN_S, STEP_S = 5.0, 1.3e-3, 0.05
SESSION = 4096  # (a kept session's rows a short request that diverges from it early may write)
TILES = (16, 64)  # (a window's tiles of an expert's rows: the first that holds its rows an expert on average)

class Engine:
  def __init__(self, model:Transformer, m:dict, slots:int=SLOTS, tokens:int=1, guess:str="", draft:Eagle3|DSpark|None=None, logits:bool=False,
               chunk:int=0, window:int=0, taps:list[tuple[int, bool]]|None=None, widths:tuple[int, ...]=()):
    """the moe blocks of model with their routed experts streamed from the store of manifest m into a pool of slots on each gpu, a part of
    every expert each (its units), in steps of a token or (a draft's verify steps) of tokens, a prompt in windows of up to window tokens (their
    attention in chunks), keeping their logits or not. taps: without a draft, the (block, mixed) streams a window keeps (draft.taps). widths:
    a scored draft's verify steps of fewer tokens too (their graphs captured), each step's the one its guesses' scores expect most tokens a
    second of"""
    self.model, self.m, self.devs, self.draft, self.keep_logits = model, m, model.token_embd.weight.device, draft, logits
    self.t_stage = time.perf_counter()
    self.dump_taps = taps or []
    blocks = [model.blk[b] for b in m["layers"] if b < len(model.blk)]
    L, E, K, G, self.lat, self.units, self.part = len(blocks), m["n_experts"], model.config.num_experts_per_tok, len(self.devs), m["lat"], \
      m["units"][0], m["part_bytes"][0]
    assert len(set(m["part_bytes"])) == 1 and len(set(m["units"])) == 1 and len(m["units"]) == G, "a part of every expert on each gpu"
    assert E <= 4096 and K <= 64 and tokens <= 8 and tokens * K <= 512 and 32 * tokens <= 1024, "past the host's fixed arrays"
    self.tokens, self.chunk, n = tokens, chunk, max(tokens, chunk, window)  # (n: a step's or a window's most tokens)
    self.qtype, self.block = {n.upper(): t for t, n in QUANT_NAMES.items()}[m["expert_type"]], m["block"]
    # each layer's host buffers on every gpu: its tag, router scores (a token's rows: the router's, the next layer's guess; a gpu its experts')
    # and the two parts of a step's slots and weights, [ready | slots | weights]: entry j*T+t is token t's j-th expert and its weight (a
    # window's: [ready | an expert's slot], [ready | an expert's count])
    self.ks, self.act = (K // 2, K - K // 2), functools.partial(gated_activation, config=model.config)
    part = [1 + max(2 * k * tokens, E) for k in self.ks]  # (a part's words)
    bufs = [[host_tensor(self.devs, sz, dt) for sz, dt in ((1, dtypes.uint32), (n * 2 * E // G, dtypes.float32), *((p, dtypes.uint32) for p in part))]
            for _ in blocks]
    self.bufs, self.tags, self.parts = bufs, [b[0][0].uop.reshape(1, 1).unshard(0) for b in bufs], [[b[2 + i][0] for b in bufs] for i in range(2)]
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
    weights = {id(t) for t in nn.state.get_parameters(model.blk)}  # (not realized here: some are decoded in the step's graph)
    # (a draft holding the trunk's weights would realize them as its state: decoded, their quantized kernels lost)
    assert not {id(t) for t in nn.state.get_parameters(draft)} & {id(t) for t in nn.state.get_parameters(model)}, "a draft with the trunk's weights"
    for b in model.blk: b._init_state(Tensor.empty(1, 1, model.config.dim, device=self.devs))
    # the context's caches (the mla blocks' latents, their indexers' keys, the draft's): lent from the pool (never allocated here)
    owners = [(o, k) for b in model.blk for o, k in ((b, "cache_k"), (getattr(b, "indexer", None), "cache_kg")) if hasattr(o, k)]
    self.caches = owners + ([(draft, "cache")] if draft is not None else [])
    weights |= {id(getattr(o, k)) for o, k in self.caches}
    Tensor.realize(*(states:=[t for t in nn.state.get_parameters([model.blk, draft]) if id(t) not in weights and t.numel()]))
    # (the empty ones allocated too, else at their first use: the ones with a buffer, not the folded constants)
    for buf in (t.uop.base.buffer for t in states if t.uop.base.op in (Ops.BUFFER, Ops.UNSHARD)):
      for x in getattr(buf, "bufs", [buf]): x.ensure_allocated()
    del states
    model.save([])  # (and the two sets of saved recurrent states)
    # a step's graph for each count of tokens, their tokens and the draft's given tokens in host memory
    self.v_pos, self.v_live = UOp.variable("start_pos", 0, model.max_context - 1), UOp.variable("live", 1, max(chunk, 1))
    # (a read window's tokens: up to the context's)
    (self.tok, (self.tok_np,)), (self.given, (self.given_np,)) = (host_tensor(self.devs[0], max(n, model.max_context), dtypes.int32)
                                                                  for _ in range(2))
    # ([done | tokens | guesses | their scores (a scored draft's, f32)])
    self.res, (self.res_np,) = host_tensor(self.devs[0], 1 + n * (1 + 2 * max(self.width, 1)), dtypes.int32)
    self.jits = {T: TinyJit(functools.partial(self.run, T)) for T in sorted({1, tokens, *widths})}
    self.keep_jits = (TinyJit(self.keep), TinyJit(self.unkeep))
    self.widths = sorted({tokens, *widths}) if draft is not None and draft.scored else [tokens]
    self.step_s:dict[int, collections.deque] = {T: collections.deque(maxlen=64) for T in self.jits}  # (measured: a step's seconds)
    # a draft's chain of guesses after its first (a verify step's other tokens): its steps on its own hidden states (dx: a step's), from a row
    self.cres, (self.cres_np,) = host_tensor(self.devs[0], max(tokens - 1, 1), dtypes.int32)  # ([done | guesses])
    self.dx, self.v_row = Tensor.empty(n, model.config.dim, device=self.devs).realize(), UOp.variable("row", 0, tokens - 1)
    self.chain_jit = TinyJit(self.run_chain) if draft is not None and draft.width < tokens - 1 else None
    self.times, self.verified = np.zeros(2), np.zeros(3)  # (launching the graphs, after the last layer), (verify steps, guesses, right ones)
    self.window_s:collections.deque = collections.deque(maxlen=64)  # (measured: a window's tokens, seconds)
    # each gpu's pool in the experts' words (a bitcast of it is a copy), of what's free after the blocks' caches and states and the window's
    # graphs (captured on a pool of a slot), then the slots a window's state is lent (its own while it runs): slots of decode's otherwise
    self.word = dtypes.uint16 if self.qtype in HALFWORD_QUANTS else dtypes.uint32
    def pool(n:int) -> Tensor: return Tensor.zeros(G * n * self.part // self.word.itemsize, dtype=self.word, device=self.devs, axis=0).realize()
    self.W = 0
    if window: self.window_init(window)
    # (the caches lent from the pool's first slots (a gpu its shard of each), their rows of positions in stripes: the rows a context reached
    # kept from the experts (grow), the rest the experts' (never read))
    shapes = {str(i): (getattr(o, k).uop.shard_shape, getattr(o, k).dtype) for i, (o, k) in enumerate(self.caches)}
    axes = [getattr(o, k).uop.axis for o, k in self.caches]
    # a kept session (a long one a short request diverges from early): copies of the caches' rows that request may write, its recurrent states
    self.B, self.session = min(SESSION, model.max_context // 2), cast(tuple[list[int], int]|None, None)
    self.v_at = UOp.variable("at", 0, model.max_context - self.B)
    def rows0(t:Tensor) -> tuple[int, ...]: return tuple(self.B if j == next(j for j, d in enumerate(t.shape) if d >= model.max_context) else d
                                                         for j, d in enumerate(t.shape))
    self.stash = [Tensor.empty(*rows0(t:=getattr(o, k)), dtype=t.dtype, device=self.devs, axis=t.uop.axis).realize() for o, k in self.caches]
    self.stash_states = [s.clone().realize() for s in model._states()]
    for o, k in self.caches: setattr(o, k, None)
    free = (lambda: min(free_bytes(d) for d in self.devs)) if not slots else (lambda: 0)  # (what's free on the gpus: sizing the pool by it)
    self.stage(f"states ({free() / 1e9:.2f} GB free before the pool)")
    # (lent: the window's compute state and a piece's, the slots of more pieces lent at a prefill)
    self.win, lent, self.layer = {}, self.slots_of(self.window_bytes) + self.slots_of(self.piece_bytes) if window else 0, 0
    self.tail = slots or (free() - SPARE) // self.part - lent
    self.slots = slots = self.tail + lent
    self.lent_from = self.tail
    self.pool = pool(slots)
    caches, self.cache_bytes = self.lend(shapes, 0)
    # (a cache's: its positions' axis, its first byte, a position's bytes, its blocks of every position's rows and their bytes)
    self.stripes = [(a, caches[i].uop.base.buffer.bufs[0].offset, row, prod(shape[:a]), shape[a] * row) for i, (shape, dt) in shapes.items()
                    for a in [next(j for j, d in enumerate(shape) if d >= model.max_context)] for row in [prod(shape[a + 1:]) * dt.itemsize]]
    for i, (o, k) in enumerate(self.caches): setattr(o, k, caches[str(i)] if axes[i] is None else Tensor(caches[str(i)].uop.unshard(axes[i])))
    self.grown = 0
    if window:
      self.stage(f"pool: {slots} slots ({slots * self.part / 1e9:.2f} GB, of it lent: the caches' {self.cache_bytes / 1e9:.2f} GB as they grow, "
                 f"the window's {(slots - self.tail) * self.part / 1e9:.2f} GB)")
      f0 = free()
      self.window_lend()
      self.stage(f"window graphs (their scratch {(f0 - free()) / 1e9:.2f} GB of SPARE's {SPARE / 1e9:.2f})")
    # the host's state, kept by step_io: each code's (layer * E + expert) slot (-1: not cached) and each slot's code (-1: free), the decayed
    # use counts, the guesses' reads in flight, the guesses their layer hasn't routed, pinned slots, the victims of the next reads
    self.state = {"slot_of": np.full(L * E, -1, np.int32), "expert_in": np.full(slots, -1, np.int64), "freq": np.zeros(L * E, np.float32),
                  "flying": np.zeros(L * E, np.int32), "guess": np.zeros(L * E, np.uint8), "mask": np.zeros(slots, np.uint8),
                  "victims": np.zeros(max(32 * tokens, E), np.int64), "bias": np.array([b.ctypes.data for b in self.bias], np.uint64),
                  "bufs": np.array([[[v.ctypes.data for v in views] for _, views in b] for b in bufs], np.uint64)}
    self.io = IoT(layers=L, E=E, ngpu=G, k=K, kmin=K // 4, k1=self.ks[0], rows=2, slots=slots, tau=TAU, tauc=TAUC, beta=BETA, pf=PREFETCH,
                  fill=FILL, decay=0.5 ** (1 / 8))
    for name, a in self.state.items(): setattr(self.io, name, a.ctypes.data)

  def drives(self, disks:list[list[str]]):
    """step_io's reads from the store's drives (each gpu's disk) into the pools"""
    self.tables, scalars = reads(self.m, disks, self.pool.uop.buffer.bufs, len(self.bias) * self.m["n_experts"])
    for name, v in {**scalars, **{k: a.ctypes.data for k, a in self.tables.items()}}.items(): setattr(self.io, name, v)

  # *** the step's graph

  @property
  def taps(self) -> list[tuple[int, bool]]: return self.draft.taps if self.draft is not None else self.dump_taps
  @property
  def width(self) -> int: return self.draft.width if self.draft is not None else 0  # (a row's guesses)

  def mixer(self, i:int) -> AttentionResidual:  # (the attention residual that next reads the streams after block i)
    return self.model.blk[i + 1].attn_res_score if i + 1 < len(self.model.blk) else self.model.output_res_score

  def forward(self, tokens:Tensor, pos:UOp) -> tuple[Tensor, Tensor|None]:
    """the logits (T, vocab) after each of the tokens (1, T) at positions pos.., the draft's taps (1, T, taps, dim): after a block its last
    stream (the block's sum so far) or (mixed) the streams as the next block's attention residual mixes them"""
    m, hidden, taps = self.model, {}, self.draft.taps if self.draft is not None else []
    x = m.token_embd(tokens.to(m.token_embd.weight.device)).float().unsqueeze(2)  # (the attention residuals' streams: (1, T, 1, dim))
    mask = None
    for i, b in enumerate(m.blk):
      x, mask = b(x, (pos,), mask)
      for t in taps:
        if t[0] == i: hidden[t] = self.mixer(i).mix(x) if t[1] else x[:, :, -1]
    return m.output(m.output_norm(m.output_res_score.mix(x)))[0].float(), Tensor.stack(*(hidden[t] for t in taps), dim=2) if taps else None

  def run(self, T:int, pos:UOp) -> tuple[Tensor, Tensor]:
    """the greedy token after each of the step's tokens and (a draft) its guesses of the tokens after that (after the given one where there's
    one: a prompt's next) into host memory, then its done word (the host reads them without a sync), their logits (kept, else done)"""
    done, logits, *dx = self.outputs(T, pos)
    Tensor.realize(done, logits, *dx)
    return done, logits

  def outputs(self, T:int, pos:UOp) -> list[Tensor]:
    """run's tensors: the done word, the logits (kept, else the done word) and the draft's hidden states"""
    logits, hidden = self.forward(self.tok[:T].reshape(1, T), pos)
    out, dx = (toks:=logits.argmax(-1)), []
    if self.draft is not None:
      # the token after each row (the given one, else the target's) and the rows kept (the step's tokens it matches from the second on): the
      # draft's guesses after the last kept row's
      truth = ((g:=self.given[:T].to(toks.device)) >= 0).where(g, toks)
      kept = ((self.tok[1:T].to(toks.device) == truth[:T - 1]).cast(dtypes.int32).cumsum(0) == Tensor.arange(1, T).to(toks.device)).sum() \
        if T > 1 else 0
      guesses, scores, x = self.draft.guess(truth.reshape(1, T), hidden, pos, kept)
      out = toks.cat(guesses, *([] if scores is None else [scores.float().bitcast(dtypes.int32)]))
      dx = [Tensor(self.dx.uop)[:T].assign(x[0].float())] if x is not None else []
    outs = Tensor(self.res.uop)[1:1 + out.shape[0]].assign(out.to(self.devs[0]).cast(dtypes.int32))
    done = Tensor(self.res.uop.after(outs.uop))[:1].assign(Tensor.ones(1, dtype=dtypes.int32, device=self.devs[0]))
    return [done, logits if self.keep_logits else done, *dx]  # (realized, the logits are a buffer of their own: only for an eval)

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
    of the experts it gives in two parts, summed over the gpus"""
    T, s = x.shape[1], w(x[0]) if isinstance(w:=self.scorers[li], Linear) else x[0] @ w.T
    scores = Tensor(self.scores[li])[:T].assign(s.float().reshape(-1, len(self.devs), 2, self.m["n_experts"] // len(self.devs))
                                                .permute(0, 2, 1, 3).reshape(-1, 2, self.m["n_experts"]))
    out = self.experts(h[0], self.host(li, 0, scores), self.ks[0] * T).contiguous()  # (a buffer: part 2 waited for after it)
    return partial_sum(out + self.experts(h[0], self.host(li, 1, out), self.ks[1] * T)).unsqueeze(0)

  def host(self, li:int, i:int, dep:Tensor) -> Tensor:
    """part i of layer li's slots and weights once the host releases it, waited for after dep (part 1: the scores, the host told they're in:
    a gpu's tag set 1 after its scores, 2 on a NaN)"""
    if i == 0: dep = Tensor(self.tags[li]).assign(((s:=dep[0, 0].reshape(len(self.devs), -1)[:, :1]) == s).where(1, 2).cast(dtypes.uint32))
    return Tensor((t:=self.parts[i][li].uop).after(wait(t, dep.uop)))

  def experts(self, h:Tensor, table:Tensor, n:int) -> Tensor:
    """each gpu's units of the n experts of a table (n slots, padded with slot 0, their weights a token's: j*T + t) on h (T, lat): (gpus, T,
    lat). entry j*T+t is token t's j-th expert, the act in the q8 of the down gemv's input, the down gemv a token's sum"""
    T, u, off = h.shape[0], self.units, 2 * self.units * self.lat // self.block[0] * self.block[1]
    slots, w = table[1:1 + n].bitcast(dtypes.int32), table[1 + n:1 + 2 * n].bitcast(dtypes.float32)
    if amd_custom_kernels_supported(h.device):
      gu = expert_linear(self.pool, slots, h, 2 * u, self.lat, self.qtype, self.part, split=True, live=w)
      return expert_linear(self.pool, slots, gu, self.lat, u, self.qtype, self.part, off, act=self.act, w=w, rows=T)
    down = self.gemv(slots, self.act(*self.gemv(slots, h, 2 * u, self.lat, 0).split(u, -1)), self.lat, u, off)
    return (down * w.reshape(n, 1)).reshape(-1, n // T, T, self.lat).sum(1)

  def gemv(self, slots:Tensor, x:Tensor, out_f:int, in_f:int, off:int) -> Tensor:
    """(gpus, n, out_f): W @ x[k % rows] (or a gpu's x[k]) of the W at off bytes in slots[k] of each gpu's pool, decoded (no kernel)"""
    n, size, isz = slots.shape[0], out_f * in_f // self.block[0] * self.block[1], self.word.itemsize
    w = self.pool.reshape(-1, self.slots, self.part // isz)[:, slots][..., off // isz:(off + size) // isz].bitcast(dtypes.uint8)
    w = ggml_data_to_tensor(w.flatten(), prod(w.shape) // self.block[1] * self.block[0], self.qtype).reshape(-1, n, out_f, in_f)
    return (w * (x if x.ndim == 3 else x[Tensor.arange(n) % x.shape[0]]).unsqueeze(-2)).sum(-1)

  # *** a prompt's window: the blocks a layer at a time over up to W tokens (layer major: a layer's experts read once a window). a layer's
  # attention, the mixes and the shared experts in steps of a chunk, its tokens' routed experts once over the window. the residual streams of
  # the window's tokens in a bank, a layer adds to its last one. the graphs are a kind's (blocks of the same buffers): a block's buffers its
  # args, the first of the kind's block captures it

  def window_init(self, W:int):
    """the window's variables and graphs (captured once its state is lent from the pool: window_capture), its state's bytes"""
    m = self.model
    self.W, S = W, 1 + sum(b.attn_res_score.new_block for b in m.blk)
    self.window_bytes, self.piece_bytes = (sum(round_up(prod(sh) * dt.itemsize, 256) for sh, dt in f().values())
                                           for f in (self.window_shapes, self.piece_shapes))
    self.v_p0 = UOp.variable("p0", 0, self.max_context - 1)  # (a piece's first token in the read window)
    self.v_c0, self.v_wlive = UOp.variable("c0", 0, W - self.chunk), UOp.variable("wlive", 1, W)
    self.v_n, self.v_s = UOp.variable("streams", 1, S), UOp.variable("stream", 0, S - 1)
    self.v_tap = UOp.variable("tap", 0, self.piece_shapes()["taps"][0][2] - 1)
    self.arenas:dict = {}  # (the window's graphs never run at the same time: one arena for their scratch)
    self.tap_jit, self.mix_jit = TinyJit(self.tap, arenas=self.arenas), TinyJit(self.tap_mix, arenas=self.arenas)
    self.fill_jit = TinyJit(self.fill, arenas=self.arenas) if self.draft else None
    # (a window's routed experts in tiles of an expert's rows sized by its rows an expert: few tokens, 16 rows, else 64)
    self.routes = {t: (TinyJit(functools.partial(self.window_route, t), arenas=self.arenas),
                       TinyJit(functools.partial(self.window_experts, t), arenas=self.arenas)) for t in TILES}
    self.start_jit = TinyJit(self.start, arenas=self.arenas)

  def window_lend(self):  # (the window's state lent from the pool's last slots, its graphs captured on it)
    self.tail = self.lent_from = self.slots - self.slots_of(self.window_bytes) - self.slots_of(self.piece_bytes)
    self.win = {**self.window_state(self.tail), **self.piece(self.slots - self.slots_of(self.piece_bytes))}
    self.window_capture()

  def window_capture(self):
    """the window's graphs captured on its lent state: every kind's on its first block (its buffers the args) on dry chunks at position 0 (the
    host's flags preset: slot 0), the routes and experts, the rest. their scratch: one arena (SPARE's)"""
    m, self.kinds = self.model, {}
    moe = [b for b in m.blk if hasattr(b, "ffn_gate_inp")]
    for b, (name, fxn) in itertools.product(m.blk, (("attend", self.attend), ("post", self.post))):
      if name == "post" and b not in moe: continue
      jit, args, first = self.kind(name, b, fxn)
      if first is not b: continue
      vs = (self.v_pos.bind(0), self.v_live.bind(self.chunk), self.v_c0.bind(0), self.v_n.bind(1), self.v_s.bind(0)) if name == "attend" else \
        (self.v_live.bind(self.chunk), self.v_c0.bind(0), self.v_s.bind(0))
      for _ in range(3): self.graph(jit, *vs, *args)
    for _ in range(3):
      self.graph(self.start_jit, self.v_p0.bind(0))
      self.dry(moe[0])
      self.graph(self.tap_jit, self.v_s.bind(0), self.v_tap.bind(0))
      if self.fill_jit: self.graph(self.fill_jit, self.v_pos.bind(0), self.v_p0.bind(0), self.v_c0.bind(0))
    for b in moe[1:]: self.dry(b)  # (and once on every moe layer's host buffers: their first use maps them on each gpu, a first window would wait)

  def dry(self, b:FFNBlock):  # (moe block b's route and experts in each tile size on a dry chunk)
    li = self.m["layers"].index(self.model.blk.index(b))
    for route, experts in self.routes.values():
      self.graph(route, self.v_wlive.bind(self.chunk), b.exp_probs_b["bias"], Tensor(self.tags[li]), self.parts[1][li])
      self.graph(experts, self.pool, self.parts[0][li])

  def window_shapes(self) -> dict[str, tuple[tuple[int, ...], DType]]:
    """the window's compute state (a piece of W tokens'): the moe's inputs and outputs (the routed experts' inputs and outputs, the shared
    experts' outputs, the router's logits), its routes, a gpu's rows of the experts' activations and outputs"""
    c, W, E, K = self.model.config, self.W, self.m["n_experts"], self.model.config.num_experts_per_tok
    R, f32, i32 = max(map(self.rows, TILES)), dtypes.float32, dtypes.int32
    return {"h": ((W, self.lat), f32), "shared": ((W, c.dim), f32), "logits": ((W, E), f32), "out": ((W, self.lat), f32), "w": ((W, K), f32),
            "pos": ((W * K,), i32), "entries": ((R,), i32), "tiles": ((R // TILES[0],), i32), "act": ((R, self.units), dtypes.half),
            "down": ((R, self.lat), dtypes.half)}

  def window_state(self, at:int) -> dict[str, Tensor]:  # (lent from the pool's slots from at on)
    return {k: Tensor(t.uop.unshard(0)) if k in ("act", "down") else t for k, t in self.lend(self.window_shapes(), at)[0].items()}

  def piece_shapes(self) -> dict[str, tuple[tuple[int, ...], DType]]:  # (a piece's: the bank of its W tokens' residual streams, the draft's inputs)
    c, S = self.model.config, 1 + sum(b.attn_res_score.new_block for b in self.model.blk)
    return {"bank": ((1, self.W, S, c.dim), dtypes.float), "taps": ((1, self.W, max(len(self.taps), 1), c.dim), dtypes.float)}

  def piece(self, at:int) -> dict[str, Tensor]: return self.lend(self.piece_shapes(), at)[0]  # (lent from the pool's slots from at on)

  def lend(self, shapes:dict[str, tuple[tuple[int, ...], DType]], at:int) -> tuple[dict[str, Tensor], int]:
    """buffers of shapes on every gpu lent from the pool's slots from at on, and their bytes"""
    state, off = {}, at * self.part
    for k, (shape, dt) in shapes.items():
      n = prod(shape) * dt.itemsize
      (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = [Buffer(d, n, base=p, offset=off).allocate()
                                                     for d, p in zip(self.devs, self.pool.uop.buffer.bufs)]
      state[k], off = Tensor(UOp.from_buffer(mb, dt, self.devs)).reshape(shape), off + round_up(n, 256)
    return state, off - at * self.part

  def scorer(self, b:FFNBlock) -> Tensor|Linear|None:  # (a moe block's router, the engine's)
    return self.scorers[self.m["layers"].index(i)] if (i:=self.model.blk.index(b)) in self.m["layers"] else None

  kinds:dict[tuple, tuple[FFNBlock, Callable, list[Tensor]]] = {}  # (a kind's, and a stage of a block's: its args)

  def kind(self, name:str, b:FFNBlock, fxn:Callable) -> tuple[Callable, list[Tensor], FFNBlock]:
    """b's graph of a stage (its kind's), b's buffers and its router's (its args) and the kind's first block (the one it's captured on)"""
    if (name, b) not in self.kinds:
      seen:dict[UOp, None] = {}
      for t in nn.state.get_state_dict([b, self.scorer(b)]).values():
        for u in t.uop.toposort():
          if u.op is Ops.BUFFER: seen.setdefault(u, None)
      key = (name, type(b), tuple((u.shape, u.dtype, u.nbytes()) for u in seen))
      if key not in self.kinds: self.kinds[key] = (b, TinyJit(functools.partial(fxn, b), arenas=self.arenas), [])
      self.kinds[(name, b)] = (*self.kinds[key][:2], [Tensor(u) for u in seen])
    first, jit, args = self.kinds[(name, b)]
    return jit, args, first

  def attend(self, b:FFNBlock, pos:UOp, live:UOp, c0:UOp, n:UOp, s:UOp, *_) -> Tensor:
    """a chunk's live rows' attention at c0 of the window (positions pos..) on its n streams, added to stream s, then its feed forward's mix: a
    moe's parts before the routed experts into the window's buffers, a dense block's feed forward added to stream s. the rows symbolic: a
    kernel's work the live rows'"""
    bank = Tensor(self.win["bank"].uop)
    h = b.attn_res_score.mix(bank, b.attn_norm, n, (c0, live))
    a = b._attention(h, (pos,), b._indexer_mask(h, (pos,), None), live)
    bank = Tensor(self.win["bank"].uop.after(self.add(s, c0, a).uop))
    x = b.ffn_res_score.mix(bank, b.ffn_norm, s + 1, (c0, live))
    if not hasattr(b, "ffn_gate_inp"): return self.add(s, c0, b._feed_forward(x)).realize()
    logits = (w(x[0]) if isinstance(w:=self.scorer(b), Linear) else x[0] @ w.T).float()
    logits = logits.reshape(live, len(self.devs), 2, -1)[:, :, 0].reshape(live, -1).contiguous()
    logits = allgather_last(logits)  # (a gpu's experts' logits, everyone's on every gpu)
    outs = [(self.win["h"], b._expert_input(x)[0]), (self.win["shared"], b._shared(x)[0]), (self.win["logits"], logits)]
    return Tensor.realize(*[Tensor(t.uop)[c0:c0 + self.chunk].assign(v.pad_to((self.chunk, v.shape[1]))) for t, v in outs])[0]

  def add(self, s:UOp, c0:UOp, y:Tensor) -> Tensor:  # (y (1, live, dim) added to stream s of the bank's chunk at c0, its rows past live 0)
    # (a slice of a symbolic offset, a symbolic length and a symbolic index isn't written: the chunk's whole rows)
    bank, C = Tensor(self.win["bank"].uop), self.chunk
    return bank[:, c0:c0 + C, s].assign((bank[:, c0:c0 + C, s] + y.pad_to((1, C, y.shape[2]))).cast(bank.dtype))

  def rows(self, tile:int) -> int:  # (a window's rows in tiles of tile rows: its selections, each expert's padded)
    return round_up(self.W * self.model.config.num_experts_per_tok + (tile - 1) * self.m["n_experts"], tile)
  def state_rows(self, tile:int) -> tuple[int, int]:  # (the state's rows, its tiles of tile rows: the largest tile size's)
    return (R:=max(map(self.rows, TILES))), R // tile

  def window_route(self, tile:int, live:UOp, bias:Tensor, tag:Tensor, part2:Tensor, *_) -> Tensor:
    """the gpus route the window's live tokens exactly (its tiles of tile rows of an expert), the expert counts to the host (part 2, then the
    tag)"""
    K, E = self.model.config.num_experts_per_tok, self.m["n_experts"]
    w, pos, entries, tiles, counts = moe_route(Tensor(self.win["logits"].uop), bias, K, tile, live=live)
    R, NT = self.state_rows(tile)  # (the routes padded to the state's rows: the experts graph's tiles past them skipped)
    vals = {"w": w, "pos": pos, "entries": entries.pad((0, R - entries.shape[0]), value=-1), "tiles": tiles.pad((0, NT - tiles.shape[0]), value=-1)}
    kept = Tensor.realize(*(Tensor(self.win[k].uop)[:v.shape[0]].assign(v) for k, v in vals.items()))
    counted = Tensor(part2.uop)[1:1 + E].assign(counts.bitcast(dtypes.uint32))
    tag = Tensor(tag.uop.after(counted.uop, kept.uop))
    return tag.assign(tag * 0 + 1).realize()

  def window_experts(self, tile:int, pool:Tensor, part1:Tensor, *_) -> Tensor:
    """the window's routed experts once the host gave their slots (part 1): each gpu its units of the experts in tiles of an expert's rows,
    summed over the gpus"""
    K, E, u, off = self.model.config.num_experts_per_tok, self.m["n_experts"], self.units, 2 * self.units * self.lat // self.block[0] * self.block[1]
    w, pos, entries, h1, h2 = (Tensor(self.win[k].uop) for k in ("w", "pos", "entries", "act", "down"))
    tiles = Tensor(self.win["tiles"].uop)[:self.state_rows(tile)[1]]
    slots = Tensor(part1.uop)[1:1 + E].bitcast(dtypes.int32)
    tslot, xrow = (tiles >= 0).where(slots[tiles.maximum(0)], -1), (entries >= 0).where(entries // K, -1)
    a = self.tiles(tile, pool, tslot, Tensor(self.win["h"].uop), 2 * u, self.lat, 0, h1, xrow=xrow, act=self.act)
    y = self.tiles(tile, pool, tslot, a, self.lat, u, off, h2)
    out = (y.reshape(len(self.devs), -1, self.lat)[:, pos.maximum(0)].float() * w.reshape(1, -1, 1)).reshape(len(self.devs), -1, K, self.lat).sum(2)
    return Tensor(self.win["out"].uop).assign(partial_sum(out)).realize()

  def tiles(self, tile:int, pool:Tensor, tslot:Tensor, x:Tensor, out_f:int, in_f:int, off:int, out:Tensor, xrow:Tensor|None=None,
            act:Callable|None=None) -> Tensor:
    """(gpus, tile * tiles, out_f) into out: each gpu's units of the experts at off bytes in the slots of tiles of tile rows, on their rows of x
    (xrow: its rows, -1 its first), act(gate, up) of the outputs' halves: the AMD tensor cores' expert tiles for IQ3_XXS, else decoded"""
    if amd_custom_kernels_supported(x.device) and self.qtype == IQ3_XXS:
      return expert_tiles(pool, tslot, x, out_f, in_f, self.part, off, xrow=xrow, act=act, out=out, tile=tile)
    NT, size, isz = tslot.shape[0], out_f * in_f // self.block[0] * self.block[1], self.word.itemsize
    w = pool.reshape(len(self.devs), -1, self.part // isz)[:, tslot.maximum(0)][..., off // isz:(off + size) // isz].bitcast(dtypes.uint8)
    w = ggml_data_to_tensor(w.flatten(), prod(w.shape) // self.block[1] * self.block[0], self.qtype).reshape(-1, NT, out_f, in_f)
    y = (x if xrow is None else x[xrow.maximum(0)]).reshape(-1, NT, tile, in_f) @ w.transpose(-1, -2)
    y = (y if act is None else act(*y.split(out_f // 2, -1))).cast(out.dtype)
    return Tensor(out.uop).assign(y.reshape(out.shape)).reshape(len(self.devs), NT * tile, -1)

  def post(self, b:FFNBlock, live:UOp, c0:UOp, s:UOp, *_) -> Tensor:
    """a chunk's live rows' routed and shared experts' outputs into the block's, added to stream s"""
    y = b._expert_output(Tensor(self.win["out"].uop)[c0:c0 + live].unsqueeze(0), Tensor(self.win["shared"].uop)[c0:c0 + live].unsqueeze(0))
    return self.add(s, c0, y).realize()

  def tap(self, s:UOp, j:UOp, *_) -> Tensor:  # (the draft's input: stream s after a block its tap j)
    return Tensor(self.win["taps"].uop)[:, :, j].assign(Tensor(self.win["bank"].uop)[:, :, s].float()).realize()

  def tap_mix(self, s:UOp, j:UOp, w:Tensor, *_) -> Tensor:  # (tap j: the streams up to s mixed by an attention residual's weight w)
    bank = Tensor(self.win["bank"].uop).float()
    scores = (bank.rmsnorm(eps=self.model.config.norm_eps) * w).sum(-1, keepdim=True)
    scores = (Tensor.arange(bank.shape[2]).to(bank.device).reshape(-1, 1) <= Tensor(s)).where(scores, -math.inf)
    return Tensor(self.win["taps"].uop)[:, :, j].assign((scores.softmax(2) * bank).sum(2)).realize()

  def fill(self, pos:UOp, p0:UOp, c0:UOp, *_) -> Tensor:  # (the draft's cache of a chunk of the piece at p0 of the read window)
    C = self.chunk
    given = self.given[p0 + c0:p0 + c0 + C].reshape(1, C).to(self.devs)
    return self.draft.fill(given, Tensor(self.win["taps"].uop)[:, c0:c0 + C].float(), pos).realize()

  def start(self, p0:UOp, *_) -> Tensor:  # (a piece's bank of streams: its tokens' embeddings, the rest zero)
    emb = self.model.token_embd(self.tok[p0:p0 + self.W].reshape(1, self.W).to(self.devs)).unsqueeze(2)
    bank = Tensor(self.win["bank"].uop)
    return bank.assign(emb.pad(((0, 0), (0, 0), (0, bank.shape[2] - 1), (0, 0))).cast(bank.dtype)).realize()

  def stage(self, name:str):  # (a start's stage: its seconds since the stage before)
    print(f"start: {name} {time.perf_counter() - self.t_stage:.1f} s", flush=True)
    self.t_stage = time.perf_counter()

  def slots_of(self, n:int) -> int: return -(-n // self.part)  # (the pool's slots n bytes take)

  def grow(self, n:int):
    """the caches' rows of the first n positions (in steps of a window piece) kept from the experts: their slots' experts dropped (once read,
    if in flight), the slots pinned (step_io's mask, read's keep), the rows zeroed (they were experts': a cache's keys past a context masked)"""
    if n <= self.grown: return
    new, st = min(round_up(n, self.W or 8192), self.max_context), self.state
    for (a, first, row, blocks, size), (o, k) in zip(self.stripes, self.caches):
      starts = first + np.arange(blocks) * size
      sl = np.unique(np.concatenate([np.arange((s + self.grown * row) // self.part, (s + new * row - 1) // self.part + 1) for s in starts]))
      c = np.ascontiguousarray((c:=st["expert_in"][sl])[c >= 0], np.int64)
      if len(c) and (err:=host_c().read_wait(ctypes.byref(self.io), len(c), c.ctypes.data)): raise RuntimeError(f"reads failed {err}")
      st["slot_of"][c], st["expert_in"][sl], st["mask"][sl] = -1, -1, 1
      rows = getattr(o, k).shrink(tuple((self.grown, new) if j == a else None for j in range(getattr(o, k).ndim)))
      rows.assign(rows.zeros_like()).realize()
    self.grown = new

  def pieces(self) -> int:  # (a read window's pieces: the pool past two layers' experts (the one routed, the one read ahead) lent)
    return 1 + max(0, (self.tail - int(self.state["mask"].sum()) - 2 * self.m["n_experts"]) // self.slots_of(self.piece_bytes))

  def window(self, toks:list[int], pos:int, given:list[int]):
    """the tokens at positions pos.. (a read window) through every block, a layer at a time over its pieces of up to W tokens (each its bank):
    a layer's experts read once a read window. the caches and states as after steps of them, the draft's cache"""
    T, C, W, t0 = len(toks), self.chunk, self.W, time.perf_counter()
    self.grow(pos + round_up(T, C))
    assert T <= W * self.pieces() and pos + round_up(T, C) <= self.max_context
    self.tok_np[:T], self.given_np[:T] = toks, given
    ps, st = self.slots_of(self.piece_bytes), self.state
    # (the pieces' states: the first the always lent one, the rest lent below the compute state's slots, their experts dropped)
    banks = [{k: self.win[k] for k in ("bank", "taps")}] + [self.piece(self.tail - i * ps) for i in range(1, -(-T // W))]
    self.lent_from = self.tail - (len(banks) - 1) * ps
    st["slot_of"][(c:=st["expert_in"][self.lent_from:])[c >= 0]], st["expert_in"][self.lent_from:] = -1, -1
    pieces = [(p0, min(W, T - p0), bank) for p0, bank in zip(range(0, T, W), banks)]
    for p0, _, bank in pieces:
      self.win.update(bank)
      self.graph(self.start_jit, self.v_p0.bind(p0))
    host_c().flags(ctypes.byref(self.io), 0)  # (the tags cleared)
    ahead = self.layers(pieces, pos)
    for p0, n, bank in pieces if self.fill_jit else []:
      self.win.update(bank)
      for c0 in range(0, n, C): self.graph(self.fill_jit, self.v_pos.bind(pos + p0 + c0), self.v_p0.bind(p0), self.v_c0.bind(c0))
    for d in self.devs: Device[d].synchronize()
    for li in ahead: self.counted(li)
    self.win.update(banks[0])
    self.lent_from = self.tail
    self.window_s.append((T, time.perf_counter() - t0))

  def steps_faster(self, n:int) -> bool:
    """n tokens a step each faster than a window of them: a step's measured seconds, a window's a fixed part and its tokens' (a token's the
    measured windows' median past the fixed part: a cold one doesn't move it). before any are measured, tinys3's"""
    step = float(np.median(self.step_s[1])) if self.step_s[1] else STEP_S
    token = float(np.median([max(t - WINDOW_S, 0) / m for m, t in self.window_s])) if self.window_s else WINDOW_TOKEN_S
    return n * step < WINDOW_S + n * token

  def layers(self, pieces:list[tuple[int, int, dict[str, Tensor]]], pos:int) -> list[int]:
    """a read window's blocks, a layer at a time over its pieces. a moe layer's experts a piece's tokens route to read in (the host waits for
    the gpus' counts), their slots given to the gpus, kept till the layer's done. when a piece's tokens route to nearly all of a layer's
    experts, every next layer's are read ahead while the gpus work (and not waited for: those layers' counted after the window). returns
    those"""
    n, E, K, ahead = 1, self.m["n_experts"], self.model.config.num_experts_per_tok, []
    def chunks(T:int) -> list[tuple[int, int]]: return [(c0, min(self.chunk, T - c0)) for c0 in range(0, T, self.chunk)]
    for i, b in enumerate(self.model.blk):
      s = n - 1 + int(b.attn_res_score.new_block)
      for p0, T, bank in pieces:
        self.win.update(bank)
        jit, args, _ = self.kind("attend", b, self.attend)
        for c0, live in chunks(T):
          self.graph(jit, self.v_pos.bind(pos + p0 + c0), self.v_live.bind(live), self.v_c0.bind(c0), self.v_n.bind(n), self.v_s.bind(s), *args)
          self.pump()
        if hasattr(b, "ffn_gate_inp"):
          self.layer = li = self.m["layers"].index(i)
          route, experts = self.routes[next(t for t in TILES if T * K <= t * E or t == TILES[-1])]
          for t in self.bufs[li][0][1]: t[0] = 0  # (the tag: this piece's counts)
          self.graph(route, self.v_wlive.bind(T), b.exp_probs_b["bias"], Tensor(self.tags[li]), self.parts[1][li])
          codes = li * E + (np.arange(E) if li in ahead else np.nonzero(self.counted(li))[0])
          t0 = time.perf_counter()
          self.read(codes, codes, wait=True)
          self.io.t_reads += time.perf_counter() - t0
          for p in self.bufs[li][2][1]: p[1:1 + E] = self.state["slot_of"][li * E:(li + 1) * E].view(np.uint32)
          self.graph(experts, self.pool, self.parts[0][li])
          if li + 1 < len(self.parts[0]) and li + 1 not in ahead and (li in ahead or len(codes) >= 0.9 * E):
            self.read((li + 1) * E + np.arange(E), codes, wait=False)
            ahead.append(li + 1)
          jit, args, _ = self.kind("post", b, self.post)
          for c0, live in chunks(T):
            self.graph(jit, self.v_live.bind(live), self.v_c0.bind(c0), self.v_s.bind(s), *args)
            self.pump()
        for j, (blk, mixed) in enumerate(self.taps):
          if blk != i: continue
          if mixed: self.graph(self.mix_jit, self.v_s.bind(s), self.v_tap.bind(j), self.mixer(i).weight)
          else: self.graph(self.tap_jit, self.v_s.bind(s), self.v_tap.bind(j))
      n = s + 1
    return ahead

  def graph(self, jit:Callable, *args) -> Tensor:  # (a window graph on the window's state)
    return jit(*args, *self.win.values())

  def pump(self):  # (the reads ahead queued as the drives have room)
    if host_c().pump(ctypes.byref(self.io)): raise RuntimeError("window reads failed")

  def counted(self, li:int) -> np.ndarray:
    """layer li's expert counts of a window's tokens once the gpus give them, added to its decayed use counts"""
    E, io, t0 = self.m["n_experts"], self.io, time.perf_counter()
    tags = self.bufs[li][0][1]
    while not all(t[0] == 1 for t in tags):
      if any(t[0] == 2 for t in tags) or time.perf_counter() - t0 > 60: raise RuntimeError(f"window layer {li}: no counts from the gpus")
    counts = self.bufs[li][3][1][0][1:1 + E].copy()
    f = self.state["freq"][li * E:(li + 1) * E]
    f[:] = f * io.decay + counts
    io.t_gpu += time.perf_counter() - t0
    return counts

  def read(self, codes:np.ndarray, busy:np.ndarray, wait:bool):
    """the codes not cached read into the least used slots but those of busy's codes (in use or read): waited for (and those read ahead), or
    not (read ahead)"""
    st, io = self.state, self.io
    reads = [np.ascontiguousarray(a, np.int64) for a in (codes[st["flying"][codes] > 0], codes[st["slot_of"][codes] < 0])]
    if wait and len(reads[0]) and (err:=host_c().read_wait(ctypes.byref(io), len(reads[0]), reads[0].ctypes.data)):
      raise RuntimeError(f"window reads failed {err}")
    if not len(miss:=reads[1]): return
    old, E = st["expert_in"], self.m["n_experts"]
    keep = (np.arange(self.slots) >= self.lent_from) | ((old >= self.layer * E) & (old < (self.layer + 2) * E)) | (st["mask"] > 0)
    keep[st["slot_of"][busy][st["slot_of"][busy] >= 0]] = True
    keep |= (old >= 0) & (st["flying"][np.maximum(old, 0)] > 0)
    victims = np.ascontiguousarray(np.argsort(np.where(keep, np.inf, np.where(old >= 0, st["freq"][np.maximum(old, 0)], -1.0)),
                                              kind="stable")[:len(miss)], np.int64)
    if keep[victims].any(): raise RuntimeError(f"window layer {self.layer}: {len(miss)} reads, the pool's free slots fewer")
    st["slot_of"][old[victims][old[victims] >= 0]] = -1
    st["slot_of"][miss], st["expert_in"][victims] = victims, miss
    io.misses += len(miss)
    if not wait: return host_c().read_ahead(ctypes.byref(io), len(miss), miss.ctypes.data)
    if (err:=host_c().read_codes(ctypes.byref(io), len(miss), miss.ctypes.data, victims.ctypes.data)):
      raise RuntimeError(f"window reads failed {err}")

  # *** the host's side

  def warm(self):
    """every step's graph (and the rollback's, the saves') captured on dry steps: the host's flags preset (its tables still zero: no experts)"""
    host_c().flags(ctypes.byref(self.io), 1)
    self.t_stage = time.perf_counter()
    for T, jit in self.jits.items():
      for _ in range(3): jit(self.v_pos.bind(0))[0].tolist()
      self.stage(f"step graph of {T}")
    for n, _ in itertools.product(range(1, self.tokens), range(3)): self.model.rollback(n)  # (their jits captured too, after a verify step)
    for (i, save), _ in itertools.product(itertools.product(range(2), (True, False)), range(3)): self.model._copy(i, save)  # (and save's)
    for _ in range(3 if self.chain_jit else 0): self.chain_jit(self.v_pos.bind(0), self.v_row.bind(0)).tolist()
    for jit, _ in itertools.product(self.keep_jits, range(3)): jit(self.v_at.bind(0))  # (a kept session's copies)
    self.stage("rollbacks, saves, chain, session copies")

  def step(self, toks:list[int], pos:int, given:list[int]|None=None) -> tuple[list[int], list[int], np.ndarray|None]:
    """a step of the tokens at positions pos..: the gpus run its graph, the host every layer's routing and reads. the greedy token after each,
    the draft's guesses after the one after the kept tokens and their scores (a scored draft's: each right if the ones before are) (their logits
    kept)"""
    assert DEBUG < 2, "DEBUG>=2 syncs in the step's launch, before the host's part runs: a hang"
    io, T, t0 = self.io, len(toks), time.perf_counter()
    assert T in self.jits and pos + T <= self.max_context
    self.grow(pos + T)
    io.tokens, io.victims_n, self.res_np[0] = T, 32 * T, 0
    self.tok_np[:T], self.given_np[:T] = toks, ((given or []) + [-1] * T)[:T]
    host_c().flags(ctypes.byref(io), 0)
    _, self.logits = self.jits[T](self.v_pos.bind(pos))
    t1 = time.perf_counter()
    err = host_c().step_io(ctypes.byref(io))
    if err: host_c().flags(ctypes.byref(io), 1)  # (the gpus run the graph out: one killed waiting on a layer wedges)
    t2 = time.perf_counter()
    while self.res_np[0] != 1:
      if time.perf_counter() - t2 > 60: raise RuntimeError("the step's graph didn't finish")
    if err: raise RuntimeError({-1: "a read failed", -2: "timeout", -3: "no victims left", -4: "NaN scores"}[err])
    ret, G = self.res_np[1:1 + T + 2 * self.width], self.width
    self.times += (t1 - t0, time.perf_counter() - t2)
    self.step_s[T].append(time.perf_counter() - t0)
    scores = ret[T + G:].view(np.float32).copy() if self.draft is not None and self.draft.scored else None
    return ret[:T].tolist(), ret[T:T + G].tolist(), scores

  def proposal(self, g:list[int], score:np.ndarray|None, pos:int, row:int) -> list[int]:
    """a verify step's guesses after the token at pos: the row's of the step before (guessed at its row), the draft's chain after them. scored:
    as many as the step of the widths that expects most tokens a second (by the scores, the steps' measured seconds: one not measured first)"""
    n = self.tokens - 1
    if score is not None and len(self.widths) > 1:
      def rate(T:int) -> float: return (1 + np.cumprod(score[:T - 1]).sum()) / np.mean(s) if (s:=self.step_s[T]) else math.inf
      n = max(self.widths, key=rate) - 1
    return g[:n] if len(g) >= n else [*g, *self.chain(g[-1], pos, row)]

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

  def session_rows(self, i:int, at:UOp) -> Tensor:  # (cache i's rows of a kept session's at..)
    o, k = self.caches[i]
    return getattr(o, k).shrink(tuple((at, at + self.B) if j == self.stripes[i][0] else None for j in range(getattr(o, k).ndim)))

  def keep(self, at:UOp):  # (a session's rows at.. and recurrent states into the copies)
    states = zip(self.stash_states, self.model._states())
    Tensor.realize(*[c.assign(self.session_rows(i, at)) for i, c in enumerate(self.stash)], *[c.assign(s) for c, s in states])

  def unkeep(self, at:UOp):  # (and back)
    states = zip(self.stash_states, self.model._states())
    Tensor.realize(*[self.session_rows(i, at).assign(c) for i, c in enumerate(self.stash)], *[s.assign(c) for c, s in states])

  def resume(self, tokens:list[int]):
    """the kept session back when tokens share more with it than with the tokens fed (if the request since wrote only its copied rows), else
    the tokens fed kept when tokens diverge from them early and are fewer (a short request: a title, a summary) and longer than one kept"""
    m = self.model
    def shared(a:list[int]) -> int: return sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens, a)))
    if self.session is not None and shared(kept:=self.session[0]) > shared(m._cached_tokens):
      if round_up(len(m._cached_tokens) + self.tokens, max(self.chunk, 1)) <= (at:=self.session[1]) + self.B:
        self.keep_jits[1](self.v_at.bind(at))
        m._cached_tokens, m._saved = kept[:], []
      self.session = None
    elif len(tokens) < len(m._cached_tokens) and (at:=shared(m._cached_tokens)) + self.B <= self.max_context and len(tokens) - at < self.B // 2 \
         and len(m._cached_tokens) > (len(self.session[0]) if self.session else 0):
      self.keep_jits[0](self.v_at.bind(at))
      self.session = (m._cached_tokens[:], at)

  def prefill(self, tokens:list[int], pos:int) -> int:
    """tokens from pos on but the last fed in windows of up to W of them (an even split, a layer at a time over each), the last one's chunk
    partial (its rows padded), unless steps of them are faster (steps_faster), the position after them"""
    if not self.W: return pos
    end = len(tokens) - 1
    P = self.W * self.pieces()  # (a read window's most tokens)
    size = -(-(end - pos) // -(-(end - pos) // P)) if end > pos else P
    while pos < end and pos + round_up(n:=min(size, end - pos), self.chunk) <= self.max_context:
      if self.steps_faster(n): break
      self.window(tokens[pos:pos + n], pos, tokens[pos + 1:pos + 1 + n])
      pos += n
    return pos

  def generate(self, tokens:list[int], temperature:float=0.0):
    """the greedy tokens after tokens (any temperature), appended as they come: fed from get_start_pos in chunks (prefill) and the rest a token
    a step, then a token a step or (a draft) a verify step of [the next, the draft's chain of guesses after it]: the guesses kept up to the first
    wrong one, the target's token after them next, the recurrent states rolled back to after the kept ones"""
    self.resume(tokens)
    start = self.get_start_pos(tokens)
    # (where tokens diverge from the tokens fed before, a window's any length: its states saved)
    if (fork:=self.model.fork(tokens, start, 1 if self.W else self.chunk or 1)) > start:
      for pos in range(self.prefill(tokens[:fork + 1], start), fork): self.step(tokens[pos:pos + 1], pos, tokens[pos + 1:pos + 2])
      self.model.save(tokens[:fork])
    for pos in range(self.prefill(tokens, fork), len(tokens)):
      (t,), g, score = self.step(tokens[pos:pos + 1], pos, tokens[pos + 1:pos + 2])
    self.model.save(tokens)
    row = 0
    while len(tokens) + self.tokens <= self.max_context:  # (a verify step's last token in the context)
      tokens.append(t)
      self.model._cached_tokens = tokens[:-1]
      yield t
      if self.draft is None:
        (t,), _, _ = self.step([t], len(tokens) - 1)
        continue
      gs = self.proposal(g, score, len(tokens) - 1, row)
      toks, g, score = self.step([t, *gs], len(tokens) - 1)
      row = next((j for j, g in enumerate(gs) if toks[j] != g), len(gs))
      self.verified += (1, len(gs), row)
      if row < len(gs): self.model.rollback(row + 1)  # (before the kept ones are yielded: a request may stop at any)
      for kept in gs[:row]:
        tokens.append(kept)
        self.model._cached_tokens = tokens[:]
        yield kept
      t = toks[row]

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
  # DRAFT=dir: the draft (eagle's head, dspark's blocks), speculative decoding with verify steps of VERIFY tokens (memory: fewer SLOTS)
  # VERIFY=n: a verify step's tokens, the next and the draft's n-1 guesses (eagle's: a chain) (an EVAL: its steps of n tokens)
  # (VERIFY=n,n..: a scored draft's verify steps of any of them, each step's by the guesses' scores)
  widths = [int(x) for x in str(getenv("VERIFY", "2")).split(",")]
  draft = load(d, lambda t: model.token_embd(t).float(), model.output, model.token_embd.weight.device, model.max_context,
               max(widths) - 1) if (d:=getenv("DRAFT", "")) else None
  print(f"start: store, draft {time.perf_counter() - st:.0f} s in", flush=True)
  # CHUNK=n: a prompt fed in steps of n tokens (0: a token a step), WINDOW=n: in windows of up to n tokens (a layer at a time, chunks of CHUNK)
  # TAPS=b,b..: without a draft, the streams a window keeps after blocks b (bm: as the next block's attention residual mixes them), for a DUMP
  taps = [(int(t.rstrip("m")), t.endswith("m")) for t in getenv("TAPS", "").split(",") if t]
  eng = Engine(model, m, tokens=max(widths) if draft else 1, guess=getenv("GUESS", ""), draft=draft, logits=bool(getenv("EVAL", "")),
               chunk=getenv("CHUNK", 256), window=getenv("WINDOW", 0), taps=taps, widths=tuple(widths))
  eng.drives(disks)
  eng.warm()
  print(f"warm in {time.perf_counter() - st:.0f} s, {eng.slots} slots a gpu", flush=True)
  if (spec:=getenv("EVAL", "") or getenv("WEVAL", "") or getenv("BENCH", "")):  # EVAL=/WEVAL=/BENCH=corpus.jsonl:nseq:ntok:first (eval.py)
    from extra.k3.eval import evaluate, window_evaluate, window_dump, bench
    if (dump:=getenv("DUMP", "")): window_dump(eng, spec, late, dump)  # DUMP=dir with WEVAL=: the taps and the target's tokens (eval.py)
    elif getenv("BENCH", ""): bench(eng, spec, late, getenv("DEPTH", ""))  # (DEPTH=corpus.jsonl:n: n tokens of text before each prompt)
    else: (evaluate if getenv("EVAL", "") else window_evaluate)(eng, spec, late)
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
