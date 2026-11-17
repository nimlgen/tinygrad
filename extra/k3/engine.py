#!/usr/bin/env python3
# K3 decode with the routed experts streamed from the nvme drives: master's model (a trunk gguf without the routed experts, sharded over the
# gpus) in a jitted step graph of the engine's own forward. at a moe block every gpu writes its router scores and a tag to host memory, the
# host routes, reads the missing experts from the drives into each gpu's pool of slots and gives the gpus the slots and weights in two parts
# (host.py's step_io, in c alongside the graph), the gpus wait for each (host_wait) and run their units of the experts over the pool
# DEV=AMD:HIP PYTHONPATH=. python3 extra/k3/engine.py trunk.gguf "prompt"
import collections, ctypes, functools, itertools, math, os, signal, sys, time
import numpy as np
from typing import Callable, Iterator, cast
from tinygrad import Tensor, TinyJit, UOp, Device, dtypes, nn
from tinygrad.dtype import DType
from tinygrad.helpers import DEBUG, getenv, prod, round_up
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.engine.jit import _TinyJit
from tinygrad.uop.ops import Ops, AxisType
from tinygrad.llm.model import Transformer, FFNBlock, GatedDeltaNetBlock, AttentionResidual, gated_activation
from tinygrad.llm.gguf import ggml_data_to_tensor, _GGML_QUANT
from tinygrad.llm.kernels.amd import Linear, amd_custom_kernels_supported, partial_sum, allgather_last, reduce_scatter, PARTIALS
from tinygrad.llm.kernels.amd_gguf import expert_linear, expert_tiles, moe_route, q8_0_linear, QUANT_NAMES, HALFWORD_QUANTS, TILE_QUANTS
from extra.k3.host import IoT, host_c, wait, reads, free_bytes, host_tensor, VICTIMS
from extra.k3.draft import Eagle3, DSpark, MTP, load
from extra.k3 import persist

# TAU: a token's experts under this weight left out unless cached, TAUC: those under it left out cached or not, BETA: a cached expert's
# routing bonus, FILL: the experts tau left out read anyway (a layer), PREFETCH: the next layer's experts guessed a layer. SLOTS: on a gpu
# (default: what's free but SPARE bytes). BIG: a token's experts of a rank under it read in the store's first copy (of copies: host.py)
TAU, TAUC, BETA, FILL, PREFETCH = getenv("TAU", 0.065), getenv("TAUC", 0.0), getenv("BETA", 0.07), getenv("FILL", 2), getenv("PREFETCH", 5)
BIG = getenv("BIG", 2)
SLOTS, SPARE = getenv("SLOTS", 0), getenv("SPARE", 2 << 30)
# (a window's seconds: a fixed part and a token's, a step's (of a token) before any are measured: tinys3's, 4.8 s and 1.26 ms, 50 ms)
WINDOW_S, WINDOW_TOKEN_S, STEP_S = 5.0, 1.3e-3, 0.05
SESSION = 4096  # (a kept session's rows a short request that diverges from it early may write)
TILES = (16, 64)  # (a window's tiles of an expert's rows: the first that holds its rows an expert on average)

class Engine:
  bank:DType = dtypes.half  # (a window's residual streams: in half, the attends read half the bytes)
  bank_streams:int|None = None  # (the bank's streams: the model's blocks' if None)
  def __init__(self, model:Transformer, m:dict, slots:int=SLOTS, tokens:int=1, guess:str="", draft:Eagle3|DSpark|MTP|None=None, logits:bool=False,
               chunk:int=0, window:int=0, taps:list[tuple[int, bool]]|None=None, widths:tuple[int, ...]=(), source:str="",
               copies:tuple[dict, ...]=()):
    """the moe blocks of model with their routed experts streamed from the store of manifest m into a pool of slots on each gpu, a part of
    every expert each (its units), in steps of a token or (a draft's verify steps) of tokens, a prompt in windows of up to window tokens (their
    attention in chunks), keeping their logits or not. taps: without a draft, the (block, mixed) streams a window keeps (draft.taps). widths:
    a scored draft's verify steps of fewer tokens too (their graphs captured), each step's the one its guesses' scores expect most tokens a
    second of. source: the model's gguf (a start's graphs are kept for the next start of the same code, model and config: persist.py). copies:
    the manifests of the store's other copies of the experts (of some of its layers: a step's expert read in one of them, BIG)"""
    self.model, self.m, self.devs, self.draft, self.keep_logits = model, m, model.token_embd.weight.device, draft, logits
    self.source, self.saved, self.names = source, cast(dict, {}), {}
    self.t_stage = time.perf_counter()
    self.dump_taps = taps or []
    stack = [*model.blk, *(draft.blk if draft is not None else [])]
    blocks = [stack[b] for b in m["layers"] if b < len(stack)]
    self.ms = [m, *copies]
    L, E, K, G, self.lat, self.units = len(blocks), m["n_experts"], model.config.num_experts_per_tok, len(self.devs), m["lat"], m["units"][0]
    assert all(len(set(c["part_bytes"])) == 1 and c["units"] == [self.units] * G and c["lat"] == self.lat for c in self.ms), \
      "a part of every expert on each gpu"
    # (a pool's slot: the smallest part, a copy's part its span of them)
    self.part = min(c["part_bytes"][0] for c in self.ms)
    self.spans = [-(-c["part_bytes"][0] // self.part) for c in self.ms]
    assert E <= 4096 and K <= 64 and tokens <= 8 and tokens * K <= 512 and 32 * tokens <= 1024, "past the host's fixed arrays"
    self.tokens, self.chunk, n = tokens, chunk, max(tokens, chunk, window)  # (n: a step's or a window's most tokens)
    qtype, layers = {n.upper(): t for t, n in QUANT_NAMES.items()}, [b for b in m["layers"] if b < len(stack)]
    self.copies = [[(qtype[t[0]], qtype[t[2]]) if t else None for t in (c["types"].get(str(b)) if "types" in c else [c["expert_type"]] * 3
                                                                      if b in c["layers"] else None for b in layers)] for c in self.ms]
    self.types = self.copies[0]
    # each layer's host buffers on every gpu: its tag, router scores (a token's rows: the router's, the next layer's guess; a gpu its experts')
    # and the two parts of a step's slots and weights, [ready | slots | weights]: entry j*T+t is token t's j-th expert and its weight (a
    # window's: [ready | an expert's slot], [ready | an expert's count])
    self.ks, self.act = (K // 2, K - K // 2), functools.partial(gated_activation, config=model.config)
    part = [1 + max(2 * k * tokens * len(self.ms), E) for k in self.ks]  # (a part's words)
    bufs = [[host_tensor(self.devs, sz, dt) for sz, dt in ((1, dtypes.uint32), (n * 2 * E // G, dtypes.float32), *((p, dtypes.uint32) for p in part))]
            for _ in blocks]
    self.bufs, self.tags, self.parts = bufs, [b[0][0].uop.reshape(1, 1).unshard(0) for b in bufs], [[b[2 + i][0] for b in bufs] for i in range(2)]
    self.scores = [b[1][0].uop.reshape(n, 2, E // G).unshard(2) for b in bufs]
    # the scorers: a layer's router and the guess of the next layer's experts on its input (GUESS=dir: guess-<block>.npy maps fitted on traces,
    # else the next layer's router), a gpu its experts' rows of each (in q8_0 where there's a kernel). the routers' memory freed: the scorers route
    # (the routers' float copies not held: the scorers' graphs only, freed once they're realized)
    def router(li:int) -> Tensor: return blocks[li].ffn_gate_inp.weight.to(self.devs[0]).float()
    def guessed(li:int) -> Tensor:
      return Tensor(np.load(f)).to(self.devs[0]) if os.path.exists(f:=f"{guess}/guess-{m['layers'][li]}.npy") else router(li)
    def scorer(w:Tensor) -> Tensor|Linear:  # (rows a gpu's: its experts' of the router, then of the guess)
      w = w.reshape(2, G, E // G, -1).permute(1, 0, 2, 3).reshape(2 * E, -1).half()
      return q8_0_linear(w, self.devs, 0) if amd_custom_kernels_supported(self.devs) else w.shard(self.devs, 0)
    # (the scorers kept for the next start: of the gguf and the guess maps)
    loaded = persist.load_scorers(self, guess)
    self.scorers = loaded or [scorer(Tensor.stack(router(li), guessed(min(li + 1, L - 1)))) for li in range(L)]
    Tensor.realize(*(biases:=[b.exp_probs_b["bias"] for b in blocks]), *[s.weight if isinstance(s, Linear) else s for s in self.scorers])
    if loaded is None: persist.save_scorers(self, guess, self.scorers)
    self.bias = list(Tensor.stack(*[t.float() for t in biases]).numpy())
    for b in model.blk:  # (a verify step's recurrent states after its first token kept: a rollback)
      if isinstance(b, GatedDeltaNetBlock): b.keep = tokens - 1
    for li, b in enumerate(blocks):
      b._routed = functools.partial(self.routed, li)  # type: ignore[method-assign]
      b.ffn_gate_inp.weight = Tensor.empty(0)
    weights = {id(t) for t in nn.state.get_parameters(stack)}  # (not realized here: some are decoded in the step's graph)
    # (a draft holding the trunk's weights would realize them as its state: decoded, their quantized kernels lost)
    assert not {id(t) for t in nn.state.get_parameters(draft)} - weights & {id(t) for t in nn.state.get_parameters(model)}, \
      "a draft with the trunk's weights"
    for b in stack: b._init_state(Tensor.empty(1, 1, model.config.dim, device=self.devs))
    # the context's caches (the mla blocks' latents, their indexers' keys, the draft's): lent from the pool (never allocated here)
    owners = [(o, k) for b in stack for o, k in ((b, "cache_k"), (getattr(b, "indexer", None), "cache_kg")) if hasattr(o, k)]
    self.caches = owners + ([(draft, "cache")] if hasattr(draft, "cache") else [])
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
    self.jits = {T: TinyJit(functools.partial(self.run, T), capture_first=True) for T in sorted({1, tokens, *widths})}
    self.keep_jits = (TinyJit(self.keep, capture_first=True), TinyJit(self.unkeep, capture_first=True))
    self.widths = sorted({tokens, *widths}) if draft is not None and draft.scored else [tokens]
    # (measured: a step's seconds, the chain's that guessed for it too)
    self.step_s:dict[int, collections.deque] = {T: collections.deque(maxlen=64) for T in self.jits}
    self.chain_s = 0.0
    # a draft's chain of guesses after its first (a verify step's other tokens): its steps on its own hidden states (dx: a step's), from a row
    # ([done | a chain step's guesses | their probabilities (f32)])
    self.links = 1 if draft is not None and draft.blk else max(tokens - 2, 0)
    self.cres, (self.cres_np,) = host_tensor(self.devs[0], 1 + 2 * self.links, dtypes.int32)
    self.dx, self.v_row = Tensor.empty(n, model.config.dim, device=self.devs).realize(), UOp.variable("row", 0, tokens - 1)
    self.chain_jit = TinyJit(self.run_chain, capture_first=True) if draft is not None and draft.width < tokens - 1 else None
    self.times, self.verified = np.zeros(2), np.zeros(3)  # (launching the graphs, after the last layer), (verify steps, guesses, right ones)
    self.window_s:collections.deque = collections.deque(maxlen=64)  # (measured: a window's tokens, seconds)
    # each gpu's pool in the experts' words (a bitcast of it is a copy), of what's free after the blocks' caches and states and the window's
    # graphs (captured on a pool of a slot), then the slots a window's state is lent (its own while it runs): slots of decode's otherwise
    words = {t: dtypes.uint16 if t in HALFWORD_QUANTS else dtypes.uint32 for t in itertools.chain(*(t for ts in self.copies for t in ts if t))}
    self.word = min(words.values(), key=lambda dt: dt.itemsize)
    def pool(n:int) -> Tensor: return Tensor.zeros(G * n * self.part // self.word.itemsize, dtype=self.word, device=self.devs, axis=0).realize()
    self.W = 0
    if window: self.window_init(window)
    # (the caches lent from the pool's first slots (a gpu its shard of each), their rows of positions in stripes: the rows a context reached
    # kept from the experts (grow), the rest the experts' (never read))
    shapes = {str(i): (getattr(o, k).uop.shard_shape, getattr(o, k).dtype) for i, (o, k) in enumerate(self.caches)}
    axes = [getattr(o, k).uop.axis for o, k in self.caches]
    # a kept session (a long one a short request diverges from early): copies of the caches' rows that request may write, its recurrent states
    self.B, self.session = min(SESSION, model.max_context // 2), cast(tuple[list[int], int, list[tuple[list[int], int]]]|None, None)
    self.v_at = UOp.variable("at", 0, model.max_context - self.B)
    def rows0(t:Tensor) -> tuple[int, ...]: return tuple(self.B if j == next(j for j, d in enumerate(t.shape) if d >= model.max_context) else d
                                                         for j, d in enumerate(t.shape))
    self.stash = [Tensor.empty(*rows0(t:=getattr(o, k)), dtype=t.dtype, device=self.devs, axis=t.uop.axis) for o, k in self.caches]
    self.stash_states = [s.empty_like() for s in self.session_states()]
    Tensor.realize(*self.stash, *self.stash_states)
    for o, k in self.caches: setattr(o, k, None)
    free = (lambda: min(free_bytes(d) for d in self.devs)) if not slots else (lambda: 0)  # (what's free on the gpus: sizing the pool by it)
    self.stage(f"states ({free() / 1e9:.2f} GB free before the pool)")
    # (lent: the window's compute state and a piece's, the slots of more pieces lent at a prefill)
    self.win, lent, self.layer = {}, self.slots_of(self.window_bytes) + self.slots_of(self.piece_bytes) if window else 0, 0
    self.slots = slots = ((slots or (free() - SPARE) // self.part - lent) + lent) // (w:=math.lcm(*self.spans)) * w
    self.tail = self.lent_from = slots - lent
    self.pool = pool(slots)
    pools = {dt: self.pool if dt == self.word else Tensor(self.lend({"pool": ((slots * self.part // dt.itemsize,), dt)}, 0)[0]["pool"].uop.unshard(0))
             for dt in set(words.values())}
    self.pools = {t: pools[dt] for t, dt in words.items()}
    caches, self.cache_bytes = self.lend(shapes, 0)
    # (a cache's: its positions' axis, its first byte, a position's bytes, its blocks of every position's rows and their bytes)
    self.stripes = [(a, caches[i].uop.base.buffer.bufs[0].offset, row, prod(shape[:a]), shape[a] * row) for i, (shape, dt) in shapes.items()
                    for a in [next(j for j, d in enumerate(shape) if d >= model.max_context)] for row in [prod(shape[a + 1:]) * dt.itemsize]]
    for i, (o, k) in enumerate(self.caches): setattr(o, k, caches[str(i)] if axes[i] is None else Tensor(caches[str(i)].uop.unshard(axes[i])))
    # (the caches grow by a step of rows, zeroed in a graph: a step's rows at a position)
    self.grown, self.grow_step = 0, min(window or 8192, model.max_context)
    self.v_grown, self.zero_jit = UOp.variable("grown", 0, model.max_context - self.grow_step), TinyJit(self.zero, capture_first=True)
    # the host's state, kept by step_io: each code's (layer * E + expert) slot (-1: not cached) and each slot's code (-1: free), the decayed
    # use counts, the guesses' reads in flight, the guesses their layer hasn't routed, pinned slots, the victims of the next reads
    codes = len(self.ms) * L * E
    self.state = {"slot_of": np.full(codes, -1, np.int32), "expert_in": np.full(slots, -1, np.int64), "freq": np.zeros(codes, np.float32),
                  "flying": np.zeros(codes, np.int32), "guess": np.zeros(codes, np.uint8), "mask": np.zeros(slots, np.uint8),
                  "victims": np.zeros(len(self.ms) * VICTIMS, np.int64), "nvict": np.zeros(len(self.ms), np.int64),
                  "span": np.array(self.spans, np.int64),
                  "bias": np.array([b.ctypes.data for b in self.bias], np.uint64),
                  "bufs": np.array([[[v.ctypes.data for v in views] for _, views in b] for b in bufs], np.uint64)}
    self.io = IoT(layers=L, E=E, ngpu=G, k=K, kmin=K // 4, k1=self.ks[0], rows=2, slots=slots, tau=TAU, tauc=TAUC, beta=BETA, pf=PREFETCH,
                  fill=FILL, decay=0.5 ** (1 / 8), copies=len(self.ms), big=BIG)
    for name, a in self.state.items(): setattr(self.io, name, a.ctypes.data)
    if window:
      self.stage(f"pool: {slots} slots ({slots * self.part / 1e9:.2f} GB, of it lent: the caches' {self.cache_bytes / 1e9:.2f} GB as they grow, "
                 f"the window's {(slots - self.tail) * self.part / 1e9:.2f} GB)")
      self.window_lend()
      self.stage(f"window graphs (their scratch {(self.arena[1] - self.arena[0]) * self.part / 1e9:.2f} GB, the pool's)")
    else: self.restore()

  def drives(self, *disks:list[list[str]]):
    """step_io's reads from the store's drives (each copy's: each gpu's disk) into the pools"""
    def sizes(c:dict, types:list) -> list[tuple[int, int]|None]:
      return [None if t is None else (c["layers"].index(b), round_up(self.nbytes(t[0], 2 * self.units * self.lat) + self.nbytes(t[1], self.units *
              self.lat), 4096)) for b, t in zip(self.m["layers"], types)]
    self.tables, scalars = reads([(c, d, sizes(c, ts)) for c, d, ts in zip(self.ms, disks, self.copies)], self.pool.uop.buffer.bufs, self.part)
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
    m, hidden, taps, streams = self.model, {}, self.draft.taps if self.draft is not None else [], self.model.config.attn_res_block_size
    x = m.token_embd(tokens.to(m.token_embd.weight.device)).float()
    if streams: x = x.unsqueeze(2)  # (the attention residuals' streams: (1, T, 1, dim))
    mask = None
    for i, b in enumerate(m.blk):
      x, mask = b(x, (pos,), mask)
      for t in taps:
        if t[0] == i: hidden[t] = (self.mixer(i).mix(x) if t[1] else x[:, :, -1]) if streams else x
    x = m.output_res_score.mix(x) if streams else x
    return m.output(m.output_norm(x))[0].float(), Tensor.stack(*(hidden[t] for t in taps), dim=2) if taps else None

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
    row of the verify step's) into host memory and their probabilities, then the done word"""
    tok, hid, cache, out, probs = self.given[:1].to(self.devs), Tensor(self.dx.uop)[row:row + 1], None, [], []
    for k in range(self.links):
      logits, hid, cache = self.draft(tok.reshape(1, 1), None, pos + k, hid=hid.reshape(1, 1, -1), cache=cache)
      out.append(tok:=logits[0].argmax(-1))
      probs.append(logits[0].softmax(-1).max(-1).float().bitcast(dtypes.int32))
    outs = [Tensor(self.cres.uop)[1:].assign(Tensor.cat(*[t.to(self.devs[0]).cast(dtypes.int32) for t in out + probs])).uop]
    if self.draft.blk: outs.append(Tensor(self.dx.uop)[row:row + 1].assign(hid.reshape(1, -1).float()).uop)
    return Tensor(self.cres.uop.after(*outs))[:1].assign(Tensor.ones(1, dtype=dtypes.int32, device=self.devs[0])).realize()

  def routed(self, li:int, x:Tensor, h:Tensor) -> Tensor:
    """layer li's routed experts on the tokens x (1, T, dim), h their expert inputs (1, T, lat): the gpus' scores to the host, each gpu's units
    of the experts it gives in two parts, summed over the gpus"""
    T, s = x.shape[1], w(x[0]) if isinstance(w:=self.scorers[li], Linear) else x[0] @ w.T
    scores = Tensor(self.scores[li])[:T].assign(s.float().reshape(-1, len(self.devs), 2, self.m["n_experts"] // len(self.devs))
                                                .permute(0, 2, 1, 3).reshape(-1, 2, self.m["n_experts"]))
    y1, w1 = self.experts(li, h[0], self.host(li, 0, scores), self.ks[0] * T)
    y1 = y1.contiguous()  # (a buffer: part 2 waited for after it)
    y2, w2 = self.experts(li, h[0], self.host(li, 1, y1), self.ks[1] * T)
    return (partial_sum(self.heaviest_first([y1, y2], [w1, w2], T)) * self.model.config.routed_scaling_factor).unsqueeze(0)

  def host(self, li:int, i:int, dep:Tensor) -> Tensor:
    """part i of layer li's slots and weights once the host releases it, waited for after dep (part 1: the scores, the host told they're in:
    a gpu's tag set 1 after its scores, 2 on a NaN)"""
    if i == 0: dep = Tensor(self.tags[li]).assign(((s:=dep[0, 0].reshape(len(self.devs), -1)[:, :1]) == s).where(1, 2).cast(dtypes.uint32))
    return Tensor((t:=self.parts[i][li].uop).after(wait(t, dep.uop)))

  def experts(self, li:int, h:Tensor, table:Tensor, n:int) -> tuple[Tensor, Tensor]:
    """each gpu's units of layer li's n experts of a table (a copy's n slots, padded with slot 0, their weights a token's: j*T + t, then the next
    copy's) on h (T, lat): (gpus, n, lat) and the weights (n,). entry j*T+t is token t's j-th expert, from the copy where its weight isn't 0,
    the act in the q8 of the down gemv's input"""
    u, ys, ws = self.units, [], []
    for c, types in enumerate(self.copies):
      if types[li] is None: continue
      (gu_type, down_type), stride, at = types[li], self.spans[c] * self.part, 1 + 2 * c * n
      off = self.nbytes(gu_type, 2 * u * self.lat)
      slots, w = table[at:at + n].bitcast(dtypes.int32), table[at + n:at + 2 * n].bitcast(dtypes.float32)
      if amd_custom_kernels_supported(h.device):
        gu = expert_linear(self.pools[gu_type], slots, h, 2 * u, self.lat, gu_type, stride, split=True, live=w)
        ys.append(expert_linear(self.pools[down_type], slots, gu, self.lat, u, down_type, stride, off, act=self.act, live=w))
      else: ys.append(self.gemv(c, slots, self.act(*self.gemv(c, slots, h, 2 * u, self.lat, gu_type, 0).split(u, -1)), self.lat, u, down_type, off))
      ws.append(w)
    return functools.reduce(Tensor.add, [(w.reshape(1, -1, 1) != 0).where(y, 0) for y, w in zip(ys, ws)]), functools.reduce(Tensor.add, ws)

  @staticmethod
  def heaviest_first(ys:list[Tensor], ws:list[Tensor], T:int) -> Tensor:
    """(gpus, T, lat): each token's expert outputs of the parts ys (gpus, n, lat), entry j*T+t token t's, weighted by ws (n,) and summed
    heaviest first (a tie: the earlier entry first), the same sum whichever part an expert came in"""
    rows = [(y[:, j * T:(j + 1) * T], w[j * T:(j + 1) * T].reshape(1, T, 1)) for y, w in zip(ys, ws) for j in range(w.shape[0] // T)]
    rank = Tensor.stack(*[sum(((v >= w) if i < j else (v > w)).cast(dtypes.int32) for i, (_, v) in enumerate(rows) if i != j)
                          for j, (_, w) in enumerate(rows)]).contiguous()
    return functools.reduce(Tensor.add, [functools.reduce(Tensor.add, [(rank[j] == r).where(y * w, 0) for j, (y, w) in enumerate(rows)])
                                         for r in range(len(rows))])

  def gemv(self, c:int, slots:Tensor, x:Tensor, out_f:int, in_f:int, qtype:int, off:int) -> Tensor:
    """(gpus, n, out_f): W @ x[k % rows] (or a gpu's x[k]) of the W of quant type qtype at off bytes in copy c's slots[k] of each gpu's pool,
    decoded (no kernel)"""
    n, size, isz = slots.shape[0], self.nbytes(qtype, out_f * in_f), self.word.itemsize
    w = self.pool.reshape(len(self.devs), -1, self.spans[c] * self.part // isz)[:, slots][..., off // isz:(off + size) // isz]
    w = w.bitcast(dtypes.uint8)
    w = ggml_data_to_tensor(w.flatten(), prod(w.shape) // size * out_f * in_f, qtype).reshape(-1, n, out_f, in_f)
    return (w * (x if x.ndim == 3 else x[Tensor.arange(n) % x.shape[0]]).unsqueeze(-2)).sum(-1)

  # *** a prompt's window: the blocks a layer at a time over up to W tokens (layer major: a layer's experts read once a window). a layer's
  # attention, the mixes and the shared experts in steps of a chunk, its tokens' routed experts once over the window. the residual streams of
  # the window's tokens in a bank, a layer adds to its last one. the graphs are a kind's (blocks of the same buffers): a block's buffers its
  # args, the first of the kind's block captures it

  def window_init(self, W:int):
    """the window's variables and graphs (captured once its state is lent from the pool: window_capture), its state's bytes"""
    m = self.model
    self.W, S = W, 1 + sum(map(self.new_stream, m.blk))
    self.window_bytes, self.piece_bytes = (sum(round_up(prod(sh) * dt.itemsize, 256) for sh, dt in f().values())
                                           for f in (self.window_shapes, self.piece_shapes))
    self.v_p0 = UOp.variable("p0", 0, self.max_context - 1)  # (a piece's first token in the read window)
    self.v_c0, self.v_wlive = UOp.variable("c0", 0, W - self.chunk), UOp.variable("wlive", 1, W)
    self.v_n, self.v_s = UOp.variable("streams", 1, S), UOp.variable("stream", 0, S - 1)
    self.v_tap = UOp.variable("tap", 0, self.piece_shapes()["taps"][0][3] - 1)
    self.arenas:dict = {}  # (the window's graphs never run at the same time: one arena for their scratch)
    self.wjit = functools.partial(TinyJit, arenas=self.arenas, capture_first=True)
    self.tap_jit, self.mix_jit = self.wjit(self.tap), self.wjit(self.tap_mix)
    self.fill_jit = self.wjit(self.fill) if self.draft else None
    # (a window's routed experts in tiles of an expert's rows sized by its rows an expert: few tokens, 16 rows, else 64)
    self.routes = {t: (self.wjit(functools.partial(self.window_route, t)),
                       {ty: self.wjit(functools.partial(self.window_experts, t, ty)) for ty in dict.fromkeys(self.types)}) for t in TILES}
    self.start_jit = self.wjit(self.start)

  def window_lend(self):
    """the window's state lent from the pool's last slots, its graphs captured on it, their scratch (one arena) lent from the pool's first slots
    past the caches' (what of them it takes kept from the experts: pinned)"""
    self.tail = self.lent_from = self.slots - self.slots_of(self.window_bytes) - self.slots_of(self.piece_bytes)
    self.win = {**self.window_state(self.tail), **self.piece(self.slots - self.slots_of(self.piece_bytes))}
    first, key = self.slots_of(self.cache_bytes), (self.devs, 0)
    if first >= self.tail:
      raise MemoryError(f"the pool's {self.slots * self.part / 1e9:.2f} GB under the caches' {self.cache_bytes / 1e9:.2f} GB and the window's "
                        f"{(self.slots - self.tail) * self.part / 1e9:.2f} GB: no room for the window's graphs' scratch (a smaller CTX or WINDOW)")
    room = self.lend({"arena": (((self.tail - first) * self.part,), dtypes.int8)}, first)[0]["arena"].uop.base
    self.arenas.clear()
    self.arenas[key] = room
    self.restore()
    try: self.window_capture()
    except MemoryError as e: raise MemoryError(f"the window's graphs' scratch past the pool's {room.max_numel() / 1e9:.2f} GB for it") from e
    if (need:=self.arenas[key].max_numel()) > room.max_numel():
      raise MemoryError(f"the window's graphs' scratch {need / 1e6:.1f} MB past the pool's {room.max_numel() / 1e6:.1f} MB for it")
    jits = [j for _, j, _ in self.kinds.values()] + [j for r, e in self.routes.values() for j in (r, *e.values())] + \
           [self.start_jit, self.tap_jit, self.mix_jit, self.fill_jit]
    used = max((sum(u.marg[0]) for j in jits if j is not None and j.captured is not None for u in j.captured._linear.toposort()
                if u.op is Ops.SHRINK and u.src[0] is room), default=0)  # (a planned buffer: the arena's (start, bytes))
    self.arena, st, sl = (first, first + self.slots_of(used)), self.state, slice(first, first + self.slots_of(used))
    self.drop(sl)
    st["mask"][sl] = 1

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
      self.graph(jit, *vs, *args)
    self.graph(self.start_jit, self.v_p0.bind(0))
    self.graph(self.tap_jit, self.v_s.bind(0), self.v_tap.bind(0))
    if self.fill_jit: self.graph(self.fill_jit, self.v_pos.bind(0), self.v_p0.bind(0), self.v_c0.bind(0))
    for b in moe: self.dry(b)  # (and once on every moe layer's host buffers: their first use maps them on each gpu, a first window would wait)

  def dry(self, b:FFNBlock):  # (moe block b's route and experts in each tile size on a dry chunk)
    li = self.m["layers"].index(self.model.blk.index(b))
    for route, experts in self.routes.values():
      self.graph(route, self.v_wlive.bind(self.chunk), b.exp_probs_b["bias"], Tensor(self.tags[li]), self.parts[1][li])
      self.graph(experts[self.types[li]], self.pool, self.parts[0][li])

  def window_shapes(self) -> dict[str, tuple[tuple[int, ...], DType]]:
    """the window's compute state (a piece of W tokens'): the moe's inputs and outputs (the routed experts' inputs and outputs, the shared
    experts' outputs, the router's logits), its routes, a gpu's rows of the experts' activations and outputs"""
    c, W, E, K = self.model.config, self.W, self.m["n_experts"], self.model.config.num_experts_per_tok
    R, f32, i32 = max(map(self.rows, TILES)), dtypes.float32, dtypes.int32
    return {"h": ((W, self.lat), f32), "shared": ((1, W // len(self.devs), c.dim), f32), "logits": ((W, E), f32), "out": ((W, self.lat), f32),
            "w": ((W, K), f32), "pos": ((W * K,), i32), "entries": ((R,), i32), "tiles": ((R // TILES[0],), i32),
            "act": ((R, self.units), dtypes.half), "down": ((R, self.lat), dtypes.half)}

  def window_state(self, at:int) -> dict[str, Tensor]:  # (lent from the pool's slots from at on)
    return {k: Tensor(t.uop.unshard(0)) if k in ("act", "down", "shared") else t for k, t in self.lend(self.window_shapes(), at)[0].items()}

  def piece_shapes(self) -> dict[str, tuple[tuple[int, ...], DType]]:
    """a piece's: the bank of its W tokens' residual streams, the draft's inputs. a gpu's rows: its part of every chunk (1, 1, W/gpus, ...)"""
    c, W = self.model.config, self.W // len(self.devs)
    S = self.bank_streams or 1 + sum(map(self.new_stream, self.model.blk))
    return {"bank": ((1, 1, W, S, c.dim), self.bank), "taps": ((1, 1, W, max(len(self.taps), 1), c.dim), dtypes.float)}

  def piece(self, at:int) -> dict[str, Tensor]:  # (lent from the pool's slots from at on: (1, gpus, W/gpus, ...) split on the gpus)
    return {k: Tensor(t.uop.unshard(1)) for k, t in self.lend(self.piece_shapes(), at)[0].items()}

  def own(self, t:Tensor, c0:UOp) -> Tensor:  # (the rows of the chunk at c0 of a piece's (1, gpus, W/gpus, ...): each gpu's own)
    n = self.chunk // len(self.devs)
    return t[:, :, c0 // len(self.devs):c0 // len(self.devs) + n]

  def whole(self, t:Tensor) -> Tensor:  # (t (1, R, ...) split on its rows, on every gpu: each gpu's rows copied to the others once, a buffer)
    n = t.shape[1] // len(self.devs)
    return Tensor.cat(*(t[:, i * n:(i + 1) * n] for i in range(len(self.devs))), dim=1).contiguous()

  def in_order(self, t:Tensor, r0:int=0, n:int|None=None) -> Tensor:
    """a piece's (1, gpus, W/gpus, ...) rows r0..r0+n on every gpu in its tokens' order (1, n, ...): their chunks' gathered"""
    G, C, rest, n = len(self.devs), self.chunk, t.shape[3:], self.W - r0 if n is None else n
    k0, k1 = r0 // C, -(-(r0 + n) // C)
    w = self.whole(t[:, :, k0 * C // G:k1 * C // G].flatten(1, 2)).reshape(1, G, k1 - k0, C // G, *rest)
    return w.permute(0, 2, 1, 3, *range(4, 4 + len(rest))).reshape(1, (k1 - k0) * C, *rest)[:, r0 - k0 * C:r0 - k0 * C + n]

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
      if key not in self.kinds:
        self.kinds[key] = (b, self.saved_graph(f"kind{sum(len(k) == 3 for k in self.kinds)}", self.wjit(functools.partial(fxn, b))), [])
      self.kinds[(name, b)] = (*self.kinds[key][:2], [Tensor(u) for u in seen])
    first, jit, args = self.kinds[(name, b)]
    return jit, args, first

  def attend(self, b:FFNBlock, pos:UOp, live:UOp, c0:UOp, n:UOp, s:UOp, *_) -> Tensor:
    """a chunk's live rows' attention at c0 of the window (positions pos..) on its n streams, added to stream s, then its feed forward's mix: a
    moe's parts before the routed experts into the window's buffers, a dense block's feed forward added to stream s. the rows symbolic: a
    kernel's work the live rows'"""
    rows = (c0 // len(self.devs), self.chunk // len(self.devs))
    bank = Tensor(self.win["bank"].uop)
    h = self.whole(self.mix(getattr(b, "attn_res_score", None), bank.flatten(1, 2), b.attn_norm, n, rows).cast(self.bank))[:, :live]
    mask = b._indexer_mask(h, (pos,), None)
    a = self.partials(lambda: b._attention(h, (pos,), mask, live))
    bank = Tensor(self.win["bank"].uop.after(self.add(s, c0, a).uop))
    x = self.whole(self.mix(getattr(b, "ffn_res_score", None), bank.flatten(1, 2), b.ffn_norm, s + 1, rows).cast(self.bank))[:, :live]
    if not hasattr(b, "ffn_gate_inp"): return self.add(s, c0, self.partials(lambda: b._feed_forward(x))).realize()
    logits = (w(x[0]) if isinstance(w:=self.scorer(b), Linear) else x[0] @ w.T).float()
    logits = logits.reshape(live, len(self.devs), 2, -1)[:, :, 0].reshape(live, -1).contiguous()
    logits = allgather_last(logits)  # (a gpu's experts' logits, everyone's on every gpu)
    outs = [(self.win["h"], b._expert_input(x)[0]), (self.win["logits"], logits)]
    shared = self.own(Tensor(self.win["shared"].uop).unsqueeze(0), c0)
    return Tensor.realize(*[Tensor(t.uop)[c0:c0 + self.chunk].assign(v.pad_to((self.chunk, v.shape[1])).cast(t.dtype)) for t, v in outs],
                          shared.assign(self.split(self.partials(lambda: b._shared(x))).cast(shared.dtype)))[0]

  @staticmethod
  def new_stream(b:FFNBlock) -> bool: return hasattr(b, "attn_res_score") and b.attn_res_score.new_block

  def mix(self, res:AttentionResidual|None, xs:Tensor, norm:Callable[[Tensor], Tensor], n:UOp, rows:tuple[UOp, int]) -> Tensor:
    """the bank's streams xs (1, W, S, dim) of each gpu's rows at rows mixed by an attention residual (none: the one stream) and normed"""
    if res is not None: return res.mix(xs, norm, n, rows)
    return norm(xs.reshape(1, len(self.devs), -1, *xs.shape[2:])[:, :, rows[0]:rows[0] + rows[1]].flatten(1, 2)[:, :, 0].float())

  def partials(self, f:Callable[[], Tensor]) -> Tensor:  # (f's output, its row parallel linears' partials (gpus, ...) as they are, if any)
    PARTIALS[0] = True
    try: return f()
    finally: PARTIALS[0] = False

  def add(self, s:UOp, c0:UOp, y:Tensor, more:Tensor|None=None) -> Tensor:
    """y (1, live, dim) (or its partials: split) and more (each gpu its rows) added to stream s of the bank's chunk at c0, its rows past live 0"""
    # (a slice of a symbolic offset, a symbolic length and a symbolic index isn't written: the chunk's whole rows. each gpu its own)
    bank = self.own(Tensor(self.win["bank"].uop), c0)[:, :, :, s]
    return bank.assign((bank + self.split(y) + (0 if more is None else more)).cast(bank.dtype))

  def split(self, y:Tensor) -> Tensor:
    """y (1, live, dim) a chunk's, or its partials (gpus, 1, live, dim) summed (reduce scattered): (1, gpus, chunk/gpus, dim), each gpu its rows
    (the rows past live 0)"""
    G, C = len(self.devs), self.chunk
    if y.ndim == 4: return reduce_scatter(y[:, 0].pad_to((G, C, y.shape[-1]))).reshape(1, G, C // G, -1)
    return self.kept(y.pad_to((1, C, y.shape[2])).reshape(1, G, C // G, -1))

  def kept(self, t:Tensor) -> Tensor:  # (t (1, gpus, ...) on every gpu split on axis 1: each gpu keeps its own, nothing copied)
    return Tensor(t.uop._shard(1, UOp.range(len(self.devs), 0, AxisType.DEVICE)).unshard(1))

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

  def window_experts(self, tile:int, types:tuple[int, int], pool:Tensor, part1:Tensor, *_) -> Tensor:
    """the window's routed experts (of gate/up and down quant types) once the host gave their slots (part 1): each gpu its units of the experts
    in tiles of an expert's rows, summed over the gpus"""
    K, E, u, off = self.model.config.num_experts_per_tok, self.m["n_experts"], self.units, self.nbytes(types[0], 2 * self.units * self.lat)
    w, pos, entries, h1, h2 = (Tensor(self.win[k].uop) for k in ("w", "pos", "entries", "act", "down"))
    tiles = Tensor(self.win["tiles"].uop)[:self.state_rows(tile)[1]]
    slots = Tensor(part1.uop)[1:1 + E].bitcast(dtypes.int32)
    tslot, xrow = (tiles >= 0).where(slots[tiles.maximum(0)], -1), (entries >= 0).where(entries // K, -1)
    a = self.tiles(tile, pool, tslot, Tensor(self.win["h"].uop), 2 * u, self.lat, types[0], 0, h1, xrow=xrow, act=self.act, live=xrow)
    y = self.tiles(tile, pool, tslot, a, self.lat, u, types[1], off, h2, live=xrow)
    out = (y.reshape(len(self.devs), -1, self.lat)[:, pos.maximum(0)].float() * w.reshape(1, -1, 1)).reshape(len(self.devs), -1, K, self.lat).sum(2)
    return Tensor(self.win["out"].uop).assign(partial_sum(out) * self.model.config.routed_scaling_factor).realize()

  def tiles(self, tile:int, pool:Tensor, tslot:Tensor, x:Tensor, out_f:int, in_f:int, qtype:int, off:int, out:Tensor, xrow:Tensor|None=None,
            act:Callable|None=None, live:Tensor|None=None) -> Tensor:
    """(gpus, tile * tiles, out_f) into out: each gpu's units of the experts (of quant type qtype) at off bytes in the slots of tiles of tile
    rows, on their rows of x (xrow: its rows, -1 its first), act(gate, up) of the outputs' halves: the AMD tensor cores' expert tiles for
    TILE_QUANTS (live: the rows' padding -1, left out), else decoded"""
    if amd_custom_kernels_supported(x.device) and qtype in TILE_QUANTS:
      return expert_tiles(self.pools[qtype], tslot, x, out_f, in_f, qtype, self.spans[0] * self.part, off, xrow=xrow, act=act, out=out, tile=tile,
                          live=live)
    NT, size, isz = tslot.shape[0], self.nbytes(qtype, out_f * in_f), self.word.itemsize
    w = pool.reshape(len(self.devs), -1, self.spans[0] * self.part // isz)[:, tslot.maximum(0)][..., off // isz:(off + size) // isz]
    w = w.bitcast(dtypes.uint8)
    w = ggml_data_to_tensor(w.flatten(), prod(w.shape) // size * out_f * in_f, qtype).reshape(-1, NT, out_f, in_f)
    y = (x if xrow is None else x[xrow.maximum(0)]).reshape(-1, NT, tile, in_f) @ w.transpose(-1, -2)
    y = (y if act is None else act(*y.split(out_f // 2, -1))).cast(out.dtype)
    return Tensor(out.uop).assign(y.reshape(out.shape)).reshape(len(self.devs), NT * tile, -1)

  def post(self, b:FFNBlock, live:UOp, c0:UOp, s:UOp, *_) -> Tensor:
    """a chunk's live rows' routed and shared experts' outputs into the block's, added to stream s"""
    y = b._expert_output(Tensor(self.win["out"].uop)[c0:c0 + live].unsqueeze(0), None)
    return self.add(s, c0, y, self.own(Tensor(self.win["shared"].uop).unsqueeze(0), c0)).realize()

  def tap(self, s:UOp, j:UOp, *_) -> Tensor:  # (the draft's input: stream s after a block its tap j)
    return Tensor(self.win["taps"].uop)[:, :, :, j].assign(Tensor(self.win["bank"].uop)[:, :, :, s].float()).realize()

  def tap_mix(self, s:UOp, j:UOp, w:Tensor, *_) -> Tensor:  # (tap j: the streams up to s mixed by an attention residual's weight w)
    bank = Tensor(self.win["bank"].uop).float()
    scores = (bank.rmsnorm(eps=self.model.config.norm_eps) * w).sum(-1, keepdim=True)
    scores = (Tensor.arange(bank.shape[3]).to(bank.device).reshape(-1, 1) <= Tensor(s)).where(scores, -math.inf)
    return Tensor(self.win["taps"].uop)[:, :, :, j].assign((scores.softmax(3) * bank).sum(3)).realize()

  def fill(self, pos:UOp, p0:UOp, c0:UOp, *_) -> Tensor:  # (the draft's cache of a chunk of the piece at p0 of the read window)
    C = self.chunk
    given = self.given[p0 + c0:p0 + c0 + C].reshape(1, C).to(self.devs)
    return self.draft.fill(given, self.whole(self.own(Tensor(self.win["taps"].uop), c0).flatten(1, 2).cast(self.bank)).float(), pos).realize()

  def start(self, p0:UOp, *_) -> Tensor:  # (a piece's bank of streams: its tokens' embeddings, the rest zero. each gpu its rows of the chunks)
    G, C, bank = len(self.devs), self.chunk, Tensor(self.win["bank"].uop)
    emb = self.model.token_embd(self.tok[p0:p0 + self.W].reshape(1, self.W).to(self.devs))
    emb = emb.reshape(1, self.W // C, G, C // G, -1).permute(0, 2, 1, 3, 4).reshape(1, G, self.W // G, 1, -1)
    return bank.assign(self.kept(emb).pad(((0, 0), (0, 0), (0, 0), (0, bank.shape[3] - 1), (0, 0))).cast(bank.dtype)).realize()

  def stage(self, name:str):  # (a start's stage: its seconds since the stage before)
    print(f"start: {name} {time.perf_counter() - self.t_stage:.1f} s", flush=True)
    self.t_stage = time.perf_counter()

  def slots_of(self, n:int) -> int: return -(-n // self.part)  # (the pool's slots n bytes take)

  def drop(self, units:slice|np.ndarray):  # (the experts in the pool's slots units dropped: all of their slots)
    st, n = self.state, len(self.types) * self.m["n_experts"]
    for c in np.unique(st["expert_in"][units]):
      if c >= 0: st["expert_in"][st["slot_of"][c]:st["slot_of"][c] + self.spans[c // n]], st["slot_of"][c] = -1, -1
  @staticmethod
  def nbytes(qtype:int, n:int) -> int: return n // _GGML_QUANT[qtype][0] * _GGML_QUANT[qtype][1]

  def grow(self, n:int):
    """the caches' rows of the first n positions (in steps of a window piece) kept from the experts: their slots' experts dropped (once read,
    if in flight), the slots pinned (step_io's mask, read's keep), the rows zeroed (they were experts': a cache's keys past a context masked)"""
    if n <= self.grown: return
    new, st = min(round_up(n, self.grow_step), self.max_context), self.state
    for (a, first, row, blocks, size), (o, k) in zip(self.stripes, self.caches):
      starts = first + np.arange(blocks) * size
      sl = np.unique(np.concatenate([np.arange((s + self.grown * row) // self.part, (s + new * row - 1) // self.part + 1) for s in starts]))
      c = np.ascontiguousarray((c:=st["expert_in"][sl])[c >= 0], np.int64)
      if len(c) and (err:=host_c().read_wait(ctypes.byref(self.io), len(c), c.ctypes.data)): raise RuntimeError(f"reads failed {err}")
      self.drop(sl)
      st["mask"][sl] = 1
    for g in range(self.grown, new, self.grow_step): self.zero_jit(self.v_grown.bind(min(g, self.max_context - self.grow_step)))
    self.grown = new

  def zero(self, g:UOp):  # (every cache's step of rows at g zeroed)
    rows = [getattr(o, k).shrink(tuple((g, g + self.grow_step) if j == a else None for j in range(getattr(o, k).ndim)))
            for (a, *_), (o, k) in zip(self.stripes, self.caches)]
    Tensor.realize(*[r.assign(r.zeros_like()) for r in rows])

  def pieces(self) -> int:
    """a read window's pieces: lent below the window's state, past the caches' stripes and the scratch (their rows pinned as they grow, the
    rest the experts'), the pool past them and two layers' experts (the one routed, the one read ahead)"""
    free = min(self.tail - self.arena[1], self.tail - int(self.state["mask"].sum()) - 2 * self.m["n_experts"] * self.spans[0])
    return 1 + max(0, free // self.slots_of(self.piece_bytes))

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
    assert self.lent_from >= self.arena[1] and not st["mask"][self.lent_from:self.tail].any(), "a piece over the caches or the scratch"
    self.drop(slice(self.lent_from, None))
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
    moe = sum(hasattr(b, "ffn_gate_inp") for b in self.model.blk)
    def chunks(T:int) -> list[tuple[int, int]]: return [(c0, min(self.chunk, T - c0)) for c0 in range(0, T, self.chunk)]
    for i, b in enumerate(self.model.blk):
      s = n - 1 + int(self.new_stream(b))
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
          for p in self.bufs[li][2][1]: p[1:1 + E] = (self.state["slot_of"][li * E:(li + 1) * E] // self.spans[0]).view(np.uint32)
          self.graph(experts[self.types[li]], self.pool, self.parts[0][li])
          if li + 1 < moe and li + 1 not in ahead and (li in ahead or len(codes) >= 0.9 * E):
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
    old, E, w = st["expert_in"], self.m["n_experts"], self.spans[0]
    keep = (np.arange(self.slots) >= self.lent_from) | ((old >= self.layer * E) & (old < (self.layer + 2) * E)) | (st["mask"] > 0)
    keep[st["slot_of"][busy][st["slot_of"][busy] >= 0]] = True
    keep |= (old >= 0) & (st["flying"][np.maximum(old, 0)] > 0)
    key = np.where(keep, np.inf, np.where(old >= 0, st["freq"][np.maximum(old, 0)], -1.0))[:self.slots // w * w].reshape(-1, w).sum(1)
    victims = np.ascontiguousarray(np.argsort(key, kind="stable")[:len(miss)] * w, np.int64)
    if np.isinf(key[victims // w]).any(): raise RuntimeError(f"window layer {self.layer}: {len(miss)} reads, the pool's free slots fewer")
    self.drop(units:=(victims[:, None] + np.arange(w)).flatten())
    st["slot_of"][miss], st["expert_in"][units] = victims, np.repeat(miss, w)
    io.misses += len(miss)
    if not wait: return host_c().read_ahead(ctypes.byref(io), len(miss), miss.ctypes.data)
    if (err:=host_c().read_codes(ctypes.byref(io), len(miss), miss.ctypes.data, victims.ctypes.data)):
      raise RuntimeError(f"window reads failed {err}")

  # *** the host's side

  def warm(self):
    """every step's graph (and the rollback's, the saves') captured on dry steps (or the start's file's, checked by them): the host's flags
    preset (its tables still zero: no experts). the graphs to the start's file, unless they're its"""
    host_c().flags(ctypes.byref(self.io), 1)
    self.t_stage = time.perf_counter()
    self.dry_steps()
    if self.saved and self.saved["fps"] != self.fingerprints():
      print("start: the graphs file's graphs don't run as they were saved: dropped, capturing", flush=True)
      self.recapture()
    if not self.saved and self.source:
      persist.save(self, self.names, self.graphs(), self.fingerprints())
      self.stage("graphs kept for the next start")

  def dry_steps(self):
    for T, jit in self.jits.items():
      jit(self.v_pos.bind(0))[0].tolist()
      self.stage(f"step graph of {T}")
    for n in range(1, self.tokens): self.model.rollback(n)  # (their jits captured too, after a verify step)
    for i, save in itertools.product(range(len(self.model._sets)), (True, False)): self.model._copy(i, save)  # (and save's)
    if self.chain_jit: self.chain_jit(self.v_pos.bind(0), self.v_row.bind(0)).tolist()
    for jit in self.keep_jits: jit(self.v_at.bind(0))  # (a kept session's copies)
    self.zero_jit(self.v_grown.bind(0))  # (and the caches' growth: its rows zeroed, there's no context yet)
    self.stage("rollbacks, saves, chain, session copies")

  # *** a start's graphs kept (persist.py): every graph named

  def graphs(self) -> dict[str, TinyJit]:
    m = self.model
    return {**{f"step{T}": j for T, j in self.jits.items()}, "keep": self.keep_jits[0], "unkeep": self.keep_jits[1], "zero": self.zero_jit,
            **({"chain": self.chain_jit} if self.chain_jit else {}), **{f"rollback{n}": j for n, j in m._rollbacks.items()},
            **{f"copy{i}{save}": j for (i, save), j in m._copies.items()},
            **({"start": self.start_jit, "tap": self.tap_jit, "mix": self.mix_jit, **({"fill": self.fill_jit} if self.fill_jit else {}),
                **{f"route{t}": r for t, (r, _) in self.routes.items()},
                **{f"experts{t}_{gu}_{d}": j for t, (_, e) in self.routes.items() for (gu, d), j in e.items()},
                **{f"kind{i}": j for i, (_, j, _) in enumerate(v for k, v in self.kinds.items() if len(k) == 3)}} if self.W else {})}

  def saved_graph(self, name:str, jit:TinyJit) -> TinyJit:  # (jit, the start's file's graph of the name if there's one)
    if (captured:=self.saved.get("graphs", {}).get(name)) is not None: jit.captured, jit.cnt = captured, 2
    return jit

  def restore(self):
    """the start's file's graphs into theirs (the model's rollbacks' and saves' made), else they're captured"""
    self.names = persist.buffers(self)
    self.saved = persist.load(self, self.names)
    for name, jit in self.graphs().items(): self.saved_graph(name, jit)
    for name, captured in self.saved.get("graphs", {}).items():
      if name.startswith("rollback"): self.model._rollbacks[int(name[8:])] = _TinyJit(None, captured=captured)
      if name.startswith("copy"): self.model._copies[(int(name[4]), name[5:] == "True")] = _TinyJit(None, captured=captured)

  def fingerprints(self) -> dict[str, str]:
    """the graphs' outputs on defined inputs, run after every graph's first (the gpus' first uses of the host's buffers done): a step of each
    count of tokens of token 0 at position 0 on zeroed recurrent states (and the draft's chain), the window's route of zero logits (its
    selections' rows sorted: within an expert their order is the atomics'). only what's exact: tokens, guesses, the routes' weights"""
    if (states:=self.model._states()): Tensor.realize(*[s.assign(s.zeros_like()) for s in states])
    self.tok_np[:], self.given_np[:] = 0, -1
    fps = {}
    for T, jit in self.jits.items():
      jit(self.v_pos.bind(0))[0].tolist()
      fps[f"step{T}"] = persist.fingerprint(self.res_np[:1 + T + self.width])  # (the tokens and guesses, not the scores' floats)
    if self.chain_jit:
      self.chain_jit(self.v_pos.bind(0), self.v_row.bind(0)).tolist()
      fps["chain"] = persist.fingerprint(self.cres_np[:1 + self.links])
    if self.W:
      Tensor(self.win["logits"].uop).assign(Tensor(self.win["logits"].uop).zeros_like()).realize()
      b = next(b for b in self.model.blk if hasattr(b, "ffn_gate_inp"))
      route, li = self.routes[TILES[0]][0], self.m["layers"].index(self.model.blk.index(b))
      self.graph(route, self.v_wlive.bind(self.chunk), b.exp_probs_b["bias"], Tensor(self.tags[li]), self.parts[1][li])
      fps["route"] = persist.fingerprint(*(self.win[k].numpy() for k in ("w", "tiles")), *(np.sort(self.win[k].numpy()) for k in ("pos", "entries")))
    return fps

  def recapture(self):
    """the start's file dropped, every graph captured again (the window's on a state lent again)"""
    persist.discard(self)
    self.saved = {}
    for jit in self.graphs().values():
      if jit.fxn is not None: jit.reset()
    self.model._rollbacks.clear()
    self.model._copies.clear()
    if self.W: self.window_lend()
    self.dry_steps()

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
    self.step_s[T].append(time.perf_counter() - t0 + self.chain_s)
    self.chain_s = 0.0
    scores = ret[T + G:].view(np.float32).copy() if self.draft is not None and self.draft.scored else None
    return ret[:T].tolist(), ret[T:T + G].tolist(), scores

  def proposal(self, g:list[int], score:np.ndarray|None, pos:int, row:int) -> list[int]:
    """a verify step's guesses after the token at pos: the row's of the step before (guessed at its row), the draft's chain after them. scored:
    as many as the step of the widths that expects most tokens a second (by the scores, a guess the chain hasn't made yet scored as the last,
    the steps' measured seconds: one not measured first), the chain run a guess further while a wider step's expected"""
    chain = None
    while True:
      n = self.tokens - 1
      if score is not None and len(self.widths) > 1:
        s = np.concatenate([score, np.full(n, score[-1])])
        def rate(T:int) -> float: return (1 + np.cumprod(s[:T - 1]).sum()) / np.mean(t) if (t:=self.step_s[T]) else math.inf
        n = max(self.widths, key=rate) - 1
      if len(g) >= n: return g[:n]
      guess, prob = next(chain:=chain or self.chain(g[-1], pos, row))
      g, score = [*g, guess], None if score is None else np.append(score, prob)

  def chain(self, g:int, pos:int, row:int) -> Iterator[tuple[int, float]]:
    """the draft's guesses after g (at pos+1, guessed at the step's row) for a verify step and their probabilities, a chain step's at a time"""
    while self.chain_jit is not None:
      t0 = time.perf_counter()
      self.given_np[0], self.cres_np[0], self.io.tokens, self.io.victims_n = g, 0, 1, 32
      host_c().flags(ctypes.byref(self.io), 0)
      self.chain_jit(self.v_pos.bind(pos), self.v_row.bind(row))
      if (err:=host_c().layer(ctypes.byref(self.io), len(self.bias) - 1) if self.draft.blk else 0): host_c().flags(ctypes.byref(self.io), 1)
      st = time.perf_counter()
      while self.cres_np[0] != 1:
        if time.perf_counter() - st > 60: raise RuntimeError("the draft's chain didn't finish")
      if err: raise RuntimeError(f"the draft's layer: {err}")
      self.chain_s += time.perf_counter() - t0
      guesses, pos = self.cres_np[1:1 + self.links].tolist(), pos + self.links
      yield from zip(guesses, self.cres_np[1 + self.links:].view(np.float32).tolist())
      g = guesses[-1]

  # *** master's model's generate (LLMServer's): the caches only go forward, a prompt that extends the tokens fed (or the last prompt) feeds
  # the rest

  @property
  def max_context(self) -> int: return self.model.max_context
  def get_start_pos(self, tokens:list[int]) -> int:
    self.resume(tokens)
    return self.model.get_start_pos(tokens)

  def session_rows(self, i:int, at:UOp) -> Tensor:  # (cache i's rows of a kept session's at..)
    o, k = self.caches[i]
    return getattr(o, k).shrink(tuple((at, at + self.B) if j == self.stripes[i][0] else None for j in range(getattr(o, k).ndim)))

  def session_states(self) -> list[Tensor]: return [*self.model._states(), *itertools.chain(*self.model._sets)]
  def keep(self, at:UOp):  # (a session's rows at.. and recurrent states into the copies)
    states = zip(self.stash_states, self.session_states())
    Tensor.realize(*[c.assign(self.session_rows(i, at)) for i, c in enumerate(self.stash)], *[c.assign(s) for c, s in states])

  def unkeep(self, at:UOp):  # (and back)
    states = zip(self.stash_states, self.session_states())
    Tensor.realize(*[self.session_rows(i, at).assign(c) for i, c in enumerate(self.stash)], *[s.assign(c) for c, s in states])

  def resume(self, tokens:list[int]):
    """the kept session back when tokens share more with it than the tokens fed are (if the request since wrote only its copied rows), else
    the tokens fed kept when tokens diverge from them early and are fewer (a short request: a title, a summary), unless they're a short
    request's too and one longer is kept"""
    m = self.model
    def shared(a:list[int]) -> int: return sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens, a)))
    if self.session is not None and m._cached_tokens != self.session[0] and \
       round_up(len(m._cached_tokens) + self.tokens, max(self.chunk, 1)) > self.session[1] + self.B: self.session = None
    if self.session is not None and shared(kept:=self.session[0]) > max(shared(m._cached_tokens), len(m._cached_tokens)):
      self.keep_jits[1](self.v_at.bind(self.session[1]))
      m._cached_tokens, m._saved, self.session = kept[:], self.session[2], None
    elif len(tokens) < len(m._cached_tokens) and (at:=shared(m._cached_tokens)) + self.B <= self.max_context and len(tokens) - at < self.B // 2 \
         and (self.session is None or len(fed:=m._cached_tokens) > len(self.session[0]) or len(fed) - at >= self.B and fed != self.session[0]):
      self.keep_jits[0](self.v_at.bind(at))
      self.session = (m._cached_tokens[:], at, m._saved[:])

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
    start = self.get_start_pos(tokens)
    # (where tokens diverge from the tokens fed before, a window's any length: its states saved)
    if (fork:=self.model.fork(tokens, start, 1 if self.W else self.chunk or 1)) > start:
      for pos in range(self.prefill(tokens[:fork + 1], start), fork): self.step(tokens[pos:pos + 1], pos, tokens[pos + 1:pos + 2])
      self.model.save(tokens[:fork])
    for pos in range(self.prefill(tokens, fork), len(tokens)):
      (t,), g, score = self.step(tokens[pos:pos + 1], pos, tokens[pos + 1:pos + 2])
    self.model.save(tokens)
    row, c0, v0, widths = 0, self.counters(), self.verified.copy(), collections.Counter()
    try:
      while len(tokens) + self.tokens <= self.max_context:  # (a verify step's last token in the context)
        tokens.append(t)
        self.model._cached_tokens = tokens[:-1]
        yield t
        if self.draft is None:
          (t,), _, _ = self.step([t], len(tokens) - 1)
          widths[1] += 1
          continue
        gs = self.proposal(g, score, len(tokens) - 1, row)
        toks, g, score = self.step([t, *gs], len(tokens) - 1)
        widths[len(toks)] += 1
        row = next((j for j, g in enumerate(gs) if toks[j] != g), len(gs))
        self.verified += (1, len(gs), row)
        if row < len(gs): self.model.rollback(row + 1)  # (before the kept ones are yielded: a request may stop at any)
        for kept in gs[:row]:
          tokens.append(kept)
          self.model._cached_tokens = tokens[:]
          yield kept
        t = toks[row]
    finally:
      if (n:=sum(widths.values())):
        dt, _, refs, hits, *_, gpu, _, reads, _, _, after = (self.counters() - c0) / n
        print(f"decode: {n} steps, {1 + (self.verified[2] - v0[2]) / n:.2f} tokens a step (widths {dict(sorted(widths.items()))}), "
              f"{dt * 1e3:.1f} ms a step: {gpu * 1e3:.1f} waiting on the gpus, {reads * 1e3:.1f} on reads, {after * 1e3:.1f} after the last layer; "
              f"{hits / max(refs, 1):.1%} of experts cached", flush=True)

  def counters(self) -> np.ndarray:
    io = self.io
    return np.array([time.perf_counter(), io.bytes, io.refs, io.hits, io.late, io.misses, io.guesses, io.used, io.fills,
                     io.t_gpu, io.t_route, io.t_reads, io.t_after, *self.times])

  def report(self, c0:np.ndarray, n:int):
    """a step's time, reads and their split over the n steps since the counters c0"""
    dt, reads, refs, hits, late, misses, guesses, used, fills, *ms = (self.counters() - c0) / n
    print(f"{dt * 1e3:.1f} ms/token = {1 / dt:.2f} tok/s, {reads / 1e9:.2f} GB/token read: {misses:.0f} misses, "
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
  # STORE=at,..: the store's copies of the experts (their manifests' offsets on disk 0)
  stores = [open_store(list(model.token_embd.weight.device), int(at)) for at in getenv("STORE", str(512 << 30)).split(",")]  # (k3's iq3_xxs store)
  # DRAFT=dir: the draft (eagle's head, dspark's blocks; mtp: the model's own next token prediction layer), speculative decoding with verify
  # steps of VERIFY tokens (memory: fewer SLOTS)
  # VERIFY=n: a verify step's tokens, the next and the draft's n-1 guesses (eagle's: a chain) (an EVAL: its steps of n tokens)
  # (VERIFY=n,n..: a scored draft's verify steps of any of them, each step's by the guesses' scores)
  widths = [int(x) for x in str(getenv("VERIFY", "2")).split(",")]
  draft = (MTP(model) if d == "mtp" else load(d, lambda t: model.token_embd(t).float(), model.output, model.token_embd.weight.device,
                                               model.max_context, max(widths) - 1)) if (d:=getenv("DRAFT", "")) else None
  print(f"start: store, draft {time.perf_counter() - st:.0f} s in", flush=True)
  # CHUNK=n: a prompt fed in steps of n tokens (0: a token a step), WINDOW=n: in windows of up to n tokens (a layer at a time, chunks of CHUNK)
  # TAPS=b,b..: without a draft, the streams a window keeps after blocks b (bm: as the next block's attention residual mixes them), for a DUMP
  taps = [(int(t.rstrip("m")), t.endswith("m")) for t in getenv("TAPS", "").split(",") if t]
  eng = Engine(model, stores[0][1], tokens=max(widths) if draft else 1, guess=getenv("GUESS", ""), draft=draft, logits=bool(getenv("EVAL", "")),
               chunk=getenv("CHUNK", 256), window=getenv("WINDOW", 0), taps=taps, widths=tuple(widths), source=sys.argv[1],
               copies=tuple(m for _, m in stores[1:]))
  eng.drives(*(d for d, _ in stores))
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
