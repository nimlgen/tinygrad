#!/usr/bin/env python3
# extra/k3/host.py's step_io (the host's part of a step) against fake drives: a thread completes every command rung on a queue (in any
# order, as drives do), the gpus' tags are set up front. checks the slots and weights the gpus get, where each read lands (its prp), the
# cache tables, the guesses and the counts.
# PYTHONPATH=. python3 extra/k3/test_host.py
import ctypes, threading, time, unittest
import numpy as np
from extra.k3.host import IoT, host_c, BUFS

E, L, NGPU, K, KMIN, ND, ENTRIES, ROWS = 64, 6, 2, 8, 2, 4, 64, 2

class FakeDrives(threading.Thread):
  """every command rung on a queue completed, a random one of those in flight at a time. slow: the first of each queue only after two
  rings' worth of others (or when it's the last in flight for 5 ms). dups: the commands whose id was in flight already. written: the
  drive and lba of each read by its prp1"""
  def __init__(self, sq, db, cq, slow=False):
    super().__init__(daemon=True)
    self.sq, self.db, self.cq, self.stop, self.dups, self.rng, self.written = sq, db, cq, False, 0, np.random.default_rng(2), {}
    self.head, self.tail, self.phase, self.flying = [0] * len(sq), [0] * len(sq), [1] * len(sq), [[] for _ in sq]
    self.held, self.passed, self.last = [0 if slow else -1] * len(sq), [0] * len(sq), [0.0] * len(sq)
  def run(self):
    while not self.stop:
      for q in range(len(self.sq)):
        while self.head[q] != int(self.db[q][0]):
          sqe = self.sq[q][self.head[q] * 16:(self.head[q] + 1) * 16]
          cid = int(sqe[0]) >> 16
          self.dups += cid in self.flying[q]
          self.flying[q].append(cid)
          self.written[int(sqe[6]) | int(sqe[7]) << 32] = (q % ND, int(sqe[10]))
          self.head[q], self.last[q] = (self.head[q] + 1) % ENTRIES, time.perf_counter()
        fl = self.flying[q]
        if not fl: continue
        alone = len(fl) == 1 and time.perf_counter() - self.last[q] > 5e-3
        hold = self.held[q] >= 0 and self.held[q] in fl and not alone and self.passed[q] < 2 * ENTRIES
        i = int(self.rng.integers(len(fl)))
        if hold and fl[i] == self.held[q]: continue
        cid = fl.pop(i)
        if cid == self.held[q]: self.held[q] = -1
        self.passed[q] += 1
        self.cq[q][self.tail[q] * 4 + 3] = cid | self.phase[q] << 16
        if (t:=self.tail[q] + 1) == ENTRIES: t, self.phase[q] = 0, self.phase[q] ^ 1
        self.tail[q] = t

class TestStepIO(unittest.TestCase):
  def setUp(self, slots=200, T=1, slow=False, parts=1, chunk=False):
    """gpu g reads its part of a code (parts pages on each of its ND // NGPU drives, a command each) into its pool, a prp table each. chunk:
    steps of a prompt's chunk of T tokens"""
    SLOTS, rng, self.keep = slots, np.random.default_rng(0), []
    def arr(a) -> np.ndarray:
      self.keep.append(a := np.ascontiguousarray(a))
      return a
    nq, dpg, ncodes = 2 * ND, ND // NGPU, L * E
    self.sq, self.cq = [arr(np.zeros(ENTRIES * 16, np.uint32)) for _ in range(nq)], [arr(np.zeros(ENTRIES * 4, np.uint32)) for _ in range(nq)]
    self.db, self.cq_db = [arr(np.zeros(1, np.uint32)) for _ in range(nq)], [arr(np.zeros(1, np.uint32)) for _ in range(nq)]
    # the commands of a code: gpu, drive, page in the part
    gpu, drive, inpart = (np.tile(np.array(c), ncodes) for c in zip(*[(g, g * dpg + d, d * parts + p) for g in range(NGPU) for d in range(dpg)
                                                                       for p in range(parts)]))
    n = NGPU * dpg * parts
    # the pages of gpu g's pool: g << 32 | page << 12 (and lists after each)
    self.prp = [arr(np.array([(g << 32 | p << 12, 1) for p in range(SLOTS * dpg * parts + 1)], np.uint64).reshape(-1)) for g in range(NGPU)]
    self.tables:dict[str, np.ndarray] = {
      "ptr": np.arange(0, n * ncodes + 1, n, dtype=np.int64), "drive": drive.astype(np.int32), "gpu": gpu.astype(np.int32),
      "lba": np.arange(n * ncodes, dtype=np.int64), "nbytes": np.full(n * ncodes, 4096, np.int64), "inpart": inpart.astype(np.int64),
      "part_pages": np.full(NGPU, dpg * parts, np.int64), "prp": np.array([p.ctypes.data for p in self.prp], np.uint64),
      "sq": np.array([a.ctypes.data for a in self.sq], np.uint64), "db": np.array([a.ctypes.data for a in self.db], np.uint64),
      "cq": np.array([a.ctypes.data for a in self.cq], np.uint64), "cq_db": np.array([a.ctypes.data for a in self.cq_db], np.uint64),
      **{k: np.zeros(nq, np.int64) for k in ("tails", "db_tails", "heads", "inflight", "cids")}, "phases": np.ones(nq, np.int64),
      "owner": np.full(nq * 65536, -2, np.int64)}
    self.bias = [arr(rng.normal(0, 0.1, E).astype(np.float32)) for _ in range(L)]
    self.state:dict[str, np.ndarray] = {
      "slot_of": np.full(L * E, -1, np.int32), "expert_in": np.full(SLOTS, -1, np.int64), "freq": np.zeros(L * E, np.float32),
      "flying": np.zeros(L * E, np.int32), "guess": np.zeros(L * E, np.uint8), "mask": np.zeros(SLOTS, np.uint8),
      "victims": np.zeros(32 * T, np.int64), "bias": np.array([b.ctypes.data for b in self.bias], np.uint64)}
    self.keep += [*self.tables.values(), *self.state.values()]
    # the gpus' buffers of every layer: tag, scores (ROWS of E / NGPU a token), the two parts' tables [ready | slots | weights] (a chunk's in
    # part 1: an entry a token's expert)
    nr, self.k1 = E // NGPU, K // 2
    def gpu_bufs():
      return {"tag": arr(np.ones(1, np.uint32)), "scores": arr(rng.normal(0, 2, (T * ROWS, nr)).astype(np.float32)),
              "part1": arr(np.zeros(1 + 2 * K * T + 17 * (K * T // 16 + E) if chunk else 1 + 2 * self.k1 * T, np.uint32)),
              "part2": arr(np.zeros(1 + 2 * (K - self.k1) * T, np.uint32))}
    self.gbuf = [[gpu_bufs() for _ in range(NGPU)] for _ in range(L)]
    self.bufs = arr(np.array([[[g[b].ctypes.data for g in self.gbuf[li]] for b in BUFS] for li in range(L)], np.uint64))
    self.io = io = IoT(layers=L, E=E, ngpu=NGPU, k=K, kmin=KMIN, k1=self.k1, rows=ROWS, tokens=T, chunk=T if chunk else 0,
                       nd=ND, entries=ENTRIES, shift=9, slots=SLOTS,
                       victims_n=32 * T, pf=3, tau=0.03, decay=0.5 ** (1 / 8), bufs=self.bufs.ctypes.data)
    for name, a in {**self.tables, **self.state}.items(): setattr(io, name, a.ctypes.data)
    self.drives = FakeDrives(self.sq, self.db, self.cq, slow)
    self.drives.start()
  def tearDown(self):
    self.drives.stop = True
    self.assertEqual(self.drives.dups, 0)

  def step(self):
    host_c().flags(ctypes.byref(self.io), 0)
    for li in range(L):
      for g in self.gbuf[li]: g["tag"][0] = 1
    self.assertEqual(host_c().step_io(ctypes.byref(self.io)), 0)

  def parts(self, li:int, g:int) -> tuple[np.ndarray, np.ndarray]: # a gpu's slots and weights of a layer, both parts
    p = [(t, (len(t) - 1) // 2) for t in (self.gbuf[li][g][b] for b in ("part1", "part2"))]
    return np.concatenate([t[1:1 + n].view(np.int32) for t, n in p]), np.concatenate([t[1 + n:].view(np.float32) for t, n in p])

  def check_tables(self):
    slot_of, expert_in, t = self.state["slot_of"], self.state["expert_in"], self.tables
    for sl, c in enumerate(expert_in):
      if c >= 0: self.assertEqual(slot_of[c], sl)
    for c in np.nonzero(slot_of >= 0)[0]: self.assertEqual(expert_in[slot_of[c]], c)
    # every cached expert's commands read into the pages of its slot
    for c in np.nonzero((slot_of >= 0) & (self.state["flying"] == 0))[0]:
      for r in range(t["ptr"][c], t["ptr"][c + 1]):
        page = slot_of[c] * t["part_pages"][t["gpu"][r]] + t["inpart"][r]
        self.assertEqual(self.drives.written[int(t["gpu"][r]) << 32 | int(page) << 12], (t["drive"][r], t["lba"][r]))

  def check_counts(self):
    io = self.io
    self.assertEqual(io.refs, io.hits + io.late + io.misses)
    self.assertGreater(io.guesses, 0)
    self.assertGreater(io.hits, 0)
    # every guess is of a layer of the step: settled (used or not) by its end
    self.assertEqual(io.used + io.wasted, io.guesses)
    self.assertEqual(int(self.state["guess"].sum()), 0)

  def test_read_ahead(self): # reads pending, queued as the drives have room (pump), waited for: each into its slot's pages
    codes, slots = np.arange(3 * E, 3 * E + 40, dtype=np.int64), np.arange(100, 140, dtype=np.int64)
    self.state["slot_of"][codes], self.state["expert_in"][slots] = slots, codes
    host_c().read_ahead(ctypes.byref(self.io), len(codes), codes.ctypes.data)
    self.assertEqual(host_c().pump(ctypes.byref(self.io)), 0)
    self.assertEqual(host_c().read_wait(ctypes.byref(self.io), len(codes), codes.ctypes.data), 0)
    self.assertEqual(int(self.state["flying"].sum()), 0)
    self.check_tables()

  def test_steps(self):
    for _ in range(5):
      self.step()
      self.check_tables()
      for li in range(L):
        so = self.state["slot_of"][li * E:(li + 1) * E]
        for g in range(NGPU):
          self.assertEqual((self.gbuf[li][g]["part1"][0], self.gbuf[li][g]["part2"][0]), (1, 1))
          slots, ws = self.parts(li, g)
          self.assertAlmostEqual(float(ws.sum()), 1.0, places=5)
          # every slot given with a weight holds one of this layer's experts
          self.assertTrue(all(self.state["expert_in"][s] // E == li for s, w in zip(slots, ws) if w > 0))
          self.assertEqual(len(set(slots[ws > 0].tolist())), int((ws > 0).sum()))
          self.assertTrue(all(so[self.state["expert_in"][s] % E] == s for s in slots[ws > 0]))
        # the gpus get the same slots and weights
        self.assertTrue(all((a == b).all() for a, b in zip(self.parts(li, 0), self.parts(li, 1))))
    self.check_counts()

  def test_nan_scores(self): # a gpu's tag 2 (its scores NaN): the step fails at once, not after the wait's timeout
    host_c().flags(ctypes.byref(self.io), 0)
    for g, buf in enumerate(self.gbuf[0]): buf["tag"][0] = 2 if g == 1 else 1
    st = time.perf_counter()
    self.assertEqual(host_c().step_io(ctypes.byref(self.io)), -4)
    self.assertLess(time.perf_counter() - st, 5)

  def test_small_cache(self): # (the slots go to later layers within a step: only the tables and counts)
    self.tearDown()
    self.setUp(slots=40)
    for _ in range(5):
      self.step()
      self.check_tables()
    self.check_counts()

  def test_tokens(self): # a step of 2 tokens: every token its own entries (i*T + t), a weight each
    self.tearDown()
    self.setUp(T=2)
    for _ in range(3):
      self.step()
      self.check_tables()
      for li in range(L):
        slots, ws = self.parts(li, 0)
        self.assertTrue(np.allclose(ws.reshape(-1, 2).sum(0), 1, atol=1e-5))
        self.assertTrue(all(self.state["expert_in"][s] // E == li for s, w in zip(slots, ws) if w > 0))
    self.check_counts()

  def test_token_entries(self): # exact routing: each token's entries only, shared experts read once, no writes past either table
    for T in (2, 4, 8):
      self.tearDown()
      self.setUp(slots=L * E, T=T)
      self.io.tau, self.io.tauc, self.io.beta, self.io.pf, self.io.fill = 0, 0, 0, 0, 0
      guards = []
      for li in range(L):
        for g in range(NGPU):
          # Two tokens share all experts; the others choose their own, so both shared and distinct entries are exercised.
          self.gbuf[li][g]["scores"][ROWS:2 * ROWS] = self.gbuf[li][g]["scores"][:ROWS]
          for kind in ("part1", "part2"):
            old = self.gbuf[li][g][kind]
            buf = np.full(len(old) + 16, 0xa5a5a5a5, np.uint32)
            self.keep.append(buf)
            self.gbuf[li][g][kind] = buf[:len(old)]
            self.bufs[li, BUFS.index(kind), g] = buf.ctypes.data
            guards.append(buf[len(old):])
      self.step()
      refs = 0
      for li in range(L):
        logits = np.concatenate([g["scores"][::ROWS] for g in self.gbuf[li]], axis=1).astype(np.float64)
        sig = 1 / (1 + np.exp(-logits))
        top = np.argsort(-(sig + self.bias[li]), axis=1, kind="stable")[:, :K]
        refs += len(np.unique(top))
        expected = np.take_along_axis(sig, top, axis=1)
        expected /= expected.sum(1, keepdims=True)
        for g in range(NGPU):
          slots, weights = self.parts(li, g)
          codes = self.state["expert_in"][slots].reshape(K, T).T - li * E
          weights = weights.reshape(K, T).T
          for t in range(T):
            order, want_order = np.argsort(codes[t]), np.argsort(top[t])
            np.testing.assert_array_equal(codes[t, order], top[t, want_order])
            np.testing.assert_allclose(weights[t, order], expected[t, want_order], rtol=1e-6)
      self.assertEqual((self.io.refs, self.io.misses), (refs, refs))
      for guard in guards: np.testing.assert_array_equal(guard, 0xa5a5a5a5)

  def test_chunk(self):
    for T in (40, 137): self.check_chunk(T)  # (137: its routes on threads, the tokens split unevenly)

  def check_chunk(self, T:int): # a prompt's chunk: each token's exact top k read (the first layer's every expert, then the guessed and the
    # missed), entry j*T + t's tile row (an expert's entries in tiles of 16, a row its token), the same on every gpu
    self.tearDown()
    self.setUp(slots=L * E + 16, T=T, chunk=True)  # (none evicted: every layer's slots checked after)
    self.step()
    self.check_tables()
    self.assertEqual(int(self.state["flying"].sum()), 0)
    self.assertGreater(self.io.guesses, 0)
    self.assertLessEqual(self.io.misses + self.io.guesses, L * E)
    for li in range(L):
      so = self.state["slot_of"][li * E:(li + 1) * E]
      logits = np.concatenate([g["scores"][::ROWS] for g in self.gbuf[li]], axis=1).astype(np.float64)
      sig = 1 / (1 + np.exp(-logits))
      top = np.argsort(-(sig + self.bias[li]), axis=1, kind="stable")[:, :K]
      for g in self.gbuf[li]:
        self.assertEqual(g["part1"][0], 1)
        n, NT, p = K * T, K * T // 16 + E, g["part1"][1:].view(np.int32)
        tslot, ttok, rows = p[:NT], p[NT:17 * NT], p[17 * NT:17 * NT + n]
        ws = g["part1"][1 + 17 * NT + n:].view(np.float32).reshape(K, T)
        np.testing.assert_allclose(ws.sum(0), 1, atol=1e-5)
        np.testing.assert_equal(ttok[rows], np.arange(n) % T)  # (an entry's tile row: its token's, of its expert's tile)
        nt = rows.max() // 16 + 1  # (the tiles used)
        self.assertEqual(sorted(set((rows // 16).tolist())), list(range(nt)))
        for t in range(T): self.assertEqual(sorted(tslot[rows // 16].reshape(K, T)[:, t].tolist()), sorted(so[top[t]].tolist()))

  def test_partial_chunk(self):
    for C, live in ((40, 7), (256, 137)): self.check_partial_chunk(C, live)  # (137: its routes on threads)

  def check_partial_chunk(self, C:int, live:int): # the table stays a whole chunk's, but only its live rows route and read
    self.tearDown()
    self.setUp(slots=L * E + 16, T=C, chunk=True)
    self.io.tokens = live
    self.step()
    refs = 0
    for li in range(L):
      logits = np.concatenate([g["scores"][::ROWS][:live] for g in self.gbuf[li]], axis=1).astype(np.float64)
      top = np.argsort(-(1 / (1 + np.exp(-logits)) + self.bias[li]), axis=1, kind="stable")[:, :K]
      refs += len(np.unique(top))
      n, NT = K * C, K * C // 16 + E
      for g in self.gbuf[li]:
        p = g["part1"][1:].view(np.int32)
        rows, ws = p[17 * NT:17 * NT + n].reshape(K, C), p[17 * NT + n:].view(np.float32).reshape(K, C)
        np.testing.assert_allclose(ws[:, :live].sum(0), 1, atol=1e-5)
        np.testing.assert_equal(ws[:, live:], 0)
        np.testing.assert_equal(rows[:, live:], 0)
        np.testing.assert_equal(p[NT:17 * NT][rows[:, :live]], np.broadcast_to(np.arange(live), (K, live)))
        so = self.state["slot_of"][li * E:(li + 1) * E]
        for t in range(live): self.assertEqual(sorted(p[:NT][rows[:, t] // 16].tolist()), sorted(so[top[t]].tolist()))
    self.assertEqual(self.io.refs, refs)
    self.assertLessEqual(self.io.misses + self.io.guesses, L * K * live)

  def test_fill(self): # the experts tau skipped for not being cached are read: kept by the next step (same scores)
    self.io.tau, self.io.pf, self.io.fill = 0.2, 0, 4
    self.step()
    refs, fills = self.io.refs, self.io.fills
    self.assertGreater(fills, 0)
    self.step()
    self.check_tables()
    self.assertGreater(self.io.refs - refs, refs)

  def test_slow_read(self): # a read passed by two rings' worth of others, a layer's reads more than a ring: no command id reused in flight
    self.tearDown()
    self.setUp(slots=L * E, slow=True, parts=12)
    for _ in range(3):
      self.step()
      self.check_tables()
    self.assertTrue(all(h == -1 for h in self.drives.held))

if __name__ == "__main__":
  unittest.main()
