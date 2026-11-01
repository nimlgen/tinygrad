#!/usr/bin/env python3
# extra/k3/engine.py's step_io (the host's part of a step) against fake drives: a thread completes every command rung on a queue (in any
# order, as drives do), the gpus' router tags are set up front. checks the slots and weights the gpus get, the cache tables, the guesses
# and the counts.
# PYTHONPATH=. python3 extra/k3/test_step_io.py
import ctypes, threading, time, unittest
import numpy as np
from extra.k3.engine import IoT, host_c, BUFS, CBUFS

E, L, NGPU, K, KMIN, ND, ENTRIES, ROWS = 64, 6, 2, 8, 2, 2, 64, 2

class FakeDrives(threading.Thread):
  """every command rung on a queue completed, a random one of those in flight at a time. slow: the first of each queue only after two
  rings' worth of others (or when it's the last in flight for 5 ms). dups: the commands whose id was in flight already"""
  def __init__(self, sq, db, cq, slow=False):
    super().__init__(daemon=True)
    self.sq, self.db, self.cq, self.stop, self.dups, self.rng = sq, db, cq, False, 0, np.random.default_rng(2)
    self.head, self.tail, self.phase, self.flying = [0] * len(sq), [0] * len(sq), [1] * len(sq), [[] for _ in sq]
    self.held, self.passed, self.last = [0 if slow else -1] * len(sq), [0] * len(sq), [0.0] * len(sq)
  def run(self):
    while not self.stop:
      for q in range(len(self.sq)):
        while self.head[q] != int(self.db[q][0]):
          cid = int(self.sq[q][self.head[q] * 16]) >> 16
          self.dups += cid in self.flying[q]
          self.flying[q].append(cid)
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
  def setUp(self, slots=200, T=1, slow=False, parts=1):
    SLOTS = self.slots = slots
    rng = np.random.default_rng(0)
    self.keep:list[np.ndarray] = []
    def arr(a) -> np.ndarray:
      self.keep.append(a := np.ascontiguousarray(a))
      return a
    nq = 2 * ND
    self.sq, self.cq = [arr(np.zeros(ENTRIES * 16, np.uint32)) for _ in range(nq)], [arr(np.zeros(ENTRIES * 4, np.uint32)) for _ in range(nq)]
    self.db, self.cq_db = [arr(np.zeros(1, np.uint32)) for _ in range(nq)], [arr(np.zeros(1, np.uint32)) for _ in range(nq)]
    # a code's parts on each drive: a command each
    ncodes, P = L * E, ND * parts
    tables:dict[str, np.ndarray] = {"ptr": np.arange(0, P * ncodes + 1, P, dtype=np.int64),
              "drive": np.tile(np.arange(ND, dtype=np.int32), ncodes * parts),
              "lba": np.zeros(P * ncodes, np.int64), "nlb": np.zeros(P * ncodes, np.uint32), "kind": np.zeros(P * ncodes, np.int8),
              **{k: np.zeros(P * ncodes, np.int64) for k in ("inpart", "page_base", "list_base")}, "part_pages": np.ones(P * ncodes, np.int64),
              "pages": np.arange(SLOTS + 1, dtype=np.int64), "lists": np.zeros(1, np.int64),
              "sq": np.array([a.ctypes.data for a in self.sq], np.uint64), "db": np.array([a.ctypes.data for a in self.db], np.uint64),
              "cq": np.array([a.ctypes.data for a in self.cq], np.uint64), "cq_db": np.array([a.ctypes.data for a in self.cq_db], np.uint64),
              **{k: np.zeros(nq, np.int64) for k in ("tails", "db_tails", "heads", "inflight", "cids")}, "phases": np.ones(nq, np.int64),
              "owner": np.full(nq * 65536, -2, np.int64)}
    self.bias = [arr(rng.normal(0, 0.1, E).astype(np.float32)) for _ in range(L)]
    self.state:dict[str, np.ndarray] = {
      "slot_of": np.full(L * E, -1, np.int32), "expert_in": np.full(SLOTS, -1, np.int64), "freq": np.zeros(L * E, np.float32),
      "flying": np.zeros(L * E, np.int32), "guess": np.zeros(L * E, np.uint8), "mask": np.zeros(SLOTS, np.uint8),
      "victims": np.zeros(max(32 * T, E), np.int64), "bias": np.array([b.ctypes.data for b in self.bias], np.uint64)}
    self.keep += [*tables.values(), *self.state.values()]
    # the gpus' buffers of every layer: tag, ready, ready1, scores (ROWS of E / NGPU a token), slots and weights (slot, token) of both parts
    nr, k1 = E // NGPU, K // 2
    def gpu():
      return {"tag": arr(np.ones(1, np.uint32)), "ready": arr(np.zeros(1, np.uint32)), "ready1": arr(np.zeros(1, np.uint32)),
              "scores": arr(rng.normal(0, 2, (T * ROWS, nr)).astype(np.float32)), "slots": arr(np.zeros(k1 * T, np.int32)),
              "ws": arr(np.zeros(k1 * T, np.float32)), "slots2": arr(np.zeros((K - k1) * T, np.int32)),
              "ws2": arr(np.zeros((K - k1) * T, np.float32))}
    self.gbuf = [[gpu() for _ in range(NGPU)] for _ in range(L)]
    self.bufs = arr(np.array([[[g[b].ctypes.data for g in self.gbuf[li]] for b in BUFS] for li in range(L)], np.uint64))
    self.io = io = IoT(layers=L, E=E, ngpu=NGPU, k=K, kmin=KMIN, rows=ROWS, slots=SLOTS, tokens=T, k1=k1, nd=ND, entries=ENTRIES,
                       victims_n=32 * T, pf=3, tau=0.03, decay=0.5 ** (1 / 8), bufs=self.bufs.ctypes.data)
    for name, a in {**tables, **self.state}.items(): setattr(io, name, a.ctypes.data)
    self.drives = FakeDrives(self.sq, self.db, self.cq, slow)
    self.drives.start()
  def tearDown(self):
    self.drives.stop = True
    self.assertEqual(self.drives.dups, 0)

  def test_small_cache(self):  # (the slots go to later layers within a step: only the tables and counts)
    self.tearDown()
    self.setUp(slots=40)
    for _ in range(5):
      self.step()
      self.check_tables()
    self.check_counts()

  def step(self):
    for li in range(L):
      for g in self.gbuf[li]: g["ready"][0] = g["ready1"][0] = 0
    self.assertEqual(host_c().step_io(ctypes.byref(self.io)), 0)

  def check_tables(self):
    slot_of, expert_in = self.state["slot_of"], self.state["expert_in"]
    for sl, c in enumerate(expert_in):
      if c >= 0: self.assertEqual(slot_of[c], sl)
    for c in np.nonzero(slot_of >= 0)[0]: self.assertEqual(expert_in[slot_of[c]], c)

  def check_counts(self):
    io = self.io
    self.assertEqual(io.refs, io.hits + io.late + io.misses)
    self.assertGreater(io.guesses, 0)
    self.assertGreater(io.hits, 0)
    # every guess is of a layer of the step: settled (used or not) by its end
    self.assertEqual(io.used + io.wasted, io.guesses)
    self.assertEqual(int(self.state["guess"].sum()), 0)

  def test_steps(self):
    for _ in range(5):
      self.step()
      self.check_tables()
      for li in range(L):
        so = self.state["slot_of"][li * E:(li + 1) * E]
        for g in self.gbuf[li]:
          self.assertEqual((g["ready"][0], g["ready1"][0]), (1, 1))
          slots, ws = np.concatenate([g["slots"], g["slots2"]]), np.concatenate([g["ws"], g["ws2"]])
          self.assertAlmostEqual(float(ws.sum()), 1.0, places=5)
          # every slot given with a weight holds one of this layer's experts
          self.assertTrue(all(self.state["expert_in"][s] // E == li for s, w in zip(slots, ws) if w > 0))
          self.assertEqual(len(set(slots[ws > 0].tolist())), int((ws > 0).sum()))
          self.assertTrue(all(so[self.state["expert_in"][s] % E] == s for s in slots[ws > 0]))
      # the gpus get the same slots and weights
      for li in range(L): self.assertTrue(all((self.gbuf[li][0][b] == self.gbuf[li][1][b]).all() for b in ("slots", "ws", "slots2", "ws2")))
    self.check_counts()

  def test_tokens(self):  # a step of 2 tokens: every token its own entries (i*T + t), a weight each
    self.tearDown()
    self.setUp(T=2)
    for _ in range(3):
      self.step()
      self.check_tables()
      for li in range(L):
        g = self.gbuf[li][0]
        slots, ws = np.concatenate([g["slots"], g["slots2"]]), np.concatenate([g["ws"], g["ws2"]]).reshape(-1, 2)
        self.assertTrue(np.allclose(ws.sum(0), 1, atol=1e-5))
        self.assertTrue(all(self.state["expert_in"][s] // E == li for s, w in zip(slots, ws.sum(1)) if w > 0))
    self.check_counts()

  def test_fill(self):  # the experts tau skipped for not being cached are read: kept by the next step (same scores)
    self.io.tau, self.io.pf, self.io.fill = 0.2, 0, 4
    self.step()
    refs, fills = self.io.refs, self.io.fills
    self.assertGreater(fills, 0)
    self.step()
    self.check_tables()
    self.assertGreater(self.io.refs - refs, refs)

  def chunk(self, T=40) -> tuple[list, list]:
    """a prompt's chunk of T tokens (random router logits): chunk_io's return checked, the logits and every layer's gpu buffers"""
    nr, rng = E // NGPU, np.random.default_rng(1)
    P = T * K + 16 * E
    def gpu(scores):
      return {"scores": scores, "tslot": np.zeros(P // 16, np.int32), "ntiles": np.zeros(1, np.int32), "rows": np.zeros(P, np.int32),
              "dest": np.zeros(T * K, np.int32), "w": np.zeros(T * K, np.float32)}
    logits = [rng.normal(0, 2, (T, E)).astype(np.float32) for _ in range(L)]
    cbuf = [[gpu(np.ascontiguousarray(logits[li][:, g * nr:(g + 1) * nr])) for g in range(NGPU)] for li in range(L)]
    self.keep += [a for lb in cbuf for gb in lb for a in gb.values()]
    cbufs = np.array([[[gb[b].ctypes.data for gb in cbuf[li]] for b in CBUFS] for li in range(L)], np.uint64)
    for li in range(L):
      for g in self.gbuf[li]: g["ready"][0] = 0
    self.io.tokens, self.io.cbufs = T, cbufs.ctypes.data
    self.assertEqual(host_c().chunk_io(ctypes.byref(self.io)), 0)
    self.check_tables()
    return logits, cbuf

  def test_chunk(self):  # a prompt's chunk: every expert of each layer read in, the exact top k of every token in the grouped gemm's tables
    self.tearDown()
    self.setUp(slots=L * E + 16)  # (none evicted: every layer's slots checked after)
    T = 40
    logits, cbuf = self.chunk(T)
    for li in range(L):
      so = self.state["slot_of"][li * E:(li + 1) * E]
      sig = 1 / (1 + np.exp(-logits[li].astype(np.float64)))
      top = np.argsort(-(sig + self.bias[li]), axis=1, kind="stable")[:, :K]
      g0 = cbuf[li][0]
      tiles = int(g0["ntiles"][0])
      for t in range(T):
        dest = g0["dest"][t * K:(t + 1) * K]
        self.assertTrue((g0["rows"][dest] == t).all())
        self.assertEqual(sorted(so[top[t]].tolist()), sorted(g0["tslot"][dest // 16].tolist()))
        self.assertTrue((dest // 16 < tiles).all())
        self.assertAlmostEqual(float(g0["w"][t * K:(t + 1) * K].sum()), 1.0, places=5)
      # a tile is one expert's: the rows of a tile (padding too) are tokens routed to it
      for p in range(tiles * 16):
        e = int(self.state["expert_in"][g0["tslot"][p // 16]]) - li * E
        self.assertIn(e, top[g0["rows"][p]].tolist())
      for b, n in (("tslot", tiles), ("rows", tiles * 16), ("ntiles", 1), ("dest", T * K), ("w", T * K)):
        self.assertTrue((cbuf[li][1][b][:n] == g0[b][:n]).all())
      self.assertTrue(all(g["ready"][0] == 1 for g in self.gbuf[li]))

  def test_slow_read(self):  # a read passed by two rings' worth of others (a layer's are 8 rings): no command id reused while in flight
    self.tearDown()
    self.setUp(slots=L * E + 16, slow=True, parts=4)
    self.chunk()
    self.assertTrue(all(h == -1 for h in self.drives.held))
    self.assertTrue((self.state["flying"] == 0).all())

if __name__ == "__main__":
  unittest.main()
