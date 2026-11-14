# the host's part of a decode step of a streamed moe, in c (step_io): while the gpus run the step's graph, at every moe layer each gpu
# stores its router scores and its tag (store), the host routes on them, reads the missing experts straight from the drives into the
# gpus' slot pools and gives the gpus the slots and weights in two parts (a wait on each): the first those cached, the gpus work on it
# while the rest are read. the reads go on two host-owned queues of each drive (demand, guesses) the c fills, rings and reaps itself
import ctypes, functools, hashlib, os, subprocess
from typing import cast
import numpy as np
from tinygrad import Device, Tensor
from tinygrad.helpers import cache_dir
from tinygrad.device import Buffer, BufferSpec, MultiBuffer
from tinygrad.dtype import DType
from tinygrad.uop.ops import UOp
from tinygrad.runtime.ops_nvme import NVMEDevice, NVMEAllocator
from extra.k3.store import STRIPE, pieces, part_off

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

# the graph's side: wait holds a queue till the host sets the first word of buf ([ready | ...]) to 1, after deps are written
def wait(buf:UOp, *deps:UOp) -> UOp: return UOp.custom_function("host_wait").call(buf, *deps)

# io_t: the host's state, the drives' queues and command tables, the counts (IO: name, c type, a pointer is an address). bufs: every layer's
# buffers (BUFS) of each gpu: its tag, router scores (rows a token: the router's, the next layer's guess) and the two parts' tables, each
# [ready | slots | weights], k1*T then (k-k1)*T entries: the union of the tokens' first k1 experts, then of the rest, an expert once, its
# weight a token (entry i's at i*T + t, 0: not the token's), padded with slot 0
IO = [*((n, "int64_t") for n in "layers E ngpu k kmin k1 rows tokens chunk nd entries shift slots victims_n nvict pf fill".split()),
      *((n, "float") for n in "tau tauc beta decay".split()), ("bufs", "const uint64_t *"), ("bias", "const float **"),
      ("slot_of", "int32_t *"), ("expert_in", "int64_t *"), ("freq", "float *"), ("flying", "int32_t *"), ("guess", "uint8_t *"),
      ("mask", "uint8_t *"), ("victims", "int64_t *"), ("owner", "int64_t *"),
      ("ptr", "const int64_t *"), ("drive", "const int32_t *"), ("gpu", "const int32_t *"),
      *((n, "const int64_t *") for n in "lba nbytes inpart part_pages".split()),
      *((n, "const uint64_t *") for n in "prp sq db cq cq_db".split()),
      *((n, "int64_t *") for n in "tails db_tails heads phases inflight cids".split()),
      *((n, "int64_t") for n in "refs hits late misses guesses used wasted fills".split()),
      *((n, "double") for n in "t_gpu t_route t_reads t_after slowest".split())]
BUFS = ["tag", "scores", "part1", "part2"]
CTYPES:dict[str, type[ctypes._SimpleCData]] = {"int64_t": ctypes.c_int64, "float": ctypes.c_float, "double": ctypes.c_double}
class IoT(ctypes.Structure): _fields_ = [(n, ctypes.c_void_p if "*" in t else CTYPES[t]) for n, t in IO]

HOST_C = "#include <stdint.h>\ntypedef struct { " + " ".join(f"{t} {n};" for n, t in IO) + " } io_t;\nenum { " + \
  ", ".join(f"B_{b.upper()}" for b in BUFS) + ", NB };\n" + r"""
#include <stdio.h>
#include <time.h>
typedef struct { uint32_t cdw0, nsid; uint64_t rsvd, mptr, prp1, prp2; uint32_t cdw10, cdw11, cdw12, pad[3]; } sqe_t;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#define BUF(kind, g, type) ((type *)b[(kind) * s->ngpu + (g)])

// a timeout's state on stderr: what it waited on, layer li's experts in flight, every queue's commands in flight and unrung
static int64_t stuck(const io_t *s, const char *on, int64_t li, double t0) {
  int64_t fl = 0;
  for (int64_t e = 0; e < s->E; e++) fl += s->flying[li * s->E + e] > 0;
  fprintf(stderr, "timeout waiting on %s at layer %ld after %.1f s: %ld of its experts in flight; in flight, unrung:", on, (long)li,
          now() - t0, (long)fl);
  for (int64_t q = 0; q < 2 * s->nd; q++)
    fprintf(stderr, " %ld,%ld", (long)s->inflight[q], (long)((s->tails[q] - s->db_tails[q] + s->entries) % s->entries));
  fprintf(stderr, "\n");
  return -2;
}

// the reads of expert codes (layer * E + expert) into slots on queue (0: demand, 1: guesses) of their drives, rung later. a command's id
// is the queue's next one not in flight (the drives complete out of order: a slow command can be passed by a whole ring's), its owner its
// code when owned (guesses), else -1, -2 when free: flying counts an owned code's commands till they're reaped. prp1 and prp2 from the prp
// table of the gpu's pool: every page and a pointer to the list of the pages after it. a full queue is reaped and rung till it has room
#define NCID 65536
static int64_t reap(io_t *s);
static void ring(io_t *s);
static void submit(io_t *s, int64_t n, const int64_t *codes, const int64_t *slots, int64_t queue, int64_t owned) {
  for (int64_t i = 0; i < n; i++) for (int64_t r = s->ptr[codes[i]]; r < s->ptr[codes[i] + 1]; r++) {
    int64_t q = s->drive[r] + queue * s->nd, cid = s->cids[q], *own = s->owner + q * NCID, page = slots[i] * s->part_pages[s->gpu[r]] + s->inpart[r];
    int64_t pages = (s->nbytes[r] + 0xfff) >> 12;
    const uint64_t *prp = (const uint64_t *)s->prp[s->gpu[r]];
    for (double t0 = now(); s->inflight[q] == s->entries - 1; ring(s))  // (a failed read or a stuck drive: the layer's wait says so)
      if (reap(s) < 0 || now() - t0 > 60) return;
    while (own[cid] != -2) cid = (cid + 1) % NCID;
    s->cids[q] = (cid + 1) % NCID;
    sqe_t *e = (sqe_t *)s->sq[q] + s->tails[q];
    e->cdw0 = 2 | (uint32_t)cid << 16, e->nsid = 1, e->rsvd = 0, e->mptr = 0;
    e->prp1 = prp[2 * page], e->prp2 = pages == 1 ? 0 : pages == 2 ? prp[2 * page + 2] : prp[2 * page + 1];
    e->cdw10 = (uint32_t)s->lba[r], e->cdw11 = (uint32_t)(s->lba[r] >> 32), e->cdw12 = (uint32_t)((s->nbytes[r] >> s->shift) - 1);
    own[cid] = owned ? codes[i] : -1;
    s->tails[q] = (s->tails[q] + 1) % s->entries, s->inflight[q]++;
    if (owned) s->flying[codes[i]]++;
  }
}

// the doorbells of the demand queues and of the guess queues of drives without demand reads: a drive reads the guesses once its demand
// is in. only the queues that moved (a doorbell write that doesn't move a queue wedges the controller)
static void ring(io_t *s) {
  __sync_synchronize();  // the entries before the doorbells
  for (int64_t q = 0; q < 2 * s->nd; q++) if (s->tails[q] != s->db_tails[q] && (q < s->nd || !s->inflight[q - s->nd]))
    *(volatile uint32_t *)s->db[q] = (uint32_t)(s->db_tails[q] = s->tails[q]);
}

// the new completions of every queue: heads and phases on, the completion doorbells written, inflight and the guesses' flying down. -1
// when a command failed
static int64_t reap(io_t *s) {
  for (int64_t q = 0; q < 2 * s->nd; q++) {
    int64_t h = s->heads[q], ph = s->phases[q], m = 0;
    const volatile uint32_t *c = (const volatile uint32_t *)s->cq[q];
    uint32_t dw3;
    while (((dw3 = c[h * 4 + 3]) >> 16 & 1) == ph) {
      if (dw3 >> 17) return -1;
      int64_t *own = s->owner + q * NCID + (dw3 & 0xffff), o = *own;
      *own = -2;
      if (o >= 0) s->flying[o]--;
      m++;
      if (++h == s->entries) h = 0, ph ^= 1;
    }
    if (m) s->heads[q] = h, s->phases[q] = ph, *(volatile uint32_t *)s->cq_db[q] = (uint32_t)h, s->inflight[q] -= m;
  }
  return 0;
}

// the k experts of the largest sigmoid + bias (+ beta those with a slot), heaviest first, their weights (sigmoids) normalized over the k.
// kept: the top kmin, then those of weight >= tauc that are >= tau or (resident) have a slot; the rest without a slot go on skip (if any,
// ns of them). sig: every expert's sigmoid. returns the number kept
static int64_t route(const io_t *s, const float *logits, const float *bias, const int32_t *slot, int64_t resident, int64_t *sel, float *w,
                     float *sig, int64_t *skip, int64_t *ns) {
  int64_t n = s->E, k = s->k, idx[64], m = 0; float key[64], sv[64];
  for (int64_t i = 0; i < n; i++) sig[i] = 1.0f / (1.0f + __builtin_expf(-logits[i]));
  for (int64_t i = 0; i < n; i++) {
    float v = sig[i], kk = v + bias[i] + (slot[i] >= 0 ? s->beta : 0);
    if (m == k && kk <= key[m - 1]) continue;
    int64_t j = m < k ? m++ : m - 1;
    for (; j > 0 && key[j - 1] < kk; j--) key[j] = key[j - 1], sv[j] = sv[j - 1], idx[j] = idx[j - 1];
    key[j] = kk, sv[j] = v, idx[j] = i;
  }
  float sum = 0;
  for (int64_t j = 0; j < m; j++) sum += sv[j];
  for (int64_t a = 1; a < m; a++) for (int64_t j = a; j > 0 && sv[j - 1] < sv[j]; j--) {  // by weight
    float t = sv[j]; sv[j] = sv[j - 1], sv[j - 1] = t; int64_t u = idx[j]; idx[j] = idx[j - 1], idx[j - 1] = u;
  }
  int64_t c = 0;
  for (int64_t j = 0; j < m; j++) {
    float wj = sv[j] / sum;
    if (j < s->kmin || (wj >= s->tauc && (wj >= s->tau || (resident && slot[idx[j]] >= 0)))) sel[c] = idx[j], w[c++] = wj;
    else if (skip && slot[idx[j]] < 0) skip[(*ns)++] = idx[j];
  }
  return c;
}

// a slot a victim can be: not masked (pinned), not of an expert in flight or guessed
static int64_t free_slot(const io_t *s, int64_t sl) {
  int64_t c = s->expert_in[sl];
  return !s->mask[sl] && !(c >= 0 && (s->flying[c] || s->guess[c]));
}

// the victims of the next misses and guesses: the n least used free slots (an empty one first), least used first
static void pick(io_t *s, int64_t n) {
  float key[1024]; int64_t m = 0;
  for (int64_t sl = 0; sl < s->slots; sl++) {
    int64_t c = s->expert_in[sl];
    if (!free_slot(s, sl)) continue;
    float f = c >= 0 ? s->freq[c] : -1;
    if (m == n && f >= key[m - 1]) continue;
    int64_t j = m < n ? m++ : m - 1;
    for (; j > 0 && key[j - 1] > f; j--) key[j] = key[j - 1], s->victims[j] = s->victims[j - 1];
    key[j] = f, s->victims[j] = sl;
  }
  s->nvict = m;
}

// the first victim still free, taken out of the list, into code's slot (-1: none left)
static int64_t take(io_t *s, int64_t code) {
  for (int64_t i = 0; i < s->nvict; i++) if (free_slot(s, s->victims[i])) {
    int64_t sl = s->victims[i];
    for (s->nvict--; i < s->nvict; i++) s->victims[i] = s->victims[i + 1];
    if (s->expert_in[sl] >= 0) s->slot_of[s->expert_in[sl]] = -1;
    s->expert_in[sl] = code, s->slot_of[code] = (int32_t)sl;
    return sl;
  }
  return -1;
}

// the codes of a chunk's reads not on a queue yet: a ring of at most two layers' (npend and pend_at count a step's)
#define NPEND 4096
static int64_t pend[NPEND], npend, pend_at;

// the completions reaped, the pending reads queued on the demand queues while every drive of the next has room, the queues that moved
// rung. -1 a read failed
static int64_t poll_io(io_t *s) {
  if (reap(s) < 0) return -1;
  for (; pend_at < npend; pend_at++) {
    int64_t c = pend[pend_at % NPEND], sl = s->slot_of[c], r = s->ptr[c];
    while (r < s->ptr[c + 1] && s->inflight[s->drive[r]] < s->entries - 8) r++;
    if (r < s->ptr[c + 1]) break;
    submit(s, 1, &c, &sl, 0, 1);
  }
  ring(s);
  return 0;
}

// (a chunk) every uncached expert of layer lj into a victim, the least used slots but layer lj's and the one's before it, its read pending.
// -3 too few victims
static int64_t queue_layer(io_t *s, int64_t lj) {
  int64_t E = s->E, c0 = (lj ? lj - 1 : 0) * E, c1 = (lj + 1) * E;
  for (int64_t c = c0; c < c1; c++) if (s->slot_of[c] >= 0) s->mask[s->slot_of[c]] = 1;
  pick(s, E);
  for (int64_t c = c0; c < c1; c++) if (s->slot_of[c] >= 0) s->mask[s->slot_of[c]] = 0;
  for (int64_t c = lj * E; c < c1; c++) if (s->slot_of[c] < 0) {
    if (take(s, c) < 0) return -3;
    pend[npend++ % NPEND] = c, s->misses++;
  }
  return 0;
}

// waits for the gpus' tags of the layer li of buffers b, the reads go on. 0, -1 a read failed, -2 a timeout
static int64_t tags(io_t *s, const uint64_t *b, int64_t li, double t0) {
  for (int64_t g = 0; g < s->ngpu; g++) while (*BUF(B_TAG, g, volatile uint32_t) != 1) {
    if (poll_io(s) < 0) return -1;
    if (now() - t0 > 60) return stuck(s, "the gpus", li, t0);
  }
  return 0;
}

// a row of every expert's logit: each gpu's scores hold E / ngpu of them
static void logits_of(const io_t *s, const uint64_t *b, int64_t row, float *out) {
  for (int64_t g = 0, nr = s->E / s->ngpu; g < s->ngpu; g++) for (int64_t i = 0; i < nr; i++) out[g * nr + i] = BUF(B_SCORES, g, float)[row * nr + i];
}

// the guesses of layer lj on a row of the scores of the layer of buffers b: the first n its router would keep that aren't cached, in the
// router's order (sigmoid + bias: a guess high in it is likelier routed), read into victims on the guess queues, held till lj routes
static void prefetch(io_t *s, const uint64_t *b, int64_t lj, int64_t row, int64_t n) {
  float logits[4096], sig[4096], w[64]; int64_t sel[64], codes[64], ms[64], m = 0;
  logits_of(s, b, row, logits);
  const float *bias = s->bias[lj];
  int32_t *so = s->slot_of + lj * s->E;
  int64_t c = route(s, logits, bias, so, 0, sel, w, sig, 0, 0);
  #define KEY(e) (sig[e] + bias[e])
  for (int64_t a = 1; a < c; a++) for (int64_t j = a; j > 0 && KEY(sel[j - 1]) < KEY(sel[j]); j--) {
    int64_t u = sel[j]; sel[j] = sel[j - 1], sel[j - 1] = u;
  }
  for (int64_t j = 0; j < c && m < n; j++) {
    if (so[sel[j]] >= 0) continue;
    if ((ms[m] = take(s, codes[m] = lj * s->E + sel[j])) < 0) break;
    s->guess[codes[m++]] = 1;
  }
  submit(s, m, codes, ms, 1, 1);
  s->guesses += m;
}

// layer li of a step of tokens (each its rows of the gpus' scores): waits for the gpus' tags, routes every token on its row 0 (the union
// of their experts read once), reads the misses into victims, queues the guesses of the next layer (row 1, only while every guess queue is
// under half full: they can't wrap one, the demand's are drained every layer), gives the gpus every token's slots and weights in the two
// parts, each in the order they're ready (cached, cached and in flight, read now): the first k1 a token, those in (one in flight or read
// only when the second can't hold the rest), released while the rest are read; padded with slot 0, weight 0 (the gemvs skip it). then the
// counts, li's guesses settled (used when routed), the decayed use counts and the next layer's victims (but this layer's slots).
// 0, -1 a read failed, -2 a timeout, -3 no victims
static int64_t layer(io_t *s, int64_t li) {
  int64_t E = s->E, ng = s->ngpu, T = s->tokens, nd = s->nd, K1 = s->k1, K2 = s->k - K1;
  const uint64_t *b = s->bufs + li * NB * ng;
  double t0 = now();
  int64_t r = tags(s, b, li, t0);  // (the guesses go on)
  if (r) return r;
  double t1 = now();
  float logits[4096], sig[4096], tw[512];
  int64_t sel[64], pos[4096], usel[512], codes[512], ms[512], skip[512], tj[512], tc[8], p1[8], nu = 0, nm = 0, nh = 0, late = 0, ns = 0, nf = 0;
  uint8_t hit[512];
  int32_t *so = s->slot_of + li * E;
  for (int64_t e = 0; e < E; e++) pos[e] = -1;
  for (int64_t t = 0; t < T; t++) {
    logits_of(s, b, t * s->rows, logits);
    int64_t c = tc[t] = route(s, logits, s->bias[li], so, 1, sel, tw + t * 64, sig, skip, &ns);
    for (int64_t j = 0; j < c; j++) {
      if (pos[sel[j]] < 0) pos[sel[j]] = nu, usel[nu++] = sel[j];
      tj[t * 64 + j] = pos[sel[j]];
    }
  }
  for (int64_t j = 0; j < nu; j++) {
    if ((hit[j] = so[usel[j]] >= 0)) nh++, late += s->flying[li * E + usel[j]] > 0, s->mask[so[usel[j]]] = 1;
    else codes[nm++] = li * E + usel[j];
  }
  int64_t free = 0;
  for (int64_t i = 0; i < s->nvict; i++) free += free_slot(s, s->victims[i]);
  if (free < nm) pick(s, s->victims_n);  // too few: new ones, but this layer's cached
  for (int64_t i = 0; i < nm; i++) if ((ms[i] = take(s, codes[i])) < 0) return -3;
  submit(s, nm, codes, ms, 0, 0);
  ring(s);
  double t2 = now();
  #define FLYING(u) (hit[u] && s->flying[li * E + usel[u]] > 0)
  #define RANK(u) (!hit[u] ? 2 : FLYING(u) ? 1 : 0)
  #define BUSY(u) (hit[u] ? FLYING(u) : reads)
  // each token's experts by readiness (stable), its part 1 those in
  for (int64_t t = 0; t < T; t++) {
    int64_t *o = tj + t * 64, c = tc[t], nht = 0; float *ow = tw + t * 64;
    for (int64_t a = 1; a < c; a++) for (int64_t j = a; j > 0 && RANK(o[j - 1]) > RANK(o[j]); j--) {
      int64_t u = o[j]; o[j] = o[j - 1], o[j - 1] = u; float x = ow[j]; ow[j] = ow[j - 1], ow[j - 1] = x;
    }
    for (int64_t j = 0; j < c; j++) nht += RANK(o[j]) == 0;
    p1[t] = nht < K1 ? nht : K1;
    if (c - p1[t] > K2) p1[t] = c - K2;
  }
  // the parts' unions: an expert in part 1 when it's in a token's first p1, its weight each token's (0: not its)
  int64_t in1[512], n1 = 0, n2 = 0, at[512];
  float wu[512 * 8];
  for (int64_t u = 0; u < nu; u++) in1[u] = 0;
  for (int64_t i = 0; i < nu * T; i++) wu[i] = 0;
  for (int64_t t = 0; t < T; t++) for (int64_t n = 0; n < tc[t]; n++) {
    wu[tj[t * 64 + n] * T + t] = tw[t * 64 + n];
    if (n < p1[t]) in1[tj[t * 64 + n]] = 1;
  }
  for (int64_t u = 0; u < nu; u++) at[u] = in1[u] ? n1++ : n2++;
  for (int64_t g = 0; g < ng; g++) for (int64_t second = 0; second < 2; second++) {
    int64_t cap = (second ? K2 : K1) * T;
    uint32_t *part = BUF(second ? B_PART2 : B_PART1, g, uint32_t);
    for (int64_t i = 0; i < cap; i++) ((int32_t *)part)[1 + i] = 0;
    for (int64_t i = 0; i < cap * T; i++) ((float *)part)[1 + cap + i] = 0;
    for (int64_t u = 0; u < nu; u++) if (in1[u] != second) {
      ((int32_t *)part)[1 + at[u]] = so[usel[u]];
      for (int64_t t = 0; t < T; t++) ((float *)part)[1 + cap + at[u] * T + t] = wu[u * T + t];
    }
  }
  for (int64_t r1 = 0, first = 1;; first = 0) {
    if (poll_io(s) < 0) return -1;
    int64_t reads = 0, busy = 0, busy1 = 0;
    for (int64_t q = 0; q < nd; q++) reads |= s->inflight[q] != 0;
    for (int64_t u = 0; u < nu; u++) busy |= BUSY(u);
    for (int64_t t = 0; t < T; t++) for (int64_t n = 0; n < p1[t]; n++) busy1 |= BUSY(tj[t * 64 + n]);
    if (!r1 && !busy1) {
      __sync_synchronize();  // the slots and weights before the gpus go on
      for (int64_t g = 0; g < ng; g++) *BUF(B_PART1, g, volatile uint32_t) = 1;
      r1 = 1;
    }
    if (first) {  // (after part 1 is out: the gpus work on it) the next layer's guesses, the fills, this layer's hits unpinned
      int64_t room = 1;
      for (int64_t q = nd; q < 2 * nd; q++) room &= s->inflight[q] < s->entries / 2;
      if (room && li + 1 < s->layers) for (int64_t t = 0; t < T; t++) prefetch(s, b, li + 1, t * s->rows + 1, s->pf);
      for (int64_t i = 0; room && i < ns && nf < s->fill; i++) if (pos[skip[i]] == -1) {  // (each once, none another token kept)
        pos[skip[i]] = -2;
        if ((ms[nm + nf] = take(s, codes[nm + nf] = li * E + skip[i])) < 0) break;
        nf++;
      }
      submit(s, nf, codes + nm, ms + nm, 1, 1);
      ring(s);
      for (int64_t j = 0; j < nu; j++) if (hit[j]) s->mask[so[usel[j]]] = 0;
    }
    if (!busy) break;
    if (now() - t0 > 60) return stuck(s, "reads", li, t0);
  }
  __sync_synchronize();
  for (int64_t g = 0; g < ng; g++) *BUF(B_PART1, g, volatile uint32_t) = 1, *BUF(B_PART2, g, volatile uint32_t) = 1;
  double t3 = now();
  s->refs += nu, s->hits += nh - late, s->late += late, s->misses += nm, s->fills += nf;
  for (int64_t e = 0; e < E; e++) if (s->guess[li * E + e]) {
    s->guess[li * E + e] = 0;
    if (pos[e] >= 0) s->used++; else s->wasted++;
  }
  float *f = s->freq + li * E;
  for (int64_t e = 0; e < E; e++) f[e] *= s->decay;
  for (int64_t j = 0; j < nu; j++) f[usel[j]] += 1;
  for (int64_t i = 0; i < nf; i++) f[codes[nm + i] - li * E] += 1;  // (a fill counts as routed: it was)
  for (int64_t j = 0; j < nu; j++) s->mask[so[usel[j]]] = 1;
  pick(s, s->victims_n);
  for (int64_t j = 0; j < nu; j++) s->mask[so[usel[j]]] = 0;
  s->t_gpu += t1 - t0, s->t_route += t2 - t1, s->t_reads += t3 - t2, s->t_after += now() - t3;
  return 0;
}

// layer li of a prompt's chunk (many tokens): waits for the gpus' tags, queues the next layer's every uncached expert (a chunk's tokens route
// to nearly all of them), routes every token exactly (its top k: no tau, no cached bonus), gives the gpus part 1 an entry a token's expert
// (j*T + t: token t's j-th, its slot and weight) once li's reads are in. then the counts and the decayed use counts (a use a token).
// 0, -1 a read failed, -2 a timeout, -3 no victims
static int64_t chunk(io_t *s, int64_t li) {
  int64_t E = s->E, ng = s->ngpu, T = s->tokens, K = s->k, sel[64], cnt[4096];
  const uint64_t *b = s->bufs + li * NB * ng;
  double t0 = now();
  int64_t r = tags(s, b, li, t0), lend = npend;  // (li's reads: those pending before li + 1's)
  if (r) return r;
  if (li + 1 < s->layers && (r = queue_layer(s, li + 1))) return r;
  if (poll_io(s) < 0) return -1;
  double t1 = now();
  io_t exact = *s;
  exact.tau = exact.tauc = exact.beta = 0, exact.kmin = K;
  float logits[4096], sig[4096], w[64];
  int32_t *so = s->slot_of + li * E, *part = BUF(B_PART1, 0, int32_t);
  for (int64_t e = 0; e < E; e++) cnt[e] = 0;
  for (int64_t t = 0; t < T; t++) {
    logits_of(s, b, t * s->rows, logits);
    route(&exact, logits, s->bias[li], so, 0, sel, w, sig, 0, 0);
    for (int64_t j = 0; j < K; j++) cnt[sel[j]]++, part[1 + j * T + t] = so[sel[j]], ((float *)part)[1 + (K + j) * T + t] = w[j];
  }
  for (int64_t g = 1; g < ng; g++) for (int64_t i = 1; i <= 2 * K * T; i++) BUF(B_PART1, g, int32_t)[i] = part[i];
  double t2 = now();
  for (int64_t busy = 1; busy;) {
    if (poll_io(s) < 0) return -1;
    busy = pend_at < lend;
    for (int64_t e = 0; e < E && !busy; e++) busy = s->flying[li * E + e] > 0;
    if (now() - t2 > 60) return stuck(s, "reads", li, t2);
  }
  __sync_synchronize();
  for (int64_t g = 0; g < ng; g++) *BUF(B_PART1, g, volatile uint32_t) = 1;
  float *f = s->freq + li * E;
  for (int64_t e = 0; e < E; e++) f[e] = f[e] * s->decay + cnt[e], s->refs += cnt[e] > 0;
  s->t_gpu += t1 - t0, s->t_route += t2 - t1, s->t_reads += now() - t2;
  return 0;
}

// n codes read into slots on the demand queues, waited for. 0, -1 a read failed, -2 a timeout
int64_t read_codes(io_t *s, int64_t n, const int64_t *codes, const int64_t *slots) {
  double t0 = now();
  submit(s, n, codes, slots, 0, 0);
  for (int64_t busy = 1; busy;) {
    if (reap(s) < 0) return -1;
    ring(s);
    busy = 0;
    for (int64_t q = 0; q < s->nd; q++) busy |= s->inflight[q] != 0;
    if (now() - t0 > 60) return stuck(s, "reads", 0, t0);
  }
  return 0;
}

// every layer's ready words set to value, with 0 the tags cleared too (before a step's graph runs)
void flags(io_t *s, uint32_t value) {
  for (int64_t li = 0; li < s->layers; li++) for (int64_t g = 0; g < s->ngpu; g++) {
    const uint64_t *b = s->bufs + li * NB * s->ngpu;
    *BUF(B_PART1, g, volatile uint32_t) = value, *BUF(B_PART2, g, volatile uint32_t) = value;
    if (!value) *BUF(B_TAG, g, volatile uint32_t) = 0;
  }
}

// every layer of a step (a chunk: layer 0's reads queued first): the host's whole part of it while the gpus run the graph. 0, else the
// layer's error. slowest: its slowest layer
int64_t step_io(io_t *s) {
  int64_t r = 0;
  s->slowest = 0, npend = pend_at = 0;
  if (s->chunk && (r = queue_layer(s, 0))) return r;
  for (int64_t li = 0; li < s->layers; li++) {
    double t = now();
    r = s->chunk ? chunk(s, li) : layer(s, li);
    if (r) return r;
    if (now() - t > s->slowest) s->slowest = now() - t;
  }
  return 0;
}
"""

@functools.cache
def host_c() -> ctypes.CDLL:
  so = os.path.join(cache_dir, f"k3_host_{hashlib.md5(HOST_C.encode()).hexdigest()}.so")  # (the user's, not a shared tmp)
  if not os.path.exists(so):  # (the sigmoids in vector expfs: libmvec)
    os.makedirs(cache_dir, exist_ok=True)
    subprocess.run(["clang", "-O2", "-march=native", "-fno-math-errno", "-fveclib=libmvec", "-shared", "-fPIC", "-x", "c", "-", "-o",
                    tmp:=f"{so}.{os.getpid()}", "-lmvec"], input=HOST_C.encode(), check=True)
    os.replace(tmp, so)
  lib = ctypes.CDLL(so)
  lib.step_io.restype, lib.step_io.argtypes = ctypes.c_int64, [ctypes.POINTER(IoT)]
  lib.read_codes.restype, lib.read_codes.argtypes = ctypes.c_int64, [ctypes.POINTER(IoT), ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
  lib.flags.argtypes = [ctypes.POINTER(IoT), ctypes.c_uint32]
  return lib

def reads(m:dict, disks:list[list[str]], pools:list[Buffer], ncodes:int) -> tuple[dict[str, np.ndarray], dict[str, int]]:
  """step_io's reads of every expert code (layer * E + expert) into the gpus' pools (a part a slot): two new host-owned queues on every
  drive of the disks (queue j of drive k at j * nd + k: the demand's, then the guesses'), the drive commands of each code (a piece of a
  stripe each, in code order) and the prp table of each pool, the one master maps it with on its disk's first drive (the drives share an
  address space). the tables and the io_t scalars"""
  nvs = [cast(NVMEDevice, Device[d]) for disk in disks for d in disk]
  qs, (nd, shift) = [nv.nvme.new_queue() for _ in range(2) for nv in nvs], (len(nvs), nvs[0].nvme.lba_shift)
  assert all(nv.nvme.lba_shift == shift and nv.nvme.max_xfer >= STRIPE for nv in nvs), "the drives read a stripe a command, of one block size"
  rows, ptr = [], [0]
  for code in range(ncodes):
    for g, (disk, size) in enumerate(zip(disks, m["part_bytes"])):
      done = 0
      for d, o, n in pieces(part_off(m, g, code), size, len(disk)):
        rows.append((g * len(disk) + d, g, o >> shift, n, done >> 12))
        done += n
    ptr.append(len(rows))
  drive, gpu, lba, nbytes, inpart = (np.array(c, np.int64) for c in zip(*rows))
  prps = [cast(NVMEAllocator, Device[disk[0]].allocator).prp_tables[pool.get_buf(disk[0])].get_storage().host for disk, pool in zip(disks, pools)]
  def addrs(xs) -> np.ndarray: return np.array([x.addr for x in xs], np.uint64)
  tables:dict[str, np.ndarray] = {
    "ptr": np.array(ptr, np.int64), "drive": drive.astype(np.int32), "gpu": gpu.astype(np.int32), "lba": lba, "nbytes": nbytes, "inpart": inpart,
    "part_pages": np.array(m["part_bytes"], np.int64) >> 12, "prp": addrs(prps), "sq": addrs(q.sq for q in qs), "cq": addrs(q.cq for q in qs),
    "db": np.array([q.dev.regs.addr + q.sq_db for q in qs], np.uint64), "cq_db": np.array([q.dev.regs.addr + q.cq_db for q in qs], np.uint64),
    **{k: np.zeros(2 * nd, np.int64) for k in ("tails", "db_tails", "heads", "inflight", "cids")}, "phases": np.ones(2 * nd, np.int64),
    "owner": np.full(2 * nd * 65536, -2, np.int64)}
  return tables, {"nd": nd, "entries": qs[0].entries, "shift": shift}
