# EVAL=corpus.jsonl:nseq:ntok:first (engine.py): production completions teacher forced through an engine, each from position 0. per output
# token the top-1 agreement with production and KL(production's top 5 and the rest || ours), per sequence and so far
import json, sys, time
from typing import Callable
import numpy as np
from tinygrad import Tensor

def score(lg:np.ndarray, top:list) -> tuple[float, float]:
  """logits lg against production's top 5 (logprob, token): top-1 agreement, KL(production's top 5 and the rest || ours)"""
  lp = lg - lg.max() - np.log(np.exp(lg - lg.max()).sum())
  p, q = np.exp([x[0] for x in top]), lp[[x[1] for x in top]]
  rest_p, rest_q = max(1 - p.sum(), 1e-12), max(1 - np.exp(q).sum(), 1e-12)
  return float(lp.argmax() == top[0][1]), float((p * (np.log(p) - q)).sum() + rest_p * (np.log(rest_p) - np.log(rest_q)))

def evaluate(eng, spec:str, late:Callable[[], bool]):
  path, *nums = spec.split(":")
  nseq, ntok, first = [int(x) for x in nums] + [16, 256, 0][len(nums):]
  stats:list[tuple[float, float]] = []
  for r in [json.loads(line) for line in open(path)][first:first + nseq]:
    ids, outs, n0, c0 = r["prompt_ids"], r["output_ids"][:ntok], len(stats), eng.counters()
    seq, T, at = ids + outs[:-1], eng.tokens, eng.prefill(ids, 0)  # (the prompt in chunks, then steps of T tokens: a verify step's numerics)
    while at < len(seq):
      if late(): sys.exit(f"deadline: {len(stats)} tokens, top-1 agree and KL {np.mean(stats, 0) if stats else '-'}")
      n = T if len(seq) - at >= T else 1
      eng.step(seq[at:at + n], at)
      for pos, lg in enumerate(eng.logits.numpy().astype(np.float64), at):
        if pos >= len(ids) - 1: stats.append(score(lg, r["output_top"][pos - len(ids) + 1]))
      at += n
    (a1, k1), (a, k) = np.mean(stats[n0:], 0), np.mean(stats, 0)
    print(f"{r['task_id']}: top-1 agree {a1:.1%}, KL {k1:.4f}; all {len(stats)} tokens: top-1 agree {a:.1%}, KL {k:.4f}")
    eng.report(c0, len(ids) + len(outs) - 2)

def window_evaluate(eng, spec:str, late:Callable[[], bool], rows:int=128, size:int=0) -> list[tuple[float, float]]:
  """WEVAL=corpus.jsonl:nseq:ntok:first: evaluate's completions through windows (of up to size tokens, else the engine's), every position's logits
  from the window's last streams (a pass of the sequence, no steps), rows at a time"""
  path, *nums = spec.split(":")
  nseq, ntok, first = [int(x) for x in nums] + [16, 256, 0][len(nums):]
  m, stats = eng.model, []
  for r in [json.loads(line) for line in open(path)][first:first + nseq]:
    if late(): sys.exit(f"deadline: {len(stats)} tokens, top-1 agree and KL {np.mean(stats, 0) if stats else '-'}")
    ids, outs, n0 = r["prompt_ids"], r["output_ids"][:ntok], len(stats)
    seq = ids + outs[:-1]
    for at in range(0, len(seq), size or eng.W):
      toks = seq[at:at + (size or eng.W)]
      eng.window(toks, at, (seq[at + 1:at + 1 + len(toks)] + [-1])[:len(toks)])
      for r0 in range(0, len(toks), rows):
        x = m.output_res_score.mix(Tensor(eng.win["bank"].uop)[:, r0:r0 + rows])
        for pos, lg in enumerate(m.output(m.output_norm(x))[0].float().numpy()[:len(toks) - r0].astype(np.float64), at + r0):
          if pos >= len(ids) - 1: stats.append(score(lg, r["output_top"][pos - len(ids) + 1]))
    (a1, k1), (a, k) = np.mean(stats[n0:], 0), np.mean(stats, 0)
    print(f"{r['task_id']}: top-1 agree {a1:.1%}, KL {k1:.4f}; all {len(stats)} tokens: top-1 agree {a:.1%}, KL {k:.4f}", flush=True)
  return stats

def bench(eng, spec:str, late:Callable[[], bool]):
  """BENCH=corpus.jsonl:nseq:ntok:first: verify steps of [the next, the draft's guess] over production completions teacher forced (the text
  fixed: the steps' time and the guesses right don't move with what the engine would have written)"""
  path, *nums = spec.split(":")
  nseq, ntok, first = [int(x) for x in nums] + [16, 256, 0][len(nums):]
  steps, right, t_steps = 0, 0, 0.0
  for r in [json.loads(line) for line in open(path)][first:first + nseq]:
    toks, n = r["prompt_ids"] + r["output_ids"][:ntok], len(r["prompt_ids"])
    for pos in range(eng.prefill(toks[:n], 0), n): _, (g,) = eng.step(toks[pos:pos + 1], pos, toks[pos + 1:pos + 2])
    pos = n
    while pos + 2 < len(toks) and not late():
      st = time.perf_counter()
      _, guesses = eng.step([toks[pos], g], pos, toks[pos + 1:pos + 3])
      t_steps, steps, ok = t_steps + time.perf_counter() - st, steps + 1, g == toks[pos + 1]
      right += ok
      if not ok: eng.model.rollback(1)
      pos, g = pos + 1 + ok, guesses[ok]
    print(f"{r['task_id']}: {steps} verify steps, {t_steps / steps * 1e3:.1f} ms a step, {right / steps:.1%} of the guesses right, "
          f"{(steps + right) / t_steps:.2f} tok/s", flush=True)
