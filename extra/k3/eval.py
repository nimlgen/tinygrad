# EVAL=corpus.jsonl:nseq:ntok:first (engine.py): production completions teacher forced through an engine, each from position 0. per output
# token the top-1 agreement with production and KL(production's top 5 and the rest || ours), per sequence and so far
import json, sys
from typing import Callable
import numpy as np

def evaluate(eng, spec:str, late:Callable[[], bool]):
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
      lg = eng.logits.numpy()[0].astype(np.float64)
      lp, top = lg - lg.max() - np.log(np.exp(lg - lg.max()).sum()), r["output_top"][pos - len(ids) + 1]
      p, q = np.exp([x[0] for x in top]), lp[[x[1] for x in top]]
      rest_p, rest_q = max(1 - p.sum(), 1e-12), max(1 - np.exp(q).sum(), 1e-12)
      stats.append((float(lp.argmax() == top[0][1]), float((p * (np.log(p) - q)).sum() + rest_p * (np.log(rest_p) - np.log(rest_q)))))
    (a1, k1), (a, k) = np.mean(stats[n0:], 0), np.mean(stats, 0)
    print(f"{r['task_id']}: top-1 agree {a1:.1%}, KL {k1:.4f}; all {len(stats)} tokens: top-1 agree {a:.1%}, KL {k:.4f}")
    eng.report(c0, len(ids) + len(outs) - 2)
