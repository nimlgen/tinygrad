"""Reorders a captured llama training graph before memory planning (the TinyJit sched_fixer hook), after gptoss_sched_fixer."""
import collections
from tinygrad.helpers import getenv
from tinygrad.uop.ops import UOp, Ops, KernelInfo

def _name(call:UOp) -> str:
  if call.op is not Ops.CALL: return str(call.op)
  body = call.src[0]
  if body.op is Ops.PROGRAM: return body.src[0].arg.name
  return body.arg.name if isinstance(body.arg, KernelInfo) else str(body.op)

def inspect(linear:UOp) -> None:
  kinds = collections.Counter((c.op, c.src[0].op if c.op is Ops.CALL else None) for c in linear.src)
  print(f"llama_sched_fixer: {len(linear.src)} calls", {f"{k[0]}/{k[1]}": v for k, v in kinds.most_common(8)}, flush=True)
  for c in [c for c in linear.src if c.op is not Ops.CALL or c.src[0].op is Ops.LINEAR][:3]:
    print("  nested:", c.op, c.src[0].op if c.src else None, len(c.src[0].src) if c.src and c.src[0].op is Ops.LINEAR else "", flush=True)

def install():
  from tinygrad.engine import jit
  def sched_fixer(linear:UOp, held_bufs:set[UOp]) -> UOp:
    if getenv("LLAMA_SCHED_INSPECT", 0) and len(linear.src) > 100: inspect(linear)
    return linear
  jit.sched_fixer = sched_fixer
