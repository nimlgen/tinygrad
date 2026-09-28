"""Reorders a captured gpt-oss step before memory planning. `plan` is the policy; the cpu emulator scores the same function."""
import pickle
from tinygrad.helpers import getenv
from tinygrad.uop.ops import UOp, Ops, KernelInfo, GroupOp

def _name(call:UOp) -> str:
  body = call.src[0]
  if body.op is Ops.PROGRAM: return body.src[0].arg.name
  return body.arg.name if isinstance(body.arg, KernelInfo) else ""

def _bufs(u:UOp) -> list[UOp]:
  if u.op is Ops.BUFFER: return [u]
  return [b for s in u.src for b in _bufs(s)] if u.op in {Ops.MSELECT, Ops.MSTACK} else []

def _devs(c:UOp) -> tuple[str, ...]:
  return tuple(dict.fromkeys(d for s in c.src[1:] for d in (s.device if isinstance(s.device, tuple) else (s.device,)) if d is not None))

def _lanes(u:UOp, devs:tuple[str, ...], lane:int|None=None) -> list[tuple[UOp, str]]:
  # (storage, device) pairs a call argument touches, through any view: each device's shard of a multi-device buffer is its own
  # allocation. anything unrecognized raises: a dependency nobody sees is a wrong answer
  if u.op in {Ops.BUFFER, Ops.PARAM, Ops.ALLOC}:
    bdevs = u.device if isinstance(u.device, tuple) else (u.device,)
    if lane is not None: return [(u, bdevs[lane] if len(bdevs) > 1 else bdevs[0])]
    return [(u, d) for d in bdevs if len(bdevs) == 1 or d in devs]
  if u.op is Ops.MSELECT: return _lanes(u.src[0], devs, u.arg)
  if u.op is Ops.MSTACK: return [x for s in u.src for x in _lanes(s, devs, lane)]
  if u.op in GroupOp.Movement or u.op in {Ops.BITCAST, Ops.AFTER, Ops.UNSHARD, Ops.SHRINK, Ops.CAST}: return _lanes(u.src[0], devs, lane)
  raise RuntimeError(f"gptoss_sched_fixer: can't resolve call argument {u.op}")

def _access(c:UOp, devs:tuple[str, ...]|None=None) -> tuple[list[tuple[UOp, str]], list[tuple[UOp, str]]]:
  # (writes, reads): a copy reads its source and writes its destination, a kernel reads and writes everything it touches
  if devs is None: devs = _devs(c)
  args = [s for s in c.src[1:] if not s.is_bound_var]
  if c.src[0].op is Ops.STORE: return _lanes(args[0], devs), _lanes(args[1], devs)
  return [x for s in args for x in _lanes(s, devs)], []

def _preds(calls:list[UOp], raw_only:bool=False, acc:list|None=None) -> list[set[int]]:
  # raw_only: data flow alone (a read after the write of its lane), no ordering edges from reusing a lane
  last:dict = {}
  readers:dict = {}
  preds:list[set[int]] = []
  for i, (c, (writes, reads)) in enumerate(zip(calls, acc or map(_access, calls))):
    if raw_only: preds.append({last[b] for b in (reads if c.src[0].op is Ops.STORE else writes + reads) if b in last}); continue
    preds.append({last[b] for b in writes + reads if b in last} | {r for b in writes for r in readers.get(b, ()) if r != i})
    for b in reads: readers.setdefault(b, []).append(i)
    for b in writes: last[b], readers[b] = i, []
  return preds

def _raw_preds(calls:list[UOp], acc:list|None=None) -> list[set[int]]:
  # data flow alone: the last writer of every lane a call reads (a kernel reads everything it touches, a copy its source)
  last:dict = {}
  preds:list[set[int]] = []
  for i, (c, (writes, reads)) in enumerate(zip(calls, acc or map(_access, calls))):
    preds.append({last[b] for b in (reads if c.src[0].op is Ops.STORE else writes) if b in last})
    for b in writes: last[b] = i
  return preds

def regions(names:list[str]) -> list[range]:
  # an update's optimizer up to the next forward's embedding
  fwds = [i for i, n in enumerate(names) if n.startswith("gptoss_embedding_fwd")]
  adams = [i for i, n in enumerate(names) if n.startswith("fused_adam")]
  return [range(min(a for a in adams if prev < a < f), f) for prev, f in zip([-1] + fwds, fwds) if any(prev < a < f for a in adams)]

def plan(names:list[str], is_copy:list[bool], preds:list[set[int]]) -> list[int]:
  """In each optimizer-to-forward region every call goes out in the order the forward needs it (optimizer kernels and gather
  copies first for the embedding, then layer 0, ...), and kernels that only prepare gathered weights move right before their
  consumer. Everything outside the regions keeps its order."""
  n = len(names)
  children:list[list[int]] = [[] for _ in range(n)]
  for i, ps in enumerate(preds):
    for p in ps: children[p].append(i)
  key:list[tuple[int, int, int, int]] = [(i, 0, 0, i) for i in range(n)]
  for reg in regions(names):
    inside = set(reg)
    need:dict[int, int] = {}
    for i in reversed(reg): need[i] = min([need[c] if c in inside else c for c in children[i]], default=n)
    fed:set[int] = set()
    for i in reg:
      if not is_copy[i] and any(p in inside and (is_copy[p] or p in fed) for p in preds[i]): fed.add(i)
    assembly = {i for i in fed if not names[i].startswith("fused_adam") and not any(is_copy[c] for c in children[i])}
    rest = [i for i in reg if i not in assembly]
    waiting = {i: sum(p in inside and p not in assembly for p in preds[i]) for i in rest}
    ready, rank = [i for i in rest if waiting[i] == 0], 0
    while ready:
      i = min(ready, key=lambda j: (need[j], j))
      ready.remove(i)
      key[i], rank = (reg.start, rank, 0, i), rank + 1
      for c in children[i]:
        if c in waiting:
          waiting[c] -= 1
          if waiting[c] == 0: ready.append(c)
    assert rank == len(rest), "gptoss_sched_fixer: region is not a dag"
    for i in sorted(assembly, reverse=True):
      key[i] = (lambda k: (k[0], k[1], k[2] - 1, i))(min(key[c] for c in children[i])) if children[i] else (reg.start, rank, 0, i)
  order = sorted(range(n), key=lambda i: key[i])
  pos = {c: p for p, c in enumerate(order)}
  assert all(pos[p] < pos[i] for i, ps in enumerate(preds) for p in ps), "gptoss_sched_fixer broke a dependency"
  return order

def plan_chain_delay(names:list[str], is_copy:list[bool], preds:list[set[int]], k:int, lanes:list[int]|None=None,
                     raw_preds:list[set[int]]|None=None) -> list[int]:
  """The reduce of a gradient hops gpu to gpu: add, copy, add on the next gpu. Every gpu runs the same program, so hop i sits at the
  same stream position everywhere and each hop exposes its transfer. A kernel goes k positions later per copy hop on its chain
  (cumulative), so when a gpu reaches it the data of the previous hop has landed. Host calls and everything after the gradient
  norm keep their place: the tail has nothing to overlap with."""
  n = len(names)
  norm = next((i for i, nm in enumerate(names) if nm.startswith("grad_norm")), n)
  # the chain: single-lane kernels (the adds of a reduce) and copies. a replicated kernel is the main compute: depth 0
  depth = [0] * n
  for i in range(n):
    if is_copy[i]: depth[i] = max([depth[p] for p in preds[i]], default=0) # a copy carries its producer's depth
    elif i < norm and (lanes is None or lanes[i] == 1): depth[i] = max([depth[p] + is_copy[p] for p in preds[i]], default=0)
  import heapq
  children:list[list[int]] = [[] for _ in range(n)]
  for i, ps in enumerate(preds):
    for p in ps: children[p].append(i)
  raw_children:list[list[int]] = [[] for _ in range(n)]
  for i, ps in enumerate(raw_preds if raw_preds is not None else preds):
    for p in ps: raw_children[p].append(i)
  # only slack chains move: a reduce whose result feeds a replicated kernel before the norm is on the backward's own path
  on_path = [False] * n
  for i in reversed(range(norm)):
    on_path[i] = (lanes is not None and lanes[i] > 1 and not is_copy[i]) or any(on_path[c] for c in raw_children[i] if c < norm)
  prio = [i + (depth[i] * k if not is_copy[i] and i < norm and depth[i] and not on_path[i] else 0) for i in range(n)]
  waiting, out = [len(p) for p in preds], []
  ready = [(prio[i], i) for i in range(n) if waiting[i] == 0]
  heapq.heapify(ready)
  while ready:
    _, i = heapq.heappop(ready); out.append(i)
    for c in children[i]:
      waiting[c] -= 1
      if waiting[c] == 0: heapq.heappush(ready, (prio[c], c))
  assert len(out) == n, "gptoss_sched_fixer: not a dag"
  return out

def plan_cone_hoist(names:list[str], is_copy:list[bool], preds:list[set[int]], ntargets:int=1, data_preds:list[set[int]]|None=None) -> list[int]:
  """The first forward kernel of the next update waits for the vocab table's gather, queued behind every other layer's gather in the
  copy fifos. In each optimizer-to-forward region the dependency cone of the region's first `ntargets` forward kernels moves to the
  front (cone by cone, original order inside); everything else keeps its order."""
  n, dp = len(names), data_preds if data_preds is not None else preds
  key = [(i, 0) for i in range(n)]
  for reg in regions(names):
    inside = set(reg)
    targets = [reg.stop + t for t in range(ntargets)]
    rank = {}
    for r, t in enumerate(targets): # backward slice of the target inside the region, data dependencies only
      stack = [p for p in dp[t] if p in inside]
      while stack:
        i = stack.pop()
        if i in rank: continue
        rank[i] = r
        stack += [p for p in dp[i] if p in inside and p not in rank]
    for i in reg: key[i] = (reg.start, rank.get(i, ntargets) * n + i)
  order = sorted(range(n), key=lambda i: key[i])
  pos = {c: p for p, c in enumerate(order)}
  assert all(pos[p] < pos[i] for i, ps in enumerate(preds) for p in ps), "gptoss_sched_fixer: cone hoist broke a dependency"
  return order

def plan_need(names:list[str], is_copy:list[bool], preds:list[set[int]], data_preds:list[set[int]]) -> list[int]:
  """Optimizer-to-forward regions in consumption order: every call ranked by the first post-region kernel that needs it (data
  dependencies, transitive), stable inside a rank, so an optimizer kernel and its gather copies stay together and the copy fifos
  fill layer by layer. Nothing else moves."""
  n = len(names)
  children:list[list[int]] = [[] for _ in range(n)]
  for i, ps in enumerate(data_preds):
    for p in ps: children[p].append(i)
  key = [(i, 0) for i in range(n)]
  for reg in regions(names):
    inside, need = set(reg), {}
    for i in reversed(reg): need[i] = min([need[c] if c in inside else c for c in children[i]], default=n)
    for i in reg: key[i] = (reg.start, need[i] * n + i)
  order = sorted(range(n), key=lambda i: key[i])
  pos = {c: p for p, c in enumerate(order)}
  assert all(pos[p] < pos[i] for i, ps in enumerate(preds) for p in ps), "gptoss_sched_fixer: need order broke a dependency"
  return order

def first_need_cone(reg:range, k:int, data_preds:list[set[int]]) -> set[int]:
  cone, stack = set(), list(range(reg.stop, reg.stop + k))
  while stack:
    for p in data_preds[stack.pop()]:
      if p in reg and p not in cone: cone.add(p); stack.append(p)
  return cone

def start_region(names:list[str], data_preds:list[set[int]], is_copy:list[bool], hosts:set[int]) -> list[range]:
  # GPTOSS_START_REGION=1: with GPTOSS_DEFER_EXPERT_GATHER the graph starts with the previous graph's expert gathers ahead of the first
  # forward. They are scheduled like a mid-graph region, from the first device copy that depends on nothing (after the token prep)
  fwds = [i for i, n in enumerate(names) if n.startswith("gptoss_embedding_fwd")]
  if not getenv("GPTOSS_START_REGION", 0) or not fwds or not any(n.startswith("fused_adam") for n in names[fwds[0]:]): return []
  s = next((i for i in range(fwds[0]) if is_copy[i] and not data_preds[i] and i not in hosts), None)
  return [] if s is None else [range(s, fwds[0])]

def lmhead_late(names:list[str], data_preds:list[set[int]], children:list[list[int]], is_copy:list[bool], prio:dict[int, float], delay:int):
  """GPTOSS_LMHEAD_LATE=k: the LM head is gathered inside each forward (deferred_lmhead_forward_release). Its nic-forward staging kernels
  wait for the far node's shard and its assembly waits for every shard, both early in the forward's compute stream, where the
  in-order queue idles until the transfers land. Staging kernels go k calls later, assembly kernels right before their consumer."""
  n = len(names)
  for rel in [i for i, nm in enumerate(names) if nm.startswith("deferred_lmhead_forward_release")]:
    stop = next((i for i in range(rel, n) if names[i].startswith("custom_fa_backward")), n)
    # the gather: calls fed only by the release and by each other (anything also fed by the forward's activations is a consumer)
    chain, frontier = {rel}, [rel]
    while frontier:
      nxt = []
      for x in frontier:
        for c in children[x]:
          if c < stop and c not in chain and all(p in chain or p < rel for p in data_preds[c]): chain.add(c); nxt.append(c)
      frontier = nxt
    need:dict[int, int] = {}
    for i in sorted(chain, reverse=True): need[i] = min([need[c] if c in chain else c for c in children[i]], default=stop)
    for i in sorted(chain):
      if i == rel or is_copy[i] or not any(is_copy[p] for p in data_preds[i]): continue
      feeds_copy = any(is_copy[c] for c in children[i])
      target = min(i + delay, need[i]) if feeds_copy else need[i]
      prio[i] = prio.get(target, float(target)) - 0.5 / n + i / n / n

def jit_optimizer(reg:range, names:list[str], data_preds:list[set[int]], raw_preds:list[set[int]], children:list[list[int]],
                  is_copy:list[bool], need:dict[int, int], prio:dict[int, float], lead:int):
  """GPTOSS_JIT_OPT=lead: the optimizer region runs on the next forward's own axis instead of ahead of it. Every region call goes
  `lead` calls before its first forward consumer (Adam, the nic send staging and the gather copies of layer k go out while the forward
  runs ~lead/42 layers earlier), the assembly kernels (fed by the gather) right before their consumer, the embedding's cone first.
  The copy fifos then deliver layer 0 first and the whole gather hides under the forward."""
  n, inside = len(names), set(reg)
  fwd_end = next((i for i in range(reg.stop, n) if names[i].startswith("custom_fa_backward")), n)
  def axis(f:int) -> float: return reg.start + 0.5 + (min(max(f, reg.stop), fwd_end) - reg.stop) / n
  fed:set[int] = set()
  for i in reg:
    if not is_copy[i] and any(p in inside and (is_copy[p] or p in fed) for p in raw_preds[i]): fed.add(i)
  assembly = {i for i in fed if not names[i].startswith("fused_adam") and not any(is_copy[c] for c in children[i])}
  for i in range(reg.stop, fwd_end): prio[i] = axis(i)
  for i in reg:
    if i not in assembly: prio[i] = axis(need[i] - lead) - 0.5 / n + i / n / n
  for i in sorted(assembly, reverse=True): prio[i] = min([prio[c] for c in children[i]], default=axis(fwd_end)) - 0.5 / n / n
  for i in first_need_cone(reg, 1, data_preds): prio[i] = reg.start - 2 + i / n

def plan_nic_need(names:list[str], preds:list[set[int]], data_preds:list[set[int]], cross:list[bool], is_copy:list[bool],
                  hosts:set[int]|None=None, raw_preds:list[set[int]]|None=None) -> list[int]:
  """Only the cross-node sends of an optimizer-to-forward region move: each goes right before the first post-region kernel that
  needs it. The xgmi fifos first drain the local shards in their order, the nic (the slow leg) then delivers in consumption order
  against the running forward, and no xgmi forward of a shard that arrives over the nic sits ahead of local shards."""
  n = len(names)
  children:list[list[int]] = [[] for _ in range(n)]
  for i, ps in enumerate(data_preds):
    for p in ps: children[p].append(i)
  prio = {i: float(i) for i in range(n)}
  for reg in start_region(names, data_preds, is_copy, hosts or set()) + regions(names):
    inside, need = set(reg), {}
    for i in reversed(reg): need[i] = min([need[c] if c in inside else c for c in children[i]], default=n)
    if (lead:=getenv("GPTOSS_JIT_OPT", 0)) and raw_preds is not None:
      jit_optimizer(reg, names, data_preds, raw_preds, children, is_copy, need, prio, lead)
      continue
    for i in reg:
      if cross[i]: prio[i] = need[i] - 0.5 + i / n
    # GPTOSS_FIRST_NEED=k: everything the forward's first k calls wait on (the vocab gather) leads the region, so its xgmi
    # forwards don't queue behind the bulk expert gathers in the copy fifos
    for i in first_need_cone(reg, getenv("GPTOSS_FIRST_NEED", 0), data_preds): prio[i] = reg.start - 1 + i / n
    # GPTOSS_FWD_INTERLEAVE=1: the next forward's kernels run as soon as their weights are assembled instead of queueing behind
    # every other gather assembly on the compute queue (the fifos deliver the last layers first, so the wait was the whole gather)
    if getenv("GPTOSS_FWD_INTERLEAVE", 0):
      fwd_end = next((i for i in range(reg.stop, n) if names[i].startswith("custom_fa_backward")), n)
      for i in range(reg.stop, fwd_end): prio[i] = reg.start + 0.5 + (i - reg.stop) / n
  if (delay:=getenv("GPTOSS_LMHEAD_LATE", 0)): lmhead_late(names, data_preds, children, is_copy, prio, delay)
  # GPTOSS_DEFER_ADDS=k: in the backward, a kernel reading a copy moves up to k calls later (not past its first consumer), so the
  # compute queue runs the next layer's gemms while the peers' reduce copies are still in flight
  if (k:=getenv("GPTOSS_DEFER_ADDS", 0)):
    fwds, adams = [i for i, nm in enumerate(names) if nm.startswith("gptoss_embedding_fwd")], [i for i, nm in enumerate(names) if nm.startswith("fused_adam")]
    copy_out = {i for i in range(n) if is_copy[i]}
    for f in fwds:
      stop = min((a for a in adams if a > f), default=n)
      for i in range(f, stop):
        if i in copy_out or not data_preds[i] & copy_out or not children[i]: continue
        if getenv("GPTOSS_DEFER_ADDS_LOCAL", 0) and any(c in copy_out for c in children[i]): continue
        prio[i] = min(i + k, min(children[i]) - 0.5)
  import heapq
  waiting, out = [len(p) for p in preds], []
  kids:list[list[int]] = [[] for _ in range(n)]
  for i, ps in enumerate(preds):
    for p in ps: kids[p].append(i)
  ready = [(prio[i], i) for i in range(n) if waiting[i] == 0]
  heapq.heapify(ready)
  while ready:
    _, i = heapq.heappop(ready); out.append(i)
    for c in kids[i]:
      waiting[c] -= 1
      if waiting[c] == 0: heapq.heappush(ready, (prio[c], c))
  assert len(out) == n, "gptoss_sched_fixer: not a dag"
  return out

def _cross_node(c:UOp, devs:tuple[str, ...]) -> bool:
  if c.src[0].op is not Ops.STORE: return False
  from tinygrad.device import Device
  d = [str(x) for x in (devs + ("", ""))[:2]]
  try: return all(x.startswith("AMD") for x in d) and Device[d[0]].peer_group != Device[d[1]].peer_group
  except Exception: return False

def _hosts(devs:list[tuple[str, ...]]) -> set[int]:
  return {i for i, d in enumerate(devs) if not d or any(not str(x).startswith(("AMD", "RDMA")) for x in d)}

def _host_barriers(preds:list[set[int]], hosts:set[int]) -> list[set[int]]:
  # a host call splits the graph into separate submissions: everything keeps its side of it
  out = [set(p) for p in preds]
  for h in sorted(hosts):
    out[h] |= set(range(h))
    for j in range(h + 1, len(preds)): out[j].add(h)
  return out

def gptoss_sched_fixer(linear:UOp) -> UOp:
  calls = list(linear.src)
  if not regions(names:=[_name(c) for c in calls]): return linear # eval, or an update without an optimizer after it
  devs = [_devs(c) for c in calls]
  acc, hosts = [_access(c, d) for c, d in zip(calls, devs)], _hosts(devs)
  is_copy, data_preds = [c.src[0].op is Ops.STORE for c in calls], _preds(calls, acc=acc)
  preds = _host_barriers(data_preds, hosts)
  if (k:=getenv("GPTOSS_CHAIN_DELAY", 0)):
    order = plan_chain_delay(names, is_copy, preds, k, [len(d) for d in devs], _preds(calls, raw_only=True, acc=acc))
  elif (t:=getenv("GPTOSS_CONE_HOIST", 0)): order = plan_cone_hoist(names, is_copy, preds, t, data_preds)
  elif getenv("GPTOSS_NEED_ORDER", 0): order = plan_need(names, is_copy, preds, data_preds)
  elif getenv("GPTOSS_NIC_NEED", 0): order = plan_nic_need(names, preds, data_preds, [_cross_node(c, d) for c, d in zip(calls, devs)], is_copy, hosts,
                                                          _raw_preds(calls, acc))
  else: order = plan(names, is_copy, preds)
  print(f"gptoss_sched_fixer: {len(calls)} calls, {sum(p != i for p, i in enumerate(order))} moved")
  return linear.replace(src=tuple(calls[i] for i in order))

def dump(linear:UOp, path:str, held_bufs:set[UOp]):
  # the step's calls for the cpu schedule emulator: kind, devices, the (buffer, device) lanes written and read, copy endpoints
  ids:dict[UOp, int] = {}
  out = []
  for c in linear.src:
    writes, reads = _access(c)
    ent = {"name": _name(c), "copy": c.src[0].op is Ops.STORE, "devs": _devs(c),
           "writes": [(ids.setdefault(b, len(ids)), d) for b, d in writes], "reads": [(ids.setdefault(b, len(ids)), d) for b, d in reads]}
    if ent["copy"]: ent |= {"dst": c.src[1].device, "src": c.src[2].device, "nbytes": c.src[2].max_numel() * c.src[2].dtype.itemsize}
    out.append(ent)
  # what the memory planner sees: which storages it may place, their size and device
  from tinygrad.schedule.memory import _can_plan
  bufinfo = {i: {"nbytes": b.max_numel() * b.dtype.itemsize, "device": b.device, "plan": b.op is Ops.BUFFER and _can_plan(b, held_bufs)}
             for b, i in ids.items()}
  with open(path, "wb") as f: pickle.dump({"calls": out, "bufs": bufinfo}, f)
  print(f"gptoss_sched_fixer: dumped {len(out)} calls to {path}")

def apply_saved(linear:UOp, path:str) -> UOp:
  # an order the cpu emulator found for exactly this capture: same calls by name, and it must keep every buffer dependency
  with open(path, "rb") as f: saved = pickle.load(f)
  calls = list(linear.src)
  if len(calls) != len(saved["names"]): return linear
  names = [_name(c) for c in calls]
  assert names == saved["names"], "gptoss_sched_fixer: saved order is for a different capture"
  order, preds = saved["order"], _preds(calls)
  pos = {c: p for p, c in enumerate(order)}
  assert sorted(order) == list(range(len(calls))) and all(pos[p] < pos[i] for i, ps in enumerate(preds) for p in ps), \
    "gptoss_sched_fixer: saved order breaks a buffer dependency"
  # host calls split the graph into separately submitted batches: they must keep their place relative to everything else
  host = [i for i, c in enumerate(calls) if any(not str(d).startswith(("AMD", "RDMA")) for d in _devs(c)) or not _devs(c)]
  assert all(pos[h] == h for h in host), "gptoss_sched_fixer: saved order moves a host call"
  print(f"gptoss_sched_fixer: applied saved order {path}, {sum(p != i for p, i in enumerate(order))} moved")
  return linear.replace(src=tuple(calls[i] for i in order))

def install():
  from tinygrad.engine import jit
  def sched_fixer(linear:UOp, held_bufs:set[UOp]) -> UOp:
    if (path:=getenv("GPTOSS_SCHED_DUMP", "")) and len(linear.src) > 10000: dump(linear, f"{path}.{len(linear.src)}", held_bufs)
    if (path:=getenv("GPTOSS_SCHED_ORDER", "")): return apply_saved(linear, path)
    return gptoss_sched_fixer(linear) if getenv("GPTOSS_SCHED_FIXER", 0) else linear
  jit.sched_fixer = sched_fixer
