import functools, itertools
from tinygrad.helpers import all_int, prod, DEBUG, RING, ALL2ALL, getenv
from tinygrad.uop.ops import UOp
from tinygrad.device import Device

# *** allreduce implementation ***
def handle_allreduce(buf:UOp, red:UOp) -> UOp|None:
  if not isinstance(buf.device, tuple): return None
  ndev, shape, numel = len(buf.device), buf.shape, prod(buf.shape)
  op, device = red.arg

  # ring allreduce doesn't provide a benefit with only 2 nodes or where number of elements is less than 256k (empirically)
  # fallback to naive allreduce to save on kernel dispatch, chunking and reassembling chunks.
  concrete = all_int(shape)
  use_all2all = concrete and (ALL2ALL >= 2 or (ndev > 2 and numel > getenv("RING_ALLREDUCE_THRESHOLD", 256_000) and ALL2ALL >= 1))
  use_ring = concrete and not use_all2all and (RING >= 2 or (ndev > 2 and numel > getenv("RING_ALLREDUCE_THRESHOLD", 256_000) and RING >= 1))
  if DEBUG >= 2: print(f"{'ALL2ALL' if use_all2all else 'RING' if use_ring else 'NAIVE'} ALLREDUCE {ndev}x{numel} | {buf.dtype}")

  buf = buf.pad_to(buf.max_shape)
  # contiguous before we copy it
  buf = buf.contiguous()

  # naive: copy to all devices. if you shrink later, that'll be handled
  if not use_ring and not use_all2all:
    return functools.reduce(lambda x,y: x.alu(op, y), [buf.mselect(i).copy_to_device(device) for i in range(ndev)]).shrink_to(shape)

  # two nodes: reduce-scatter in each node, every chunk owner exchanges with its counterpart on the other node, all-gather in each node
  nodes = [[i for i, d in enumerate(buf.device) if Device[d].peer_group == g] for g in dict.fromkeys(Device[d].peer_group for d in buf.device)]
  if use_all2all and len(nodes) == 2 and len(nodes[0]) == len(nodes[1]):
    n, flat = len(nodes[0]), buf.reshape((numel,))
    chunks = list(itertools.pairwise(itertools.accumulate([numel // n + (i < numel % n) for i in range(n)], initial=0)))
    owned = {i: functools.reduce(lambda x, y: x.alu(op, y), [flat.mselect(j).shrink((chunks[k],)).copy_to_device(buf.device[i]) for j in node])
             for node in nodes for k, i in enumerate(node)}
    for a, b in zip(*nodes): # the counterparts, linked by their nics
      owned[a], owned[b] = owned[a].alu(op, owned[b].copy_to_device(buf.device[a])), owned[b].alu(op, owned[a].copy_to_device(buf.device[b]))
    gathered = [UOp.mstack(*(owned[node[k]].copy_to_device(buf.device[j]) for node in nodes for j in node)) for k in range(n)]
    return UOp.usum(*[c.pad(((s, numel - e),)) for (s, e), c in zip(chunks, gathered)]).reshape(shape)

  # chunk data into ndev pieces
  assert isinstance(numel, int)
  factor = next((f for f in [32, 16, 8, 4, 2] if numel % f == 0), 1)
  base, left = divmod(numel // factor,  ndev)
  chunks = list(itertools.pairwise(itertools.accumulate([(base + 1) * factor] * left + [base * factor] * (ndev - left), initial=0)))

  # reduce-scatter
  reduced_chunks:list[UOp] = []
  for i,(s,e) in enumerate(chunks):
    if use_all2all:
      chunks_on_i = [buf.mselect(j).reshape((numel,)).shrink(((s,e),)).copy_to_device(buf.device[i]) for j in range(ndev)]
      reduced_chunks.append(functools.reduce(lambda x,y: x.alu(op, y), chunks_on_i))
    else:
      chunk, reduced = buf.reshape((numel,)).shrink(((s,e),)), buf.reshape((numel,)).shrink(((s,e),))
      for step in range(ndev-1):
        src, dest = (i+step)%ndev, (i+step+1)%ndev
        cp = reduced.copy_to_device(buf.device[dest], src if isinstance(reduced.device, tuple) else None)
        reduced = cp.alu(op, chunk.copy_to_device(buf.device[dest], dest))
      reduced_chunks.append(reduced)

  # allgather
  copied_chunks:list[UOp] = []
  for i,rc in enumerate(reduced_chunks):
    if isinstance(device, str): copied_chunks.append(rc.copy_to_device(device))
    elif use_all2all: copied_chunks.append(UOp.mstack(*(rc.copy_to_device(buf.device[j]) for j in range(ndev))))
    else:
      chain:list[UOp] = [rc]
      for step in range(ndev-1):
        chain.append(rc := rc.copy_to_device(buf.device[(i+step)%ndev]))
      copied_chunks.append(UOp.mstack(*(chain[(j-i+1)%ndev] for j in range(ndev))))

  # reassemble
  return UOp.usum(*[c.pad(((s,numel-e),)) for (s,e),c in zip(chunks, copied_chunks)]).reshape(shape)

def create_allreduce_function(buf:UOp, red:UOp, output:UOp|None=None) -> UOp|None:
  if output is None: output = UOp.invalids(red.shape, dtype=red.dtype, device=red.device)
  to = red.param_like(0)
  src = buf.param_like(1)
  red = src.allreduce(*red.arg)
  return output.after(to.after(to.store(handle_allreduce(src, red))).sink().call(output, buf.contiguous(), name="allreduce", precompile=True))
