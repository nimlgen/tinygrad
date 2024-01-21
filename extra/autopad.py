from tinygrad.tensor import Tensor
from tinygrad.ops import LoadOps
from tinygrad.codegen.linearizer import Linearizer
from test.external.fuzz_linearizer import run_linearizer
from tinygrad.features.search import get_linearizer_actions, bufs_from_lin
from tinygrad.codegen.kernel import Opt, OptOps
from tinygrad.helpers import Context
import numpy as np

N = 17**3

a = Tensor.rand(N, N)
b = Tensor.rand(N, N)
c = a @ b
sched = [si for si in c.lazydata.schedule() if si.ast.op not in LoadOps]
assert len(sched) == 1
lin = Linearizer(sched[0].ast)

lin.apply_opt(Opt(op=OptOps.PADTO, axis=0, amt=32))
lin.apply_opt(Opt(op=OptOps.PADTO, axis=1, amt=32))
lin.hand_coded_optimizations()
lin.linearize()
print(f"{lin.applied_opts=}")

run_linearizer(lin)

###

a = Tensor.rand(61, 61).sum(axis=0)
sched = [si for si in a.lazydata.schedule() if si.ast.op not in LoadOps]
assert len(sched) == 1
lin = Linearizer(sched[0].ast)

lin.apply_opt(Opt(op=OptOps.PADTO, axis=0, amt=32))
lin.hand_coded_optimizations()
lin.linearize()
print(f"{lin.applied_opts=}")

run_linearizer(lin)

###

a = Tensor.ones(10, 30522).sum(axis=1)
sched = [si for si in a.lazydata.schedule() if si.ast.op not in LoadOps]
assert len(sched) == 1
lin = Linearizer(sched[0].ast)

lin.apply_opt(Opt(op=OptOps.PADTO, axis=1, amt=32))
lin.apply_opt(Opt(op=OptOps.GROUPTOP, axis=0, amt=32))
lin.linearize()
print(f"{lin.applied_opts=}")

rawbufs = bufs_from_lin(lin)
run_linearizer(lin, rawbufs)
assert np.allclose(np.frombuffer(rawbufs[0].as_buffer(), rawbufs[0].dtype.np), a.numpy())

###

with Context(BEAM=4):
    x = Tensor.rand(10, 30522).sum(axis=1).realize()