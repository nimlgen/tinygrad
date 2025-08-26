# ruff: noqa: E501
import unittest
from tinygrad import dtypes
from tinygrad.codegen.opt.kernel import Kernel
from tinygrad.codegen.opt.search import Opt, OptOps, bufs_from_lin
from extra.optimization.helpers import time_linearizer

# stuff needed to unpack a kernel
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View

def _test_overflow(ast, opts):
  lin = Kernel(ast)
  lin.apply_opts(opts)
  bufs = bufs_from_lin(lin)
  print(bufs)
  time_linearizer(lin, bufs, disable_cache=True)

class TestLinearizerOverflow(unittest.TestCase):
  def test_overflow_1(self):
    ast = UOp(Ops.SINK, dtypes.void, arg=KernelInfo(name='test', axis_types=(), dont_use_locals=False, applied_opts=(), opts_to_apply=(Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.LOCAL, axis=1, arg=4))), src=(
  UOp(Ops.STORE, dtypes.void, arg=None, src=(
    UOp(Ops.VIEW, dtypes.bfloat16.ptr(134217728), arg=ShapeTracker(views=(View(shape=(16, 1024, 8192, 1), strides=(8388608, 8192, 1, 0), offset=0, mask=None, contiguous=True),)), src=(
      UOp(Ops.DEFINE_GLOBAL, dtypes.bfloat16.ptr(134217728), arg=0, src=()),)),
    UOp(Ops.MUL, dtypes.bfloat16, arg=None, src=(
      UOp(Ops.MUL, dtypes.bfloat16, arg=None, src=(
        x5:=UOp(Ops.CAST, dtypes.bfloat16, arg=None, src=(
          UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (3,)), src=(
            UOp(Ops.CAST, dtypes.float, arg=None, src=(
              UOp(Ops.MUL, dtypes.bfloat16, arg=None, src=(
                UOp(Ops.LOAD, dtypes.bfloat16, arg=None, src=(
                  UOp(Ops.VIEW, dtypes.bfloat16.ptr(33554432), arg=ShapeTracker(views=(View(shape=(16, 1024, 8192, 2048), strides=(2097152, 2048, 0, 1), offset=0, mask=None, contiguous=False),)), src=(
                    UOp(Ops.DEFINE_GLOBAL, dtypes.bfloat16.ptr(33554432), arg=1, src=()),)),)),
                UOp(Ops.LOAD, dtypes.bfloat16, arg=None, src=(
                  UOp(Ops.VIEW, dtypes.bfloat16.ptr(16777216), arg=ShapeTracker(views=(View(shape=(16, 1024, 8192, 2048), strides=(0, 0, 2048, 1), offset=0, mask=None, contiguous=False),)), src=(
                    UOp(Ops.DEFINE_GLOBAL, dtypes.bfloat16.ptr(16777216), arg=2, src=()),)),)),)),)),)),)),
        UOp(Ops.RECIP, dtypes.bfloat16, arg=None, src=(
          UOp(Ops.ADD, dtypes.bfloat16, arg=None, src=(
            UOp(Ops.CONST, dtypes.bfloat16, arg=1.0, src=(
              x18:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(16, 1024, 8192, 1), strides=(0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
            UOp(Ops.EXP2, dtypes.bfloat16, arg=None, src=(
              UOp(Ops.MUL, dtypes.bfloat16, arg=None, src=(
                 x5,
                UOp(Ops.CONST, dtypes.bfloat16, arg=-1.4426950408889634, src=(
                   x18,)),)),)),)),)),)),
      UOp(Ops.CAST, dtypes.bfloat16, arg=None, src=(
        UOp(Ops.LOAD, dtypes.float, arg=None, src=(
          UOp(Ops.VIEW, dtypes.float.ptr(134217728), arg=ShapeTracker(views=(View(shape=(16, 1024, 8192, 1), strides=(8388608, 8192, 1, 0), offset=0, mask=None, contiguous=True),)), src=(
            UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(134217728), arg=3, src=()),)),)),)),)),)),))
    opts = [Opt(op=OptOps.TC, axis=0, arg=(-1, 0, 1)), Opt(op=OptOps.UPCAST, axis=0, arg=4), Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.LOCAL, axis=1, arg=4)]
    _test_overflow(ast, opts)

if __name__ == '__main__':
  unittest.main()
