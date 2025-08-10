# ruff: noqa: E501
import unittest
from tinygrad import dtypes
from tinygrad.opt.kernel import Kernel
from tinygrad.opt.search import Opt, OptOps, bufs_from_lin
from extra.optimization.helpers import time_linearizer

# stuff needed to unpack a kernel
from tinygrad.uop.ops import UOp, Ops
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View

def _test_kernel(ast, opts):
  lin = Kernel(ast)
  lin.apply_opts(opts)
  bufs = bufs_from_lin(lin)
  print(bufs)
  time_linearizer(lin, bufs)

class TestLinearizerOverflow(unittest.TestCase):
  def test_kernel_failure_1(self):
    # mi350
    ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
  UOp(Ops.STORE, dtypes.void, arg=None, src=(
    UOp(Ops.VIEW, dtypes.half.ptr(5767168), arg=ShapeTracker(views=(View(shape=(11, 512, 1024, 1, 1), strides=(524288, 1024, 1, 0, 0), offset=0, mask=None, contiguous=True),)), src=(
      UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(5767168), arg=0, src=()),)),
    UOp(Ops.REDUCE_AXIS, dtypes.half, arg=(Ops.ADD, (3,)), src=(
      UOp(Ops.MUL, dtypes.half, arg=None, src=(
        UOp(Ops.CAST, dtypes.half, arg=None, src=(
          UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
            UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
              UOp(Ops.ADD, dtypes.int, arg=None, src=(
                UOp(Ops.REDUCE_AXIS, dtypes.int, arg=(Ops.ADD, (4,)), src=(
                  UOp(Ops.WHERE, dtypes.int, arg=None, src=(
                    UOp(Ops.VALID, dtypes.bool, arg=None, src=(
                      UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(30523, 61043), strides=(0, 0), offset=0, mask=((0, 30523), (30521, 61043)), contiguous=False), View(shape=(11, 512, 1024, 30522, 30522), strides=(0, 0, 0, 1, 61044), offset=0, mask=None, contiguous=False))), src=()),)),
                    UOp(Ops.CONST, dtypes.int, arg=1, src=(
                      x14:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(11, 512, 1024, 30522, 30522), strides=(0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                    UOp(Ops.CONST, dtypes.int, arg=0, src=(
                       x14,)),)),)),
                UOp(Ops.CONST, dtypes.int, arg=-1, src=(
                  x17:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(11, 512, 1024, 30522, 1), strides=(0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
              UOp(Ops.LOAD, dtypes.int, arg=None, src=(
                UOp(Ops.VIEW, dtypes.int.ptr(5632), arg=ShapeTracker(views=(View(shape=(11, 512, 1024, 30522, 1), strides=(512, 1, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(5632), arg=1, src=()),)),)),)),
            UOp(Ops.CONST, dtypes.bool, arg=True, src=(
               x17,)),)),)),
        UOp(Ops.LOAD, dtypes.half, arg=None, src=(
          UOp(Ops.VIEW, dtypes.half.ptr(31254528), arg=ShapeTracker(views=(View(shape=(11, 512, 1024, 30522, 1), strides=(0, 0, 1, 1024, 0), offset=0, mask=None, contiguous=False),)), src=(
            UOp(Ops.DEFINE_GLOBAL, dtypes.half.ptr(31254528), arg=2, src=()),)),)),)),)),)),))
    opts = [Opt(op=OptOps.LOCAL, axis=1, arg=16), Opt(op=OptOps.LOCAL, axis=1, arg=16), Opt(op=OptOps.PADTO, axis=5, arg=32)]
    _test_kernel(ast, opts)

if __name__ == '__main__':
  unittest.main()
