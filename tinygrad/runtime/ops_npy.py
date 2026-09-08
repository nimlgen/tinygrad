import numpy as np
from typing import Any
from tinygrad.device import Compiled, Buffer
from tinygrad.runtime.ops_cpu import HostAllocator
from tinygrad.runtime.support.memory import MMIOInterface

class NpyAllocator(HostAllocator):
  def _alloc(self, buf:Buffer, opaque:Any=None) -> tuple[int|None, MMIOInterface|None, Any]:
    arr = np.require(opaque, requirements='C') if opaque is not None else np.empty(buf.nbytes, dtype=np.uint8)
    return arr.ctypes.data, MMIOInterface(arr.ctypes.data, buf.nbytes), arr

class NpyDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, NpyAllocator(self), [], None)
