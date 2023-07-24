import ctypes
from collections import defaultdict, deque
import numpy as np
from typing import TypeVar, Type, Any, Optional, Dict, List, Deque, Union
from tinygrad.helpers import DType, dtypes, prod, GlobalCounters, ImageDType
from dataclasses import dataclass

_T = TypeVar("_T")
class RawBuffer:  # pylint: disable=abstract-method
  def __init__(self, size:int, dtype:DType, buf:Any=None):
    self.size: int = size
    self.dtype: DType = dtype
    self._buf = buf
    self._memsz: int = size*dtype.itemsize
    GlobalCounters.mem_used += self._memsz

  def __del__(self):  # NOTE: if it fails on init (bad dtype), it won't have a _memsz
    if hasattr(self, '_memsz'): GlobalCounters.mem_used -= self._memsz

  def __repr__(self): return f"buffer<{self.size}, {self.dtype}>"
  @property
  def key(self): return (self.size, self.dtype.key)

  # NOTE: this interface allows for 0 copy
  @classmethod
  def fromCPU(cls:Type[_T], x:np.ndarray) -> _T: raise NotImplementedError("must be implemented")
  def toCPU(self) -> np.ndarray: raise NotImplementedError("must be implemented")

class RawBufferCopyIn(RawBuffer):
  def _copyin(self, x:np.ndarray) -> None: raise NotImplementedError("must be implemented")

  @classmethod
  def fromCPU(cls, x:np.ndarray, **kwargs):
    ret = cls(prod(x.shape), dtypes.from_np(x.dtype), **kwargs)
    ret._copyin(x)
    return ret

class RawBufferMapped(RawBufferCopyIn):
  def _buffer(self) -> memoryview: raise NotImplementedError("must be implemented")
  # NOTE: this metadata prevents the backing buffer from being freed. hack can be removed with PEP688
  def toCPU(self) -> np.ndarray: return np.frombuffer(self._buffer(), dtype=np.dtype(self.dtype.np, metadata={"backing": self}))  # type: ignore
  def _copyin(self, x:np.ndarray) -> None: np.copyto(self.toCPU(), x.reshape(-1))

# this one is simple enough that i moved it out of the runtimes
class RawMallocBuffer(RawBufferMapped):
  def __init__(self, size, dtype: DType): super().__init__(size, dtype, ({dtypes.float32: ctypes.c_float, dtypes.float16: ctypes.c_int16, dtypes.bfloat16: ctypes.c_int16, dtypes.int8: ctypes.c_int8, dtypes.uint8: ctypes.c_uint8, dtypes.bool: ctypes.c_uint8, dtypes.int32: ctypes.c_int32, dtypes.int64: ctypes.c_int64}[dtype] * size)())
  def _buffer(self): return memoryview(self._buf)

class RawBufferCopyInOut(RawBufferCopyIn):
  def _copyout(self, x:np.ndarray) -> None: raise NotImplementedError("must be implemented")

  def toCPU(self) -> np.ndarray:
    x: np.ndarray = np.empty(self.size, dtype=self.dtype.np)
    self._copyout(x)
    return x

class RawConst(RawBuffer): # pylint: disable=abstract-method
  def __repr__(self): return f"const<{self._buf}, {self.dtype}>"
  @property
  def key(self): return (str(self._buf), self.dtype.key)

def buf_is_kernel_arg(x) -> bool:
  return x.realized is not None and x.realized.__class__ is not RawConst

# **** Device info classes *****

@dataclass
class DeviceInfo:
  memory_size: Optional[int] = None # Memory available on the device

# **** Allocators *****

class AllocatedBufferWrapper:
  def __init__(self, wrapped_buffer, allocator):
    self._wrapped_buffer = wrapped_buffer
    self._allocator = allocator
  def __getattr__(self, name):
    if name in ['_wrapped_buffer', '_allocator']: return object.__getattribute__(self, name)
    return getattr(self._wrapped_buffer, name)
  def __setattr__(self, name, value):
    if name in ['_wrapped_buffer', '_allocator']: super().__setattr__(name, value)
    else: setattr(self._wrapped_buffer, name, value)
  def __delattr__(self, name): delattr(self._wrapped_buffer, name)
  def __del__(self): self._allocator._on_buffer_free(self._wrapped_buffer)
  def __repr__(self): return self._wrapped_buffer.__repr__()

  @property # type: ignore
  def __class__(self): return self._wrapped_buffer.__class__
def is_buffer_wrapped(buf: Union[RawBuffer, AllocatedBufferWrapper]) -> bool: return hasattr(buf, '_wrapped_buffer')

class Allocator:
  def __init__(self, buftype):
    self.buftype = buftype
  def __call__(self, size, dtype, **kwargs):
    return self.buftype(size, dtype, **kwargs)

class LRUAllocator(Allocator):
  oom_threshold: float = 0.8
  oom_rem_buffer_count: int = 16 # Buffers count to be freed when OOM.

  def __init__(self, buftype, device_info:Optional[DeviceInfo]=None):
    super().__init__(buftype)
    self.aging_order: List[RawBuffer] = []
    self.cached_buffers: Dict[int, Deque[RawBuffer]] = defaultdict(deque)
    self.dev_memsz: int = device_info.memory_size if device_info and device_info.memory_size is not None else (4<<30) # When no devinfo defaulting to 4Gbs.

  def _get_bin(self, size, dtype): return (size, dtype)
  def _cache_add_buffer(self, buf):
    GlobalCounters.mem_used -= buf._memsz
    GlobalCounters.mem_cached += buf._memsz
    self.aging_order.append(buf)
    self.cached_buffers[self._get_bin(buf.size, buf.dtype)].appendleft(buf)

  def _cache_rem_buffer(self, buf):
    GlobalCounters.mem_used += buf._memsz
    GlobalCounters.mem_cached -= buf._memsz
    self.aging_order.remove(buf)
    self.cached_buffers[self._get_bin(buf.size, buf.dtype)].remove(buf)

  def _cached_buffer_matches(self, buf, size, dtype):
    if isinstance(buf.dtype, ImageDType) and isinstance(dtype, ImageDType): return buf.dtype == dtype and buf.dtype.shape == dtype.shape
    return buf.dtype == dtype and buf.size == size

  def _check_for_cache_unload(self):
    if self.free_space < self.dev_memsz * (1 - LRUAllocator.oom_threshold): # Unload cache if the device is about to be OOM.
      for _ in range(min(LRUAllocator.oom_rem_buffer_count, len(self.aging_order))): self._cache_rem_buffer(self.aging_order[0])

  def _unload_cache_to_fit_buffer(self, memsize):
    while len(self.aging_order) and self.free_space < memsize: self._cache_rem_buffer(self.aging_order[0])

  def __call__(self, size, dtype, **kwargs):
    bin_id = self._get_bin(size, dtype)
    best_fit = next((i for i,buf in enumerate(self.cached_buffers[bin_id]) if self._cached_buffer_matches(buf, size, dtype)), None)
    rawbuf = self.cached_buffers[bin_id][best_fit] if best_fit is not None else None
    if not rawbuf:
      self._unload_cache_to_fit_buffer(size*dtype.itemsize)
      rawbuf = self.buftype(size, dtype, **kwargs)
    else:
      self._cache_rem_buffer(rawbuf)
    return AllocatedBufferWrapper(rawbuf, self)

  def _on_buffer_free(self, buf) -> bool:
    self._check_for_cache_unload()
    self._cache_add_buffer(buf)
    return False

  @property
  def free_space(self): return self.dev_memsz - GlobalCounters.mem_used - GlobalCounters.mem_cached
