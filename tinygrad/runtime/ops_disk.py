import os, sys, mmap, io, ctypes, contextlib, pathlib
from tinygrad.helpers import OSX, mv_address
from tinygrad.device import Compiled, Allocator, Buffer
from tinygrad.runtime.support.memory import MMIOInterface
with contextlib.suppress(ImportError):
  import _posixshmem
  from tinygrad.runtime.autogen import io_uring, libc

class DiskDevice(Compiled):
  _tried_io_uring_init = False

  def __init__(self, device:str):
    if not DiskDevice._tried_io_uring_init: self._iouring_setup()

    self.size: int|None = None
    self.fd: int|None = None
    self.refcount = 0
    super().__init__(device, DiskAllocator(self), [], None)
  def _might_open(self, size:int):
    assert self.size is None or size <= self.size, f"can't reopen Disk tensor with larger size, opened with {self.size}, tried to open with {size}"
    if self.size is not None and hasattr(self, "mem"):
      self.refcount += 1
      return
    filename = self.device[len("disk:"):]

    if sys.platform != "win32" and filename.startswith("shm:"):
      fd = _posixshmem.shm_open("/"+filename[4:].lstrip("/"), os.O_RDWR, 0o600)
      self.mem = mmap.mmap(fd, size, mmap.MAP_SHARED | MAP_POPULATE | MAP_LOCKED)
      os.close(fd)
    else:
      try: self.fd = os.open(filename, os.O_RDWR|os.O_CREAT|getattr(os, "O_DIRECT", 0))
      except OSError: self.fd = os.open(filename, os.O_RDWR|os.O_CREAT)
      if not pathlib.Path(filename).is_block_device() and os.fstat(self.fd).st_size < size: os.ftruncate(self.fd, size)
      self.mem = mmap.mmap(self.fd, size)
    self.size = size
    if hasattr(self.mem, 'madvise') and (hp := getattr(mmap, "MADV_HUGEPAGE", None)) is not None:
      with contextlib.suppress(OSError): self.mem.madvise(hp) # some systems have transparent_hugepage disabled
    self.refcount += 1
  def _might_close(self):
    self.refcount -= 1
    if self.refcount == 0:
      if self.fd is not None:
        os.close(self.fd)
      if hasattr(self, "mem"):
        try: self.mem.close()
        except BufferError: pass
      self.size = None
  def _iouring_setup(self):
    DiskDevice._tried_io_uring_init = True

    if sys.platform == 'linux' and not hasattr(sys, "getandroidapilevel"):
      fd = libc.syscall(io_uring.NR_io_uring_setup, 4096, ctypes.byref(p:=io_uring.struct_io_uring_params()))
      if fd < 0: return

      sq_ptr = libc.mmap(0, p.sq_off.array + p.sq_entries * 4, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | MAP_POPULATE, fd, 0)
      cq_ptr = libc.mmap(0, p.cq_off.cqes + p.cq_entries * ctypes.sizeof(io_uring.struct_io_uring_cqe),
                        mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | MAP_POPULATE, fd, io_uring.IORING_OFF_CQ_RING)
      sqes = libc.mmap(0, p.sq_entries * ctypes.sizeof(io_uring.struct_io_uring_sqe),
                      mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | MAP_POPULATE, fd, io_uring.IORING_OFF_SQES)

      def u32ptr(val): return ctypes.cast(val, ctypes.POINTER(ctypes.c_uint32))
      sqdesc = io_uring.struct_io_uring_sq(khead=u32ptr(sq_ptr+p.sq_off.head), ktail=u32ptr(sq_ptr+p.sq_off.tail),
                                           array=u32ptr(sq_ptr+p.sq_off.array),
        kring_mask=u32ptr(sq_ptr+p.sq_off.ring_mask), sqes=ctypes.cast(sqes, ctypes.POINTER(io_uring.struct_io_uring_sqe)))

      cqdesc = io_uring.struct_io_uring_cq(khead=u32ptr(cq_ptr+p.cq_off.head), ktail=u32ptr(cq_ptr+p.cq_off.tail),
        kring_mask=u32ptr(sq_ptr+p.cq_off.ring_mask), cqes=ctypes.cast(cq_ptr+p.cq_off.cqes, ctypes.POINTER(io_uring.struct_io_uring_cqe)))

      DiskDevice.io_uring = io_uring.struct_io_uring(ring_fd=fd, sq=sqdesc, cq=cqdesc) # type: ignore

MAP_LOCKED, MAP_POPULATE = 0 if OSX else 0x2000, getattr(mmap, "MAP_POPULATE", 0 if OSX else 0x008000)
class DiskAllocator(Allocator):
  def __init__(self, dev:DiskDevice): super().__init__(dev)
  def _alloc(self, buf:Buffer, opaque=None):
    self.dev._might_open(buf.nbytes)
    return None, MMIOInterface(mv_address(memoryview(self.dev.mem)), buf.nbytes), None
  def _free(self, buf:Buffer): self.dev._might_close()
  def _copyout(self, dst:memoryview, buf:Buffer):
    if OSX and self.dev.fd is not None:
      # OSX doesn't seem great at mmap, this is faster
      with io.FileIO(self.dev.fd, "a+b", closefd=False) as fo:
        fo.seek(buf.offset)
        bytes_read = 0
        while (n := fo.readinto(dst[bytes_read:])) is not None and n > 0: bytes_read += n
    else: super()._copyout(dst, buf)
