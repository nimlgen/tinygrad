from __future__ import annotations
from typing import Tuple, List, Any, Optional, cast
import os, ctypes, ctypes.util, functools, pathlib, mmap, errno, array, contextlib, sys, subprocess, select, struct, time
assert sys.platform != 'win32'
from dataclasses import dataclass
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HWQueue, CLikeArgsState, HCQSignal, HCQProgram
from tinygrad.ops import sint
from tinygrad.device import BufferSpec
from tinygrad.helpers import getenv, to_mv, round_up, data64_le, mv_address, from_mv, lo32, hi32
from tinygrad.renderer.cstyle import AMDRenderer
from tinygrad.runtime.autogen import kfd, hsa, amd_gpu, libc, libpciaccess, vfio, pci
from tinygrad.runtime.autogen.am import am
from tinygrad.runtime.support.compiler_hip import AMDCompiler
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.am.amdev import AMDev
from tinygrad.runtime.support.am.usb import Asm236x
if getenv("IOCTL"): import extra.hip_gpu_driver.hip_ioctl  # noqa: F401 # pylint: disable=unused-import
if getenv("MOCKGPU"): import extra.mockgpu.mockgpu # noqa: F401 # pylint: disable=unused-import

regBIF_BX_PF1_GPU_HDP_FLUSH_REQ, regBIF_BX_PF1_GPU_HDP_FLUSH_DONE = 0x0106, 0x0107

EVENT_INDEX_PARTIAL_FLUSH = 4 # based on a comment in nvd.h
WAIT_REG_MEM_FUNCTION_GEQ = 5 # >=

COMPUTE_SHADER_EN, FORCE_START_AT_000, CS_W32_EN = (1 << 0), (1 << 2), (1 << 15)

def gfxreg(reg): return reg + 0x00001260 - amd_gpu.PACKET3_SET_SH_REG_START
def nbioreg(reg): return reg + 0x00000d20 # NBIO_BASE__INST0_SEG2

# def read_physical_memory(vaddr):
#   # Define the page size and calculate the page index
#   page_size = 4096
#   page_index = vaddr // page_size
#   offset = page_index * 8

#   # Open /proc/self/pagemap to read the physical address mapping
#   with open(f"/proc/self/pagemap", "rb") as f:
#     f.seek(offset)
#     # Read the entry for the page and extract the physical address
#     paddr = struct.unpack("Q", f.read(8))[0] & ((1 << 54) - 1)
#     paddr = paddr * page_size + (vaddr % page_size)

#   # Open /dev/mem and use mmap to read from the physical address
#   with open("/dev/mem", "r") as f:
#     # Memory map the physical address to the process's address space
#     mapped_addr = libc.mmap(0,
#             8,
#             mmap.PROT_READ,
#             mmap.MAP_SHARED,
#             f.fileno(),
#             paddr)
    
#     # print(mapped_addr)

#     # Read the 8-byte data from the mapped memory (you can change the number of bytes if needed)
#     value = struct.unpack("Q", ctypes.string_at(mapped_addr, 8))[0]
    
#     # Close the memory map after reading
#     # mem_map.close()
    
#     return value

class AMDSignal(HCQSignal):
  def __init__(self, dev, base_addr:Optional[int]=None, **kwargs):
    self.dev = dev
    super().__init__(dev.signals_pool.pop() if base_addr is None else base_addr, **kwargs, timestamp_divider=100)

  def __del__(self):
    if isinstance(self.base_addr, int): self.dev.signals_pool.append(self.base_addr)

  def _sleep(self, time_spent_waiting_ms:int):
    # Resonable to sleep for long workloads (which take more than 2s) and only timeline signals.
    pass
    # self.timeline_for_device.dev_iface.adev.gmc.flush_hdp()
    # self.timeline_for_device.dev_iface.adev.gmc.flush_tlb(ip="GC", vmid=0, flush_type=2)
    # self.timeline_for_device.dev_iface.adev.gmc.flush_tlb(ip="MM", vmid=0, flush_type=2)
    # if time_spent_waiting_ms > 5000: print("HMM", self.value)

    # if time_spent_waiting_ms > 2000 and self.timeline_for_device is not None: self.timeline_for_device.dev_iface.sleep(timeout=200)

class AMDComputeQueue(HWQueue):
  def __del__(self):
    if self.binded_device is not None:
      self.binded_device.allocator.free(self.hw_page, self.hw_page.size, BufferSpec(cpu_access=True, nolru=True, uncached=True))

  def pkt3(self, cmd, *vals):
    self.q(amd_gpu.PACKET3(cmd, len(vals) - 1), *vals)
    # self._h = array.array('I', self._q)

  def wait_reg_mem(self, value, mask=0xffffffff, mem=None, reg_req=None, reg_done=None):
    wrm_info_dw = amd_gpu.WAIT_REG_MEM_MEM_SPACE(int(mem is not None)) | amd_gpu.WAIT_REG_MEM_OPERATION(int(mem is None)) \
                | amd_gpu.WAIT_REG_MEM_FUNCTION(WAIT_REG_MEM_FUNCTION_GEQ) | amd_gpu.WAIT_REG_MEM_ENGINE(0)

    self.pkt3(amd_gpu.PACKET3_WAIT_REG_MEM, wrm_info_dw, *(data64_le(mem) if mem is not None else (reg_req, reg_done)), value, mask, 4)

  def acquire_mem(self, addr=0x0, sz=(1 << 64)-1, gli=1, glm=1, glk=1, glv=1, gl1=1, gl2=1):
    cache_flags_dw = amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV(gli) \
                   | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_INV(glm) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_WB(glm) \
                   | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV(glk) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_WB(glk) \
                   | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV(glv) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV(gl1) \
                   | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV(gl2) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_WB(gl2)

    self.pkt3(amd_gpu.PACKET3_ACQUIRE_MEM, 0, *data64_le(sz), *data64_le(addr), 0, cache_flags_dw)

  def release_mem(self, address, value, data_sel, int_sel, ctxid=0, cache_flush=False):
    cache_flags_dw = 0 if not cache_flush else (amd_gpu.PACKET3_RELEASE_MEM_GCR_GLV_INV | amd_gpu.PACKET3_RELEASE_MEM_GCR_GL1_INV \
                   | amd_gpu.PACKET3_RELEASE_MEM_GCR_GL2_INV | amd_gpu.PACKET3_RELEASE_MEM_GCR_GLM_WB \
                   | amd_gpu.PACKET3_RELEASE_MEM_GCR_GLM_INV | amd_gpu.PACKET3_RELEASE_MEM_GCR_GL2_WB | amd_gpu.PACKET3_RELEASE_MEM_GCR_SEQ)

    event_dw = amd_gpu.PACKET3_RELEASE_MEM_EVENT_TYPE(amd_gpu.CACHE_FLUSH_AND_INV_TS_EVENT) \
             | amd_gpu.PACKET3_RELEASE_MEM_EVENT_INDEX(amd_gpu.event_index__mec_release_mem__end_of_pipe)

    memsel_dw = amd_gpu.PACKET3_RELEASE_MEM_DATA_SEL(data_sel) | amd_gpu.PACKET3_RELEASE_MEM_INT_SEL(int_sel) | amd_gpu.PACKET3_RELEASE_MEM_DST_SEL(0)

    self.pkt3(amd_gpu.PACKET3_RELEASE_MEM, event_dw | cache_flags_dw, memsel_dw, *data64_le(address), *data64_le(value), ctxid)

  def memory_barrier(self):
    self.wait_reg_mem(reg_req=nbioreg(regBIF_BX_PF1_GPU_HDP_FLUSH_REQ), reg_done=nbioreg(regBIF_BX_PF1_GPU_HDP_FLUSH_DONE), value=0xffffffff)
    self.acquire_mem()
    return self

  def exec(self, prg:AMDProgram, args_state:CLikeArgsState, global_size:Tuple[sint, ...], local_size:Tuple[sint, ...]):
    self.bind_args_state(args_state)

    self.acquire_mem(gli=0, gl2=0)

    if prg.enable_private_segment_sgpr:
      scratch_hilo = data64_le(prg.dev.scratch.va_addr)
      # sgpr word1 bit31 enables swizzle
      # sgpr word3 = 0x14 << 12 | 2 << 28 | 2 << 21 | 1 << 23
      user_regs = [scratch_hilo[0], scratch_hilo[1] | 1 << 31, 0xffffffff, 0x20c14000] if prg.enable_private_segment_sgpr else []
    else: user_regs = []
    if prg.enable_dispatch_ptr:
      dp = hsa.hsa_kernel_dispatch_packet_t.from_address(dp_addr:=args_state.ptr + prg.kernargs_segment_size)

      self.bind_sints(*local_size, struct=dp, start_field='workgroup_size_x', fmt='H')
      self.bind_sints(*[g*l for g,l in zip(global_size, local_size)], struct=dp, start_field='grid_size_x', fmt='I')
      dp.group_segment_size, dp.private_segment_size, dp.kernarg_address = prg.group_segment_size, prg.private_segment_size, args_state.ptr
      user_regs += [*data64_le(dp_addr)]

    user_regs += [*data64_le(args_state.ptr)]

    assert prg.prog_addr > 0, f"Program address is not set {prg.prog_addr}"
    self.pkt3(amd_gpu.PACKET3_SET_SH_REG, gfxreg(amd_gpu.regCOMPUTE_PGM_LO), *data64_le(prg.prog_addr >> 8))
    self.pkt3(amd_gpu.PACKET3_SET_SH_REG, gfxreg(amd_gpu.regCOMPUTE_PGM_RSRC1), prg.rsrc1, prg.rsrc2)
    self.pkt3(amd_gpu.PACKET3_SET_SH_REG, gfxreg(amd_gpu.regCOMPUTE_PGM_RSRC3), 0)
    self.pkt3(amd_gpu.PACKET3_SET_SH_REG, gfxreg(amd_gpu.regCOMPUTE_TMPRING_SIZE), prg.dev.tmpring_size)
    if prg.dev.has_scratch_base_registers:
      self.pkt3(amd_gpu.PACKET3_SET_SH_REG, gfxreg(amd_gpu.regCOMPUTE_DISPATCH_SCRATCH_BASE_LO), *data64_le(prg.dev.scratch.va_addr >> 8))
    if prg.dev.target < 110000: self.pkt3(amd_gpu.PACKET3_SET_SH_REG, gfxreg(amd_gpu.mmCP_COHER_START_DELAY), 0x20)
    self.pkt3(amd_gpu.PACKET3_SET_SH_REG, gfxreg(amd_gpu.regCOMPUTE_RESTART_X), 0, 0, 0, 0)
    self.pkt3(amd_gpu.PACKET3_SET_SH_REG, gfxreg(amd_gpu.regCOMPUTE_STATIC_THREAD_MGMT_SE0), 0xFFFFFFFF, 0xFFFFFFFF)
    self.pkt3(amd_gpu.PACKET3_SET_SH_REG, gfxreg(amd_gpu.regCOMPUTE_STATIC_THREAD_MGMT_SE2), 0xFFFFFFFF, 0xFFFFFFFF)
    self.pkt3(amd_gpu.PACKET3_SET_SH_REG, gfxreg(amd_gpu.regCOMPUTE_STATIC_THREAD_MGMT_SE4), 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
    self.pkt3(amd_gpu.PACKET3_SET_SH_REG, gfxreg(amd_gpu.regCOMPUTE_USER_DATA_0), *user_regs)

    self.pkt3(amd_gpu.PACKET3_SET_SH_REG, gfxreg(amd_gpu.regCOMPUTE_START_X), 0, 0, 0, *local_size, 0, 0)
    self.pkt3(amd_gpu.PACKET3_SET_SH_REG, gfxreg(amd_gpu.regCOMPUTE_RESOURCE_LIMITS), 0)

    self.pkt3(amd_gpu.PACKET3_DISPATCH_DIRECT, *global_size, CS_W32_EN | FORCE_START_AT_000 | COMPUTE_SHADER_EN)
    self.pkt3(amd_gpu.PACKET3_EVENT_WRITE, amd_gpu.EVENT_TYPE(amd_gpu.CS_PARTIAL_FLUSH) | amd_gpu.EVENT_INDEX(EVENT_INDEX_PARTIAL_FLUSH))
    return self

  def wait(self, signal:AMDSignal, value:sint=0):
    self.wait_reg_mem(mem=signal.value_addr, value=value, mask=0xffffffff)
    return self

  def timestamp(self, signal:AMDSignal):
    self.release_mem(signal.timestamp_addr, 0, amd_gpu.data_sel__mec_release_mem__send_gpu_clock_counter, amd_gpu.int_sel__mec_release_mem__none)
    return self

  def signal(self, signal:AMDSignal, value:sint=0):
    # NOTE: this needs an EOP buffer on the queue or it will NULL pointer
    self.release_mem(signal.value_addr, value, amd_gpu.data_sel__mec_release_mem__send_32_bit_low,
                     amd_gpu.int_sel__mec_release_mem__send_interrupt_after_write_confirm, cache_flush=True)

    # if not AMDDevice.driverless and (dev:=signal.timeline_for_device) is not None:
    #   self.release_mem(dev.queue_event_mailbox_ptr, dev.queue_event.event_id, amd_gpu.data_sel__mec_release_mem__send_32_bit_low,
    #                    amd_gpu.int_sel__mec_release_mem__send_interrupt_after_write_confirm, ctxid=dev.queue_event.event_id)
    return self

  def bind(self, dev:AMDDevice):
    self.binded_device = dev
    self.hw_page = dev.allocator.alloc(len(self._q) * 4, BufferSpec(cpu_access=True, nolru=True, uncached=True))
    hw_view = to_mv(self.hw_page.va_addr, self.hw_page.size).cast("I")
    for i, value in enumerate(self._q): hw_view[i] = value

    self.indirect_cmd = [amd_gpu.PACKET3(amd_gpu.PACKET3_INDIRECT_BUFFER, 2), *data64_le(self.hw_page.va_addr),
                         len(self._q) | amd_gpu.INDIRECT_BUFFER_VALID]
    self._q = hw_view # type: ignore
    return self

  def _submit(self, dev:AMDDevice):
    cmds = self.indirect_cmd if dev == self.binded_device else self._q

    for i, value in enumerate(cmds): dev.compute_queue.ring[(dev.compute_queue.put_value + i) % len(dev.compute_queue.ring)] = value

    dev.compute_queue.put_value += len(cmds)
    print("will send")
    dev.compute_queue.write_ptr[0] = dev.compute_queue.put_value
    dev.compute_queue.doorbell[0] = dev.compute_queue.put_value

SDMA_MAX_COPY_SIZE = 0x400000
class AMDCopyQueue(HWQueue):
  def __init__(self):
    self.internal_cmd_sizes = []
    super().__init__()

  def q(self, *arr):
    super().q(*arr)
    self.internal_cmd_sizes.append(len(arr))

  def copy(self, dest:sint, src:sint, copy_size:int):
    copied, copy_commands = 0, (copy_size + SDMA_MAX_COPY_SIZE - 1) // SDMA_MAX_COPY_SIZE

    for _ in range(copy_commands):
      step_copy_size = min(copy_size - copied, SDMA_MAX_COPY_SIZE)

      self.q(amd_gpu.SDMA_OP_COPY | amd_gpu.SDMA_PKT_COPY_LINEAR_HEADER_SUB_OP(amd_gpu.SDMA_SUBOP_COPY_LINEAR),
        amd_gpu.SDMA_PKT_COPY_LINEAR_COUNT_COUNT(step_copy_size - 1), 0, *data64_le(src + copied), *data64_le(dest + copied))

      copied += step_copy_size
    return self

  def signal(self, signal:AMDSignal, value:sint=0):
    self.q(amd_gpu.SDMA_OP_FENCE | amd_gpu.SDMA_PKT_FENCE_HEADER_MTYPE(3), *data64_le(signal.value_addr), value)
    # self.q(amd_gpu.SDMA_OP_GCR_REQ, 0, amd_gpu.SDMA_GCR_GLK_WB | amd_gpu.SDMA_GCR_GL2_WB, 0, 0)

    # if not AMDDevice.driverless and (dev:=signal.timeline_for_device) is not None:
    #   self.q(amd_gpu.SDMA_OP_FENCE | amd_gpu.SDMA_PKT_FENCE_HEADER_MTYPE(3), *data64_le(dev.queue_event_mailbox_ptr), dev.queue_event.event_id)
    #   self.q(amd_gpu.SDMA_OP_TRAP, amd_gpu.SDMA_PKT_TRAP_INT_CONTEXT_INT_CONTEXT(dev.queue_event.event_id))

    return self

  def wait(self, signal:AMDSignal, value:sint=0):
    self.q(amd_gpu.SDMA_OP_POLL_REGMEM | amd_gpu.SDMA_PKT_POLL_REGMEM_HEADER_FUNC(WAIT_REG_MEM_FUNCTION_GEQ) | \
           amd_gpu.SDMA_PKT_POLL_REGMEM_HEADER_MEM_POLL(1), *data64_le(signal.value_addr), value, 0xffffffff,
           amd_gpu.SDMA_PKT_POLL_REGMEM_DW5_INTERVAL(0x04) | amd_gpu.SDMA_PKT_POLL_REGMEM_DW5_RETRY_COUNT(0xfff))
    return self

  def timestamp(self, signal:AMDSignal):
    self.q(amd_gpu.SDMA_OP_TIMESTAMP | amd_gpu.SDMA_PKT_TIMESTAMP_GET_HEADER_SUB_OP(amd_gpu.SDMA_SUBOP_TIMESTAMP_GET_GLOBAL),
           *data64_le(signal.timestamp_addr))
    return self

  def _submit(self, dev:AMDDevice):
    if dev.sdma_queue.put_value - dev.sdma_queue.read_ptr[0] > dev.sdma_queue.ring.nbytes: raise RuntimeError("SDMA queue overrun")

    tail_blit_dword = 0
    for cmdsz in self.internal_cmd_sizes:
      if (tail_blit_dword + cmdsz) * 4 >= dev.sdma_queue.ring.nbytes - dev.sdma_queue.put_value % dev.sdma_queue.ring.nbytes: break
      tail_blit_dword += cmdsz

    start_idx = (dev.sdma_queue.put_value % dev.sdma_queue.ring.nbytes) // 4
    dev.sdma_queue.ring[start_idx : start_idx + tail_blit_dword] = array.array('I', self._q[:tail_blit_dword])
    dev.sdma_queue.put_value += tail_blit_dword * 4

    if (rem_packet_cnt := len(self._q) - tail_blit_dword) > 0:
      zero_fill = dev.sdma_queue.ring.nbytes - dev.sdma_queue.put_value % dev.sdma_queue.ring.nbytes
      ctypes.memset(mv_address(dev.sdma_queue.ring) + (dev.sdma_queue.put_value % dev.sdma_queue.ring.nbytes), 0, zero_fill)
      dev.sdma_queue.put_value += zero_fill

      dev.sdma_queue.ring[0:rem_packet_cnt] = array.array('I', self._q[tail_blit_dword:])
      dev.sdma_queue.put_value += rem_packet_cnt * 4

    dev.sdma_queue.write_ptr[0] = dev.sdma_queue.put_value
    dev.sdma_queue.doorbell[0] = dev.sdma_queue.put_value

class AMDProgram(HCQProgram):
  def __init__(self, dev:AMDDevice, name:str, lib:bytes):
    # TODO; this API needs the type signature of the function and global_size/local_size
    self.dev: AMDDevice = dev
    self.name, self.lib = name, lib
    image, sections, _ = elf_loader(self.lib)
    self.lib_gpu = self.dev.allocator.alloc(round_up(image.nbytes, 0x1000), BufferSpec(cpu_access=True, nolru=True))
    ctypes.memmove(self.lib_gpu.va_addr, mv_address(image), image.nbytes)

    entry_point = min(sh.header.sh_addr for sh in sections if sh.header.sh_type == libc.SHT_PROGBITS and sh.header.sh_flags & libc.SHF_ALLOC)
    self.group_segment_size = image[entry_point:entry_point+4].cast("I")[0]
    self.private_segment_size = image[entry_point+4:entry_point+8].cast("I")[0]
    self.kernargs_segment_size = image[entry_point+8:entry_point+12].cast("I")[0]

    lds_size = ((self.group_segment_size + 511) // 512) & 0x1FF
    if lds_size > (self.dev.dev_iface.properties['lds_size_in_kb'] * 1024) // 512: raise RuntimeError("Too many resources requsted: group_segment_size")
    if self.private_segment_size > self.dev.max_private_segment_size: raise RuntimeError("Too many resources requsted: private_segment_size")

    code = hsa.amd_kernel_code_t.from_address(self.lib_gpu.va_addr + entry_point) # NOTE: this is wrong, it's not this object
    assert code.kernel_code_properties & 0x400 == 0x400 # ENABLE_WAVEFRONT_SIZE32

    # Set rsrc1.priv=1 on gfx11 to workaround cwsr.
    self.rsrc1: int = code.compute_pgm_rsrc1 | ((1 << 20) if 110000 <= self.dev.target < 120000 else 0)
    self.rsrc2: int = code.compute_pgm_rsrc2 | (lds_size << 15)
    self.prog_addr: int = self.lib_gpu.va_addr + entry_point + code.kernel_code_entry_byte_offset

    # Some programs use hsa_kernel_dispatch_packet_t to read workgroup sizes during execution.
    # The packet is represented as a pointer and set up in SGPRs. Space for the packet is allocated as part of the kernel arguments.
    self.enable_dispatch_ptr: int = code.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR
    self.enable_private_segment_sgpr: int = code.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER
    additional_alloc_sz = ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t) if self.enable_dispatch_ptr else 0

    super().__init__(CLikeArgsState, self.dev, self.name, kernargs_alloc_size=self.kernargs_segment_size+additional_alloc_sz)

  def __del__(self):
    if hasattr(self, 'lib_gpu'): self.dev.allocator.free(self.lib_gpu, self.lib_gpu.size, BufferSpec(cpu_access=True, nolru=True))

class AMDDriverAllocator(HCQAllocator['AMDDevice']):
  def __init__(self, dev:AMDDevice): super().__init__(dev, batch_size=SDMA_MAX_COPY_SIZE)

  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    if options.host: return self.dev.dev_iface.alloc(size, host=True)
    if options.cpu_access and options.uncached: return self.dev.dev_iface.alloc(size, uncached=True, cpu_access=options.cpu_access)
    return self.dev.dev_iface.alloc(size, cpu_access=options.cpu_access)

  def _free(self, opaque, options:BufferSpec):
    self.dev.synchronize()
    self.dev.dev_iface.free(opaque)

  def map(self, buf:HCQBuffer): self.dev.dev_iface.map(buf._base if buf._base is not None else buf)

MAP_FIXED, MAP_NORESERVE, MAP_LOCKED = 0x10, 0x400, 0x2000

@dataclass
class AMDQueueDesc:
  ring: memoryview
  read_ptr: memoryview
  write_ptr: memoryview
  doorbell: memoryview
  put_value: int = 0

class KFDIface:
  kfd:int = -1

  @staticmethod
  def is_usable_gpu(gpu_id):
    with contextlib.suppress(OSError): return int(pathlib.Path(gpu_id).read_text()) != 0
    return False

  def __init__(self, dev, device_id):
    self.dev = dev

    if KFDIface.kfd == -1:
      KFDIface.kfd = os.open("/dev/kfd", os.O_RDWR)
      gpus = [g.parent for g in pathlib.Path("/sys/devices/virtual/kfd/kfd/topology/nodes").glob("*/gpu_id") if KFDIface.is_usable_gpu(g)]
      gpus = sorted(gpus, key=lambda x: int(x.name.split('/')[-1]))
      visible_devices = [int(x) for x in (getenv('VISIBLE_DEVICES', getenv('HIP_VISIBLE_DEVICES', ''))).split(',') if x.strip()]
      KFDIface.gpus = [gpus[x] for x in visible_devices] if visible_devices else gpus

    if device_id >= len(KFDIface.gpus): raise RuntimeError(f"No device found for {device_id}. Requesting more devices than the system has?")

    with open(f"{KFDIface.gpus[device_id]}/gpu_id", "r") as f: self.gpu_id = int(f.read())
    with open(f"{KFDIface.gpus[device_id]}/properties", "r") as f: self.properties = {line.split()[0]: int(line.split()[1]) for line in f}
    self.drm_fd = os.open(f"/dev/dri/renderD{self.properties['drm_render_minor']}", os.O_RDWR)

    kfd.AMDKFD_IOC_ACQUIRE_VM(KFDIface.kfd, drm_fd=self.drm_fd, gpu_id=self.gpu_id)

    # Set these for our device.
    # self.dev.queue_event = kfd.AMDKFD_IOC_CREATE_EVENT(KFDIface.kfd, event_type=kfd.KFD_IOC_EVENT_SIGNAL, auto_reset=1)
    # self.dev.queue_event_mailbox_ptr = AMDDevice.event_page.va_addr + self.queue_event.event_slot_index * 8
    # self.dev.queue_event_arr = (kfd.struct_kfd_event_data)(event_id=self.queue_event.event_id)
    # self.queue_event_arr_ptr = ctypes.addressof(self.queue_event_arr)

    # self.mem_fault_event = kfd.AMDKFD_IOC_CREATE_EVENT(KFDIface.kfd, event_type=kfd.KFD_IOC_EVENT_MEMORY)
    # self.hw_fault_event = kfd.AMDKFD_IOC_CREATE_EVENT(KFDIface.kfd, event_type=kfd.KFD_IOC_EVENT_HW_EXCEPTION)

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False) -> HCQBuffer:
    flags = kfd.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | kfd.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE

    if uncached: flags |= kfd.KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | kfd.KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED | kfd.KFD_IOC_ALLOC_MEM_FLAGS_GTT
    else: flags |= (kfd.KFD_IOC_ALLOC_MEM_FLAGS_USERPTR if host else kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)

    if cpu_access or host: flags |= kfd.KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC

    if host: buf = addr = libc.mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | mmap.MAP_ANONYMOUS, -1, 0)
    else: buf, addr = 0, libc.mmap(0, size, 0, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_NORESERVE, -1, 0)
    assert addr != 0xffffffffffffffff

    try: mem = kfd.AMDKFD_IOC_ALLOC_MEMORY_OF_GPU(self.kfd, va_addr=addr, size=size, base=addr, length=size, gpu_id=self.gpu_id,
                                                  flags=flags, mmap_offset=buf)
    except OSError as e:
      if e.errno == errno.EINVAL and (flags & kfd.KFD_IOC_ALLOC_MEM_FLAGS_VRAM) and cpu_access:
        raise MemoryError("Cannot allocate host-visible VRAM. Ensure the resizable BAR option is enabled on your system.") from e
      if e.errno == errno.ENOMEM: raise MemoryError("Cannot allocate memory: no memory is available.") from e
      raise

    if not host:
      buf = libc.mmap(mem.va_addr, mem.size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | MAP_FIXED, self.drm_fd, mem.mmap_offset)
      assert addr == buf == mem.va_addr

    self.map(hcqbuf:=HCQBuffer(mem.va_addr, mem.size, meta=mem))
    return hcqbuf

  def free(self, mem):
    if len(gpus:=getattr(mem.meta, "mapped_gpu_ids", [])):
      c_gpus = (ctypes.c_int32 * len(gpus))(*gpus)
      stm = kfd.AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU(self.kfd, handle=mem.meta.handle, device_ids_array_ptr=ctypes.addressof(c_gpus), n_devices=len(gpus))
      assert stm.n_success == len(gpus)
    if mem.va_addr: libc.munmap(mem.va_addr, mem.size)
    kfd.AMDKFD_IOC_FREE_MEMORY_OF_GPU(self.kfd, handle=mem.meta.handle)

  def map(self, mem):
    if self.gpu_id in getattr(mem.meta, "mapped_gpu_ids", []): return
    mem.meta.__setattr__("mapped_gpu_ids", getattr(mem.meta, "mapped_gpu_ids", []) + [self.gpu_id])
    c_gpus = (ctypes.c_int32 * len(mem.meta.mapped_gpu_ids))(*mem.meta.mapped_gpu_ids)
    # print("MAPPING", hex(mem.va_addr), hex(mem.size), hex(mem.handle), self.gpu_id)
    stm = kfd.AMDKFD_IOC_MAP_MEMORY_TO_GPU(self.kfd, handle=mem.meta.handle, device_ids_array_ptr=ctypes.addressof(c_gpus),
                                           n_devices=len(mem.meta.mapped_gpu_ids))
    assert stm.n_success == len(mem.meta.mapped_gpu_ids)

  def create_queue(self, queue_type, ring, gart, eop_buffer=None, ctl_stack_size=0, ctx_save_restore_size=0, debug_memory_size=0):
    cwsr_ctx = self.alloc(round_up(ctx_save_restore_size + debug_memory_size, mmap.PAGESIZE)) if ctx_save_restore_size else None
    queue = kfd.AMDKFD_IOC_CREATE_QUEUE(KFDIface.kfd, ring_base_address=ring.va_addr, ring_size=ring.size, gpu_id=self.gpu_id,
      queue_type=queue_type, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE, queue_priority=kfd.KFD_MAX_QUEUE_PRIORITY,
      eop_buffer_address=eop_buffer.va_addr if eop_buffer else 0, eop_buffer_size=eop_buffer.size if eop_buffer else 0, ctl_stack_size=ctl_stack_size,
      ctx_save_restore_address=cwsr_ctx.va_addr if cwsr_ctx else 0, ctx_save_restore_size=ctx_save_restore_size,
      write_pointer_address=gart.va_addr, read_pointer_address=gart.va_addr + 8)

    if not hasattr(self, 'doorbells'):
      self.doorbells_base = queue.doorbell_offset & (~0x1fff) # doorbell is two pages
      self.doorbells = libc.mmap(0, 0x2000, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, KFDIface.kfd, self.doorbells_base)

    return AMDQueueDesc(ring=to_mv(ring.va_addr, ring.size).cast("I"),
                        read_ptr=to_mv(queue.read_pointer_address, 8).cast("Q"), write_ptr=to_mv(queue.write_pointer_address, 8).cast("Q"),
                        doorbell=to_mv(self.doorbells + queue.doorbell_offset - self.doorbells_base, 8).cast("Q"))

  def sleep(self, tm:int): pass # kfd.AMDKFD_IOC_WAIT_EVENTS(KFDIface.kfd, events_ptr=self.queue_event_arr_ptr, num_events=1, wait_for_all=1, timeout=tm)

class VFIOIface:
  vfio_fd:int = -1
  iommu_set:bool = False
  gpus:List[Any] = []

  def __init__(self, dev, dev_id):
    self.dev = dev

    if VFIOIface.vfio_fd == -1:
      subprocess.run(['modprobe', 'vfio_pci'], capture_output=True, text=True, check=True)
      with open("/sys/module/vfio/parameters/enable_unsafe_noiommu_mode", 'w') as f: f.write("1")

      VFIOIface.vfio_fd = os.open("/dev/vfio/vfio", os.O_RDWR)
      assert vfio.VFIO_CHECK_EXTENSION(VFIOIface.vfio_fd, vfio.VFIO_NOIOMMU_IOMMU), "VFIO does not support IOMMU"

      libpciaccess.pci_system_init()
      pci_iter = libpciaccess.pci_id_match_iterator_create(None)
      while pcidev:=libpciaccess.pci_device_next(pci_iter):
        if pcidev.contents.vendor_id == 0x1002 and pcidev.contents.device_id == 0x744c: VFIOIface.gpus.append(pcidev.contents)

      # TODO: visible_devices should be handled layer above this?
      visible_devices = [int(x) for x in (getenv('VISIBLE_DEVICES', getenv('HIP_VISIBLE_DEVICES', ''))).split(',') if x.strip()]
      VFIOIface.gpus = [VFIOIface.gpus[x] for x in visible_devices] if visible_devices else VFIOIface.gpus

    self.pcidev = VFIOIface.gpus[dev_id]
    self.pcibus = f"{self.pcidev.domain_16:04x}:{self.pcidev.bus:02x}:{self.pcidev.dev:02x}.{self.pcidev.func:d}"

    if os.path.exists(f"/sys/bus/pci/devices/{self.pcibus}/driver"):
      with open(f"/sys/bus/pci/devices/{self.pcibus}/driver/unbind", 'w') as f: f.write(self.pcibus)
    with open(f"/sys/bus/pci/devices/{self.pcibus}/resource0_resize", 'w') as f: f.write("15")

    libpciaccess.pci_device_probe(ctypes.byref(self.pcidev))

    if getenv("VFIO", 1):
      with open(f"/sys/bus/pci/devices/{self.pcibus}/driver_override", 'w') as f: f.write("vfio-pci")
      with open(f"/sys/bus/pci/drivers_probe", 'w') as f: f.write(self.pcibus)

      iommu_group = os.readlink(f"/sys/bus/pci/devices/{self.pcibus}/iommu_group").split('/')[-1]

      self.vfio_group = os.open(f"/dev/vfio/noiommu-{iommu_group}", os.O_RDWR)
      vfio.VFIO_GROUP_SET_CONTAINER(self.vfio_group, ctypes.c_int(VFIOIface.vfio_fd))

      if not VFIOIface.iommu_set:
        vfio.VFIO_SET_IOMMU(VFIOIface.vfio_fd, vfio.VFIO_NOIOMMU_IOMMU)
        VFIOIface.iommu_set = True
      self.vfio_dev = vfio.VFIO_GROUP_GET_DEVICE_FD(self.vfio_group, (ctypes.c_char * (len(self.pcibus) + 1))(*bytearray(self.pcibus.encode() + b'\0')))

      self.irq_fd = os.eventfd(0, 0)
      self.irq_poller = select.poll()
      self.irq_poller.register(self.irq_fd, select.POLLIN)

      # irqs = vfio.struct_vfio_irq_set(index=vfio.VFIO_PCI_MSI_IRQ_INDEX, flags=vfio.VFIO_IRQ_SET_DATA_EVENTFD|vfio.VFIO_IRQ_SET_ACTION_TRIGGER,
      #   argsz=ctypes.sizeof(vfio.struct_vfio_irq_set), count=1, data=(ctypes.c_int * 1)(self.irq_fd))
      # vfio.VFIO_DEVICE_SET_IRQS(self.vfio_dev, irqs)
    else: libpciaccess.pci_device_enable(ctypes.byref(self.pcidev))

    self.fds = {bar: os.open(f"/sys/bus/pci/devices/{self.pcibus}/resource{bar}", os.O_RDWR | os.O_SYNC) for bar in [0, 2, 5]}

    self.adev = AMDev(self.pcidev, self._map_pci_range(0), dbell:=self._map_pci_range(2).cast('Q'), self._map_pci_range(5).cast('I'))
    self.doorbell_cpu_addr = mv_address(dbell)

    # TODO: think of a way to handle this
    self.properties = {'simd_count': 192, 'simd_per_cu': 2, 'max_waves_per_simd': 16, 'gfx_target_version': 110000, 'max_slots_scratch_cu': 32,
                       'array_count': 12, 'simd_arrays_per_engine': 2, 'lds_size_in_kb': 64}

  def _map_pci_range(self, bar, off=0, addr=0, size=None):
    if getenv("VFIO", 1):
      vfio.VFIO_DEVICE_GET_REGION_INFO(self.vfio_dev, reg:=vfio.struct_vfio_region_info(argsz=ctypes.sizeof(vfio.struct_vfio_region_info), index=bar))
      fd, sz, off = self.vfio_dev, size or reg.size, reg.offset + off
    else: fd, sz, off = self.fds[bar], size or self.pcidev.regions[bar].size, off

    x = to_mv(z:=libc.mmap(addr, sz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if addr else 0), fd, off), sz)
    # print("mmp", bar, hex(z), hex(sz))
    return x

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False):
    if host:
      vaddr = self.adev.mm.alloc_vaddr(size, align=mmap.PAGESIZE)
      va = libc.mmap(vaddr, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | MAP_LOCKED | MAP_FIXED, -1, 0)

      # Read pagemap to get the physical address of each page. The pages are locked.
      with open("/proc/self/pagemap", "rb") as f:
        for off in range(0, size, mmap.PAGESIZE):
          f.seek(((va + off) // mmap.PAGESIZE) * 8)
          pt_entry = struct.unpack("Q", f.read(8))[0] & ((1 << 55) - 1)
          self.adev.mm.map_range(vaddr=vaddr + off, size=mmap.PAGESIZE, paddr=pt_entry * mmap.PAGESIZE, system=True, snooped=True, uncached=True)
      return HCQBuffer(vaddr, size, meta=(self.dev, [self.dev], None))

    vm = self.adev.mm.valloc(size:=round_up(size, 0x1000), uncached=uncached, contigous=cpu_access)
    if cpu_access:
      # print("want")
      self._map_pci_range(bar=0, off=vm.paddr, addr=vm.va_addr, size=vm.size)
    return HCQBuffer(vm.va_addr, size, meta=(self.dev, [self.dev], vm))

  def free(self, mem):
    if mem.meta[2] is not None:
      for dev in mem.meta[1][1:]: dev.dev_iface.adev.mm.unmap_range(mem.va_addr, mem.size, free_paddrs=False)
      self.adev.mm.vfree(mem.meta[2])

  def map(self, mem):
    if mem.meta[0] == self.dev or self.dev in mem.meta[1]: return
    mem.meta[1].append(self.dev)
    self.adev.mm.map_from(mem.va_addr, mem.size, mem.meta[0].dev_iface.adev)

  def create_queue(self, queue_type, ring, gart, eop_buffer=None, ctl_stack_size=0, ctx_save_restore_size=0, debug_memory_size=0):
    if queue_type == kfd.KFD_IOC_QUEUE_TYPE_SDMA:
      self.adev.sdma.setup_ring(ring_addr=ring.va_addr, ring_size=ring.size, rptr_addr=gart.va_addr, wptr_addr=gart.va_addr+0x10,
                                doorbell=(doorbell_index:=am.AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0), pipe=0, queue=0)
    else:
      self.adev.gfx.setup_ring(ring_addr=ring.va_addr, ring_size=ring.size, rptr_addr=gart.va_addr, wptr_addr=gart.va_addr+0x10,
        eop_addr=eop_buffer.va_addr, eop_size=eop_buffer.size, doorbell=(doorbell_index:=am.AMDGPU_NAVI10_DOORBELL_MEC_RING0), pipe=0, queue=0)

    return AMDQueueDesc(ring=to_mv(ring.va_addr, ring.size).cast("I"), doorbell=to_mv(self.doorbell_cpu_addr + doorbell_index * 8, 8).cast("Q"),
                        read_ptr=to_mv(gart.va_addr, 8).cast("Q"), write_ptr=to_mv(gart.va_addr+0x10, 8).cast("Q"))

  def sleep(self, timeout):
    if getenv("VFIO", 1):
      x = self.irq_poller.poll(timeout)
      if len(x): os.read(self.irq_fd, 1024)

class USBTrackedMemoryView:
  def __init__(self, bar_info, usb, elsz=1):
    (self.addr, self.size), self.usb = bar_info, usb
    self.elsz = elsz
    assert self.elsz <= 4 or self.elsz % 4 == 0

  def _read(self, index):
    val = 0
    for off in range(max(1, self.elsz // 4)):
      part = self.usb.pcie_mem_req(self.addr + index * self.elsz + off * 4, None, min(4, self.elsz))
      val += part << (32 * off)
    return val

  def _write(self, index, value):
    for off in range(max(1, self.elsz // 4)):
      self.usb.pcie_mem_req(self.addr + index * self.elsz + off * 4, (value >> (32 * off)) & 0xFFFFFFFF, min(4, self.elsz))

  def __getitem__(self, index): return self._read(index)
  def __setitem__(self, index, value): return self._write(index, value)
  def cast(self, new_type, **kwargs): return self
  def offset(self, offset=0, elsz=None): return USBTrackedMemoryView((self.addr + offset, self.size - offset), self.usb, elsz or self.elsz)

  def copyin(self, mv):
    print("copyin", len(mv))
    x = mv.cast({1:'B', 2:'H', 4:'I', 8:'Q'}[self.elsz])
    for i in range(len(x)): self._write(i, x[i])

  def copyout(self, mv):
    x = mv.cast({1:'B', 2:'H', 4:'I', 8:'Q'}[self.elsz])
    for i in range(len(x)): x[i] = self._read(i)

  @property
  def nbytes(self): return self.size
  def __len__(self): return self.size // self.elsz
  def __repr__(self): return "USBTrackedMemoryView"

class USBIface(VFIOIface):
  iommu_set:bool = False
  gpus:List[Any] = []

  def __init__(self, dev, dev_id):
    self.dev = dev
    self.usb = Asm236x("/dev/sg0")

    # setup pci switch
    self.usb.pcie_cfg_req(pci.PCI_SUBORDINATE_BUS, bus=0, dev=0, fn=0, value=3, size=1)
    self.usb.pcie_cfg_req(pci.PCI_SECONDARY_BUS, bus=0, dev=0, fn=0, value=1, size=1)
    self.usb.pcie_cfg_req(pci.PCI_PRIMARY_BUS, bus=0, dev=0, fn=0, value=0, size=1)

    self.usb.pcie_cfg_req(pci.PCI_MEMORY_BASE, bus=0, dev=0, fn=0, value=0x1000, size=2)
    self.usb.pcie_cfg_req(pci.PCI_MEMORY_LIMIT, bus=0, dev=0, fn=0, value=0x2000, size=2)

    self.usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_BASE, bus=0, dev=0, fn=0, value=0x4000, size=2)
    self.usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_LIMIT, bus=0, dev=0, fn=0, value=0xffff, size=2)

    for bus in [1, 2]:
      self.usb.pcie_cfg_req(pci.PCI_MEMORY_BASE, bus=bus, dev=0, fn=0, value=0x1000, size=2)
      self.usb.pcie_cfg_req(pci.PCI_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0x2000, size=2)

      self.usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_BASE, bus=bus, dev=0, fn=0, value=0x4000, size=2)
      self.usb.pcie_cfg_req(pci.PCI_PREF_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=0xffff, size=2)

      self.usb.pcie_cfg_req(pci.PCI_SUBORDINATE_BUS, bus=bus, dev=0, fn=0, value=3, size=1)
      self.usb.pcie_cfg_req(pci.PCI_SECONDARY_BUS, bus=bus, dev=0, fn=0, value=bus+1, size=1)
      self.usb.pcie_cfg_req(pci.PCI_PRIMARY_BUS, bus=bus, dev=0, fn=0, value=bus-1, size=1)

      self.usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_BUS_RESET, size=1)
      time.sleep(1)
      self.usb.pcie_cfg_req(pci.PCI_BRIDGE_CONTROL, bus=bus, dev=0, fn=0, value=pci.PCI_BRIDGE_CTL_PARITY|pci.PCI_BRIDGE_CTL_SERR, size=1)      
      self.usb.pcie_cfg_req(pci.PCI_COMMAND, bus=bus, dev=0, fn=0, value=pci.PCI_COMMAND_IO | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER, size=1)

    bar_next_addr, bar_off, self.bars = [0x10000000, 0x40000000], 0, {}
    for bar_id in range(4):
      bar_cfg = self.usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=3, dev=0, fn=0, size=4)
      if bar_cfg & pci.PCI_BASE_ADDRESS_SPACE == pci.PCI_BASE_ADDRESS_SPACE_MEMORY:
        self.usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=3, dev=0, fn=0, value=0xffffffff, size=4)
        bar_size = 0xffffffff - (self.usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=3, dev=0, fn=0, size=4) & 0xFFFFFFF0) + 1
        if bar_id in {0, 1, 3}:
          is_pref = int(bool(bar_cfg & pci.PCI_BASE_ADDRESS_MEM_PREFETCH))
          self.usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=3, dev=0, fn=0, value=bar_next_addr[is_pref], size=4)
          self.bars[bar_id] = (bar_next_addr[is_pref], bar_size)
          bar_next_addr[is_pref] += round_up(bar_size, 2 << 20)
          print(bar_id, hex(self.usb.pcie_cfg_req(pci.PCI_BASE_ADDRESS_0 + bar_off, bus=3, dev=0, fn=0, size=4)))
      print(bar_id, hex(bar_cfg))
      bar_off += 8 if bar_cfg & pci.PCI_BASE_ADDRESS_MEM_TYPE_64 else 4

    cap_ptr = self.usb.pcie_cfg_req(0x34, bus=3, dev=0, fn=0, value=None, size=1)
    caps = {}
    while True:
        cap_0 = self.usb.pcie_cfg_req(cap_ptr, bus=3, dev=0, fn=0, value=None, size=1)
        cap_nxt = self.usb.pcie_cfg_req(cap_ptr+1, bus=3, dev=0, fn=0, value=None, size=1)
        cap_len = self.usb.pcie_cfg_req(cap_ptr+2, bus=3, dev=0, fn=0, value=None, size=1)
        # print(cap_0, cap_nxt, cap_len)
        caps[cap_0] = cap_ptr
        if cap_nxt == 0: break
        cap_ptr = cap_nxt

    print(caps)

    pm_state = self.usb.pcie_cfg_req(caps[0x1]+4, bus=3, dev=0, fn=0, value=None, size=1)
    self.usb.pcie_cfg_req(caps[0x1]+4, bus=3, dev=0, fn=0, value=0x0, size=1)
    # print("PM cap now", pm_state)

    self.usb.pcie_cfg_req(pci.PCI_COMMAND, bus=3, dev=0, fn=0, value=pci.PCI_COMMAND_MEMORY, size=1)

    # print(caps)

    # print(self.bars)

    aa, size = self.bars[3]
    # print(hex(aa))
    # print(self.usb.pcie_mem_req(aa, None, size=4))
    # exit(0)

    # print(self.bars)
    self.vram_bar = USBTrackedMemoryView(self.bars[0], self.usb, elsz=1)
    self.doorbell_bar = USBTrackedMemoryView(self.bars[1], self.usb, elsz=8)
    self.mmio_bar = USBTrackedMemoryView(self.bars[3], self.usb, elsz=4)
    self.adev = AMDev(None, self.vram_bar, self.doorbell_bar, self.mmio_bar)
    # print(bars)

    self.properties = {'simd_count': 192, 'simd_per_cu': 2, 'max_waves_per_simd': 16, 'gfx_target_version': 110000, 'max_slots_scratch_cu': 32,
                       'array_count': 12, 'simd_arrays_per_engine': 2, 'lds_size_in_kb': 64}

  # def _tracked_view(self, va):
  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, zero=False):
    # if host:
    #   vaddr = self.adev.mm.alloc_vaddr(size, align=mmap.PAGESIZE)
    #   va = libc.mmap(vaddr, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | MAP_LOCKED | MAP_FIXED, -1, 0)

    #   # Read pagemap to get the physical address of each page. The pages are locked.
    #   with open("/proc/self/pagemap", "rb") as f:
    #     for off in range(0, size, mmap.PAGESIZE):
    #       f.seek(((va + off) // mmap.PAGESIZE) * 8)
    #       pt_entry = struct.unpack("Q", f.read(8))[0] & ((1 << 55) - 1)
    #       self.adev.mm.map_range(vaddr=vaddr + off, size=mmap.PAGESIZE, paddr=pt_entry * mmap.PAGESIZE, system=True, snooped=True, uncached=True)
    #   return HCQBuffer(vaddr, size, meta=(self.dev, [self.dev], None))

    vm = self.adev.mm.valloc(size:=round_up(size, 0x1000), uncached=uncached, contigous=True, zero=zero)
    tracked_mv = self.vram_bar.offset(vm.paddr, elsz=1)
    # assert tracked_mv.size >= 0, f"{vm.paddr} {tracked_mv.size}, {tracked_mv.}"
    return HCQBuffer(vm.va_addr, size, meta=(self.dev, [self.dev], vm, tracked_mv))

  def create_queue(self, queue_type, ring, gart, eop_buffer=None, ctl_stack_size=0, ctx_save_restore_size=0, debug_memory_size=0):
    # print(ring.va_addr)

    if queue_type == kfd.KFD_IOC_QUEUE_TYPE_SDMA:
      doorbell_index = 0
      pass
      # self.adev.sdma.setup_ring(ring_addr=ring.va_addr, ring_size=ring.size, rptr_addr=gart.va_addr, wptr_addr=gart.va_addr+0x10,
      #                           doorbell=(doorbell_index:=am.AMDGPU_NAVI10_DOORBELL_sDMA_ENGINE0), pipe=0, queue=0)
    else:
      self.adev.gfx.setup_ring(ring_addr=ring.va_addr, ring_size=ring.size, rptr_addr=gart.va_addr, wptr_addr=gart.va_addr+0x10,
        eop_addr=eop_buffer.va_addr, eop_size=eop_buffer.size, doorbell=(doorbell_index:=am.AMDGPU_NAVI10_DOORBELL_MEC_RING0), pipe=0, queue=0)

    return AMDQueueDesc(ring=ring.meta[3].offset(elsz=4), doorbell=self.doorbell_bar.offset(doorbell_index * 8, elsz=8),
                        read_ptr=gart.meta[3].offset(elsz=8), write_ptr=gart.meta[3].offset(offset=0x10, elsz=8))

class AMDDevice(HCQCompiled):
  driverless:bool = False
  event_page:Any = None  # TODO: fix types in kfd, Optional[kfd.struct_kfd_ioctl_alloc_memory_of_gpu_args]
  signals_page:Any = None
  signals_pool:List[int] = []

  def __init__(self, device:str=""):
    AMDDevice.driverless = not os.path.isdir('/sys/module/amdgpu/') or bool(getenv("AMD_DRIVERLESS", 0))

    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    # self.dev_iface = VFIOIface(self, self.device_id) if AMDDevice.driverless else KFDIface(self, self.device_id)
    self.dev_iface = USBIface(self, self.device_id)

    self.target = int(self.dev_iface.properties['gfx_target_version'])
    self.arch = "gfx%d%x%x" % (self.target // 10000, (self.target // 100) % 100, self.target % 100)
    if self.target < 100300 or self.target >= 120000: raise RuntimeError(f"Unsupported arch: {self.arch}")

    # TODO: think of moving this out.
    self.signals_page = self.dev_iface.alloc(0x1000, uncached=True, cpu_access=True, zero=True)
    self.signals_pool = [self.signals_page.va_addr + off for off in range(0, self.signals_page.size, 16)]
    for dev in self.devices: 
      self.dev_iface.map(dev.signals_page)
      dev.dev_iface.map(self.signals_page)

    # if AMDDevice.signals_page is None:
    #   AMDDevice.signals_page = self.dev_iface.alloc(16 * 65536, host=True, uncached=True, cpu_access=True)
    #   AMDDevice.signals_pool = [self.signals_page.va_addr + off for off in range(0, AMDDevice.signals_page.size, 16)]
    # else: self.dev_iface.map(AMDDevice.signals_page)

    # Scratch setup
    max_cu_id = self.dev_iface.properties['simd_count'] // self.dev_iface.properties['simd_per_cu'] - 1
    max_wave_id = self.dev_iface.properties['max_waves_per_simd'] * self.dev_iface.properties['simd_per_cu'] - 1
    self.max_private_segment_size = 4096
    # <gfx103 requires alignment of 1024, >=gfx11 requires 256
    wave_scratch_len = round_up(((max_wave_id + 1) * self.max_private_segment_size), 256 if self.target >= 110000 else 1024)
    self.scratch_len = (max_cu_id + 1) * self.dev_iface.properties['max_slots_scratch_cu'] * wave_scratch_len
    self.scratch = self.dev_iface.alloc(0x1000)
    # self.scratch = self.dev_iface.alloc(self.scratch_len)
    self.has_scratch_base_registers = self.target >= 110000
    engines = self.dev_iface.properties['array_count'] // self.dev_iface.properties['simd_arrays_per_engine']
    waves = wave_scratch_len // (256 if self.target >= 110000 else 1024)
    # >=gfx11 wavesize is per SE
    wavesize = self.scratch_len // ((wave_scratch_len * engines) if self.target >= 110000 else wave_scratch_len)
    self.tmpring_size = waves << 12 | wavesize

    # https://gitlab.freedesktop.org/agd5f/linux/-/blob/a1fc9f584c4aaf8bc1ebfa459fc57a3f26a290d8/drivers/gpu/drm/amd/amdkfd/kfd_queue.c#L391
    sgrp_size_per_cu, lds_size_per_cu, hwreg_size_per_cu = 0x4000, 0x10000, 0x1000
    vgpr_size_per_cu = 0x60000 if self.target in {110000, 110001, 120000, 120001} else 0x40000
    wg_data_size = round_up((vgpr_size_per_cu + sgrp_size_per_cu + lds_size_per_cu + hwreg_size_per_cu) * (max_cu_id + 1), mmap.PAGESIZE)
    ctl_stack_size = round_up(12 * (max_cu_id + 1) * (max_wave_id + 1) + 8 + 40, mmap.PAGESIZE)
    debug_memory_size = round_up((max_cu_id + 1) * (max_wave_id + 1) * 32, 64)

    self.compute_queue = self.create_queue(kfd.KFD_IOC_QUEUE_TYPE_COMPUTE, 16 << 10, ctx_save_restore_size=wg_data_size + ctl_stack_size,
                                           eop_buffer_size=0x1000, ctl_stack_size=ctl_stack_size, debug_memory_size=debug_memory_size)

    # self.sdma_queue = self.create_queue(kfd.KFD_IOC_QUEUE_TYPE_SDMA, 0x800000)

    super().__init__(device, AMDDriverAllocator(self), AMDRenderer(), AMDCompiler(self.arch), functools.partial(AMDProgram, self),
                     functools.partial(AMDSignal, self), AMDComputeQueue, AMDCopyQueue)

    AMDComputeQueue().signal(self.timeline_signal, self.timeline_value).submit(self)
    self.timeline_value += 1

    time.sleep(1)

    sig_view = self.signals_page.meta[3].offset(self.timeline_signal.value_addr - self.signals_page.va_addr, elsz=4)
    print(sig_view[0])
    print(self.compute_queue.write_ptr[0], self.compute_queue.read_ptr[0])

    print("OK")
    exit(0)

    # self.synchronize()
    
    # print(read_physical_memory(self.timeline_signal.value_addr))

  def create_queue(self, queue_type, ring_size, ctx_save_restore_size=0, eop_buffer_size=0, ctl_stack_size=0, debug_memory_size=0):
    ring = self.dev_iface.alloc(ring_size, uncached=True, cpu_access=True, zero=True)
    gart = self.dev_iface.alloc(0x1000, uncached=True, cpu_access=True, zero=True)
    eop_buffer = self.dev_iface.alloc(eop_buffer_size, zero=True) if eop_buffer_size else None
    return self.dev_iface.create_queue(queue_type, ring, gart, eop_buffer=eop_buffer, debug_memory_size=debug_memory_size,
                                       ctx_save_restore_size=ctx_save_restore_size, ctl_stack_size=ctl_stack_size)

  def invalidate_caches(self):
    AMDComputeQueue().memory_barrier().signal(self.timeline_signal, self.timeline_value).submit(self)
    self.timeline_value += 1
    self.synchronize()

  def on_device_hang(self):
    self.dev_iface.adev.ih.interrupt_handler()
    self.dev_iface.adev.gmc.on_interrupt()

    for dev in self.devices:
      print(dev.device_id, dev.sdma_queue.read_ptr[0], dev.sdma_queue.write_ptr[0])
      print(hex(self.dev_iface.adev.regSDMA0_F32_CNTL.read()))

    raise RuntimeError("Device hang detected")

    # raise RuntimeError("Device hang detected")

    # report = []

    # ev = (kfd.struct_kfd_event_data)(event_id=self.mem_fault_event.event_id)
    # kfd.AMDKFD_IOC_WAIT_EVENTS(KFDIface.kfd, events_ptr=ctypes.addressof(ev), num_events=1, wait_for_all=1)
    # if ev.memory_exception_data.gpu_id:
    #   pfstatus = ' '.join(f'{k[0]}={getattr(ev.memory_exception_data.failure, k[0])}' for k in ev.memory_exception_data.failure._fields_)
    #   report += [f"MMU fault: 0x{ev.memory_exception_data.va:X} | {pfstatus}"]

    # ev = (kfd.struct_kfd_event_data)(event_id=self.hw_fault_event.event_id)
    # kfd.AMDKFD_IOC_WAIT_EVENTS(KFDIface.kfd, events_ptr=ctypes.addressof(ev), num_events=1, wait_for_all=1)
    # if ev.hw_exception_data.gpu_id:
    #   report += [f"HW fault: {' '.join(f'{k[0]}={getattr(ev.hw_exception_data, k[0])}' for k in ev.hw_exception_data._fields_)}"]

    # raise RuntimeError("\n".join(report))
