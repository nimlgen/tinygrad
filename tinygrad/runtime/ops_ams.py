from __future__ import annotations
from typing import Tuple, List, Any
import os, ctypes, ctypes.util, functools, pathlib, mmap, errno, time, array, contextlib, decimal
from dataclasses import dataclass
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HWComputeQueue, HWCopyQueue, HCQArgsState, \
                            HCQSignal, HCQProgram, BufferOptions, LRUAllocator
from tinygrad.helpers import getenv, to_mv, round_up, data64_le, DEBUG, mv_address, from_mv
from tinygrad.renderer.cstyle import AMDRenderer
from tinygrad.runtime.autogen import kfd, hsa, amd_gpu, libc
from tinygrad.runtime.support.compiler_hip import AMDCompiler
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.am.amdev import AMDev
from tinygrad.runtime.autogen import libpciaccess, amdgpu_2, amdgpu_mp_13_0_0, amdgpu_nbio_4_3_0, amdgpu_discovery, amdgpu_mmhub_3_0_0
if getenv("IOCTL"): import extra.hip_gpu_driver.hip_ioctl  # noqa: F401 # pylint: disable=unused-import
if getenv("MOCKGPU"): import extra.mockgpu.mockgpu # noqa: F401 # pylint: disable=unused-import


regBIF_BX_PF1_GPU_HDP_FLUSH_REQ, regBIF_BX_PF1_GPU_HDP_FLUSH_DONE = 0x0106, 0x0107

# VGT_EVENT_TYPE in navi10_enum.h
CACHE_FLUSH_AND_INV_TS_EVENT = 0x14

WAIT_REG_MEM_FUNCTION_EQ = 3 # ==
WAIT_REG_MEM_FUNCTION_GEQ = 5 # >=

COMPUTE_SHADER_EN, FORCE_START_AT_000, CS_W32_EN = (1 << 0), (1 << 2), (1 << 15)

def gfxreg(reg): return reg + 0x00001260 - amd_gpu.PACKET3_SET_SH_REG_START
def nbioreg(reg): return reg + 0x00000d20 # NBIO_BASE__INST0_SEG2

class AMSignal(HCQSignal):
  def __init__(self, value=0, is_timeline=False):
    self._o = AMSDevice.signals_pool.pop()
    x, off = self._o
    self._gpu_addr = x.va_addr + off
    # self._mc_addr = x.mc_addr + off
    self._signal = to_mv(x.cpu_addr + off, 16).cast("Q")
    super().__init__(value)
  def __del__(self): AMSDevice.signals_pool.append(self._o)
  def _get_value(self) -> int: return self._signal[0]
  def _get_timestamp(self) -> decimal.Decimal: return decimal.Decimal(self._signal[1]) / decimal.Decimal(100)
  def _set_value(self, new_value:int): self._signal[0] = new_value

class AMComputeQueue(HWComputeQueue):
  def __init__(self):
    self.cmd_idx_to_local_offset, self.cmd_idx_to_global_offset, self.cmd_idx_to_dispatch_packet = {}, {}, {}
    super().__init__()

  def _acquire_mem(self, addr=0x0, sz=(1 << 64)-1, gli=1, glm=1, glk=1, glv=1, gl1=1, gl2=1):
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_ACQUIRE_MEM, 6), 0, *data64_le(sz), *data64_le(addr), 0,
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV(gli) | \
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_INV(glm) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_WB(glm) | \
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV(glk) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_WB(glk) | \
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV(glv) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV(gl1) | \
               amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV(gl2) | amd_gpu.PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_WB(gl2)]

  def _release_mem(self, mem_event_type, mem_data_sel, mem_int_sel, address, value=0, cst=0, cache_flush=False):
    cache_flush_flags = 0

    if cache_flush:
      cache_flush_flags = amd_gpu.PACKET3_RELEASE_MEM_GCR_GLV_INV | amd_gpu.PACKET3_RELEASE_MEM_GCR_GL1_INV | \
        amd_gpu.PACKET3_RELEASE_MEM_GCR_GL2_INV | amd_gpu.PACKET3_RELEASE_MEM_GCR_GLM_WB | amd_gpu.PACKET3_RELEASE_MEM_GCR_GLM_INV | \
        amd_gpu.PACKET3_RELEASE_MEM_GCR_GL2_WB | amd_gpu.PACKET3_RELEASE_MEM_GCR_SEQ

    # event_index__mec_release_mem__end_of_pipe = 5
    # event_index__mec_release_mem__shader_done = 6
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_RELEASE_MEM, 6),
      amd_gpu.PACKET3_RELEASE_MEM_EVENT_TYPE(mem_event_type) | amd_gpu.PACKET3_RELEASE_MEM_EVENT_INDEX(5) | cache_flush_flags,
      amd_gpu.PACKET3_RELEASE_MEM_DATA_SEL(mem_data_sel) | amd_gpu.PACKET3_RELEASE_MEM_INT_SEL(mem_int_sel) | amd_gpu.PACKET3_RELEASE_MEM_DST_SEL(0),
      *data64_le(address), *data64_le(value), cst]

  def _memory_barrier(self):
    pass
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_WAIT_REG_MEM, 5), amd_gpu.WAIT_REG_MEM_MEM_SPACE(0) | amd_gpu.WAIT_REG_MEM_OPERATION(1) | \
      amd_gpu.WAIT_REG_MEM_FUNCTION(WAIT_REG_MEM_FUNCTION_EQ) | amd_gpu.WAIT_REG_MEM_ENGINE(0), nbioreg(regBIF_BX_PF1_GPU_HDP_FLUSH_REQ),
      nbioreg(regBIF_BX_PF1_GPU_HDP_FLUSH_DONE), 0xffffffff, 0xffffffff, 0x20]
    self._acquire_mem()

  def _exec(self, prg, args_state, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1)):
    self._acquire_mem(gli=0, gl2=0)

    cmd_idx = self._cur_cmd_idx()
    user_regs = [*data64_le(args_state.ptr)]

    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 2), gfxreg(amd_gpu.regCOMPUTE_PGM_LO), *data64_le(prg.prog_addr >> 8)]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 2), gfxreg(amd_gpu.regCOMPUTE_PGM_RSRC1), prg.rsrc1, prg.rsrc2]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 1), gfxreg(amd_gpu.regCOMPUTE_PGM_RSRC3), 0]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 1), gfxreg(amd_gpu.regCOMPUTE_TMPRING_SIZE), prg.device.tmpring_size]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 2),
               gfxreg(amd_gpu.regCOMPUTE_DISPATCH_SCRATCH_BASE_LO), *data64_le(prg.device.scratch.va_addr >> 8)]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 4), gfxreg(amd_gpu.regCOMPUTE_RESTART_X), 0, 0, 0, 0]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 2), gfxreg(amd_gpu.regCOMPUTE_STATIC_THREAD_MGMT_SE0)] + [0xFFFFFFFF] * 2
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 2), gfxreg(amd_gpu.regCOMPUTE_STATIC_THREAD_MGMT_SE2)] + [0xFFFFFFFF] * 2
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 4), gfxreg(amd_gpu.regCOMPUTE_STATIC_THREAD_MGMT_SE4)] + [0xFFFFFFFF] * 4
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, len(user_regs)), gfxreg(amd_gpu.regCOMPUTE_USER_DATA_0)] + user_regs

    self.cmd_idx_to_local_offset[cmd_idx] = len(self.q) - self.cmds_offset[cmd_idx] + 5 # +1 to skip PACKET3_SET_SH_REG + reg + 3 zeros.
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 8), gfxreg(amd_gpu.regCOMPUTE_START_X), 0, 0, 0, *local_size, 0, 0]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_SET_SH_REG, 1), gfxreg(amd_gpu.regCOMPUTE_RESOURCE_LIMITS), 0x0]

    self.cmd_idx_to_global_offset[cmd_idx] = len(self.q) - self.cmds_offset[cmd_idx] + 1 # +1 to skip PACKET3_DISPATCH_DIRECT.
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_DISPATCH_DIRECT, 3), *global_size, CS_W32_EN | FORCE_START_AT_000 | COMPUTE_SHADER_EN]
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_EVENT_WRITE, 0), amd_gpu.EVENT_TYPE(7) | amd_gpu.EVENT_INDEX(4)]
    pass

  def _update_exec(self, cmd_idx, global_size, local_size):
    if local_size is not None: self._patch(cmd_idx, offset=self.cmd_idx_to_local_offset[cmd_idx], data=local_size)
    if global_size is not None: self._patch(cmd_idx, offset=self.cmd_idx_to_global_offset[cmd_idx], data=global_size)

    if (dp:=self.cmd_idx_to_dispatch_packet.get(cmd_idx)) is not None:
      if local_size is not None: dp.workgroup_size_x, dp.workgroup_size_y, dp.workgroup_size_z = local_size[0], local_size[1], local_size[2]
      if global_size is not None:
        dp.grid_size_x,dp.grid_size_y,dp.grid_size_z = [g*l for g,l in zip(global_size,[dp.workgroup_size_x,dp.workgroup_size_y,dp.workgroup_size_z])]

  def _wait(self, signal, value=0):
    pass
    self.q += [amd_gpu.PACKET3(amd_gpu.PACKET3_WAIT_REG_MEM, 5),
      amd_gpu.WAIT_REG_MEM_MEM_SPACE(1) | amd_gpu.WAIT_REG_MEM_OPERATION(0) | amd_gpu.WAIT_REG_MEM_FUNCTION(WAIT_REG_MEM_FUNCTION_GEQ) | \
      amd_gpu.WAIT_REG_MEM_ENGINE(0), *data64_le(signal._gpu_addr), value, 0xffffffff, 4]

  def _timestamp(self, signal): 
    self._release_mem(CACHE_FLUSH_AND_INV_TS_EVENT, mem_data_sel=3, mem_int_sel=0, address=signal._gpu_addr + 8, cache_flush=False)

  def _signal(self, signal, value=0):
    # NOTE: this needs an EOP buffer on the queue or it will NULL pointer
    self._release_mem(CACHE_FLUSH_AND_INV_TS_EVENT, mem_data_sel=1, mem_int_sel=2, address=signal._gpu_addr, value=value, cache_flush=False)

  def _update_wait(self, cmd_idx, signal=None, value=None):
    if signal is not None: self._patch(cmd_idx, offset=2, data=data64_le(signal._gpu_addr))
    if value is not None: self._patch(cmd_idx, offset=4, data=[value])

  def _update_signal(self, cmd_idx, signal=None, value=None):
    if signal is not None: self._patch(cmd_idx, offset=3, data=data64_le(signal._gpu_addr))
    if value is not None: self._patch(cmd_idx, offset=5, data=data64_le(value))

  def bind(self, device):
    self.binded_device = device
    self.hw_page = device.allocator.alloc(len(self.q) * 4, BufferOptions(cpu_access=True, nolru=True, uncached=True))
    hw_view = to_mv(self.hw_page.cpu_addr, self.hw_page.size).cast("I")
    for i, value in enumerate(self.q): hw_view[i] = value

    self.indirect_cmd = [amd_gpu.PACKET3(amd_gpu.PACKET3_INDIRECT_BUFFER, 2), *data64_le(self.hw_page.va_addr),
                         len(self.q) | amd_gpu.INDIRECT_BUFFER_VALID]
    self.q = hw_view # type: ignore
  
  def _submit(self, device):
    # device.adev.smu.smu_set_power_profile(enabled=True)
    cmds = self.indirect_cmd if device == self.binded_device else self.q
    for value in cmds: device.adev.gfx.kcq_ring.write(value)
    device.adev.wdoorbell64(device.adev.gfx.kcq_ring.doorbell_index, device.adev.gfx.kcq_ring.next_ptr)

    for i in range(100000):
      val = device.adev.gfx.kcq_ring.mqd_vm.cpu_view().cast('I')[0x17]
      if val != 0xffffffff:
        print(val)
    print(device.adev.gfx.kcq_ring.mqd_vm.cpu_view().cast('I')[0x17])

class AMArgsState(HCQArgsState):
  def __init__(self, ptr:int, prg:AMProgram, bufs:Tuple[HCQBuffer, ...], vals:Tuple[int, ...]=()):
    super().__init__(ptr, prg, bufs, vals=vals)

    self.paddr_ptr = self.prg.device.kernargs_page.paddr + self.ptr - self.prg.device.kernargs_page.va_addr
    self.cpu_ptr = self.prg.device.kernargs_page.cpu_addr + self.ptr - self.prg.device.kernargs_page.va_addr

    self.bufs = to_mv(self.cpu_ptr, len(bufs) * 8).cast('Q')
    self.vals = to_mv(self.cpu_ptr + len(bufs) * 8, len(vals) * 4).cast('I')

    self.bufs[:] = array.array('Q', [b.va_addr for b in bufs])
    self.vals[:] = array.array('I', vals)

  def update_buffer(self, index:int, buf:HCQBuffer): self.bufs[index] = buf.va_addr
  def update_var(self, index:int, val:int): self.vals[index] = val

class AMProgram(HCQProgram):
  def __init__(self, device:AMDDevice, name:str, lib:bytes):
    # TODO; this API needs the type signature of the function and global_size/local_size
    self.device, self.name, self.lib = device, name, lib

    # if DEBUG >= 6: print(disasm(lib))

    image, sections, _ = elf_loader(self.lib)
    self.lib_gpu = self.device.allocator.alloc(round_up(image.nbytes, 0x4000), BufferOptions(cpu_access=True, nolru=True))
    ctypes.memmove(self.lib_gpu.cpu_addr, mv_address(image), image.nbytes)

    entry_point = min(sh.header.sh_addr for sh in sections if sh.header.sh_type == libc.SHT_PROGBITS and sh.header.sh_flags & libc.SHF_ALLOC)
    self.group_segment_size = image[entry_point:entry_point+4].cast("I")[0]
    self.private_segment_size = image[entry_point+4:entry_point+8].cast("I")[0]
    self.kernargs_segment_size = image[entry_point+8:entry_point+12].cast("I")[0]

    lds_size = ((self.group_segment_size + 511) // 512) & 0x1FF
    if self.private_segment_size > self.device.max_private_segment_size: raise RuntimeError("Too many resources requsted: private_segment_size")

    # print(self.name, lds_size, self.private_segment_size)

    code = hsa.amd_kernel_code_t.from_address(self.lib_gpu.cpu_addr + entry_point) # NOTE: this is wrong, it's not this object
    assert code.kernel_code_properties & 0x400 == 0x400 # ENABLE_WAVEFRONT_SIZE32

    self.rsrc1 = code.compute_pgm_rsrc1
    self.rsrc2 = code.compute_pgm_rsrc2 | (lds_size << 15)
    self.prog_addr = self.lib_gpu.va_addr + entry_point + code.kernel_code_entry_byte_offset

    # Some programs use hsa_kernel_dispatch_packet_t to read workgroup sizes during execution.
    # The packet is represented as a pointer and set up in SGPRs. Space for the packet is allocated as part of the kernel arguments.
    self.enable_dispatch_ptr = code.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR
    self.enable_private_segment_sgpr = code.kernel_code_properties & hsa.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER
    additional_alloc_sz = ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t) if self.enable_dispatch_ptr else 0

    super().__init__(AMArgsState, self.device, self.name, kernargs_alloc_size=self.kernargs_segment_size+additional_alloc_sz)

  def __del__(self):
    if hasattr(self, 'lib_gpu'): self.device.allocator.free(self.lib_gpu, self.lib_gpu.size, BufferOptions(cpu_access=True, nolru=True))

class AMBuffer(HCQBuffer):
  def __init__(self, adev, vm):
    self.va_addr, self.size = vm.vaddr, vm.size
    self.paddr = vm.paddr
    # self.mc_addr = vm.mc_addr()
    self.cpu_addr = vm.cpu_addr()
    self.cpu_view = vm.cpu_view()

class AMAllocator(LRUAllocator):
  def __init__(self, device:AMSDevice):
    self.device = device
    super().__init__()

  def _alloc(self, size:int, options:BufferOptions) -> HCQBuffer:
    return self.device._gpu_alloc(size, uncached=options.uncached)

  def _free(self, opaque, options:BufferOptions):
    pass
    # self.device.synchronize()
    # TODO

  def copyin(self, dest:HCQBuffer, src:memoryview):
    # pass
    ctypes.memmove(dest.cpu_addr, mv_address(src), src.nbytes)
    self.device.adev.gmc.flush_hdp()

  def copyout(self, dest:memoryview, src:HCQBuffer):
    # for i in range(10):
    #   print(hex(self.device.adev.regSPI_TMPRING_SIZE.read()))

    # for i in range(1000):
    #   val = self.device.adev.gfx.kcq_ring.mqd_vm.cpu_view().cast('I')[0x17]
    #   if val != 0xffffffff:
    #     print(val)
      # print(hex(self.device.adev.gfx.kcq_ring.mqd_vm.cpu_view().cast('I')[0x17]))
    
    self.device.synchronize()
    # self.device.adev.vmm.collect_pfs()

    # print(hex(self.device.adev.gfx.kcq_ring.mqd_vm.cpu_view().cast('I')[0x17]))
    ctypes.memmove(from_mv(dest), src.cpu_addr, dest.nbytes)

class AMSDevice(HCQCompiled):
  signals_pool = []

  def __init__(self, device:str=""):
    # if os.path.isdir('/sys/module/amdgpu/'): raise RuntimeError("amdgpu module is loaded, unload it first")

    self.pcidev = self._find_pci_dev()
    self.adev = AMDev(self.pcidev)
    # print("ok")
    # exit(0)

    self.arch = "gfx1100"
    self.tmpring_size = 0x200200
    self.max_private_segment_size = 4096
    self.scratch = self._gpu_alloc(0x18000000)

    signals_alloc = self._gpu_alloc(0x4000, uncached=True)
    AMSDevice.signals_pool = [(signals_alloc, off) for off in range(0, signals_alloc.size, 16)]

    self.adev.smu.smu_set_power_profile(enabled=True)

    super().__init__(device, AMAllocator(self), AMDRenderer(), AMDCompiler(self.arch), functools.partial(AMProgram, self),
                     AMSignal, AMComputeQueue, None)

  def _gpu_alloc(self, size:int, uncached=False):
    size = round_up(size, 0x1000)
    vm = self.adev.mm.valloc(size, uncached=uncached)
    return AMBuffer(self.adev, vm)

  def _find_pci_dev(self):
    libpciaccess.pci_system_init()

    pci_iter = libpciaccess.pci_id_match_iterator_create(None)

    pcidev = None
    while True:
      pcidev = libpciaccess.pci_device_next(pci_iter)
      if not pcidev: break
      dev_fmt = "{:04x}:{:02x}:{:02x}.{:d}".format(pcidev.contents.domain_16, pcidev.contents.bus, pcidev.contents.dev, pcidev.contents.func)

      if pcidev.contents.vendor_id == 0x1002 and pcidev.contents.device_id == 0x744c:
          dev_fmt = "{:04x}:{:02x}:{:02x}.{:d}".format(pcidev.contents.domain_16, pcidev.contents.bus, pcidev.contents.dev, pcidev.contents.func)
          if dev_fmt == "0000:03:00.0": continue
          # if dev_fmt == "0000:44:00.0": continue
          # if dev_fmt == "0000:83:00.0": continue
          # if dev_fmt == "0000:86:00.0": continue
          # if dev_fmt == "0000:c3:00.0": continue
          print(f"Select device {dev_fmt}")
          break

    assert pcidev is not None
    pcidev = pcidev.contents

    libpciaccess.pci_device_probe(ctypes.byref(pcidev))
    return pcidev
