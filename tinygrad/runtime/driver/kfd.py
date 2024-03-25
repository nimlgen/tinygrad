import os, ctypes, pathlib, re, fcntl, functools, mmap
import tinygrad.runtime.autogen.kfd as kfd
import tinygrad.runtime.autogen.hsa as hsa
from tinygrad.helpers import to_mv, from_mv
from extra.hip_gpu_driver import hip_ioctl
from tinygrad.runtime.ops_hsa import HSACompiler

libc = ctypes.CDLL("libc.so.6")
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mmap.restype = ctypes.c_void_p
MAP_NORESERVE = 0x4000
MAP_FIXED = 0x10

def kfd_ioctl(idir, nr, user_struct, fd, **kwargs):
  made = user_struct(**kwargs)
  ret = fcntl.ioctl(fd, (idir<<30) | (ctypes.sizeof(user_struct)<<16) | (ord('K')<<8) | nr, made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  return made

def format_struct(s):
  sdats = []
  for field_name, field_type in s._fields_:
    dat = getattr(s, field_name)
    if isinstance(dat, int): sdats.append(f"{field_name}:0x{dat:X}")
    else: sdats.append(f"{field_name}:{dat}")
  return sdats

idirs = {"IOW": 1, "IOR": 2, "IOWR": 3}
def ioctls_from_header():
  hdr = pathlib.Path("/usr/include/linux/kfd_ioctl.h").read_text().replace("\\\n", "")
  pattern = r'#define\s+(AMDKFD_IOC_[A-Z0-9_]+)\s+AMDKFD_(IOW?R?)\((0x[0-9a-fA-F]+),\s+struct\s([A-Za-z0-9_]+)\)'
  matches = re.findall(pattern, hdr, re.MULTILINE)

  fxns = {}
  for name, idir, nr, sname in matches:
    fxns[name.replace("AMDKFD_IOC_", "").lower()] = functools.partial(kfd_ioctl, idirs[idir], int(nr, 0x10), getattr(kfd, "struct_"+sname))
  return type("KIO", (object, ), fxns)
kio = ioctls_from_header()

def get_binary_info(binary):
  with open("/home/nimlgen/amd.elf", 'wb') as file:
    file.write(binary)

  from io import BytesIO
  from elftools.elf.elffile import ELFFile
  from elftools.elf.sections import NoteSection
  from elftools.elf.constants import P_FLAGS
  import msgpack
  with BytesIO(binary) as f:
    elffile = ELFFile(f)
    print(elffile.header['e_type'])

    dynsym = elffile.get_section_by_name('.dynsym')
    if not dynsym:
      print("Dynamic symbol table not found.")
      return None

    # for segment in elffile.iter_segments():
    #   if segment.header.p_type == 'PT_LOAD':
    #     flags = segment.header.p_flags
    #     segment_type = []

    #     if flags & P_FLAGS.PF_X:
    #       segment_type.append("Executable (.text)")
    #     if flags & P_FLAGS.PF_W:
    #       segment_type.append("Writable (.data)")
    #     if not segment_type:  # If neither executable nor writable, could be .rodata, etc.
    #       segment_type.append("Other")

    #     print(f"Segment: {segment.header.p_type}, Offset: {segment.header.p_offset}, Size: {segment.header.p_filesz}, Type: {', '.join(segment_type)}")
    #   else:
    #     print(f"Segment: {segment.header.p_type}, Offset: {segment.header.p_offset}, Size: {segment.header.p_filesz}")
                
    # print("\nSections in the ELF file:")
    # for section in elffile.iter_sections():
    #   print(f"Section Name: {section.name}, Type: {section['sh_type']}, Address: {section['sh_addr']}, Size: {section['sh_size']}")
    for segment in elffile.iter_segments():
      segment_range = range(segment.header.p_offset, segment.header.p_offset + segment.header.p_filesz)
      print(f"\nSegment {segment.header.p_type} [Offset: {segment.header.p_offset}, Size: {segment.header.p_filesz}]")
      print("Contains sections:")
      
      for section in elffile.iter_sections():
        section_range = range(section.header.sh_offset, section.header.sh_offset + section.header.sh_size)
        # Check if the section's offset range falls within the segment's offset range
        if segment_range.start <= section_range.start and segment_range.stop >= section_range.stop:
          print(f"  - {section.name} [Offset: {section.header.sh_offset}, Size: {section.header.sh_size}]")

    kern_info = None
    for section in elffile.iter_sections():
      if isinstance(section, NoteSection):
        for note in section.iter_notes():
          kern_info = msgpack.unpackb(note['n_descdata'])
          print(kern_info)

    text_section = elffile.get_section_by_name('.text')
    if not text_section:
      print("The .text section could not be found.")
      return None
        
    # Simulate loading the section into a "buffer"
    buffer = text_section.data()
    # print(buffer)

    assert kern_info is not None
    return kern_info, buffer

if __name__ == "__main__":
  fd = os.open("/dev/kfd", os.O_RDWR)
  drm_fd = os.open("/dev/dri/renderD128", os.O_RDWR)
  GPU_ID = 17213

  ver = kio.get_version(fd)
  st = kio.acquire_vm(fd, drm_fd=drm_fd, gpu_id=GPU_ID)

  # 0xF0000001 = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
  # 0xD6000002 = KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
  # 0xD6000004 = KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
  # 0x94000010 = KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
  #addr = libc.mmap(0, 0x1000, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS, -1, 0)
  #addr = libc.mmap(0, 0x1000, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|mmap.MAP_ANONYMOUS, -1, 0)
  #mem = kio.AMDKFD_IOC_ALLOC_MEMORY_OF_GPU(fd, va_addr=addr, size=0x1000, gpu_id=GPU_ID, flags=0xD6000004)

  addr = libc.mmap(0, 0x1000, 0, mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS|MAP_NORESERVE, -1, 0)
  mem = kio.alloc_memory_of_gpu(fd, va_addr=addr, size=0x1000, gpu_id=GPU_ID, flags=0xF0000001)
  buf = libc.mmap(mem.va_addr, mem.size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|MAP_FIXED, drm_fd, mem.mmap_offset)

  addr = libc.mmap(0, 0x1000, 0, mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS|MAP_NORESERVE, -1, 0)
  mem = kio.alloc_memory_of_gpu(fd, va_addr=addr, size=0x1000, gpu_id=GPU_ID, flags=0xF0000001)
  buf2 = libc.mmap(mem.va_addr, mem.size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|MAP_FIXED, drm_fd, mem.mmap_offset)

  arr = (ctypes.c_int32 * 1)(GPU_ID)
  stm = kio.map_memory_to_gpu(fd, handle=mem.handle, device_ids_array_ptr=ctypes.addressof(arr), n_devices=1)
  assert stm.n_success == 1

  nq = kio.create_queue(fd, ring_base_address=buf, ring_size=0x1000, gpu_id=GPU_ID,
                        queue_type=kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL, queue_percentage=kfd.KFD_MAX_QUEUE_PERCENTAGE,
                        queue_priority=kfd.KFD_MAX_QUEUE_PRIORITY, write_pointer_address=buf2, read_pointer_address=buf2 + 0x8)
  print(nq)

  # map doorbells
  doorbell = libc.mmap(0, 0x8, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|MAP_FIXED, fd, nq.doorbell_offset)
  print("doorbell", hex(doorbell))

  #mv = to_mv(buf, 0x1000)

  # Load kernel binary
  addr = libc.mmap(0, 0x1000, 0, mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS|MAP_NORESERVE, -1, 0)
  mem = kio.alloc_memory_of_gpu(fd, va_addr=addr, size=0x1000, gpu_id=GPU_ID, flags=0xF0000001)
  codebuf = libc.mmap(mem.va_addr, mem.size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|MAP_FIXED, drm_fd, mem.mmap_offset)

  addr = libc.mmap(0, 0x1000, 0, mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS|MAP_NORESERVE, -1, 0)
  mem = kio.alloc_memory_of_gpu(fd, va_addr=addr, size=0x1000, gpu_id=GPU_ID, flags=0xF0000001)
  inpbuf = libc.mmap(mem.va_addr, mem.size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|MAP_FIXED, drm_fd, mem.mmap_offset)
  
  addr = libc.mmap(0, 0x1000, 0, mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS|MAP_NORESERVE, -1, 0)
  mem = kio.alloc_memory_of_gpu(fd, va_addr=addr, size=0x1000, gpu_id=GPU_ID, flags=0xF0000001)
  args = libc.mmap(mem.va_addr, mem.size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED|MAP_FIXED, drm_fd, mem.mmap_offset)
  args_pointer = ctypes.cast(args, ctypes.POINTER(ctypes.c_uint64))
  args_pointer[0] = inpbuf

  compiler = HSACompiler(arch="gfx1100")
  binary = compiler.compile("""
  extern "C" __attribute__((device)) __attribute__((const)) int __ockl_get_local_id(unsigned int);
  extern "C" __attribute__((device)) __attribute__((const)) int __ockl_get_group_id(unsigned int);
  extern "C" __attribute__((device)) __attribute__((const)) int __ockl_get_local_size(unsigned int);
  extern "C" __attribute__((device)) void test(int* data0) {
  int gidx0 = __ockl_get_group_id(0);
  *((int*)(data0+gidx0)) = int(5);
}""")

  ctypes.memset(codebuf, 0, 0x1000)
  header = kfd.amd_kernel_code_t.from_address(codebuf)
  header.kernel_code_properties = 1 << kfd.AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_SHIFT
  header.compute_pgm_rsrc1 |= 32 << kfd.AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT
  header.compute_pgm_rsrc1 |= 32 << kfd.AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT
  header.compute_pgm_rsrc1 |= 3 << kfd.AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_SHIFT
  header.compute_pgm_rsrc1 |= 1 << kfd.AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE_SHIFT
  header.compute_pgm_rsrc2 |= kfd.AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X
  # header.compute_pgm_rsrc1 |= 1 << kfd.AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFR
  header.kernarg_segment_byte_size = 8
  info, bincode = get_binary_info(binary)
  print(ctypes.sizeof(kfd.amd_kernel_code_t))
  ctypes.memmove(codebuf + ctypes.sizeof(kfd.amd_kernel_code_t), from_mv(bytearray(bincode)), len(bincode))

  AQL_PACKET_SIZE = ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t)
  EMPTY_SIGNAL = hsa.hsa_signal_t()

  DISPATCH_KERNEL_SETUP = 3 << hsa.HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS
  DISPATCH_KERNEL_HEADER  = 1 << hsa.HSA_PACKET_HEADER_BARRIER
  DISPATCH_KERNEL_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
  DISPATCH_KERNEL_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
  DISPATCH_KERNEL_HEADER |= hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH << hsa.HSA_PACKET_HEADER_TYPE

  BARRIER_HEADER  = 1 << hsa.HSA_PACKET_HEADER_BARRIER
  BARRIER_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
  BARRIER_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
  BARRIER_HEADER |= hsa.HSA_PACKET_TYPE_BARRIER_AND << hsa.HSA_PACKET_HEADER_TYPE

  packet = hsa.hsa_kernel_dispatch_packet_t.from_address(buf)
  packet.workgroup_size_x = 1
  packet.workgroup_size_y = 1
  packet.workgroup_size_z = 1
  packet.reserved0 = 0
  packet.grid_size_x = 32
  packet.grid_size_y = 1
  packet.grid_size_z = 1
  packet.private_segment_size = 0
  packet.group_segment_size = 0
  packet.kernel_object = codebuf
  packet.kernarg_address = args
  packet.reserved2 = 0
  packet.completion_signal = EMPTY_SIGNAL
  packet.setup = DISPATCH_KERNEL_SETUP
  packet.header = DISPATCH_KERNEL_HEADER

  print(hex(nq.doorbell_offset))

  inpbuf_ptr = ctypes.cast(inpbuf, ctypes.POINTER(ctypes.c_uint32))
  for i in range(10): print(inpbuf_ptr[0])

  # https://llvm.org/docs/AMDGPUUsage.html#code-object-v3-kernel-descriptor
  # aql_kernel_header = 

  # print(buf)

  #addr = libc.mmap(0, 0x1000, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS, -1, 0)

  #print('\n'.join(format_struct(ver)))
  #print('\n'.join(format_struct(st)))
