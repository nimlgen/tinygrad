from __future__ import annotations
import os, ctypes, functools, mmap, struct, math, array, sys, contextlib
assert sys.platform != 'win32'
from typing import Any, cast
from tinygrad.device import Buffer, BufferSpec, Device, TinyELF
from tinygrad.runtime.support.hcq import HCQBuffer, MMIOInterface, FileIOInterface
from tinygrad.runtime.support.hcq2 import HCQ2Compiled, HCQAllocator, HCQInfo, HCQ_RUNTIME_DEV
from tinygrad.runtime.support.hcq2 import make_binary_patch, make_buf, make_cmdbuf, make_patches
from tinygrad.runtime.support.system import System
from tinygrad.runtime.autogen import kgsl, mesa, libc
from tinygrad.renderer.cstyle import QCOMCLRenderer
from tinygrad.renderer.nir import IR3Renderer
from tinygrad.helpers import getenv, round_up, ceildiv, data64_le, next_power2, to_tuple, mv_address, unwrap, is_image_shape, IMAGE, PROFILE
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher, KernelInfo, graph_rewrite
from tinygrad.engine.realize import get_call_arg_uops, pm_flatten_linear
if getenv("IOCTL"): import extra.qcom_gpu_driver.opencl_ioctl  # noqa: F401  # pylint: disable=unused-import

BUFTYPE_BUF, BUFTYPE_TEX, BUFTYPE_IBO = 0, 1, 2

@functools.cache
def dcache_flush():
  from tinygrad.codegen import to_program
  buf, n = UOp.param(0, dtypes.uint8, shape=(1,)), UOp.param(1, dtypes.int, shape=(), name="n", addrspace=AddrSpace.ALU)
  i = UOp.range(n, 0, dtype=dtypes.int)
  flush = UOp(Ops.CUSTOM, src=(buf.index(i * 64),), arg='__asm__ volatile("dc cvac, %0" :: "r"({0}) : "memory");')
  sink = UOp.sink(flush.end(i), UOp(Ops.CUSTOM, arg='__asm__ volatile("dsb sy" ::: "memory");'), arg=KernelInfo(name="dcache_flush"), tag=1)
  return Device["CPU"].runtime(to_program(sink, Device["CPU"].renderer).to_elf())

#Parse C-style defines: <regname>_<field_x>__SHIFT and <regname>_<field_y>__MASK from the adreno module into the following format:
# qreg.<regname>(<field_x>=..., <field_y>=..., ..., <field_n>=...)
def _qreg_exec(__reg, __val=0, **kwargs):
  for k, v in kwargs.items():
    reg_name = f"{__reg[4:]}_{k.removeprefix('_').upper()}"
    __val |= (getattr(mesa, reg_name) if v else 0) if type(v) is bool else (v << getattr(mesa, f'{reg_name}__SHIFT'))
  return __val
qreg: Any = type("QREG", (object,), {name[4:].lower(): functools.partial(_qreg_exec, name) for name in mesa.__dict__.keys() if name[:4] == 'REG_'})

def ctz(v): return (v & -v).bit_length() - 1

def parity(val: int):
  for i in range(4,1,-1): val ^= val >> (1 << i)
  return (~0x6996 >> (val & 0xf)) & 1

def pkt7_hdr(opcode: int, cnt: int): return mesa.CP_TYPE7_PKT | cnt & 0x3FFF | parity(cnt) << 15 | (opcode & 0x7F) << 16 | parity(opcode) << 23

def pkt4_hdr(reg: int, cnt: int): return mesa.CP_TYPE4_PKT | cnt & 0x7F | parity(cnt) << 7 | (reg & 0x3FFFF) << 8 | parity(reg) << 27

def _read_lib(lib, off) -> int: return struct.unpack("I", lib[off:off+4])[0]

def flag(nm, val): return (val << getattr(kgsl, f"{nm}_SHIFT")) & getattr(kgsl, f"{nm}_MASK")

class QCOMProgram: # the descriptors a dispatch needs, the image itself is uploaded by the linker
  def __init__(self, obj: TinyELF, nir:bool):
    self.signature, self.name, self.NIR = obj.signature, obj.name, nir

    if self.NIR:
      from tinygrad.runtime.support.compiler_mesa import IR3Compiler
      v, cs, imm_vals, self.image = IR3Compiler.unpack_lib(obj.lib)
      self.prg_offset, self.brnchstck, self.image_size, self.pvtmem, self.shmem = 0, v.branchstack, v.info.size, v.pvtmem_size, v.shared_size
      self.wgsz = alloc.offset_vec4 * 4 + 8 if (alloc:=cs.allocs.consts[mesa.IR3_CONST_ALLOC_DRIVER_PARAMS]).size_vec4 else 0xfc

      self.wgid, self.lid = v.cs.work_group_id, v.cs.local_invocation_id # register ids
      self.buf_off, imm_off = cs.ubo_state.range[0].offset, cs.allocs.max_const_offset_vec4 * 16
      self.consts_info = [(struct.unpack_from("<I", imm_vals, i)[0], imm_off + i, 4) for i in range(0, len(imm_vals), 4)]

      # see https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_shader.h#L525
      # and https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_compiler_nir.c#L5389
      self.samp_cnt, self.tex_cnt, self.ibo_cnt = (nt:=v.image_mapping.num_tex), nt, v.num_uavs - nt
      self.tex_to_image = v.image_mapping.tex_to_image[:]
      # IR3 outputs a sampler for every texture (https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_compiler_nir.c#L1714)
      self.samplers = [qreg.a6xx_tex_samp_0(wrap_s=(clamp_mode:=mesa.A6XX_TEX_CLAMP_TO_BORDER), wrap_t=clamp_mode, wrap_r=clamp_mode),
                       qreg.a6xx_tex_samp_1(unnorm_coords=True, cubemapseamlessfiltoff=True), 0, 0] * self.samp_cnt

      self.tex_off, self.ibo_off, self.samp_off = 2048, 2048 + 0x40 * self.tex_cnt, 2048 + 0x40 * (self.tex_cnt + self.ibo_cnt)
      self.fregs, self.hregs = v.info.max_reg + 1, v.info.max_half_reg + 1
    else: self._parse_lib(obj.lib)

    self.pvtmem_size_per_item: int = round_up(self.pvtmem, 512) >> 9
    self.pvtmem_size_total: int = self.pvtmem_size_per_item * 128 * 2
    self.hw_stack_offset: int = round_up(next_power2(round_up(self.pvtmem, 512)) * 128 * 16, 0x1000)
    self.shared_size: int = max(1, (self.shmem - 1) // 1024)
    self.kernargs_alloc_size = round_up(2048 + (self.tex_cnt + self.ibo_cnt) * 0x40 + len(self.samplers) * 4, 0x100)

  def _parse_lib(self, lib):
    # Extract image binary
    self.image_size = _read_lib(lib, 0x100)
    self.image = lib[(image_offset:=_read_lib(lib, 0xc0)):image_offset+self.image_size]

    # Parse image descriptors
    image_desc_off = _read_lib(lib, 0x110)
    self.prg_offset, self.brnchstck = _read_lib(lib, image_desc_off+0xc4), _read_lib(lib, image_desc_off+0x108) // 2
    self.pvtmem, self.shmem = _read_lib(lib, image_desc_off+0xc8), _read_lib(lib, image_desc_off+0xd8)

    # Fill up constants and buffers info
    self.consts_info = []

    # Collect sampler info.
    self.samp_cnt = samp_cnt_in_file = _read_lib(lib, image_desc_off + 0xdc)
    assert self.samp_cnt <= 1, "Up to one sampler supported"
    if self.samp_cnt:
      self.samp_cnt += 1
      self.samplers = [qreg.a6xx_tex_samp_0(wrap_s=(clamp_mode:=mesa.A6XX_TEX_CLAMP_TO_BORDER), wrap_t=clamp_mode, wrap_r=clamp_mode),
                       qreg.a6xx_tex_samp_1(unnorm_coords=True, cubemapseamlessfiltoff=True), 0, 0, 0, 0, 0, 0]
    else: self.samplers = []

    # Collect kernel arguments (buffers) info.
    bdoff, binfos = round_up(image_desc_off + 0x158 + len(self.name), 4) + 8 * samp_cnt_in_file, []
    while bdoff + 32 <= len(lib):
      length, _, _, offset_words, _, _, _, typ = struct.unpack("8I", lib[bdoff:bdoff+32])
      if length == 0: break
      binfos.append((offset_words * 4, typ))
      bdoff += length
    self.buf_offs = [off for off,typ in binfos if typ not in {BUFTYPE_TEX, BUFTYPE_IBO}]

    # Setting correct offsets to textures/ibos.
    self.tex_cnt, self.ibo_cnt = sum(typ is BUFTYPE_TEX for _,typ in binfos), sum(typ is BUFTYPE_IBO for _,typ in binfos)
    self.ibo_off, self.tex_off, self.samp_off = 2048, 2048 + 0x40 * self.ibo_cnt, 2048 + 0x40 * self.tex_cnt + 0x40 * self.ibo_cnt

    if _read_lib(lib, 0xb0) != 0: # check if we have constants.
      cdoff = _read_lib(lib, 0xac)
      while cdoff + 40 <= image_offset:
        cnst, offset_words, _, is32 = struct.unpack("I", lib[cdoff:cdoff+4])[0], *struct.unpack("III", lib[cdoff+16:cdoff+28])
        self.consts_info.append((cnst, offset_words * (sz_bytes:=(2 << is32)), sz_bytes))
        cdoff += 40

    # Registers info
    reg_desc_off = _read_lib(lib, 0x34)
    self.fregs, self.hregs = _read_lib(lib, reg_desc_off + 0x14), _read_lib(lib, reg_desc_off + 0x18)

_prg_cache:dict[tuple[bytes, tuple[str, ...]], UOp] = {}
def build_program(prg:UOp) -> UOp:
  if (cached:=_prg_cache.get(key:=(prg.src[3].arg, to_tuple(prg.device)))) is None:
    data = QCOMProgram(prg.to_elf(), isinstance(Device[key[1][0]].renderer, IR3Renderer))
    buf = UOp.placeholder((data.image_size,), dtypes.uint8, next(UOp.unique_num), device=prg.device).rtag("program")
    cached = _prg_cache[key] = prg.replace(src=(buf.after(make_binary_patch(buf, bytes(data.image))),), arg=(data, prg.arg))
  return cached

# *****************
# A6XX packets

def pkt7(op:int, *vals) -> UOp: return UOp(Ops.INS, arg=op, src=tuple(UOp.const(v, dtypes.uint32) for v in (pkt7_hdr(op, len(vals)), *vals)))
def pkt4(reg:int, *vals) -> UOp: return UOp(Ops.INS, arg=reg, src=tuple(UOp.const(v, dtypes.uint32) for v in (pkt4_hdr(reg, len(vals)), *vals)))

def cache_flush(devs, write_back=True, invalidate=False, sync=True, memsync=False) -> list[UOp]:
  # TODO: 7xx support.
  ins = [pkt7(mesa.CP_EVENT_WRITE, mesa.CACHE_FLUSH_TS, *data64_le(make_buf(devs, tag="dummy").getaddr(devs)), 0)] if write_back else []
  if invalidate: ins.append(pkt7(mesa.CP_EVENT_WRITE, mesa.CACHE_INVALIDATE)) # invalidate cache lines (following reads from RAM).
  if memsync: ins.append(pkt7(mesa.CP_WAIT_MEM_WRITES))
  if sync: ins.append(pkt7(mesa.CP_WAIT_FOR_IDLE))
  return ins

def qcom_wait(ctx, dst, val):
  return pkt7(mesa.CP_WAIT_REG_MEM, qreg.cp_wait_reg_mem_0(function=mesa.WRITE_GE, poll=mesa.POLL_MEMORY), *data64_le(dst.getaddr(ctx)),
              val & 0xFFFFFFFF, qreg.cp_wait_reg_mem_4(mask=0xFFFFFFFF), qreg.cp_wait_reg_mem_5(delay_loop_cycles=32))

def qcom_store(ctx, dst, val):
  return UOp(Ops.LINEAR, dtypes.void, (pkt7(mesa.CP_WAIT_FOR_IDLE),
    pkt7(mesa.CP_EVENT_WRITE, mesa.CACHE_FLUSH_TS, *data64_le(dst.getaddr(ctx)), val & 0xFFFFFFFF), *cache_flush(ctx, sync=False)))

def qcom_timestamp(ctx, dst):
  return UOp(Ops.LINEAR, dtypes.void, (pkt7(mesa.CP_WAIT_FOR_IDLE),
    pkt7(mesa.CP_REG_TO_MEM, qreg.cp_reg_to_mem_0(reg=mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER, cnt=2, _64b=True), *data64_le(dst.getaddr(ctx)))))

# *****************
# kernargs

def tex_const(imgdt, shape, ibo:bool) -> tuple[int, ...]: # 16 dwords, the buffer address at dword 4 is patched in separately
  pitch = shape[1] * 4 * imgdt.itemsize
  fmt = mesa.FMT6_32_32_32_32_FLOAT if imgdt.itemsize == 4 else mesa.FMT6_16_16_16_16_FLOAT
  return (qreg.a6xx_tex_const_0(fmt=fmt) if ibo else qreg.a6xx_tex_const_0(0x8, swiz_x=0, swiz_y=1, swiz_z=2, swiz_w=3, fmt=fmt),
          qreg.a6xx_tex_const_1(width=shape[1], height=shape[0]),
          qreg.a6xx_tex_const_2(type=mesa.A6XX_TEX_2D, pitch=pitch, pitchalign=ctz(pitch)-6), 0, 0, 0,
          qreg.a6xx_tex_const_6(plane_pitch=0x400000), qreg.a6xx_tex_const_7(13), 0, 0, 0, 0, 0, 0, 0, 0)

def encode_kernargs(call:UOp, prg:UOp, devs) -> UOp:
  data, info = prg.arg
  bufs = [get_call_arg_uops(call)[gi] for gi in info.globals]
  ubos = [bufs[slot] for _,slot,_,shape in data.signature if slot < len(bufs) and not is_image_shape(shape)]
  uavs = [(dt,shape,bufs[slot]) for _,slot,dt,shape in data.signature if slot < len(bufs) and is_image_shape(shape)]
  ibos, texs = uavs[:data.ibo_cnt], [uavs[data.ibo_cnt + (data.tex_to_image[i] if data.NIR else i)] for i in range(data.tex_cnt)]

  # constants, samplers and the static half of every texture descriptor are known at link time
  blob = bytearray(data.kernargs_alloc_size)
  for cnst,off,sz in data.consts_info: blob[off:off+sz] = cnst.to_bytes(sz, 'little')
  struct.pack_into(f'<{len(data.samplers)}I', blob, data.samp_off, *data.samplers)
  for off, uav, ibo in ((data.tex_off, texs, False), (data.ibo_off, ibos, True)):
    for i,(dt,shape,_) in enumerate(uav): struct.pack_into('<16I', blob, off + i*0x40, *tex_const(dt, shape, ibo))
  if data.NIR and data.wgsz != 0xfc: struct.pack_into('<3I', blob, data.wgsz*4, *(info.local_size or (1,1,1)))

  # buffer addresses and vars are patched over it
  var_offs = [o for o,_ in TinyELF.iter_sig(data.signature[len(bufs):], len(ubos)*8)] if data.NIR else []
  offs = ([data.buf_off + o for o in range(0, len(ubos)*8, 8)] + [data.buf_off + o for o in var_offs]) if data.NIR else data.buf_offs
  patches = list(zip(offs, [b.getaddr(devs) for b in ubos] + list(info.vars)))
  patches += [(o + i*0x40 + 16, b.getaddr(devs)) for o,uav in ((data.tex_off, texs), (data.ibo_off, ibos)) for i,(_,_,b) in enumerate(uav)]

  buf = UOp.placeholder((data.kernargs_alloc_size // 4,), dtypes.uint32, next(UOp.unique_num), device=devs).rtag("kernargs")
  return buf.after(make_binary_patch(buf, bytes(blob)), *make_patches(buf, patches))

# *****************
# host copies: the gpu must not touch cacheable host memory, and qcom memory is cpu mapped, so host<->device copies run on the host

def dev_idle(devs, loop_id:int) -> UOp: # a host call is outside the queues, so it has to wait out every device it touches
  v = make_buf(devs, tag="timeline_signal").after(loop:=UOp.loop(loop_id)).index(0).load()
  return v.end(loop, v + 1 < make_buf(devs, tag="timeline_value").index(0).load())

def qcom_host_copy(dst:UOp, src:UOp) -> UOp|None:
  if {to_tuple(b.device)[0].split(":")[0] for b in (dst, src)} != {"QCOM", "CPU"}: return None
  devs, host = to_tuple(next(b for b in (dst, src) if to_tuple(b.device)[0].startswith("QCOM")).device), (HCQ_RUNTIME_DEV.value,)
  fn = make_buf(devs, tag="func:memcpy").after(*[dev_idle(to_tuple(b.device), i) for i,b in enumerate((dst, src))])
  copy = fn.index(0).load().call(dst.getaddr(host), src.getaddr(host), UOp.const(dst.nbytes(), dtypes.uint64), ret_dtype=dtypes.void)
  return UOp.custom_function("hcq", copy.sink()).call(dst, src, name="hcq_copy", aux=HCQInfo(devs))

# *****************
# dispatch

def cast_int(x, ceil=False): return (math.ceil(x) if ceil else int(x)) if isinstance(x, float) else x

def qcom_program(ctx, call, prg):
  data, info = prg.arg
  args_addr, lib_addr = encode_kernargs(call, prg, ctx).getaddr(ctx), prg.src[0].getaddr(ctx)
  stack_addr = UOp.placeholder((data.hw_stack_offset * 4,), dtypes.uint8, 0, device=ctx).rtag("scratch").getaddr(ctx)
  global_size, local_size = info.global_size, info.local_size or (1, 1, 1)
  global_size_mp = [cast_int(g*l) for g,l in zip(global_size, local_size)]

  ins = [pkt7(mesa.CP_SET_MARKER, qreg.a6xx_cp_set_marker_0(mode=mesa.RM6_COMPUTE)),
    pkt4(mesa.REG_A6XX_SP_UPDATE_CNTL, qreg.a6xx_sp_update_cntl(cs_state=True, cs_uav=True)),
    pkt4(mesa.REG_A6XX_SP_UPDATE_CNTL, 0x0),
    pkt4(mesa.REG_A6XX_SP_CS_TSIZE, qreg.a6xx_sp_cs_tsize(0x80)), # is this right? mesa uses 1
    pkt4(mesa.REG_A6XX_SP_CS_USIZE, qreg.a6xx_sp_cs_usize(0x40)), # mesa also uses 1
    pkt4(mesa.REG_A6XX_SP_MODE_CNTL, qreg.a6xx_sp_mode_cntl(isammode=mesa.ISAMMODE_GL if data.NIR else mesa.ISAMMODE_CL,
                                                            constant_demotion_enable=data.NIR)),
    pkt4(mesa.REG_A6XX_SP_PERFCTR_SHADER_MASK, qreg.a6xx_sp_perfctr_shader_mask(cs=True)),
    pkt4(mesa.REG_A6XX_TPL1_MODE_CNTL, qreg.a6xx_tpl1_mode_cntl(isammode=mesa.ISAMMODE_GL if data.NIR else mesa.ISAMMODE_CL)),
    pkt4(mesa.REG_A6XX_TPL1_DBG_ECO_CNTL, 0),
    pkt7(mesa.CP_WAIT_FOR_IDLE),
    pkt4(mesa.REG_A6XX_SP_CS_NDRANGE_0,
      qreg.a6xx_sp_cs_ndrange_0(kerneldim=3, localsizex=local_size[0] - 1, localsizey=local_size[1] - 1, localsizez=local_size[2] - 1),
      global_size_mp[0], 0, global_size_mp[1], 0, global_size_mp[2], 0, 0xccc0cf, 0xfc | qreg.a6xx_sp_cs_wge_cntl(threadsize=mesa.THREAD64),
      *[cast_int(g, ceil=True) for g in global_size]),
    pkt4(mesa.REG_A6XX_SP_CS_CNTL_0,
      qreg.a6xx_sp_cs_cntl_0(threadsize=mesa.THREAD64, halfregfootprint=data.hregs, fullregfootprint=data.fregs, branchstack=data.brnchstck),
      qreg.a6xx_sp_cs_cntl_1(constantrammode=mesa.CONSTLEN_256, shared_size=data.shared_size), # should this be CONSTLEN_512?
      0, data.prg_offset, *data64_le(lib_addr),
      qreg.a6xx_sp_cs_pvt_mem_param(memsizeperitem=data.pvtmem_size_per_item), *data64_le(stack_addr),
      qreg.a6xx_sp_cs_pvt_mem_size(totalpvtmemsize=data.pvtmem_size_total)),
    pkt7(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS, state_src=mesa.SS6_INDIRECT,
                                                         state_block=mesa.SB6_CS_SHADER, num_unit=1024 // 4), *data64_le(args_addr)),
    pkt7(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                                                         state_block=mesa.SB6_CS_SHADER, num_unit=ceildiv(data.image_size, 128)),
         *data64_le(lib_addr)),
    pkt4(mesa.REG_A6XX_SP_REG_PROG_ID_0, 0xfcfcfcfc, 0xfcfcfcfc, 0xfcfcfcfc, 0xfc, qreg.a6xx_sp_cs_const_config(constlen=1024 // 4, enabled=True)),
    pkt4(mesa.REG_A6XX_SP_CS_PVT_MEM_STACK_OFFSET, qreg.a6xx_sp_cs_pvt_mem_stack_offset(data.hw_stack_offset)),
    # image_size is in bytes, but INSTR_SIZE is measured in units of instruction groups (16 instructions, 8 bytes each)
    # https://elixir.bootlin.com/mesa/mesa-26.1.5/source/src/freedreno/ir3/ir3_shader.h#L719-L723
    pkt4(mesa.REG_A6XX_SP_CS_INSTR_SIZE, qreg.a6xx_sp_cs_instr_size(ceildiv(data.image_size, 128)))]

  if data.samp_cnt > 0:
    border = UOp.placeholder((0x1000,), dtypes.uint8, 0, device=ctx).rtag("border_color")
    border_addr = border.after(make_binary_patch(border, bytes(border.max_numel()))).getaddr(ctx)
    ins += [pkt7(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                                                                 state_block=mesa.SB6_CS_TEX, num_unit=data.samp_cnt),
                 *data64_le(args_addr + data.samp_off)),
            pkt4(mesa.REG_A6XX_SP_CS_SAMPLER_BASE, *data64_le(args_addr + data.samp_off)),
            pkt4(mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE, *data64_le(border_addr))]

  if data.tex_cnt > 0:
    ins += [pkt7(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS, state_src=mesa.SS6_INDIRECT,
                                                                 state_block=mesa.SB6_CS_TEX, num_unit=min(16, data.tex_cnt)),
                 *data64_le(args_addr + data.tex_off)),
            pkt4(mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE, *data64_le(args_addr + data.tex_off))]

  if data.ibo_cnt > 0:
    ins += [pkt7(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST6_UAV, state_src=mesa.SS6_INDIRECT,
                                                                 state_block=mesa.SB6_CS_SHADER, num_unit=data.ibo_cnt),
                 *data64_le(args_addr + data.ibo_off)),
            pkt4(mesa.REG_A6XX_SP_CS_UAV_BASE, *data64_le(args_addr + data.ibo_off))]

  ins += [pkt4(mesa.REG_A6XX_SP_CS_CONFIG, qreg.a6xx_sp_cs_config(enabled=True, nsamp=data.samp_cnt, ntex=data.tex_cnt, nuav=data.ibo_cnt))]

  if data.NIR:
    ins += [pkt4(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0,
                 qreg.a6xx_sp_cs_const_config_0(wgidconstid=data.wgid, wgsizeconstid=data.wgsz, wgoffsetconstid=0xfc, localidregid=data.lid),
                 qreg.a6xx_sp_cs_wge_cntl(linearlocalidregid=0xfc, threadsize=mesa.THREAD64)),
            pkt7(mesa.CP_EXEC_CS, 0, qreg.cp_exec_cs_1(ngroups_x=global_size[0]), qreg.cp_exec_cs_2(ngroups_y=global_size[1]),
                 qreg.cp_exec_cs_3(_ngroups_z=global_size[2]))]
  else: ins += [pkt7(mesa.CP_RUN_OPENCL, 0)]

  return UOp(Ops.LINEAR, dtypes.void, tuple(ins + cache_flush(ctx, sync=False)))

pm_opsel = PatternMatcher([
  (UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="prg"),), name="call", allow_any_len=True), qcom_program),

  (UPat(Ops.INS, arg="barrier"), lambda ctx: UOp(Ops.LINEAR, dtypes.void, tuple(cache_flush(ctx, invalidate=True, memsync=True)))),
  (UPat(Ops.INS, arg="wait", src=(UPat(name="dst"), UPat(name="val"))), qcom_wait),
  (UPat(Ops.INS, arg="timestamp", src=(UPat(name="dst"),)), qcom_timestamp),
  (UPat(Ops.INS, arg="store", src=(UPat((Ops.BUFFER, Ops.PARAM), name="dst"), UPat(name="val"))), qcom_store),
])

# *****************
# submit: kgsl owns the ringbuffer, so a submit is one ioctl over structs the linker fills in

def ioctl_nr(ioc) -> int: # rebuild the raw request number the autogen wrapper would pass to ioctl(2)
  idir, base, nr, typ = ioc.args
  return (idir << 30) | (typ.SIZE << 16) | (base << 8) | nr

def qcom_submit(devs, lin):
  u32, u64 = functools.partial(UOp.const, dtype=dtypes.uint32), functools.partial(UOp.const, dtype=dtypes.uint64)
  cmdbuf = UOp.placeholder((sum(len(ins.src) for ins in lin.src),), dtypes.uint32, next(UOp.unique_num), device=devs).rtag("cmdbuf")

  # struct kgsl_gpu_command followed by the single command object it points at. kgsl structs are packed, so the fields just go back to back
  req = UOp.placeholder((24,), dtypes.uint32, next(UOp.unique_num), device=devs).rtag("cmdbuf")
  fields = (u64(0), req.getaddr(devs) + u64(kgsl.struct_kgsl_gpu_command.SIZE), u32(kgsl.struct_kgsl_command_object.SIZE), u32(1),
            u64(0), u32(0), u32(0), u64(0), u32(0), u32(0), u32(cast(QCOMDevice, Device[devs[0]]).ctx), u32(0),
            u64(0), cmdbuf.getaddr(devs), u64(cmdbuf.nbytes()), u32(kgsl.KGSL_CMDLIST_IB), u32(0))

  # addresses of both come out of the link, so only the ioctl itself is left to run per submit
  fn = make_buf(devs, tag="func:ioctl").after(make_cmdbuf(lin, devs, buf=cmdbuf),
                                              make_cmdbuf(UOp(Ops.LINEAR, src=(UOp(Ops.INS, src=fields),)), devs, buf=req))
  return fn.index(0).load().call(make_buf(devs, tag="kgsl_fd").index(0).load(), u64(ioctl_nr(kgsl.IOCTL_KGSL_GPU_COMMAND)),
                                 req.getaddr(devs), ret_dtype=dtypes.void)

def encode_queue(q:UOp) -> UOp:
  return qcom_submit(devs:=to_tuple(q.arg[0]), graph_rewrite(q, pm_opsel+pm_flatten_linear, walk=True, ctx=devs, name=f"{q.arg[1]} opsel"))

# *****************

class QCOMAllocator(HCQAllocator['QCOMDevice']):
  def __init__(self, dev:QCOMDevice): super().__init__(dev, supports_copy_from_disk=False, supports_transfer=False)
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    return self.dev._gpu_map(options.external_ptr, size) if options.external_ptr else self.dev._gpu_alloc(size)
  def _do_free(self, opaque, options:BufferSpec): self.dev._gpu_free(opaque)

class QCOMDevice(HCQ2Compiled):
  has_copy_queue = False
  timestamp_divider = 19.2 # QCOM always-on counter: ticks/us
  rt_nbytes = 16 << 20 # phones are tight on memory
  _stack:Buffer|None = None
  pm_stage_copy = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.COPY), UPat(name="dst"), UPat(name="src"))), qcom_host_copy)])
  pm_lower = PatternMatcher([
    (UPat(Ops.PROGRAM, src=(UPat(), UPat(), UPat(), UPat(Ops.BINARY)), name="prg"), build_program),
    (UPat(Ops.CUSTOM_FUNCTION, arg="submit_cmdbuf", src=(UPat(Ops.LINEAR, name="q"),)), encode_queue),
  ])

  def __init__(self, device:str=""):
    self.fd = FileIOInterface('/dev/kgsl-3d0', os.O_RDWR)

    flags = kgsl.KGSL_CONTEXT_PREAMBLE | kgsl.KGSL_CONTEXT_PWR_CONSTRAINT | kgsl.KGSL_CONTEXT_NO_FAULT_TOLERANCE | kgsl.KGSL_CONTEXT_NO_GMEM_ALLOC \
      | flag("KGSL_CONTEXT_PRIORITY", getenv("QCOM_PRIORITY", 8)) | flag("KGSL_CONTEXT_PREEMPT_STYLE", kgsl.KGSL_CONTEXT_PREEMPT_STYLE_FINEGRAIN)
    self.ctx = kgsl.IOCTL_KGSL_DRAWCTXT_CREATE(self.fd, flags=flags).drawctxt_id

    # Set max power
    struct.pack_into('IIQQ', pwr:=memoryview(bytearray(0x18)), 0, 1, self.ctx, mv_address(_:=memoryview(array.array('I', [1]))), 4)
    kgsl.IOCTL_KGSL_SETPROPERTY(self.fd, type=kgsl.KGSL_PROP_PWR_CONSTRAINT, value=mv_address(pwr), sizebytes=pwr.nbytes)

    # Load info about qcom device
    info = kgsl.struct_kgsl_devinfo()
    kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY(self.fd, type=kgsl.KGSL_PROP_DEVICE_INFO, value=ctypes.addressof(info), sizebytes=ctypes.sizeof(info))
    self.gpu_id = (info.chip_id >> 24, (info.chip_id >> 16) & 0xFF, (info.chip_id >> 8) & 0xFF)

    # a7xx start with 730x or 'Cxxx', a8xx starts 'Exxx'
    if self.gpu_id[:2] >= (7, 3): raise RuntimeError(f"Unsupported GPU: chip_id={info.chip_id:#x}")

    if PROFILE:
      System.write_sysfs("/sys/class/kgsl/kgsl-3d0/idle_timer", value="4000000000", msg="Failed to disable suspend mode", expected="4294967276")

    super().__init__(device, QCOMAllocator(self), [QCOMCLRenderer, IR3Renderer], None,
                     arch=("a%d%d%d" + (",IMAGE_PITCH_ALIGNMENT=64" if IMAGE else "")) % self.gpu_id)

    self.pm_bufferize = PatternMatcher([
      (UPat(Ops.PARAM, tag="scratch", name="b"), lambda ctx, b: ctx[0].stack_buffer(b.max_numel())),
      (UPat(Ops.PARAM, tag="kgsl_fd", name="b"), lambda ctx, b: ctx[0].signal(b.tag, ctx[0].fd.fd, device="CPU")),
      (UPat(Ops.PARAM, name="b"), lambda ctx, b: None if not isinstance(b.tag, str) or not b.tag.startswith("func:") else
       ctx[0].signal(b.tag, unwrap(ctypes.cast(getattr(libc.dll, b.tag[5:]), ctypes.c_void_p).value), device="CPU")),
    ]) + self.pm_bufferize

  def stack_buffer(self, size:int) -> Buffer:
    if self._stack is None or self._stack.nbytes < size:
      self._stack = Buffer(self.device, size, dtypes.uint8, options=BufferSpec(nolru=True), preallocate=True)
    return self._stack

  def _wait_signal(self, sig, value:int, timeout:int|None=None):
    # block in kgsl rather than spinning. synchronize() drains the runtime device first, so everything is queued by now
    if sig[0] < value:
      ts = kgsl.IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID(self.fd, context_id=self.ctx, type=kgsl.KGSL_TIMESTAMP_QUEUED).timestamp
      with contextlib.suppress(OSError):
        kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID(self.fd, context_id=self.ctx, timestamp=ts, timeout=int(timeout or self.wait_timeout_ms))
    super()._wait_signal(sig, value, timeout)

  def _gpu_alloc(self, size:int) -> HCQBuffer:
    flags = flag("KGSL_MEMALIGN", alignment_hint:=12) | kgsl.KGSL_MEMFLAGS_USE_CPU_MAP
    alloc = kgsl.IOCTL_KGSL_GPUOBJ_ALLOC(self.fd, size=(bosz:=round_up(size, 1<<alignment_hint)), flags=flags, mmapsize=bosz)
    va_addr = self.fd.mmap(0, bosz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, alloc.id * 0x1000)
    return HCQBuffer(va_addr=va_addr, size=size, meta=(alloc, True), view=MMIOInterface(va_addr, size, fmt='B'), owner=self)

  def _gpu_map(self, ptr:int, size:int) -> HCQBuffer:
    ptr_aligned, size_aligned = (ptr & ~0xfff), round_up(size + (ptr & 0xfff), 0x1000)
    dcache_flush().fxn(ctypes.c_uint64(ptr_line_aligned:=ptr & ~63), ceildiv(ptr + size - ptr_line_aligned, 64))
    try:
      mi = kgsl.IOCTL_KGSL_MAP_USER_MEM(self.fd, hostptr=ptr_aligned, len=size_aligned, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
      return HCQBuffer(mi.gpuaddr + (ptr - ptr_aligned), size=size, meta=(mi, False), view=MMIOInterface(ptr, size, fmt='B'), owner=self)
    except OSError as e:
      if e.errno == 14: return HCQBuffer(va_addr=ptr, size=size, meta=(None, False), view=MMIOInterface(ptr, size, fmt='B'), owner=self)
      raise RuntimeError("Failed to map external pointer to GPU memory") from e

  def _gpu_free(self, mem:HCQBuffer):
    if mem.meta[0] is None: return # external (gpu) ptr
    if not mem.meta[1]: kgsl.IOCTL_KGSL_SHAREDMEM_FREE(self.fd, gpuaddr=mem.meta[0].gpuaddr) # external (cpu) ptr
    else:
      kgsl.IOCTL_KGSL_GPUOBJ_FREE(self.fd, id=mem.meta[0].id)
      FileIOInterface.munmap(mem.va_addr, mem.meta[0].mmapsize)

  def _at_profile_finalize(self):
    super()._at_profile_finalize()
    with contextlib.suppress(RuntimeError): System.write_sysfs("/sys/class/kgsl/kgsl-3d0/idle_timer", "10", "Failed to reenable suspend mode")

if getenv("HCQ1"): from extra.hcq1.ops_qcom import * # noqa: F401, F403 # pylint: disable=unused-import
