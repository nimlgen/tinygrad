# pip3 install pyobjc-framework-Metal pyobjc-framework-Cocoa pyobjc-framework-libdispatch
import os, subprocess, pathlib, functools, ctypes, struct
import Metal, Cocoa, libdispatch # type: ignore
from typing import List, Any, Tuple
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage
from tinygrad.helpers import prod, getenv, DEBUG, DType, dtypes
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawBufferMapped, LRUAllocator

METAL_XCODE = getenv("METAL_XCODE")

class MetalAllocator(LRUAllocator):
  def _do_alloc(self, size, dtype, device, **kwargs): return METAL.device.newBufferWithLength_options_(size*dtype.itemsize, Metal.MTLResourceStorageModeShared)
  def _do_free(self, buf): buf.release()
  def _cached_bufkey(self, size, dtype, device): return (device, size*dtype.itemsize) # Buffers of the same length could be reused, no matter what dtype.

class _METAL:
  def __init__(self):
    self.mtl_buffers_in_flight: List[Any] = []
    self.device = Metal.MTLCreateSystemDefaultDevice()
    self.mtl_queue = self.device.newCommandQueueWithMaxCommandBufferCount_(1024)
    self.allocator = MetalAllocator(self.device.dedicatedMemorySize() or self.device.sharedMemorySize())
  # TODO: is there a better way to do this?
  def synchronize(self):
    for cbuf in self.mtl_buffers_in_flight: cbuf.waitUntilCompleted()
    self.mtl_buffers_in_flight.clear()
METAL = _METAL()

class RawMetalBuffer(RawBufferMapped):
  def __init__(self, size:int, dtype:DType):
    assert dtype != dtypes.double, f"METAL does not support {dtype.name}"
    super().__init__(size, dtype, allocator=METAL.allocator)
  def _buffer(self):
    METAL.synchronize()
    return self._buf.contents().as_buffer(self._buf.length())

def unwrap(x):
  ret, err = x
  assert err is None, str(err)
  return ret

THE_IBUF = None
THE_CNT = 0

class MetalProgram:
  def __init__(self, name:str, prg:str, binary:bool=False):
    if METAL_XCODE:
      air = subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metal', '-x', 'metal', '-c', '-', '-o', '-'], input=prg.encode('utf-8'))
      # NOTE: if you run llvm-dis on "air" you can see the llvm bytecode
      lib = subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metallib', '-', '-o', '-'], input=air)
      data = libdispatch.dispatch_data_create(lib, len(lib), None, None)
      self.library = unwrap(METAL.device.newLibraryWithData_error_(data, None))
    else:
      options = Metal.MTLCompileOptions.alloc().init()
      self.library = unwrap(METAL.device.newLibraryWithSource_options_error_(prg, options, None))
    self.fxn = self.library.newFunctionWithName_(name)
    self.argument_buf = None
    self.packed_data = None
    # hacks to disassemble shader
    if DEBUG >= 5:
      arc = unwrap(METAL.device.newBinaryArchiveWithDescriptor_error_(Metal.MTLBinaryArchiveDescriptor.alloc().init(), None))
      desc = Metal.MTLComputePipelineDescriptor.alloc().init()
      desc.setComputeFunction_(self.fxn)
      unwrap(arc.addComputePipelineFunctionsWithDescriptor_error_(desc, None))
      unwrap(arc.serializeToURL_error_(Cocoa.NSURL.URLWithString_("file:///tmp/shader.bin"), None))
      # clone https://github.com/dougallj/applegpu.git in tinygrad/disassemblers
      os.system(f"cd {pathlib.Path(__file__).parents[2]}/disassemblers/applegpu && python3 compiler_explorer.py /tmp/shader.bin")
    self.pipeline_state = unwrap(METAL.device.newComputePipelineStateWithFunction_error_(self.fxn, None))

  def start_graph(self):
    global THE_IBUF, THE_CNT
    print("string metal graph")
    desc = Metal.MTLIndirectCommandBufferDescriptor.alloc().init()
    desc.setCommandTypes_(Metal.MTLIndirectCommandTypeConcurrentDispatchThreads)
    desc.setInheritBuffers_(False)
    desc.setMaxKernelBufferBindCount_(32) # TODO: Set to 1 with argument buffers
    THE_IBUF = METAL.device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options_(desc, 1024, 0)
    print(THE_IBUF)
    THE_CNT = 0
    pass
    # global GH_STREAM, LAUNCH_INFO
    # GH_STREAM = cuda.Stream()
    # GH_STREAM.begin_capture()
    # LAUNCH_INFO = []
    # # event_init = cuda.Event()
    # return GH_STREAM

  def get_graph(self):
    global THE_IBUF
    return None, THE_IBUF, None
    pass
    # global GH_STREAM, LAUNCH_INFO
    # graph = GH_STREAM.end_capture()
    # # graph.debug_dot_print("test.dot")  # print dotfile of graph
    # instance = graph.instantiate()
    # rr = LAUNCH_INFO
    # GH_STREAM = None
    # LAUNCH_INFO = []
    # return graph, instance, rr

  def capture_node(self, global_size, local_size, *bufs):
    # global THE_IBUF, THE_CNT
    # print(dir(THE_IBUF))
    # encoder = THE_IBUF.indirectComputeCommandAtIndex_(THE_CNT)
    # print(dir(encoder))
    # print(encoder)
    # encoder.setComputePipelineState_(self.pipeline_state)
    # for i,a in enumerate(bufs):
    #   if isinstance(a, RawMetalBuffer): encoder.setKernelBuffer_offset_atIndex_(a._buf, 0, i)
    #   elif isinstance(a, int): encoder.setBytes_length_atIndex_((arg:=ctypes.c_int32(a)), ctypes.sizeof(arg), i)
    #   else: raise RuntimeError(f"arg at index {i} has unsupported type {type(a)}")
    # encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
    THE_CNT += 1
    # encoder.endEncoding()
    # pass
    # global GH_STREAM, LAUNCH_INFO
    # stream_id = GH_STREAM
    # assert getattr(self, 'graph_node', None) is None
    # _, _, graph, deps = stream_id.get_capture_info_v2()
    # graph_node = graph.add_kernel_node(*[x._buf if isinstance(x, RawCUDABuffer) else np.int32(x) if (isinstance(x, int) and not getenv("CUDACPU")) else x for x in args], block=tuple(local_size), grid=tuple(global_size), func=self.prg, dependencies=deps)
    # stream_id.update_capture_dependencies([graph_node], 1)
    # LAUNCH_INFO.append((global_size, local_size, self.prg, graph_node))

  def update_node(self, instance, global_size, local_size, f, graph_node, *args):
    pass
    # instance.kernel_node_set_params(*[x._buf if isinstance(x, RawCUDABuffer) else np.int32(x) if (isinstance(x, int) and not getenv("CUDACPU")) else x for x in args], block=tuple(local_size), grid=tuple(global_size), func=f, kernel_node=graph_node)

  def replay_graph(self, instance):
    # command_buffer = METAL.mtl_queue.commandBuffer()
    # encoder = command_buffer.computeCommandEncoder()
    # print(dir(encoder))
    pass
    # instance.launch()

  def process_args(self, *bufs):
    data = []
    format = ""
    for i,a in enumerate(bufs):
      if isinstance(a, RawMetalBuffer):
        format += "P"
        data.append(a._buf.gpuAddress())
      elif isinstance(a, int):
        format += "i"
        data.append(a)
      else: raise RuntimeError(f"arg at index {i} has unsupported type {type(a)}")
    return struct.pack(format, *data)

  def __call__(self, global_size, local_size, *bufs, wait=False):
    global THE_IBUF
    if THE_IBUF is not None: return self.capture_node(global_size, local_size, *bufs)
    assert prod(local_size) <= self.pipeline_state.maxTotalThreadsPerThreadgroup(), f"local size {local_size} bigger than {self.pipeline_state.maxTotalThreadsPerThreadgroup()} with exec width {self.pipeline_state.threadExecutionWidth()} memory length {self.pipeline_state.staticThreadgroupMemoryLength()}"
    
    packed_data = self.process_args(*bufs)
    if self.argument_buf is None:
      self.argument_buf = METAL.device.newBufferWithLength_options_(len(packed_data), Metal.MTLResourceStorageModeShared)
    self.argument_buf.contents().as_buffer(len(packed_data))[:len(packed_data)] = packed_data
    
    command_buffer = METAL.mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(self.pipeline_state)
    encoder.setBuffer_offset_atIndex_(self.argument_buf, 0, 0)
    for i,a in enumerate(bufs):
      if isinstance(a, RawMetalBuffer): encoder.useResource_usage_(a._buf, Metal.MTLResourceUsageWrite | Metal.MTLResourceUsageRead)
    # print(dir(encoder))
    # for i,a in enumerate(bufs):
    #   if isinstance(a, RawMetalBuffer): encoder.setBuffer_offset_atIndex_(a._buf, 0, i)
    #   elif isinstance(a, int): encoder.setBytes_length_atIndex_((arg:=ctypes.c_int32(a)), ctypes.sizeof(arg), i)
    #   else: raise RuntimeError(f"arg at index {i} has unsupported type {type(a)}")
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
    encoder.endEncoding()
    command_buffer.commit()
    if wait:
      command_buffer.waitUntilCompleted()
      return command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    METAL.mtl_buffers_in_flight.append(command_buffer)

class MTLLanguage(CStyleLanguage):
  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,DType]], local_size:List[int], prekernel:List[str]) -> Tuple[str, List[int], List[int]]:
    prg = super().render_kernel(function_name, kernel, bufs, local_size, prekernel)
    res = ""
    for line in prg.splitlines():
      if line.startswith("kernel "):
        res += f"""
struct Args {{
  {";".join([(f"device {buf[1].name}* {buf[0]}" if buf[1] != dtypes._arg_int32 else f"int {buf[0]}") for buf in bufs])};
}};
kernel void {function_name}(device Args& args, {','.join(self.extra_args)}) {{
  {";".join([(f"device {buf[1].name}* {buf[0]} = args.{buf[0]}" if buf[1] != dtypes._arg_int32 else f"int {buf[0]} = args.{buf[0]}") for buf in bufs])};
\n"""
      else: res += line+"\n"
    return res

renderer = functools.partial(uops_to_cstyle, MTLLanguage(
  kernel_prefix = "#include <metal_stdlib>\nusing namespace metal;\nkernel ", buffer_prefix = "device ", smem_prefix = "threadgroup ", arg_int_prefix = "constant int&",
  barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);", float4 = "float4", uses_ptr_arithmetic=True,
  gid = [f"gid.{chr(120+i)}" for i in range(3)], lid = [f"lid.{chr(120+i)}" for i in range(3)],
  extra_args = ['uint3 gid [[threadgroup_position_in_grid]]', 'uint3 lid [[thread_position_in_threadgroup]]']))
MetalBuffer = Compiled(RawMetalBuffer, LinearizerOptions(), renderer, MetalProgram, METAL.synchronize)
