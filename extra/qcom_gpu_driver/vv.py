from hexdump import hexdump
from tinygrad.runtime.ops_qcom import QcomDevice
from tinygrad.runtime.ops_gpu import GPUDevice
from tinygrad.helpers import getenv
from tinygrad.device import BufferOptions
from tinygrad.dtype import dtypes, ImageDType
if getenv("IOCTL"): import extra.qcom_gpu_driver.opencl_ioctl  # noqa: F401  # pylint: disable=unused-import

entry = """
__kernel void E_81_3_4n2(__global float* data0, read_only image2d_t data1, write_only image2d_t data2, write_only image2d_t data3) {
 const sampler_t smp = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP;
 //const sampler_t smp2 = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP;
  int gidx0 = get_group_id(0); /* 81 */
  int lidx0 = get_local_id(0); /* 3 */
  float4 val0 = read_imagef(data1, smp, (int2)((((gidx0%9)*3)+lidx0),(gidx0/9))); // + read_imagef(data2, smp2, (int2)((((gidx0%9)*3)+lidx0),(gidx0/9)));
  *((__global float4*)(data0+(gidx0*12)+(lidx0*4))) = val0;
  write_imagef(data2, (int2)((((gidx0%9)*3)+lidx0),(gidx0/9)), val0);
  write_imagef(data3, (int2)((((gidx0%9)*3)+lidx0),(gidx0/9)), val0);
}
"""

if __name__ == "__main__":
  dev = GPUDevice()
  lib = dev.compiler.compile(entry)
  hexdump(lib[0xa0c:0xa0c+0xa4])
  hexdump(lib[0xab0:0xab0+0xa4])
  hexdump(lib[0xb54:0xb54+0xa4])

  # data0 = dev.allocator.alloc(0x10000)
  # data1 = dev.allocator.alloc(100 * 100 * 32, BufferOptions(image=dtypes.imagef((100, 100))))
  # data2 = dev.allocator.alloc(100 * 100 * 32, BufferOptions(image=dtypes.imagef((100, 100))))
  # data3 = dev.allocator.alloc(100 * 100 * 32, BufferOptions(image=dtypes.imagef((100, 100))))
  
  # app = dev.runtime("E_81_3_4n2", lib)
  # z = app(data0, data1, data2, data3, wait=True)
