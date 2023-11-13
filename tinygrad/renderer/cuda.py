import functools
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage

class CUDALanguage(CStyleLanguage):
  kernel_prefix = "__global__ "
  smem_prefix = "__shared__ "
  smem_prefix_for_cast = False
  arg_int_prefix = "const int"
  barrier = "__syncthreads();" 
  float4 = "make_float4"
  launch_bounds=True
  gid = [f'blockIdx.{chr(120+i)}' for i in range(3)]
  lid = [f'threadIdx.{chr(120+i)}' for i in range(3)]
  xid = [f'(blockIdx.{chr(120+i)}*blockDim.{chr(120+i)}+threadIdx.{chr(120+i)})' for i in range(3)]
  half_prekernel = """
    #include <cuda_fp16.h>
    #include <mma.h>
    using namespace nvcuda;
    struct __align__(8) half4 {
      half x, y, z, w;
      __device__ __forceinline__ half4() = default;
      __device__ __forceinline__ explicit half4(const float4& a): x(__float2half(a.x)), y(__float2half(a.y)), z(__float2half(a.z)), w(__float2half(a.w)) {}
      __device__ __forceinline__ explicit operator float4() const {return make_float4(__half2float(x), __half2float(y), __half2float(z), __half2float(w)); }
    };
    __device__ __forceinline__ half4 make_half4(const half& x, const half& y, const half& z, const half& w) {
      half4 res;
      res.x = x;
      res.y = y;
      res.z = z;
      res.w = w;
      return res;
    }
    __device__ __forceinline__ half max(const half& a, const half& b) { return (a > b ? a : b); }
    """ # if not getenv("PTX") else fromimport("tinygrad.renderer.assembly_ptx", "uops_to_ptx_asm") # assembly_ptx currently isn't supported

CUDARenderer = functools.partial(uops_to_cstyle, CUDALanguage())