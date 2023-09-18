import time
import numpy as np
from tinygrad.helpers import dtypes, getenv, prod
from tinygrad.runtime.ops_hip import RawHIPBuffer, HIPProgram
import extra.hip_wrapper as hip

import numpy
a = numpy.zeros((4, 4)).astype(numpy.float32)
b = numpy.zeros((4, 4)).astype(numpy.float32)
result = numpy.zeros_like(b)

a_gpu = RawHIPBuffer.fromCPU(a)
b_gpu = RawHIPBuffer.fromCPU(b)

prog_plus = HIPProgram("plus", """
#define F32
extern "C" __global__ void plus(float *a, int num)
{
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] += num;
}
""")

prog_times = HIPProgram("times", """
#define F32
extern "C" __global__ void times(float *a, float *b)
{
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= b[idx];
}""")

print(prog_plus)
hip.hipSetDevice(b_gpu._device)

stream = hip.hipStreamCreate()
print(stream)
hip.hipStreamBeginCapture(stream)
prog_plus((1, 1, 1), (4, 4, 1), a_gpu, 2, stream=stream)
prog_plus((1, 1, 1), (4, 4, 1), b_gpu, 1, stream=stream)
_, _, graph, deps = hip.hipStreamGetCaptureInfo_v2(stream)

# print(deps)
print("prgs", prog_plus.prgs)
params = hip.buildKernelNodeParams(a_gpu, 3, func=prog_plus.prgs[b_gpu._device], block=(4, 4, 1))
node1 = hip.hipGraphAddKernelNode(graph, deps, params)
hip.hipStreamUpdateCaptureDependencies(stream, [node1])
_, _, graph, deps = hip.hipStreamGetCaptureInfo_v2(stream)
print(node1, deps)

# prog_plus((1, 1, 1), (4, 4, 1), b_gpu, 3, stream=stream)
prog_times((1, 1, 1), (4, 4, 1), b_gpu, a_gpu, stream=stream)
graph = hip.hipStreamEndCapture(stream)
print(graph)
instance = hip.hipGraphInstantiate(graph)
print(instance)

params2 = hip.buildKernelNodeParams(a_gpu, 100, func=prog_plus.prgs[b_gpu._device], block=(4, 4, 1))
hip.hipGraphExecKernelNodeSetParams(instance, node1, params2)
hip.hipGraphLaunch(instance)

print("original arrays:")
print(a)
print(b)
print("(0+2)x(0+3) = 6, using a kernel graph of 3 kernels:")
print(a_gpu.toCPU())
print(b_gpu.toCPU())
