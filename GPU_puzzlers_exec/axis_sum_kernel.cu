#include <cuda_runtime.h>

__global__ void AxisSum(float* A, float* C, int size) {
  extern __shared__ float cache[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int local_i = threadIdx.x;
  int batch = blockIdx.y;

  /// CODE HERE (approx 14 lines) ///


}
