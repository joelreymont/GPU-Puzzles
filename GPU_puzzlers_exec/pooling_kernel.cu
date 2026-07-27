#include <cuda_runtime.h>

__global__ void Pooling(float* A, float* C, float size) {
  extern __shared__ float sharedMem[];
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int local_i = threadIdx.x;

  /// CODE HERE (approx 7 lines) ///

}
