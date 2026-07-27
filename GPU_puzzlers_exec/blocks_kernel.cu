#include <cuda_runtime.h>

__global__ void Blocks(float* A, float* C, float size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  /// CODE HERE (approx 3 lines) ///

}
