#include <cuda_runtime.h>

__global__ void Map2DBlock(float* A, float* C, float size) {
  int local_i = blockDim.x * blockIdx.x + threadIdx.x;
  int local_j = blockDim.y * blockIdx.y + threadIdx.y;

  /// CODE HERE (approx 4 lines) ///

}
