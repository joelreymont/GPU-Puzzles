#include <cuda_runtime.h>

__global__ void Map2D(float* A, float* C, float size) {
  int local_i = threadIdx.x;
  int local_j = threadIdx.y;

  /// CODE HERE (approx 4 lines) ///

}
