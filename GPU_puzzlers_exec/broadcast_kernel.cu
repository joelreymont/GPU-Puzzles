#include <cuda_runtime.h>

__global__ void Broadcast(float* A, float* B, float* C, int size) {
  int local_i = threadIdx.x;
  int local_j = threadIdx.y;

  /// CODE HERE (approx 4 lines) ///

}
