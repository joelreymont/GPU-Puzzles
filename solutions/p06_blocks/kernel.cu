#include <cuda_runtime.h>

// out[i] = a[i] + 10 over n elements spread across a 1D grid of blocks.
__global__ void Blocks(const float* a, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN CODE HERE (approx 3 lines) ///
  if (i < n) {
    out[i] = a[i] + 10.0f;
  }
  /// END CODE HERE ///
}
