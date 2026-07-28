#include <cuda_runtime.h>

// out[i] = a[i] + b[i]. One block, one thread per element.
__global__ void VecAdd(const float* a, const float* b, float* out) {
  int i = threadIdx.x;

  /// BEGIN CODE HERE (approx 1 line) ///
  /// END CODE HERE ///
}
