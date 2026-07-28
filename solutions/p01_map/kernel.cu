#include <cuda_runtime.h>

// out[i] = a[i] + 10. One block, one thread per element.
__global__ void ScalarAdd(const float* a, float* out) {
  int i = threadIdx.x;

  /// BEGIN CODE HERE (approx 1 line) ///
  out[i] = a[i] + 10.0f;
  /// END CODE HERE ///
}
