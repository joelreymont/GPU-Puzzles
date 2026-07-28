#include <cuda_runtime.h>

// out[i] = a[i] + 10 for the n elements that exist. The block is launched with
// 1024 threads over n = 1000 elements, so 24 threads have no element.
__global__ void Guards(const float* a, float* out, int n) {
  int i = threadIdx.x;

  /// BEGIN CODE HERE (approx 3 lines) ///
  /// END CODE HERE ///
}
