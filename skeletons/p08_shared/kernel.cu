#include <cuda_runtime.h>

// out[i] = a[i] + 10, but routed through shared memory: each block stages its
// tile of `a` into `tile`, synchronises, then reads it back. `tile` is dynamic
// shared memory -- the runner requests blockDim.x floats at launch.
__global__ void Shared(const float* a, float* out, int n) {
  extern __shared__ float tile[];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int local_i = threadIdx.x;

  /// BEGIN CODE HERE (approx 7 lines) ///
  /// END CODE HERE ///
}
