#include <cuda_runtime.h>

// out[0] = sum over i of a[i] * b[i], from one block of blockDim.x threads.
//
// Each thread forms one product into `cache` -- dynamic shared memory, one
// float per thread -- and the block then folds those blockDim.x partials down
// to a single number by halving the number of live threads each step:
// blockDim.x/2 additions, then /4, then /8, and so on. blockDim.x is a power
// of two, so the halving is exact and the total is left in cache[0].
//
// Threads with no element must still take part: their slot has to hold the
// additive identity, and they have to reach every barrier the others reach.
__global__ void DotProduct(const float* a, const float* b, float* out, int n) {
  extern __shared__ float cache[];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int local_i = threadIdx.x;

  /// BEGIN CODE HERE (approx 13 lines) ///
  /// END CODE HERE ///
}
