#include <cuda_runtime.h>

// Inclusive prefix sum within each block: out[i] = a[base] + ... + a[i], where
// base is the first element of i's block. Blocks cannot see each other, so the
// scan restarts at every block boundary -- this kernel is the per-block half
// of a full-array scan, and the runner's reference is segmented to match.
//
// Hillis-Steele: after the step with offset `off`, every slot holds the sum of
// itself and the `off` slots before it, so the offsets 1, 2, 4, 8, ... reach
// the whole block in log2(blockDim.x) steps. Each step reads a slot another
// thread is about to write, which is what the barriers are for.
//
// `cache` is dynamic shared memory, one float per thread.
__global__ void PrefixSum(const float* a, float* out, int n) {
  extern __shared__ float cache[];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int local_i = threadIdx.x;

  /// BEGIN CODE HERE (approx 14 lines) ///
  /// END CODE HERE ///
}
