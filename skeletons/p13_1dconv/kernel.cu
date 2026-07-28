#include <cuda_runtime.h>

// The block shape is a compile-time fact here, because the shared-memory
// layout is built out of it: one signal tile of TPB elements plus room for the
// widest halo any permitted kernel length needs, then the filter itself.
// MAX_CONV is the cap on `k`, not its value -- the runtime `k` is smaller.
const int TPB = 256;
const int MAX_CONV = 32;
const int TPB_MAX_CONV = TPB + MAX_CONV;

// out[i] = sum over j in [0, k) of a[i + j] * b[j], with a[i + j] treated as 0
// once i + j runs off the end of the signal.
//
// Every output reads k consecutive signal samples, and consecutive outputs
// overlap in k-1 of them, so a block of TPB threads touches TPB + k - 1
// distinct samples: its own tile, plus a halo of k-1 samples belonging to the
// block on its right. Staging the union once and reading it k times from
// shared memory is the whole point.
//
// The runner hands over one dynamic shared allocation; carving it into the two
// arrays is part of the task. shared_a is TPB_MAX_CONV floats, shared_b is
// MAX_CONV.
__global__ void Conv1D(const float* a, const float* b, float* out, int n,
                       int k) {
  extern __shared__ float smem[];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int local_i = threadIdx.x;

  /// BEGIN CODE HERE (approx 19 lines) ///
  /// END CODE HERE ///
}
