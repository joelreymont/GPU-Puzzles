#include <cuda_runtime.h>

// The tile is a compile-time square, TPB x TPB, and the block has exactly one
// thread per tile element -- so a thread stages one element of A's tile and
// one of B's tile per step, and the shared-memory layout is known at compile
// time.
const int TPB = 16;

// out = a * b for square row-major n x n matrices.
//
// The naive version reads a whole row of `a` and a whole column of `b` per
// output, so each element of `a` is fetched from global memory n times. Tiling
// fixes that: the K axis is walked TPB elements at a time, each block stages
// one TPB x TPB tile of `a` and one of `b` into shared memory, and every
// thread in the block then reads those tiles TPB times each. One global load
// per element per tile step instead of one per multiply.
//
// The runner hands over one dynamic shared allocation; carving it into the two
// TPB x TPB tiles is part of the task. Both are indexed row-major within the
// tile, [local_row * TPB + local_col].
//
// Out-of-range tile slots are staged as 0 rather than skipped, so the inner
// product loop needs no bounds test: a zero contributes nothing to the sum.
__global__ void Matmul(const float* a, const float* b, float* out, int n) {
  extern __shared__ float smem[];

  int col = blockIdx.x * blockDim.x + threadIdx.x;  // column of `out`
  int row = blockIdx.y * blockDim.y + threadIdx.y;  // row of `out`
  int lcol = threadIdx.x;
  int lrow = threadIdx.y;

  /// BEGIN CODE HERE (approx 20 lines) ///
  /// END CODE HERE ///
}
