#include <cuda_runtime.h>

// out[r] = sum over c of a[r][c], for a row-major rows x cols matrix: one
// number per row, produced by one block per row.
//
// The grid is two-dimensional and only one of its axes indexes elements.
// blockIdx.y is the row -- the whole block works on that row and nothing else,
// and the launch puts exactly one block on each row, so `rows` is not a kernel
// argument -- while blockIdx.x tiles the row itself, which for this launch is
// a single block wide. The reduction inside the block is puzzle 12's,
// unchanged.
//
// `cache` is dynamic shared memory, one float per thread.
__global__ void AxisSum(const float* a, float* out, int cols) {
  extern __shared__ float cache[];

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int local_i = threadIdx.x;
  int row = blockIdx.y;

  /// BEGIN CODE HERE (approx 13 lines) ///
  cache[local_i] = (col < cols) ? a[row * cols + col] : 0.0f;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (local_i < s) {
      cache[local_i] += cache[local_i + s];
    }
    __syncthreads();
  }

  if (local_i == 0) {
    out[row] = cache[0];
  }
  /// END CODE HERE ///
}
