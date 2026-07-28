#include <cuda_runtime.h>

// out[r][c] = a[r][c] + 10 over a row-major rows x cols matrix, from a single
// 32 x 32 block. threadIdx.x walks the columns, threadIdx.y the rows.
__global__ void Map2D(const float* a, float* out, int rows, int cols) {
  int col = threadIdx.x;
  int row = threadIdx.y;

  /// BEGIN CODE HERE (approx 4 lines) ///
  /// END CODE HERE ///
}
