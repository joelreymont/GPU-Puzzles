#include <cuda_runtime.h>

// out[r][c] = a[r] + b[c]: a column vector of `rows` elements plus a row
// vector of `cols` elements, into a row-major rows x cols matrix. One 32 x 32
// block; threadIdx.x walks the columns, threadIdx.y the rows.
__global__ void Broadcast(const float* a, const float* b, float* out, int rows,
                          int cols) {
  int col = threadIdx.x;
  int row = threadIdx.y;

  /// BEGIN CODE HERE (approx 4 lines) ///
  /// END CODE HERE ///
}
