#include <cuda_runtime.h>

// out[r][c] = a[r][c] + 10 over a row-major rows x cols matrix, tiled by a 2D
// grid of 2D blocks. Both grid dimensions overhang the matrix.
__global__ void Map2DBlock(const float* a, float* out, int rows, int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  /// BEGIN CODE HERE (approx 4 lines) ///
  /// END CODE HERE ///
}
