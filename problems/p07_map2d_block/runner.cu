#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void Map2DBlock(const float* a, float* out, int rows,
                                  int cols);

int main() {
  print_device_info();

  // 500 x 700 = 350000 elements, far past one block, tiled by 32 x 8 blocks:
  // 22 x 63 = 1386 of them. 22 * 32 = 704 and 63 * 8 = 504, so the grid
  // overhangs the matrix by 4 columns and 4 rows and both guards are
  // load-bearing on the two edges.
  const int rows = 500;
  const int cols = 700;
  const int n = rows * cols;
  const dim3 block(32, 8);
  const dim3 grid(cdiv(cols, block.x), cdiv(rows, block.y));

  float* a = new float[n];
  float* want = new float[n];
  float* got = new float[n];
  fill_rand(a, n, 7);
  ref_add_scalar(a, want, n, 10.0f);

  float *d_a, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));

  Map2DBlock<<<grid, block>>>(d_a, d_out, rows, cols);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  const bool ok = compare("map2d_block", got, want, n, 0.0f);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_out));
  delete[] a;
  delete[] want;
  delete[] got;
  return ok ? 0 : 1;
}
