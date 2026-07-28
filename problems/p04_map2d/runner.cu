#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void Map2D(const float* a, float* out, int rows, int cols);

int main() {
  print_device_info();

  // A 25 x 31 matrix under a 32 x 32 block: 1024 threads over 775 elements, so
  // both dimensions overhang and both guards are load-bearing. Neither extent
  // is a multiple of the block dimension, and neither is a multiple of 32.
  const int rows = 25;
  const int cols = 31;
  const int n = rows * cols;
  const dim3 block(32, 32);

  float* a = new float[n];
  float* want = new float[n];
  float* got = new float[n];
  fill_rand(a, n, 4);
  ref_add_scalar(a, want, n, 10.0f);

  float *d_a, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));

  Map2D<<<1, block>>>(d_a, d_out, rows, cols);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  const bool ok = compare("map2d", got, want, n, 0.0f);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_out));
  delete[] a;
  delete[] want;
  delete[] got;
  return ok ? 0 : 1;
}
