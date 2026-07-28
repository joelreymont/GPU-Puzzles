#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void Broadcast(const float* a, const float* b, float* out,
                                 int rows, int cols);

int main() {
  print_device_info();

  // Column vector a (rows elements) plus row vector b (cols elements) into a
  // rows x cols matrix. rows != cols on purpose: a kernel that swaps the two
  // vectors, or indexes the output the wrong way round, cannot pass a
  // non-square case.
  const int rows = 25;
  const int cols = 31;
  const int n = rows * cols;
  const dim3 block(32, 32);

  float* a = new float[rows];
  float* b = new float[cols];
  float* want = new float[n];
  float* got = new float[n];
  fill_rand(a, rows, 5);
  fill_rand(b, cols, 105);
  ref_bcast_add(a, b, want, rows, cols);

  float *d_a, *d_b, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, rows * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b, cols * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a, rows * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b, cols * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));

  Broadcast<<<1, block>>>(d_a, d_b, d_out, rows, cols);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  const bool ok = compare("broadcast", got, want, n, 0.0f);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_out));
  delete[] a;
  delete[] b;
  delete[] want;
  delete[] got;
  return ok ? 0 : 1;
}
