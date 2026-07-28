#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void VecAdd(const float* a, const float* b, float* out);

int main() {
  print_device_info();

  const int n = 1000;  // one block, one thread per element

  float* a = new float[n];
  float* b = new float[n];
  float* want = new float[n];
  float* got = new float[n];
  // Two inputs, two seeds: a[i] == b[i] would let a kernel that reads the
  // wrong pointer still pass.
  fill_rand(a, n, 2);
  fill_rand(b, n, 102);
  ref_add(a, b, want, n);

  float *d_a, *d_b, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));

  VecAdd<<<1, n>>>(d_a, d_b, d_out);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  const bool ok = compare("zip", got, want, n, 0.0f);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_out));
  delete[] a;
  delete[] b;
  delete[] want;
  delete[] got;
  return ok ? 0 : 1;
}
