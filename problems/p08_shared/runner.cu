#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void Shared(const float* a, float* out, int n);

int main() {
  print_device_info();

  // Same shape as puzzle 6 -- 100003 elements, 391 blocks of 256, a partial
  // last block -- so the only new thing is the shared-memory staging step.
  const int n = 100003;
  const int TPB = 256;
  const int grid = cdiv(n, TPB);
  // Dynamic shared memory: one float per thread in the block, requested at
  // launch and seen by the kernel as `extern __shared__`.
  const size_t smem = TPB * sizeof(float);

  float* a = new float[n];
  float* want = new float[n];
  float* got = new float[n];
  fill_rand(a, n, 8);
  ref_add_scalar(a, want, n, 10.0f);

  float *d_a, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));

  Shared<<<grid, TPB, smem>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  const bool ok = compare("shared", got, want, n, 0.0f);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_out));
  delete[] a;
  delete[] want;
  delete[] got;
  return ok ? 0 : 1;
}
