#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void Blocks(const float* a, float* out, int n);

int main() {
  print_device_info();

  // Past one block: 100003 elements at 256 threads per block is 391 blocks,
  // and 100003 = 390 * 256 + 163, so the last block is 163/256 full and its
  // guard decides 93 threads' behaviour. A single-block kernel, or one that
  // forgets blockIdx, cannot get past element 255.
  const int n = 100003;
  const int TPB = 256;
  const int grid = cdiv(n, TPB);

  float* a = new float[n];
  float* want = new float[n];
  float* got = new float[n];
  fill_rand(a, n, 6);
  ref_add_scalar(a, want, n, 10.0f);

  float *d_a, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));

  Blocks<<<grid, TPB>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  const bool ok = compare("blocks", got, want, n, 0.0f);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_out));
  delete[] a;
  delete[] want;
  delete[] got;
  return ok ? 0 : 1;
}
