#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void ScalarAdd(const float* a, float* out);

int main() {
  print_device_info();

  // One block, one thread per element. 1000 is inside the 1024-thread block
  // limit and is deliberately not a multiple of 32: the last warp is partial,
  // which is true of almost every real launch and is worth seeing from the
  // start.
  const int n = 1000;

  float* a = new float[n];
  float* want = new float[n];
  float* got = new float[n];
  fill_rand(a, n, 1);
  ref_add_scalar(a, want, n, 10.0f);

  float *d_a, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
  // Poison the output with 0xff bytes, which read back as NaN. An element the
  // kernel never writes then fails the comparison instead of quietly passing
  // because zero happened to be close enough.
  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));

  ScalarAdd<<<1, n>>>(d_a, d_out);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  // One IEEE float add on the CPU, the same one on the GPU: bit-identical, so
  // the tolerance is exactly zero. Nothing here reassociates.
  const bool ok = compare("map", got, want, n, 0.0f);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_out));
  delete[] a;
  delete[] want;
  delete[] got;
  return ok ? 0 : 1;
}
