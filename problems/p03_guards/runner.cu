#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void Guards(const float* a, float* out, int n);

int main() {
  print_device_info();

  // More threads than elements. This is the whole puzzle: the launch geometry
  // does not fit the data, and 24 threads have no element to work on. Nothing
  // in the launch stops them from reading and writing anyway -- only the
  // kernel can.
  const int n = 1000;
  const int TPB = 1024;

  float* a = new float[n];
  float* want = new float[n];
  float* got = new float[n];
  fill_rand(a, n, 3);
  ref_add_scalar(a, want, n, 10.0f);

  float *d_a, *d_out;
  // Exactly n floats, not a rounded-up TPB floats. An unguarded kernel then
  // runs off the end of both allocations, which is what compute-sanitizer
  // reports and what `make check P=03` exists to catch.
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));

  Guards<<<1, TPB>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  const bool ok = compare("guards", got, want, n, 0.0f);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_out));
  delete[] a;
  delete[] want;
  delete[] got;
  return ok ? 0 : 1;
}
