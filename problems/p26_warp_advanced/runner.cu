#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void ButterflyAllReduce(const float* a, float* out, int n);
extern __global__ void WarpInclusiveScan(const float* a, float* out, int n);
extern __global__ void WarpScanCG(const float* a, float* out, int n);

int main() {
  print_device_info();

  const int n = 1000;  // not a multiple of 32 or TPB: guards must work
  const int TPB = 256;
  // Looser than puzzle 25: these kernels accumulate 32 floats in a different
  // association order than the reference, so a few ulps of drift is expected.
  const float tol = 1e-5f;

  float* a = new float[n];
  float* got = new float[n];
  float* want = new float[n];
  fill_rand(a, n, 26);

  float *d_a, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

  const dim3 grid(cdiv(n, TPB));
  bool ok = true;

  // Every kernel owns all n outputs. Poisoning the buffer with 0xff (a NaN bit
  // pattern) before each launch means an element the kernel forgets to write is
  // a FAIL, not a lucky zero.
  ref_group_sum_all(a, want, n);
  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  ButterflyAllReduce<<<grid, TPB>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("butterfly_all_reduce", got, want, n, tol) && ok;

  // Both scan kernels answer to the same reference.
  ref_group_incl_scan(a, want, n);

  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  WarpInclusiveScan<<<grid, TPB>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("warp_inclusive_scan", got, want, n, tol) && ok;

  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  WarpScanCG<<<grid, TPB>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("warp_scan_cg", got, want, n, tol) && ok;

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_out));
  delete[] a;
  delete[] got;
  delete[] want;
  return ok ? 0 : 1;
}
