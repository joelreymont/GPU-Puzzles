#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void NeighborDiff(const float* a, float* out, int n);
extern __global__ void WarpBroadcastBase(const float* a, float* out, int n);
extern __global__ void WarpShiftUp(const float* a, float* out, int n);

int main() {
  print_device_info();

  const int n = 1000;  // not a multiple of 32 or TPB: guards must work
  const int TPB = 256;
  const float tol = 1e-6f;

  float* a = new float[n];
  float* got = new float[n];
  float* want = new float[n];
  fill_rand(a, n, 25);

  float *d_a, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

  const dim3 grid(cdiv(n, TPB));
  bool ok = true;

  // Every kernel owns all n outputs, including the boundary zeros. Poisoning
  // the buffer with 0xff (a NaN bit pattern) before each launch means an
  // element the kernel forgets to write is a FAIL, not a lucky zero.
  ref_neighbor_diff(a, want, n);
  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  NeighborDiff<<<grid, TPB>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("neighbor_diff", got, want, n, tol) && ok;

  ref_sub_warp_base(a, want, n);
  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  WarpBroadcastBase<<<grid, TPB>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("warp_broadcast_base", got, want, n, tol) && ok;

  ref_shift_up(a, want, n);
  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  WarpShiftUp<<<grid, TPB>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("warp_shift_up", got, want, n, tol) && ok;

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_out));
  delete[] a;
  delete[] got;
  delete[] want;
  return ok ? 0 : 1;
}
