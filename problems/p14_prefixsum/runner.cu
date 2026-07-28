#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void PrefixSum(const float* a, float* out, int n);

int main() {
  print_device_info();

  // 100003 elements in 391 blocks of 256. The scan restarts at every block
  // boundary -- this is the per-block half of a full scan, and blocks cannot
  // see each other -- so the reference is the same segmented scan puzzle 27
  // checks CUB against, with the segment set to the block size.
  const int n = 100003;
  const int TPB = 256;
  const int grid = cdiv(n, TPB);
  const size_t smem = TPB * sizeof(float);

  float* a = new float[n];
  float* want = new float[n];
  float* got = new float[n];
  fill_rand(a, n, 14);
  ref_seg_incl_scan(a, want, n, TPB);

  float *d_a, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));

  PrefixSum<<<grid, TPB, smem>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  // Hillis-Steele reassociates every partial sum in the segment, and the last
  // element of a segment is a 256-term float sum against the reference's
  // sequential double. Measured worst deviation over all 100003 outputs:
  // 2.1e-6 for Hillis-Steele, 8.1e-6 for the one-thread sequential scan, which
  // is the other legitimate way to produce this answer. 1e-4 is the tightest
  // decade clearing both by more than 10x. See the README.
  const bool ok = compare("prefix_sum", got, want, n, 1e-4f);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_out));
  delete[] a;
  delete[] want;
  delete[] got;
  return ok ? 0 : 1;
}
