#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void BlockSumTwoStage(const float* a, float* out, int n);
extern __global__ void BlockSumCUB(const float* a, float* out, int n);
extern __global__ void BlockScanCUB(const float* a, float* out, int n);
extern __global__ void HistogramShared(const float* a, int* hist, int nbins,
                                       int n);

int main() {
  print_device_info();

  const int n = 1000;    // not a multiple of 32 or TPB: guards must work
  const int TPB = 256;   // must match the kernels' compile-time TPB
  const int NBINS = 64;  // equal-width bins over [-1, 1)
  // Block reductions accumulate 256 floats per block in a different
  // association order than the reference's sequential double, then combine
  // blocks in nondeterministic atomic order.
  const float tol = 1e-5f;

  float* a = new float[n];
  float* got = new float[n];
  float* want = new float[n];
  int* got_h = new int[NBINS];
  int* want_h = new int[NBINS];
  fill_rand(a, n, 27);

  float *d_a, *d_scan, *d_sum;
  int* d_hist;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_scan, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_sum, sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_hist, NBINS * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

  const dim3 grid(cdiv(n, TPB));
  bool ok = true;

  // The two reduction outputs and the histogram are atomicAdd accumulators:
  // zero is the identity and the only correct initial value, so unlike the
  // per-element buffers below they cannot be poisoned with a NaN pattern. A
  // kernel that writes nothing still fails loudly, because the expected sum
  // and every expected bin count are non-zero.
  const float want_sum = ref_sum(a, n);
  float got_sum = 0.0f;

  CHECK_CUDA(cudaMemset(d_sum, 0, sizeof(float)));
  BlockSumTwoStage<<<grid, TPB>>>(d_a, d_sum, n);
  CHECK_LAUNCH();
  CHECK_CUDA(
      cudaMemcpy(&got_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("block_sum_two_stage", &got_sum, &want_sum, 1, tol) && ok;

  CHECK_CUDA(cudaMemset(d_sum, 0, sizeof(float)));
  BlockSumCUB<<<grid, TPB>>>(d_a, d_sum, n);
  CHECK_LAUNCH();
  CHECK_CUDA(
      cudaMemcpy(&got_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("block_sum_cub", &got_sum, &want_sum, 1, tol) && ok;

  // The scan owns every element, so poison with 0xff (a NaN bit pattern):
  // an output the kernel forgets to write is a FAIL, not a lucky zero.
  ref_seg_incl_scan(a, want, n, TPB);
  CHECK_CUDA(cudaMemset(d_scan, 0xff, n * sizeof(float)));
  BlockScanCUB<<<grid, TPB>>>(d_a, d_scan, n);
  CHECK_LAUNCH();
  CHECK_CUDA(
      cudaMemcpy(got, d_scan, n * sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("block_scan_cub", got, want, n, tol) && ok;

  // Integer counts do not reassociate: compare exactly, across all NBINS.
  ref_hist(a, want_h, NBINS, n);
  CHECK_CUDA(cudaMemset(d_hist, 0, NBINS * sizeof(int)));
  HistogramShared<<<grid, TPB, NBINS * sizeof(int)>>>(d_a, d_hist, NBINS, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got_h, d_hist, NBINS * sizeof(int),
                        cudaMemcpyDeviceToHost));
  ok = compare("histogram_shared", got_h, want_h, NBINS) && ok;

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_scan));
  CHECK_CUDA(cudaFree(d_sum));
  CHECK_CUDA(cudaFree(d_hist));
  delete[] a;
  delete[] got;
  delete[] want;
  delete[] got_h;
  delete[] want_h;
  return ok ? 0 : 1;
}
