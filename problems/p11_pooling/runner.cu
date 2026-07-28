#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void Pooling(const float* a, float* out, int n);

int main() {
  print_device_info();

  // Puzzle 6's geometry -- 100003 elements, 391 blocks of 256, a partial last
  // block -- but now every output depends on the two elements before it, so
  // the block boundaries are real: thread 0 of block 1 needs a[254] and
  // a[255], which no thread of block 1 stages. That is the halo.
  const int n = 100003;
  const int TPB = 256;
  const int W = 3;  // window length; must match the kernel's compile-time W
  const int grid = cdiv(n, TPB);
  // The tile is one element per thread plus the W-1 halo elements sitting to
  // its left, which is what makes it wider than the block.
  const size_t smem = (TPB + W - 1) * sizeof(float);

  float* a = new float[n];
  float* want = new float[n];
  float* got = new float[n];
  fill_rand(a, n, 11);
  ref_trail_sum(a, want, n, W);

  float *d_a, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));

  Pooling<<<grid, TPB, smem>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  // Tolerance zero, and it is not luck: fill_rand emits multiples of 2^-23 in
  // [-1, 1), so any two of them add exactly in float, and the third addition
  // is then the single rounding the double reference also performs. Measured:
  // 0.000e+00 deviation over all 100003 outputs, in either summation order.
  // See the README.
  const bool ok = compare("pooling", got, want, n, 0.0f);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_out));
  delete[] a;
  delete[] want;
  delete[] got;
  return ok ? 0 : 1;
}
