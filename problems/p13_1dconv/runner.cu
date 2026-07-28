#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void Conv1D(const float* a, const float* b, float* out, int n,
                              int k);

int main() {
  print_device_info();

  // These three must match the kernel's compile-time constants: the block
  // shape is baked into the shared-memory layout, and the launch has to
  // request exactly the tile the kernel indexes.
  const int TPB = 256;
  const int MAX_CONV = 32;
  const int TPB_MAX_CONV = TPB + MAX_CONV;

  // 100003 signal samples, 391 blocks of 256, partial last block. k = 15 is
  // odd, is not a divisor of 256, and is well under MAX_CONV -- so the halo is
  // 14 elements wide, the staging loop's guard on k is live, and the tail of
  // the tile that the kernel never reads is a real part of the allocation
  // rather than a coincidence.
  const int n = 100003;
  const int k = 15;
  // The kernel's shared layout is sized from MAX_CONV, and its staging assumes
  // one thread per halo slot and one per filter tap. Both are properties of
  // the launch, so they are checked here, at compile time, rather than left as
  // something a future edit can quietly violate.
  static_assert(k <= MAX_CONV, "filter longer than the tile has room for");
  static_assert(k <= TPB, "fewer threads than filter taps to stage them");
  const size_t smem = (TPB_MAX_CONV + MAX_CONV) * sizeof(float);
  const int grid = cdiv(n, TPB);

  float* a = new float[n];
  float* b = new float[k];
  float* want = new float[n];
  float* got = new float[n];
  fill_rand(a, n, 13);
  fill_rand(b, k, 113);
  ref_conv1d(a, b, want, n, k);

  float *d_a, *d_b, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b, k * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b, k * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));

  Conv1D<<<grid, TPB, smem>>>(d_a, d_b, d_out, n, k);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  // 15 products accumulated in float against the reference's double. Measured
  // worst deviation over all 100003 outputs: 4.2e-7 walking the filter
  // forwards, 5.4e-7 backwards. 1e-5 is the tightest decade clearing both by
  // more than 10x. See the README.
  const bool ok = compare("conv1d", got, want, n, 1e-5f);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_out));
  delete[] a;
  delete[] b;
  delete[] want;
  delete[] got;
  return ok ? 0 : 1;
}
