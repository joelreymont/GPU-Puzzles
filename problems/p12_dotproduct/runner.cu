#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void DotProduct(const float* a, const float* b, float* out,
                                  int n);

int main() {
  print_device_info();

  // One block, because the reduction this puzzle teaches is a block-level one:
  // 1024 threads is the hardware maximum and 1000 elements is deliberately not
  // a multiple of it, so 24 lanes hold no element and must contribute the
  // additive identity rather than whatever was in shared memory.
  const int n = 1000;
  const int TPB = 1024;
  const size_t smem = TPB * sizeof(float);

  float* a = new float[n];
  float* b = new float[n];
  fill_rand(a, n, 12);
  fill_rand(b, n, 112);
  const float want = ref_dot(a, b, n);
  float got = 0.0f;

  float *d_a, *d_b, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice));
  // One thread writes this output outright -- it is not an atomic accumulator
  // -- so poison it with a NaN pattern: a kernel that never writes fails.
  CHECK_CUDA(cudaMemset(d_out, 0xff, sizeof(float)));

  DotProduct<<<1, TPB, smem>>>(d_a, d_b, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));

  // A 1000-term dot product summed as a depth-10 float tree, against the
  // reference's sequential double. One seed is one draw, so the tolerance is
  // set from a 512-seed sweep of this exact reduction: worst 2.7e-6 for the
  // tree and 9.0e-6 for the one-thread sequential sum, which is the other
  // legitimate way to finish this kernel. 1e-4 is the tightest decade clearing
  // both by more than 10x. See the README.
  const bool ok = compare("dot_product", &got, &want, 1, 1e-4f);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_out));
  delete[] a;
  delete[] b;
  return ok ? 0 : 1;
}
