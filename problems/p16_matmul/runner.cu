#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void Matmul(const float* a, const float* b, float* out,
                              int n);

int main() {
  print_device_info();

  // Must match the kernel's compile-time TPB: the tile is a TPB x TPB square
  // of shared memory and the block is a TPB x TPB square of threads, one
  // thread per tile element.
  const int TPB = 16;

  // 500 x 500. 500 is not a multiple of 16: the grid is 32 x 32 blocks
  // covering 512 x 512, so the right-hand and bottom blocks overhang by 12 in
  // each direction, and the K loop's last tile is 12 wide. Every guard in the
  // kernel is load-bearing at 500 and none of them is at 512.
  const int n = 500;
  const int nn = n * n;
  const dim3 block(TPB, TPB);
  const dim3 grid(cdiv(n, TPB), cdiv(n, TPB));
  const size_t smem = 2 * TPB * TPB * sizeof(float);

  float* a = new float[nn];
  float* b = new float[nn];
  float* want = new float[nn];
  float* got = new float[nn];
  // Different seeds, so A != B and C = A*B is not symmetric. That is what
  // makes a transposed index a wrong answer rather than a relabelled one.
  fill_rand(a, nn, 16);
  fill_rand(b, nn, 116);
  ref_gemm(a, b, want, n, n, n);

  float *d_a, *d_b, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, nn * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b, nn * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, nn * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a, nn * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b, nn * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_out, 0xff, nn * sizeof(float)));

  Matmul<<<grid, block, smem>>>(d_a, d_b, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(
      cudaMemcpy(got, d_out, nn * sizeof(float), cudaMemcpyDeviceToHost));

  // 500 products per output, accumulated as one 500-long serial float chain,
  // against the reference's double. Measured worst deviation over all 250000
  // outputs: 1.27e-5 for that chain, 3.6e-6 when each tile is summed into its
  // own partial first -- both legitimate. 1e-3 is the tightest decade clearing
  // both by more than 10x; 1e-4 would leave only 7.9x on the shipped kernel.
  // The worst case is an output of magnitude 0.29 built from 500 products of
  // magnitude up to 1, so that figure is cancellation, not a wrong answer.
  // See the README.
  const bool ok = compare("matmul", got, want, nn, 1e-3f);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_out));
  delete[] a;
  delete[] b;
  delete[] want;
  delete[] got;
  return ok ? 0 : 1;
}
