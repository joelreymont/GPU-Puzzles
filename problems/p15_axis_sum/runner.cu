#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void AxisSum(const float* a, float* out, int cols);

int main() {
  print_device_info();

  // 64 rows of 1000, one block per row: the grid is one block wide and 64
  // blocks tall, so blockIdx.y selects the row and threadIdx.x walks it.
  // 1000 is not a multiple of the 1024-thread block, so the last 24 lanes of
  // every row hold no element and must contribute the additive identity.
  const int rows = 64;
  const int cols = 1000;
  const int n = rows * cols;
  const int TPB = 1024;
  const dim3 grid(cdiv(cols, TPB), rows);
  const size_t smem = TPB * sizeof(float);

  float* a = new float[n];
  float* want = new float[rows];
  float* got = new float[rows];
  fill_rand(a, n, 15);
  // Row sums are ref_sum applied to each row in turn -- the row is contiguous,
  // so the reference is a pointer offset, not a new function.
  for (int r = 0; r < rows; r++) want[r] = ref_sum(a + r * cols, cols);

  float *d_a, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, rows * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
  // Each output is written outright by one block's thread 0, so a NaN pattern
  // is the right poison: a row no block wrote fails instead of reading zero.
  CHECK_CUDA(cudaMemset(d_out, 0xff, rows * sizeof(float)));

  // `rows` is not a kernel argument: the launch puts exactly one block on
  // each row, so gridDim.y is the row count and blockIdx.y is the row.
  AxisSum<<<grid, TPB, smem>>>(d_a, d_out, cols);
  CHECK_LAUNCH();
  CHECK_CUDA(
      cudaMemcpy(got, d_out, rows * sizeof(float), cudaMemcpyDeviceToHost));

  // 1000 floats per row folded as a depth-10 float tree against the
  // reference's sequential double -- 64 independent draws of puzzle 12's
  // reduction. Measured worst deviation over the 64 rows: 2.6e-6 for the tree,
  // 9.8e-6 for the one-thread sequential sum. 1e-4 is the tightest decade
  // clearing both by more than 10x. See the README.
  const bool ok = compare("axis_sum", got, want, rows, 1e-4f);

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_out));
  delete[] a;
  delete[] want;
  delete[] got;
  return ok ? 0 : 1;
}
