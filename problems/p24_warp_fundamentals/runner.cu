#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void WarpDotShfl(const float* a, const float* b, float* out,
                                   int n);
extern __global__ void WarpDotCG(const float* a, const float* b, float* out,
                                 int n);
extern __global__ void WarpCountGT(const float* a, int* out, float thresh,
                                   int n);

int main() {
  print_device_info();

  const int n = 1000;  // not a multiple of 32 or TPB: guards must work
  const int TPB = 256;
  const float thresh = 0.5f;
  const float tol = 1e-5f;

  float* a = new float[n];
  float* b = new float[n];
  fill_rand(a, n, 1);
  fill_rand(b, n, 2);
  const float want_dot = ref_dot(a, b, n);
  const int want_cnt = ref_count_gt(a, n, thresh);

  float *d_a, *d_b, *d_out;
  int* d_cnt;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_cnt, sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

  const dim3 grid(cdiv(n, TPB));
  bool ok = true;

  float got = 0.0f;
  CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float)));
  WarpDotShfl<<<grid, TPB>>>(d_a, d_b, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("warp_dot_shfl", &got, &want_dot, 1, tol) && ok;

  CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float)));
  WarpDotCG<<<grid, TPB>>>(d_a, d_b, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("warp_dot_cg", &got, &want_dot, 1, tol) && ok;

  int gotc = 0;
  CHECK_CUDA(cudaMemset(d_cnt, 0, sizeof(int)));
  WarpCountGT<<<grid, TPB>>>(d_a, d_cnt, thresh, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(&gotc, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));
  ok = compare("warp_count_ballot", &gotc, &want_cnt, 1) && ok;

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_out));
  CHECK_CUDA(cudaFree(d_cnt));
  delete[] a;
  delete[] b;
  return ok ? 0 : 1;
}
