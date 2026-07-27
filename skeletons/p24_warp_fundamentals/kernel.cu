#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// Dot product: hand-rolled warp reduction with register shuffles.
__global__ void WarpDotShfl(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Guard with a value, not an early return: every lane must reach the
  // full-mask shuffles below.
  float partial = (i < n) ? a[i] * b[i] : 0.0f;

  /// BEGIN CODE HERE (approx 6 lines) ///
  /// END CODE HERE ///
}

// Same reduction through cooperative groups.
__global__ void WarpDotCG(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float partial = (i < n) ? a[i] * b[i] : 0.0f;
  cg::thread_block_tile<32> warp =
      cg::tiled_partition<32>(cg::this_thread_block());

  /// BEGIN CODE HERE (approx 2 lines) ///
  /// END CODE HERE ///
}

// Count elements above thresh: one ballot per warp, no per-thread atomics.
__global__ void WarpCountGT(const float* a, int* out, float thresh, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  bool pred = (i < n) && (a[i] > thresh);

  /// BEGIN CODE HERE (approx 3 lines) ///
  /// END CODE HERE ///
}
