#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// out[i] = the total of i's aligned 32-element group, written by EVERY lane.
// Butterfly with __shfl_xor_sync: after the last step all 32 lanes hold the
// total, unlike the __shfl_down_sync ladder which leaves it only in lane 0.
__global__ void ButterflyAllReduce(const float* a, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Guard with a value, not an early return: every lane must reach the
  // full-mask shuffles below.
  float v = (i < n) ? a[i] : 0.0f;

  /// BEGIN CODE HERE (approx 4 lines) ///
  for (int mask = 16; mask > 0; mask >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, mask);
  }
  if (i < n) out[i] = v;
  /// END CODE HERE ///
}

// Inclusive prefix sum within each aligned 32-element group:
// out[i] = a[base] + ... + a[i], base = (i/32)*32. Hand-rolled Hillis-Steele.
__global__ void WarpInclusiveScan(const float* a, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x % 32;
  float v = (i < n) ? a[i] : 0.0f;

  /// BEGIN CODE HERE (approx 5 lines) ///
  for (int delta = 1; delta < 32; delta <<= 1) {
    float up = __shfl_up_sync(0xffffffffu, v, delta);
    if (lane >= delta) v += up;
  }
  if (i < n) out[i] = v;
  /// END CODE HERE ///
}

// The same segmented inclusive scan through cooperative groups. The tile is
// already built; cg::inclusive_scan generates the ladder for you.
__global__ void WarpScanCG(const float* a, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = (i < n) ? a[i] : 0.0f;
  cg::thread_block_tile<32> warp =
      cg::tiled_partition<32>(cg::this_thread_block());

  /// BEGIN CODE HERE (approx 2 lines) ///
  float s = cg::inclusive_scan(warp, v, cg::plus<float>());
  if (i < n) out[i] = s;
  /// END CODE HERE ///
}
