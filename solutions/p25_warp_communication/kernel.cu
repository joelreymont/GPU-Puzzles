#include <cuda_runtime.h>

// Forward difference out[i] = a[i+1] - a[i], with out[n-1] = 0.
// The neighbour lives in the next lane's register — except for lane 31, which
// has no next lane and must fetch a[i+1] from memory itself.
__global__ void NeighborDiff(const float* a, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x % 32;
  // Guard with a value, not an early return: every lane must reach the
  // full-mask shuffle below.
  float v = (i < n) ? a[i] : 0.0f;

  /// BEGIN CODE HERE (approx 6 lines) ///
  float nxt = __shfl_down_sync(0xffffffffu, v, 1);
  if (lane == 31) {
    nxt = (i + 1 < n) ? a[i + 1] : 0.0f;
  }
  if (i < n) {
    out[i] = (i + 1 < n) ? (nxt - v) : 0.0f;
  }
  /// END CODE HERE ///
}

// out[i] = a[i] - a[base], base = (i/32)*32 — the global index lane 0 of this
// warp is holding, since blockDim.x is a multiple of 32.
__global__ void WarpBroadcastBase(const float* a, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = (i < n) ? a[i] : 0.0f;

  /// BEGIN CODE HERE (approx 3 lines) ///
  float base = __shfl_sync(0xffffffffu, v, 0);
  if (i < n) {
    out[i] = v - base;
  }
  /// END CODE HERE ///
}

// Shift each warp's values one lane up: out[i] = a[i-1], and 0 at lane 0.
// __shfl_up_sync leaves the bottom `delta` lanes holding their own value, so
// the zero at the warp head is yours to write.
__global__ void WarpShiftUp(const float* a, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x % 32;
  float v = (i < n) ? a[i] : 0.0f;

  /// BEGIN CODE HERE (approx 4 lines) ///
  float prev = __shfl_up_sync(0xffffffffu, v, 1);
  if (i < n) {
    out[i] = (lane == 0) ? 0.0f : prev;
  }
  /// END CODE HERE ///
}
