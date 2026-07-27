#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>

// The block size the runner launches with. CUB's block collectives are
// templated on it, so it must be a compile-time constant here and it must
// match the launch configuration exactly.
constexpr int TPB = 256;
constexpr int WARP = 32;
constexpr int WARPS = TPB / WARP;  // 8 warp partials per block

// Grid-wide sum into out[0], hand-rolled. Stage 1 reduces inside each warp
// with shuffles, exactly as in puzzle 24. Stage 2 reduces the WARPS partials
// stage 1 left behind. Shuffles move values between lanes of one warp and
// nothing else, so the only way across a warp boundary is memory plus a
// barrier -- that is what makes this block-level rather than warp-level.
__global__ void BlockSumTwoStage(const float* a, float* out, int n) {
  __shared__ float partials[WARPS];  // one slot per warp, written by its lane 0
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x % WARP;
  int warp = threadIdx.x / WARP;
  // Guard with a value, not an early return: every thread must reach both the
  // full-mask shuffles and the barrier below.
  float v = (i < n) ? a[i] : 0.0f;

  /// BEGIN CODE HERE (approx 12 lines) ///
  /// END CODE HERE ///
}

// The same grid-wide sum from cub::BlockReduce. The temp storage and the
// typedef are given; what you supply is the collective call and the single
// thread that publishes the block's result.
__global__ void BlockSumCUB(const float* a, float* out, int n) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = (i < n) ? a[i] : 0.0f;

  /// BEGIN CODE HERE (approx 4 lines) ///
  /// END CODE HERE ///
}

// Segmented inclusive prefix sum: block b owns indices [b*TPB, (b+1)*TPB), and
// out[i] is the sum from that segment's first index up to and including i.
__global__ void BlockScanCUB(const float* a, float* out, int n) {
  using BlockScan = cub::BlockScan<float, TPB>;
  __shared__ typename BlockScan::TempStorage tmp;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = (i < n) ? a[i] : 0.0f;

  /// BEGIN CODE HERE (approx 4 lines) ///
  /// END CODE HERE ///
}

// Application: histogram of a[] over [-1, 1) into nbins equal-width bins.
// Each block privatises the bins in dynamic shared memory, so the atomics
// contend inside a block instead of across the whole grid, then merges its
// private bins into the global histogram once per bin.
__global__ void HistogramShared(const float* a, int* hist, int nbins, int n) {
  extern __shared__ int bins[];  // nbins ints, sized by the launch
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN CODE HERE (approx 10 lines) ///
  /// END CODE HERE ///
}
