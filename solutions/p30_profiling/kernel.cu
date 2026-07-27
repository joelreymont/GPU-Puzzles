#include <cuda_runtime.h>

// Two kernels, one answer: out[i] is the total of i's aligned 32-element
// group, written by every thread of that group. A trailing group shorter than
// 32 sums only the elements that exist.
//
// The groups are 32 elements wide and aligned, and the runner launches blocks
// of 256 threads -- a multiple of 32 -- so a group is exactly one warp's worth
// of consecutive elements and every thread of a warp shares a group. That is
// the whole reason both kernels are possible; it is also the reason they
// perform so differently.
constexpr int GROUP = 32;  // = warpSize: one group is one warp's elements

// The obvious way. Each thread works alone: it walks all GROUP elements of its
// own group, pulling every one of them out of GLOBAL memory, and accumulates
// the total itself. No shuffles, no shared memory, no cooperation of any kind
// -- and therefore every element of the group is fetched GROUP times over,
// once by each thread that needs it.
__global__ void GroupSumRedundant(const float* a, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int base = (i / GROUP) * GROUP;  // first element of i's group

  /// BEGIN CODE HERE (approx 6 lines) ///
  float s = 0.0f;
  for (int j = 0; j < GROUP; j++) {
    const int k = base + j;
    s += (k < n) ? a[k] : 0.0f;  // value guard: the last group is short
  }
  if (i < n) out[i] = s;
  /// END CODE HERE ///
}

// Puzzle 26's butterfly, applied. Each thread reads exactly ONE element -- its
// own -- so the warp's 32 loads coalesce into one request, and the group total
// is then built in registers by log2(GROUP) __shfl_xor_sync steps. Every lane
// ends up holding the total, so every lane can store its own output.
__global__ void GroupSumStream(const float* a, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN CODE HERE (approx 4 lines) ///
  float v = (i < n) ? a[i] : 0.0f;  // value guard, so no lane leaves the warp
  for (int mask = GROUP / 2; mask > 0; mask >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, mask);
  if (i < n) out[i] = v;
  /// END CODE HERE ///
}
