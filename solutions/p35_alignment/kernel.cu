#include <cuda_runtime.h>

// Three kernels, one arithmetic result: out[i] = s * a[i] + b[i]. Two flops
// and twelve bytes of traffic per element, which is as memory-bound as a
// kernel gets -- so nothing here is about arithmetic and everything is about
// how the twelve bytes are asked for.
//
// The unit the memory system actually moves is the 32-byte SECTOR, four to a
// 128-byte cache line. A warp-wide access is broken into requests, each request
// is broken into sectors, and the sectors are what cost time. Thirty-two lanes
// reading thirty-two consecutive floats want 128 contiguous bytes, which is
// exactly four sectors when the first of them starts on a 32-byte boundary --
// the minimum possible, and the number every one of these kernels should be
// measured against.
//
// SaxpyScalar asks for those four sectors with one 32-bit load per lane.
// SaxpyVec4 asks for the same bytes with one 128-bit load per lane, four times
// fewer instructions. SaxpyMisaligned asks for the same *count* of bytes, one
// float to the right, and gets five sectors instead of four. Which of those
// three differences the clock notices is the puzzle.

// out[i] = s * a[i] + b[i], one element per thread, one plain 32-bit load per
// operand. The baseline the other two are measured against, and not a straw
// man: consecutive lanes read consecutive floats, so the access is perfectly
// coalesced and its request is the 4-sector minimum.
//
// `n` is not a multiple of the block size, so the guard is load-bearing: the
// last block runs with a fraction of its threads live.
__global__ void SaxpyScalar(const float* a, const float* b, float* out, float s,
                            int n) {
  /// BEGIN CODE HERE (approx 3 lines) ///
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = s * a[i] + b[i];
  /// END CODE HERE ///
}

// The same arithmetic, four elements per thread, moved as float4. One thread
// per float4 element, so the launch is n / 4 threads wide and each of them
// issues one 128-bit load per operand instead of four 32-bit ones.
//
// Two things have to be true before `reinterpret_cast<const float4*>(a)` is
// anything other than undefined behaviour:
//
//   1. The base is 16-byte aligned. cudaMalloc guarantees at least 256, so all
//      three of these pointers qualify -- but only the pointers cudaMalloc
//      handed back. `a + 1` does not, and the runner demonstrates what the
//      hardware does about it.
//   2. Indexing the float4 view by `i` reaches floats 4i .. 4i+3, so the whole
//      array has to be a whole number of float4s. `n` is NOT a multiple of 4
//      here, and the last n % 4 elements are outside the float4 view entirely.
//      They still have to be computed, with scalar code, by somebody.
//
// That tail is the guard this kernel is really about. Skip it and the last
// elements keep the runner's 0xff poison and come back NaN.
__global__ void SaxpyVec4(const float* a, const float* b, float* out, float s,
                          int n) {
  /// BEGIN CODE HERE (approx 12 lines) ///
  const int nv = n / 4;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nv) {
    const float4 av = reinterpret_cast<const float4*>(a)[i];
    const float4 bv = reinterpret_cast<const float4*>(b)[i];
    const float4 ov = make_float4(s * av.x + bv.x, s * av.y + bv.y,
                                  s * av.z + bv.z, s * av.w + bv.w);
    reinterpret_cast<float4*>(out)[i] = ov;
  }
  const int t = nv * 4 + i;
  if (t < n) out[t] = s * a[t] + b[t];
  /// END CODE HERE ///
}

// out[i] = s * a[i + 1] + b[i]: the same scalar code with the `a` operand read
// one element to the right. `a` holds n + 1 floats so nothing runs off the end,
// the arithmetic is the same two flops, the instruction count is identical to
// SaxpyScalar's, and the access is still perfectly coalesced -- consecutive
// lanes still read consecutive floats.
//
// What changes is where the warp's 128 bytes begin. Lane 0 of warp w wants
// a[32w + 1], four bytes past the 128-byte boundary its aligned twin started
// on, so the warp's 128 contiguous bytes now straddle five 32-byte sectors
// instead of four. Nothing else about this kernel differs from SaxpyScalar.
__global__ void SaxpyMisaligned(const float* a, const float* b, float* out,
                                float s, int n) {
  /// BEGIN CODE HERE (approx 3 lines) ///
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = s * a[i + 1] + b[i];
  /// END CODE HERE ///
}
