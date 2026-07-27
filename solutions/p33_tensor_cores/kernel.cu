#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

// Everything the tensor-core API exposes lives in nvcuda::wmma. `mma.h` is the
// only header involved -- there is no library to link and no launch API to
// change. The instruction is just an instruction.
using namespace nvcuda;

// Two kernels, one product:
//
//   A: M x K row-major     B: K x N row-major     C: M x N row-major
//
// and they are fed the same numbers, not merely similar ones. The runner
// rounds its random inputs to __half and then converts them straight back to
// float, so every float element GemmNaiveFp32 reads is the exact value of the
// __half element GemmWmma reads at the same index. Nothing about the inputs
// differs. What differs is which pipe the multiply-accumulate issues to, and
// -- because one mma_sync computes a whole 16x16x16 product -- how many
// instructions there are: 141,760 warp instructions against 2,432,000, both
// measured with ncu and both in the README's table.

// The fragment shape, and the reason the runner's M, N and K are all multiples
// of 16. These are not tuning knobs: 16x16x16 is a shape the silicon
// implements, wmma::fragment is a template parameterised on it, and the set of
// legal triples is fixed by the hardware. A warp owns one MMA_M x MMA_N tile
// of C and walks the shared dimension MMA_K elements at a time.
constexpr int MMA_M = 16;
constexpr int MMA_N = 16;
constexpr int MMA_K = 16;

constexpr int WARP = 32;

// 128 threads per block = 4 warps, laid out 2 x 2 over the tile grid, so one
// block covers a 32 x 32 patch of C. The warp is the unit here, not the
// thread: a wmma fragment is warp-collective state, so "which warp am I" is
// the only decomposition question this kernel has.
constexpr int WARPS_X = 2;
constexpr int WARPS_Y = 2;

// One thread per element of C, one plain FP32 multiply-add per K step. This is
// the correctness anchor -- it is checked against the same CPU reference
// GemmWmma is -- and it is the timing baseline.
//
// It is deliberately not a straw man. The launch is dim3(16, 16) threads over
// dim3(cdiv(N,16), cdiv(M,16)) blocks, so threadIdx.x indexes the column of C:
// consecutive lanes read consecutive elements of a row of B, which is a
// coalesced 128-byte transaction, and all 32 lanes of a warp read at most two
// elements of A per step, which the L1 broadcasts. The gap you measure is not
// this kernel being careless with memory.
__global__ void GemmNaiveFp32(const float* a, const float* b, float* c, int M,
                              int N, int K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN CODE HERE (approx 5 lines) ///
  if (row >= M || col >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < K; k++) acc += a[row * K + k] * b[k * N + col];
  c[row * N + col] = acc;
  /// END CODE HERE ///
}

// The same product on the tensor cores. One warp computes one 16 x 16 tile of
// C, which is 256 outputs from 32 threads -- there is no thread-to-element map
// here at all, and you should not go looking for one.
//
// You need three fragments and four calls:
//
//   wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, __half, LAYOUT>
//   wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, __half, LAYOUT>
//   wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float>
//
//   wmma::fill_fragment(frag, value)
//   wmma::load_matrix_sync(frag, ptr, ldm)
//   wmma::mma_sync(d, a, b, c)
//   wmma::store_matrix_sync(ptr, frag, ldm, layout)
//
// Three things decide whether this compiles and then whether it is right:
//
//   1. The layout of a and b fragments is a TYPE parameter -- row_major or
//      col_major, chosen when you declare the fragment. The accumulator has no
//      layout in its type; store_matrix_sync takes it as an argument instead
//      (wmma::mem_row_major / wmma::mem_col_major), because the accumulator is
//      only laid out at the moment it lands in memory.
//   2. `ldm` is the leading dimension of the matrix the fragment is a window
//      into -- the distance in elements from one row to the next in the FULL
//      array, not 16.
//   3. Every one of those calls is warp-collective. All 32 lanes execute it
//      together, the fragment lives spread across the warp's registers in a
//      layout you are not told, and a lane that skips a call or reaches it
//      under a different branch is undefined behaviour.
__global__ void GemmWmma(const __half* a, const __half* b, float* c, int M,
                         int N, int K) {
  // This warp's tile of C. Both coordinates are built from blockIdx and the
  // warp id and never from the lane, so they are uniform across the warp --
  // which is what makes the guard on the next line legal. All 32 lanes
  // evaluate it identically and either all return or none do. The same
  // `return` written against a per-lane condition, anywhere in this kernel,
  // would not be a slow path; it would be undefined behaviour.
  //
  // At the runner's M and N the guard never fires -- 320/16 = 20 and
  // 256/16 = 16 are both even, so the 2 x 2 warp grid tiles them exactly. It
  // is here because M and N are runtime arguments and the next caller's may
  // not be. What it can never rescue is a dimension that is not a multiple of
  // 16: a fragment is 16 wide whether or not the matrix is, and the fix for a
  // ragged edge is to pad the problem, not to guard the warp.
  const int warp = threadIdx.x / WARP;
  const int tile_m = (blockIdx.y * WARPS_Y + warp / WARPS_X) * MMA_M;
  const int tile_n = (blockIdx.x * WARPS_X + warp % WARPS_X) * MMA_N;
  if (tile_m >= M || tile_n >= N) return;

  /// BEGIN CODE HERE (approx 10 lines) ///
  wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, __half, wmma::row_major>
      fa;
  wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, __half, wmma::row_major>
      fb;
  wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> acc;
  wmma::fill_fragment(acc, 0.0f);
  for (int k = 0; k < K; k += MMA_K) {
    wmma::load_matrix_sync(fa, a + tile_m * K + k, K);
    wmma::load_matrix_sync(fb, b + k * N + tile_n, N);
    wmma::mma_sync(acc, fa, fb, acc);
  }
  wmma::store_matrix_sync(c + tile_m * N + tile_n, acc, N, wmma::mem_row_major);
  /// END CODE HERE ///
}
