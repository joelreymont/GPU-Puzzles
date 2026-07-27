#include <cuda_runtime.h>

// Three kernels, one answer: `out` is the transpose of `in`. There is no
// arithmetic anywhere -- every output is a bit-exact copy of exactly one input
// -- so anything that separates these three is pure memory behaviour, and all
// of it happens inside shared memory.
//
// The global side is deliberately identical and deliberately optimal in all
// three. A 32x32 tile is read from `in` row-major and written to `out`
// row-major, so both global accesses are perfectly coalesced. The transpose
// happens on the shared tile in between, which is the only place left for the
// three kernels to differ.
constexpr int TILE = 32;

// Threads per block: TILE x BLOCK_ROWS = 32 x 8 = 256, so each thread carries
// TILE / BLOCK_ROWS = 4 rows of the tile.
//
// blockDim.x == 32 == warpSize is load-bearing for everything below: a warp is
// exactly one row of the block, threadIdx.y is the warp id, and every
// bank-mapping question in this puzzle is therefore a question about what the
// 32 values of threadIdx.x do inside one warp while threadIdx.y is fixed.
constexpr int BLOCK_ROWS = 8;

// The plain tile: 32 rows of 32 floats, row-major, so logical element [r][c]
// sits at float offset r * 32 + c -- and the row length is exactly the number
// of banks. Work out what that does to a column before you write anything.
//
// Both loops are yours: a coalesced load of the tile from `in`, then a
// coalesced store to `out` that reads the tile transposed. Between them goes
// the barrier, and it is not optional -- a warp reads tile rows that other
// warps wrote, so without it this kernel is a data race that happens to pass
// most of the time. `make check P=32` will say so.
__global__ void TransposeNaive(const float* in, float* out, int rows, int cols) {
  __shared__ float tile[TILE][TILE];

  // Where this block's tile sits in `in`: column x, rows y .. y + TILE - 1.
  const int x = blockIdx.x * TILE + threadIdx.x;
  const int y = blockIdx.y * TILE + threadIdx.y;

  // Where the transposed tile sits in `out`, which is cols x rows. Row ty of
  // `out` is column ty of `in`, and column tx of `out` is row tx of `in` -- so
  // the block indices swap, and threadIdx.x now walks the direction that used
  // to be rows. That swap is what keeps the global store coalesced, and it is
  // also what forces the shared read to run down a tile column.
  const int tx = blockIdx.y * TILE + threadIdx.x;
  const int ty = blockIdx.x * TILE + threadIdx.y;

  /// BEGIN CODE HERE (approx 8 lines) ///
  /// END CODE HERE ///
}

// Identical to TransposeNaive in every respect except the shared declaration:
// the row is one float longer than it needs to be. Nothing else changes -- the
// indices you write are the same indices you wrote above, the pad column is
// never read and never written, and the block spends 128 bytes of shared
// memory on nothing.
//
// Work out where logical [r][c] sits now, and what that does to the bank a
// whole column lands in.
__global__ void TransposePadded(const float* in, float* out, int rows,
                                int cols) {
  __shared__ float tile[TILE][TILE + 1];

  const int x = blockIdx.x * TILE + threadIdx.x;
  const int y = blockIdx.y * TILE + threadIdx.y;
  const int tx = blockIdx.y * TILE + threadIdx.x;
  const int ty = blockIdx.x * TILE + threadIdx.y;

  /// BEGIN CODE HERE (approx 8 lines) ///
  /// END CODE HERE ///
}

// The square tile again -- not one byte of padding -- but each row is permuted
// before it is stored. Logical element [r][c] of the tile lives at physical
// [r][c ^ r]: row 0 unchanged, row 1 with neighbours swapped in pairs, row 2
// swapped in pairs of two, and so on up to row 31.
//
// Two properties are worth convincing yourself of before writing anything,
// because the kernel is correct only if both hold:
//
//   1. For a fixed r, `c -> c ^ r` is a permutation of 0..31. Nothing is lost
//      and nothing collides, so the tile still holds every value exactly once.
//   2. XOR is its own inverse, so a thread hunting for logical [a][b] applies
//      the same transform the writer applied.
//
// Both loops are yours again, and they have to agree on the physical address
// of a logical element. That agreement is the whole trick.
__global__ void TransposeSwizzle(const float* in, float* out, int rows,
                                 int cols) {
  __shared__ float tile[TILE][TILE];

  const int x = blockIdx.x * TILE + threadIdx.x;
  const int y = blockIdx.y * TILE + threadIdx.y;
  const int tx = blockIdx.y * TILE + threadIdx.x;
  const int ty = blockIdx.x * TILE + threadIdx.y;

  /// BEGIN CODE HERE (approx 8 lines) ///
  /// END CODE HERE ///
}
