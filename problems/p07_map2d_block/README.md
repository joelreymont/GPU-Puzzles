# Puzzle 7 — Map 2D, blocked

New idea: **a 2D grid of 2D blocks.**

Puzzle 4 covered a matrix with one block, which caps you at 1024 elements.
Puzzle 6 covered a long vector with many blocks. This is both at once: a
500×700 matrix — 350000 elements — tiled by `dim3(32, 8)` blocks, so the grid
is `dim3(22, 63)` and there are 1386 blocks in flight.

The global coordinate of a thread is puzzle 6's line, once per axis:

```
col = blockIdx.x * blockDim.x + threadIdx.x
row = blockIdx.y * blockDim.y + threadIdx.y
```

Everything after that is puzzle 4: flatten row-major with `row * cols + col`,
guard both dimensions. Nothing new happens in the kernel body. The new thing is
entirely in the launch geometry, and it is worth looking at the numbers:

- `22 * 32 = 704` columns of threads over 700 columns of matrix — 4 columns of
  overhang.
- `63 * 8 = 504` rows of threads over 500 rows of matrix — 4 rows of overhang.

Both extents were chosen so neither divides evenly. The blocks along the right
edge and along the bottom edge are partial, the corner block is partial in both
directions, and the two guards are what keeps 6032 threads out of memory that
is not theirs.

Why `32 x 8` and not `16 x 16`, given both are 256 threads? Because
`threadIdx.x` varies fastest, a 32-wide block means each warp is exactly one
row of 32 consecutive columns — one fully coalesced 128-byte load. A 16-wide
block splits each warp across two matrix rows, so every warp issues two
transactions instead of one. Same thread count, same result, more memory
traffic. Block *shape* is a performance decision even when block *size* is not.

## Task

Fill in `skeletons/p07_map2d_block/kernel.cu`:

```
out[r][c] = a[r][c] + 10   for every r in [0, rows), c in [0, cols)
```

- Input `a` and output `out`: `rows * cols` floats, row-major, `rows = 500`,
  `cols = 700`.
- Launch: `dim3(22, 63)` blocks of `dim3(32, 8)` threads.
- `row` and `col` are already computed for you.

**approx 4 lines.**

## Verification

Tolerance zero, `NaN` poison, seeded input.

The guards are checked by `make check` rather than by the values, for the
reason puzzle 3 spells out: the overhanging threads write past the end of the
allocation, which corrupts nothing the comparison looks at. Measured against
this runner, dropping the row guard still prints `PASS map2d_block` — and
compute-sanitizer reports 1301 errors on the same binary. Run both.

## Signature correction

Upstream is `Map2DBlock(float* A, float* C, float size)`; here,
`int rows, int cols`. See puzzles 3 and 4.

## Run

```
make run P=07
make run P=07 MODE=solution
make check P=07
```

Expected: one `PASS map2d_block` line.
