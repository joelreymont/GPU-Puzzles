# Puzzle 4 — Map 2D

New idea: **blocks are three-dimensional, and matrices are one-dimensional.**

Both halves matter.

A block is not a line of threads, it is a box: `threadIdx` has `.x`, `.y` and
`.z`, and the launch supplies the extents as a `dim3`. This puzzle launches
`dim3(32, 32)` — 1024 threads arranged 32 across by 32 down — so each thread
gets a distinct `(threadIdx.x, threadIdx.y)` pair, which is exactly the shape
of a 2D problem.

Memory, meanwhile, is flat. There is no `float**` here and no
`out[row][col]`. A `rows x cols` matrix is `rows * cols` consecutive floats in
**row-major** order: row 0's elements first, then row 1's, and so on. So
element `(r, c)` lives at

```
index = r * cols + c
```

`cols`, not `rows` — the row stride is the length of a row.

Which thread index maps to which matrix axis is a convention, and this
repository uses the one that matters later: **`threadIdx.x` is the column**,
`threadIdx.y` is the row. Threads are numbered with `.x` fastest, so
consecutive `.x` threads land in the same warp, and making them read
consecutive columns of a row makes their loads consecutive in memory. That is
what the hardware coalesces into one transaction. Bind `.x` to the row instead
and every thread in a warp reads an address `cols` floats away from its
neighbour's — correct, but many times the memory traffic. The skeleton names
the variables `row` and `col` so there is nothing to keep straight.

Second thing to notice: the block is 32×32 but the matrix is 25×31. **Both**
dimensions overhang, so puzzle 3's guard is now two guards.

## Task

Fill in `skeletons/p04_map2d/kernel.cu`:

```
out[r][c] = a[r][c] + 10   for every r in [0, rows), c in [0, cols)
```

- Input `a` and output `out`: `rows * cols` floats, row-major, `rows = 25`,
  `cols = 31`.
- Launch: one block, `dim3(32, 32)`.
- `row` and `col` are already computed for you. Guard both, then flatten.

**approx 4 lines.**

## Verification

Tolerance zero, `NaN` poison on the output, seeded input — as in puzzles 1–3.

Be aware of one thing the value check cannot see here, because it is honest
about it: this operation is elementwise, so a kernel that consistently used
column-major indexing for *both* the load and the store would produce a
bit-identical output buffer and pass. The row-major flattening is part of the
stated contract rather than something the numbers can prove. Puzzle 5 is where
getting the two axes the wrong way round starts failing loudly, because there
the output at `(r, c)` depends on `r` and `c` differently.

What the harness *does* catch is a missing guard, via `make check`: drop either
of the two bounds tests and compute-sanitizer reports out-of-bounds accesses
from the overhanging threads (measured: 126 errors with both guards removed,
2 with only the column guard removed).

## Signature correction

Upstream is `Map2D(float* A, float* C, float size)` — a `float` element count
again, and a single `size` that forces the matrix to be square. Here it is
`int rows, int cols`. Square is the shape in which every 2D indexing mistake
looks harmless: the row stride and the column count are the same number, so
`r * cols + c` and `c * rows + r` are interchangeable and the two bounds tests
are the same test. Rectangular is the shape that makes the stride real.

## Run

```
make run P=04
make run P=04 MODE=solution
make check P=04
```

Expected: one `PASS map2d` line.
