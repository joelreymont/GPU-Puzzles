# Puzzle 16 — Matrix multiply

New idea: **tiling** — the two-level loop that turns a memory-bound kernel into
a compute-bound one, and the reason shared memory exists at all.

```
out[r][c] = sum over k of a[r][k] * b[k][c]
```

for square row-major 500×500 matrices. This is the last of the beginner
sequence and it uses everything before it: 2D blocks and a 2D grid (puzzles 4
and 7), staging into shared memory (8), halo-free tiles with edge guards (11),
and a barrier discipline that is now genuinely two-sided (14).

## Why the naive version is slow

Write it without shared memory and each thread walks a whole row of `a` and a
whole column of `b`:

```
for (int k = 0; k < n; k++) sum += a[row*n + k] * b[k*n + col];
```

Correct, and it does `n` global loads per multiply-add. But `a[row][k]` is read
by *every* thread in the same row of the output — 500 of them — and each one
fetches it from global memory separately. The arithmetic is `2n³` flops against
`2n³` loads: one load per flop, from the slowest memory in the machine. The
GPU spends the entire kernel waiting.

Measured on this box with `ncu`, over the full 500×500 problem:

```
sudo /usr/local/cuda/bin/ncu \
  --metrics smsp__inst_executed_op_global_ld.sum,smsp__inst_executed_op_shared_ld.sum \
  ./build/solution/p16_matmul
```

| kernel | global load instructions | shared load instructions |
|---|---|---|
| naive, no shared memory | 8,000,000 | 0 |
| this puzzle's tiled kernel | 524,288 | 5,242,880 |

**15.3× fewer global loads.** The traffic did not vanish, it moved: the same
values are now read out of shared memory ten times as often as they are read
out of global memory, which is precisely the trade shared memory exists to
make.

## The tile

Walk the `k` axis in chunks of `TPB = 16` instead of one element at a time. At
each chunk, the block cooperatively stages a 16×16 tile of `a` and a 16×16 tile
of `b` into shared memory — one element each per thread, since the block is
16×16 threads — and then every thread computes the part of its dot product that
lives in those tiles, reading each staged value 16 times.

So there are two nested loops where the naive kernel had one:

```
for (k0 = 0; k0 < n; k0 += TPB) {     // outer: which tile pair
   ...stage, barrier...
   for (k = 0; k < TPB; k++)          // inner: within the tiles
      sum += a_shared[...] * b_shared[...];
   ...barrier...
}
```

`sum` lives in a register across the whole outer loop; nothing is written back
to global memory until the end. Each element of `a` is now loaded from global
memory once per *block* that needs it — `n / TPB` tile steps instead of `n`
multiplies — which is where the 15.3× comes from.

Which element each thread stages is the part worth thinking about slowly. A
thread stages one element of `a`'s tile and one of `b`'s, and they are not the
same `(k)`:

- from `a` it takes the element in **its own output row**, at column
  `k0 + threadIdx.x` — the tile is a horizontal strip of `a`.
- from `b` it takes the element in **its own output column**, at row
  `k0 + threadIdx.y` — the tile is a vertical strip of `b`.

Both staged tiles are indexed `[threadIdx.y * TPB + threadIdx.x]` in shared
memory, so a warp writes 32 consecutive floats. Its global reads are two runs
of 16 consecutive floats rather than one of 32, because a 16-wide block splits
each warp across two matrix rows — the cost puzzle 7 describes, accepted here
because the tile has to be square for the inner loop to work. Write the store
index the other way round and the kernel still computes *something*, and it
will not be `a * b`.

## Two barriers per tile step

One after the staging, for the reason every shared-memory kernel needs one:
the values a thread reads in the inner loop were written by 255 other threads.

And one **after** the inner loop, before the next iteration overwrites the
tiles. That second barrier is the write-after-read hazard from puzzle 14, in a
different costume: without it, a fast warp that has finished the inner loop
races ahead and stages tile `k0 + 16` on top of values a slow warp is still
reading from tile `k0`. It is the barrier people leave out — including the
upstream answer to this puzzle — because the loop "obviously" ends before the
next one begins, which is true of the source text and not of 8 warps.

Both barriers go at the top level of the outer loop, where all 256 threads
reach them. Edge blocks make this sharp: 500 is not a multiple of 16, so in the
bottom and right-hand blocks some threads have no output at all, and they must
still stage (zeros), still hit both barriers, and still run the inner loop.
Anything of the form `if (row < n && col < n) { ...whole body... }` is undefined
behaviour here, and there is no `return` that is safe before the last barrier.

## Guards, by zero-filling

500 = 31·16 + 4, so the last tile of every row and column is 4 wide and 12
columns of the staged tile refer to elements that do not exist. Stage `0.0f`
into those slots rather than skipping them: a zero contributes nothing to the
product, so the inner loop needs no bounds test and runs the same 16 iterations
every time. Same idea as the truncated windows in puzzle 11 — the identity
element *is* the boundary condition.

Three guards in total, and each covers a different edge:

- staging `a`: `row < n` (the block hangs off the bottom) and `k0 + lcol < n`
  (the tile hangs off the right of `a`);
- staging `b`: `k0 + lrow < n` and `col < n`;
- the final store: `row < n && col < n`.

## Task

Fill in `skeletons/p16_matmul/kernel.cu`:

```
out = a * b,   square row-major, n = 500, TPB = 16
```

- Inputs `a`, `b` and output `out`: `n * n` floats each, row-major.
- Launch: `dim3(32, 32)` blocks of `dim3(16, 16)` threads, 2048 bytes of
  dynamic shared memory each.
- `row`, `col`, `lrow` and `lcol` are computed for you; `TPB` is at the top of
  the file.
- Carve `a_shared` and `b_shared` out of the one dynamic allocation (puzzle 13),
  both `TPB * TPB` floats indexed `[lrow * TPB + lcol]`.

**approx 20 lines.**

## Verification

Seeded inputs, `NaN` poison, **relative tolerance `1e-3`** — the loosest in
this repository, and the reason is worth stating rather than hiding.

Each output is 500 products summed into one `float` register, a 500-long serial
rounding chain, compared against a reference that accumulates the same 500
products in `double`. Measured worst deviation over all 250000 outputs:

| association | worst relative deviation | tolerance | headroom |
|---|---|---|---|
| one running `sum` (the solution) | `1.267e-05` (index 229810) | `1e-3` | 79× |
| per-tile partial, then accumulate | `3.636e-06` (index 31015) | `1e-3` | 275× |

The rule this repository uses — the tightest power of ten clearing the measured
deviation by at least 10× — puts the shipped kernel at `1e-3`, because `1e-4`
would leave only 7.9×.

The worst case is also worth looking at rather than being alarmed by: at index
229810 the expected value is `-0.2938673` and the kernel produces `-0.29388`.
That is an output of magnitude 0.29 assembled from 500 products of magnitude up
to 1 — near-total cancellation — and since `compare()` divides by
`max(|want|, 1)`, the figure quoted is really an absolute error of `1.27e-05`
against a floor of 1. For a 500-term float sum of terms that size, that is
ordinary.

A loose tolerance is only dangerous if the failures it must catch are subtle,
and measured here they are not. The barrier-deletion experiment below is the
mildest defect of the obvious ones — it corrupts a few terms of a 500-term sum
rather than the whole answer — and across five runs its first failure was off
by 2.0%, 2.5%, 6.5%, 18% and 19% of the expected value. The smallest of those
is 20× the tolerance; the rest are hundreds of times it. Structural mistakes in
this kernel are wrong by a factor, not by a rounding.

Two properties of the runner make the indexing checkable: `a` and `b` are
seeded differently, so `a * b` is not symmetric and computing the transpose is
a *different answer* rather than a relabelling of the same one; and the output
is poisoned with `NaN`, so the 12 rows and columns of overhang cannot be
quietly skipped.

Build output for the solution on this box: `Used 36 registers, used 1 barriers`.

## Race-freedom

`make check P=16 MODE=solution` is clean under memcheck, racecheck and
synccheck.

Delete the barrier at the **end** of the outer loop — the one the upstream
answer does not have — and, measured on this box, the plain run fails **20
times out of 20**, at a different index each time:

```
FAIL matmul at index 32288: got 5.40682 want 5.29859
FAIL matmul at index 27272: got 7.57627 want 7.77006
FAIL matmul at index 32384: got -11.7403 want -12.5623
```

Under `racecheck` the same binary fails at index 0 instead — `got 1.00296 want
2.99742`, the instrumentation shifting the schedule far enough to corrupt the
very first output — and reports a write racing 16 separate reads:

```
Error: Race reported between Write access at Matmul(...)+0x2f0
    and Read access at Matmul(...)+0x320 [288052 hazards]
    and Read access at Matmul(...)+0x340 [254272 hazards]
    ... 16 read sites in all, 254k-520k hazards each
RACECHECK SUMMARY: 3 hazards displayed (1 error, 2 warnings)
```

The 16 read sites are the inner loop, fully unrolled by the compiler: one
staging write racing every iteration of the product loop. (The summary count
moves between 2 and 3 across runs — it is a count of what the tool observed,
not of what the code permits.)

## Signature correction

Upstream (dshah3's puzzle 14, `matmul`) is `Matmul(float* A, float* B, float*
C, int size)`; here, `const float*` inputs. The size stays a single `int`
because the matrices are square, as upstream.

Three differences beyond the harness:

- **The missing barrier.** The upstream answer has `__syncthreads()` after
  staging and none after the inner loop, which is the race measured above. Its
  own harness runs a 2×2 matrix in a single 3×3 block, where the outer loop
  executes once and the missing barrier can never be reached a second time.
- **`TPB = 3` and `size = 2`.** One block, one tile step, no edge case that is
  not also the only case. Here: 1024 blocks, 32 tile steps, and a 4-wide
  remainder tile.
- **The inner-loop guard.** Upstream tests `k + local_k < size` inside the
  product loop instead of zero-filling the tile, which is correct but puts a
  branch in the innermost loop of the kernel.

## Run

```
make run P=16
make run P=16 MODE=solution
make check P=16
```

Expected: one `PASS matmul` line, and `make check` clean under all three
sanitizers.

That is the end of the beginner sequence. Puzzle 24 picks up at warp-level
programming, and the first thing it does is replace puzzle 12's shared-memory
tree with three instructions.
