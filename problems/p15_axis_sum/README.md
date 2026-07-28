# Puzzle 15 — Axis sum

New idea: **a grid dimension that indexes work instead of data.**

The kernel is puzzle 12's reduction, unchanged. What is new is the launch: a
batch of independent reductions, one per row of a matrix, dispatched by giving
the grid a second dimension.

```
out[r] = a[r][0] + a[r][1] + ... + a[r][cols-1]
```

64 rows of 1000, so 64 totals.

## One block per row

Puzzle 7 used a 2D grid to tile a 2D array — both axes indexed elements. Here
only one does:

- `blockIdx.y` is **which row**. The whole block works on that row and no
  other. This axis indexes *tasks*, not elements.
- `blockIdx.x` tiles the row itself. With 1000 columns and 1024 threads it is
  one block wide, so `blockIdx.x` is always 0 — but the index expression is
  written as though it were not, because that is what makes the kernel say what
  it means.

The grid is `dim3(1, 64)`. Nothing coordinates the 64 blocks and nothing needs
to: each owns one row of the input and one element of the output, and blocks
being independent is exactly the property this launch is built on.

Because the launch puts exactly one block on each row, the row count is not a
kernel argument at all — `gridDim.y` is the row count and `blockIdx.y` is the
row. A parameter the kernel cannot use is a parameter that will eventually
disagree with the launch.

The row is contiguous in memory (row-major, puzzle 4), so the address is
`a[row * cols + col]` and a warp reads 32 consecutive floats: coalesced, the
same as every 1D kernel so far. The second grid dimension changes which data
the block gets, not how it reads it.

## What carries over from puzzle 12

All of it. Stage one element per thread into shared memory with the identity
for the 24 lanes past the end of the row, barrier, halve the stride until one
value is left, and have `local_i == 0` write the block's answer — to `out[row]`
this time rather than `out[0]`.

If you wrote puzzle 12, this puzzle is three characters of difference and one
new line in the launch. That is the point: block-level collectives compose with
grid geometry without changing.

## Task

Fill in `skeletons/p15_axis_sum/kernel.cu`:

```
out[r] = sum over c in [0, cols) of a[r][c],   rows = 64, cols = 1000
```

- Input: `a`, `rows * cols` floats, row-major. Output: `out`, `rows` floats.
- Launch: `dim3(1, 64)` blocks of 1024 threads, 4096 bytes of dynamic shared
  memory each.
- `col`, `local_i` and `row` are computed for you.

**approx 13 lines.**

## Verification

Seeded input, `NaN` poison on all 64 outputs, **relative tolerance `1e-4`**.

64 rows is 64 independent draws of the same 1000-element reduction puzzle 12
performs once, which makes this runner the better measurement of the two.
Measured worst deviation over the 64 rows against the reference's sequential
`double`:

| association | worst relative deviation |
|---|---|
| tree (the solution) | `2.623e-06` (row 62) |
| one thread summing sequentially | `9.775e-06` (row 56) |

`1e-4` is the tightest decade clearing both by more than 10×, and it agrees
with the 512-seed sweep quoted in puzzle 12 — `2.742e-06` and `9.030e-06` — as
it should, since it is the same reduction over the same distribution.

The `NaN` poison is doing real work here. A kernel that writes `out[0]` instead
of `out[row]`, or that lets every thread write rather than only `local_i == 0`,
leaves rows untouched or fills them with partial sums, and both fail loudly
instead of averaging out.

Build output for the solution on this box: `Used 12 registers, used 1 barriers`.

## Race-freedom, and the thing puzzle 12 could not show you

`make check P=15 MODE=solution` is clean under memcheck, racecheck and
synccheck.

Delete the `__syncthreads()` inside the reduction loop — the same edit that
puzzle 12 survived 30 runs out of 30 — and here, measured on this box, the
plain run fails **20 times out of 20**:

```
FAIL axis_sum at index 0: got -3.65694 want 7.58294
```

with racecheck reporting

```
Race reported between Read access at AxisSum(...)+0x1b0
    and Write access at AxisSum(...)+0x1d0 [129152 hazards]
RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)
```

Same kernel body, same missing barrier, opposite outcome on the plain run. The
difference is the launch: puzzle 12 puts one block of 32 warps on an idle GPU,
where they advance close enough to lockstep that the bad interleaving never
comes up. This one puts 64 blocks on 48 SMs, so some SMs host two blocks, the
warp schedulers have something else to run, and the drift the race needs
happens immediately.

Neither run is more correct than the other about the kernel — it is racy in
both. The plain test was simply not able to observe it in one case. That is the
argument for `make check` in a sentence: correctness tests sample the schedules
the machine happened to pick, and racecheck reasons about the ones it was
allowed to pick.

## Signature correction

Upstream (dshah3's puzzle 13, `axis_sum`) is `AxisSum(float* A, float* C, int
size)`; here, `const float* a` and `int cols`.

Its harness runs `numBatches = 1` — a batch dimension of one, which is the one
size at which a batch dimension cannot be wrong. Its answer also wraps the
whole body, barriers included, in `if (i < size)`, which is undefined behaviour
the moment a row is not a multiple of the block size (puzzle 8), and computes
its doubling offset with `pow(2, p)` (puzzle 14).

## Run

```
make run P=15
make run P=15 MODE=solution
make check P=15
```

Expected: one `PASS axis_sum` line, and `make check` clean under all three
sanitizers.
