# Puzzle 13 — 1D convolution

New idea: **halo loading done properly**, and **two arrays carved out of one
dynamic shared allocation.**

Puzzle 11's window was three elements wide and reached two elements to the
left. This is the same structure with the window a parameter, reaching the
other way:

```
out[i] = a[i]*b[0] + a[i+1]*b[1] + ... + a[i+k-1]*b[k-1]
```

with `a[i+j]` read as `0` once `i+j` runs past the end of the signal, so the
last `k-1` outputs are windows that hang off the end. (Signal-processing
pedantry: this is a correlation, not a flipped convolution. It is what upstream
computes and what the reference computes.)

## The two staged arrays

A block of `TPB = 256` threads produces 256 outputs, and those outputs read
`256 + k - 1` distinct signal samples: the block's own tile, plus a **halo** of
`k - 1` samples that belong to the block on its right. Every sample in the
interior is read by `k` different outputs, so staging pays for itself `k` times
over.

The filter `b` is staged too, for a different reason. It is only `k` floats,
but *every* thread reads *all* of it, so leaving it in global memory means the
same `k` values are fetched 256 times per block. In shared memory they are
fetched once. Small arrays that everyone reads are as good a reason to use
shared memory as big arrays that overlap.

Both live in the one dynamic allocation the launch provides — there is only one
`extern __shared__` array per kernel, so multiple logical arrays are carved out
of it by pointer arithmetic. The launch requests
`TPB_MAX_CONV + MAX_CONV` floats, laid out as:

```
smem ─┬─ shared_a ── TPB floats (tile) │ k-1 (halo) │ slack ─┐  TPB_MAX_CONV
      └─ shared_b ── k floats (filter) │ slack ─────────────┘  MAX_CONV
```

Doing the carve — two `float*` into that one block, the second offset by the
full length of the first — is part of the task. Offset it by anything smaller
and the two arrays overlap: the filter lands on top of the tail of the signal
tile, which is exactly the sort of bug that produces plausible-looking wrong
numbers rather than a crash.

`MAX_CONV` is a compile-time **cap** on the filter length, not its value. The
runtime `k` is 15. Sizing shared memory from a cap rather than from the actual
parameter is what real convolution kernels do, and it means the last
`MAX_CONV - (k - 1)` floats of `shared_a` are allocated and never touched.

## Who loads the halo

The obvious staging line covers `shared_a[0 .. TPB-1]`. Somebody still has to
fill `shared_a[TPB .. TPB+k-2]`, and the only threads available are the same
256. So a subset of them — the first `k - 1` is the simplest choice — do a
second load, from `TPB` elements further along the signal than their own.

That second load needs its own guard against the end of the array, and its
result must be `0` when it falls off, for the same reason as puzzle 11: the
zero *is* the definition of what happens past the end, so it makes the tail
outputs come out right with no special case in the inner loop.

Then one barrier, and the product loop reads `shared_a[local_i + j]` for
`j` in `[0, k)` — the widest read is `shared_a[TPB - 1 + k - 1]`, precisely the
last halo slot.

## Task

Fill in `skeletons/p13_1dconv/kernel.cu`:

```
out[i] = sum over j in [0, k) of a[i+j] * b[j],   n = 100003, k = 15
```

- Inputs: `a` (`n` floats), `b` (`k` floats). Output: `out`, `n` floats.
- Launch: 391 blocks of 256 threads; dynamic shared memory for
  `TPB + MAX_CONV` plus `MAX_CONV` floats.
- `i` and `local_i` are computed for you; `TPB`, `MAX_CONV` and `TPB_MAX_CONV`
  are given at the top of the file.
- Carve the two arrays, stage the tile, stage the halo, stage the filter,
  barrier, then accumulate in a `float`.
- One barrier, and every thread must reach it — so it goes outside the staging
  conditionals, not inside them.

**approx 19 lines.**

## Verification

Seeded input, `NaN` poison, **relative tolerance `1e-5`**.

Each output is 15 products accumulated in `float` against the reference's
`double`. Measured worst deviation over all 100003 outputs:

| association | worst relative deviation |
|---|---|
| filter walked forwards, `j = 0 .. k-1` | `4.172e-07` (index 4434) |
| filter walked backwards | `5.364e-07` (index 64281) |

`1e-5` is the tightest decade clearing both by more than 10×.

Sizes are chosen so that nothing is a special case that happens to work:
`n = 100003` is prime, so the last block is partial and its last warp is
partial; `k = 15` is odd, does not divide 256, and is under `MAX_CONV`, so the
staging guard on `k` and the unused tail of the tile are both real.

Build output for the solution on this box: `Used 28 registers, used 1 barriers`.

## Race-freedom

`make check P=13 MODE=solution` is clean under memcheck, racecheck and
synccheck. Delete the `__syncthreads()` and, measured on this box, the plain
run fails **20 times out of 20** (`FAIL conv1d at index 32: got 0.608565 want
-0.376893`), with racecheck reporting

```
Race reported between Write access at Conv1D(...)+0x150
    and Read access at Conv1D(...)+0x390 [393016 hazards]
    ... 29 read sites in total, across three write sites
RACECHECK SUMMARY: 3 hazards displayed (3 errors, 0 warnings)
```

Three separate write→read races, one per staging write — tile, halo, filter —
each of them read by threads that did not perform it. The 29 read sites are the
unrolled product loop: every `j` is its own instruction, and every one of them
reads shared memory somebody else wrote.

## Signature correction

Upstream (dshah3's puzzle 11, `1dconv`) is `Conv1D(float* A, float* B, float*
C, int a_size, int b_size)`; here, `const float*` inputs and `int n, int k`.
Its accumulator is `int sum = 0` — see puzzle 12 for what that does to
non-integer data.

Its constants are `TPB = 8`, `MAX_CONV = 4` against a 5-element signal, which
is one partial block and no interior. Real ones here: 256 threads, 391 blocks,
a 14-element halo. The halo-loading logic upstream is also written in terms of
`b_size` rather than `k - 1`, which loads one slot more than the window needs
and only works while `TPB >= 2 * b_size`.

## Run

```
make run P=13
make run P=13 MODE=solution
make check P=13
```

Expected: one `PASS conv1d` line, and `make check` clean under all three
sanitizers.
