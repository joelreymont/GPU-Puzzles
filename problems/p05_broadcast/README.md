# Puzzle 5 — Broadcast

New idea: **the output shape is not the input shape.**

Every kernel so far read element `i` and wrote element `i`. Here a thread reads
two vectors and writes a matrix:

```
a is a column vector, rows elements
b is a row vector,    cols elements
out[r][c] = a[r] + b[c]
```

Each element of `a` is read by an entire row of threads; each element of `b` is
read by an entire column. That fan-out is what "broadcast" means, and it is
free to write — the index expression simply drops a subscript. It is also the
first hint of why shared memory exists (puzzle 8): `a[r]` is fetched from
global memory `cols` separate times, and eventually you will want to fetch it
once.

Note what the two reads look like across one warp. Consecutive threads have
consecutive `col`, so `b[col]` is 32 consecutive floats — one coalesced
transaction. `a[row]` is the *same* address for all 32 of them, which the
hardware broadcasts from a single fetch. Both are ideal, for opposite reasons.

The launch geometry and guards are exactly puzzle 4's: a `dim3(32, 32)` block
over a 25×31 matrix, both dimensions overhanging.

## Task

Fill in `skeletons/p05_broadcast/kernel.cu`:

```
out[r][c] = a[r] + b[c]   for every r in [0, rows), c in [0, cols)
```

- Inputs: `a` (`rows` floats), `b` (`cols` floats).
- Output: `out`, `rows * cols` floats, row-major.
- Launch: one block, `dim3(32, 32)`. `row` and `col` are already computed.
- Guard both dimensions before touching memory. The three arrays have three
  different lengths, and `col` reaching 31 while `a` holds 25 floats is an
  out-of-bounds read waiting to happen.

**approx 4 lines.**

## Verification

Tolerance zero, `NaN` poison, seeded inputs — and unlike puzzle 4, the value
check here really does pin the index mapping down. `out[r][c]` depends on `r`
and `c` through different arrays, so swapping the two axes is a different
answer, not a relabelled one. Measured against this runner, a kernel that
computes `a[col] + b[row]` fails at index 1.

`rows != cols` is what makes that true.

## Run

```
make run P=05
make run P=05 MODE=solution
make check P=05
```

Expected: one `PASS broadcast` line.
