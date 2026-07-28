# Puzzle 12 — Dot product

New idea: **the tree reduction** — how a block turns `blockDim.x` numbers into
one, and why there is a barrier between every level of it.

This is the canonical shared-memory collective. Puzzles 15, 24, 25 and 27 are
all variations on it, and puzzle 24 exists specifically to replace the pattern
you write here with warp primitives. Get it right once and read the rest as
edits to it.

```
out[0] = sum over i of a[i] * b[i]
```

## The shape

Each thread computes one product into its own slot of a shared array. Then the
block folds those `blockDim.x` partials down to one number by halving the
number of live threads at each step:

```
step 1:  512 threads:  cache[t] += cache[t + 512]      for t < 512
step 2:  256 threads:  cache[t] += cache[t + 256]      for t < 256
step 3:  128 threads:  cache[t] += cache[t + 128]      ...
...
step 10:   1 thread:   cache[0] += cache[0 + 1]
```

Ten steps for 1024 threads, not 1024 — the depth is `log2(blockDim.x)`, which
is the entire reason this is worth writing instead of having thread 0 loop over
the array. The total lands in `cache[0]`.

Write it as a loop whose stride starts at half the block, halves on every
iteration, and stops after the stride of 1.

`blockDim.x` is 1024 here — a power of two — so the halving is exact and no
partial level needs special handling. That is a real constraint on this
pattern, not an accident of the launch: with 1000 threads instead, `s` would
be 500, then 250, then 125, then 62 — and the element at index 124 would never
be added into anything. Powers of two, or a shape that handles the odd element
explicitly.

## Why `__syncthreads()` goes *inside* the loop

At every level, some threads read slots that other threads wrote at the
previous level. Thread 0 reads `cache[1]` at the last step, and `cache[1]` was
written by thread 1 at the step before. There is no ordering between two
different warps unless you impose one, so the barrier belongs **inside** the
loop, after the additions of each level:

```
for (...) {
  if (local_i < s) { ... }      // this level's additions
  __syncthreads();              // every thread waits for all of them
}
```

Note where it sits: after the `if`, not inside it. Only the low `s` threads do
the addition, but **all** `blockDim.x` threads must reach the barrier — the
same rule as puzzle 8, and here it is easy to violate by tucking the barrier
inside the conditional where the work is.

Two `__syncthreads()` statements in the source, then: one after staging, and
one at the end of the loop body — the second of which executes ten times, once
per level.

## Guards, without killing any threads

`n = 1000` and the block has 1024 threads, so 24 threads have no element. They
must still:

- put something in their shared slot — the additive identity, `0.0f`, because
  their slot *will* be read by the reduction. Leaving it unwritten reads
  whatever the last kernel left in that shared memory;
- reach every barrier, so they cannot `return` early.

"Give the idle lanes the identity and keep them alive" is the standard answer
to a reduction that does not divide evenly, and it comes back in every later
reduction puzzle.

## Task

Fill in `skeletons/p12_dotproduct/kernel.cu`:

```
out[0] = sum over i in [0, n) of a[i] * b[i],   n = 1000
```

- Inputs: `a`, `b`, `n` floats each. Output: `out`, one float.
- Launch: **one** block of 1024 threads, 4096 bytes of dynamic shared memory.
- `i` and `local_i` are computed for you.
- Stage the product, barrier, fold, then have one thread write `out[0]`.

**approx 13 lines.**

One block is the whole grid here, which is why `out[0]` can just be assigned.
A multi-block dot product needs the block totals combined somehow — an
`atomicAdd` per block, or a second kernel over the per-block partials. Puzzle
24 does it with atomics.

## Verification

Seeded inputs, `NaN` poison on the single output, **relative tolerance `1e-4`**.

A float reduction reassociates: the kernel adds the 1000 products in a
different order from the reference's sequential `double`, so bitwise equality
is not expected and `==` would be a lie. Setting the tolerance from one
measurement would also be a lie — a single run is a single draw — so it is set
from a 512-seed sweep of this exact kernel and this exact size:

| association | worst relative deviation over 512 seeds | exactly equal |
|---|---|---|
| tree (the solution) | `2.742e-06` | 143 seeds |
| one thread summing sequentially | `9.030e-06` | 23 seeds |

Both are legitimate answers to this puzzle — the second is what the upstream
solution does — so the tolerance has to accept both. `1e-4` is the tightest
decade clearing both by more than 10×. At the runner's own seed both happen to
land on the correctly-rounded value, deviation `0.000e+00`, which is exactly
why the tolerance is not set from it.

The table also shows the tree is the more accurate of the two, by about 3×.
Shallower accumulation chains round fewer times: depth 10 versus depth 1000.
The parallel algorithm is not a numerical compromise here, it is an
improvement.

Build output for the solution on this box: `Used 12 registers, used 1 barriers`.

## The barrier deletion experiment — and why `make run` is not enough

Delete the `__syncthreads()` inside the reduction loop, leaving the staging
one. Measured on this box:

```
$ ./p12_noloopsync
# NVIDIA GB10, sm_121, 48 SMs
PASS dot_product
$ echo $?
0
```

**Thirty runs out of thirty.** The kernel is racy and the answer is right every
time, because a single block of 32 warps on an otherwise idle GPU tends to
advance in near-lockstep — the schedule that would corrupt it is allowed but
does not happen to occur.

Now the same binary under `compute-sanitizer --tool racecheck`:

```
========= Error: Race reported between Read access at DotProduct(...)+0x1d0
=========     and Write access at DotProduct(...)+0x1f0 [1916 hazards]
# NVIDIA GB10, sm_121, 48 SMs
FAIL dot_product at index 0: got -8.77355 want -12.7413
========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)
```

Two things happened there. racecheck named the race — 1916 hazards between one
read and one write of shared memory — and the *value came out wrong*, because
the tool's instrumentation perturbs the schedule enough to expose what was
always permitted.

This is the puzzle-8 caveat cashed in. There the barrier was not load-bearing
and racecheck honestly found nothing. Here it is load-bearing, the tool finds
it, and the plain run does not. `make check` is not a formality.

If you want to see the same defect fail without a sanitizer, look at puzzle 15:
it is this reduction with 64 blocks instead of one, and with the loop barrier
deleted it fails the plain run **20 times out of 20**. Same kernel, same bug,
more blocks competing for SMs, no lockstep to hide behind. A race that "passes"
is a race that has not been given a reason to lose yet.

## Signature correction

Upstream (dshah3's puzzle 10, `dotproduct`) is `DotProduct(float* A, float* B,
float* C, float size)`; here, `const float*` inputs and `int n`. See puzzle 3.

Its answer also accumulates into an `int`:

```
int sum = 0;
for (int k = 0; k < size; k++) sum = sum + sharedMem[k];
```

which truncates every partial sum to a whole number. It passes upstream because
that harness feeds it `A[i] = i`, `B[i] = i`, so every value in play is already
an integer. Every product here has magnitude below 1, so each truncation takes
the running total straight back to zero — measured against this runner,
`FAIL dot_product at index 0: got 0 want -12.7413`. Accumulate in `float`.

## Run

```
make run P=12
make run P=12 MODE=solution
make check P=12
```

Expected: one `PASS dot_product` line, and `make check` clean under all three
sanitizers.
