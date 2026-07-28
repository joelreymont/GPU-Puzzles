# Puzzle 11 — Pooling

New idea: **a window that reaches outside the block.**

Puzzle 8 put shared memory in the right shape — stage, barrier, compute — but
was honest that the barrier there was not load-bearing: every thread read back
the slot it had just written itself, so there was no cross-thread dependency
and nothing to synchronise. This is the puzzle where that changes.

Each output is a sliding window over the input:

```
out[i] = a[i-2] + a[i-1] + a[i]
```

so every element of `a` is read by **three** different outputs. That is the
reason to stage it: fetch each element from global memory once, then read it
three times from shared memory, which is tens of cycles away instead of
hundreds. And it is the reason the barrier now matters — thread 7 reads the
slots holding `a[5]` and `a[6]`, and threads 5 and 6 are the ones that put them
there.

## The halo

A block of 256 threads produces 256 outputs, but those outputs read **258**
distinct inputs: the block's own 256, plus the two that belong to the block on
its left. Thread 0 of block 1 needs `a[254]` and `a[255]`, and no thread of
block 1 loads them.

Those extra elements are the **halo**, and staging them is the block's job.
The tile is therefore wider than the block, and the mapping is shifted:

```
tile[m] holds a[blockIdx.x * blockDim.x + m - HALO],  HALO = 2
```

so a thread's own element is at `tile[HALO + threadIdx.x]`, and the three
elements of its window are the three consecutive slots ending there:
`tile[local_i]`, `tile[local_i + 1]`, `tile[local_i + 2]`. The shift is the
whole trick; index the tile by the *global* `i` and you run off the end of a
258-float array in the first block.

Two threads per block have to do a second load to fill the halo, and that
second load is guarded differently from the first: it can fall off the *front*
of the array (block 0's halo does not exist), while the ordinary load can fall
off the *back* (the last block's tail).

## Truncation at the front

`out[0]` and `out[1]` have no full window. They sum only the elements that
exist:

```
out[0] = a[0]
out[1] = a[0] + a[1]
```

which is what you get for free if the missing halo slots hold `0` — zero is the
additive identity, so a nonexistent element contributing zero *is* the
truncation rule. That is worth internalising now, because every reduction in
puzzles 12–16 handles its out-of-range lanes exactly the same way: give them
the identity, keep them alive, let them reach the barrier.

Note what the alternative would be. Guarding the *store* instead — "only write
`out[i]` when the whole window exists" — leaves `out[0]` and `out[1]` untouched,
which is what the upstream answer does. Against this runner that is an
immediate `FAIL`, because the output buffer is poisoned with `NaN` and an
output nobody wrote stays `NaN`.

## Task

Fill in `skeletons/p11_pooling/kernel.cu`:

```
out[i] = a[max(0, i-2)] + ... + a[i]   for every i in [0, n),  n = 100003
```

- Input: `a`, `n` floats. Output: `out`, `n` floats.
- Launch: 391 blocks of 256 threads, `(256 + HALO) * 4` bytes of dynamic
  shared memory each.
- `i` and `local_i` are computed for you; `W` and `HALO` are given at the top
  of the file.
- Stage your own element, stage the halo if you are one of the first `HALO`
  threads, barrier, then compute. Keep the barrier outside every `if` — all
  256 threads must reach it, for the reason puzzle 8 spells out.

**approx 12 lines.**

## Verification

Seeded input, `NaN` poison, and **tolerance zero** — which for a sum is worth
justifying rather than asserting.

`fill_rand` builds each value as `k / 2^24 * 2 - 1` for an integer `k`, so
every input is an exact multiple of `2^-23` with magnitude below 1. Two such
values always add *exactly* in `float`: their sum is a multiple of `2^-23`
smaller than 2, and every such number is representable. So whichever pair the
kernel adds first, that addition is exact, and the one remaining addition
rounds the true three-element sum exactly once — which is precisely the single
rounding the reference performs when it casts its `double` accumulator back to
`float`. The two must agree bit for bit, in whichever order the slots are
summed.

Measured, over all 100003 outputs: deviation `0.000e+00`, both summing the
window left to right and summing it right to left. This is a property of *this
input*, not of pooling — feed the same kernel arbitrary floats and the
double-rounding reappears. It is here because a tolerance that can be zero
should be zero.

## The barrier is real now, and both tools say so

Delete the `__syncthreads()` and run it, measured on this box:

```
FAIL pooling at index 2: got 0.617922 want 0.571111
```

**20 runs out of 20, always at index 2** — the first output whose window
reaches a slot the thread did not write itself. Outputs 0 and 1 read halo slots
that thread 0 and thread 1 filled themselves, so they survive; output 2 needs
`tile[2]`, which is thread 0's, and it is wrong every time.

Note that thread 2 and thread 0 are in the *same warp*, which is the part worth
absorbing. "Same warp" is not a substitute for a barrier: the compiler is free
to schedule this thread's loads before that thread's stores, and with the
barrier gone there is nothing telling it not to. A missing barrier is not only
a scheduling risk, it is a licence for the compiler.

Under `compute-sanitizer --tool racecheck` the same binary reports

```
Race reported between Write access at Pooling(...)+0x130
    and Read access at Pooling(...)+0x120 [393960 hazards]
    and Read access at Pooling(...)+0x200 [395564 hazards]
RACECHECK SUMMARY: 2 hazards displayed (1 error, 1 warning)
```

Compare that with puzzle 8, where deleting the same line changed nothing and
racecheck found `0 hazards` — correctly, because there was no race. The tool
was not being lenient there and is not being strict here. It reports what the
data flow actually is, and the data flow is what changed.

Build output for the solution on this box: `Used 12 registers, used 1 barriers`.

## Signature correction

Upstream (dshah3's puzzle 9, `pooling`) is `Pooling(float* A, float* C, float
size)`; here, `const float* a, float* out, int n`. See puzzle 3 for the
`float size` correction.

The upstream answer also writes nothing for `i < 2` and nothing at all for the
first two threads of *every* block after the first — its `local_i - 2 >= 0`
test is a local index, so with more than one block it silently leaves two
outputs per block unwritten. Its harness runs one block of 4 elements and
asserts only `i >= 2`, so it never sees this. The halo is what fixes it.

## Run

```
make run P=11
make run P=11 MODE=solution
make check P=11
```

Expected: one `PASS pooling` line, and `make check` clean under all three
sanitizers.
