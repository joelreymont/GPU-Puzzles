# Puzzle 8 — Shared memory

New idea: **memory that belongs to a block, and a barrier that makes it safe.**

Everything so far has used *global* memory: the big off-chip pool `cudaMalloc`
hands out, visible to every thread in the grid, and slow — a few hundred cycles
of latency, and on this box roughly 273 GB/s of bandwidth for the whole GPU to
share. **Shared memory** is a small block of SRAM physically on the SM, tens of
cycles away, and it is per-block: every thread in a block sees the same shared
array, and no thread in any other block sees it at all. It is a scratchpad you
manage by hand, not a cache.

The pattern this puzzle installs is the one every later shared-memory kernel
uses:

1. **Stage.** Each thread copies one element from global memory into shared.
2. **Barrier.** `__syncthreads()`.
3. **Compute.** Threads read shared memory and write their results.

Step 2 is the part that is easy to get wrong. `__syncthreads()` is a barrier
for the block: no thread proceeds past it until every thread in the block has
reached it, and — just as important — every shared-memory write issued before
it is visible to every thread after it. Without that guarantee, step 3 is a
race: thread 5 could read `tile[6]` before thread 6 has written it.

Two rules about it that cost people days:

- **Every thread in the block must reach the same `__syncthreads()`.** The
  programming model permits a barrier inside a conditional only if the
  condition evaluates identically for the whole block; otherwise the behaviour
  is undefined. Putting it inside `if (i < n) { ... }` breaks that in the last
  block, where 163 threads enter and 93 do not. Guard the *load*; leave the
  barrier outside the branch, where all 256 threads run into it. This is why
  the skeleton's guards are two separate `if` statements with the barrier
  between them rather than one `if` around everything.
- It synchronises **one block**, not the grid. There is no `__syncgrid()` in an
  ordinary kernel; blocks stay independent.

This puzzle's shared array is **dynamic**: declared `extern __shared__ float
tile[];` with no size, and sized at launch by the third argument in
`<<<grid, block, smem>>>`. The runner asks for `blockDim.x` floats. The
alternative — `__shared__ float tile[256];` — is fixed at compile time; both
appear in later puzzles.

## Task

Fill in `skeletons/p08_shared/kernel.cu`:

```
out[i] = a[i] + 10   for every i in [0, n),  n = 100003
```

routed through shared memory: stage `a`'s tile into `tile`, synchronise, then
compute from `tile`.

- Input: `a`, `n` floats. Output: `out`, `n` floats.
- Launch: 391 blocks of 256 threads, 1024 bytes of dynamic shared memory each.
- `i` (global index) and `local_i` (index within the block, i.e. within `tile`)
  are already computed for you. Note that these are different numbers and that
  `tile` is indexed by the local one — `tile[i]` would run off the end of a
  256-float array on the very first block.

**approx 7 lines.**

## Verification

Tolerance zero, `NaN` poison, seeded input, same 391-block geometry as puzzle 6
so the only new thing is the staging.

One honest caveat about what this test can and cannot prove. In *this* kernel
each thread writes `tile[local_i]` and then reads back `tile[local_i]` — its
own slot, which no other thread touches. So there is no cross-thread dependency
yet, and the barrier is not actually load-bearing: measured on this box, a
version with `__syncthreads()` deleted prints `PASS shared` and is clean under
`compute-sanitizer --tool racecheck` (0 hazards), because there is genuinely no
race to find. You are learning the *shape* here, with the data flow kept
trivial on purpose.

The barrier stops being optional the moment a thread reads a slot it did not
write — which is the entire point of shared memory and is what the pooling,
convolution and reduction puzzles do. Write it now, understand why it is there,
and when a later kernel corrupts its answer without one you will recognise the
symptom.

`-Xptxas -v` reports the cost of the barrier in the build output for this
kernel: `Used 8 registers, used 1 barriers`.

## Signature correction

Upstream is `Shared(float* A, float* C, float size)`; here, `int n`. See
puzzle 3.

## Run

```
make run P=08
make run P=08 MODE=solution
make check P=08
```

Expected: one `PASS shared` line, and `make check` clean under all three
sanitizers.

Do not read a clean `synccheck` as a licence on the barrier-in-a-branch rule
above. Measured on this box: the variant with the whole body wrapped in
`if (i < n) { ... }`, barrier included, prints `PASS shared` and reports
`ERROR SUMMARY: 0 errors` under `compute-sanitizer --tool synccheck`. The rule
is a rule of the programming model, and the tools did not enforce it here —
which makes undefined behaviour in a barrier more dangerous than the kind that
crashes, not less. It is a property of the code that all 256 threads reach the
barrier, and you have to establish it by reading the code.
