# Puzzle 24 — Warp Fundamentals

A block of 256 threads is not the smallest unit the hardware schedules. The SM
issues instructions to **warps** of 32 threads that execute in lockstep (SIMT).
Within one warp, threads can exchange registers directly — no shared memory, no
`__syncthreads()` — via the warp-level primitives. This puzzle replaces the
shared-memory tree reduction you would write at block level with warp
primitives.

## Task

Complete three kernels in `skeletons/p24_warp_fundamentals/kernel.cu`. The
runner allocates `n = 1000` floats — deliberately **not** a multiple of 32 or
of the 256-thread block — so your guard logic is exercised.

### 1. `WarpDotShfl` (approx 6 lines)

Dot product `out[0] = Σ a[i]·b[i]`, hand-rolled:

- Each thread computes one product (already done: out-of-range lanes hold `0`).
- Reduce the 32 partials of each warp into lane 0 with `__shfl_down_sync`,
  halving the stride each step: 16, 8, 4, 2, 1.
- Lane 0 of each warp adds its warp's total to `out[0]` with `atomicAdd`.
- Lane index: `threadIdx.x % 32`.

### 2. `WarpDotCG` (approx 2 lines)

Same reduction through cooperative groups: the tile is already constructed;
use `cg::reduce(warp, partial, cg::plus<float>())` and have tile rank 0 do the
`atomicAdd`.

### 3. `WarpCountGT` (approx 3 lines)

Count how many elements exceed a threshold — without any per-thread atomics:

- `__ballot_sync(mask, pred)` returns a 32-bit word with one bit per lane whose
  predicate is true.
- `__popc` counts the bits; one lane per warp adds the count to `out[0]`.

## The divergence trap

Every `*_sync` primitive names the lanes that participate (`0xffffffff` = all
32). Each participating lane **must reach the call** — if a lane exits early
(e.g. `if (i >= n) return;`), the remaining lanes call the primitive with a
mask naming a dead lane, which is undefined behavior. That is why the skeletons
guard with a *value* (`partial = 0` for out-of-range lanes) and keep every lane
alive through the reduction.

Since Volta, threads in a warp have independent program counters and are *not*
guaranteed to execute in lockstep through divergent code. `__syncwarp(mask)`
reconverges the named lanes; the `*_sync` primitives have that barrier built
in. `__activemask()` tells you which lanes happen to be converged right now —
it is a query, not a fence, and is **not** a safe substitute for an explicit
mask.

## Run

```
make run P=24                 # your kernel
make run P=24 MODE=solution   # reference
make check P=24               # sanitizers
```

Expected: three `PASS` lines. The dot products are compared with relative
tolerance — warp reduction reassociates floating-point addition, so bitwise
equality with the CPU reference is not expected and not required.
