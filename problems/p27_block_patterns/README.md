# Puzzle 27 — Block-Level Patterns

Puzzles 24–26 stayed inside one warp on purpose. Every primitive there —
`__shfl_down_sync`, `__shfl_xor_sync`, `__shfl_up_sync`, `__ballot_sync` — moves
a value from one lane's register file to another lane's register file **inside a
single warp**, and that is the entire extent of what it can do. No shared
memory, no `__syncthreads()`, no staging array, because there was never anything
to stage: the 32 values were already sitting in one warp's registers and the
warp executes them in lockstep.

A thread block is 256 threads: eight warps. There is no shuffle across a warp
boundary. There is no instruction that lets warp 3 read warp 5's registers. The
*only* channel between warps is memory, and the only way to know the other warp
has finished writing is a barrier. That is the whole content of this puzzle:

```
within a warp    registers -> registers      __shfl_*_sync
across warps     registers -> shared -> registers, with __syncthreads() between
```

Everything block-level — `cub::BlockReduce`, `cub::BlockScan`, every "block sum"
in every library you will ever read — is that sandwich. Build it by hand once
and you can recognise it everywhere.

## Task

Complete four kernels in `skeletons/p27_block_patterns/kernel.cu`. The runner
allocates `n = 1000` floats in `[-1, 1)` — deliberately **not** a multiple of 32
or of the 256-thread block, so guard logic is exercised — and launches a grid of
4 blocks.

`TPB = 256`, `WARP = 32` and `WARPS = TPB / WARP = 8` are already defined at the
top of the file. `TPB` is a `constexpr` because CUB's block collectives are
templated on the block size; it must match the launch configuration exactly.

### I/O contract

| kernel | output | meaning |
|---|---|---|
| `BlockSumTwoStage` | `out[0]` | `Σ a[j]`, `j = 0 .. n-1` — the whole grid, accumulated across blocks with `atomicAdd` |
| `BlockSumCUB` | `out[0]` | the same total, the same way |
| `BlockScanCUB` | `out[0..n-1]` | segmented inclusive prefix sum: `out[i] = Σ a[j]`, `j = seg .. i`, where `seg = (i / 256) * 256` |
| `HistogramShared` | `hist[0..nbins-1]` | count of elements falling in each of `nbins = 64` equal-width bins over `[-1, 1)` |

The two sums and the histogram are `atomicAdd` accumulators, so the runner
zeroes them before each launch — zero is the identity and the only correct
initial value. `BlockScanCUB` owns every element, so its buffer is poisoned
with `0xff` (a NaN bit pattern) instead: an output you forget to write is a
`FAIL`, not a lucky zero.

Every kernel is given its value guard already:

```cpp
float v = (i < n) ? a[i] : 0.0f;
```

Use it. Do **not** replace it with `if (i >= n) return;`. A block collective —
a shuffle with a full mask, a `__syncthreads()`, a `cub::Block*` call — is a
*collective*: every thread named in it must reach it. The tail block here has
232 in-range threads out of 256, so an early return leaves 24 threads absent
from every barrier in the kernel. The identity element `0.0f` costs nothing and
keeps all 256 threads in the collective.

### 1. `BlockSumTwoStage` (approx 12 lines)

The canonical shape, by hand:

```
stage 1   each of the 8 warps reduces its 32 values with __shfl_down_sync
          lane 0 of warp w writes the result to partials[w]
          __syncthreads()
stage 2   warp 0 loads the 8 partials, reduces them with shuffles
          thread 0 atomicAdds the block total into out[0]
```

`__shared__ float partials[WARPS]` is already declared, as are `lane` and
`warp`. Two things to get right:

- **Stage 2 needs its own value guard.** Warp 0 has 32 lanes and there are only
  8 partials. Lanes 8..31 must contribute `0`, not `partials[lane]` — reading
  `partials[8]` is a shared-memory buffer overrun.
- **The barrier goes between the stages, not inside a branch.** Every thread in
  the block reaches it, including the ones whose warp is not warp 0.

### 2. `BlockSumCUB` (approx 4 lines)

The same total from `cub::BlockReduce<float, TPB>`. The typedef and the
`__shared__ TempStorage` are given; you supply the collective call and the one
thread that publishes the block's result. `Sum()` returns a meaningful value in
thread 0 only.

### 3. `BlockScanCUB` (approx 4 lines)

`cub::BlockScan<float, TPB>::InclusiveSum(v, s)` — an *all-to-each* collective:
where the reduce leaves one answer in one thread, the scan leaves a different
answer in all 256. Each in-range thread writes its own `out[i]`. Note that the
scan is segmented by construction, not by any code you write: a block sees only
its own 256 elements, so the running sum restarts at every block boundary.

### 4. `HistogramShared` (approx 10 lines)

The application. `extern __shared__ int bins[]` is sized by the launch
(`nbins * sizeof(int)`). Four phases:

```
zero the shared bins,  striding k = threadIdx.x, k += blockDim.x
__syncthreads()
each in-range thread bins its own value and atomicAdds 1 into shared
__syncthreads()
merge shared into global with atomicAdd, striding the same way
```

The bin index must be computed exactly like this, because the CPU reference
computes it with the character-identical expression and float rounding has to
agree on both sides:

```cpp
int b = (int)((v + 1.0f) * 0.5f * nbins);
```

then clamped into `[0, nbins-1]`. The clamp is not decoration. `v = 1.0f` maps
to `b = nbins`, one past the end of `bins[]`. Verified: dropping the clamp and
feeding one element of `1.0f` gives
`Invalid __shared__ atomic of size 4 bytes` under `compute-sanitizer memcheck`.
The runner's LCG tops out at `1 - 2⁻²³`, so this input never triggers it —
which is exactly why the clamp is a correctness requirement rather than
something to add after a test fails.

Both barriers are load-bearing. The first separates zeroing from binning; the
second separates binning from the merge. Neither is optional and neither can be
replaced by `__syncwarp()` — the whole point is that eight warps share one
`bins[]`.

## Why the two-stage shape, and not a tree in shared memory

The textbook block reduction stages all 256 values into shared memory and folds
with `stride = 128, 64, 32, ...`, barriering at every level: 8 barriers, 256
floats of shared memory, and every level is a shared load, an add, and a store.

The two-stage version does the first five levels in registers, where a warp
already provides the lockstep for free, and only spills the 8 surviving partials
to memory. One barrier. 32 bytes of shared memory. That is why it is the shape
every library converged on, and why puzzles 24–26 come first: the warp reduce
is not a warm-up for the block reduce, it is stage 1 of it.

## What CUB actually generates

Measured on this box (GB10, `sm_121`, CUDA 13.0) with `-Xptxas -v` on the built
solution and `cuobjdump -sass` on the resulting binary:

| kernel | regs | static smem | shuffles | FADD | LDS | STS | BAR.SYNC |
|---|---|---|---|---|---|---|---|
| `BlockSumTwoStage` | 13 | 32 B | 8 `SHFL.DOWN` | 8 | 1 | 1 | 1 |
| `BlockSumCUB` | 12 | 48 B | 5 `SHFL.DOWN` | 12 | 3 | 1 | 1 |
| `BlockScanCUB` | 25 | 1184 B | 12 `SHFL.UP` | 25 | 9 | 9 | 2 |
| `HistogramShared` | 12 | 0 B (256 B dynamic) | — | — | 1 | 1 | 2 |

Read the two sum rows against each other. Both are the same sandwich — warp
shuffles, one `STS`, one barrier, shared reload — and both finish with a single
`REDG.E.ADD.F32` (a fire-and-forget global reduction, not `ATOM`, because
nothing reads the return value of `atomicAdd`). The difference is what they do
with the 8 partials:

- The hand-rolled version runs a **second shuffle ladder**: 5 `SHFL.DOWN` for
  stage 1 plus 3 more for the 8 partials, 8 in total, 8 `FADD`, one 32-bit
  `LDS`.
- CUB runs **5 `SHFL.DOWN` and then rakes**: the three loads are `LDS.128`,
  `LDS.64` and a 32-bit `LDS`, pulling all 8 partials into one thread with
  vector accesses, then 7 serial `FADD` on top of stage 1's 5 — 12 total.

Neither is obviously better and the measurement is the point: CUB did not
produce your code, it produced a different point on the shuffle-vs-load
trade-off (fewer cross-lane ops, more scalar adds, wider memory instructions),
and it chose that without being asked. It also reserves 48 bytes of temp
storage where the hand-rolled version declares 32.

The scan row is the one worth staring at. A block-wide inclusive scan costs
**25 registers and 1184 bytes of shared memory**, against 12 registers and 48
bytes for the reduce — roughly 25× the shared memory, two barriers instead of
one, and 9 loads and 9 stores instead of one each. All-to-each is genuinely
more expensive than all-to-one. If a scan is only being used to produce the
total, use the reduce.

`HistogramShared` has one more measured surprise: the shared increment compiles
to a single `ATOMS.POPC.INC.32`. `atomicAdd(&bins[b], 1)` with a literal `1` is
recognised and warp-aggregated in hardware — the lanes of a warp that land in
the same bin are counted with a population count and applied as one atomic,
rather than 32 serialised read-modify-writes. You get that for free from
writing the increment in the obvious way.

## Why privatise the bins at all

The alternative is one line: every thread `atomicAdd`s straight into
`hist[b]` in global memory. It is correct. Measured on this box, `n = 2²⁴`
elements into 64 bins, 20 iterations each, both kernels verified to produce
identical counts:

| variant | time |
|---|---|
| shared-privatised bins, merged per block | 0.304 ms |
| direct `atomicAdd` to global | 3.96 ms |

**13.0×.** The reason is contention, not bandwidth: 16.7M atomics land on 64
global addresses, so every SM in the machine serialises on the same 64 cache
lines. Privatising confines that contention to one block's shared memory —
where it is further collapsed by the `ATOMS.POPC.INC` aggregation above — and
leaves only `gridDim.x * nbins` global atomics at the end. The global variant's
SASS is a single `REDG.E.ADD` per thread with no aggregation at all.

The ratio moves with the bin count, because more bins means less contention for
the version that has none to reduce:

| bins | shared | global | ratio |
|---|---|---|---|
| 8 | 0.241 ms | 2.95 ms | 12.2× |
| 64 | 0.305 ms | 3.97 ms | 13.0× |
| 256 | 0.600 ms | 3.61 ms | 6.0× |

And it disappears entirely at this puzzle's size. At `n = 1000` the two are
indistinguishable (0.0039 ms vs 0.0038 ms, launch-bound): 4 blocks cannot
contend with each other. The pattern is here because it is the right one at
scale, not because it is faster on the puzzle's input. Do not take "I measured
no difference" as evidence about a technique when the input is too small to
express the effect it targets.

## Tolerance, and what the numbers mean

The runner compares floats with a relative tolerance of `1e-5` and the
histogram **exactly** — integer counts do not reassociate, so anything other
than an exact match on all 64 bins is a bug, not rounding.

Measured at `n = 1000`, seed 27, against the reference's sequential double
accumulator:

```
want                35.5244560
BlockSumTwoStage    35.5244522     rel. dev. 1.1e-7
BlockSumCUB         35.5244598     rel. dev. 1.1e-7
BlockScanCUB        worst rel. dev. 1.9e-6, at index 229
```

Note that the two sum kernels do not agree with each other in the last two
digits. They are summing the same 1000 floats and are both correct; they
associate the additions differently, so they round differently. This is the
concrete reason the harness never uses `==` on a float reduction. Both were
bit-stable across 200 consecutive runs on this box, but that is an observation
about four blocks racing on one address, not a guarantee — `atomicAdd` ordering
across blocks is not defined.

If a *large* deviation appears it is not floating point:

- `block_sum_*` off by roughly a factor of 2, or by one block's worth — a
  barrier in the wrong place, or a partial that was never written.
- `block_scan_cub` correct at index 0 of each 256-element segment and wrong
  after — the scan is not seeing the whole block.
- `histogram_shared` low by a few counts — a missing barrier, so some block
  merged its bins before every thread had finished incrementing them.

## Race-freedom is part of the answer here

`make check` is not a formality for this puzzle. Both correctness and the
sanitizers are load-bearing, and **neither catches everything** — measured, by
breaking the solution three ways:

| defect | `make run` | racecheck | memcheck |
|---|---|---|---|
| stage-2 reads `partials[lane]` with no `lane < WARPS` guard | **PASS** | clean | `Invalid __shared__ read` |
| `__syncthreads()` removed from `BlockSumTwoStage` | `FAIL … got 18.7 want 35.5` | 1 hazard | clean |
| second `__syncthreads()` removed from `HistogramShared` | `FAIL … got 6 want 7` | clean | clean |

The first row is the one to remember: a kernel that reads 24 words past the end
of a shared array produced the right answer on this input, on this hardware, on
every run. The test suite said `PASS`. Only `memcheck` disagreed. The third row
is its mirror — a real race that `racecheck` did not report and only the
numerical check caught.

A puzzle 27 solution is done when it passes `make run` **and** all three
sanitizers report zero.

## Run

```
make run P=27                  # your kernel
make run P=27 MODE=solution    # reference
make check P=27                # memcheck + racecheck + synccheck, all must be 0
```

Expected: four `PASS` lines.
