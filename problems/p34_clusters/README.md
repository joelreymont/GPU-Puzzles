# Puzzle 34 — Thread Block Clusters and Distributed Shared Memory

Every puzzle so far has had two levels of cooperation. Puzzles 24–26 stayed
inside a warp, where lanes exchange registers directly. Puzzle 27 stepped up to
the block, where the only channel between warps is shared memory and the only
way to know a write landed is `__syncthreads()`. Above the block there was
nothing: two blocks of the same grid could not synchronise, could not see each
other's shared memory, and were not even guaranteed to be resident at the same
time.

SM90 added a level in between:

```
grid
└── cluster        <- new: a fixed number of blocks, co-scheduled, able to
    └── block         barrier with each other and address each other's smem
        └── warp
            └── thread
```

A cluster is a group of thread blocks that the hardware guarantees are resident
**simultaneously on the same GPC**. That guarantee buys two things a grid can
never offer:

- `cluster.sync()` — a barrier across all blocks of the cluster.
- **Distributed shared memory (DSMEM)** — every block can read, write and
  atomically update the shared memory of every other block in its cluster.
  `cluster.map_shared_rank(ptr, rank)` takes a pointer to a shared variable and
  returns a pointer to *that same variable in another block's shared memory*.

The grid is still the grid: clusters do not synchronise with each other, and
nothing here makes the whole grid cooperative. What has changed is that "shared
memory is private to one block" — true in every earlier puzzle — is now false
for the four blocks of a cluster.

## Does this hardware even have clusters?

This is the one puzzle in this repo whose existence was in question, and the
question was not answerable from documentation. GB10 is `sm_121`, consumer-class
Blackwell. Published analyses of SM120 state that only cluster size 1 is
supported, with a *silent downgrade* on launch and a subsequent `cluster.sync()`
deadlock; NVIDIA forum discussion claims consumer Blackwell supports 8-block
clusters minus multicast. Both cannot be right, and a puzzle that hangs the box
is worse than no puzzle.

So it was measured. `scripts/probe_caps.cu` queries the attribute, asks the
occupancy API, launches a **real** 2-block cluster and has the kernel report
`cluster_group::num_blocks()`, and only then — behind a host-side 10-second
deadlock timeout — attempts `cluster.sync()`. Its findings are cached in
`common/caps.mk` and every `make` reads them:

```
CAPS_ARCH := sm_121
CAPS_CLUSTER_LAUNCH := 1      cudaDevAttrClusterLaunch
CAPS_CLUSTER_MAX := 8         cudaOccupancyMaxPotentialClusterSize
CAPS_CLUSTER_REAL := 2        a real 2-block cluster genuinely grouped
CAPS_CLUSTER_SYNC := 1        cluster.sync() completed, no deadlock
```

Clusters work here. The root `Makefile` gates this puzzle on
`CAPS_CLUSTER_SYNC = 1` and refuses to build it otherwise, naming the probe in
the error, so that if the hardware story ever changes the build stops instead of
shipping a kernel that hangs.

What `sm_121` still does not have: **multicast TMA** (the cluster-wide
`cp.async.bulk.tensor` broadcast that Hopper uses to load one tile into all
blocks of a cluster at once) and **TMEM**/`tcgen05`. This puzzle uses no TMA at
all. DSMEM and `cluster.sync()` are the parts that exist, and they are the parts
this puzzle teaches.

Two facts measured on this box while building the puzzle, both worth knowing:

| experiment | result |
|---|---|
| `__cluster_dims__(8,1,1)`, grid 3912, both kernels | `num_blocks()` = **8**, both outputs correct |
| `__cluster_dims__(16,1,1)` | compiles; `cudaOccupancyMaxPotentialClusterSize` returns `cudaErrorInvalidClusterSize` |

Eight is real, not just reported — but 8 is also the documented non-opt-in
ceiling and 16 is rejected at run time rather than at compile time. This puzzle
ships `CLUSTER = 4`.

## The runtime check that makes this puzzle honest

The failure mode §5 warns about is not a crash. It is a launch that is
**accepted** while the attribute is quietly ignored, leaving you with a grid of
ordinary blocks, `cluster.sync()` as an expensive no-op, and every
`map_shared_rank(p, r)` returning your own shared memory under another rank's
name. Nothing on the host side can detect that: the attribute lives in the
kernel's fatbin metadata, not in the launch call.

So the kernel reports it. `ClusterHistogram` has one extra output, `meta`, and
thread 0 of rank 0 writes `cluster.num_blocks()` into `meta[0]`. The runner
poisons it with `0xff` first (so an unimplemented kernel reads `-1`) and checks
it equals `CLUSTER` before it looks at any histogram bin. That check is the
`cluster_geometry` PASS line, and it is the only reason to trust the two below
it.

## Task

Complete two kernels in `skeletons/p34_clusters/kernel.cu`. Both carry
`__cluster_dims__(CLUSTER, 1, 1)` already — the attribute sits outside the
fill-in regions, along with the includes, the constants, the shared-memory
declarations and `cg::this_cluster()`.

```
n = 1000003 floats in [-1, 1), seed 34
TPB = 256, CLUSTER = 4, NBINS = 256, SLICE = NBINS / CLUSTER = 64
grid = 3907 blocks needed, rounded up to 3908 = 977 clusters of 4
```

### I/O contract

| kernel | output | meaning |
|---|---|---|
| `ClusterHistogram` | `hist[0..255]` | count of elements in each of 256 equal-width bins over `[-1, 1)`, built in DSMEM |
| `ClusterHistogram` | `meta[0]` | `cluster.num_blocks()`, written by thread 0 of rank 0 |
| `ClusterShift` | `out[0..n-1]` | `out[i] = a[i] + a[i+256]` when `a[i+256]` is staged by a block of the **same** cluster, otherwise `out[i] = a[i]` |

`hist` is an `atomicAdd` accumulator, so the runner zeroes it — zero is the
identity and the only correct initial value, exactly as in puzzle 27. `meta` and
`out` are poisoned with `0xff` instead: a value you forget to write is a `FAIL`,
not a lucky zero. That matters most for `out`, where "add nothing" is the
*correct* answer for the last rank of every cluster and for the tail of the
array — a kernel that skips those elements entirely would otherwise look right.

### The API you need

```cpp
cg::cluster_group cluster = cg::this_cluster();

cluster.block_rank()                  // 0 .. CLUSTER-1, this block's index in its cluster
cluster.num_blocks()                  // CLUSTER, as the hardware actually built it
cluster.map_shared_rank(ptr, rank)    // ptr, as seen in block `rank`'s shared memory
cluster.sync()                        // barrier across all blocks of the cluster
```

`map_shared_rank` is a template over the pointer type: hand it `bins` (an
`int[]` in shared memory) and it hands back an `int*` into another block's copy
of `bins`. It does not copy anything and it does not check anything — it is
address arithmetic (see the SASS section below). The pointer you pass must be a
shared-memory pointer of *this* block, and the rank must be a member of *this*
cluster.

### 1. `ClusterHistogram` (approx 14 lines)

Puzzle 27's histogram gave every block a private copy of all the bins, then
merged `gridDim.x` copies into global memory. Here the cluster holds **one**
copy of the bins, split across the shared memory of its four blocks:

```
rank 0 owns bins [0, 64)      rank 2 owns bins [128, 192)
rank 1 owns bins [64, 128)    rank 3 owns bins [192, 256)
```

A thread whose value lands in bin `b` does not get a private copy and does not
go to global memory. It finds the owner, maps that owner's `bins[]`, and
atomically increments the right slot inside it. The phases:

```
zero this block's own slice,  striding k = threadIdx.x, k += blockDim.x
rank 0's thread 0 publishes cluster.num_blocks() into meta[0]
cluster.sync()
each in-range thread: bin its value, find the owning rank, map, atomicAdd
cluster.sync()
flush this block's own slice to hist[] with atomicAdd, striding the same way
```

The bin index must be computed with the character-identical expression puzzle 27
and `ref_hist()` use, because float rounding has to agree on both sides:

```cpp
float v = a[i];
int b = (int)((v + 1.0f) * 0.5f * nbins);
b = min(max(b, 0), nbins - 1);
```

From `b`, the owning rank is `b / slice` and the offset inside that owner's
slice is `b - (b / slice) * slice`, where `slice = nbins / CLUSTER` is given.

Note what the final flush costs: one global atomic per bin per **cluster**, not
per block. 250,112 global atomics here against puzzle 27's 1,000,448 for the
same launch shape. Whether that is a good trade is measured at the bottom of
this file, and the answer is not the one the arithmetic suggests.

### 2. `ClusterShift` (approx 10 lines)

The same address space, read instead of written. Every block stages its own 256
elements into `tile[]`, then reads its **right-hand neighbour's** staged tile at
the same offset:

```
tile[threadIdx.x] = the element this thread owns, or 0.0f if out of range
cluster.sync()
if this block is not the last rank of its cluster:
    map the next rank's tile and read the element at threadIdx.x
in-range threads write out[i] = a[i] + that element (0 if there was none)
cluster.sync()
```

Two things fall out of this that are worth pausing on:

- **Nothing crosses a cluster boundary.** Rank 3 has no rank 4 to read, so
  `out[i] = a[i]` for the last quarter of every 1024-element span. That is not
  a limitation being worked around, it is what a cluster *is*: the guarantee of
  co-residency covers exactly these four blocks.
- **The tail needs no code.** For the last 67 elements, `a[i+256]` does not
  exist — but the block that would have staged it does exist (the grid was
  rounded up) and it staged `0.0f`. The out-of-range guard on the staging store
  is what makes the read side unconditional. This is the whole reason the excess
  blocks must run the staging phase instead of returning early.

### The grid must be a multiple of `CLUSTER`

`cdiv(1000003, 256) = 3907` blocks are needed and 3907 is not divisible by 4, so
the runner launches 3908. This is not a rounding convenience, it is a hard
launch requirement — measured, by launching the finished kernel on a grid that
violates it:

| grid | launch result |
|---|---|
| 3908 (977 × 4) | `cudaSuccess` |
| 3907 | `cudaErrorInvalidClusterSize` — "a kernel launch error has occurred due to cluster misconfiguration" |
| 4 | `cudaSuccess` |
| 6 | `cudaErrorInvalidClusterSize` |

The consequence is block 3907: a full block of 256 threads, none of which owns
an element. It has no work. It is still a member of its cluster, and it must
reach every `cluster.sync()` the other three reach.

### `if (i >= n) return;` is now a cluster-scale bug

Puzzle 24 taught that an early return before a full-mask shuffle breaks the
warp. Puzzle 27 taught the same thing about `__syncthreads()`. At cluster scope
the blast radius is a whole block: put an early return in front of a
`cluster.sync()` and the block leaves three peers waiting on a barrier one
participant will never reach — and, worse, deallocates shared memory those peers
may still be pointing at.

Measured, by adding exactly that line to the working solution — five
uninstrumented runs, five identical results:

```
CUDA error cudaErrorLaunchFailure ... unspecified launch failure
```

It does not hang on this hardware; it faults. That is a mercy, not a guarantee:
the same defect is documented elsewhere as a deadlock, and the difference is a
scheduling detail you do not control. Guard with a value, not a return — every
kernel here is written so that out-of-range threads compute nothing but still
arrive at every barrier.

## The three barrier placements, and what each one protects

Shared memory belongs to a block and dies with it. Once four blocks share one
address space, that single sentence generates three distinct obligations, and
`cluster.sync()` is the only instruction that can discharge any of them:

| barrier | protects against |
|---|---|
| after zeroing (histogram) | a peer incrementing a slice **before its owner zeroed it** — or before the owner is even resident |
| after scattering (histogram) | an owner flushing its slice to global memory **while peers are still incrementing it** |
| after remote reads (shift) | a peer **exiting** — and deallocating its shared memory — while this block is still reading it |

`__syncthreads()` cannot substitute for any of them. It synchronises the 256
threads of one block and knows nothing about the other 768.

Measured, by deleting one barrier at a time from the working solution and
running the unmodified runner:

| defect | `make run` | racecheck | memcheck |
|---|---|---|---|
| histogram: no `cluster.sync()` after zeroing | `cudaErrorLaunchFailure` (5/5 runs) | 750,038 errors | **0 errors, all three checks PASS** |
| histogram: no `cluster.sync()` before the flush | `cudaErrorLaunchFailure` | 390,656 errors | 1 error (the API failure itself) |
| shift: no trailing `cluster.sync()` | `cluster_shift` never completes, `cudaErrorLaunchFailure` | 293,100 errors | 1 error (the API failure itself) |
| histogram: `bins[]` instead of `map_shared_rank(bins, owner)` | `FAIL cluster_histogram at index 0: got 3909 want 3999` | 0 errors | 0 errors |

Two rows are worth reading twice.

**Row 1.** Removing the barrier after zeroing produces a hard launch failure on
every ordinary run — and under `memcheck` the same binary prints three `PASS`
lines and `ERROR SUMMARY: 0 errors`. Instrumentation serialises the machine
enough that the peer block is always resident and always zeroed before anyone
increments its slice. Only `racecheck` sees it, and it sees three quarters of a
million instances of it. A sanitizer that reports clean is evidence about the
run you did, not about the code.

**Row 4.** Replacing the DSMEM write with a plain local one — precisely what a
silently-downgraded cluster would compute — leaves both sanitizers perfectly
clean and every barrier correct. It is not a memory error and it is not a race.
It is simply the wrong answer, caught by comparing against a CPU reference and
by nothing else. This is why `cluster_geometry` is asserted separately from
`cluster_histogram`: one detects the hardware lying about the launch, the other
detects the kernel lying about where it wrote.

## What DSMEM compiles to

Measured on this box with `-Xptxas -v` and `cuobjdump -sass` on the built
solution:

| kernel | regs | static smem |
|---|---|---|
| `ClusterHistogram` | 28 | 256 B (64 bins × 4 B) |
| `ClusterShift` | 16 | 1024 B (256 floats) |

`cudaOccupancyMaxPotentialClusterSize` reports **8** for both, at 256 threads
per block — the runner prints it on every run, next to the cached probe values.

`cluster.sync()` is five instructions, not one:

```
MEMBAR.ALL.GPU      make this block's writes visible at GPU scope
ERRBAR              error barrier
CGAERRBAR           the same, cluster-wide (CGA = cooperative grid array = cluster)
UCGABAR_ARV         arrive at the cluster barrier
UCGABAR_WAIT        wait for the other blocks
```

Note the arrive/wait split in the last two: the barrier is built from the same
two-phase primitive `cg::cluster_group::barrier_arrive()` /
`barrier_wait()` exposes, which is what lets a kernel overlap independent work
between arriving and waiting.

`map_shared_rank` generates **no instruction of its own**. In `ClusterShift`:

```
CS2R.32 R4, SR_CgaSize      cluster size, from a special register
S2R     R5, SR_CgaCtaId     this block's rank, from a special register
LEA     R6, R5, R4, 0x18    peer address = base + (rank << 24)
```

The peer's shared memory is a window in the generic address space at a fixed
stride — the rank is literally shifted into bit 24 of the address. That is why
the mapping is free and also why it is unchecked: a bad rank produces a valid
address into nothing in particular.

What is *not* free is the access itself:

| operation | instruction |
|---|---|
| staging store to own shared memory | `STS` |
| **remote** read of a peer's tile | `LD.E` — a generic load, not `LDS` |
| local shared increment (puzzle 27) | `ATOMS.POPC.INC.32` — warp-aggregated |
| **remote** shared increment (this puzzle) | `ATOM.E.ADD.STRONG.GPU` — generic, GPU-scope, **no aggregation** |

The last row is the whole performance story of this puzzle. Puzzle 27 measured
that `atomicAdd(&bins[b], 1)` on local shared memory compiles to a single
warp-aggregated population count — the lanes of a warp landing in the same bin
are counted once and applied as one atomic. Route the same increment through
`map_shared_rank` and that optimisation is gone: it becomes a generic
GPU-scoped atomic, one per lane.

## The measurement that does not go the way the arithmetic does

Same input (`n = 1000003`, 256 bins, 3908 blocks), same bin formula, both
verified against the same CPU reference. Best of 12 reps of 50 iterations,
interleaved:

| variant | time | global atomics |
|---|---|---|
| puzzle 27 shape: private copy of all 256 bins per block | **0.037 ms** | 1,000,448 |
| this puzzle: one copy per cluster, split across 4 blocks in DSMEM | 0.239 ms | 250,112 |

The cluster version issues **4× fewer global atomics and runs 6.4× slower.**
Reproduced across three consecutive runs (0.2392 / 0.2393 / 0.2399 ms against
0.0369 / 0.0379 / 0.0377 ms).

Two reasons, both visible in the SASS table above. The increment lost hardware
warp aggregation, so 1,000,003 lane-level atomics are issued where puzzle 27
issued roughly one per warp per distinct bin. And each one is a remote,
GPU-scoped atomic crossing the GPC interconnect rather than an `ATOMS` into the
SM's own shared memory — while *four* blocks now contend for each 64-bin slice
instead of one block owning a private copy.

The global-atomic count, which is the number the technique is usually sold on,
was the wrong thing to optimise at this size.

So why learn it? Because DSMEM is not a faster histogram, it is a **larger
working set**. Four blocks' shared memory is one 4× bigger scratchpad with a
barrier across it: a tile that does not fit in 48 KB fits in 192 KB, a halo
exchange becomes a pointer instead of a global round trip, and a producer block
can hand a consumer block data without either touching L2. This puzzle uses a
histogram because the puzzle-27 version is right there to compare against, not
because the histogram wants a cluster. Measure before you reach for this.

## Race-freedom is mandatory here

`make check` must report zero from all three tools. The solution does — 0
memcheck errors, 0 racecheck hazards, 0 synccheck errors — and the defect table
above shows what each tool does and does not catch when it is broken. Note in
particular that racecheck understands DSMEM well enough to attribute hundreds of
thousands of hazards to it, and that memcheck alone would have signed off on a
kernel that fails every real run.

## Run

```
make run P=34                  # your kernel
make run P=34 MODE=solution    # reference
make check P=34                # memcheck + racecheck + synccheck, all must be 0
```

Expected: `PASS cluster_geometry`, `PASS cluster_histogram`, `PASS
cluster_shift`, above the info header that prints the occupancy API's answer for
both kernels and the cached probe results they have to agree with.
