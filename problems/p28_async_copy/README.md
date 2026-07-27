# Puzzle 28 — Async Copy and Copy Overlap

This is the first timing puzzle in the series. Puzzles 24–27 asked only whether
your kernel got the right answer; this one also asks how fast it went, and the
honest answer is going to be more interesting than the one you expect.

Every tiled kernel you have written so far has the same shape:

```
load a tile from global into shared
__syncthreads()
compute from shared
__syncthreads()
load the next tile
```

Look at what the *load* line actually costs on the hardware. `buf[j] = a[base+j]`
is two instructions, not one: an `LDG` that pulls a word from memory **into a
register**, and an `STS` that pushes that register **into shared memory**. The
thread issuing them cannot retire the `STS` until the `LDG` has returned, so the
thread stalls on the full memory latency, holding a register the whole time,
purely as a courier between two memories it does not care about.

Ampere added an instruction that removes the courier: `LDGSTS` — global to
shared, one instruction, no register, and *asynchronous*: the thread issues it
and keeps going. You do not write `LDGSTS` directly; you write
`__pipeline_memcpy_async` and the compiler emits it.

Asynchronous means you now need a way to ask "has it landed yet?", and that is
what the rest of the API is for:

| you write | SASS you get | means |
|---|---|---|
| `__pipeline_memcpy_async(dst, src, 4)` | `LDGSTS.E` | issue one global→shared copy |
| `__pipeline_commit()` | `LDGDEPBAR` | close the current *stage*: everything issued since the last commit is now one group |
| `__pipeline_wait_prior(N)` | `DEPBAR.LE SB0, N` | block until at most `N` of my committed stages are still in flight |

Those three (plus `#include <cuda_pipeline.h>`) are the whole interface, and
they are per-thread: a stage belongs to the thread that issued it, and
`__pipeline_wait_prior` only tells *that* thread its own copies landed. Any
thread reading a word another thread copied still needs `__syncthreads()`.
That is the single most important sentence in this puzzle.

Once loads no longer block, you can have two tiles alive at once — the point of
the whole exercise:

```
issue copy of tile 0 -> buffer A ; commit
loop t:
    issue copy of tile t+1 -> buffer B ; commit      <- starts moving now
    wait until only 1 stage is in flight             <- tile t has landed
    __syncthreads()
    compute tile t from buffer A                     <- overlaps tile t+1's transfer
    __syncthreads()
    swap A and B
```

## Task

Complete two kernels in `skeletons/p28_async_copy/kernel.cu`. They compute the
same thing and must produce the same output; only the load path differs.

### Problem

A sliding-window sum with `WINDOW = 16`:

    out[i] = a[i] + a[i+1] + ... + a[i+15],   for i in [0, n)

### I/O contract

| | |
|---|---|
| `a` | `n + WINDOW - 1` floats. The trailing 15 are **halo**: inputs the last windows reach into, not outputs of their own. |
| `out` | `n` floats, every one of them written. The runner poisons `out` with `0xff` (a NaN pattern) before each launch, so an element you never write is a `FAIL`, not a lucky zero. |
| `n` | `1000003` — not a multiple of 32, of `TPB`, or of `WINDOW`. |
| grid | `192` blocks (4 per SM on this box's 48 SMs) × `256` threads, **fixed**. |

The grid is fixed and `n` is not, so both kernels **grid-stride over tiles**:
block `b` handles tiles `b`, `b + gridDim.x`, `b + 2*gridDim.x`, … The tile loop
header is already written for you.

`TILE = TPB = 256` outputs per tile, and therefore `TILE_IN = TILE + WINDOW - 1
= 271` inputs per tile — a tile's last thread reads 15 elements past the tile's
last output. All the constants are given at the top of the file.

### 1. `WindowSumSync` (approx 12 lines)

The baseline, inside the given tile loop: cooperatively load `TILE_IN` floats
into `buf[]` with ordinary loads, barrier, sum a 16-element window out of shared
memory, write `out[i]`, barrier before the next tile overwrites the buffer.

Two guards to get right, both of which are value guards, not early returns —
every thread has to reach both barriers:

- **The store guard.** `3907 * 256 = 1000192` outputs exist in tile space but
  only `1000003` are real. Thread `i >= n` computes nothing and stores nothing,
  but still hits both `__syncthreads()`.
- **The load guard.** The final tile only has `1000018 - 3906*256 = 82` inputs
  left, not 271. Loading a fixed 271 reads past the end of `a`. Clamp the count.

That second one is the whole reason `n` was chosen as it was, and it is worth
being precise about *why* clamping is sufficient: if a tile has `cnt` inputs
left, then its last real output is at local index `cnt - WINDOW`, so every
window a surviving thread reads lies entirely inside `[0, cnt)`. The unloaded
slots of `buf[]` are never read. You do not need to zero them, and you do not
need a special case for the last tile — you need one `min`.

### 2. `WindowSumAsync` (approx 25 lines)

The same math, from a two-stage double buffer. `__shared__ float buf[2][TILE_IN]`
is declared for you; everything else is yours.

Structure it as: a prologue that stages the block's first tile into buffer 0,
then a loop that — for tile `t` in buffer `p` — stages tile `t + gridDim.x` into
buffer `p^1`, waits for `t`, barriers, computes `t`, barriers, and flips `p`.

Four things this puzzle is actually testing:

1. **Every thread must `__pipeline_commit()` the same number of times.** In the
   ragged tile most threads have no word to copy. A thread that skips the commit
   because it had nothing to issue is one stage behind its neighbours forever,
   and every subsequent `__pipeline_wait_prior(1)` in that thread waits for the
   wrong stage. Commit an empty stage rather than skipping it. (Branching on
   `blockIdx`/`gridDim` is fine: those are uniform across the block. Branching
   on `threadIdx` is not.)
2. **`__pipeline_wait_prior(1)` after issuing the prefetch, not before.** If you
   wait first you have written the sync kernel with extra steps.
3. **`__syncthreads()` after the wait.** Your wait covers your copies. Thread 5
   is about to read `buf[p][5..20]`, most of which threads 6–20 copied.
4. **`__syncthreads()` at the end of the loop body.** Buffer `p` gets refilled
   two iterations later; that barrier is what stands between "thread 200 is
   still reading `buf[0]`" and "thread 3 has already issued an `LDGSTS` into
   `buf[0]`". It is a write-after-read hazard and it is invisible in the output
   — see the table further down.

## Run

```
make run   P=28                  # your kernel — fails loudly
make run   P=28 MODE=solution    # reference
make check P=28 MODE=solution    # memcheck + racecheck + synccheck, all must be 0
make prof  P=28 MODE=solution    # LDGSTS counts + SpeedOfLight
```

Expected: two `PASS` lines, then a timing table. The runner verifies both
kernels before it times anything and skips timing entirely if either is wrong;
a fast wrong kernel is not a result.

`make check` runs the timing loop under the sanitizers too. That is fine — the
whole thing takes about 24 s on this box.

## What the compiler actually emitted

`cuobjdump -sass` on the built solution, GB10 / `sm_121` / CUDA 13.0. The load
path, side by side:

```
WindowSumSync                          WindowSumAsync
  LDG.E  R13, desc[UR6][R12.64]          LDGSTS.E [R19],       desc[UR4][R10.64]
  LDG.E  R15, desc[UR6][R14.64]          LDGSTS.E [R19+0x400], desc[UR4][R12.64]
  LDG.E  R17, desc[UR6][R16.64]          LDGSTS.E [R19+0x800], desc[UR4][R14.64]
  LDG.E  R19, desc[UR6][R18.64]          LDGSTS.E [R19+0xc00], desc[UR4][R16.64]
  STS    [R10],       R13                LDGDEPBAR
  STS    [R10+0x400], R15                ...
  STS    [R10+0x800], R17                DEPBAR.LE SB0, 0x1
  STS    [R10+0xc00], R19                DEPBAR.LE SB0, 0x0
  BAR.SYNC.DEFER_BLOCKING 0x0            BAR.SYNC.DEFER_BLOCKING 0x0
```

Two instructions and a live register per element on the left; one instruction
and no register on the right. Both loops got unrolled ×4 by the compiler, which
is why you see four of each. `LDGDEPBAR` is the commit, and the two `DEPBAR.LE
SB0` are the two `__pipeline_wait_prior` calls — `0x1` for the steady state,
`0x0` for the drain on the last tile.

`-Xptxas -v` on the same build:

| kernel | registers | static shared |
|---|---|---|
| `WindowSumSync` | 30 | 1084 B (271 floats) |
| `WindowSumAsync` | 32 | 2168 B (2 × 271 floats) |

The async version costs 2 more registers and exactly double the shared memory.
That is the price of the technique and you pay it whether or not it helps.

`make prof P=28 MODE=solution` confirms the path was taken at all:

```
WindowSumSync                                                            WindowSumAsync
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum        0        41164
  smsp__inst_executed_op_ldgsts.sum                          0 inst   39063 inst
```

Zero against 39 063. `39063 = 3906 * 10 + 3`: ten `LDGSTS` warp-instructions
for each of the 3906 full tiles (eight warps × the unrolled strided loop) and
three for the 82-element ragged tail.

> `ncu` needs permission to read the GPU's performance counters, which on this
> box is admin-only (`RmProfilingAdminOnly: 1`). If `make prof` reports
> `ERR_NVGPUCTRPERM`, run the same command under `sudo` — there is a sudoers
> rule for exactly `/usr/local/cuda/bin/ncu`.

## What it actually cost

Measured on this box. Runner output, three consecutive runs:

```
# timing: 5 warmup + 50 timed iterations, 8.00 MB of global traffic per iteration
# kernel                avg ms   eff GB/s
  window_sum_sync       0.0166      480.8
  window_sum_async      0.0164      487.8
# sync/async time ratio: 1.014x
```

`1.014x`, `1.007x`, `1.039x`. **That is nothing.** The double buffer, the
pipeline stages, the extra kilobyte of shared memory, the extra registers, the
race-hazard surface — for between zero and four percent, inside run-to-run
noise.

This is the puzzle. Do not skip to the next one; work out why.

### First clue: the bandwidth number is impossible

GB10 has about 273 GB/s of memory bandwidth. The table says 487. Effective
bandwidth above the DRAM roofline means the traffic is not reaching DRAM: `a`
plus `out` is 8 MB, and this GPU has a **24 MB L2**. After the first iteration
the entire working set is cache-resident and every "load" is an L2 hit.

So push the problem out of L2 (same kernels, same fixed 192-block grid, `n`
swept, best of both launch orders):

| n | working set | sync | async | sync GB/s | async GB/s | ratio |
|---|---|---|---|---|---|---|
| 1 000 003 | 8 MB | 0.0164 ms | 0.0164 ms | 487 | 488 | 1.003 |
| 2 000 003 | 16 MB | 0.0310 ms | 0.0307 ms | 516 | 521 | 1.008 |
| 4 000 003 | 32 MB | 0.1034 ms | 0.0987 ms | 310 | 324 | 1.048 |
| 8 000 003 | 64 MB | 0.2711 ms | 0.2711 ms | 236 | 236 | 1.000 |
| 16 000 003 | 128 MB | 0.5514 ms | 0.5529 ms | 232 | 232 | 0.997 |
| 32 000 003 | 256 MB | 1.1078 ms | 1.1209 ms | 231 | 228 | 0.988 |

Out of L2 the effective bandwidth collapses to a flat **232 GB/s — 85% of this
part's 273 GB/s spec** — and stays there, and the async version is still worth
nothing. Both kernels are pinned against the memory system, in cache and out of
it. You cannot prefetch your way past a bus that is already full.

### Second clue: it depends entirely on occupancy

`__pipeline_memcpy_async` hides latency. But hiding latency is what the warp
scheduler already does for free, by switching to another warp. At 192 blocks on
48 SMs this kernel runs 4 blocks = 32 warps per SM; while one warp waits on a
load, the SM has 31 others to issue from, and the memory pipe never goes idle.
There is no bubble left for a prefetch to fill.

So take the warps away. Same kernels, same code, only the block count changes:

| blocks | blocks/SM | 8 MB working set (in L2) | | 128 MB working set (DRAM) | |
|---|---|---|---|---|---|
| | | sync → async | ratio | sync → async | ratio |
| 48 | 1 | 0.0307 → 0.0246 ms | **1.25×** | 0.9071 → 0.6386 ms | **1.42×** |
| 96 | 2 | 0.0205 → 0.0185 ms | 1.11× | 0.6307 → 0.5467 ms | 1.15× |
| 192 | 4 | 0.0165 → 0.0164 ms | 1.00× | 0.5475 → 0.5570 ms | 0.98× |
| 384 | 8 | 0.0164 → 0.0164 ms | 1.00× | 0.5825 → 0.5747 ms | 1.01× |
| 768 | 16 | 0.0162 → 0.0163 ms | 0.99× | 0.5642 → 0.5704 ms | 0.99× |

There it is. **At one block per SM the async version is 1.25× to 1.42× faster.**
By four blocks per SM the advantage is gone, and it never comes back.

Async copy and occupancy are two solutions to the same problem, and they do not
add up. The technique earns its keep exactly where occupancy cannot be raised:

- tiles so large that shared memory limits you to one or two blocks per SM
  (which is precisely the tiled-matmul regime the instruction was designed for);
- kernels with enough register pressure that more resident warps would spill;
- an inner loop with real arithmetic in it, so there is compute to overlap with
  rather than a 16-element add;
- and the register-pressure relief itself, which is a second, independent
  benefit — the sync version burns a live register per element in flight, which
  is exactly what caps occupancy in the kernels that need this most.

The corollary is the part worth carrying forward: this kernel, at this
occupancy, on this part, was **never latency-bound in the first place**. It sits
at 85% of DRAM spec with ordinary loads. Applying a latency-hiding optimisation
to a bandwidth-bound kernel buys you a race-condition surface and 4% of noise.
Measure which of the two you have before you pick the tool — that is what
`SpeedOfLight` in `make prof` is for.

### A warning about the profiler

`ncu`'s default settings lock the GPU to base clocks and serialise every launch.
Under those settings this box reports `WindowSumSync` 28.5 µs against
`WindowSumAsync` 22.9 µs — a 1.24× win that **does not exist** in the runner's
own event timing (1.01×), and that shrinks to 0.96× if you profile the same two
launches with `--clock-control none`. The profiler is authoritative about
*counters* — LDGSTS was issued 39 063 times, that is a fact. It is not
authoritative about *wall time*. Take timings from the timing harness and
counters from `ncu`, and never the other way round.

## Race-freedom is part of the answer here

A double buffer is a shared-memory buffer that one thread reads while another
thread is scheduled to overwrite it. That is the exact shape `racecheck` exists
for, and two of the three ways to get it wrong still print `PASS`. Measured, by
breaking the solution three ways:

| defect | `make run` | memcheck | racecheck |
|---|---|---|---|
| trailing `__syncthreads()` removed — the next tile's `LDGSTS` can land in a buffer still being read | **PASS** | clean | 3 hazards (2 errors) |
| copy count not clamped to the ragged tail | **PASS** | 65 × `Invalid __global__ read of size 4 bytes` | clean |
| `__syncthreads()` after `__pipeline_wait_prior` removed | `FAIL … at index 81` | clean | 5 hazards (5 errors) |

Read the first two rows again. Both produce correct output on this input, on
this hardware, on every run — and one of them reads past the end of a device
allocation on every single launch while the other has a live write-after-read
race on shared memory. Only the sanitizers disagreed, and *different* sanitizers
caught them: `racecheck` was silent on the out-of-bounds read, `memcheck` was
silent on the race.

The third row is the useful one to see fail properly. Dropping the barrier after
the wait makes each thread trust only its own copies; the first wrong element is
at index 81, in the middle of the first tile, because a thread got to
`buf[p][81..96]` before the threads that copy 82…96 had theirs land. The failure
index is not near a tile edge and not near `n` — it is wherever the schedule
happened to lose the race that run.

A puzzle 28 solution is done when it passes `make run` **and** all three
sanitizers report zero.

## The other spelling

There is a second, higher-level API for the same hardware:
`cuda::memcpy_async` with `cuda::barrier<cuda::thread_scope_block>` from
`<cuda/barrier>`, which folds the commit and the wait into the barrier's
arrive/wait protocol and can copy a whole `cuda::aligned_size_t<16>` chunk per
call. It compiles to the same `LDGSTS`. This puzzle uses the primitive spelling
deliberately, because the primitives make the stage accounting visible — and
the stage accounting is the part you can get wrong. Puzzle 29 picks up the
barrier side of that API.
