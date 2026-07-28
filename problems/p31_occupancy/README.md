# Puzzle 31 — Occupancy: The Budget and the Bottleneck

Puzzle 30 was about a metric that says the opposite of the truth. This one is
about a metric that is *exactly correct* and still does not answer the question
you wanted to ask.

Occupancy is how many warps an SM is running at once, as a fraction of how many
it could. It is not a measurement — it is an **integer division of four fixed
budgets**, and CUDA will compute it for you from your compiled binary before
you launch anything. Half this puzzle is learning to do that. The other half is
finding out what it buys you, which on this box is: sometimes everything, and
sometimes nothing at all.

You get three kernels that compute the identical polynomial and differ only in
what they ask the SM for. The runner prints their occupancy table on every run,
correct or not, then times them. Two of the three have **exactly the same
occupancy** and are **1.3× apart**. The third has the lowest occupancy of the
three and the runner proves *why* with arithmetic. Your job is to explain both.

## The resource model

An SM is four budgets, handed out per block. A block gets resident on an SM
only if all four still have room, so the number of blocks that fit is the
**minimum** over four independent integer divisions. On this box (the runner
prints these from `cudaDeviceProp`, they are not universal):

| budget | this SM | blocks of 256 threads it allows |
|---|---|---|
| threads per SM | 1536 | `1536 / 256` = **6** |
| blocks per SM | 24 | **24** |
| registers per SM | 65536 | `65536 / (256 × regs)` |
| shared memory per SM | 102400 B | `102400 / (smem + 1024)` |

The `+1024` is real: the driver reserves 1 KiB of shared memory per block for
itself, so a block asking for 24 KiB actually costs 25 KiB of the budget. The
runner prints the reserved amount rather than assuming it.

Occupancy is then `blocks × 256 / 1536`. Note what this means: on this SM, six
blocks of 256 is already 100 %. A kernel using few enough registers and no
shared memory is *pinned* at 100 % and cannot be improved — which is worth
knowing before you spend a day trying.

## Reading what the build already tells you

Every `make` in this repo passes `-Xptxas -v`, so the register and
shared-memory numbers are already scrolling past you:

```
ptxas info    : Compiling entry function '_Z7SmemHogPKfPfi' for 'sm_121'
ptxas info    : Function properties for _Z7SmemHogPKfPfi
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 1 barriers, 24576 bytes smem
```

Three things to take from a block like that:

1. **`Used N registers`** — per thread. Multiply by the block size to get the
   block's register cost, and divide that into 65536.
2. **`bytes smem`** — the *static* `__shared__` cost of one block. Dynamic
   shared memory (the third `<<<>>>` argument) is not in this number and has to
   be added by hand; the occupancy API takes it as a parameter for that reason.
3. **`spill stores` / `spill loads`** — nonzero means ptxas ran out of
   registers and started spilling to local memory, which is off-chip. Spills
   are the failure mode that makes "just reduce registers to raise occupancy"
   backfire: you buy warps and pay for them with DRAM traffic. All three
   kernels here should show **zero**. Check.

## The API

Two calls, both cheap, both host-side, both exact:

```cpp
cudaFuncAttributes fa;
cudaFuncGetAttributes(&fa, MyKernel);          // fa.numRegs, fa.sharedSizeBytes
int blocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, MyKernel, tpb, dynSmem);
```

The second one does the four-way minimum for you, accounting for static shared
memory automatically. There is also `cudaOccupancyMaxPotentialBlockSize`, which
sweeps block sizes and hands you the one with the best occupancy — useful, and
also a trap, for the reason this puzzle exists.

The important property of both: they read the *binary*, not the clock. They
return the same answer under `compute-sanitizer`, under `ncu`, and on a machine
that is busy doing something else. The runner leans on that — the occupancy
check is asserted even on instrumented runs, while the timing check is not.

## `__launch_bounds__(maxThreads, minBlocksPerSM)`

A promise and a request, in that order:

- **`maxThreads`** — you promise never to launch this kernel with more threads
  per block than this. ptxas may assume it. Break the promise and the launch
  fails at runtime.
- **`minBlocksPerSM`** — you request that ptxas compile the kernel so at least
  this many blocks fit on an SM. The only lever it has is the register count,
  so it caps registers at `65536 / (maxThreads × minBlocksPerSM)` and spills if
  the kernel does not fit under that cap.

Two consequences worth internalising, both visible in this puzzle:

- The request can be **rejected**. `maxThreads × minBlocksPerSM` cannot exceed
  the SM's thread budget. Ask for too much and ptxas prints a warning, ignores
  the annotation entirely, and compiles as if you had not written it — your
  build succeeds, your binary is not what you asked for, and nothing at runtime
  will tell you. Try it: change `SmemHog`'s `6` to an `8` and read the build
  log.
- The request only controls **registers**. It has no authority over shared
  memory. A kernel can be granted its register budget and still be limited to
  fewer blocks by a `__shared__` array, and the annotation will not warn you,
  because from ptxas's point of view it did its job.

## The three kernels

`TPB = 256` for all of them, `n = 2000003` (a multiple of neither 32, nor 256,
nor 1024, so the tails are ragged and guard logic is exercised). All three
compute `out[i] = sum_k a[i]^k / (k+1)` for `k = 0..63`, evaluated by Horner:

    s = c_63;  s = s*x + c_62;  s = s*x + c_61;  ...  s = s*x + c_0

63 chained `fmaf()`s, each needing the previous one's result. The coefficient
helper, the `__launch_bounds__`, and the shared array are given to you above
the fill-in regions; you write the bodies. They live in
`skeletons/p31_occupancy/kernel.cu`.

| | | |
|---|---|---|
| `a` | `n` floats, read-only | |
| `out` | `n` floats, **every one written** | poisoned with `0xff` (a NaN pattern) before each launch, so an element you never write is a `FAIL`, not a lucky zero |
| tolerance | `1e-5` relative | against a double-accumulator Horner in `common/reference.hpp` |
| guards | **value guards, not early returns** | `(i < n) ? a[i] : 0.0f` |

### 1. `PolyOne` (approx 5 lines)

One element per thread, `cdiv(n, 256)` blocks. Load `a[i]`, run the 63-step
chain, store `out[i]`. The baseline, and the shape you would write without
thinking about it.

### 2. `PolyFour` (approx 16 lines)

Four elements per thread, and the runner sizes the grid at `cdiv(n, 4)` threads
— a quarter of `PolyOne`'s — so the two kernels do **exactly the same total
arithmetic**. Stride the four elements by the whole grid (`i`, `i + stride`,
`i + 2*stride`, `i + 3*stride`, where `stride = gridDim.x * blockDim.x`) so
each warp's loads stay as coalesced as `PolyOne`'s.

Write it as four chains that do not depend on each other: four inputs, four
accumulators, all live across the loop. Guard each of the four indices — the
last thread runs off the end of `a` on three of its four elements.

### 3. `SmemHog` (approx 7 lines)

`PolyOne`'s arithmetic exactly, one element per thread, same grid — but staged
through the 24 KiB `__shared__` array declared for you. Write your slot,
`__syncthreads()`, read it back, evaluate, store. The array is 24 times larger
than the 1 KiB a block actually touches; it is there to be a budget, not a
buffer.

One warning the runner will enforce: a shared array that nothing **writes** is
dead code, and ptxas deletes it. If your `smem B` column reads `0`, that is not
the harness being generous — it is telling you the truth about your binary.

## Run

```
make run   P=31                  # your kernel — fails loudly
make run   P=31 MODE=solution    # reference
make check P=31 MODE=solution    # memcheck + racecheck + synccheck, all zero
make prof  P=31 MODE=solution    # Occupancy + SpeedOfLight + WarpStateStats
```

Expected from `make run`: the occupancy table, three `PASS` lines, `PASS
smem_limits_occupancy`, a timing table, the paired ratio lines, and `PASS
occupancy_not_predictive`.

The occupancy table prints **even when your kernels are wrong** — it is the
instrument, not a diagnostic. The two relationship checks are guarded
differently on purpose: `smem_limits_occupancy` is integer arithmetic over the
compiled binary and is asserted on every run including instrumented ones,
while the timing check is skipped when `P31_SKIP_TIMING` is set (see
`problems/p31_occupancy/prof.mk`, which sets it for the sanitizer runs). Under
`compute-sanitizer` or `ncu` a wall clock measures the tool.

### How the timing number is formed, and why it is not a minimum

Worth reading before you trust — or argue with — the ratio, because the
estimator is doing real work:

- All three kernels are timed **inside the same rep**, a few hundred
  microseconds apart, and the ratio is formed *within* the rep. This box is a
  shared-memory SoC — one LPDDR5X behind the GPU and the CPUs — so an
  unrelated process slows both kernels of a pair at once, and a ratio formed
  from adjacent samples is largely indifferent to that. A ratio formed from two
  minima drawn from *different* machine states is not: that estimator inverted
  on this box, reporting `PolyFour` slower than `PolyOne`, on runs where the
  paired median read 1.20×.
- The 201 reps are ranked by their three-kernel total and the **quietest 21**
  are kept; the estimate is the median of those. Contention only ever makes a
  measurement slower, so the cheapest reps are the least contaminated ones. The
  ranking is on the rep *total*, so it cannot flatter one kernel — a kernel
  that is uniformly slow is slow in every rep.
- The SM clock is measured **by the GPU**, with one warp spinning on
  `clock64()`. `nvidia-smi` reports 208 MHz on this box while the device is
  actually at ~2540 MHz, so it is not usable as a check on whether your
  measurement was taken at a sane clock.
- Before it asserts anything, the runner establishes that it **got a
  measurement**. Two checks, deliberately different in kind: it asks the driver
  (through NVML) which other processes held a CUDA context on this GPU while the
  section was being timed, and it holds `PolyFour`'s quiet window to an absolute
  throughput floor. The first is causal and names the competing PIDs; the second
  is the only one that sees a CPU process on the other side of this SoC's
  LPDDR5X, which holds no CUDA context and appears in no GPU-side accounting.
- That gives the timing section three outcomes and no silent one. A valid run
  whose ratio clears `MARGIN` prints `PASS occupancy_not_predictive`. A valid
  run whose ratio does not prints `FAIL occupancy_not_predictive` — the claim
  did not hold on a quiet machine, and the message says which kernel to look
  at. A run that was **not a measurement** prints `FAIL measurement_invalid`
  with the evidence — the competing PIDs, or the throughput against the floor —
  and exits non-zero without evaluating the assert at all.
- The third outcome is a `FAIL` and not a `SKIP`, and that is the point. Under
  enough external load every effect in this puzzle genuinely disappears, which
  is a fact about your box and not about your kernel — but a run that proves
  nothing is not a run that passed, and a green suite has to mean the asserts
  were evaluated. So `measurement_invalid` never says your kernel is wrong, and
  it never exits 0 either. Re-run on an idle box.
- Correctness and `smem_limits_occupancy` sit outside all of this. They are not
  wall-clock measurements — one is arithmetic on your outputs, the other is
  arithmetic on your binary — so they run on every run, including instrumented
  ones, and they can always fail.

> `ncu` needs permission to read the GPU's performance counters, which on this
> box is admin-only (`RmProfilingAdminOnly: 1`). If `make prof` reports
> `ERR_NVGPUCTRPERM`, run the same command under `sudo` — there is a sudoers
> rule for exactly `/usr/local/cuda/bin/ncu`.

## The puzzle

Predict the occupancy table from the `-Xptxas -v` output **before** you run it.
Three kernels, four budgets each, twelve integer divisions, one minimum apiece.
Then run it and check yourself. If you got all three, you understand occupancy
completely — and you are about to find out that this is not the same thing as
understanding the timing table.

Now `make run P=31 MODE=solution` and sit with these:

1. `SmemHog` runs at two thirds of `PolyOne`'s occupancy and is slower by about
   the same proportion. Which of the four budgets is the one that binds it, and
   what is the exact division? Confirm it against `Block Limit Shared Mem` in
   `make prof`. Then ask the harder half: **is `SmemHog` slower *because* its
   occupancy is lower**, or because of the shared-memory round trip and the
   barrier it added along the way? Those are different claims and the timing
   table alone cannot separate them. Design the control that can. (You are
   allowed to write a fourth kernel to do it.)

2. `PolyOne` and `PolyFour` do **the same total arithmetic** and run at
   **exactly the same occupancy — 100 %, both of them, pinned at the ceiling**.
   `PolyFour` uses 2.5× the registers and it costs it nothing, because the
   register budget was never the binding one. And it is ~1.3× faster per
   element. Occupancy is identical, so occupancy explains *nothing* about that
   gap. **What does?**

   Write down your hypothesis before you profile. Then go and try to kill it,
   because the obvious one is wrong on this hardware and the profiler will say
   so if you ask it the right question. Three places to look:

   - `--section WarpStateStats` reports what warps are stalled *on*, per issued
     instruction. `Stall Wait` is a warp waiting on a fixed-latency arithmetic
     dependency — exactly what a 63-deep FMA chain produces. `Stall Long
     Scoreboard` is a warp waiting for a memory operand. Get both numbers for
     both kernels. One of them barely moves between the two kernels; that is
     your falsification.
   - `cuobjdump -sass` on the built binary. Count the `FFMA`s between one
     `LDG.E` and the next. Whatever you assumed the compiler did with your four
     independent chains, check it.
   - A control: `PolyFour` differs from `PolyOne` in **three** ways at once —
     work per thread, threads per grid, and blocks per grid. Hold two of them
     fixed and vary the third. Only one of the three is responsible.

3. And the regime question, the one puzzle 30 ended on too. The working set
   here is 16 MB against a 25 MB L2, chosen deliberately. Predict what happens
   to the `PolyOne / PolyFour` ratio as `n` grows past the L2 — not just
   whether the gap shrinks, but whether it can *invert*, and why the quarter-
   sized grid would stop being an advantage there.

When you have answers — including a mechanism you have actively tried to
falsify — check them against `solutions/p31_occupancy/SOLUTION.md`, which
carries every number measured on this box, the sweep behind the choice of `n`,
and the three hypotheses that had to die before the right one was left.
