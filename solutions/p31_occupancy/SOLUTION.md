# Puzzle 31 — Solution: The Budget and the Bottleneck

Every number below was measured on this box: NVIDIA GB10, `sm_121`, 48 SMs,
1536 threads/SM, 24 blocks/SM, 65536 registers/SM, 102400 B shared/SM (1024 B
per block reserved by the driver), 25.17 MB L2, CUDA 13.0, driver 580.159.03.
Nothing here is quoted from anywhere.

This puzzle has two halves and they point in opposite directions, which is the
point. Occupancy is computed exactly and it explains `SmemHog` completely. The
same number explains nothing whatever about `PolyOne` versus `PolyFour`.

## 1. What the compiler produced

`-Xptxas -v`, from the build the runner uses:

| | `PolyOne` | `PolyFour` | `SmemHog` |
|---|---|---|---|
| registers/thread | 8 | **20** | 8 |
| static shared/block | 0 B | 0 B | **24576 B** |
| barriers | 0 | 0 | 1 |
| **stack frame** | **0 B** | **0 B** | **0 B** |
| **spill stores / loads** | **0 / 0** | **0 / 0** | **0 / 0** |

**No kernel spills.** That matters because register spilling is the trap that
makes naive occupancy-chasing backfire, and it would have contaminated every
comparison below. `PolyFour` holds four inputs and four accumulators live
across a 63-iteration unrolled loop for 20 registers, and ptxas fits it
comfortably.

## 2. The occupancy table, and its arithmetic

From `cudaFuncGetAttributes` + `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
at 256 threads/block, printed by the runner on every run:

```
# kernel       regs      smem B  blocks/SM   warps/SM  occupancy
  PolyOne         8           0          6         48     100.0%
  PolyFour       20           0          6         48     100.0%
  SmemHog         8       24576          4         32      66.7%
```

Every one of those is the minimum of four integer divisions, and all twelve are
reproducible by hand:

| budget | `PolyOne` | `PolyFour` | `SmemHog` |
|---|---|---|---|
| threads: `1536 / 256` | 6 | 6 | 6 |
| blocks: hardware cap | 24 | 24 | 24 |
| registers: `65536 / (256 × regs)` | `/2048` = 32 | `/5120` = 12 | 32 |
| shared: `102400 / (smem + 1024)` | ∞ | ∞ | `102400/25600` = **4** |
| **minimum** | **6** | **6** | **4** |

`ncu --section Occupancy` computes the same thing from the other side and
agrees exactly, including which budget wins:

| `Block Limit …` | `PolyOne` | `PolyFour` | `SmemHog` |
|---|---|---|---|
| `Warps` | **6** | **6** | 6 |
| `Registers` | 16 | 10 | 16 |
| `Shared Mem` | 16 | 16 | **4** |
| `Barriers` | 24 | 24 | 24 |
| `SM` | 24 | 24 | 24 |
| `Theoretical Occupancy` | 100 % | 100 % | **66.67 %** |

with ncu stating the conclusion outright for `SmemHog`: *"This kernel's
theoretical occupancy (66.7%) is limited by the required amount of shared
memory."* (ncu's `Block Limit Registers` differs from the hand calculation
because it quantises registers to allocation granularity — 16 and 10 against
32 and 12 — which changes none of the minima.)

Three things worth extracting from that table:

- **`PolyOne` is pinned.** Six blocks of 256 is 1536 threads is 100 %. There is
  no register count and no shared-memory budget that would make it *better*.
  Occupancy optimisation on this kernel has nothing to sell you.
- **`PolyFour`'s 2.5× register footprint is free.** 20 registers × 256 threads
  × 6 blocks = 30720, less than half the 65536 budget. The register budget was
  never the binding constraint, so spending it cost nothing. This is the
  general shape of the thing: a resource you are not short of is not a cost.
- **`SmemHog`'s 24 KiB is what binds.** `24576 + 1024 = 25600`, and
  `102400 / 25600 = 4` exactly. Not 12 KiB, which was the first size tried:
  `12288 + 1024 = 13312`, `102400 / 13312 = 7`, and 7 is *more* than the 6 the
  thread budget already allows, so a 12 KiB array would have changed the
  occupancy by nothing at all and the assert would have failed. The array has
  to beat 6 to be visible, and the smallest power-of-two that does is 24 KiB.

### The `__launch_bounds__` interaction

`SmemHog` is declared `__launch_bounds__(256, 6)`: never more than 256 threads
per block, and please fit 6 blocks per SM. Both halves are instructive here
because **neither one ends up mattering**, for two different reasons.

The request is *satisfiable*: `6 × 256 = 1536` is exactly this SM's thread
budget. ptxas accepts it and caps registers at `65536 / 1536 = 42`. The kernel
uses 8. The cap does nothing.

And 6 is the largest number this hardware will take. Measured, same kernel,
only `minBlocksPerSM` changed:

| `minBlocksPerSM` | ptxas response | registers | smem |
|---|---|---|---|
| 4 | accepted, silent | 8 | 24576 |
| 6 | accepted, silent | 8 | 24576 |
| 7 | `ptxas warning : Value of threads per SM for entry _Z3HogPKfPfi is out of range. .minnctapersm will be ignored` | 8 | 24576 |
| 8 | same warning | 8 | 24576 |

`7 × 256 = 1792 > 1536`, so the request is unsatisfiable and ptxas discards the
**entire annotation** rather than clamping it. The build succeeds. The binary
is not what was asked for. Nothing at runtime reports this — only the build
log. (This is why the puzzle ships `6` and not the `8` it was originally
specced with: on `sm_121`, `__launch_bounds__(256, 8)` is a no-op that looks
like a directive.)

Then the part that actually decides the outcome: ptxas's only lever is the
register count. It has no authority over shared memory. So the annotation was
granted in full, and the kernel still gets 4 blocks per SM instead of the 6 it
asked for, because a completely different budget ran out. **`__launch_bounds__`
is a request about registers wearing the costume of a request about
occupancy.** The occupancy API is what tells you which budget actually
answered.

## 3. What the runner measured

```
# timing: fastest of 15 interleaved reps of (3 warmup + 20 timed iterations)
# kernel       best ms   ns/element   occupancy
  PolyOne       0.0184      0.00921      100.0%
  PolyFour      0.0143      0.00716      100.0%
  SmemHog       0.0224      0.01121       66.7%
PASS smem_limits_occupancy (SmemHog 4 blocks/SM < PolyOne 6, 66.7% vs 100.0%)
PASS occupancy_not_predictive (poly_one / poly_four = 1.286x >= 1.05x, at 100.0% vs 100.0% occupancy)
```

Two rows of that table have identical occupancy and are 1.29× apart. One row
has lower occupancy and is 1.22× slower. Both facts are real. Sections 4 and 5
take them one at a time.

## 4. `SmemHog`: occupancy explains it, and here is the control that proves it

`SmemHog` runs at 66.7 % occupancy and takes 1.22× as long as `PolyOne` for
identical arithmetic. Tempting to call that causal and move on — but `SmemHog`
changed *two* things at once. It lost occupancy, and it also added a shared
store, a barrier, and a shared load. Either could be the cost.

The control: a fourth kernel, identical staging work — same store, same
`__syncthreads()`, same load — with a **1 KiB** array instead of 24 KiB, so it
keeps 100 % occupancy. If the staging is what costs, it will be slow. If the
occupancy is what costs, it will be free. Measured, all four in one interleaved
process, best of 15 reps, twice:

| variant | regs | smem | blocks/SM | occupancy | best ms | vs `PolyOne` |
|---|---|---|---|---|---|---|
| `PolyOne` (no staging) | 8 | 0 | 6 | 100 % | 0.0184 | 1.000× |
| **1 KiB + barrier** (control) | 8 | 1024 | **6** | **100 %** | **0.0184** | **1.000×** |
| 24 KiB, no barrier | 10 | 24576 | 4 | 66.7 % | 0.0205 | 1.114× |
| `SmemHog`: 24 KiB + barrier | 8 | 24576 | 4 | 66.7 % | 0.0222 | 1.206× |

**The shared-memory round trip and the barrier are free — 1.000×, twice, to
four digits.** Staging every input through shared memory, with a full block
barrier, at 100 % occupancy, costs nothing measurable. The entire 1.21 % → 21 %
penalty appears when, and only when, the array grows large enough to cost
blocks. `SmemHog` is slow *because* its occupancy is lower, and the control is
what licenses that "because".

Which is the honest case *for* occupancy, and it is worth being precise about
why it applies here: `PolyOne`'s warps spend most of their life stalled on a
global load (section 5), so throughput is set by how many warps the SM has
available to overlap. Cut the resident warps from 48 to 32 and you cut the
latency-hiding capacity by a third — 48/32 = 1.5 against a measured 1.21,
so hiding was not the *only* thing going on, but it was most of it.

That is the correct statement of what occupancy buys: **it is a latency-hiding
budget.** It matters exactly as much as you are relying on other warps to fill
your stalls, and not at all beyond that.

(One measurement in that table not to lean on: the no-barrier variant at
1.114×. It also compiles to 10 registers instead of 8, so it is not a clean
one-variable change, and across runs it moved between 1.11× and 1.19×. The
1 KiB control is the clean one and it is the one the argument rests on.)

## 5. `PolyOne` vs `PolyFour`: three hypotheses, two of them dead

Same arithmetic, same 100 % occupancy, 1.29× apart. ncu confirms the work is
genuinely identical: `sm__inst_executed_pipe_fma.sum` is 4 187 573 for
`PolyOne` against 4 156 357 for `PolyFour`, a 0.75 % difference on 4.2 million
FMA pipe instructions.

### Hypothesis 1 — ILP: "four independent FMA chains interleave in the pipeline"

The obvious story, and the one this puzzle was originally specified around. A
63-deep dependent FMA chain can issue one instruction every ~4 cycles;
`PolyFour` has four such chains per thread with no dependency between them, so
the compiler can interleave them and fill the gaps.

It is wrong on this hardware, and there are three independent ways to see it.

**Falsification A — the SASS.** `cuobjdump -sass` on the shipped binary,
`PolyFour`'s instruction order:

```
LDG.E ×4   →   63 FFMA → STG.E   →   63 FFMA → STG.E   →   63 FFMA → STG.E   →   63 FFMA → STG.E
```

ptxas hoisted the four loads, and then **serialised the four chains anyway**.
It did not interleave them. Whatever the source says about independence, the
machine is running the same one-chain-at-a-time code `PolyOne` runs — four
times per thread.

**Falsification B — the stall reasons.** If a dependent arithmetic chain were
the bottleneck, the stall reason would be `Stall Wait`, which is precisely a
warp waiting on a fixed-latency arithmetic dependency. Per issued instruction,
from `ncu`:

| stall reason (cycles per issued instruction) | `PolyOne` | `PolyFour` | `SmemHog` |
|---|---|---|---|
| **`wait`** (arithmetic dependency) | **3.01** | **2.98** | 2.91 |
| **`long_scoreboard`** (global memory operand) | **26.87** | **22.47** | 18.47 |
| `short_scoreboard` (shared/local) | 1.10 | 0.57 | 1.86 |
| `barrier` | 0 | 0 | 2.38 |
| `Warp Cycles Per Issued Instruction` (total) | 27.32 | 28.55 | 22.27 |

`Stall Wait` is **3.01 versus 2.98 — it does not move.** The FMA chain was
never the bottleneck in either kernel, so removing a dependency that was not
costing anything cannot be what made `PolyFour` faster. What *does* move is
`long_scoreboard`, 26.87 → 22.47: less time waiting on global loads. ncu's own
rule text says the same in words, attributing 75.8 % of `PolyOne`'s stall
cycles to "waiting for a scoreboard dependency on a L1TEX operation".

**Falsification C — the control kernel.** Write `PolyFour` the *sequential*
way: element 0 loaded, evaluated and stored before element 1 is even loaded. No
independent chains, no hoisted loads — the SASS is literally
`LDG → 63 FFMA → STG` four times over, and it compiles to 12 registers instead
of 20, confirming only one chain is live at a time. If ILP were the mechanism,
this version should lose the entire advantage. Measured, six runs:

| formulation | registers | SASS shape | ratio vs `PolyOne` |
|---|---|---|---|
| independent chains (shipped) | 20 | 4×LDG, then 4 serial chains | 1.28× (mode) |
| **sequential chains (control)** | **12** | LDG → chain → STG, ×4 | **1.20–1.36×** |

The same win, from a kernel with no instruction-level parallelism in it at all.
Hypothesis 1 is dead three times over.

### Hypothesis 2 — grid size: "fewer blocks means less launch overhead"

`PolyFour` launches 1954 blocks where `PolyOne` launches 7813. Fewer blocks to
dispatch, fewer tail effects, cheaper.

Also wrong, and one experiment kills it. Hold the **grid geometry identical**
— both 1954 blocks × 256 threads, same block count, same wave count, same
occupancy — and vary only the work per thread, by running `PolyOne` over
`n = 500001` and `PolyFour` over `n = 2000003`:

```
blocks: PolyOne(n=500001)=1954   PolyFour(n=2000003)=1954
PolyOne   0.0061 ms over  500001 el -> 0.01225 ns/el
PolyFour  0.0143 ms over 2000003 el -> 0.00715 ns/el
per-element advantage at IDENTICAL grid geometry: 1.714x
```

Five trials: 1.507×, 1.714×, 1.724×, 1.700×, 1.713×. With block count and
occupancy pinned, the advantage does not shrink — it **grows** to 1.7×. Nothing
about the grid explains it. It is entirely a per-thread effect.

### Hypothesis 3 — the one that survives: per-thread amortisation of exposed load latency

Put the surviving evidence together. The stall is a memory stall
(`long_scoreboard` = 26.87 cycles per issued instruction, 75.8 % of the total).
It gets smaller when each thread handles more elements (26.87 → 22.47), and it
gets smaller *at identical grid geometry*, so it is not about how many blocks
there are.

The mechanism: **every thread pays an exposed load latency at the start of its
life, and `PolyFour` amortises that cost over four elements instead of one.**

At one element per thread, all 48 resident warps of an SM are launched
together, execute an identical short prologue, and issue their `LDG` at
essentially the same instant. Every warp on the SM is then stalled on
`long_scoreboard` simultaneously — there is no other warp to switch to, because
every warp is at the same point in the same program. 100 % occupancy does not
help: occupancy gives you 48 warps to hide latency *with*, and it is worth
nothing when all 48 are blocked on the same thing at the same time. The SM
idles for one full memory latency, computes for one chain, and the block ends.

At four elements per thread the first round is identical, but after it the
warps have diverged in time — different warps reach their second, third and
fourth loads at different moments, because they were released from the first
stall in whatever order the memory system returned their data. Later loads then
overlap other warps' arithmetic, which is exactly what occupancy is *for*, and
it is only reachable once the warps stop marching in lockstep. `PolyOne` never
gets there, because a `PolyOne` thread's entire life is one prologue, one
stall, one chain, one store. It has no second round in which to desynchronise.

This also explains the two sweeps that made no sense under the ILP story:

- **Raising the polynomial degree makes the effect smaller, not larger.**
  Measured at `n = 2000003`: 64 terms → 1.33×, 128 → 1.18×, 192 → 1.13×,
  256 → 1.09×. Under the ILP hypothesis a longer dependent chain should widen
  the gap. Under this one it narrows it, because a longer chain means more
  arithmetic per exposed load, so the fixed per-thread stall is amortised by
  the *arithmetic* instead — and at 256 terms both kernels are simply at the
  FMA roofline with nothing left to win. 64 terms sits near the peak of the
  effect (16 → 1.32×, 48 → 1.29×, 96 → 1.22×) and is what ships.
- **More elements per thread does not keep helping.** Best-of-8 ratios:
  2 elements → 1.33×, **4 → 1.49×**, 8 → 1.42×, 16 → 1.29×. Registers grow
  with the count but never enough to cost occupancy on this SM. Four is the
  measured optimum, which is why `PolyFour` is `PolyFour`.

### So what is occupancy for?

Both halves of this puzzle are true at once, and together they are the whole
lesson:

- **`SmemHog`**: occupancy dropped from 100 % to 66.7 % and the kernel got
  1.21× slower doing identical work, with a control proving the staging itself
  was free. Occupancy was the cause.
- **`PolyOne` vs `PolyFour`**: occupancy identical at 100 %, pinned, unable to
  go higher — and a 1.29× gap that occupancy has nothing to say about.

Occupancy is a **budget for hiding latency**. It bounds how much overlap is
*available*. It tells you nothing about whether your kernel is in a position to
*use* it — and a kernel whose every warp stalls on the same load at the same
instant is not, no matter how many warps it has. `PolyOne` is the case where
maximum occupancy and poor latency hiding coexist, which is precisely the case
the metric is least able to warn you about.

## 6. Speed of Light, and why it names nothing

For completeness, `--section SpeedOfLight` on the three correctness launches:

| | `PolyOne` | `PolyFour` | `SmemHog` |
|---|---|---|---|
| `Compute (SM) Throughput` | 27.13 % | 28.66 % | 16.98 % |
| `Memory Throughput` | 20.87 % | 24.27 % | 12.36 % |
| `L1/TEX Cache Throughput` | 11.93 % | 14.20 % | 14.47 % |
| `L2 Cache Throughput` | 20.87 % | 24.27 % | 12.36 % |
| `SM Active Cycles` | 77 951 | 74 110 | 89 971 |
| `Elapsed Cycles` | 100 811 | 88 376 | 168 777 |
| `SM Frequency` | 2.15 GHz | 2.13 GHz | 2.13 GHz |
| `Duration` (see caveat 1) | 46.91 µs | 41.41 µs | 79.20 µs |

Every one of them is under 30 % of peak, and ncu says the same thing about all
three: *"Achieved compute throughput and/or memory bandwidth below 60.0% of
peak typically indicate latency issues. Look at Scheduler Statistics and Warp
State Statistics."* That is the correct advice and it is why `WarpStateStats`
is in `NCU_ARGS`. `SpeedOfLight` can tell you a kernel is latency-bound; it
cannot tell you *what* latency, and in this puzzle the entire answer is in that
distinction.

## 7. The regime: why `n = 2000003`

A working set of 16 MB against a 25.17 MB L2, chosen with the sweep, not by
taste. Same three kernels, only `n` changed, best of 5:

| n | a + out | `PolyOne` | `PolyFour` | ratio | `SmemHog`/`PolyOne` |
|---|---|---|---|---|---|
| 250 003 | 2 MB | 0.0042 | 0.0041 | 1.01× | 1.17× |
| 500 003 | 4 MB | 0.0062 | 0.0061 | 1.01× | 1.29× |
| 1 000 003 | 8 MB | 0.0103 | 0.0082 | 1.25× | 1.20× |
| **2 000 003** | **16 MB** | **0.0184** | **0.0144** | **1.28×** | **1.22×** |
| 3 000 003 | 24 MB | 0.0614 | 0.0575 | 1.07× | 1.03× |
| 4 000 003 | 32 MB | 0.1085 | 0.1046 | 1.04× | 1.03× |
| 6 000 003 | 48 MB | 0.1904 | 0.2002 | **0.95×** | 1.03× |
| 8 000 003 | 64 MB | 0.2596 | 0.2728 | **0.95×** | 1.02× |
| 16 000 003 | 128 MB | 0.5147 | 0.5425 | **0.95×** | 1.02× |

Two walls, one on each side, and the puzzle lives between them.

**Below 8 MB there is not enough work.** `PolyFour` at `n = 500003` is 245
blocks against 288 concurrent block slots (48 SMs × 6) — less than a single
full wave. The measurement is dominated by ramp-up and the two kernels are
indistinguishable.

**Above 24 MB the L2 is gone.** Past the cliff both kernels are DRAM-bound and
every effect in this document collapses: the `PolyOne`/`PolyFour` ratio to
1.04×, and — the more interesting one — `SmemHog`'s occupancy penalty to 1.02×.
Occupancy stops mattering at exactly the same point, because when the memory
system is the wall, extra resident warps have nothing to do but queue.

And past 48 MB the ratio **inverts**: `PolyFour` becomes 5 % *slower*. Its
quarter-sized grid is a quarter as many threads with loads in flight, and once
throughput is set by how many independent memory requests you can keep
outstanding, fewer threads is straightforwardly worse. The advantage was never
free — it was a trade of memory-level parallelism for per-thread amortisation,
and which side wins is decided by the regime, not by the code.

24 MB (`n = 3000003`) is already past the edge despite nominally fitting in a
25.17 MB L2: `a` and `out` are both live and replacement is not clairvoyant.
16 MB is comfortably interior, which is why it ships.

**This puzzle was originally specified at `n = 4000003`.** At 32 MB the
measured ratio is 1.04× and there is no puzzle: the assert would be a coin flip
and the lesson would be inverted by the next size up. The sweep is the reason
the shipped value is different.

## 8. Correctness and tolerance

All three kernels compute the same 63-step float Horner over character-
identical coefficients, so they agree with each other bit for bit and differ
from `ref_poly64`'s double accumulator only by accumulated rounding. Measured
over all 2 000 003 outputs:

| | worst relative deviation | at index | expected value |
|---|---|---|---|
| `PolyOne` | **2.364e-07** | 813 587 | 4.03478 |
| `PolyFour` | **2.364e-07** | 813 587 | 4.03478 |
| `SmemHog` | **2.364e-07** | 813 587 | 4.03478 |

Identical to the last digit across the three, which is itself a check: the
kernels are not merely within tolerance of the reference, they are equal to
each other, as three implementations of the same arithmetic should be.

`tol = 1e-5` clears the measured worst case by **42×**. The next decade down,
`1e-6`, clears it by only 4.2× and would have been too tight to call a margin.
1e-5 is the tightest decade that clears the measurement by more than 10×.

## 9. Measurement caveats, all of them found on this box

1. **Do not take `Duration` from `ncu`.** The profiler's injection library adds
   tens of microseconds per launch, and these kernels run in 15–25 µs. The SoL
   `Duration` row above reports 46.91 / 41.41 / 79.20 µs against the runner's
   18.4 / 14.3 / 22.4 µs — inflated roughly 2.5×, and unevenly. `prof.mk`
   passes `--kill yes` and `-c 3` so the process stops after the three
   correctness launches and the runner's own timing loop never executes under
   the profiler. Counters from `ncu`, timings from `make run`.

2. **`Achieved Occupancy` can exceed 100 %.** `PolyOne` reports **103.19 %**
   (49.53 achieved active warps per SM against 48 theoretical). It is a
   sampling artefact of a short kernel, not a real state — an SM cannot hold 49
   warps when the budget is 48. `PolyFour` reports 89.01 % and `SmemHog`
   91.99 %, the latter *above* its own 66.67 % theoretical for the same reason.
   The theoretical numbers are exact arithmetic and agree with the occupancy
   API to the block; the achieved numbers on kernels this short are not worth
   quoting, and none of the conclusions above rests on one.

3. **The timing assert is measuring a shared SoC.** GB10 is not a discrete
   card: the GPU and the CPUs are behind the same LPDDR5X. On an idle machine
   `PolyOne`/`PolyFour` is extremely reproducible — 1.279, 1.280, 1.283, 1.284,
   1.285, 1.286, 1.287, 1.288 across consecutive runs, a spread under 1 %, with
   the cleanest windows reaching 1.49×. Let anything else run and both kernels
   inflate together, both go memory-bound, and the ratio compresses: across
   ~80 whole-run measurements taken while this box was also running builds and
   other agents, the distribution has a mode at **1.28**, a clean maximum of
   **1.49**, and a contended tail reaching **1.052**. A 10 Hz `nvidia-smi`
   poll running alongside is enough to drag it to 1.055.

   `MARGIN` is therefore **1.05**, below the entire observed distribution,
   rather than the ~1.14 that "half the effect" would suggest. The measured
   ratio is printed on the `PASS` line every run, so the *size* of the effect
   is always visible; the assert only has to be true. When it does fire, the
   FAIL text names box contention as the first thing to check, because it is
   overwhelmingly the likeliest cause.

   Two things were tried and did not fix this, both worth recording as dead
   ends: a wall-clock ramp of 25–250 ms before timing (the SM clock is a stable
   2.55 GHz under sustained load either way — this is contention, not DVFS
   warm-up), and raising the rep count to 25 or 50 (contention persists for
   whole runs, so more reps inside one contended run buys nothing). What *did*
   help is interleaving: timing all reps of one kernel and then all reps of the
   next biases the ratio by ~0.05 against whichever kernel goes last, and one
   rep of each in turn removes it. That is why `bench_all` is shaped the way it
   is.

4. **`dram__*` counters do not exist on GB10**, as puzzle 30 found — this SoC
   does not expose DRAM-side performance counters, so `lts__*` is the deepest
   measurable level and every DRAM claim in section 7 rests on the behavioural
   evidence of the sweep rather than on a counter.

## 10. Three specification corrections, all forced by measurement

Recorded because they are findings about this hardware, not silent fixes:

| specified | shipped | why |
|---|---|---|
| `n = 4000003` | `n = 2000003` | 32 MB is past the 25.17 MB L2; the measured ratio there is 1.04× and inverts to 0.95× by 48 MB (section 7) |
| `__launch_bounds__(256, 8)` | `__launch_bounds__(256, 6)` | `8 × 256 = 2048 > 1536` threads/SM; ptxas discards the whole annotation with a warning (section 2) |
| ~12 KiB shared | 24 KiB shared | 12 KiB allows 7 blocks/SM, more than the 6 the thread budget already permits, so it would not have been the limiter at all (section 2) |

And one substantive re-framing: the puzzle was specified around
`ilp_beats_occupancy` — lower occupancy winning through instruction-level
parallelism. Neither half of that survived measurement. `PolyFour`'s occupancy
is not lower (both are pinned at 100 %), and ILP is not the mechanism (section
5, three falsifications). The assert ships as **`occupancy_not_predictive`**,
which is what the hardware actually demonstrates, and the dead hypotheses are
kept above because the trail from "obviously ILP" to "actually exposed load
latency, and here are the three experiments that got me there" is the most
useful thing in this file.
