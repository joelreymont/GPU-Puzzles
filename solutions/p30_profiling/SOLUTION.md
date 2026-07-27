# Puzzle 30 — Solution: The Cache Hit Paradox

Every number below was measured on this box: NVIDIA GB10, `sm_121`, 48 SMs,
25.17 MB L2, CUDA 13.0. Nothing here is quoted from anywhere.

## What the runner measured

Three consecutive `make run P=30 MODE=solution`:

```
# timing: 3 warmup + 20 timed iterations, 8.00 MB of useful traffic per iteration
# kernel                  avg ms   eff GB/s
  group_sum_redundant     0.0206      388.6
  group_sum_stream        0.0083      967.3
PASS cache_hit_paradox (ratio 2.489x >= 1.25x)
```

Ratios over eleven runs of the same binary: 2.486, 2.489, 2.491, 2.495, 2.496,
2.498, 2.498, 2.501, 2.502, 2.504, 2.505. **The measured effect is
2.49×**, and its entire spread is 0.8 %. `MARGIN` in the runner is `1.25f`,
just over half of that — the assert is a floor on the effect, deliberately far
from the thing it is guarding.

## What `make prof P=30 MODE=solution` measured

`ncu -c 2 --kill yes`, the two correctness launches. Counter rows verbatim from
one run; percentage rows as a measured range (see immediately below the table):

| | `GroupSumRedundant` | `GroupSumStream` | redundant / stream |
|---|---|---|---|
| `l1tex__t_sector_hit_rate.pct` | **77.78 %** | **0 %** | — |
| `..._pipe_lsu_mem_global_op_ld_hit_rate.pct` | **87.50 %** | **0 %** | — |
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | 1 000 003 | 31 251 | **32.0×** |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 1 000 003 | 125 001 | **8.0×** |
| `l1tex__average_t_sectors_per_request_..._op_ld.ratio` | 1.00 | 4.00 | 0.25× |
| `l1tex__data_pipe_lsu_wavefronts.sum` | 1 093 983 | 254 942 | 4.3× |
| `lts__t_sectors.sum` | 251 632 | 251 255 | **1.0015×** |
| SoL `L1/TEX Cache Throughput` | **71.1 – 73.9 %** | 24.0 – 24.9 % | ~3× |
| SoL `L2 Cache Throughput` | 22.1 – 25.7 % | 17.1 – 21.3 % | |
| SoL `Memory Throughput` | 58.6 – 68.2 % | 17.1 – 21.3 % | |
| SoL `Compute (SM) Throughput` | 58.6 – 68.2 % | 16.9 – 21.0 % | |
| MWA `Mem Pipes Busy` | 58.6 – 68.2 % | 16.9 – 21.0 % | |
| MWA `Mem Busy` | 30.2 – 35.1 % | 14.2 – 17.6 % | |
| SoL `Duration` (see caveat 1) | 30.6 – 35.7 µs | 23.2 – 28.8 µs | ~1.3× |

The request, sector, hit-rate and sectors/request rows are **bit-identical on
every run**; the wavefront and `lts__t_sectors` rows move by about 0.2 %. The
percentage rows are given as the range over three `make prof` runs, because
most of them are normalised to *elapsed* cycles and therefore inherit the
profiler's own per-launch overhead. `L1/TEX Cache Throughput` is the
exception — it is normalised to *active* cycles, which is why it is the
steadiest number in the block and the one worth leaning on.

Every counter in that table is reproducible by hand, which is how you know the
profiler is being read correctly and not just quoted:

- 31 251 = `cdiv(1000003, 32)` groups = one warp-wide load per warp. That is
  `GroupSumStream`'s request count exactly.
- 1 000 003 = 31 251 × 32 − 29. The full 32 loop iterations for every warp,
  less the 29 iterations of the last warp whose `k < n` guard is false for
  *all* 32 lanes, so no request is issued at all. `GroupSumRedundant`'s request
  count exactly. The amplification is precisely 32×, the loop trip count.
- 125 001 = 31 251 × 4 − 3: a coalesced 128-byte warp load is four 32-byte
  sectors. `GroupSumStream`'s sector count exactly.
- 87.50 % = 28/32. Each warp's group is 128 B = 4 sectors. `GroupSumRedundant`
  touches those 4 sectors 32 times over, so 4 accesses miss and 28 hit.
- 77.78 % = 875 003 / (1 000 003 + 125 001): the load hits divided by all L1
  sector accesses, loads plus the always-missing 125 001 store sectors.

## The answer

**A hit rate is a ratio, and the thing it is a ratio *of* is work you chose to
do.** Performance is set by the numerator and denominator separately, and the
profiler prints only their quotient.

Count the absolute hits (question 3 in the README):

- `GroupSumStream`: 125 001 sector accesses, **0 hits**.
- `GroupSumRedundant`: 1 000 003 sector accesses, **875 003 hits**.

`GroupSumRedundant` did not earn 875 003 hits. It *manufactured* them. Every
one of those hits is a load instruction that `GroupSumStream` never issued,
because in `GroupSumStream` the value was already in a register and got moved
between lanes by `SHFL.BFLY` instead. An L1 hit is not free: it is still a
warp-wide request, still a tag lookup, still a slot in the LSU pipe, still a
data-path wavefront. It is merely *cheaper than a miss*. A hit rate near 90 %
tells you that most of your memory traffic was cheap. It says nothing at all
about how much of it there was, and this kernel's answer to that second
question is "thirty-two times more than necessary".

So the metric that looks like a grade is really a symptom: **in these two
kernels the L1 hit rate is a direct measurement of the redundancy**. 87.5 % of
the loads hit *because* 87.5 % of them were re-reads of something a lane in the
same warp had already fetched. Drive the redundancy up and the hit rate goes up
with it (see the counter-experiment below); remove the redundancy entirely and
the hit rate falls to exactly 0 %, which is what perfect behaviour looks like
here — every sector fetched once, used, and never asked for again.

### Where the time actually goes

`lts__t_sectors.sum` settles it: 251 632 against 251 255, a difference of
0.15 %. **The two kernels put identical traffic on L2 and everything below
it.** Both read `a` once and write `out` once at that level; split out with
`lts__t_sectors_op_read.sum` / `..._op_write.sum` the measurement is 125 899
read + 125 141 write for the redundant kernel against 125 557 + 125 114 for the
streaming one, either side of the 125 001 + 125 001 the arithmetic predicts. So
nothing below the L1 boundary can explain a 2.49× gap, and every memory-side
explanation that starts "the slow one is moving more data" is dead on arrival.

The gap is entirely above L1, and `SpeedOfLight` names it. For
`GroupSumRedundant`, `L1/TEX Cache Throughput` is **71–74 %** while `L2 Cache
Throughput` is 22–26 %; `Mem Pipes Busy` is 59–68 % and `Mem Busy` — the
*device* memory interface — only 30–35 %. `Memory Throughput` and
`Compute (SM) Throughput` are equal to each other on every run, because they
are the same bottleneck counted twice: the LSU is an SM pipe and the L1 is a
memory level, and this kernel's limiter sits exactly where they meet. `ncu`'s
own verdict, "Compute and Memory are well-balanced", is a misreading of that
coincidence; the kernel is neither compute-bound nor DRAM-bound, it is
**L1/LSU-request-bound**.

`GroupSumStream` is at 17–21 % on the same axes and `ncu` says
"Achieved compute throughput and/or memory bandwidth below 60.0% of peak
typically indicate latency issues". Read that as the good news it is: with the
redundant traffic removed there is nothing left to be bound by. Four lines of
kernel, one load, five `SHFL.BFLY`, one store, and the machine is coasting.

### The trap in `sectors/request` (README question 6)

| | `GroupSumRedundant` | `GroupSumStream` |
|---|---|---|
| sectors per global-load request | **1.00** | **4.00** |

By the usual reading — 1.00 is a perfectly coalesced access, larger values mean
threads are scattered across cache lines — the *slow* kernel scores perfectly
and the fast one scores four times worse. Both readings are wrong because the
metric is being asked the wrong question.

`sectors/request` measures how *wide* one request is, not how *efficient* it
is. In `GroupSumRedundant` every one of a warp's 32 lanes asks for the same
4-byte address, so the hardware coalesces the request down to a single sector:
32 B fetched, 4 B used, 12.5 % of the sector consumed. `ncu`'s
`MemoryWorkloadAnalysis_Tables` section states this outright —
`Average Bytes Per Sector For Global Loads: 4 byte/sector` against a maximum of
32, with `Est. Speedup: 58.97%`. (That table is not in the default `NCU_ARGS`
because it needs `--print-details all`, which prints 972 lines.) In
`GroupSumStream` the 32 lanes ask for 32 consecutive addresses, so one request
covers a full 128 B line — four sectors, 100 % of every one of them used.

`sectors/request` is only a coalescing metric when each lane wants a *distinct*
address. The moment lanes share addresses it inverts, and a broadcast — the
most degenerate access pattern in this file — reports the best possible score.

### What the SASS says

`cuobjdump -sass` on the built solution, opcode counts per kernel:

| | `GroupSumRedundant` | `GroupSumStream` |
|---|---|---|
| `LDG.E` | **32** | **1** |
| `SHFL.BFLY` | 0 | 5 |
| `FADD` | 32 | 5 |
| `ISETP.GE.AND` | 33 | 1 |
| `STG.E` | 1 | 1 |
| registers (`-Xptxas -v`) | **38** | **12** |

The loop is fully unrolled, so the 32× is visible as literal instruction count,
and 32 loads in flight is also where the 38 registers go — 3.2× the register
footprint of the streaming version, which on a bigger kernel would cost
occupancy on top of everything else. The arithmetic is identical work either
way: 32 `FADD` in a serial dependent chain against 5 `FADD` in a butterfly.
The butterfly is the *cheaper* reduction on both axes at once.

## The counter-experiment: making it coalesced makes it worse

The obvious "fix" once you see `4 byte/sector` is to make every step of the
loop a full-width warp load — have each lane start at its own position and walk
the group cyclically, so all 32 lanes read 32 distinct consecutive elements on
every iteration:

```
const int k = base + ((int)(threadIdx.x & 31) + j) % 32;   // instead of base + j
```

Same result, same 32 loop iterations, every load now perfectly coalesced.
Measured, same runner, same `n`:

| | plain `base + j` | rotated | `GroupSumStream` |
|---|---|---|---|
| avg ms | 0.0206 | **0.0288** | 0.0082 |
| ratio vs stream | 2.49× | **3.50×** | 1.00× |
| global-load requests | 1 000 003 | 1 000 032 | 31 251 |
| global-load sectors | 1 000 003 | **4 000 032** | 125 001 |
| sectors/request | 1.00 | **4.00** | 4.00 |
| load hit rate | 87.50 % | **96.88 %** | 0 % |
| `l1tex__t_sector_hit_rate` | 77.78 % | **93.94 %** | 0 % |
| `lts__t_sectors.sum` | 252 042 | 251 239 | 251 255 |

The rotated version now scores *identically to the fast kernel* on
`sectors/request` (4.00), has the **highest hit rate of all three** (96.88 %,
= 124/128: 128 sector accesses per warp over the same 4 distinct sectors), and
is the **slowest of all three**. Its L2 traffic is unchanged, again. All that
changed is that each of the same 1 000 032 requests now drags 4 sectors through
the L1 data path instead of 1 — 128 MB of L1→register traffic to deliver 4 MB
of distinct input.

This is the same lesson twice. The first version's hit rate was high because it
re-read; the second's is higher because it re-reads *more thoroughly*. Neither
number was ever describing anything but the redundancy.

## The prediction: what happens past the L2 cliff

The last README question. Same two kernels, compiled from
`solutions/p30_profiling/kernel.cu` into a sweep harness, `n` swept, best of
three repetitions of 3 warmup + 20 timed iterations:

| n | working set | redundant | stream | red GB/s | str GB/s | ratio |
|---|---|---|---|---|---|---|
| 250 003 | 2 MB | 0.0068 ms | 0.0041 ms | 295 | 485 | 1.65× |
| 500 003 | 4 MB | 0.0118 ms | 0.0061 ms | 338 | 656 | 1.94× |
| **1 000 003** | **8 MB** | **0.0205 ms** | **0.0082 ms** | **390** | **972** | **2.49×** |
| 2 000 003 | 16 MB | 0.0369 ms | 0.0144 ms | 433 | 1109 | 2.56× |
| 4 000 003 | 32 MB | 0.1258 ms | 0.1091 ms | 254 | 293 | **1.15×** |
| 8 000 003 | 64 MB | 0.2923 ms | 0.2600 ms | 219 | 246 | 1.12× |
| 16 000 003 | 128 MB | 0.5865 ms | 0.5163 ms | 218 | 248 | 1.14× |

**The paradox evaporates between 16 MB and 32 MB**, which is this box's 25.17
MB L2, and past it both kernels flatten onto the same ~220–250 GB/s. Puzzle 28
measured a flat 232 GB/s past the same cliff with entirely different kernels,
which is what a real hardware ceiling looks like, and it is the
whole point: once the working set no longer fits, the memory system becomes the
limiter, and since the two kernels' `lts__t_sectors.sum` are *identical*, the
memory system charges them the same. A 32× amplification that lives entirely
inside L1 costs nothing once something slower than L1 is the constraint.

Which is why `n = 1000003` and not something bigger. At 8 MB, DRAM is not the
wall, so the L1 request volume is free to be the wall, and the puzzle has
something to show. At 64 MB the two kernels differ by 1.12× — inside the noise,
and a relationship assert built on it would be a coin flip. **The paradox is
real but it is regime-dependent, and the regime is the first thing to establish
about any optimisation.**

## Measurement caveats, all of them found on this box

1. **Do not take `Duration` from `ncu`.** The profiler's injection library adds
   12–22 µs to every launch here. With the timing loop allowed to run under
   `ncu`, the runner reports 0.0322 ms / 0.0305 ms — a ratio of **1.06×**
   instead of the true 2.49×, which is why `prof.mk` passes `--kill yes` and
   stops the process after the two profiled launches. `--clock-control none`
   does not help (1.03×): the overhead is per launch, not clock scaling. The
   SoL `Duration` row above is subject to exactly the same distortion — it puts
   the two kernels ~1.3× apart instead of 2.49× — and is included only for
   completeness.
2. **Do not take wall time from `compute-sanitizer` either.** With the timing
   loop enabled under the sanitizers this box reports **15.43×** under
   `memcheck` and **1.03×** under `racecheck` for the same binary and the same
   input. `prof.mk` sets `P30_SKIP_TIMING=1` for all three sanitizer runs;
   correctness still runs in full under all three, and all three report zero.
3. **`dram__*` counters do not exist on GB10.** `dram__bytes.sum`,
   `dram__bytes_read.sum` and `dram__throughput.avg.pct_of_peak_sustained_elapsed`
   all return `(!) n/a` here — this SoC does not expose the DRAM-side
   performance counters, so `lts__t_sectors.sum` is the deepest level this box
   can actually measure, and every "DRAM" claim above is really an L2-traffic
   claim plus the behavioural evidence of the sweep table.
4. **`lts__t_sector_hit_rate` reads ~0.3 % in a regime where it cannot be
   right.** Both kernels report 0.37 % / 0.29 % L2 hit rate, with
   `--cache-control none`, profiled at launch #10 and #35 (i.e. deep inside the
   warm timing loop) as well as at launch #0. Yet at that working set
   `GroupSumStream` sustains 972 GB/s of useful traffic, 3.6× this part's rated
   273 GB/s of memory bandwidth, and the sweep table shows a hard cliff exactly
   at the L2 capacity. Those three facts cannot all be true. The conclusion
   drawn above rests on the counters that *are* self-consistent
   (`l1tex__*` and `lts__t_sectors.sum`, both reproducible by hand to within
   0.2 %) and on the measured sweep — not on the L2 hit-rate counter.

Point 3 and point 4 are the honest residue of profiling a part the tooling does
not fully cover, and they are worth as much as the rest of the puzzle: a
counter that disagrees with the timing harness and with arithmetic is a counter
you stop using, not a result you write down.
