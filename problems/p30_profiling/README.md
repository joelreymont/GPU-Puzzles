# Puzzle 30 — Profiling: The Cache Hit Paradox

Puzzles 24–27 asked whether your kernel got the right answer. Puzzle 28 added
"and how fast", and ended on a warning: `ncu` is authoritative about *counters*
and not about *wall time*. This puzzle is the other half of that sentence. The
kernels here are four lines each and neither of them is the puzzle. The puzzle
is a number the profiler will show you that says the opposite of the truth.

You get two kernels that compute the same thing from the same input on the same
grid. One of them is decisively slower — the runner measures it and asserts it,
so you cannot talk yourself out of the result. Then you profile both, and the
kernel that hits in L1 almost every time turns out to be the loser, while the
kernel that misses L1 *every single time* wins.

Your job is not to write the kernels. Your job is to explain that.

## The computation

`out[i]` is the total of `i`'s aligned 32-element group, stored by every thread
of that group:

    base = (i / 32) * 32
    out[i] = a[base] + a[base+1] + ... + a[base+31]

The same `ref_group_sum_all` semantics as puzzle 26's butterfly. A trailing
group shorter than 32 sums only the elements that exist.

Groups are 32 wide and aligned, and `TPB = 256` is a multiple of 32, so **a
group is exactly one warp's worth of consecutive elements** and every thread of
a warp shares a group. That is why both kernels below are possible.

## I/O contract

| | |
|---|---|
| `a` | `n` floats. Read-only. |
| `out` | `n` floats, **every one written**. The runner poisons `out` with `0xff` (a NaN pattern) before each launch, so an element you never write is a `FAIL`, not a lucky zero. |
| `n` | `1000003` — a multiple of neither 32 nor `TPB`. The last group holds 3 elements, not 32, and the last block has 3 live threads, not 256. |
| grid | `cdiv(n, TPB) = 3907` blocks × `256` threads. One thread per output; no grid-stride loop. |
| guards | **Value guards, not early returns.** `GroupSumStream` shuffles with a full `0xffffffff` mask, so no lane may leave the warp before the butterfly. |
| tolerance | `1e-5` relative, against a CPU reference that accumulates each group in double. |

`i` and (for the redundant kernel) `base` are given to you above the fill-in
region. The kernels sit in `skeletons/p30_profiling/kernel.cu`.

### 1. `GroupSumRedundant` (approx 6 lines)

The obvious way, and the way you would write it if you had not read puzzles
24–26. Each thread works alone: loop `j = 0 .. 31`, read `a[base + j]` **out of
global memory** every time, accumulate into a local `float`, store `out[i]` at
the end. No shuffles, no shared memory, no cooperation.

Every element of the group therefore gets fetched 32 times over — once by each
thread that needs it. Guard the `j`-index with a value (`(k < n) ? a[k] : 0.0f`)
so the short trailing group contributes only what exists, and guard the store
with `i < n`.

### 2. `GroupSumStream` (approx 4 lines)

Puzzle 26's `ButterflyAllReduce`, applied. Each thread reads exactly **one**
element — its own, `a[i]`, value-guarded — so the warp's 32 loads coalesce into
a single request. Then `__shfl_xor_sync(0xffffffffu, v, mask)` for `mask` =
16, 8, 4, 2, 1, summing at each step. After the last step every lane holds the
group total, so every in-range lane stores its own `out[i]`.

## Run

```
make run   P=30                  # your kernel — fails loudly
make run   P=30 MODE=solution    # reference
make check P=30 MODE=solution    # memcheck + racecheck + synccheck, all zero
make prof  P=30 MODE=solution    # the metrics this puzzle is about
```

Expected from `make run`: two `PASS` lines, a timing table, and
`PASS cache_hit_paradox (ratio ...)`. The runner verifies both kernels before
it times anything and skips timing entirely if either is wrong — a fast wrong
kernel is not a result, and a *relationship* between two kernels means nothing
unless both are right.

`make check` takes about 2 s. It sets `P30_SKIP_TIMING=1` (see
`problems/p30_profiling/prof.mk`), which runs both kernels and both correctness
checks under all three sanitizers but skips the wall-clock comparison: under
instrumentation the clock is measuring the sanitizer. Set that variable
yourself if you run `compute-sanitizer` by hand. `make prof` gets the same
protection a different way — see the comments in `prof.mk`.

> `ncu` needs permission to read the GPU's performance counters, which on this
> box is admin-only (`RmProfilingAdminOnly: 1`). If `make prof` reports
> `ERR_NVGPUCTRPERM`, run the same command under `sudo` — there is a sudoers
> rule for exactly `/usr/local/cuda/bin/ncu`.

## The puzzle

Run `make run P=30 MODE=solution` and write down which kernel is slower and by
how much. Then run `make prof P=30 MODE=solution` and read the two kernels'
blocks side by side.

The first thing you should notice is that **`L1/TEX Hit Rate` and the timing
table disagree about who is winning.** Not slightly — completely. Everything
else follows from working out why, so do not move on until you can answer all
six of these from the profiler output rather than from intuition:

1. What is each kernel's `l1tex__t_sector_hit_rate` and its global-load-only
   hit rate? Which kernel is faster? Say the contradiction out loud.
2. `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` counts *warp-wide load
   instructions* and `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` counts
   the 32-byte sectors those instructions resolved to. What is each kernel's
   number, what is the ratio between the kernels for each counter, and where in
   the source does each ratio come from? Predict both from the source before
   you look.
3. A hit rate is a *fraction*. Multiply each kernel's sector count by its hit
   rate to get the absolute number of L1 hits in each. Now say what a hit is
   evidence of.
4. `lts__t_sectors.sum` is the traffic L1 handed on to L2 — everything below
   the level where the two kernels differ. Compare the two. What does that
   comparison rule out as an explanation?
5. `SpeedOfLight` names a limiter for each kernel. What is each one limited by?
   Which one is `Mem Pipes Busy` pointing at, and note that the number
   `Max Bandwidth` reports is not a DRAM number.
6. `sectors/request` is the classic coalescing metric, and the classic reading
   is that 1.00 is perfect and large values mean an uncoalesced access pattern.
   Read that line for both kernels. Then decide what that metric is actually
   measuring and when the classic reading applies.

Finally, one prediction to make *before* you test it: the working set here is
8 MB. Say what you expect to happen to the ratio if `n` grows until `a` and
`out` no longer fit in this box's L2 — and why.

When you have answers, check them against
`solutions/p30_profiling/SOLUTION.md`, which carries the measured numbers from
this box, including the answer to that last prediction and one counter-
experiment that is worse than either kernel here.
