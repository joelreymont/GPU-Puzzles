# Puzzle 29 — Synchronization: Barriers, Fences, and a Ping-Pong Stencil

Puzzle 28 was about moving data. This one is about *agreeing* — the moment when
every thread in a block has to know that every other thread's writes have
happened. You have used `__syncthreads()` since the first tiled kernel. This
puzzle takes it apart: what it actually guarantees, what the alternative
spelling (`cuda::barrier`) decouples, and what a memory fence does *not* do
even though it looks like it should.

## The computation

One step of a 3-point 1D stencil — the discrete heat equation, in the form you
would write for an explicit Jacobi sweep:

    next[i] = 0.25*cur[i-1] + 0.5*cur[i] + 0.25*cur[i+1]

with **clamped** indexing: a neighbour off the end of the array is the element
itself, so `next[0] = 0.75*cur[0] + 0.25*cur[1]` and likewise at `n-1`.

You must apply `T = 4` of these steps. Step `k+1` reads what step `k` wrote,
for the whole array — that is a genuine global data dependency, and it is the
whole difficulty. There are exactly three ways to satisfy it:

1. **Relaunch the kernel four times.** The end of a grid *is* the only free
   grid-wide barrier CUDA gives you. Correct, trivial, and it round-trips the
   entire array through global memory four times, plus four launch latencies.
2. **A grid-wide barrier inside one launch** (`cg::this_grid().sync()` under a
   cooperative launch). Costs a residency constraint on the whole grid.
3. **Make the dependency block-local**, so a *block* barrier is enough.

This puzzle is (3), and (3) is what production stencil and multigrid codes do.

## The trick: buy locality with redundant arithmetic

A block owns `SEG = 256` consecutive outputs. It loads those 256 elements
**plus `HALO = T = 4` elements of overlap on each side**, `SEG + 2*HALO = 264`
in all, and then never touches global memory again until it writes its results.

Why that is enough: one stencil step consumes one element off each end of
whatever the block can see. So the block's *correct* window shrinks by one per
side per step:

| after | correct shared range | width |
|---|---|---|
| load | `[0, 263]` | 264 |
| step 1 | `[1, 262]` | 262 |
| step 2 | `[2, 261]` | 260 |
| step 3 | `[3, 260]` | 258 |
| step 4 | `[4, 259]` | 256 |

After `T` steps the surviving window is exactly the `SEG` central elements —
`HALO = T` is the minimum halo that works, and the block writes out the window
it has left. Neighbouring blocks redundantly recompute each other's halo: 264
lanes of arithmetic per step for 256 useful outputs, about 3% wasted work per
step, and every block does it without ever asking another block anything.

**That is the trade.** Redundant computation in exchange for the elimination of
all inter-block synchronization. It is worth it whenever `T` is small relative
to `SEG`; at `T = 128` with `SEG = 256` the halo would be bigger than the
payload and you would go back to relaunching.

Both kernels here use two shared buffers and ping-pong between them: step `k`
reads buffer `p` and writes buffer `p^1`, then the roles swap.

## `__syncthreads()` is two things at once

This is the sentence to take away from the puzzle:

> `__syncthreads()` is an **execution barrier** *and* a **block-scope memory
> fence**, and the ping-pong needs both — from the same call.

- **Execution barrier**: no thread proceeds past it until every thread in the
  block has reached it. On this hardware it is one instruction,
  `BAR.SYNC.DEFER_BLOCKING`.
- **Memory fence**: every shared (and global) write a thread performed *before*
  the barrier is visible to every thread *after* it. Without the fence half,
  the hardware and the compiler would both be free to leave your `STS` sitting
  in a store queue, or to sink it below the barrier entirely.

Now trace one iteration of the ping-pong. There are two distinct hazards across
the barrier, and each of the two properties kills exactly one:

| hazard | what it is | needs |
|---|---|---|
| **RAW** | I am about to read `sh[p^1][j-1]`, which *another* thread wrote in this step | the fence half — its write must be visible to me |
| **WAR** | I am about to write `sh[p][j]` in the next step, and another thread may still be reading `sh[p][j+1]` from this one | the barrier half — it must be *done reading*, and no fence makes a thread wait |

Take either half away and one of those two rows breaks. This is also why the
double buffer earns its keep: in a single buffer the reads and the writes of a
step hit the same array, so each thread would have to stage its three
neighbours in registers, barrier so that everyone is done reading, write, and
barrier again so that the writes are visible — **two** barriers per step. Two
buffers, one barrier per step.

## `cuda::barrier`: the same guarantee, split in half

`#include <cuda/barrier>` gives you `cuda::barrier<cuda::thread_scope_block>`,
a barrier as an *object* rather than an instruction. It splits the single
`__syncthreads()` into two operations:

```
auto token = bar.arrive();     // "I am done. My writes are published."
   ... anything that does not depend on the other threads ...
bar.wait(cuda::std::move(token));  // "Now let everyone else's writes be mine."
```

The pieces:

- **`init(&bar, blockDim.x)`** constructs the barrier with an *expected arrival
  count*. Exactly one thread does this, and a plain `__syncthreads()` must
  follow it before anyone else touches `bar` — there is no way to bootstrap the
  first barrier with the barrier itself. That code is written for you.
- **`arrive()`** decrements the pending count and acts as a **release**: this
  thread's prior shared-memory writes are published. It returns an
  `arrival_token` naming *which* barrier phase you arrived at.
- **`wait(token)`** blocks until the phase that token names has completed —
  i.e. until all `blockDim.x` threads have arrived — and acts as an
  **acquire**: everyone else's writes are now visible to you. The token is
  move-only precisely because it is a one-shot claim on one phase.
- **The barrier then flips phase and re-arms itself automatically**, with the
  same expected count. That is why one `bar` serves all four steps.
- **`arrive_and_wait()`** is the fused form, and is `__syncthreads()` with
  extra steps — use it where you have nothing to overlap.
- **`cuda::thread_scope_block`** is the scope of the guarantee: threads of this
  block, shared memory. Larger scopes (`thread_scope_device`,
  `thread_scope_system`) exist and cost more; naming the smallest correct scope
  is the whole point of the template parameter.

**What the split buys you** is the gap between `arrive()` and `wait()`. A
thread that has finished its share of the step can do useful work there instead
of stalling — as long as that work touches nothing the barrier is ordering. In
this puzzle the available work is small and honest: rotating the two buffer
pointers for the next step is pure register bookkeeping, so it is legal in the
gap. See the measurements below before you assume it makes anything faster.

## `__threadfence_block()` is not a barrier

A fence is about **ordering and visibility**. A barrier is about **waiting**.
They are not on the same axis, and the API names do not help:

| | orders my accesses | makes others wait | scope |
|---|---|---|---|
| `__threadfence_block()` | yes | **no** | this block |
| `__threadfence()` | yes | **no** | this device |
| `__threadfence_system()` | yes | **no** | device + host + peers |
| `__syncthreads()` | yes (block scope) | **yes** | this block |

`__threadfence_block()` says: my writes before this point become visible to
other threads of my block before any of my writes after this point. It says
nothing about *when*, and it does not delay a single other thread by a single
cycle. A thread can execute `__threadfence_block()` and be three iterations
ahead of its neighbours a microsecond later.

`__threadfence()` is the same statement at device scope — it is what you need
when a block publishes a flag that another block spins on (the classic
threadfence-reduction pattern), and it is *still* not a barrier.

### The classic mistake, measured

Replace the single `__syncthreads()` in the ping-pong loop with
`__threadfence_block()` — the data flow argument still sounds fine, "my writes
are visible before I read" — and build it against this exact runner. Measured
on this box (GB10, `sm_121`, CUDA 13.0), **broken variant**:

```
FAIL stencil_syncthreads at index 28: got 0.548707 want 0.527023
FAIL stencil_syncthreads at index 26: got 0.628129 want 0.629656
FAIL stencil_syncthreads at index 26: got 0.628129 want 0.629656
```

Three consecutive runs of the same binary on the same input, and the failing
index moves — the signature of a schedule-dependent race, not a logic bug. The
second kernel, untouched, still passes in the same run. Under the sanitizers,
same binary:

| tool | broken variant | the solution |
|---|---|---|
| `memcheck` | `ERROR SUMMARY: 0 errors` | 0 errors |
| `synccheck` | `ERROR SUMMARY: 0 errors` | 0 errors |
| `racecheck` | `RACECHECK SUMMARY: 4 hazards displayed (4 errors, 0 warnings)` | 0 hazards, 0 errors |

Those four displayed hazards are *deduplicated write sites*. Expanded,
`racecheck` reports 18 write→read instruction pairs totalling **2 855 064
individual hazards** in one run (2 855 192 in another — it is a race, the count
moves), all of this form:

```
Error: Race reported between Write access at StencilSyncthreads(const float *, float *, int)+0x500 in kernel_broken.cu:57
    and Read access at StencilSyncthreads(const float *, float *, int)+0x340 in kernel_broken.cu:57 [8844 hazards]
    and Read access at StencilSyncthreads(const float *, float *, int)+0x350 in kernel_broken.cu:58 [395484 hazards]
    and Read access at StencilSyncthreads(const float *, float *, int)+0x670 in kernel_broken.cu:57 [399472 hazards]
    and Read access at StencilSyncthreads(const float *, float *, int)+0x680 in kernel_broken.cu:58 [13756 hazards]
```

Lines 57–58 are the two lines of the stencil expression, so both sides of every
race are the same statement: thread A writing step `k+1` into the buffer thread
B is still reading for step `k`. That is the
**WAR** row of the table above, the one the fence cannot help with. Note also
that `memcheck` and `synccheck` are both perfectly happy — a race is not an
invalid access and it is not barrier divergence. Only `racecheck` sees it.

## What the two kernels actually compile to

`cuobjdump -sass` on the built solution, GB10 / `sm_121` / CUDA 13.0:

| | `StencilSyncthreads` | `StencilBarrierArriveWait` |
|---|---|---|
| barrier instructions | 5 × `BAR.SYNC.DEFER_BLOCKING` | 1 × `BAR.SYNC` (the bootstrap) + 5 × `SYNCS.ARRIVE.TRANS64` + 10 × `SYNCS.PHASECHK.TRANS64.TRYWAIT` |
| registers (`-Xptxas -v`) | 15 | 14 |
| static shared | 2112 B | 2120 B |

Read the second column carefully. `__syncthreads()` is *one* instruction that
the SM's barrier unit resolves. `bar.wait()` is a **spin loop**: a
`SYNCS.PHASECHK...TRYWAIT` test, and if the phase has not flipped, a
`NANOSLEEP` backoff ladder driven by `SR_GLOBALTIMERLO` that retests and
eventually times out. Ten `TRYWAIT`s for five waits — one speculative test plus
one in the retry loop each. The extra 8 bytes of shared memory are the barrier
object itself.

## What it cost

Measured on this box, runner output, three runs of each launch order:

| kernel | order A | order B |
|---|---|---|
| `stencil_syncthreads` | 0.0042 / 0.0045 / 0.0046 ms | 0.0041 / 0.0042 / 0.0042 ms |
| `stencil_arrive_wait` | 0.0061 / 0.0061 / 0.0061 ms | 0.0063 / 0.0064 / 0.0062 ms |

**The split-barrier version is about 1.5× slower here, and that is the honest
result.** It is not noise and it does not depend on which kernel is timed
first. The reason is in the SASS: a hardware barrier instruction against a
spin-wait loop, five times per block, with only a two-register pointer swap in
the gap to pay for it — and the compiler schedules that swap wherever it likes
anyway, since it touches no memory.

So why write the arrive/wait version at all? Because the *structure* is what
matters, not this measurement:

- The gap is where puzzle 28's other spelling puts its real work.
  `cuda::memcpy_async(..., bar)` issues the next tile's copy *into* the barrier,
  which then tracks the transfer as well as the arrivals; the thread arrives,
  computes a whole tile out of the other buffer, and only then waits. The thing
  being overlapped is a memory transfer, not three register moves.
- `arrive()` and `wait()` can be executed by *different* code paths — producer
  threads arrive, consumer threads wait — which `__syncthreads()` cannot
  express at all, since it requires every thread to reach the same instruction.
- The expected count is a parameter, not `blockDim.x` by decree, so a subset of
  the block can own a barrier.
- It is the same object the TMA/`mbarrier` machinery is built on, so this is
  the API shape the rest of the modern async story is written in.

Use `__syncthreads()` when the whole block synchronizes at the same point and
has nothing to do in between — which is most of the time, and it is faster.
Reach for `cuda::barrier` when you need the split.

## I/O contract

| | |
|---|---|
| `a` | `n` floats, the initial field. Read-only. |
| `out` | `n` floats, **every one written**: the field after exactly `T = 4` stencil steps. The runner poisons `out` with `0xff` (a NaN pattern) before each launch, so an element you never write is a `FAIL`, not a lucky zero. |
| `n` | `100003` — not a multiple of 32 or of `SEG`. The last block owns 163 outputs, not 256. |
| grid | `cdiv(n, SEG) = 391` blocks × `TPB = 256` threads. One block per output segment; no grid-stride loop. |
| boundary | Clamped: an out-of-array neighbour is the element itself. This applies at *every* step, not just the first. |
| tolerance | `1e-5` relative, against a CPU reference that runs the same four steps in double precision. |

Given to you at the top of the file, outside the fill-in regions: the constants
(`T`, `SEG`, `HALO`, `TPB`, `SH`), both shared buffers, the `cuda::barrier`
bootstrap, and the three index values `base`, `lo`, `hi` — where shared slot
`j` holds global element `base - HALO + j`, and `lo`/`hi` are the shared
indices of global elements `0` and `n-1` (clamping a neighbour into `[lo, hi]`
*is* the boundary condition, and it has to be re-applied on every step, not
just at load time).

## Task

Two kernels in `skeletons/p29_synchronization/kernel.cu`. They compute the same
thing and must produce identical output; only the synchronization differs.

### 1. `StencilSyncthreads` (approx 14 lines)

Cooperatively load `SH = 264` elements into buffer 0 — 264 slots over 256
threads, so the strided loop matters and the first few threads take a second
element. Clamp the global index so the halo of an edge block reads the nearest
real element instead of running off the allocation. Barrier. Then `T`
ping-pong steps, one `__syncthreads()` each, and finally each thread stores its
one output from the central region of whichever buffer the last step wrote.

Two things to get right:

- The store guard is a *value* guard, not an early return: the last block has
  163 real outputs but all 256 of its threads must reach all five barriers.
  (A thread that skips a `__syncthreads()` its block-mates execute is
  undefined behaviour, and `synccheck` is the tool that finds it.)
- Every thread computes all the slots it loaded, halo included. Do not try to
  narrow the computed range per step to match the shrinking window — the
  bookkeeping costs more than the arithmetic it saves, and the discarded values
  are provably never read.

### 2. `StencilBarrierArriveWait` (approx 17 lines)

Identical arithmetic, synchronized with the `bar` that is already initialized
for you. Use `arrive_and_wait()` after the load, where there is nothing to
overlap. Inside the step loop use the split form: compute, `arrive()`, do the
work that depends on nobody (the buffer-role rotation for the next step),
`wait()`. The token is move-only: `bar.wait(cuda::std::move(token))`.

## Run

```
make run   P=29                  # your kernel — fails loudly
make run   P=29 MODE=solution    # reference
make check P=29 MODE=solution    # memcheck + racecheck + synccheck, all zero
```

Expected: two `PASS` lines and a short timing print. `make check` takes about
4.5 s end to end on this box.

## Race-freedom is the answer here

Puzzle 28 made the point that a `PASS` proves nothing about a shared-memory
race. This puzzle is the one where you cannot pretend otherwise: a
double-buffered ping-pong is a shared array that one thread reads while another
is scheduled to overwrite it, four times per launch. A puzzle 29 solution is
done when `make run` passes **and** `memcheck`, `racecheck` and `synccheck`
each report zero, for **both** kernels.
