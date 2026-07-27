# Puzzle 33 — Tensor Cores: A Warp Is the Unit, Not the Thread

Every kernel in this repo so far has had a thread-to-data map. One thread per
element, one thread per window, one thread per tile row. Even the warp puzzles
kept it: a shuffle moves a value from *this* lane to *that* lane, and you can
name which.

Tensor cores throw that away. One warp computes a 16×16 tile of `C` — 256
outputs from 32 threads — and there is no answer to "which thread owns element
`[3][7]`". The value lives somewhere in the warp's registers in a layout the
hardware defines and the API does not tell you. You do not index it. You load
a fragment, multiply fragments, store a fragment, and every one of those verbs
is something all 32 lanes do together.

That is the whole shift, and it is why this puzzle is short.

## What `wmma` gives you

One header, no library, no launch API change:

```cpp
#include <mma.h>
using namespace nvcuda;
```

Three types and four calls.

```cpp
wmma::fragment<wmma::matrix_a,    M, N, K, __half, wmma::row_major>  fa;
wmma::fragment<wmma::matrix_b,    M, N, K, __half, wmma::row_major>  fb;
wmma::fragment<wmma::accumulator, M, N, K, float>                    acc;

wmma::fill_fragment(acc, 0.0f);          // set every element to a scalar
wmma::load_matrix_sync(fa, ptr, ldm);    // memory -> fragment
wmma::mma_sync(d, a, b, c);              // d = a * b + c
wmma::store_matrix_sync(ptr, acc, ldm, wmma::mem_row_major);
```

Four things about that block are load-bearing.

**A fragment is warp-collective opaque state.** It is declared per thread and
looks like a local variable, but it is not one: the 256 halves of a `matrix_a`
fragment are distributed across the 32 lanes' registers, 8 each, in a mapping
that is an implementation detail. All 32 lanes must reach every `load`, `mma`
and `store` **together**. Divergence around a `wmma` call is undefined
behaviour, not a slow path — not "the inactive lanes contribute nothing", not
"it serialises". A warp-uniform `if` that all 32 lanes agree on is fine; that
is not divergence. A per-lane one is not.

**The layout is a type parameter, and `ldm` is not.** `wmma::row_major` and
`wmma::col_major` are template arguments on the `a` and `b` fragments, fixed at
compile time, because they change which registers the load unit writes. The
leading dimension is a runtime *argument*, because it only affects addresses.
Getting this backwards is the most common first bug: `ldm` is the row stride of
the **whole matrix** the fragment is a window into, not 16.

**The accumulator has no layout in its type.** Look at the declaration again —
`wmma::fragment<wmma::accumulator, ...>` takes no layout parameter. An
accumulator is not a view of memory; it is only laid out at the instant it
lands in memory, so `store_matrix_sync` takes `wmma::mem_row_major` or
`wmma::mem_col_major` as an argument instead. (`load_matrix_sync` on an
accumulator, for a `D = A·B + C` with a real `C`, takes the same argument, for
the same reason.)

**`mma_sync(acc, fa, fb, acc)` is legal and is the normal shape.** The
destination and the addend are the same fragment; that is how you accumulate
down `K` without a separate copy.

## The fixed shape, and why every dimension here is a multiple of 16

`fragment<matrix_a, M, N, K, ...>` looks parameterised. It is not tunable. The
triples are a fixed, small set the silicon implements — for `__half` inputs
with a `float` accumulator, 16×16×16, 32×8×16 and 8×32×16 all compile on this
box; this puzzle uses 16×16×16. You cannot ask for 16×16×12 because `K`
happened to be 12, and the API says so at compile time rather than at runtime:
`wmma::fragment` is only *defined* for legal triples, so an illegal one is
`error: incomplete type ... is not allowed`. That error message is the shape
set being enforced by the type system, and it is the friendliest failure in
this entire puzzle — every other way of getting the shape wrong is silent.

Two separate constraints stack up here, and they are worth keeping apart
because only one of them is enforced.

**1. Coverage: the fragment is 16 wide because the instruction is.** A warp's
MMA covers a 16×16 tile of `C` and consumes a 16×16 slab of `A` and of `B`.
There is no partial fragment, no lane mask that makes one, no "16×16 but only
use 12 columns". If `K` is not a multiple of 16 then a `for (k = 0; k < K;
k += 16)` loop reads past the end of every row on its last trip. Measured here
with `K = 200`: the kernel launches clean, `compute-sanitizer --tool memcheck`
reports **0 errors** (the overrun lands inside the allocation for every row but
the last, and in `cudaMalloc`'s padding for that one), and the output is
garbage. There is no diagnostic. The only defence is the shape.

**2. Alignment: `load_matrix_sync` wants an aligned pointer and an aligned
`ldm`.** The API's contract is a 16-byte-aligned pointer for the tile origin
and an `ldm` that is a multiple of 8 for `__half` — 16 bytes either way — so
that every row of the fragment starts aligned too. `cudaMalloc` gives you 256
bytes of base alignment for free; what you have to preserve is `a + tile_m *
K + k`, and that stays aligned for every legal `tile_m` and `k` because `K` is
a multiple of 8.

Now the part that matters more than the rule. Measured on this box, holding
everything else fixed and moving `A`'s base pointer:

| offset from the aligned base | result |
|---|---|
| 0 B | correct |
| **2 B** (one `__half`) | **`cudaErrorMisalignedAddress`** — the launch fails |
| 4 B | correct |
| 8 B, 16 B | correct |

and separately, running `K = 192` with a padded row stride so `ldm` is 196 or
194 — not multiples of 8, straight violations of the documented contract — the
answer comes out **correct to 6e-07**, the same as the aligned case.

So this device enforces 4-byte alignment for this fragment shape and layout,
not the 16 the contract asks for, and ignores the `ldm` rule entirely. That is
the useful lesson, and it is the opposite of reassuring: *"I violated the
alignment contract and it worked"* is not evidence of anything. It means the
compiler happened to emit 32-bit loads for this fragment shape on this
architecture. Change the shape, the layout, the element type, the architecture
or the CUDA version and the same code can start faulting. Code to the contract;
you cannot test your way to it.

So `M = 320`, `N = 256`, `K = 192` are not a cop-out to dodge guard logic.
They are the shape of the problem the hardware will accept. **The general
lesson is that you pad the problem to the fragment, never the fragment to the
problem** — which is exactly what cuBLAS, CUTLASS and every real GEMM do with a
ragged edge: allocate to the next multiple of the tile, zero the pad, run the
full fragment grid, and ignore the extra outputs. Zeros contribute nothing to a
dot product, so a padded MMA is not an approximation; it is the same answer
with wasted flops.

Two consequences you should be able to state before you write code:

- A `k`-loop guard like `if (k + 16 <= K)` does not rescue a ragged `K`. It
  skips the tail entirely and silently computes the wrong product.
- A lane-level guard `if (my_col < N)` around a `wmma` call is worse than
  useless: it is the undefined behaviour from two sections ago.

Alignment as a general topic — `float4`, `__align__(16)`, what a misaligned
base actually costs, and why `reinterpret_cast` onto one is UB — is puzzle 35.
This puzzle is the case where the *shape* is not negotiable even though, as the
table above shows, some of the alignment around it quietly is.

## The problem

```
C = A · B        A: M×K  row-major  __half
                 B: K×N  row-major  __half
                 C: M×N  row-major  float          M=320  N=256  K=192
```

| | |
|---|---|
| warp tile grid | `M/16 × N/16` = **20 × 16** = 320 tiles — non-square, and 20 is not a power of two |
| steps down K | `K/16` = **12** `mma_sync` per warp |
| `GemmWmma` launch | `dim3(8, 10)` blocks of **128 threads** = 4 warps, laid out 2×2 over the tile grid |
| `GemmNaiveFp32` launch | `dim3(16, 20)` blocks of `dim3(16, 16)`, one thread per element of `C` |
| tolerance | `1e-4` for both — measured, see below |
| `C` | poisoned with `0xff` before each launch: a tile you never write is a `FAIL`, not a lucky zero |

The shape is deliberately awkward inside what the hardware allows. A kernel
that assumed a square tile grid, or that `K` was one fragment, produces a wrong
answer here rather than a lucky one.

### Two kernels, identical inputs

`GemmNaiveFp32` is the same product with one thread per output and a plain
`K`-loop on the general-purpose FP32 pipe. It is the correctness anchor and the
timing baseline, and it is deliberately not a straw man: `threadIdx.x` indexes
the column, so consecutive lanes read consecutive elements of a row of `B` and
the loads are coalesced.

The runner does something specific to make the comparison mean something. It
fills `A` and `B` with random floats, rounds them to `__half`, and then
converts them **straight back to float** before handing anything to anyone. So
the CPU reference, the FP32 kernel and the tensor-core kernel all multiply
bit-identical values. Nothing about the inputs differs between them. The only
divergence possible is the order and the width of the accumulation — which is
the point of the precision section below, and it is why the tolerances here are
three decades tighter than a half-precision matmul tolerance usually is.

## Precision: half in, float out, and why that is the standard recipe

The mixed-precision recipe is `__half` inputs with a `float` accumulator, and
the reason is arithmetic rather than convention. A `__half` carries 11
significant bits. The product of two of them needs at most 22, and a `float`
carries 24 — **so every individual product is exact**. All the error in a
half-precision GEMM lives in the accumulation, which is precisely where you buy
it back by accumulating in `float`. Accumulating in `half` instead is legal —
`fragment<accumulator, 16, 16, 16, __half>` compiles fine here — and a `K`-long
sum then starts losing small terms to a large running total after a few dozen
steps.

Measured on this box, worst relative deviation from the `double`-accumulated
CPU reference over all 81,920 outputs, using the same
`|got − want| / max(|want|, 1)` the runner thresholds:

| kernel | worst relative deviation | tolerance | headroom |
|---|---|---|---|
| `gemm_naive_fp32` | `4.35e-06` (index 64350) | `1e-4` | 23.0× |
| `gemm_wmma` | `2.89e-06` (index 1374) | `1e-4` | 34.6× |

Both kernels are deterministic; those digits did not move across ten
consecutive runs, and the runner prints them above the verdict lines every time
so you can check the table rather than believe it.

**The tensor-core kernel is the more accurate of the two.** That is not what
most people expect from the word "half", and it follows directly from the two
paragraphs above: with input quantisation removed by construction, the naive
kernel's 192 terms go through a single 192-long serial `float` chain with a
rounding at every step, while the tensor core's arrive as 12 chained fragment
MMAs, each of which sums 16 products in one hardware step. Whatever the MMA's
internal accumulation width is — which the API does not tell you and this
puzzle does not measure — it rounds at most 12 times where the naive kernel
rounds 192. Shallower chain, fewer roundings, better answer.

Each tolerance is the tightest power of ten clearing its measured deviation by
at least 10×. Both land on `1e-4`.

**What the round trip is worth, measured.** Delete it — reference the
*unrounded* floats, feed the kernels the rounded ones, which is what a
half-precision matmul benchmark normally does — and the same `gemm_wmma` output
deviates by **5.09e-03** instead of 2.89e-06. That is 1760× larger and a
three-decade looser tolerance, and every bit of it is `__float2half` on the way
in: the FP32 kernel run the same way (true float inputs, float reference) still
measures 4.28e-06, unchanged. So the number a loose half-GEMM tolerance is
absorbing is the *input conversion*, not the tensor core. Separating the two is
the only reason this puzzle can assert `1e-4` and mean it.

## Performance, measured on this box

Timings are best of 15 interleaved reps on an idle machine
(`make run P=33 MODE=solution`); the counters are from `make prof`. The two
sources are kept apart deliberately — under `ncu` the same kernels measure
31–42 µs and 15–17 µs, because the profiler's per-launch overhead is a
significant fraction of a kernel this short.

| kernel | best ms | GFLOP/s | warp instructions | Compute (SM) | Memory | waves |
|---|---|---|---|---|---|---|
| `GemmNaiveFp32` | 0.0246 | 1278 | 2,432,000 | 46–61 % | 46–61 % | 1.11 |
| `GemmWmma` | 0.0061 | 5159 | 141,760 | 5.7–6.1 % | 16–17 % | 0.14 |

The instruction counts and the wave counts are exact and identical on every
run. The two `SpeedOfLight` percentages are quoted as ranges because they are
throughput-over-elapsed and the elapsed side moves under instrumentation; the
ranges are from repeated `make prof` runs on this box.

Ratio **4.03×**, and the runner asserts a floor of 3.00× on it. Read the
right-hand columns before you read the ratio, because they say the 4× is a
*floor imposed by the problem size*, not the size of the hardware effect:

- `GemmWmma` issues **17× fewer instructions** than `GemmNaiveFp32` to compute
  the same product from the same values.
- `GemmNaiveFp32` is being measured around half of this device's throughput
  roofline over 1.11 waves of its 48 SMs — a reasonable kernel near its limit.
- `GemmWmma` is being measured at ~6 % of compute throughput over **0.14
  waves**. `ncu` says it outright: *"This kernel grid is too small to fill the
  available resources on this device."* 320 warps on 48 SMs is under seven
  warps per SM, and 80 blocks do not even cover the SMs once. What the clock is
  timing there is mostly launch and memory latency with a tensor-core
  instruction somewhere inside it.

So the tensor-core kernel wins by 4× *while running almost entirely idle*. The
honest statement of the result is that a 320×256×192 GEMM is too small to
measure a tensor core with, and it still wins by 4×. Question 4 below is how to
find out what the number becomes when the grid is big enough to mean
something.

The margin is a floor, not a description. On an idle box the ratio is
3.988×–4.040× over ten consecutive runs, which is tight; but this is a
shared-memory SoC where the GPU and the CPUs sit behind one LPDDR5X, and the
kernel with the least of its own latency to hide loses the most under
contention. Across 62 whole-run measurements — including two concurrent GPU
bandwidth hogs, twenty CPU memory streamers, and eight copies of the binary
racing each other — the range was 3.279× to 4.519×. `MARGIN = 3.0` sits below
that contended floor so the assert cannot lie about the machine; the measured
ratio is printed on the `PASS` line every run either way.

## What you write

Two regions in `skeletons/p33_tensor_cores/kernel.cu`. The launch geometry, the
constants and the warp-tile mapping are given.

### 1. `GemmNaiveFp32` (approx 5 lines)

`row` and `col` are computed for you. Bounds, an accumulator, the `K`-loop, the
store. Nothing here is about tensor cores; it exists so the other kernel has
something to be right about and something to beat.

### 2. `GemmWmma` (approx 10 lines)

`tile_m` and `tile_n` — the top-left corner of this warp's 16×16 tile of `C` —
are computed for you from `blockIdx` and the warp id, and both are uniform
across the warp.

Declare the three fragments, zero the accumulator, loop `k` from 0 to `K` in
steps of 16 loading one `a` and one `b` fragment and issuing one `mma_sync`,
then store once at the end.

The whole difficulty is in three expressions. For each of the three pointers,
write down on paper: **where does this tile start, and what is the row stride
of the array it lives in?**

- `A` is `M×K` row-major. This warp wants the 16×16 block at row `tile_m`,
  column `k`.
- `B` is `K×N` row-major. This warp wants the 16×16 block at row `k`, column
  `tile_n`.
- `C` is `M×N` row-major. This warp wants the 16×16 block at row `tile_m`,
  column `tile_n`.

One of those three has a different leading dimension from the other two. If you
pass 16 as `ldm` anywhere it will compile, run clean under `memcheck` (measured:
0 errors), and give a wrong answer — the load reads a contiguous 256-element
window instead of a strided tile.

Also: the accumulator must be zeroed. A fragment is uninitialised registers
until you `fill_fragment` it, and there is no accumulator-defaults-to-zero
rule.

## Running it

```
make run   P=33                  # your kernels — fails loudly on both
make run   P=33 MODE=solution    # reference
make check P=33 MODE=solution    # memcheck + racecheck + synccheck, all zero
make prof  P=33 MODE=solution    # tensor-pipe counters + SpeedOfLight
```

`make prof` is the instrument this puzzle is built around, and the check is by
inspection: every tensor counter is a large exact integer for `GemmWmma` and
**exactly zero** for `GemmNaiveFp32`, while the FP32 pipe counter says the
opposite. That is the proof the hardware ran, as distinct from the source
having said `wmma`.

```
sm__inst_executed_pipe_tensor_subpipe_hmma_op_hmma.sum   # warp-level HMMA
sm__inst_executed_pipe_tensor.sum                        # all tensor-pipe
sm__ops_path_tensor_op_hmma_src_fp16_dst_fp32.sum        # fp16 in, fp32 out
sm__pipe_tensor_cycles_active.sum                        # tensor pipe busy
sm__inst_executed_pipe_fma.sum                           # the FP32 control
smsp__inst_executed.sum                                  # total warp insts
launch__waves_per_multiprocessor                         # is the device full?
```

Note the metric name. `sm__inst_executed_pipe_tensor_op_hmma.sum`, which older
write-ups quote, **does not exist on this device** — the tensor pipe here is
split into `hmma`/`imma` subpipes and the name follows. `--query-metrics` is
the only reliable source; `dram__*` metrics are not available on GB10 either.

Predict the three tensor counters before you run it. You have everything you
need: the warp tile count, the steps down `K`, and `2·M·N·K`. Then reconcile
what you predicted for the HMMA counter against what it reports, because they
differ by an integer factor and that factor is a fact about the hardware you
cannot get from the `wmma` API.

> `ncu` needs permission to read the GPU's performance counters, which on this
> box is admin-only (`RmProfilingAdminOnly: 1`). If `make prof` reports
> `ERR_NVGPUCTRPERM`, run the same command under `sudo` — there is a sudoers
> rule for exactly `/usr/local/cuda/bin/ncu`.

## The puzzle

1. **The HMMA arithmetic.** 320 warp tiles × 12 steps = 3,840 `mma_sync` calls
   in the whole grid. The counter does not report 3,840. Work out the integer
   factor from the reported value, then find out where it comes from — and note
   that the PTX will not tell you: `nvcc -ptx` still shows a 16×16×16
   `wmma.mma.sync`, so the factor appears in `ptxas`, below the level any of
   this source can see. `cuobjdump -sass` on the built binary shows the actual
   machine instruction, and its mnemonic names its own shape. Which shape is
   it, and how many of them tile a 16×16×16?

2. **The ops counter.** `sm__ops_path_tensor_op_hmma_src_fp16_dst_fp32.sum`
   lands on an exact round number. Which one, and why is that a *stronger*
   correctness signal than the instruction count? (What would it read if your
   `K`-loop ran 11 times instead of 12, or if two warps had been given the same
   tile?)

3. **Cycles per instruction.** Divide `sm__pipe_tensor_cycles_active.sum` by
   the HMMA count. The answer is a small integer. Is that latency or issue
   cost, and which of the two would you need in order to predict this kernel's
   run time?

4. **The 4× is a floor — prove it.** The table above says `GemmWmma` runs at
   8 % of roofline over 0.14 waves. So make the problem big enough to matter:
   scale `M` and `N` up by 8× each (staying on multiples of 16) and re-measure
   both kernels. Predict first — which one degrades, and does the ratio go up,
   down, or stay put? Then explain the answer with `SpeedOfLight`, and say what
   the *next* bottleneck is once the grid is large enough to fill the machine.
   (Hint: count how many times this kernel reads each element of `A`. Every
   warp tile in a tile-row re-reads the whole 16×K strip. There is no shared
   memory in this kernel at all, and puzzles 28 and 32 are about the two things
   you would add next.)

5. **The divergence rule, tested.** The README says a per-lane branch around a
   `wmma` call is undefined behaviour. Write the tempting version — a guard
   like `if (tile_n + lane < N)` around the `store_matrix_sync` — and observe
   that it compiles, that `compute-sanitizer` is perfectly happy with it, and
   that this tells you nothing. Then say what the guard would have to be
   instead, and why the real fix is not a guard at all.

6. **The two constraints, separated.** The alignment section above reports
   three measurements; reproduce them and then say which constraint each one
   tests.
   - `K = 200`: `ldm` is still a multiple of 8, so alignment is untouched, and
     `memcheck` still reports 0 errors — yet the answer is garbage. Which trip
     of the `k`-loop did it, and why did `memcheck` stay quiet? (The answer is
     about what `cudaMalloc` hands back, not about `wmma`.)
   - Base pointer `+ 1` element vs `+ 2` elements: one faults, one does not.
     Work out what the SASS is loading — `cuobjdump -sass` — and reconcile that
     with the API contract asking for 16.
   - `ldm = 196` at `K = 192`: contract violated, answer correct. Say what you
     are allowed to conclude from that. (Nothing.)

   Then say what you would actually do for a `320 × 250 × 190` GEMM, and cost
   it: how much padding, how many wasted flops, and how that compares to the
   4× above. Puzzle 35 is where alignment becomes a general story rather than
   one API's requirement.
