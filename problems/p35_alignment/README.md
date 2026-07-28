# Puzzle 35 — Alignment: The Sector Is the Unit

Every memory puzzle so far has been about *where* the bytes are: coalescing in
puzzle 24, banks in 32, L2 residency in 30. This one is about a smaller
question that turns out to decide all of them — **how many 32-byte sectors does
one warp-wide access touch, and what does the answer cost?**

Three kernels compute the identical arithmetic on identical data. One asks for
its bytes four at a time, one sixteen at a time, and one asks for exactly the
same bytes one float to the right. They move **the same 18 MB** and they are
7 % and 15 % apart, and the sector counters explain all of it before you run the
clock — including which of those two numbers is a fact about the hardware and
which is a fact about the block size you happened to pick.

Then a fourth thing happens: the runner deliberately does the one cast this
puzzle exists to warn you about, and the CUDA context dies.

## The alignment model

Global memory on this device is not addressed in bytes by the hardware that
moves it. Three granularities stack up:

```
byte          0        32        64        96       128       160  ...
sector        |----0----|----1----|----2----|----3----|----4----|
cache line    |------------------ 0 ------------------|--------- 1 ---------
float index   0..7      8..15     16..23    24..31    32..39
```

- A **sector** is 32 bytes = 8 floats. It is the unit the L1 requests from L2
  and the unit `ncu` counts.
- A **cache line** is 128 bytes = 4 sectors. It is the unit L1 tags.
- A **request** is one warp-wide access — one `LDG` or `STG` instruction, all
  32 lanes together. One request is broken into however many sectors its lanes'
  addresses fall in.

Now do the arithmetic for a warp reading consecutive floats:

| the 32 lanes read | bytes touched | sectors | why |
|---|---|---|---|
| `a[32w .. 32w+31]`, `a` 128-B aligned | 128, starting at a line | **4** | the minimum: 128 / 32 |
| `a[32w+1 .. 32w+32]` | 128, starting 4 B in | **5** | the 128 bytes now straddle a fifth sector |
| `a[4·(32w) .. ]`, one float4 per lane | 512 | **16** | four times the bytes, so four times the sectors |
| `a[4L]` for lane `L`, four separate reads | 512 per read set | **16 per read** | stride-4: same 16 sectors, four times |

Two things follow immediately, and they are the whole puzzle:

**A misaligned start costs a sector, not a factor.** 128 contiguous bytes need
`ceil(128/32) = 4` sectors if they begin on a 32-byte boundary and 5 if they do
not — anywhere in the 31 bytes after it, so a shift of 4 bytes and a shift of
28 bytes cost exactly the same. That is 25 % more sectors *on that one access*,
which is nothing like 25 % more time unless every access in the kernel is that
one.

**Vectorising costs zero sectors and saves instructions.** A `float4` load moves
128 bits per lane in one instruction. The bytes are the same bytes; nothing
about the sector count changes. What changes is that one instruction now asks
for four times as many of them. Whether that is worth anything depends entirely
on whether the kernel was short of instructions or short of sectors — and the
measurement below says which.

### Why a vector load wants a 16-byte-aligned address

The hardware has one 128-bit global load, `LDG.E.128`, and it requires its
address to be 16-byte aligned. There is no misaligned variant. This is not a
performance rule with a slow path behind it; it is the instruction's contract,
and violating it raises a fault.

That constraint propagates up into C++ as the alignment of the type:
`sizeof(float4) == 16` and `alignof(float4) == 16`, because `float4` is declared
`__align__(16)` in `vector_types.h`. So `reinterpret_cast<float4*>(p)` is a
promise that `p` is 16-byte aligned, and the compiler believes you.

### `__align__(16)` and the struct that cannot be vector-loaded

The classic version of this is a 3-vector:

```cpp
struct Vec3 { float x, y, z; };            // sizeof 12, alignof 4
struct __align__(16) Vec3Pad { float x, y, z; };   // sizeof 16, alignof 16
```

`Vec3` is 12 bytes, so an array of them puts element 0 at offset 0, element 1
at offset 12, element 2 at offset 24. Only every fourth element lands on a
16-byte boundary, so there is no way to load one with a 128-bit instruction —
and the compiler will not try. `__align__(16)` raises both the alignment *and*
the size to 16 (size is always a multiple of alignment, so the padding is
forced), every element lands on a boundary, and the array becomes
vector-loadable at the cost of 25 % more memory and 25 % more bandwidth for the
padding you never read.

That trade — a quarter more bytes for a quarter of the instructions — is the
same trade this puzzle measures, priced in the other direction. Which one wins
is the same question: were you short of sectors, or short of instructions?

### What `cudaMalloc` guarantees

`cudaMalloc` returns memory "suitably aligned for any variable type", which in
practice on this box means **every allocation comes back on a 256-byte
boundary** — the runner prints `a % 256`, `b % 256` and `out % 256` every run
and all three are 0. 256 is 16 sixteens and 8 sectors, so a `cudaMalloc` base is
aligned for anything CUDA has.

What it does not guarantee is anything about `base + k`. That is your
arithmetic, and it is where alignment is lost.

## The problem

```
out[i] = s * a[i] + b[i]        s = 0.75, n = 1,500,003
```

| | |
|---|---|
| `a` | `n + 1` floats (the misaligned kernel reads `a[i+1]`, so `a[n]` exists) |
| `b`, `out` | `n` floats; every element of `out` is written by every kernel |
| tolerance | `1e-6` — measured deviation is **0**, see below |
| block | 256 threads |
| scalar grid | `cdiv(n, 256)` = **5860** blocks, last one 99 threads live |
| vec4 grid | `cdiv(n/4, 256)` = **1465** blocks, covering 375,000 float4 |
| tail | `n % 4` = **3** elements the float4 view does not reach |

Every digit of `n` is doing something:

- **18.0 MB against a 25.17 MB L2.** The whole working set stays resident
  between reps, so what is being timed is the L1↔L2 sector path and not DRAM.
  This is the only regime where any of this is visible; past the L2 all three
  kernels go DRAM-bound at ~250 GB/s and converge, because DRAM sees the same
  bytes in all three cases. The sweep is below.
- **`n % 4 == 3`.** The float4 view covers 1,500,000 elements and three are left
  over. They still have to be computed.
- **`n % 256 == 99`, `n % 32 == 3`.** The last block runs 99 of 256 threads and
  the last warp of it runs 3 of 32. Guards are exercised at both granularities.

`out` is poisoned with `0xff` — a NaN bit pattern — before every launch, so an
element that is never written is a `FAIL` and not a lucky zero. That is exactly
how a missing float4 tail presents.

### Tolerance, and why it is not zero

saxpy is two flops. `nvcc` contracts `s * a[i] + b[i]` into a single `FFMA`
(you can see it in the SASS), which rounds once; the CPU reference computes the
same fused form in `double` and rounds once on the way back to `float`, so it is
the more accurate of the two rather than a differently-sloppy one. Measured on
this box, worst relative deviation over all 1,500,003 outputs, using the
`|got − want| / max(|want|, 1)` the runner thresholds:

| kernel | worst relative deviation | tolerance |
|---|---|---|
| `saxpy_scalar` | **0** | `1e-6` |
| `saxpy_vec4` | **0** | `1e-6` |
| `saxpy_misaligned` | **0** | `1e-6` |

Every output is bit-identical to the reference, in all three kernels, every run.
The tolerance is `1e-6` anyway and that is deliberate: bit-exactness here is a
consequence of a compiler default (`-fmad=true`), not a property of the problem.
Written as two roundings instead of one — `float t = s * a[i]; out[i] = t +
b[i];` — the deviation would be about one ulp, 6e-08 on values of this size, and
that kernel would still be a correct saxpy. `1e-6` clears an ulp by more than a
decade and still catches every real indexing bug, all of which produce O(1)
errors.

## What you write

Three regions in `skeletons/p35_alignment/kernel.cu`. Nothing is given except
the signatures and the commentary.

### 1. `SaxpyScalar` (approx 3 lines)

One element per thread, plain indexing, one guard. The baseline, and not a straw
man — consecutive lanes read consecutive floats, so it is perfectly coalesced
and its request is the 4-sector minimum.

### 2. `SaxpyVec4` (approx 12 lines)

One `float4` per thread. `reinterpret_cast<const float4*>(a)` on the two inputs,
`reinterpret_cast<float4*>(out)` on the output, index by float4 element, and do
the four multiply-adds componentwise.

Then the part that is actually the exercise: `n` is not a multiple of 4, so the
float4 view covers `4 * (n / 4)` elements and the last `n % 4` are outside it.
They are still outputs. Somebody has to compute them with scalar code, and the
launch is only `n / 4` threads wide, so you have to decide which threads do it.

Two questions to answer before writing anything:

- Which thread indices are in range for the float4 body, and which are in range
  for the tail? They are not the same set, and one thread can be in both.
- What happens if you guard the tail with the float4 body's bound instead of its
  own? (Poison the output and find out; the runner already does.)

### 3. `SaxpyMisaligned` (approx 3 lines)

`out[i] = s * a[i + 1] + b[i]`. Character for character the same as
`SaxpyScalar` apart from one `+ 1`. `a` holds `n + 1` floats so nothing runs off
the end — `memcheck` confirms 0 errors — and the arithmetic, the instruction
count and the coalescing are all unchanged.

Work out, before you run it, what changes about lane 0 of warp `w`.

## Measured on this box

`make run P=35 MODE=solution`, best of 201 interleaved reps, on an idle machine,
in `SaxpyScalar`'s sector-roof mode (see below). All three kernels move the same
18.00 MB.

| kernel | best ms | eff GB/s | sectors/request (ld) | requests (ld) | warp instructions |
|---|---|---|---|---|---|
| `SaxpyVec4` | 0.01027 | **1753** | 16.00 | 23,440 | 363,319 |
| `SaxpyScalar` | 0.01100 | 1636 | 4.00 | 93,752 | 984,428 |
| `SaxpyMisaligned` | 0.01181 | 1524 | **4.50** | 93,752 | 984,428 |

and the ratios the runner asserts, as medians of **paired** per-rep measurements
(see "How this is measured" below):

| relationship | measured | asserted floor |
|---|---|---|
| `misaligned / vec4` | 1.102 – 1.364, median 1.15 | **1.05** |
| `scalar / vec4` | 1.031 – 1.355, median 1.07 | **1.00** |
| `misaligned / scalar` | 0.997 – 1.083 | *reported, not asserted* |

The floor on `scalar / vec4` is deliberately 1.00 and not something inside its
measured range: question 3 shows the *size* of that gap is a property of the
block size rather than of vectorisation, and at 128 threads per block it is
exactly 1.000. What is a property of vectorisation is the sign — the same bytes
in a quarter of the requests cannot cost more — and that holds at every block
size measured. `misaligned / scalar` does not even have a safe sign: on a run
where `SaxpyScalar` is latency-bound it can come out just under 1.0, which is
why it is reported and not asserted.

Ranges are over the 70 of 105 consecutive whole runs that passed the runner's
quiet-window gate; every one of them passed both asserts. The wide top ends, and
the third row's straddling of 1.0, are one phenomenon and question 5 is about
it: `SaxpyScalar` sits exactly on the boundary between sector-bound and
latency-bound, and its best time is bimodal — **0.0108–0.0111 ms** in 28 of
those 70 runs and **0.0112–0.0143 ms** in the other 42. In the first mode
`misaligned / scalar` measures 1.053–1.083, which is the 13/12 the sector model
predicts; in the second it measures 0.997–1.045, because a kernel that already
has slack in the sector path does not pay for an extra sector. `SaxpyVec4` has
no second mode — 0.0102–0.0104 ms on every gated-in run — which is precisely why
the two ratios measured *against it* are the two that get asserted.

### The counters, and what they each explain

`make prof P=35 MODE=solution` reports, for the three correctness launches:

| | `SaxpyScalar` | `SaxpyVec4` | `SaxpyMisaligned` |
|---|---|---|---|
| global **load sectors** | 375,002 | **375,002** | **421,877** |
| global **load requests** | 93,752 | **23,440** | 93,752 |
| sectors / request (ld) | 4.00 | 16.00 | **4.50** |
| global store sectors | 187,501 | 187,501 | 187,501 |
| global store requests | 46,876 | 11,720 | 46,876 |
| warp instructions | 984,428 | 363,319 | 984,428 |

Read that table twice, once per column pair.

**Scalar vs Vec4.** The load sector counts are *identical to the sector* —
375,002 both times. Vectorising did not move one byte less. What it did was ask
for those bytes in 23,440 requests instead of 93,752, exactly a quarter, and in
363,319 warp instructions instead of 984,428. So the entire vec4 effect lives in
the instruction and request columns, and the measured 7 % is what a quarter of
the requests is worth to a kernel whose real constraint is elsewhere.

**Scalar vs Misaligned.** The instruction counts are identical — 984,428 both
times, to the instruction — and so are the request counts. The *only* number
that moves in the entire table is load sectors, 375,002 → 421,877. That is
**1.125×**, and it is exactly the prediction: the `a` load goes 4 → 5 sectors,
the `b` load stays at 4, so loads go `(5+4)/(4+4) = 1.125` and the reported
average sectors-per-request goes 4.00 → **4.50**. Including the untouched
stores, total sectors go `(421,877 + 187,501) / (375,002 + 187,501)` = **1.083 =
13/12**, which is the number to hold against the clock.

Predict every number in it before running `ncu`. All of them are derivable:
1,500,003 elements is 46,876 warps (the last one partial), each issuing two
loads and one store.

### Reconciling the counters with the clock

| effect | sector prediction | measured time |
|---|---|---|
| misaligned vs scalar | 1.083 (13/12 total sectors) | **1.053 – 1.083**, in `SaxpyScalar`'s roof mode |
| misaligned vs vec4 | 1.083 sectors × the request effect | 1.102 – 1.364 |
| vec4 vs scalar | 1.000 (identical sectors) | 1.070 – 1.355 |

The alignment row lands on its prediction. The vectorisation row does not,
because sectors are not the only thing that costs — but note the *sign* of the
disagreement and how small it is. This kernel is close enough to the sector roof
that removing three quarters of its instructions buys 7 %.

**This is the result the puzzle exists for, and it contradicts the folklore.**
The usual story about `float4` is that it is how you saturate memory bandwidth.
It is not: bandwidth is bytes, vectorising does not change bytes, and the
counter says so to the sector. What `float4` saturates is the *instruction*
side, and you only notice when instructions were the constraint. Here they
nearly are not, so it is worth per cent. Misalignment does not change the byte
count either — it changes how many 32-byte pieces the same bytes arrive in — and
*that* is what costs, in every block-size configuration, which is why it is the
claim this puzzle stands on.

### Speed of Light, and why the profiler's durations look wrong

`ncu` reports all three kernels at 55–59 µs and 15–17 % of memory throughput,
against the 10–12 µs and ~1750 GB/s `make run` measures. That is not a
contradiction, it is the profiler flushing the caches between replays: under
`ncu` the 18 MB is not L2-resident any more, so all three kernels are reading
DRAM and all three converge. **Counters from `ncu`, timings from `make run`** —
and the fact that the effect vanishes when the working set leaves L2 is itself
one of the measurements below.

`dram__*` metrics do not exist on GB10 at all, so there is no way to ask this
device how many bytes reached memory. Every number above is measured at the
L1TEX/L2 boundary, which is where the effect lives anyway.

## The `__ldg` question, measured

`SaxpyScalar`'s inputs are `const float*`. Does the compiler use the read-only
data path? Measured with `cuobjdump -sass` on the shipped binary:

| source | SASS for the two input loads | time |
|---|---|---|
| `const float* a, const float* b, float* out` | `LDG.E` | 0.01110 ms |
| `__ldg(a + i)`, `__ldg(b + i)` | `LDG.E.CONSTANT` | 0.01090 ms |
| `const float* __restrict__` on all three | `LDG.E.CONSTANT` | 0.01120 ms |

(best `SaxpyScalar` time over twelve gated-in runs each, all three variants
linked against the same unmodified runner; the 3 % spread between them is
smaller than the run-to-run spread of any one of them.)

So: **`const` alone is not enough.** `const float* a` promises only that *this
pointer* will not be used to write. It says nothing about `out`, which is a
`float*` into the same address space and could perfectly well alias `a`. Without
that guarantee the compiler cannot use the non-coherent read-only path, and
emits plain `LDG.E`.

Either `__ldg` (which selects the instruction directly) or `__restrict__` on
*all three* pointers (which gives the compiler the no-aliasing fact it was
missing) produces `LDG.E.CONSTANT`, and the difference in measured time is
**nothing**: the three variants land within 3 % of each other, which is inside
the run-to-run spread of any one of them, and `__restrict__` measures *slower*
than plain `const` on this sample. This kernel reads each byte exactly once, so
the read-only cache buys reuse it does not have.

One asymmetry worth noticing, and it is measured: with `__restrict__` the
compiler upgraded `SaxpyScalar` and `SaxpyVec4` but left `SaxpyMisaligned` on
plain `LDG.E`. `__ldg` upgraded all three. `__restrict__` is a fact you give the
optimiser and it may decline to act on; `__ldg` is an instruction you select.

## How this is measured, and why it is not the earlier puzzles' estimator

Puzzles 32 and 33 assert on a ratio of *minima*: time every kernel many times
interleaved, keep each one's fastest rep, divide. That is the right estimator
when the effect is a large multiple — 1.6× for bank conflicts, 4× for tensor
cores — because a machine-state artefact would have to be enormous to matter.

At 7–15 % it is the wrong estimator, and this was measured rather than assumed.
With a GPU bandwidth hog and eight CPU memory streamers running, the
`misaligned / scalar` **ratio of minima** scattered over **0.93× – 1.48×** while
the median of the **paired** per-rep ratios stayed inside **1.094× – 1.120×** on
the same runs. So this runner keeps every sample, forms each ratio *inside* a rep — the
three kernels a few hundred microseconds apart, seeing the same machine — and
takes the median.

Two further pieces of measurement discipline, both forced by this box:

**The SM clock has to be woken, and cannot be read.** GB10 idles at 208 MHz
against a 3003 MHz maximum. `nvidia-smi` reports `clocks.sm` as 208 MHz and
pstate P8 *even at 91 % utilisation drawing 64 W*, so it is useless here. The
runner measures the clock itself, by spinning one warp on `clock64()` for a
known cycle count and timing it on the wall clock — it reports **2510–2572 MHz**
during the timed section, printed every run. That spin is also the clock ramp:
one warp draws almost no power, so it wakes the DVFS controller without heating
the part. A *heavy* warm-up is worse than none — 2 seconds of back-to-back saxpy
before timing made the results bimodal, because it drives the GPU into a
power-limited state where `SaxpyScalar` becomes issue-bound and the alignment
effect disappears into its slack.

**The runner checks that it got a measurement before asserting one.** This is a
shared-memory SoC: one LPDDR5X behind the GPU and the CPUs both. When something
else takes a large share of it, all three kernels stop being limited by their own
sector traffic and the ratios collapse toward 1.0 — which is *true*, not noisy:
under an external bottleneck the extra sector genuinely costs nothing. No
estimator recovers an effect that is not present. So before it asserts anything
the runner establishes that it **got a measurement**, with two checks that are
deliberately different in kind:

- It asks the driver, through NVML, which other processes held a CUDA context on
  this GPU while the section was being timed — sampled before and after, and
  unioned. This is causal: it names the competing PIDs and their device memory,
  and it fires whether or not the contention happened to show up in the timings.
- It holds `SaxpyVec4`'s quiet window to **1600 GB/s**, 91 % of the 1748–1759
  GB/s it reaches on a settled idle box. This is the only check that sees a CPU
  process on the other side of this SoC's LPDDR5X: such a process holds no CUDA
  context and appears in no GPU-side accounting, but it is what actually
  flattens these ratios.

That gives the timing section three outcomes and no silent one. A valid run
whose ratios clear their margins prints `PASS`. A valid run whose ratios do not
prints `FAIL misalignment_costs_sectors` or `FAIL vec4_never_slower` — the claim
did not hold on a quiet machine. A run that was **not a measurement** prints
`FAIL measurement_invalid` with the evidence, and exits non-zero without
evaluating either assert. It never says your kernel is wrong, and it never exits
0 either: a run that proves nothing is not a run that passed.

The floor is on the conservative side, and that is a deliberate choice about
which mistake to make. `MARGIN_SPAN` is 1.05 and it has very little headroom in
its low tail — over 25 runs on an idle box `misaligned / vec4` measured
1.086–1.283, but a run at 914 GB/s came in at 1.0495 and one under 40 CPU memory
streamer threads at 547 GB/s came in at 1.0257, with `scalar / vec4` at 1.0020
against a margin of 1.00. The margin only reliably holds in the settled mode, so
the floor demands the settled mode. Runs that are merely *unsettled* — this box
takes minutes to recover from a heavy memory load, and measured 1065–1505 GB/s
throughout one such window with nothing else on the GPU — are refused too, even
though their ratios were fine. Refusing a measurable run costs a re-run;
admitting an unmeasurable one costs a `FAIL` that blames `SaxpyMisaligned` for
the machine.

What the floor cannot do is stand in for asking the driver, and the numbers say
why: under twenty CPU streamer threads ten runs measured 1602–1736 GB/s and
cleared the floor while the machine was under heavy memory load. An absolute
rate is a weak proxy for who else is on the machine. Asking the driver is not a
proxy at all.

## Undefined behaviour, and what it does here

The last thing the runner does, every uninstrumented run, is this:

```cpp
const float4 v = reinterpret_cast<const float4*>(a + 1)[i];
```

`a` is a `cudaMalloc` base, so 256-byte aligned. `a + 1` is four bytes past it.
The cast asserts that a `float4` object — `alignof` 16 — lives there. It does
not. By the C++ object model this is undefined behaviour full stop, before any
hardware is consulted: you have created a pointer to an object that does not
exist at that address, and the standard has nothing further to say about your
program.

What *this* hardware does about it, measured:

```
launch -> cudaSuccess, synchronise -> cudaErrorMisalignedAddress
```

with `compute-sanitizer --tool memcheck` reporting

```
Invalid __global__ read of size 16 bytes
Access to 0x... is misaligned
```

Four properties of that, all of which matter more than the error name:

1. **The launch succeeds.** The address is computed per lane at execution time,
   so nothing on the host side can see it coming. `cudaGetLastError()`
   immediately after the launch returns `cudaSuccess`. The fault surfaces at the
   next synchronisation, which may be far away from the kernel that caused it.
2. **The error is sticky.** Every subsequent call on that context returns
   `cudaErrorMisalignedAddress` forever, including `cudaFree`. The context is
   gone. That is why the probe is the runner's last act, after every real buffer
   has already been freed, and why it does not free its own — there is nothing
   that can be done with a context in this state, and pretending otherwise would
   be the real error.
3. **The offset does not have to be small or odd.** Measured by shifting the
   base: offsets of 0, 4 and 8 *floats* (0, 16, 32 bytes) work; 1, 2, 3 and 5
   floats all fault. Anything that is not a multiple of 16 bytes faults, and 8
   bytes is just as fatal as 4.
4. **Compare puzzle 33.** A `wmma::load_matrix_sync` on a `__half` matrix
   tolerated a 4-byte offset on this same device and only faulted at 2 bytes,
   because `ptxas` had chosen 32-bit loads for that fragment shape. Same
   hardware, same kind of contract violation, different outcome — because the
   outcome depends on which instruction the compiler happened to pick. That is
   precisely why "I violated the alignment contract and it worked" is not
   evidence of anything.

`make check` runs with the probe disabled (`P35_SKIP_UB_PROBE=1`), because
`memcheck` is right to report it and a harness should not fail its own sanitizer
run for a fault it caused on purpose.

## Running it

```
make run   P=35                  # your kernels — fails loudly on all three
make run   P=35 MODE=solution    # reference
make check P=35 MODE=solution    # memcheck + racecheck + synccheck, all zero
make prof  P=35 MODE=solution    # sector and request counters + SpeedOfLight
```

`make check` is not a formality here. The two places an off-by-one hides in this
puzzle are the float4 kernel's scalar tail and the misaligned kernel's read of
`a[n]`, and `memcheck` is the instrument that finds both — a tail that walks one
element too far, or an `a[i + 1]` against an array of `n`, are both reads past
the end of an allocation and both silent without it. Measured on the reference:
**0 errors** from all three tools.

> `ncu` needs permission to read the GPU's performance counters, which on this
> box is admin-only (`RmProfilingAdminOnly: 1`). If `make prof` reports
> `ERR_NVGPUCTRPERM`, run the same command under `sudo` — there is a sudoers
> rule for exactly `/usr/local/cuda/bin/ncu`.

## The puzzle

1. **The sector arithmetic, from the source.** Before running `ncu`, write down
   all six load numbers in the counter table: sectors and requests for each of
   the three kernels. You have everything — `n`, the warp size, the sector size,
   and two loads plus one store per element. Then explain the `002` in 375,002,
   and why `SaxpyMisaligned` reports 421,877 rather than the 421,884 the clean
   `46,876 × 9` arithmetic gives.

2. **Where the misalignment penalty is *not*.** The `b` load and the `out` store
   in `SaxpyMisaligned` are unshifted and cost 4 sectors each, exactly as in
   `SaxpyScalar`. So the shifted `a` load is one of three accesses and the total
   sector inflation is 13/12, not 5/4. Now: what would you have to change to
   make it 5/4? And what does the 13/12 become for a kernel with one input and
   one output instead of two and one — say `out[i] = 2 * a[i + 1]`? Predict,
   then measure it (the kernel is three lines).

3. **The block-size result, which is not in the table above.** Sweep the block
   size with everything else fixed. Measured on this box, `scalar / vec4` at
   n = 1,500,003:

   | threads/block | 64 | 96 | 128 | 192 | **256** | 384 | 512 | 768 | 1024 |
   |---|---|---|---|---|---|---|---|---|---|
   | `SaxpyVec4` ms | .01020 | .01020 | .01024 | .01026 | **.01027** | .01025 | .01021 | .01024 | .01026 |
   | `SaxpyScalar` ms | .01776 | .01143 | .01028 | .01048 | **.01100** | .01142 | .01192 | .01224 | .01637 |
   | `scalar / vec4` | 1.738 | 1.120 | **1.000** | 1.030 | **1.080** | 1.120 | 1.179 | 1.190 | 1.597 |
   | `misaligned / vec4` | 1.737 | 1.179 | 1.119 | 1.139 | **1.159** | 1.175 | 1.189 | 1.199 | 1.598 |
   | `misaligned / scalar` | 0.995 | 1.053 | **1.119** | 1.105 | **1.074** | 1.049 | 1.009 | 1.008 | 1.000 |

   `SaxpyVec4` measures **0.01020–0.01027 ms at every one of those block sizes**
   — it does not care at all. Only the scalar kernels move, and
   `misaligned / scalar` falls monotonically from 1.119 to 1.000 as they get
   slower. So at 128 threads per block the scalar kernel catches the vectorised
   one exactly and the alignment penalty is at its largest; at 1024 it is 60 %
   slower and the alignment penalty is *zero*. Both halves of that need the same
   one-sentence explanation, and it is the same sentence as question 5's.
   (Puzzle 31 is the other half of it.)

   Note what this does to the honesty of the headline number. The 7 % vec4 win
   is measured at 256 threads per block and it is 0 % at 128, so the README's
   7 % is true *of this configuration* and is not a property of vectorisation.
   That is exactly why the runner asserts only `scalar / vec4 >= 1.00` and not a
   number inside the measured range. The misalignment penalty, by contrast, is
   nonzero at every block size from 96 to 768 and is largest exactly where the
   baseline is fastest — which is the right shape for a claim about the memory
   system, and why *that* one gets a real floor.

4. **The L2 cliff.** Sweep `n` and watch the effect disappear. Measured here,
   effective GB/s for `SaxpyScalar` and the `misaligned / scalar` ratio:

   | working set | 3 MB | 12 MB | 15 MB | **18 MB** | 21 MB | 24 MB | 48 MB | 96 MB |
   |---|---|---|---|---|---|---|---|---|
   | scalar GB/s | 743 | 1465 | 1474 | **1606** | 602 | 569 | 300 | 246 |
   | `misaligned / vec4` | 1.02 | 1.01 | 1.25 | **1.17** | 1.04 | 1.04 | 1.01 | 1.00 |
   | `misaligned / scalar` | 1.00 | 1.00 | 1.01 | **1.05** | 1.02 | 1.00 | 1.04 | 1.04 |

   Two collapses, at both ends, for two different reasons. Past ~21 MB — against
   a 25.17 MB L2, so *before* the working set nominally exceeds it — everything
   falls off a cliff to 570–600 GB/s and then settles DRAM-bound at ~250 GB/s,
   and the extra *sector* stops mattering to `SaxpyVec4` because DRAM sees the
   same *bytes*. (The `misaligned / scalar` column staying at 1.04 out there,
   rather than going to 1.00, is a separate small effect; explain it.) Below
   ~15 MB the kernel is too short to measure — and the reason is not the L2.
   Find it: launch a no-op kernel on the same grid and time it. On this box 5860
   blocks of 256 threads take **5.8 µs** to dispatch and do nothing, against the
   10.7 µs the real kernel takes. What does that do to a 3 MB measurement, and
   why does it not invalidate the 18 MB one?

   Note also where 18 MB sits: 21 MB is only 83 % of the L2 and has already
   fallen off the cliff. Work out what else is competing for that cache.

5. **The two kernels that are the same speed.** At 256 threads per block,
   `SaxpyScalar` runs at 1636 GB/s and `SaxpyVec4` at 1753 — 7 % apart, from a
   4× difference in instruction count. At 128 threads they are equal.
   `SaxpyMisaligned` is 15 % behind `SaxpyVec4` and never catches up at any
   block size. Say, in terms of the two columns of the counter table, which
   resource each of the three kernels is short of, and why exactly one of the
   three has slack. Then predict what happens to all three if you double the
   arithmetic per element (`out[i] = s*a[i]*a[i] + b[i]`), and check.

6. **The tail, deleted.** Remove the scalar tail from `SaxpyVec4` and run
   `make run` and then `make check`. Which one tells you, and what exactly does
   each one say? Then make the opposite mistake — guard the tail with `i < nv`
   instead of its own bound — and answer the same question. One of those four
   results is silence, and knowing which is the point of the exercise.

7. **`__align__(16)`, costed.** You have an array of 10 million `struct Vec3 {
   float x, y, z; }` and a kernel that reads all three components of each. Write
   down: the bytes moved and the sectors touched per warp as-is; the same after
   `__align__(16)`; and the instruction counts for both. Then say which one you
   would ship, and what measurement would change your mind. (Both answers are
   defensible. The number that decides it is in the counter table above.)

8. **The one that faults, generalised.** The runner's probe uses `a + 1`.
   Measured here, `a + 2` and `a + 3` fault too, and `a + 4` does not. Now
   suppose you are writing a library function that takes a `const float*` and
   wants to use `float4` loads. You cannot control the caller's pointer. Write
   the prologue: how do you detect the misalignment, what do you do about the
   elements before the first aligned boundary, and how much of the win is left?
   (This is what every real vectorised memcpy does, and it is why they are
   longer than you expect.)
