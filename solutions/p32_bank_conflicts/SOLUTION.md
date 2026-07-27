# Puzzle 32 — Solution: The Array Shape Is an Access Pattern

Every number below was measured on this box: NVIDIA GB10, `sm_121`, 48 SMs,
1536 threads/SM, 102400 B shared/SM (1024 B per block reserved by the driver),
32 banks × 4 B, 25.17 MB L2, CUDA 13.0. Nothing here is quoted from anywhere.

The whole puzzle is one number and its derivation. `TransposeNaive` reports
**~971 600 shared-load bank conflicts**; the other two report **~30**, which is
zero to within what the counter can resolve. That is a factor of about
3 × 10⁴, from changing an array's declared width by one.

## 1. What the compiler produced

`-Xptxas -v`, from the build the runner uses:

| | `TransposeNaive` | `TransposePadded` | `TransposeSwizzle` |
|---|---|---|---|
| registers/thread | 30 | 30 | 30 |
| static shared/block | 4096 B | **4224 B** | 4096 B |
| barriers | 1 | 1 | 1 |
| stack frame | 0 B | 0 B | 0 B |
| spill stores / loads | 0 / 0 | 0 / 0 | 0 / 0 |

Identical register footprint, identical barrier count, no spills anywhere. The
padded kernel's 128 extra bytes are the *entire* difference between it and the
naive one, and `cudaOccupancyMaxActiveBlocksPerMultiprocessor` says all three
get **6 blocks/SM** at 256 threads — the thread budget (1536 / 256 = 6) binds
every one of them, and the shared budget would have allowed 20, 19 and 20
respectively. So the pad costs nothing in occupancy here, and none of the
timing below is an occupancy effect in disguise.

## 2. The bank maps

Bank of a shared word is `(word offset) mod 32`. Write out where logical
`[r][c]` lives in each of the three tiles, and the entire puzzle is visible
before anything runs.

### `tile[32][32]` — offset `32r + c`, bank `= c`

```
         c=0   c=1   c=2   c=3  ...  c=31
 r=0       0     1     2     3  ...    31
 r=1       0     1     2     3  ...    31
 r=2       0     1     2     3  ...    31
 ...
 r=31      0     1     2     3  ...    31
```

The row index vanishes: `32r mod 32 = 0` for every `r`. A tile element's bank
**is** its column.

- Row access (`tile[fixed][threadIdx.x]`, the tile *load*): lane L → bank L.
  32 banks, 1 wavefront. Perfect.
- Column access (`tile[threadIdx.x][fixed]`, the transposed *read*): every lane
  → the same bank, at 32 different addresses. **32-way conflict, 32
  wavefronts.**

### `tile[32][33]` — offset `33r + c`, bank `= (r + c) mod 32`

```
         c=0   c=1   c=2   c=3  ...  c=31
 r=0       0     1     2     3  ...    31
 r=1       1     2     3     4  ...     0
 r=2       2     3     4     5  ...     1
 ...
 r=31     31     0     1     2  ...    30
```

`33 ≡ 1 (mod 32)`, so each successive row is the previous one **rotated by one
bank**. Column `c` now reads banks `c, c+1, ..., c+31` mod 32 — all 32, exactly
once. And the row access is still `bank = (r + c) mod 32` with `r` fixed and
`c` varying, which is also all 32. Both directions conflict-free.

### `tile[32][32]` with `[r][c ^ r]` — offset `32r + (c ^ r)`, bank `= c ^ r`

```
         c=0   c=1   c=2   c=3   c=4  ...
 r=0       0     1     2     3     4
 r=1       1     0     3     2     5
 r=2       2     3     0     1     6
 r=3       3     2     1     0     7
 ...
```

Fix `r`: `c -> c ^ r` is a bijection on `0..31` (XOR with a constant is an
involution, hence a permutation), so a row still contains every bank once and
the tile still holds every value exactly once — nothing is lost, which is the
correctness half. Fix `c`: `r -> c ^ r` is the *same* bijection with the roles
swapped, so a column also contains every bank once. That symmetry is why XOR
works and, say, `[r][(c + r) mod 32]` (which is the pad's rotation, done by
hand) works too.

**All three of these are Latin squares of order 32 — except the first one,
which is not.** That is the whole criterion: you need each of the 32 banks to
appear exactly once in every row *and* every column of the bank map. `tile[32]
[32]` has each bank appearing 32 times in its column and never elsewhere,
which is as far from a Latin square as an order-32 grid can get.

The reader finds logical `[a][b]` at physical `[a][b ^ a]` because XOR is its
own inverse; the writer and reader apply literally the same expression, which
is what makes the swizzle cheap to get right compared to, say, a rotation that
needs a modulus in one direction and a different one in the other.

## 3. The measured counters

`make prof P=32 MODE=solution`, one representative run:

| metric | `Naive` | `Padded` | `Swizzle` |
|---|---|---|---|
| `..._bank_conflicts_..._mem_shared_op_ld.sum` | **971 591** | **32** | **29** |
| `..._bank_conflicts_..._mem_shared_op_st.sum` | **0** | **0** | **0** |
| `..._wavefronts_mem_shared_op_ld.sum` | 1 003 591 | 32 032 | 32 029 |
| `..._wavefronts_mem_shared_op_st.sum` | 32 000 | 32 000 | 32 000 |
| `smsp__inst_executed_op_shared_ld.sum` | 32 512 | 32 512 | 32 512 |
| `smsp__inst_executed_op_shared_st.sum` | 32 768 | 32 768 | 32 768 |

The first thing to notice is the identity that holds in every column:

```
conflicts = wavefronts − requests,   with requests = 32 000 in all three
```

`1 003 591 − 971 591 = 32 000`. `32 032 − 32 = 32 000`. `32 029 − 29 = 32 000`.
The conflict counter is not measuring "how many collisions happened"; it is
measuring **extra data-path cycles over the minimum**, and the minimum for a
32-bit warp-wide access is one cycle. Keep that definition — section 6 breaks
the naive reading of it with a `float4`.

### Where 32 000 comes from

The grid is 32 × 32 = 1024 blocks of 32 × 8 = 256 threads = 8 warps, and each
warp runs `TILE / BLOCK_ROWS = 4` iterations of each loop. So the structural
instruction count on each side is

```
1024 blocks × 8 warps × 4 iterations = 32 768
```

which is exactly `smsp__inst_executed_op_shared_st.sum`. On the load side the
counter reads 32 512, i.e. 256 fewer, because ptxas branched around some of the
dead ones; `smsp__inst_executed_op_shared_ld_pred_off_all.sum` measures **512**
exactly, so 32 512 − 512 = **32 000 instructions with at least one live lane**.

And 768 = 256 + 512 is predicted exactly. The store loop's guard `ty + j < cols`
is *warp-uniform* — `ty + j = blockIdx.x * 32 + threadIdx.y + j` does not
depend on `threadIdx.x`. It kills a whole warp-instruction whenever
`blockIdx.x = 31` and `threadIdx.y + j ≥ 8`, which is 24 of the 32
(warp, iteration) pairs, in each of the 32 blocks with `blockIdx.x = 31`:

```
24 × 32 = 768 warp-instructions with no live lane
32 768 − 768 = 32 000 live ones
```

The load loop is the mirror image with `blockIdx.y` and `rows`, and gives the
same 768.

### Where 968 000 comes from

Each live load instruction reads `tile[threadIdx.x][threadIdx.y + j]`. Its
live-lane count is set by the *other* guard, `tx = blockIdx.y * 32 +
threadIdx.x < rows`, which is lane-varying:

```
blockIdx.y ≤ 30  ->  32 live lanes
blockIdx.y = 31  ->  lanes 0..7 only, 8 live lanes   (1000 = 31×32 + 8)
```

All live lanes of one such instruction hit **one** bank at distinct addresses,
so an instruction with `A` live lanes costs `A` wavefronts. Live instructions
per `blockIdx.y` value: 31 full `blockIdx.x` columns × 32 + one edge column × 8
= 1000. So

```
wavefronts = 31 × 1000 × 32   +   1 × 1000 × 8
           =   992 000        +      8 000     =  1 000 000
conflicts  = 1 000 000 − 32 000                =    968 000
```

1 000 000 is not a coincidence: a fully serialised access spends one bank cycle
per element, and the matrix has `rows × cols = 1 000 000` elements. **The naive
kernel's shared read costs one cycle per float, which is the definition of
having thrown the entire bank array away.**

Measured across six runs: 971 204, 971 591, 971 896, 972 098, 973 296, 973 461
— the model is low by 0.33 % to 0.56 %, and the excess moves run to run on an
identical binary with identical input.

### The residual, and what it is

The gap is not the model being wrong; it is the counter picking up L1 client
arbitration. The same run reports, on the same three kernels:

| | `Naive` | `Padded` | `Swizzle` |
|---|---|---|---|
| `l1tex__data_bank_conflicts_type_arbitration.sum` | 9 297 | 4 755 | 4 180 |
| `..._bank_conflicts_pipe_lsu_mem_gds_op_ld.sum` (global) | 2 740 | 2 541 | 3 217 |
| `..._bank_conflicts_pipe_lsu_mem_gds_op_st.sum` (global) | 679 | 1 406 | 2 134 |

Thousands of arbitration cycles exist in *all three* kernels, including the two
with a perfect shared access pattern: the L1TEX data path is shared between the
global and shared clients, and when both want the same bank in the same cycle
someone waits. A little of that lands on the shared-load counter. That is why
`Padded` and `Swizzle` read 16, 18, 20, 27, 28, 29, 32, 32, 75, 86 across ten
profiled runs instead of a clean 0 — the value is nondeterministic and is
0.05 %–0.27 % of their 32 000 requests.

A control confirms it is not a property of the access pattern. The same two
kernels at **1024 × 1024**, where every block is full and there are no guards
at all, so the model is exact:

| | model | measured (4 runs) |
|---|---|---|
| naive: `wavefronts` | 1 048 576 | 1 053 275 / 1 053 790 / 1 053 920 / 1 053 983 |
| naive: `conflicts` | **1 015 808** | 1 020 507 / 1 021 022 / 1 021 152 / 1 021 215 |
| padded: `conflicts` | **0** | 12 / 30 / 33 / 35 |
| both: `inst_executed_ld` | 32 768 | 32 768 (exact, no predication) |

Same 0.5 % excess on the conflicted kernel, same tens-of-counts on the
conflict-free one. Report the naive figure as **≈ 9.7 × 10⁵, model 968 000**,
and the other two as **0 to within the counter's noise floor** — not as literal
zeros, because on this box they are not.

### Why the store side is exactly 0, in all three

`..._mem_shared_op_st.sum = 0` and `wavefronts_st = 32 000 = requests`, exactly,
every run, for every kernel. All three write `tile[threadIdx.y + j][threadIdx.x]`
(or its swizzled column), so the varying index across a warp is the *column*,
and the column is the bank. The tile load was already optimal in the naive
kernel and neither remedy had to touch it — which is the general shape of the
thing: a bank conflict is a property of **one access**, not of a kernel or of an
array. The same `tile[32][32]` is perfect one line earlier and catastrophic one
line later.

## 4. What it costs, measured

```
# transpose: 1000 x 1000 -> 1000 x 1000, 32x32 tiles through shared memory
# grid: 32 x 32 blocks of 32 x 8 threads, 4 rows of the tile per thread
# edges: last tile column is 8 wide, last tile row is 8 tall
# working set: in + out = 8.00 MB, L2 = 25.17 MB
# timing: fastest of 15 interleaved reps of (3 warmup + 20 timed iterations)
# kernel               best ms     eff GB/s
  TransposeNaive        0.0123        651.2
  TransposePadded       0.0073       1092.9
  TransposeSwizzle      0.0073       1091.0
PASS padding_beats_conflicts (naive / padded = 1.678x >= 1.25x, ...)
PASS swizzle_matches_padding (swizzle / padded = 1.002x, within [0.90, 1.10], ...)
```

A 32× increase in shared data-path cycles buys a 1.68× increase in wall clock,
not a 32× one, and the reason is in the `SpeedOfLight` rows:

| | `Naive` | `Padded` | `Swizzle` |
|---|---|---|---|
| `L1/TEX Cache Throughput` | **50.00 %** | 14.38 % | 16.29 % |
| `Memory Throughput` | 40.05 % | 21.45 % | 20.78 % |
| `L2 Cache Throughput` | 19.87 % | 21.45 % | 20.78 % |
| `Compute (SM) Throughput` | 10.44 % | 11.28 % | 10.92 % |
| `SM Active Cycles` | 46 997 | 42 572 | 37 575 |

`L2 Cache Throughput` is the same in all three, to a percent — the global
traffic really is identical, as designed. What moves is `L1/TEX Cache
Throughput`, from ~15 % to 50 %, because that is where the shared-memory data
path lives. The naive kernel is not 32× slower because the conflict serialises
a stage that was never the only cost: the kernel still has to move 8 MB through
L2, and the extra ~9.7 × 10⁵ bank cycles spread over 48 SMs overlap with that
traffic. They stop being free at the point where the L1 data path becomes the
busier of the two, which is exactly what the 50 % row is saying.

### The distribution behind `MARGIN`

The effect is exceptionally reproducible on an idle box — **ten consecutive
runs: 1.660, 1.678, 1.685, 1.664, 1.641, 1.660, 1.683, 1.664, 1.663, 1.663**,
a spread of 2.7 %. But GB10 is a shared-memory SoC, and the *conflict-free*
kernel is the one closer to a bandwidth limit, so contention hurts it more and
squeezes the ratio down. Measured deliberately against a loaded machine:

| machine state | measured ratios | min |
|---|---|---|
| idle (10 runs) | 1.641 – 1.685 | 1.641 |
| 16 × CPU `memcpy` streamers, 128 MB each | 1.620, 1.642, 1.646, 1.650, 1.657 | 1.620 |
| 3 copies of this binary racing | 1.540, 1.613, 1.628 | 1.540 |
| concurrent GPU bandwidth hog (320 MB stream) | 1.403, 1.411, 1.414, 1.594, 1.635 | 1.403 |
| 4 copies + GPU hog | 1.404, 1.411, 1.411, 1.480 | 1.404 |
| **GPU hog + 16 CPU streamers** | 1.356, 1.367, 1.379, 1.395, 1.397, 1.404 | **1.356** |

Note the CPU-only row barely moves the number: the working set is 8 MB and
lives in L2, so LPDDR5X pressure from the CPU side is nearly irrelevant here.
It takes a *GPU* competing for L2 to compress the ratio, and even then only to
1.36.

`MARGIN = 1.25` sits below the whole observed distribution with room, rather
than at half the effect. The measured ratio is printed on the `PASS` line every
run, so the size of the effect is always visible; the assert's only job is to
never lie.

`swizzle / padded` was within **±1.5 %** on every run, idle or contended — they
are interleaved rep for rep and see the same machine — so the two-sided band
`[0.90, 1.10]` is about seven times the observed spread.

## 5. The regime: why 1000 × 1000

Same three kernels, only the matrix size changed, best of 10 interleaved reps:

| n | in + out | naive ms | padded ms | swizzle ms | ratio | padded GB/s |
|---|---|---|---|---|---|---|
| 500 | 2.0 MB | 0.0059 | 0.0040 | 0.0040 | 1.455 | 495 |
| **1000** | **8.0 MB** | **0.0123** | **0.0073** | **0.0073** | **1.682** | **1093** |
| 1400 | 15.7 MB | 0.0226 | 0.0160 | 0.0159 | 1.409 | 979 |
| 2000 | 32.0 MB | 0.0995 | 0.0988 | 0.0984 | **1.007** | 324 |
| 2828 | 64.0 MB | 0.3141 | 0.3137 | 0.3137 | 1.001 | 204 |
| 4000 | 128.0 MB | 0.5982 | 0.5990 | 0.5988 | 0.999 | 214 |
| 5656 | 255.9 MB | 1.3080 | 1.3064 | 1.3090 | 1.001 | 196 |
| 8000 | 512.0 MB | 2.4358 | 2.4379 | 2.4389 | 0.999 | 210 |

Two walls, one on each side.

**Below ~4 MB there is not enough work.** At n = 500 the grid is 16 × 16 = 256
blocks against 48 SMs × 6 = 288 concurrent block slots — less than one full
wave, so the measurement is mostly ramp-up and the ratio sags to 1.46.

**At 32 MB the L2 is gone and the effect vanishes completely — 1.007.** Past
the cliff the padded kernel drops from 1093 GB/s to 324 and then settles at
~205 GB/s, which is this box's DRAM roofline for a read-plus-write stream
(~273 GB/s peak). Both kernels are then waiting on DRAM, the extra 9.7 × 10⁵
bank cycles hide entirely underneath that wait, and a 32-way bank conflict
costs **nothing at all**.

That is the honest boundary on this puzzle's lesson: bank conflicts matter
exactly when shared memory is on the critical path, and a DRAM-bound kernel has
a different critical path. It is also why the shipped size is 8 MB — comfortably
interior to the 25.17 MB L2, and past the ramp-up regime.

## 6. Access granularity: what a `float4` does to the counters

Seven single-instruction probe kernels, 4096 blocks × 128 threads = 16 384
warps, each warp issuing exactly one shared load (except the last, which issues
four). The staging array is 256 floats.

| kernel | access, lane `L` | `inst` | wavefronts | wf/inst | conflicts | conf/inst |
|---|---|---|---|---|---|---|
| `Ld1Stride1` | 32-bit, word `L` | 16 384 | 16 594 | 1.01 | 210 | ~0 |
| `Ld1Stride2` | 32-bit, word `2L` | 16 384 | 33 937 | 2.07 | 17 553 | 1.07 |
| `Ld1Stride4` | 32-bit, word `4L` | 16 384 | 68 423 | 4.18 | 52 039 | 3.18 |
| `Ld1Stride32` | 32-bit, word `32L mod 256` | 16 384 | 134 527 | 8.21 | 118 143 | 7.21 |
| `Ld4Stride1` | **128-bit**, `float4` slot `L` | 16 384 | 68 605 | **4.19** | 3 069 | **~0.19** |
| `Ld4Stride2` | **128-bit**, `float4` slot `2L` | 16 384 | 134 544 | 8.21 | 69 008 | 4.21 |
| `Ld1x4Stride1` | 4 × 32-bit, words `L, L+32, L+64, L+96` | 65 536 | 65 900 | 1.01 | 364 | ~0 |

Four things fall out, and three of them contradict the obvious model.

**A `float4` load takes 4 wavefronts and reports 0 conflicts.** `Ld4Stride1` is
laid out perfectly — lane `L` reads `float4` slot `L`, so lane `L` touches banks
`4L .. 4L+3` and every bank is touched by exactly one lane. It still costs
**4.19 wavefronts per instruction**, because 32 lanes × 16 B = 512 B and a bank
cycle delivers 32 × 4 B = 128 B. The data simply does not fit in one cycle. And
the conflict counter reports ~0 anyway. So the counter's baseline is **the
minimum wavefronts for that access width**, not 1: `conflicts = wavefronts −
requests` where a 128-bit request costs 4. If you read the conflict counter
alone you will conclude a perfect `float4` access is free of cost; it is free of
*conflicts*, which is a different claim. Always read the wavefront counter
beside it.

**Four separate 32-bit loads cost the same data-path cycles and 4× the
instructions.** `Ld1x4Stride1` moves the identical 512 B per warp as
`Ld4Stride1`: 65 900 wavefronts against 68 605, i.e. ~4 per warp either way. The
`float4` version does it in **one** instruction instead of four. That is what
128-bit shared access buys — issue slots and address arithmetic, not bank
cycles. It is also why `LDS.128` is worth reaching for in a tight loop and why
it is not a way out of a bank conflict.

**A conflicted `float4` is not as bad as the lane model predicts.**
`Ld4Stride2` has lane `L` reading `float4` slot `2L`, so lanes `L` and `L + 4`
want the same four banks — the naive reading is a 4-way conflict on top of the
4× width, i.e. 16 wavefronts. Measured: **8.21**, exactly 2× the minimum, with
`conflicts/inst = 4.21` (and `134 544 − 69 008 = 65 536 = 4 × 16 384`, the
identity again). The hardware is splitting a 128-bit warp access into **8-lane
groups of 128 B**: lanes 0–7 of `Ld4Stride1` already cover all 32 banks, so each
group is one cycle and the warp is 4. At stride 2, lanes 0–7 cover banks
`{0–3, 8–11, 16–19, 24–27}` twice over, so each group takes 2 cycles and the
warp takes 8. The conflict degree of a wide access is a property of an 8-lane
group, not of the warp.

**Broadcast beats the stride model.** `Ld1Stride32` looks like the worst case in
this document — every lane on bank 0 — and should be 32 wavefronts. It measures
**8.21**. With a 256-float array, `(32L) mod 256` produces only **8 distinct
addresses**, each wanted by 4 lanes, and the hardware broadcasts an identical
address to all lanes that want it for free. 8 distinct addresses in one bank = 8
cycles. Conflicts count **distinct addresses per bank**, and no amount of
reasoning about strides substitutes for asking that question.

(The transpose's column read has no broadcast to save it: `tile[threadIdx.x][b]`
gives 32 lanes 32 genuinely different addresses in one bank, which is why it
pays the full 32.)

## 7. Race-freedom

Required for this puzzle, and clean:

```
========= ERROR SUMMARY: 0 errors                                  (memcheck)
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
========= ERROR SUMMARY: 0 errors                                  (synccheck)
```

all three kernels, full run, under `make check P=32 MODE=solution`.

The barrier is load-bearing and racecheck is the instrument that says so. Built
in scratchpad only, the identical kernels with `__syncthreads()` deleted:

```
========= Error: Race reported between Write access at TransposeNaive+0x3f0 (line 51)
=========     and Read access at TransposeNaive+0x430 (line 56)  [229376 hazards]
=========     and Read access at TransposeNaive+0x440 (line 56)  [253952 hazards]
=========     and Read access at TransposeNaive+0x450 (line 56)  [253952 hazards]
=========     and Read access at TransposeNaive+0x4c0 (line 56)  [253952 hazards]
  ... 4 such records per kernel (one per unrolled store), each naming the
  ... same 4 unrolled loads: 16 write/read pairs per kernel, 12 records total
========= RACECHECK SUMMARY: 13 hazards displayed (12 errors, 1 warning)
```

Two things worth taking from that run:

- **The hazard counts are enormous** — a quarter of a million per write/read
  pair — because every warp reads a tile row that another warp wrote. This is
  not a subtle race in a corner case; it is the kernel's entire data flow.
- **`synccheck` reports 0 errors on the broken build.** `synccheck` finds
  *illegal* barriers — divergent `__syncthreads()`, mismatched arrivals — not
  *missing* ones. A kernel with no barrier at all is perfectly legal as far as
  it is concerned. `racecheck` is the tool for this failure, and running one
  without the other would have passed the broken kernel.

The broken build also fails correctness (`FAIL transpose_naive at index 19: got
0 want 0.49422`) — but not deterministically, and not at the same index run to
run, which is exactly the reason to ask the sanitizer rather than the output.

## 8. Pad or swizzle?

On this box, at this tile size, the timing table cannot tell them apart
(1.002×) and the occupancy table cannot either (6 blocks/SM both ways, thread-
budget limited). So the choice is made on properties the runner does not
measure:

- **The pad's cost scales, the swizzle's does not.** `+1` column on a `T × T`
  tile is `4T` bytes on a `4T²`-byte array — 3.1 % here, and it stays 3.1 %
  at every `T`. Not free, and it grows with the tile.
- **The pad breaks 16-byte alignment; the swizzle does not.** In `tile[32][33]`
  row `r` starts at word `33r`, and `33r ≡ 0 (mod 4)` only when `r ≡ 0 (mod 4)`.
  So three rows in four cannot be read as `float4` at all. The swizzled tile is
  still `[32][32]`, every row starts at a 128-byte boundary, and section 6's
  `LDS.128` path stays available. For a kernel that wants both conflict-freedom
  and vectorised shared access — which is most tiled matmuls — this is decisive
  and the pad is simply not an option.
- **The swizzle costs an XOR**, which is one integer instruction on an index
  that is already being computed, and it is the same expression on both sides
  because XOR is an involution.

The pad's advantage is that it is one character and impossible to get wrong.
The swizzle's is that it is free in space and keeps the tile a power of two.
Both are Latin squares over the bank map, which is the actual requirement;
everything else is engineering.

## 9. Measurement caveats

1. **Do not take `Duration` from `ncu`.** The profiler reports 27.36 / 25.34 /
   26.21 µs against the runner's 12.3 / 7.3 / 7.3 µs — inflated ~2.5× and
   unevenly, which would compress the 1.68× to 1.08×. `prof.mk` passes `-c 3`
   and `--kill yes` so the runner's timing loop never executes under the
   profiler. Counters from `ncu`, timings from `make run`.

2. **The conflict counters are not perfectly deterministic on this box.** The
   naive kernel varies by ~0.2 % run to run and the conflict-free kernels
   report tens rather than zero (section 3). The instruction counters
   (`smsp__inst_executed_op_shared_{ld,st}.sum`, and the `pred_off_all`
   variants) *are* exact and identical every run, so any claim that needs an
   exact number should be built on those.

3. **`dram__*` counters do not exist on GB10**, as puzzles 30 and 31 found, so
   the DRAM-roofline reading of section 5's table rests on the behavioural
   evidence of the sweep (the ~205 GB/s plateau against a ~273 GB/s nominal
   peak) rather than on a counter.

4. **The timing assert is measuring a shared SoC.** Section 4's table is the
   whole story; the short version is that CPU-side pressure barely touches this
   puzzle because the working set is L2-resident, and GPU-side pressure is what
   compresses the ratio.

## 10. Specification corrections

Recorded because they are findings about this hardware, not silent fixes:

| specified | shipped / measured | why |
|---|---|---|
| padded and swizzle conflict counters = **0** | **16–86**, nondeterministic | L1TEX arbitration between the global and shared clients leaks into the shared-load counter; confirmed by `..._type_arbitration.sum` being in the thousands for all three kernels, and by an unguarded 1024×1024 control where the model is exactly 0 and the counter still reads 12–35 (section 3) |
| `MARGIN` "1.05 precedent, may go higher" | **1.25** | the measured floor over ~30 runs including deliberately contended ones is 1.356; the idle mode is 1.66 with a 2.7 % spread (section 4) |
