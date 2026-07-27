# Puzzle 32 — Bank Conflicts: The Array Shape Is an Access Pattern

Puzzle 30 was a metric that says the opposite of the truth. Puzzle 31 was a
metric that is exactly right and answers the wrong question. This one is
simpler and meaner: a kernel that is correct, coalesced on both sides, has no
divergence, no spills and 100 % occupancy — and throws away a third of its
time because a two-dimensional array has the wrong width.

The fix is one character. The point is being able to see, from the source
alone, that it is needed.

## Shared memory is 32 banks wide

Shared memory is not a flat buffer. It is **32 independent banks**, each 4
bytes wide, interleaved by word:

```
byte address   0    4    8   12  ...  124  128  132  ...
word index     0    1    2    3  ...   31   32   33  ...
bank           0    1    2    3  ...   31    0    1  ...
```

So `bank(addr) = (addr / 4) mod 32`, and consecutive `float`s land on
consecutive banks and wrap every 32.

A bank serves **one address per cycle**. That gives exactly three cases for a
warp-wide access:

| the 32 lanes want | cost | name |
|---|---|---|
| 32 addresses on 32 different banks | 1 cycle | conflict-free |
| `N` different addresses in some one bank | `N` cycles | `N`-way conflict |
| the *same* address, any number of lanes | 1 cycle | broadcast |

That last row is the exception worth remembering: banks conflict on
**addresses**, not on lanes. Thirty-two lanes reading one address is free.
Thirty-two lanes reading thirty-two *different* addresses that happen to share
a bank is thirty-two times the cost.

The unit the hardware and the profiler both count is the **wavefront**: one
cycle of shared-memory data-path work. A conflict-free warp-wide 32-bit access
is 1 wavefront. An `N`-way conflict is `N`. And the counter this puzzle is
named after reports the *difference* — the wavefronts you paid over the
minimum — so `conflicts = wavefronts − requests`.

## Why a column of `tile[32][32]` is the worst case there is

C stores `tile[32][32]` row-major, so `tile[r][c]` is at word offset
`r * 32 + c`, and

```
bank(tile[r][c]) = (r * 32 + c) mod 32 = c mod 32 = c
```

The row index **falls out entirely**. A tile element's bank is its column and
nothing else. Now look at the two ways a warp can walk that tile, with
`blockDim.x == 32` so a warp is one row of the block and `threadIdx.x` is the
lane:

```
tile[fixed][threadIdx.x]        lane L -> bank L         32 banks, 1 wavefront
tile[threadIdx.x][fixed]        lane L -> bank `fixed`   1 bank, 32 wavefronts
```

A row read is perfect. A **column read puts all 32 lanes on one bank**, at 32
different addresses, and the hardware serialises it 32 ways.

And a transpose has to read a column. That is what a transpose *is*.

## The problem

Transpose a `rows × cols` matrix out of place: `out[c][r] = in[r][c]`, with
`in` row-major `1000 × 1000` and `out` row-major `1000 × 1000`.

The naive one-liner — `out[x * rows + y] = in[y * cols + x]` with no shared
memory at all — makes one of the two global accesses stride by a whole row,
which costs you 32 sectors per warp instead of 4. The standard cure is to move
a **32 × 32 tile** through shared memory: read the tile from `in` row-major,
write it to `out` row-major, and let the transpose happen inside the tile,
where the hardware is not sensitive to coalescing.

All three kernels here do exactly that, with identical, optimal global
behaviour. The only thing that differs between them is the shape of the shared
tile and how it is indexed — so anything the timing table shows is a
shared-memory effect and nothing else.

| | |
|---|---|
| `in` | `rows * cols` floats, read-only, row-major |
| `out` | `cols * rows` floats, **every one written**, row-major |
| tolerance | **exactly 0** |
| block | `dim3(32, 8)` = 256 threads, so a warp is one row of the block |
| grid | `dim3(cdiv(cols, 32), cdiv(rows, 32))` = 32 × 32 blocks |
| per thread | `32 / 8 = 4` rows of the tile, strided by `blockDim.y` |
| guards | `1000 = 31 * 32 + 8`, so the last tile row and column are 8 wide |

The tolerance is zero because it should be. A transpose does no arithmetic —
every output is a bit-exact copy of one input — so there is nothing to
reassociate and nothing to round. `out` is poisoned with `0xff` (a NaN bit
pattern) before every launch, so an element you never write is a `FAIL`, not a
lucky zero.

The guards are not decoration either. 1000 is not a multiple of 32: the blocks
along the right and bottom edges have 8 valid rows or columns out of 32, and
the two loops need *different* guards, because the store walks `out`, whose
shape is the transpose of `in`'s. Get one of them wrong and you write outside
an array; `make check` will tell you exactly where.

## The three kernels

The shared declaration and the four index variables are given to you in
`skeletons/p32_bank_conflicts/kernel.cu`. You write the two loops and the
barrier between them, in each of the three.

### 1. `TransposeNaive` (approx 8 lines)

`__shared__ float tile[32][32]`. Load the tile from `in` with `threadIdx.x`
walking the fast axis, barrier, then store to `out` with `threadIdx.x` walking
the fast axis of `out` — which means reading the tile *down a column*. This is
the kernel the section above describes, written the way you would write it
without thinking about banks. It is correct. Measure it anyway.

### 2. `TransposePadded` (approx 8 lines)

Character for character the same body, against `__shared__ float tile[32][33]`.
The extra column is never read and never written; it exists only to change the
arithmetic that maps `[r][c]` to a bank. Work out the new mapping before you
run it:

- what is `bank(tile[r][c])` when the row stride is 33 words instead of 32?
- what does that make of a column, `r = 0..31` at fixed `c`?
- and does the row access — the one that was already perfect — survive?

That last question is the one people forget. A remedy that fixes the read by
breaking the write has bought nothing.

### 3. `TransposeSwizzle` (approx 8 lines)

`__shared__ float tile[32][32]` again — **no padding at all** — but logical
element `[r][c]` is stored at physical `[r][c ^ r]`. Both loops apply the
transform, and they have to agree.

Three things to satisfy yourself of, in order:

1. For fixed `r`, is `c -> c ^ r` a permutation of `0..31`? (If it is not, the
   tile loses data and the kernel is simply wrong.)
2. What is `bank(tile[r][c ^ r])`, and what happens to it when a warp fixes `c`
   and varies `r` — the column case?
3. Where does the reader find logical `[a][b]`, given that the writer put it at
   `[a][b ^ a]`?

Padding costs 128 bytes per block and one wasted column. The swizzle costs an
XOR. Both should land in the same place, and the runner asserts that they do.

## Measuring it

```
make run   P=32                  # your kernels — fails loudly
make run   P=32 MODE=solution    # reference
make check P=32 MODE=solution    # memcheck + racecheck + synccheck, all zero
make prof  P=32 MODE=solution    # bank-conflict counters + SpeedOfLight
```

`make prof` is the instrument this puzzle is built around. It reports, per
kernel:

```
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum   # load-side conflicts
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum   # store-side conflicts
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum       # what they are extra TO
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
smsp__inst_executed_op_shared_ld.sum                       # how many LDS were issued
smsp__inst_executed_op_shared_st.sum
```

Before you run it, **predict the load-side conflict count for
`TransposeNaive`**. You have everything you need: the grid, the block shape,
how many `LDS` instructions each warp issues, how many lanes are live in an
edge block, and the conflict degree of a column read. It is an integer, it is
reproducible to within about half a percent, and getting it right is the point
of the exercise. Then check the store-side counter and explain why it says what
it says for all three kernels.

Race-freedom is a **required** result for this puzzle, not an optional one. The
barrier between the tile load and the transposed store is load-bearing: a warp
reads tile rows that other warps wrote. Leave it out and the kernel still
produces the right answer most of the time on this hardware, which is precisely
why you should ask `racecheck` instead of asking the output.

> `ncu` needs permission to read the GPU's performance counters, which on this
> box is admin-only (`RmProfilingAdminOnly: 1`). If `make prof` reports
> `ERR_NVGPUCTRPERM`, run the same command under `sudo` — there is a sudoers
> rule for exactly `/usr/local/cuda/bin/ncu`.

## The puzzle

Predict the counters, then run it, then sit with these:

1. **The arithmetic.** Reconstruct `TransposeNaive`'s conflict count from the
   source: blocks × warps × `LDS` per warp × conflict degree, with the edge
   blocks handled separately because they do not have 32 live lanes. You should
   land within a fraction of a percent. Whatever gap is left over — where does
   it come from, and is it deterministic? (Run `make prof` twice.)

2. **Why the store side is already zero, in all three.** Both remedies target
   the load. Neither one *needed* to target the store. Say exactly why, in
   terms of which index varies across a warp in each of the two loops.

3. **What the pad costs.** 33 floats per row instead of 32 is 128 extra bytes
   per block. Ask `cudaOccupancyMaxActiveBlocksPerMultiprocessor` whether that
   changed the number of blocks per SM here; the answer on this box is probably
   not the one the question implies, and the reason is in puzzle 31. So find
   the *other* cost. Hint: write down the word offset at which row `r` of
   `tile[32][33]` begins, and ask whether `&tile[r][0]` is 16-byte aligned. Now
   re-read question 4. That is the argument for the swizzle, and it has nothing
   to do with either speed or occupancy.

4. **Access granularity — the measurement this puzzle owes you.** Everything
   above assumes 32-bit accesses: one lane, one word, one bank. A `float4`
   shared load is 128 bits per lane, so one lane touches **four consecutive
   banks** and a warp asks for 512 bytes when a bank cycle can deliver 128.

   - How many wavefronts *must* one warp-wide `float4` load take, before any
     conflict is even possible?
   - So what should the conflict counter report for a `float4` load that is
     laid out perfectly — 0, or 3 per instruction? Predict, then measure.
   - Compare it against moving the same 512 bytes as four separate
     conflict-free 32-bit loads. Which costs fewer data-path cycles, and which
     costs fewer instructions? They are not the same question.

   Write a small kernel and profile it. `solutions/p32_bank_conflicts/SOLUTION.md`
   has the measured table, including the case where a `float4` access *is*
   conflicted and the conflict degree turns out not to be what the naive model
   predicts.

5. **The regime, again.** The working set here is 8 MB against a 25 MB L2, and
   that is deliberate. Predict what happens to the naive/padded ratio when the
   matrix no longer fits in L2. Then predict what happens when it is far too
   *small*. Both ends collapse, for different reasons, and the sweep is in
   `SOLUTION.md`.

When you have answers — including a predicted conflict count you wrote down
*before* running `ncu` — check them against
`solutions/p32_bank_conflicts/SOLUTION.md`, which carries every number measured
on this box, the derivation, the bank maps for all three tile shapes, and the
`float4` experiment.
