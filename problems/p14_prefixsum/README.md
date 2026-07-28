# Puzzle 14 — Prefix sum

New idea: **a scan**, and the case where **one barrier per step is not enough.**

A reduction (puzzle 12) collapses `n` values to one. A **scan** keeps all `n`
and gives each one the running total up to and including itself:

```
out[i] = a[base] + a[base+1] + ... + a[i]
```

where `base` is the first element of `i`'s block. Blocks cannot see each other,
so the running total restarts at every block boundary — this kernel is the
per-block half of a full-array scan, and the reference is segmented to match.
(The other half is adding each block's total into every later block, which
needs a second pass; CUB's `BlockScan` in puzzle 27 is this same per-block
operation with the shuffle-level details hidden.)

## Hillis–Steele

The algorithm is one line, repeated with a doubling offset:

```
off = 1:   cache[t] += cache[t-1]     for t >= 1
off = 2:   cache[t] += cache[t-2]     for t >= 2
off = 4:   cache[t] += cache[t-4]     for t >= 4
...
off = 128: cache[t] += cache[t-128]   for t >= 128
```

After the step with offset `off`, every slot holds the sum of itself and the
`off` slots before it — so the reach doubles every step, and eight steps
(`log2(256)`) cover the whole block. Threads with `t < off` have nothing to add
and add zero, rather than sitting out: same trick as the idle lanes in puzzle
12, and it keeps every thread on the same path to the barrier.

This is not the cheapest scan. It performs about `n log n` additions where the
sequential version performs `n`, and Blelloch's up-sweep/down-sweep does `2n`.
It is the one to learn first because the step is a single readable line and
because it is *shallow* — depth 8 rather than 256 — which is what makes it fast
here and, as the numbers below show, more accurate as well.

## Two barriers per step, not one

Here is the trap this puzzle exists for. Consider `cache[t] += cache[t - off]`
with every thread executing it at once. Thread `t` **reads** `cache[t - off]`
while thread `t - off` **writes** `cache[t - off]`. Those are the same address,
and there is no ordering between the two threads.

A barrier after the update does not fix it: it orders this step's writes
against the *next* step's reads, but says nothing about this step's reads
against this step's writes. The read has to be separated from the write
*within* the step, which means each step is four things and not two:

```
read     the value `off` slots back, into a local variable
barrier  <- nobody may overwrite a slot another thread has not read yet
write    that value into your own slot
barrier  <- nobody may read a slot another thread has not written yet
```

The value lands in a register, everybody stops, and only then does anybody
write. Two barriers per step, guarding the two directions:

- the first is **write-after-read** (WAR): no thread may overwrite a slot
  another thread has not yet read;
- the second is **read-after-write** (RAW): no thread may read a slot another
  thread has not yet written.

Puzzle 12's reduction needs only the RAW barrier, because a thread there writes
`cache[t]` and reads `cache[t + s]` — and the thread that owns `cache[t + s]`
is not writing at that level at all, only threads below the midpoint write. The
scan has every thread writing at every step, so both hazards are live. Read the
barrier count off the data flow, not off habit.

The alternative — two buffers, read from one and write to the other, swap —
needs only one barrier per step and twice the shared memory. Puzzle 29 does
that with a double-buffered stencil.

## Task

Fill in `skeletons/p14_prefixsum/kernel.cu`:

```
out[i] = sum of a[base .. i], base = (i / TPB) * TPB,  n = 100003, TPB = 256
```

- Input: `a`, `n` floats. Output: `out`, `n` floats.
- Launch: 391 blocks of 256 threads, 1024 bytes of dynamic shared memory each.
- `i` and `local_i` are computed for you.
- Stage (out-of-range lanes get the identity), barrier, then the doubling loop,
  then write back.
- Both barriers in each step go outside the `if` — all 256 threads reach both.

**approx 14 lines.**

## Verification

Seeded input, `NaN` poison, **relative tolerance `1e-4`**.

The last element of a segment is a 256-term float sum, and every other element
is a shorter one, all reassociated relative to the reference's sequential
`double`. Measured worst deviation over all 100003 outputs:

| association | worst relative deviation |
|---|---|
| Hillis–Steele (the solution) | `2.146e-06` (index 83153) |
| one thread scanning sequentially | `8.106e-06` (index 81879) |

`1e-4` is the tightest decade clearing both by more than 10×. Both are correct
answers to this puzzle, so the tolerance has to admit both — and note again
that the parallel algorithm is the more accurate one, by about 4×, for the same
reason as in puzzle 12: it rounds along a depth-8 chain instead of a depth-256
one.

What a *large* deviation would mean, as opposed to these:

- correct at every index that is a multiple of 256 and wrong after — the scan
  is not reaching across the whole block; usually a loop bound of `blockDim.x`
  where the offset needed to keep doubling past it, or the other way around.
- a first failing index that is a multiple of 32 — a missing barrier. Measured
  over 15 runs of the WAR-barrier-deleted kernel below, every first failure
  landed on a multiple of 32 (1760, 13536, 24800, ... 37792) and none was at a
  smaller offset. Warps advance internally in step, so the corruption can only
  begin where one warp's reads cross another warp's writes.
- the last block wrong only — the out-of-range lanes did not get the identity.

Build output for the solution on this box: `Used 12 registers, used 1 barriers`.

## Race-freedom

`make check P=14 MODE=solution` is clean under memcheck, racecheck and
synccheck.

Delete the *first* barrier — the WAR one, the one that is easy to convince
yourself is redundant — and, measured on this box, the plain run fails **20
times out of 20** (`FAIL prefix_sum at index 96: got -1.22025 want -0.367466`)
with racecheck reporting

```
Race reported between Read access at PrefixSum(...)+0x1b0
    and Write access at PrefixSum(...)+0x1e0 [2492244 hazards]
RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)
```

Note the order in that message: **Read** then **Write**. Compare puzzle 12's,
which reads Write-then-Read. racecheck is telling you which of the two hazards
it found, and therefore which barrier is missing.

## Signature correction

Upstream (dshah3's puzzle 12, `prefixsum`) is `PrefixSum(float* A, float* C,
int size)`; here, `const float* a` and `int n`.

The larger correction is the puzzle itself: the upstream answer is not a scan.
It is a reduction — `cache[local_i] += cache[local_i + k]` for `k = 1, 2, 4` —
that leaves the block total in `cache[0]` and then writes every slot out, and
its harness asserts only `C[0] == 10`, the sum. This puzzle computes the prefix
sum its name promises.

Two other things in the upstream kernel are worth not copying: the step count
is the literal `3` rather than `log2(blockDim.x)`, and the offset is computed
as `pow(2, p)` — a double-precision transcendental call, per thread, per step,
to compute a small power of two. `1 << p`, or a doubling loop variable.

## Run

```
make run P=14
make run P=14 MODE=solution
make check P=14
```

Expected: one `PASS prefix_sum` line, and `make check` clean under all three
sanitizers.
