# Puzzle 26 — Advanced Warp Patterns

Puzzle 24 collapsed a warp into one value. Puzzle 25 moved values sideways by
one lane. This puzzle builds the two structured communication patterns that
everything else is made of: the **butterfly**, where every lane ends up with the
same answer, and the **scan**, where lane `k` ends up with the sum of lanes
`0..k`. Both are `O(log 32) = 5` shuffle steps, both use registers only — no
shared memory, no `__syncthreads()`, no staging array.

```
__shfl_xor_sync(mask, v, m)      lane L reads lane L ^ m      (symmetric pair)
__shfl_up_sync (mask, v, delta)  lane L reads lane L - delta  (one direction)
```

## Task

Complete three kernels in `skeletons/p26_warp_advanced/kernel.cu`. The runner
allocates `n = 1000` floats — deliberately **not** a multiple of 32 or of the
256-thread block — so your guard logic is exercised.

`TPB = 256` is a multiple of 32, so warps are aligned to the global index:
`lane == i % 32`, and the warp containing `i` covers global indices
`base .. base+31` where `base = (i / 32) * 32`. Call that an *aligned group*.

### I/O contract

| kernel | out[i] |
|---|---|
| `ButterflyAllReduce` | the total of i's aligned group: `Σ a[j]`, `j = base .. base+31` |
| `WarpInclusiveScan` | `Σ a[j]`, `j = base .. i` (inclusive prefix sum, restarted per group) |
| `WarpScanCG` | same as `WarpInclusiveScan` |

All three own **every** element `0 .. n-1`. The runner poisons the output
buffer with a NaN pattern before each launch, so an element you never write is
a failure, not a lucky zero.

The last group is partial — `n = 1000` gives a final group of indices
`992..999`. Out-of-range `j` contribute `0` to the sums, which falls out for
free from the value-guard `v = (i < n) ? a[i] : 0.0f` the skeleton already
gives you: lanes 8..31 of that warp hold `0`, participate in every shuffle, and
write nothing.

### 1. `ButterflyAllReduce` (approx 4 lines)

Five steps, `mask = 16, 8, 4, 2, 1`: add in what `__shfl_xor_sync` hands you.
After the last step every lane in the warp holds the group total, and every
in-range lane writes its own `out[i]`.

### 2. `WarpInclusiveScan` (approx 5 lines)

Hillis–Steele, hand-rolled. Five steps, `delta = 1, 2, 4, 8, 16`: pull the
value `delta` lanes below with `__shfl_up_sync` and add it — **but only in the
lanes where that value is real.** Read "The `shfl_up` predication trap" below
before you write the loop; this is the part the puzzle is actually about.

### 3. `WarpScanCG` (approx 2 lines)

The same scan through cooperative groups. The tile is already constructed;
`cg::inclusive_scan(warp, v, cg::plus<float>())` returns lane `k`'s prefix sum.
The header is already included.

## Why XOR is an *all*-reduce

Puzzle 24's `__shfl_down_sync` ladder is a *to-lane-0* reduce. Its
communication is one-directional: at `delta = 16` lanes 0..15 pull from lanes
16..31, and lanes 16..31 — whose source lane doesn't exist — silently keep
their own value. The tree has leaves. Only lane 0 is guaranteed to hold the
total, which is why puzzle 24 had exactly one lane per warp do the `atomicAdd`.

XOR has no leaves. `L ^ m` is an involution: if lane `L` reads lane `L ^ m`,
then lane `L ^ m` reads lane `L`, in the same instruction. Every lane both
sends and receives, so both halves of every pair advance together.

```
m = 16:   0↔16   1↔17   ...  15↔31        (pairs differing in bit 4)
m =  8:   0↔ 8   1↔ 9   ...  23↔31        (pairs differing in bit 3)
m =  4, 2, 1: same, bits 2, 1, 0
```

The invariant: after processing a set of bit-masks `S`, lane `L` holds the sum
over every lane whose index agrees with `L` outside `S`. Each step doubles that
set — 1, 2, 4, 8, 16, 32 lanes — and once `S` covers all five bits of a lane
index, "agrees outside `S`" is vacuous and every lane holds the whole warp's
sum. Note that the invariant depends only on *which* bits have been covered,
not on the order they were covered in, so any permutation of the five masks
works; the descending 16→1 convention is habit, not a requirement.

That property is what makes this the right primitive whenever every thread
needs the reduced value — normalising by a warp sum, softmax denominators,
comparing against a warp max — rather than one thread needing to write it out.
Replace the XOR with `__shfl_down_sync` in this kernel and index 0 is still
correct while index 1 is wrong; that is the whole difference, made visible.

## The `shfl_up` predication trap

`__shfl_up_sync(mask, v, delta)` gives lanes `delta..31` the value from `delta`
lanes below. Lanes `0..delta-1` have no source lane, so — exactly as in puzzle
25 — the instruction is a no-op for them and **they get their own `v` back**.
Not zero. Not undefined. Their own value, again.

A scan that adds unconditionally therefore doubles those lanes at every step:

```
delta = 1     lane 0 adds itself      -> 2a₀
delta = 2     lanes 0,1 add themselves
delta = 4     lanes 0..3 add themselves
...
```

Lane 0 ends at `32·a₀` instead of `a₀`, and the corruption spreads upward as
later steps read those poisoned lanes. So each step is predicated: a lane adds
the shuffled value only when `lane >= delta`. Note the shuffle itself stays
*outside* the predicate — every lane named in the mask must execute the
collective, so you compute the value in all 32 lanes and then decide whether to
use it. Predicate the **add**, never the **shuffle**.

This is the same boundary lesson as puzzle 25's `WarpShiftUp`, one level
harder: there the untouched lane was a single visible zero at the warp head,
here it is five compounding steps that produce plausible-looking wrong answers
in 31 of the 32 lanes. Exactly one lane comes out right — lane 31, which still
receives each of `a[base..base+31]` exactly once. Spot-check the last element
of a warp and a broken scan will look fine.

## What cooperative groups actually generates

`cg::inclusive_scan` does not lower to some dedicated scan unit. It is the same
Hillis–Steele ladder, written once in a header. Measured on the Spark with
`cuobjdump -sass` on the built solution:

```
WarpInclusiveScan   SHFL.UP  0x1, 0x2, 0x4, 0x8, 0x10     (5 instructions)
WarpScanCG          SHFL.UP  0x1, 0x2, 0x4, 0x8, 0x10     (5 instructions)
ButterflyAllReduce  SHFL.BFLY 0x10, 0x8, 0x4, 0x2, 0x1    (5 instructions)
```

The two scan kernels also produce **bitwise identical** output for this input —
same operations, same association order, same rounding. The library version
costs you nothing and hides the predication trap; writing it by hand once is
how you learn to recognise it in the code that doesn't use the library. (Note
also that the butterfly compiles to a named instruction, `SHFL.BFLY` — the XOR
mode is baked into the hardware shuffle unit, not synthesised from an XOR plus
an indexed shuffle.)

## Tolerance

Unlike puzzle 25's elementwise kernels, these accumulate. The warp sums 32
floats in a different association order than the CPU reference's sequential
double accumulator, so the runner compares with a relative tolerance of `1e-5`
rather than `1e-6`. Measured worst-case relative deviation across all three
kernels at `n = 1000`, seed 26: `4.8e-7` — comfortably inside the tolerance,
which is sized for the reassociation bound rather than for this one input.

If a *large* deviation appears, it is not floating point. Check the failing
index: a failure at index 0 or every index means the algorithm is wrong; a
failure only at indices `≥ 1` within each warp means you built a to-lane-0
reduce instead of an all-reduce.

## Run

```
make run P=26                 # your kernel
make run P=26 MODE=solution   # reference
make check P=26               # sanitizers
```

Expected: three `PASS` lines.
