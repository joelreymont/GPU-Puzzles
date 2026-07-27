# Puzzle 25 — Warp Communication

Puzzle 24 used shuffles to *collapse* a warp into one value. This puzzle uses
them to *move* values sideways: each lane reads another lane's register
directly. No shared memory, no `__syncthreads()`, no staging array — one
instruction, and the data is in your register.

Three primitives, one shape:

```
__shfl_sync     (mask, v, srcLane)   lane L reads lane srcLane
__shfl_up_sync  (mask, v, delta)     lane L reads lane L - delta
__shfl_down_sync(mask, v, delta)     lane L reads lane L + delta
```

`v` is the *source lane's* copy of the variable, sampled at the call. Every
lane named in `mask` executes the instruction and every lane gets a result —
these are collective operations, not loads.

## Task

Complete three kernels in `skeletons/p25_warp_communication/kernel.cu`. The
runner allocates `n = 1000` floats — deliberately **not** a multiple of 32 or
of the 256-thread block — so your guard logic is exercised.

`TPB = 256` is a multiple of 32, so warps are aligned to the global index:
`lane == i % 32`, and lane 0 of the warp containing `i` holds global index
`(i / 32) * 32`.

### I/O contract

| kernel | out[i] |
|---|---|
| `NeighborDiff` | `a[i+1] - a[i]` for `i < n-1`; `0` at `i == n-1` |
| `WarpBroadcastBase` | `a[i] - a[(i/32)*32]` |
| `WarpShiftUp` | `0` when `i % 32 == 0`, else `a[i-1]` |

All three own **every** element `0 .. n-1`, boundary zeros included. The runner
poisons the output buffer with a NaN pattern before each launch, so an element
you never write is a failure, not a lucky zero.

### 1. `NeighborDiff` (approx 6 lines)

A forward difference. The value you need, `a[i+1]`, is already sitting in the
next lane's register — pull it down with `__shfl_down_sync(0xffffffffu, v, 1)`.

Lane 31 is the exception: there is no lane 32 to read from, so that lane has to
go to memory for `a[i+1]` (when `i+1` is in range). Handle the array's own tail
too — index `n-1` has no successor and its output is `0`.

### 2. `WarpBroadcastBase` (approx 3 lines)

Subtract each warp's leading element from every element of the warp. Lane 0
already holds it; `__shfl_sync(0xffffffffu, v, 0)` hands it to all 32 lanes in
one instruction. No lane needs to touch memory twice.

### 3. `WarpShiftUp` (approx 4 lines)

Shift each warp's values one lane up with `__shfl_up_sync`, writing `0` at the
head of the warp. Read the boundary section below before you assume the
hardware zero-fills for you — it does not.

## srcLane semantics and out-of-range behaviour

`__shfl_sync(mask, v, srcLane)` takes an absolute lane index. If `srcLane` is
outside `[0, 31]` it wraps modulo the warp width — no fault, no zero, just a
different lane's data. `srcLane` need not be uniform across the warp; a lane
that supplies a computed index performs a full 32-lane gather.

`__shfl_up_sync` and `__shfl_down_sync` are the interesting case:

```
delta = 1, __shfl_down_sync:   lane 0..30 get lane+1's value
                               lane 31    gets ITS OWN value back

delta = 1, __shfl_up_sync:     lane 1..31 get lane-1's value
                               lane 0     gets ITS OWN value back
```

The source lane is out of range, so the instruction is a no-op for that lane:
it keeps its own `v`. It is not zero, it is not undefined, and it does **not**
wrap around to the other end of the warp. This is the single most common
warp-shuffle bug — `shift_up` that silently duplicates element 0, or a scan
whose first `delta` lanes are wrong — and it is why kernels 1 and 3 both
predicate on the lane index instead of trusting a boundary that does not exist.
(The underlying PTX `shfl.sync` does produce a predicate saying whether the
source lane was in range; the CUDA C intrinsics do not expose it. Test the lane
yourself.)

Note the asymmetry in kernel 1: the warp boundary at lane 31 and the array
boundary at `n-1` are different problems. Lane 31 *has* a neighbour in memory
and must load it; index `n-1` has no neighbour at all and writes `0`. A warp is
32 lanes regardless of how much of the array is left.

## The mask, again

Every `*_sync` primitive names its participating lanes. `0xffffffff` means all
32. Two rules follow, and both are load-bearing:

1. **Every lane named in the mask must reach the call.** A lane that took
   `if (i >= n) return;` is not there; the remaining lanes then execute a
   collective naming a dead lane, which is undefined behaviour. That is why
   these skeletons guard with a *value* — `v = (i < n) ? a[i] : 0.0f` — and let
   every lane run to the end.
2. **The lane you read from must be in the mask and must be executing.** If the
   source lane diverged elsewhere, the result you get back is undefined.

Since Volta, warps have per-lane program counters and no longer reconverge
implicitly. `__activemask()` returns which lanes happen to be converged *at
this instant* — it is a query of scheduler state, not a fence, and it does not
cause convergence. Feeding it into a shuffle as `__shfl_down_sync(__activemask(),
v, 1)` looks defensive and is the opposite: the returned set can differ between
runs, compiler versions and optimisation levels, and the neighbour you wanted
may simply not be in it. The mask must be a value you can reason about
statically. Use `0xffffffff` and keep every lane alive.

`__activemask()` is legitimate for *measuring* convergence — debugging, or
counting how many lanes survived a branch — never for authorising one.

## Run

```
make run P=25                 # your kernel
make run P=25 MODE=solution   # reference
make check P=25               # sanitizers
```

Expected: three `PASS` lines. These kernels are elementwise — no reassociation,
no reduction — so the comparison against the CPU reference is tight (`1e-6`).
If you see a failure at index 31, 63 or 95, you trusted a warp boundary.
