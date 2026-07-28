# Puzzle 1 — Map

The first puzzle, and the only one that explains the machine from zero. Read
this one slowly; puzzles 2 onward each add exactly one idea.

## What a kernel is

A CPU function processes an array with a loop: it visits element 0, then 1,
then 2. A GPU does not. You write the body of the loop — what happens to *one*
element — and the hardware runs thousands of copies of it at once, one per
**thread**. That body is a **kernel**: a function marked `__global__`, called
from the CPU, executed on the GPU.

```
__global__ void ScalarAdd(const float* a, float* out) { ... }
```

Every thread runs the same instructions. The only thing that differs between
them is a set of read-only built-in variables the hardware fills in:

| built-in | meaning |
|---|---|
| `threadIdx.x` | this thread's index inside its block, `0 .. blockDim.x-1` |
| `blockDim.x` | how many threads are in a block |
| `blockIdx.x` | which block this is (puzzle 6 — ignore it for now) |
| `gridDim.x` | how many blocks there are (puzzle 6) |

So `threadIdx.x` is how a thread knows *which* element is its job. That is the
whole trick: one thread, one element, and the index comes from the thread's
own identity rather than from a loop counter.

Threads are launched in **blocks**. The CPU says how many blocks and how many
threads per block using the triple-angle-bracket syntax:

```
ScalarAdd<<<blocks, threads_per_block>>>(d_a, d_out);
```

This puzzle launches `<<<1, 1000>>>` — a single block of 1000 threads, so
`threadIdx.x` runs 0 to 999 and covers the array exactly. A block can hold at
most 1024 threads, which is why puzzles 1–5 stop at 1000 elements and puzzle 6
introduces multiple blocks.

`d_a` and `d_out` are pointers into **device memory** — a separate address
space from the CPU's. The runner allocates them with `cudaMalloc`, copies the
input across with `cudaMemcpy`, launches, and copies the result back. The
kernel can only dereference device pointers, and your CPU code can only
dereference host pointers. Mixing them up is the classic first-day crash.

The launch is **asynchronous**: `<<<...>>>` returns immediately, before the
kernel has run. The runner calls `cudaGetLastError()` and
`cudaDeviceSynchronize()` afterwards (wrapped in `CHECK_LAUNCH()`) so that a
failed launch is reported instead of silently producing nothing.

## Task

Fill in `skeletons/p01_map/kernel.cu`:

```
out[i] = a[i] + 10   for every i in [0, 1000)
```

- Input: `a`, 1000 floats in device memory.
- Output: `out`, 1000 floats. Every element must be written.
- Launch: one block, 1000 threads. `i` is already computed for you.

**approx 1 line.**

There is no bounds check in this kernel and none is needed: there is exactly
one thread per element. Puzzle 3 is what happens when that stops being true.

## Verification

The runner fills `a` from a fixed seed, computes `a[i] + 10.0f` on the CPU, and
compares with tolerance **zero** — one IEEE float add on each side gives
bit-identical results, so any difference at all is a bug and never rounding.
Before the launch it fills `out` with `0xff` bytes, which read back as `NaN`;
an element you forget to write therefore fails loudly instead of accidentally
being right.

## Run

```
make run P=01                 # your kernel
make run P=01 MODE=solution   # the reference
make check P=01               # compute-sanitizer
```

Expected: one `PASS map` line.
