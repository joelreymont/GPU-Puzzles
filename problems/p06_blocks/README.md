# Puzzle 6 — Blocks

New idea: **more than one block.**

A block is capped at 1024 threads, which is why every puzzle so far has stopped
at 1000 elements. Real arrays are bigger. The grid is the answer: the launch
asks for many blocks, each with the same number of threads, and the hardware
distributes them across the GPU's streaming multiprocessors (48 of them on this
box) as capacity frees up.

Two more built-ins come into play:

| built-in | meaning |
|---|---|
| `blockIdx.x` | which block this is, `0 .. gridDim.x-1` |
| `blockDim.x` | threads per block (the same for every block) |

A thread's identity is now a pair, and the flat index is the obvious
combination:

```
i = blockIdx.x * blockDim.x + threadIdx.x
```

This is the single most-typed line in CUDA. It is already written for you here;
type it from memory anyway, a few times.

Two properties of blocks you are relying on and should know you are relying on:

- **Blocks are independent.** There is no ordering between them and no way to
  synchronise across them inside a kernel. Block 7 may finish before block 0
  starts. This kernel is fine with that because no element depends on another.
- **The grid rarely divides evenly.** 100003 elements at 256 threads per block
  needs `ceil(100003 / 256) = 391` blocks, and `391 * 256 = 100096`, so 93
  threads in the last block have no element. Puzzle 3's guard, unchanged,
  handles them — and now it is the *last block* that overhangs rather than the
  end of the only block.

The runner computes the block count with `cdiv(n, TPB)` from
`common/puzzle_utils.cuh`. Rounding down instead — `n / TPB` — silently drops
the tail of the array, which the `NaN` poison catches immediately.

## Task

Fill in `skeletons/p06_blocks/kernel.cu`:

```
out[i] = a[i] + 10   for every i in [0, n),  n = 100003
```

- Input: `a`, `n` floats. Output: `out`, `n` floats.
- Launch: 391 blocks of 256 threads. `i` is already computed for you.

**approx 3 lines.**

## Verification

Tolerance zero, `NaN` poison, seeded input. `n = 100003` is prime and is not a
multiple of 256 or of 32, so the last block is partial and the last warp of that
block is partial too.

Measured against this runner, a kernel that forgets `blockIdx` and uses
`i = threadIdx.x` fails at index 256 — the first element only block 1 could
have written.

## Signature correction

Upstream is `Blocks(float* A, float* C, float size)`; here, `int n`. See
puzzle 3.

## Run

```
make run P=06
make run P=06 MODE=solution
make check P=06
```

Expected: one `PASS blocks` line.
