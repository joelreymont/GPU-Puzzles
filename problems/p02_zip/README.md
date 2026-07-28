# Puzzle 2 — Zip

New idea: **two input arrays instead of one.**

Nothing about the thread model changes. One block, one thread per element,
`threadIdx.x` is the index. The only difference is that a thread now reads the
same index out of two arrays.

That is worth one puzzle on its own because it is the first time the pattern
"thread *i* touches element *i* of everything" appears, and it is the pattern
almost every elementwise GPU kernel uses. Note what does **not** happen: thread
`i` never looks at element `i-1` or `i+1` of anything, so no thread depends on
another thread's work, so no synchronisation of any kind is required. Puzzle 8
is the first time that stops being true.

## Task

Fill in `skeletons/p02_zip/kernel.cu`:

```
out[i] = a[i] + b[i]   for every i in [0, 1000)
```

- Inputs: `a` and `b`, 1000 floats each.
- Output: `out`, 1000 floats. Every element must be written.
- Launch: one block, 1000 threads. `i` is already computed for you.

**approx 1 line.**

## Verification

`a` and `b` are filled from two *different* seeds. That matters: if both held
the same numbers, a kernel that read `a` twice would still pass.

Tolerance is zero — one float add on each side, bit-identical. `out` is
poisoned with `NaN` before the launch, so an unwritten element fails.

## Run

```
make run P=02
make run P=02 MODE=solution
make check P=02
```

Expected: one `PASS zip` line.
