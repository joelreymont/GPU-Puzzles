# Puzzle 3 — Guards

New idea: **there are more threads than there is data.**

Puzzles 1 and 2 launched exactly one thread per element, which only works
because 1000 happens to fit inside a block. In general the array size is
whatever it is and the launch geometry is whatever divides nicely, so the
launch almost always covers *more* indices than the array has. This puzzle
launches 1024 threads over 1000 elements. Threads 1000–1023 have no element.

They still run. Every thread executes the kernel; there is no mechanism by
which the hardware knows your array is shorter than your block. If the kernel
does not check, those 24 threads read `a[1000..1023]` and write
`out[1000..1023]`, both of which are past the end of allocations that are
exactly 4000 bytes long.

That is the guard: a bounds check on the index, in the kernel, using a size the
kernel is told.

## Task

Fill in `skeletons/p03_guards/kernel.cu`:

```
out[i] = a[i] + 10   for every i in [0, n),  n = 1000
```

- Input: `a`, `n` floats. Output: `out`, `n` floats.
- Launch: one block of **1024** threads. `i` is already computed for you.
- Threads with `i >= n` must not touch memory at all.

**approx 3 lines.**

## Why this is the puzzle where the harness stops being enough

Here is the measured behaviour of a kernel with the guard removed —
`out[i] = a[i] + 10.0f;` with no `if` — against this exact runner on this box:

```
$ ./unguarded
# NVIDIA GB10, sm_121, 48 SMs
PASS guards
$ echo $?
0
```

**It passes.** Every element the test checks is correct, because the 24 stray
threads scribble *past* the checked region, into padding the allocator handed
out and nobody is looking at. Undefined behaviour is not the same as visible
misbehaviour, and a green test is not a proof.

Run the same binary under `compute-sanitizer`, which knows the exact bounds of
every allocation, and it is a different program:

```
========= Invalid __global__ read of size 4 bytes
=========     at Guards(const float *, float *, int)+0x60 in unguarded.cu:4
=========     by thread (1000,0,0) in block (0,0,0)
=========     Access to 0xe7756ae00fa0 is out of bounds
=========     and is 1 bytes after the nearest allocation at 0xe7756ae00000 of size 4000 bytes
...
========= ERROR SUMMARY: 25 errors
```

One report per stray thread — threads 1000 through 1023, 24 of them — plus the
launch failure they caused. `4000 bytes` is `n * sizeof(float)`, and thread
1000's address is `1 bytes after` its end.

Guard the read but not the write and you get the other half of the same
message:

```
========= Invalid __global__ write of size 4 bytes
=========     at Guards(const float *, float *, int)+0xf0 in writeoob.cu:5
=========     by thread (1000,0,0) in block (0,0,0)
=========     Access to 0xe9f62ae01fa0 is out of bounds
=========     and is 1 bytes after the nearest allocation at 0xe9f62ae01000 of size 4000 bytes
```

An out-of-bounds *write* is the worse of the two: it corrupts memory that
belongs to something else, and the failure surfaces somewhere unrelated, much
later, in a kernel that is not the one with the bug.

So: `make run` checks that your answer is right, and `make check` checks that
your kernel only touched memory it owns. Both, every time. That is why this
repository's definition of done for every puzzle includes a clean
`make check` — not just a `PASS`.

## Signature correction

Upstream declares this kernel `Guards(float* A, float* C, float size)` and
compares `i < size` with `i` an `int` — an element count carried in a `float`.
Here it is `int n`. Counts are integers; a `float` cannot represent every
`int` past 2^24, and the comparison silently promotes `i` to `float` on every
thread for no reason. The same correction is applied in puzzles 4, 6, 7 and 8.

## Run

```
make run P=03
make run P=03 MODE=solution
make check P=03
```

Expected: one `PASS guards` line, and `make check` clean.
