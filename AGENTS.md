# cuda-puzzles

CUDA C port of mojo-gpu-puzzles 24–35, on a fork of dshah3/GPU-Puzzles.
Full contract: `CUDA-PUZZLES-24-35-BRIEF.md` — read it before any puzzle work.

## Hardware

NVIDIA DGX Spark: GB10, `sm_121`, aarch64, CUDA 13, ~273 GB/s memory bandwidth.
Target `sm_121` only. No portability shims, no `#if __CUDA_ARCH__` ladders.
No `tcgen05`, no TMEM, no multicast TMA. `wmma` works. Cluster support unknown —
resolve empirically via `scripts/probe_caps.cu` before building puzzle 34 (§5 of brief).

## Layout

- `GPU_puzzlers_exec/` — upstream puzzles 1–14. Do not modify beyond de-Colabbing.
- `common/` — `puzzle_utils.cuh` (CHECK_CUDA, compare, timing), `reference.hpp` (CPU refs).
- `problems/pNN_*/` — README + `runner.cu` (immutable harness; solver never edits).
- `skeletons/pNN_*/kernel.cu` — fill-in regions marked `/// CODE HERE ///`.
- `solutions/pNN_*/kernel.cu` — complete, verified.
- `scripts/check_sync.py` — skeleton ≡ solution outside fill-in regions.

## Build

```
make P=24                    # skeleton
make run P=24 [MODE=solution]
make check P=24              # compute-sanitizer memcheck+racecheck+synccheck
make prof P=24               # ncu, puzzle-specific metrics
make test                    # all solutions, PASS/FAIL table
make sync                    # check_sync.py
```

`ARCH ?= native`. `NVCCFLAGS = -arch=$(ARCH) -lineinfo -O2 -Xptxas -v`.
Puzzles resolved by number via prefix glob.

## Rules

- One puzzle at a time, order 24 → 35, commit per puzzle. No up-front scaffolding.
- Runners: deterministic seeded input, CPU reference, `compare()` with relative
  tolerance (never `==` on float reductions), `PASS`/`FAIL` + non-zero exit,
  `CHECK_CUDA` on every API call, `cudaGetLastError()` + sync after every launch.
- Sizes exercise guard logic: non-multiples of 32 and of block size (`n = 1000`, not 1024).
- Never copy code or prose from mojo-gpu-puzzles; read for intent only.
- No notebooks, no CMake, no Docker, no CI, no mdBook.
- Every performance claim measured on the Spark; never quoted.
- If a puzzle can't work on this hardware, say so and propose a replacement —
  never ship one that deadlocks or teaches the wrong thing.

## Done (per puzzle)

README (no solution, no Mojo refs) · runner fails loudly on skeleton · solution
passes non-trivial size · `make check` clean (race-free mandatory for 27–29, 32) ·
`make sync` passes · timing puzzles emit their metric via `make prof`.
