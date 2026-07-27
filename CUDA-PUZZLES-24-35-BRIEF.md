# Brief: CUDA C puzzles 24–35, built on a fork of dshah3/GPU-Puzzles

**Audience:** an autonomous coding agent with shell access.
**Owner:** Joel. Personal learning repo — not for publication.
**Target hardware:** NVIDIA DGX Spark (GB10, `sm_121`, aarch64, CUDA 13).

---

## 1. What this is

[modular/mojo-gpu-puzzles](https://github.com/modular/mojo-gpu-puzzles) teaches GPU
programming through 35 puzzles in Mojo. A CUDA C port of the *first fourteen* already
exists — [dshah3/GPU-Puzzles](https://github.com/dshah3/GPU-Puzzles) — covering roughly
Modular's Parts I–III (map through naive tiled matmul). It is small: 14 host runners,
632 lines, kernel skeletons living only inside a Colab notebook.

Nothing covers Modular's **Parts VII–XII (puzzles 24–35)**: warp-level programming,
block-level collectives, async memory, synchronization, profiling, occupancy, bank
conflicts, tensor cores, clusters, alignment. That is the gap this project fills, and it
is the material where CUDA C is the *native* idiom rather than a translation target —
Mojo's `warp.shuffle_down()` and `block.sum()` are wrappers over CUDA primitives.

**Explicitly out of scope:** puzzles 1–23. Puzzles 1–16 are already done by dshah3.
Puzzles 17–22 are MAX Graph and PyTorch integration with no meaningful CUDA C analogue.
Puzzle 23 is Mojo functional patterns (`elementwise`, `tile`, `vectorize`) — language
features, not GPU concepts. Do not port any of these.

---

## 2. Setup

```bash
# Fork on GitHub first, then:
git clone git@github.com:<joel>/GPU-Puzzles.git cuda-puzzles
cd cuda-puzzles
git remote add upstream https://github.com/dshah3/GPU-Puzzles.git
git checkout -b puzzles-24-35
```

The fork gives you puzzles 1–14 plus attribution lineage. The repo is Apache-2.0
(inherited from srush); keep `LICENSE` intact and note both upstreams in the README.

### 2.1 De-Colab the existing puzzles first

The 14 existing kernel skeletons exist only as `%%writefile` cells inside
`GPU_Puzzlers_C++.ipynb`. Extract them to real files and add a Makefile — a working
script for this already exists (`spark-setup.sh`, provided separately). Run it before
anything else so the repo is editor-and-`make` driven end to end. Verify with
`make test` in `GPU_puzzlers_exec/` — all 14 should build (they will FAIL at runtime
until kernels are filled in; that is correct).

---

## 3. Target layout

Three parallel trees so a skeleton and its solution compile against the *same* runner:

```
cuda-puzzles/
├── GPU_puzzlers_exec/          # untouched upstream puzzles 1–14
├── common/
│   ├── puzzle_utils.cuh        # CHECK_CUDA, compare(), timing, device query
│   └── reference.hpp           # CPU reference implementations
├── problems/
│   └── p24_warp_fundamentals/
│       ├── README.md           # the puzzle statement
│       └── runner.cu           # host harness: alloc, launch, verify. NEVER edited by solver.
├── skeletons/
│   └── p24_warp_fundamentals/
│       └── kernel.cu           # `/// CODE HERE (approx N lines) ///`
├── solutions/
│   └── p24_warp_fundamentals/
│       └── kernel.cu           # complete, verified
├── scripts/
│   └── check_sync.py           # skeleton must equal solution outside fill-in regions
└── Makefile
```

`problems/` holds the statement and the immutable harness. `skeletons/` holds what Joel
edits. `solutions/` holds the reference. The runner declares
`extern __global__ void Foo(...);` — exactly the pattern dshah3 uses — so either kernel
tree links against it unchanged.

### 3.1 Build interface

```bash
make P=24                  # build skeleton (default)
make run   P=24            # build + run skeleton
make run   P=24 MODE=solution
make check P=24            # compute-sanitizer memcheck + racecheck + synccheck
make prof  P=24            # ncu with the metrics relevant to that puzzle
make test                  # every puzzle, solution mode, PASS/FAIL table
make sync                  # scripts/check_sync.py
```

`ARCH ?= native` (resolves to `sm_121` on the Spark). `NVCCFLAGS = -arch=$(ARCH)
-lineinfo -O2 -Xptxas -v`. Keep `-lineinfo` — compute-sanitizer and `ncu` source
attribution are useless without it. `-Xptxas -v` prints register and shared-memory usage,
which puzzles 31 and 32 actually need.

Accept the puzzle by number (`P=24`) and resolve the directory by prefix glob, so the
descriptive suffix can change without breaking commands.

### 3.2 Harness contract

Upstream's runners use hardcoded `assert(C[0][0] == 0)`. That does not scale past
toy sizes. Every new runner must instead:

1. Allocate host input, fill it deterministically (fixed seed — no `rand()` without one).
2. Compute the expected result with a CPU reference from `common/reference.hpp`.
3. Run the kernel.
4. `compare(got, expected, n, tol)` — relative tolerance, not `==`. Float reductions
   reassociate; exact equality will produce false failures on the warp and block
   reduction puzzles specifically.
5. Print `PASS <name>` / `FAIL <name> at index i: got X want Y` and exit non-zero on
   failure.
6. Wrap every CUDA API call in `CHECK_CUDA(...)` and call `cudaGetLastError()` plus
   `cudaDeviceSynchronize()` after each launch. Silent launch failures are the single
   most common way these harnesses lie.

Sizes should be large enough to be *correct-by-construction-proof*, not just lucky:
use non-multiples of 32 and of the block size so guard logic is exercised (e.g. `n = 1000`,
not `n = 1024`). This is the main defect in the upstream runners — `size = 2` with
`TPB = 3` passes with almost any wrong kernel.

---

## 4. The puzzles

Each entry gives the Mojo original and the CUDA constructs the puzzle should teach.
Read the corresponding chapter at `https://puzzles.modular.com/puzzle_NN/` for the
pedagogical intent, then design the CUDA version independently — do **not** transliterate
Mojo, and do not copy Modular's prose.

| # | Directory | Mojo original | CUDA constructs to teach |
|---|---|---|---|
| 24 | `p24_warp_fundamentals` | Warp lanes & SIMT, `warp.sum()` | `__activemask()`, `__ballot_sync`, lane/warp id from `threadIdx`, `__shfl_down_sync` reduction, `cg::tiled_partition<32>` + `cg::reduce`. Show the divergence trap: why `__syncwarp()` exists post-Volta. |
| 25 | `p25_warp_communication` | `warp.shuffle_down()`, `warp.broadcast()` | `__shfl_down_sync`, `__shfl_sync(mask, v, srcLane)`, `__shfl_up_sync`. Mask correctness under divergence is the lesson. |
| 26 | `p26_warp_advanced` | `warp.shuffle_xor()` butterfly, `warp.prefix_sum()` | `__shfl_xor_sync` butterfly all-reduce, inclusive scan via `__shfl_up_sync` + lane predication, `cg::inclusive_scan`. |
| 27 | `p27_block_patterns` | `block.sum()`, `block.prefix_sum()`, `block.broadcast()` | Hand-rolled warp-then-shared two-stage reduction **and** the CUB equivalent (`cub::BlockReduce`, `cub::BlockScan`). Build both; the point is seeing what CUB generates. Application: parallel histogram binning. |
| 28 | `p28_async_copy` | Async memory ops & copy overlap | `cuda::memcpy_async` + `cuda::pipeline`, or `__pipeline_memcpy_async`/`__pipeline_commit`/`__pipeline_wait_prior`. Double-buffered tile loads. Compare against the naive sync version and measure. |
| 29 | `p29_synchronization` | Barrier, memory barrier, double-buffered stencil | `cuda::barrier<cuda::thread_scope_block>`, `__syncthreads()` vs `__threadfence_block()` vs `__threadfence()`, arrive/wait split-barrier. Multi-stage pipeline + double-buffered stencil. |
| 30 | `p30_profiling` | GPU profiling, "the cache hit paradox" | `ncu` sections (`SpeedOfLight`, `MemoryWorkloadAnalysis`), `nsys` timeline. Construct two kernels where the higher L1 hit rate is the *slower* one, and have the solver explain why from the metrics. |
| 31 | `p31_occupancy` | Occupancy optimization | `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `__launch_bounds__`, register spill inspection via `-Xptxas -v`, shared-memory-per-block as an occupancy limiter. Include a case where *lower* occupancy is faster (ILP). |
| 32 | `p32_bank_conflicts` | Shared memory banks, conflict-free patterns | 32-bank model, `+1` padding, XOR swizzle, `float4` vs `float` access granularity. Measure with `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`. |
| 33 | `p33_tensor_cores` | Tensor core operations | `nvcuda::wmma` 16×16×16 `half`/`bf16` fragments, `load_matrix_sync`/`mma_sync`/`store_matrix_sync`, layout and alignment requirements. **See §5** for what `sm_121` does not have. |
| 34 | `p34_clusters` | Cluster programming (SM90+) | `__cluster_dims__`, `cg::cluster_group`, `cluster.sync()`, distributed shared memory via `cluster.map_shared_rank()`. **Gated — see §5.** |
| 35 | `p35_alignment` | Memory alignment for load/store | `float4`/`int4` vectorized access, `__align__(16)`, `__ldg`, misaligned-pointer penalty, why `reinterpret_cast<float4*>` on an unaligned base is UB. Measure the bandwidth delta. |

Puzzle 30 is a profiling exercise, not a kernel-writing one; its "solution" is a
`SOLUTION.md` explaining the metrics, plus the two contrasting kernels. Same for the
analysis halves of 31 and 32 — the answer is an explanation, and the harness should
check the kernel's *performance relationship* (A faster than B by some margin) rather
than only correctness.

---

## 5. Hardware gates — verify before building 33 and 34

GB10 is `sm_121`, consumer-class Blackwell. It does **not** have the datacenter
(`sm_100`) feature set. Confirmed absent: `tcgen05` async-MMA instructions, Tensor
Memory (TMEM), cluster-shared/multicast TMA. Single-CTA TMA is present.

**Thread block clusters on `sm_121` are genuinely uncertain and sources conflict.** One
analysis states SM120 supports only cluster size 1, with a silent downgrade and a
subsequent `cluster.sync()` deadlock; NVIDIA forum discussion claims consumer Blackwell
supports 8-block clusters minus multicast. Do **not** resolve this from documentation.
Resolve it empirically as the first step of puzzle 34:

```cpp
int v = 0;
cudaDeviceGetAttribute(&v, cudaDevAttrClusterLaunch, 0);
// then cudaOccupancyMaxPotentialClusterSize() on an actual __cluster_dims__ kernel
```

Commit that probe as `scripts/probe_caps.cu` and have `make` run it once, caching results
to `common/caps.mk`. Gate puzzle 34's build on it. If clusters turn out to be
size-1-only, **do not write a cluster puzzle that deadlocks** — replace 34 with a
cooperative-groups grid-wide sync puzzle (`cudaLaunchCooperativeKernel`,
`cg::this_grid().sync()`) and document the substitution in the puzzle README. A puzzle
that hangs the box is worse than no puzzle.

For 33: `wmma` works on `sm_121`. Target `wmma` only. Do not attempt `tcgen05` or
`sm_100a`-gated CUTLASS paths — they will not compile.

Also note for every timing-based puzzle (28, 30, 31, 32, 35): GB10 has ~273 GB/s of
memory bandwidth, far below a discrete Blackwell card. Bandwidth-bound kernels will hit
their roofline much earlier than published numbers suggest. Every performance claim in a
README must be measured on the Spark, not quoted from a blog post.

---

## 6. Definition of done, per puzzle

A puzzle is complete when all of these hold:

1. `problems/pNN/README.md` states the problem, the input/output contract, and a hint at
   the expected line count — no solution, no Mojo references.
2. `problems/pNN/runner.cu` compiles, and **fails loudly** against the skeleton.
3. `solutions/pNN/kernel.cu` compiles and passes with a non-trivial input size.
4. `make check P=NN MODE=solution` is clean under memcheck, racecheck, **and** synccheck.
   Race-free is not optional for 27, 28, 29 and 32.
5. `make sync` passes: skeleton and solution are byte-identical outside the marked
   fill-in region.
6. For timing puzzles, `make prof P=NN` emits the specific metric the README discusses.

Work one puzzle at a time, in order 24 → 35. Commit per puzzle. Do not scaffold all
twelve directories up front and fill them in later — the harness design will change
after the first two are real, and a batch of empty scaffolding hides that.

---

## 7. Constraints

- **Do not copy code or prose from mojo-gpu-puzzles.** Read it for intent; write the CUDA
  independently. Different licence lineage, and transliterated Mojo makes bad CUDA.
- **Do not modify `GPU_puzzlers_exec/`** beyond the `spark-setup.sh` de-Colabbing.
  Upstream merges should stay possible.
- **Do not add a notebook.** The whole point is that this runs from an editor on the Spark.
- **Do not add CMake, Docker, CI, or an mdBook.** Personal repo. A Makefile is sufficient
  and every layer added is a layer to maintain.
- **Do not target GPUs other than `sm_121`.** No portability shims, no `#if __CUDA_ARCH__`
  ladders for hardware Joel does not own.
- **Do not fabricate performance numbers.** If a claim is not measured on the Spark, do
  not write it down.
- If a puzzle turns out not to work on this hardware, say so and propose a replacement.
  Do not ship something that compiles but teaches the wrong thing.

---

## 8. First deliverable

Before writing puzzle 24, produce and get sign-off on:

- `common/puzzle_utils.cuh` and `common/reference.hpp`
- the root `Makefile`
- `scripts/probe_caps.cu` output from an actual run on the Spark
- one complete puzzle (24) end to end as the pattern for the rest

Report the cluster-support finding explicitly — it determines whether puzzle 34 exists in
its planned form.
