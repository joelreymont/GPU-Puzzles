#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void TransposeNaive(const float* in, float* out, int rows,
                                      int cols);
extern __global__ void TransposePadded(const float* in, float* out, int rows,
                                       int cols);
extern __global__ void TransposeSwizzle(const float* in, float* out, int rows,
                                        int cols);

// The relationship this puzzle exists to demonstrate: the only difference
// between TransposeNaive and TransposePadded is one extra float per shared
// tile row, and it is worth a large multiple in wall-clock time because it
// turns a 32-way bank conflict into no conflict at all.
//
// MARGIN is a floor on that effect, not a description of it. On an idle box
// the measured ratio is remarkably tight -- 1.641 to 1.685 over ten
// consecutive runs -- but this is a shared-memory SoC where the GPU and the
// CPUs sit behind one LPDDR5X, and the conflict-free kernel is the one closer
// to a bandwidth limit, so contention hurts it more and squeezes the ratio
// down. Across ~30 whole-run measurements including a concurrent GPU
// bandwidth hog, sixteen CPU memory streamers, and four copies of this binary
// racing each other, the lowest value seen was 1.356. MARGIN sits below that
// floor rather than at half the effect, so the assert can never lie. The
// measured ratio is printed on the PASS line every run, and the whole
// distribution is in solutions/p32_bank_conflicts/SOLUTION.md.
constexpr float MARGIN = 1.25f;

// The swizzle should land on the padded kernel's time, not the naive one's:
// same zero conflicts, zero bytes of padding. That is a two-sided claim, so it
// gets a two-sided band -- being much faster than the pad would be as
// suspicious as being much slower. The two agree to within 1.5% on every run
// measured here, idle or contended, because they are interleaved rep for rep
// and see the same machine; the band is ~7x that spread.
constexpr float SWIZZLE_LO = 0.90f;
constexpr float SWIZZLE_HI = 1.10f;

// The throughput floor both asserts sit behind, together with the
// competing-process detector in common/puzzle_utils.cuh. QuietCheck there
// documents the design and the three outcomes a timing section has; what belongs
// here are this puzzle's numbers.
//
// TransposePadded is the reference kernel: it is the conflict-free one, so it is
// the one closest to a bandwidth limit and the one whose throughput moves first
// when something else takes the LPDDR5X this SoC shares between the GPU and the
// CPUs. It is also the denominator of both asserted ratios.
//
// Measured on this box, and unlike most of this repository's timing puzzles both
// sides of the floor are reachable here, so it is a real discriminator rather
// than a tripwire:
//
//   from below -- fifteen consecutive runs on a settled idle box measured
//   982.5-1079.2 GB/s, and the lowest figure seen on an idle GPU across every
//   session was 965.3 GB/s. Forty CPU memory streamer threads over 40 GB did not
//   move it at all: six runs measured 1026.3-1048.9 GB/s with
//   padding_beats_conflicts at 1.606-1.635. This puzzle's 8 MB working set stays
//   inside the 25 MB L2, so CPU-side memory load barely reaches it.
//
//   from above -- what does reach it is another process on the GPU. Under two GPU
//   bandwidth hogs plus 20 CPU streamers, eight runs measured 735.6-757.8 GB/s
//   and padding_beats_conflicts fell to 1.365-1.435, closing most of the distance
//   to MARGIN's 1.25. That is the regime this floor exists to refuse.
//
// 850 GB/s sits 12 % below the lowest idle observation and 12 % above the
// highest endangered one, which is as close to the middle of the two populations
// as this box's data puts it.
constexpr double QUIET_GBPS = 850.0;

// What TransposePadded reaches on a settled idle box, quoted in the
// measurement_invalid diagnosis so a reader can see how far off an invalid run
// was. Measured, not asserted: the low end of the idle spread above.
constexpr double IDLE_GBPS = 982.5;

// All three kernels are timed together: one rep of each in turn, repeated, and
// each kernel keeps its own fastest rep.
//
// Interleaving is not cosmetic. Timing every rep of one kernel and then every
// rep of the next lets a drift in machine state land entirely on one of them;
// one rep of each in turn gives every kernel the same distribution of machine
// states. The fastest rep, not the mean, is the estimate: contention only ever
// makes a measurement slower, never faster, so the minimum is the observation
// least contaminated by whatever else the machine was doing.
//
// Launched through cudaLaunchKernel rather than <<<>>> so the loop is written
// once: CUDA does not allow <<<>>> through a function pointer, but the
// host-side stub address is a perfectly good kernel handle.
static void bench_all(const void* const* kerns, int nk, dim3 grid, dim3 block,
                      const float* d_in, float* d_out, int rows, int cols,
                      int warmup, int iters, int reps, float* ms_out) {
  void* args[] = {(void*)&d_in, (void*)&d_out, (void*)&rows, (void*)&cols};
  for (int k = 0; k < nk; k++) {
    ms_out[k] = -1.0f;
    for (int i = 0; i < warmup; i++) {
      CHECK_CUDA(cudaLaunchKernel(kerns[k], grid, block, args, 0, 0));
    }
  }
  CHECK_LAUNCH();

  for (int r = 0; r < reps; r++) {
    for (int k = 0; k < nk; k++) {
      GpuTimer timer;
      timer.start();
      for (int i = 0; i < iters; i++) {
        CHECK_CUDA(cudaLaunchKernel(kerns[k], grid, block, args, 0, 0));
      }
      const float ms = timer.stop_ms() / iters;
      CHECK_CUDA(cudaGetLastError());
      if (ms_out[k] < 0.0f || ms < ms_out[k]) ms_out[k] = ms;
    }
  }
}

int main() {
  print_device_info();

  // 1000 x 1000 is deliberate on two axes.
  //
  // Guard logic: 1000 is not a multiple of 32, so the last tile column and the
  // last tile row are both 8 wide instead of 32. Every block on the right and
  // bottom edges runs with a partial warp or with whole warps switched off,
  // and a kernel that skips its guards writes outside both arrays.
  //
  // Working set: in + out is 8 MB against this box's 25 MB L2 (printed below),
  // and that is load-bearing. Past the L2 both kernels become DRAM-bound and
  // the shared-memory effect this puzzle is about disappears underneath the
  // global traffic; the sweep that establishes that is in SOLUTION.md.
  const int rows = 1000;
  const int cols = 1000;
  const int n = rows * cols;
  const int TILE = 32;        // must match the kernels' compile-time TILE
  const int BLOCK_ROWS = 8;   // must match the kernels' compile-time BLOCK_ROWS
  // Shared memory is 32 banks of 4 B on every CUDA architecture to date, and
  // cudaDeviceProp does not report it, so it is a constant here rather than a
  // query. It is printed because the whole puzzle is arithmetic modulo it.
  const int BANKS = 32;
  const dim3 block(TILE, BLOCK_ROWS);
  const dim3 grid(cdiv(cols, TILE), cdiv(rows, TILE));
  const int WARMUP = 3;
  const int ITERS = 20;
  const int REPS = 15;  // see bench_all(): the assert needs the best, not mean

  // A transpose moves floats; it does not compute them. Every output is a
  // bit-exact copy of one input, so there is no reassociation to tolerate and
  // no rounding to absorb -- anything other than exact equality here would be
  // a bug being waved through. tol = 0 makes compare() demand g == w.
  const float tol = 0.0f;

  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

  printf("# transpose: %d x %d -> %d x %d, %dx%d tiles through shared memory\n",
         rows, cols, cols, rows, TILE, TILE);
  printf("# grid: %d x %d blocks of %d x %d threads, %d rows of the tile per"
         " thread\n",
         grid.x, grid.y, block.x, block.y, TILE / BLOCK_ROWS);
  printf("# edges: last tile column is %d wide, last tile row is %d tall\n",
         cols - (grid.x - 1) * TILE, rows - (grid.y - 1) * TILE);
  printf("# working set: in + out = %.2f MB, L2 = %.2f MB\n",
         2.0 * n * sizeof(float) / 1e6, prop.l2CacheSize / 1e6);
  printf("# shared memory: %zu B/SM, %d banks x %zu B, %zu B/block reserved by"
         " the driver\n",
         prop.sharedMemPerMultiprocessor, BANKS, sizeof(float),
         prop.reservedSharedMemPerBlock);

  float* in = new float[n];
  float* got = new float[n];
  float* want = new float[n];
  fill_rand(in, n, 32);
  ref_transpose(in, want, rows, cols);

  float *d_in, *d_out;
  CHECK_CUDA(cudaMalloc(&d_in, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_in, in, n * sizeof(float), cudaMemcpyHostToDevice));

  // Correctness first, and only then timing: a fast wrong kernel is not a
  // result, and a performance *relationship* between two kernels means nothing
  // unless both of them compute the same right answer. Every kernel owns every
  // output, so poison the buffer with 0xff (a NaN bit pattern) before each
  // launch -- an output that is never written is a FAIL, not a lucky zero.
  bool ok = true;

  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  TransposeNaive<<<grid, block>>>(d_in, d_out, rows, cols);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("transpose_naive", got, want, n, tol) && ok;

  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  TransposePadded<<<grid, block>>>(d_in, d_out, rows, cols);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("transpose_padded", got, want, n, tol) && ok;

  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  TransposeSwizzle<<<grid, block>>>(d_in, d_out, rows, cols);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("transpose_swizzle", got, want, n, tol) && ok;

  // The timing section asserts a property of the *hardware*, so it is only
  // meaningful on an uninstrumented run. Setting P32_SKIP_TIMING (to anything)
  // runs correctness alone; problems/p32_bank_conflicts/prof.mk sets it for
  // the three compute-sanitizer runs, where a wall clock measures the
  // sanitizer rather than the shared-memory pipe.
  const bool skip_timing = getenv("P32_SKIP_TIMING") != nullptr;

  if (!ok) {
    printf("# timing skipped: a kernel is incorrect\n");
  } else if (skip_timing) {
    printf("# timing skipped: P32_SKIP_TIMING is set. All three kernels ran"
           " and were verified above; only the wall-clock comparison is off.\n");
  } else {
    const void* kerns[3] = {(const void*)TransposeNaive,
                            (const void*)TransposePadded,
                            (const void*)TransposeSwizzle};
    const char* names[3] = {"TransposeNaive", "TransposePadded",
                            "TransposeSwizzle"};
    float ms[3];
    QuietCheck qc{"TransposePadded", "GB/s", QUIET_GBPS, IDLE_GBPS};
    qc.watch();
    bench_all(kerns, 3, grid, block, d_in, d_out, rows, cols, WARMUP, ITERS,
              REPS, ms);
    qc.watch();

    // A transpose reads every element once and writes it once, so the traffic
    // is fixed at 2 * rows * cols * 4 B no matter which kernel moves it. That
    // makes GB/s a fair second axis: same numerator for all three, so the
    // column is the bandwidth each shared-memory access pattern leaves on the
    // table.
    const double bytes = 2.0 * n * sizeof(float);
    printf("# timing: fastest of %d interleaved reps of (%d warmup + %d timed"
           " iterations); all three move %.2f MB\n",
           REPS, WARMUP, ITERS, bytes / 1e6);
    printf("# %-18s %9s %12s\n", "kernel", "best ms", "eff GB/s");
    for (int k = 0; k < 3; k++) {
      printf("  %-18s %9.4f %12.1f\n", names[k], ms[k], bytes / (ms[k] * 1e6));
    }

    const double padded_gbps = bytes / (ms[1] * 1e6);
    qc.report();

    const float ratio = ms[0] / ms[1];
    const float sw = ms[2] / ms[1];
    if (!qc.valid(padded_gbps)) {
      qc.fail("padding_beats_conflicts and swizzle_matches_padding");
      printf("FAIL   the two ratios above are printed anyway and are real"
             " measurements -- of a saturated machine. Under external load all"
             " three kernels go memory-bound and converge, and the"
             " shared-memory effect this puzzle is about disappears underneath"
             " the global traffic.\n");
      ok = false;
    } else if (ratio >= MARGIN) {
      printf("PASS padding_beats_conflicts (naive / padded = %.3fx >= %.2fx,"
             " from one extra float per tile row)\n",
             ratio, MARGIN);
    } else {
      printf("FAIL padding_beats_conflicts: ratio %.3fx below margin %.2fx --"
             " the puzzle's performance claim did not hold on a quiet"
             " machine\n",
             ratio, MARGIN);
      printf("FAIL   TransposeNaive reads a column of a [%d][%d] tile, whose"
             " row stride is exactly the %d banks, so all %d lanes of a warp"
             " land on one bank and the access serialises %d ways;"
             " TransposePadded does the identical work with a [%d][%d] tile"
             " and no conflict at all. If the two are close, check that the"
             " padded kernel really indexes [row][col] and did not accidentally"
             " get the naive tile's shape.\n",
             TILE, TILE, BANKS, block.x, TILE, TILE, TILE + 1);
      printf("FAIL   this run passed the validity check -- no other process had"
             " a CUDA context on this GPU, and TransposePadded cleared %.1f"
             " GB/s -- so another process loading this shared-memory SoC is NOT"
             " the explanation, and it is not what this FAIL is claiming.\n",
             padded_gbps);
      printf("FAIL   (running under compute-sanitizer or ncu? set"
             " P32_SKIP_TIMING=1 -- instrumented wall-clock time is not a"
             " measurement of this machine)\n");
      ok = false;
    }

    // Reported, then asserted. The swizzle's claim is that it buys the pad's
    // conflict-freedom for zero bytes, so it has to land on the pad's time
    // from both sides -- being much faster would be as suspicious as being
    // much slower.
    //
    // Guarded by the same validity check as the assert above: a run that is not
    // a measurement has already FAILed as measurement_invalid, and evaluating a
    // second wall-clock claim on it would only add a second wrong verdict.
    if (!qc.valid(padded_gbps)) {
      // Already diagnosed above; nothing to add and nothing to assert.
    } else if (sw >= SWIZZLE_LO && sw <= SWIZZLE_HI) {
      printf("PASS swizzle_matches_padding (swizzle / padded = %.3fx, within"
             " [%.2f, %.2f], with 0 B of padding)\n",
             sw, SWIZZLE_LO, SWIZZLE_HI);
    } else {
      printf("FAIL swizzle_matches_padding: ratio %.3fx outside band [%.2f,"
             " %.2f] -- the puzzle's performance claim did not hold on a quiet"
             " machine\n",
             sw, SWIZZLE_LO, SWIZZLE_HI);
      printf("FAIL   an XOR swizzle is conflict-free in both directions, so it"
             " should time like the padded kernel. Above the band, the two"
             " loops probably disagree about where logical [r][c] lives -- a"
             " swizzle applied on the write but not the read is still correct"
             " only if it is a no-op, and is otherwise a conflict of its own."
             " Below the band, the validity check above has already ruled out"
             " the machine -- TransposePadded cleared %.1f GB/s with no other"
             " process on this GPU -- so look at the swizzle.\n",
             padded_gbps);
      ok = false;
    }
  }

  CHECK_CUDA(cudaFree(d_in));
  CHECK_CUDA(cudaFree(d_out));
  delete[] in;
  delete[] got;
  delete[] want;
  return ok ? 0 : 1;
}
