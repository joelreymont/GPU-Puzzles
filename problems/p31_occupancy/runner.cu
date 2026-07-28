#include "puzzle_utils.cuh"
#include "reference.hpp"

#include <algorithm>
#include <vector>

extern __global__ void PolyOne(const float* a, float* out, int n);
extern __global__ void PolyFour(const float* a, float* out, int n);
extern __global__ void SmemHog(const float* a, float* out, int n);

// Two relationships this puzzle exists to demonstrate, checked very
// differently because they are different kinds of fact.
//
// smem_limits_occupancy is arithmetic on the compiled kernels. How many blocks
// of 256 threads fit on an SM is an integer division of fixed hardware
// budgets, and cudaOccupancyMaxActiveBlocksPerMultiprocessor computes it from
// the binary. Same answer every run, on every machine of this model, whether
// or not anything is instrumenting it. No margin, no tolerance, no clock, no
// gate -- it is asserted on every run, including instrumented ones.
//
// occupancy_not_predictive is a wall-clock measurement, and everything below
// exists because of that difference.
//
// MARGIN is a floor on the PolyOne/PolyFour effect, not a description of it.
// What it is a floor on is the PAIRED per-rep median (see bench_all and
// paired_ratio) taken over the quietest window of the run (see
// quietest_reps), behind the QUIET_GELEM gate.
//
// Measured with that estimator over 69 runs on this box, spanning everything
// from a settled machine to three GPU bandwidth hogs plus 16 CPU memory
// streamers plus three concurrent copies of this binary. On a quiet box the
// ratio is 1.28-1.43. Under load it falls, and it falls together with
// PolyFour's absolute throughput -- which is what the gate below reads.
//
// Over those 69 runs, the floor across every run the gate ADMITS is 1.1066.
// MARGIN is 1.05, which sits 0.057 below that floor and still decisively
// rejects what this assert exists to catch: a PolyFour with no per-thread
// amortisation measures ~1.00, and the worst gated-out run measured 0.9913.
//
// 1.10 was tried and is NOT defensible here, which is worth recording because
// the paired estimator is otherwise so much steadier than what it replaced. Per
// run it is steadier; but its coupling to the gate is loose near the threshold,
// so the gated-in floor is 1.1066 and a 1.10 margin would clear it by 0.0066 --
// one sample from flaking. Buying headroom by raising the gate does not work
// either: 110 Gelem/s only lifts the floor to 1.1249 and declines 48 % of runs
// instead of 33 %. The distribution supports 1.05 and not more.
//
// For the record of what this replaced. The first version of this runner took
// the ratio of each kernel's fastest rep across separately-timed reps. That is
// the right estimator when the effect is a large multiple, and the wrong one
// at 28 %: two minima drawn from different machine states swamp it. It FAILed
// 3 of 9 full-suite runs, twice with the ratio outright inverted (0.899x,
// 1.000x) on a box that reported itself idle. The paired median never inverted
// once in 69 runs, including under a load that dropped the SM clock to 904 MHz.
constexpr float MARGIN = 1.05f;

// The throughput floor MARGIN sits behind: half of this runner's
// measurement-validity check, the other half being the competing-process
// detector in common/puzzle_utils.cuh (see QuietCheck there for the design and
// for the three outcomes a timing section has).
//
// Why a validity check at all. This is a shared-memory SoC: one LPDDR5X behind
// both the GPU and the CPUs. When something else on the box takes a large share
// of it, PolyOne and PolyFour stop being limited by their own exposed load
// latency and start being limited by the contention -- and the ratio between
// them collapses toward 1.0. That is physics, not noise. Under an external
// bottleneck the per-thread amortisation genuinely stops paying, because the
// thing it amortises is no longer the constraint. No estimator recovers an
// effect that is not present, and a margin low enough to survive the contended
// floor would not be a claim worth making.
//
// So the runner establishes that it got a measurement before it asserts
// anything about one, and a run that did not get one FAILs as
// measurement_invalid rather than passing quietly. It never skips: a busy box is
// not a wrong kernel, but it is not a green suite either.
//
// Now the part that is specific to this puzzle, and that was re-measured for
// these semantics because a floor that fires now costs a red suite rather than
// a skipped line.
//
// PolyFour reaches 138.5-139.6 Gelem/s on a SETTLED idle box -- 30 consecutive
// runs, a 0.8 % spread -- and 100 Gelem/s is 72 % of that. What the floor is
// really demanding is that settled state, and the honest description of the
// floor is that it is a settledness check, not a contention check. The
// difference matters, and it is measurable:
//
//   this box has an idle-but-unsettled state, and it is not rare. With the GPU
//   verifiably free (the detector below reports zero competing processes), the
//   CPUs at a load average of 0.15-0.65, the GPU at 38-40 C drawing a flat
//   30-31 W, the SM clock flat at 2539-2561 MHz and cudaMalloc returning
//   byte-identical addresses every run, fourteen consecutive runs measured
//   88.2-139.0 Gelem/s. The rate is constant within any one process -- every one
//   of the 201 reps of a slow run is slow -- and varies 1.6x between processes.
//   It is not thermal, not DVFS, not which CPU the host thread lands on, and not
//   another process. It takes minutes to clear after a heavy memory load.
//
//   under load the rates OVERLAP that range completely. A sweep from 2 to 40 CPU
//   memory streamer threads produced 28.8 to 137.6 Gelem/s. So the two
//   populations do not separate on this axis and no threshold on it is a
//   contention detector. That job belongs to the competing-process detector,
//   which is causal and has none of this problem.
//
//   the floor is not redundant even so, because it is the only thing that sees
//   CPU-side load at all: a process on the other side of this SoC's LPDDR5X
//   holds no CUDA context and appears in no GPU-side accounting. Where it earns
//   its keep is the deep end. Sub-MARGIN ratios were measured at 28.8-56.5
//   Gelem/s, as low as 0.83, and 100 refuses all of them.
//
// It is left at 100 rather than lowered to fit the unsettled state, and the
// reason is which mistake each setting makes. At 100, an unsettled run FAILs as
// measurement_invalid: loud, non-zero, and explicitly not a claim about the
// kernels. Lowered to clear the unsettled runs it would admit them, and one
// contended run measured 110.3 Gelem/s with a ratio of 1.0440 -- under MARGIN --
// so what a lower floor buys is FAILs that blame PolyFour for the machine. Of
// the two, the first is the one this file is willing to print. It does mean a
// red suite on a box that has not settled; the answer to that is to let it
// settle, which is what the diagnosis says.
constexpr double QUIET_GELEM = 100.0;

// What PolyFour reaches on a settled idle box, quoted in the measurement_invalid
// diagnosis so a reader can see how far off an invalid run was. Measured, not
// asserted: the low end of the 30-run settled mode described above.
constexpr double IDLE_GELEM = 138.5;

struct Occ {
  int regs;
  size_t smem;
  int blocks;   // concurrent blocks per SM
  double frac;  // blocks * TPB / maxThreadsPerMultiProcessor
};

// cudaFuncGetAttributes reports what the compiler produced; the occupancy API
// turns that plus the launch shape into the resource-limit answer. Static
// shared memory is accounted automatically -- SmemHog never has to declare its
// 24 KiB here, which is exactly why the API is worth using instead of doing
// the arithmetic by hand.
template <typename K>
static Occ occupancy_of(K kern, int tpb, const cudaDeviceProp& prop) {
  cudaFuncAttributes fa;
  CHECK_CUDA(cudaFuncGetAttributes(&fa, kern));
  int blocks = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kern, tpb,
                                                           /*dynamic smem*/ 0));
  Occ o;
  o.regs = fa.numRegs;
  o.smem = fa.sharedSizeBytes;
  o.blocks = blocks;
  o.frac = (double)blocks * tpb / prop.maxThreadsPerMultiProcessor;
  return o;
}

// Measures the SM clock, in MHz, by spinning a single warp for a known number
// of SM cycles and timing it on the wall clock.
//
// This is not decoration. nvidia-smi on this box reports `clocks.sm` as
// 208 MHz even while the GPU is at 91 % utilisation drawing 56 W, so the only
// way to know what clock a measurement was taken at is to ask the device
// itself. One warp spinning on clock64() draws almost no power, so this also
// wakes the GPU off its idle clock before anything downstream of it is timed.
__global__ void SpinCycles(long long cycles, long long* out) {
  long long d = 0;
  const long long t0 = clock64();
  do {
    d = clock64() - t0;
  } while (d < cycles);
  if (blockIdx.x == 0 && threadIdx.x == 0) *out = d;
}

static float measure_sm_mhz(long long* d_scratch) {
  constexpr long long CYCLES = 20000000;  // ~8 ms at 2.5 GHz
  GpuTimer timer;
  timer.start();
  SpinCycles<<<1, 32>>>(CYCLES, d_scratch);
  const float ms = timer.stop_ms();
  CHECK_LAUNCH();
  return (float)CYCLES / (ms * 1000.0f);
}

// One timed kernel: a name, a handle, and its grid. All three take the same
// argument pack; only the grid differs, because PolyFour covers four elements
// per thread.
struct Job {
  const char* name;
  const void* kern;
  dim3 grid;
};

// All three kernels are timed together: one rep of each in turn, repeated, and
// every individual sample is kept.
//
// Interleaving is not cosmetic. Timing every rep of one kernel and then every
// rep of the next lets a drift in machine state land entirely on one of them;
// measured on this box that ordering bias is worth about 0.05 on the ratio,
// always against whichever kernel runs last. One rep of each in turn gives
// every kernel the same distribution of machine states.
//
// Keeping every sample rather than only the minimum is what lets the caller
// form PAIRED ratios -- PolyOne and PolyFour as measured inside the same rep,
// a few hundred microseconds apart -- and take the median of those. See
// MARGIN for why that replaced the ratio of minima this runner used first.
//
// Launched through cudaLaunchKernel rather than <<<>>> so the loop is written
// once: CUDA does not allow <<<>>> through a function pointer, but the
// host-side stub address is a perfectly good kernel handle.
static void bench_all(const Job* jobs, int nk, int tpb, void** args, int warmup,
                      int iters, int reps, std::vector<float>* samples) {
  for (int k = 0; k < nk; k++) {
    samples[k].clear();
    for (int i = 0; i < warmup; i++) {
      CHECK_CUDA(
          cudaLaunchKernel(jobs[k].kern, jobs[k].grid, dim3(tpb), args, 0, 0));
    }
  }
  CHECK_LAUNCH();

  for (int r = 0; r < reps; r++) {
    for (int k = 0; k < nk; k++) {
      GpuTimer timer;
      timer.start();
      for (int i = 0; i < iters; i++) {
        CHECK_CUDA(
            cudaLaunchKernel(jobs[k].kern, jobs[k].grid, dim3(tpb), args, 0, 0));
      }
      samples[k].push_back(timer.stop_ms() / iters);
      CHECK_CUDA(cudaGetLastError());
    }
  }
}

// Ranks the reps by how quiet the machine was during each one -- the sum of
// the three kernels' samples in that rep -- and returns the indices of the
// quietest `keep`, in rep order.
//
// Contention only ever makes a measurement slower, never faster, so the reps
// with the smallest total are the ones least contaminated by whatever else the
// box was doing. This is the same principle the earlier timing puzzles apply
// when they keep each kernel's fastest rep, moved up one level: the unit
// selected is the matched TRIPLE, so the three kernels are still being
// compared against each other inside a single machine state rather than each
// against its own best moment.
//
// It cannot flatter a slow kernel. The ranking is on the rep total, so a
// kernel that is uniformly slow is slow in every rep, quiet or not, and its
// ratio is unchanged. What the ranking removes is the rep in which some other
// process took the memory system away from all three at once.
static std::vector<int> quietest_reps(const std::vector<float>* samples, int nk,
                                      int reps, int keep) {
  std::vector<int> idx(reps);
  for (int r = 0; r < reps; r++) idx[r] = r;
  std::sort(idx.begin(), idx.end(), [&](int x, int y) {
    float sx = 0.0f, sy = 0.0f;
    for (int k = 0; k < nk; k++) {
      sx += samples[k][x];
      sy += samples[k][y];
    }
    return sx < sy;
  });
  idx.resize(keep);
  std::sort(idx.begin(), idx.end());
  return idx;
}

// Median of one kernel's samples over the given reps.
static float median_over(const std::vector<float>& v,
                         const std::vector<int>& reps) {
  std::vector<float> s;
  s.reserve(reps.size());
  for (int r : reps) s.push_back(v[r]);
  std::sort(s.begin(), s.end());
  return s[s.size() / 2];
}

// Median of the per-rep ratio of two kernels over the given reps. `lo` and
// `hi` come back as the quartiles, so the caller can print the spread the
// median came out of.
static float paired_ratio(const std::vector<float>& num,
                          const std::vector<float>& den,
                          const std::vector<int>& reps, float* lo, float* hi) {
  const int n = (int)reps.size();
  std::vector<float> r(n);
  for (int i = 0; i < n; i++) r[i] = num[reps[i]] / den[reps[i]];
  std::sort(r.begin(), r.end());
  *lo = r[n / 4];
  *hi = r[(3 * n) / 4];
  return r[n / 2];
}

int main() {
  print_device_info();

  // 2000003 is deliberate on two axes.
  //
  // Guard logic: it is a multiple of neither 32, nor TPB, nor ELEMS * TPB, so
  // PolyOne's last block has 3 live threads out of 256 and PolyFour's last
  // thread runs off the end of `a` on three of its four elements.
  //
  // Working set: a + out is 16 MB against this box's 25 MB L2 (printed below),
  // and that is load-bearing. Measured here, same three kernels, only n
  // changed: 8 MB gives 1.25x, 16 MB gives 1.28x, 24 MB gives 1.07x, 32 MB
  // gives 1.04x, 64 MB gives 0.95x. Past the L2 both kernels are DRAM-bound,
  // every difference this puzzle is about is gone, and PolyFour's
  // quarter-sized grid starts to actively hurt. A relationship you can only
  // see inside one regime is still a real relationship, but you have to say
  // which regime. The full sweep is in SOLUTION.md.
  const int n = 2000003;
  const int TPB = 256;
  const int ELEMS = 4;  // must match PolyFour's compile-time ELEMS
  const int BLOCKS_ONE = cdiv(n, TPB);
  const int BLOCKS_FOUR = cdiv(cdiv(n, ELEMS), TPB);
  const int WARMUP = 3;
  const int ITERS = 20;
  const int REPS = 201;
  // How many of those reps the ratios are taken over: the quietest KEEP of the
  // REPS, by total time across the three kernels. Odd, so the median is a
  // measured sample rather than the average of two. See quietest_reps().
  const int KEEP = 21;
  // 63 chained float FMAs against a double accumulator over character-
  // identical coefficients. The worst deviation measured over all 2000003
  // outputs on this box is 2.364e-7 relative (index 813587, want 4.03478), and
  // it is the same for all three kernels because the arithmetic is the same.
  // 1e-6 would clear that by only 4.2x; 1e-5 is the tightest decade that
  // clears it by more than 10x, at 42x.
  const float tol = 1e-5f;

  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

  printf("# poly: %d terms, Horner, c_k = 1/(k+1), n = %d\n", 64, n);
  printf("# grids: PolyOne/SmemHog %d x %d threads (1 elem/thread), PolyFour"
         " %d x %d (%d elem/thread)\n",
         BLOCKS_ONE, TPB, BLOCKS_FOUR, TPB, ELEMS);
  printf("# working set: a + out = %.2f MB, L2 = %.2f MB\n",
         2.0 * n * sizeof(float) / 1e6, prop.l2CacheSize / 1e6);

  // The four budgets an SM hands out, and therefore the four ways a kernel can
  // run out of room. Every number in the occupancy table below is an integer
  // division of one of these.
  printf("# SM budgets: %d threads, %d blocks, %d registers, %zu B shared"
         " (+%zu B/block reserved by the driver)\n",
         prop.maxThreadsPerMultiProcessor, prop.maxBlocksPerMultiProcessor,
         prop.regsPerMultiprocessor, prop.sharedMemPerMultiprocessor,
         prop.reservedSharedMemPerBlock);

  float* a = new float[n];
  float* got = new float[n];
  float* want = new float[n];
  fill_rand(a, n, 31);
  ref_poly64(a, want, n);

  float *d_a, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

  // The occupancy table is printed unconditionally, before anything is run and
  // whether or not the kernels are correct. It is not a diagnostic for the
  // timing section -- it is the instrument this puzzle is about, and it is a
  // property of the compiled code alone.
  //
  // Which means it describes the kernel you actually wrote, not the one you
  // meant to write. A shared array nothing writes is dead: ptxas deletes it,
  // the kernel's `smem B` reads 0, and its occupancy jumps to PolyOne's. That
  // is not the harness being lenient, it is the harness telling you the truth
  // about your binary.
  const Occ one = occupancy_of(PolyOne, TPB, prop);
  const Occ four = occupancy_of(PolyFour, TPB, prop);
  const Occ hog = occupancy_of(SmemHog, TPB, prop);
  printf("# occupancy at %d threads/block, from cudaFuncGetAttributes +"
         " cudaOccupancyMaxActiveBlocksPerMultiprocessor\n",
         TPB);
  printf("# %-10s %6s %11s %10s %10s %10s\n", "kernel", "regs", "smem B",
         "blocks/SM", "warps/SM", "occupancy");
  const Occ* occs[3] = {&one, &four, &hog};
  const char* names[3] = {"PolyOne", "PolyFour", "SmemHog"};
  for (int k = 0; k < 3; k++) {
    printf("  %-10s %6d %11zu %10d %10d %9.1f%%\n", names[k], occs[k]->regs,
           occs[k]->smem, occs[k]->blocks, occs[k]->blocks * TPB / 32,
           100.0 * occs[k]->frac);
  }

  // Correctness first, and only then timing: a fast wrong kernel is not a
  // result, and a performance *relationship* between two kernels means nothing
  // unless both of them compute the same right answer. Every kernel owns every
  // output, so poison the buffer with 0xff (a NaN bit pattern) -- an output
  // that is never written is a FAIL, not a lucky zero.
  //
  // Nothing in this section is gated or skipped, ever. The gate below applies
  // to one wall-clock assert; correctness is not a measurement of the machine.
  bool ok = true;

  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  PolyOne<<<BLOCKS_ONE, TPB>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("poly_one", got, want, n, tol) && ok;

  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  PolyFour<<<BLOCKS_FOUR, TPB>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("poly_four", got, want, n, tol) && ok;

  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  SmemHog<<<BLOCKS_ONE, TPB>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("smem_hog", got, want, n, tol) && ok;

  // Pure resource arithmetic on the compiled kernels: it times nothing, it
  // cannot be perturbed by an instrumented run or by a busy box, and it holds
  // or fails identically under compute-sanitizer and under ncu. So it is
  // checked whenever the kernels are correct, and it is never gated.
  if (ok) {
    if (hog.blocks < one.blocks) {
      printf("PASS smem_limits_occupancy (SmemHog %d blocks/SM < PolyOne %d,"
             " %.1f%% vs %.1f%%)\n",
             hog.blocks, one.blocks, 100.0 * hog.frac, 100.0 * one.frac);
    } else {
      printf("FAIL smem_limits_occupancy: SmemHog fits %d blocks/SM, PolyOne"
             " %d, expected strictly fewer\n",
             hog.blocks, one.blocks);
      printf("FAIL   SmemHog's shared array has to be declared AND written, and"
             " big enough that the shared-memory budget beats the thread"
             " budget; if it is not, the kernel is limited by the same thing"
             " PolyOne is and there is nothing to see\n");
      ok = false;
    }
  }

  // The timing section asserts a property of the *hardware*, so it is only
  // meaningful on an uninstrumented run. Setting P31_SKIP_TIMING (to anything)
  // runs correctness and the occupancy check alone;
  // problems/p31_occupancy/prof.mk sets it for the three compute-sanitizer
  // runs, where a wall clock measures the sanitizer.
  const bool skip_timing = getenv("P31_SKIP_TIMING") != nullptr;

  if (!ok) {
    printf("# timing skipped: a kernel is incorrect\n");
  } else if (skip_timing) {
    printf("# timing skipped: P31_SKIP_TIMING is set. All three kernels ran"
           " and were verified above, and the occupancy table is exact"
           " regardless; only the wall-clock comparison is off.\n");
  } else {
    long long* d_cycles;
    CHECK_CUDA(cudaMalloc(&d_cycles, sizeof(long long)));
    // Sampled here rather than immediately before bench_all: NVML resolves on
    // the first watch() and that costs ~14 ms one-off, which must not land
    // between the clock ramp below and the timed section. Subsequent samples
    // cost ~25 us. watch() unions, so the window simply widens to cover the
    // ramp as well.
    QuietCheck qc{"PolyFour", "Gelem/s", QUIET_GELEM, IDLE_GELEM};
    qc.watch();
    const float mhz_before = measure_sm_mhz(d_cycles);

    // Order matters by one place: PolyOne and PolyFour are adjacent, so the
    // asserted ratio is formed from samples taken back to back.
    const Job jobs[3] = {
        {"PolyOne", (const void*)PolyOne, dim3(BLOCKS_ONE)},
        {"PolyFour", (const void*)PolyFour, dim3(BLOCKS_FOUR)},
        {"SmemHog", (const void*)SmemHog, dim3(BLOCKS_ONE)},
    };
    void* args[] = {(void*)&d_a, (void*)&d_out, (void*)&n};
    std::vector<float> samples[3];
    bench_all(jobs, 3, TPB, args, WARMUP, ITERS, REPS, samples);
    qc.watch();
    const float mhz_after = measure_sm_mhz(d_cycles);
    CHECK_CUDA(cudaFree(d_cycles));

    printf("# SM clock: %.0f MHz before the timed section, %.0f MHz after"
           " (nvidia-smi reports 208 MHz throughout; it is wrong)\n",
           mhz_before, mhz_after);
    printf("# timing: %d interleaved reps of (%d warmup + %d timed iterations);"
           " all three evaluate all %d polynomials\n",
           REPS, WARMUP, ITERS, n);
    printf("# %-10s %9s %12s %11s\n", "kernel", "best ms", "ns/element",
           "occupancy");
    for (int k = 0; k < 3; k++) {
      const float best =
          *std::min_element(samples[k].begin(), samples[k].end());
      printf("  %-10s %9.4f %12.5f %10.1f%%\n", jobs[k].name, best,
             best * 1e6 / n, 100.0 * occs[k]->frac);
    }

    float rlo = 0.0f, rhi = 0.0f, hlo = 0.0f, hhi = 0.0f;
    const std::vector<int> quiet = quietest_reps(samples, 3, REPS, KEEP);
    const float ratio = paired_ratio(samples[0], samples[1], quiet, &rlo, &rhi);
    const float r_hog = paired_ratio(samples[2], samples[0], quiet, &hlo, &hhi);
    printf("# paired per-rep ratios, median over the %d quietest of %d reps:\n",
           KEEP, REPS);
    printf("#   poly_one / poly_four = %.4fx  [p25 %.4f, p75 %.4f]  asserted:"
           " two kernels at the SAME occupancy\n",
           ratio, rlo, rhi);
    printf("#   smem_hog / poly_one  = %.4fx  [p25 %.4f, p75 %.4f]  reported"
           " only: what the lost third of the occupancy actually costs\n",
           r_hog, hlo, hhi);

    // How fast the quiet window actually ran, and half of the answer to whether
    // anything above is a measurement of this GPU rather than of the rest of
    // the box. See QUIET_GELEM, and QuietCheck in common/puzzle_utils.cuh.
    const float quiet_ms = median_over(samples[1], quiet);
    const double quiet_gelem = n / (quiet_ms * 1e6);
    printf("# quiet-window PolyFour: %.5f ms = %.1f Gelem/s (validity floor:"
           " >= %.0f Gelem/s)\n",
           quiet_ms, quiet_gelem, QUIET_GELEM);
    qc.report();

    if (!qc.valid(quiet_gelem)) {
      qc.fail("occupancy_not_predictive");
      printf("FAIL   the ratios above are printed anyway and are real"
             " measurements -- of a saturated machine. Under external load both"
             " kernels become bound by the contention rather than by their own"
             " exposed load latency, and the ratio collapses toward 1.0; that"
             " is physics, not an estimator problem, and no amount of rep"
             " selection recovers it. The quietest %d of %d reps were already"
             " the best this run had.\n",
             KEEP, REPS);
      ok = false;
    } else if (ratio >= MARGIN) {
      printf("PASS occupancy_not_predictive (poly_one / poly_four = %.4fx >="
             " %.2fx, at %.1f%% vs %.1f%% occupancy)\n",
             ratio, MARGIN, 100.0 * one.frac, 100.0 * four.frac);
    } else {
      printf("FAIL occupancy_not_predictive: ratio %.4fx below margin %.2fx --"
             " the puzzle's performance claim did not hold on a quiet"
             " machine\n",
             ratio, MARGIN);
      printf("FAIL   PolyFour does the same total arithmetic in a quarter of"
             " the threads, %d elements per thread instead of one; it should be"
             " decisively faster per element, and it was not\n",
             ELEMS);
      printf("FAIL   this run passed the validity check -- no other process had"
             " a CUDA context on this GPU, and PolyFour cleared %.1f Gelem/s in"
             " its quiet window -- so external load is NOT the explanation."
             " Check that PolyFour really covers %d elements per thread strided"
             " by gridDim.x * blockDim.x, and that the runner launched it on"
             " cdiv(n, %d) threads rather than n.\n",
             quiet_gelem, ELEMS, ELEMS);
      printf("FAIL   (running under compute-sanitizer or ncu? set"
             " P31_SKIP_TIMING=1 -- instrumented wall-clock time is not a"
             " measurement of this machine)\n");
      ok = false;
    }
  }

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_out));
  delete[] a;
  delete[] got;
  delete[] want;
  return ok ? 0 : 1;
}
