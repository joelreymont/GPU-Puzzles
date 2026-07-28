#include "puzzle_utils.cuh"
#include "reference.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

extern __global__ void SaxpyScalar(const float* a, const float* b, float* out,
                                   float s, int n);
extern __global__ void SaxpyVec4(const float* a, const float* b, float* out,
                                 float s, int n);
extern __global__ void SaxpyMisaligned(const float* a, const float* b,
                                       float* out, float s, int n);

// The scalar multiplier, named rather than spelled twice. 0.75 is exactly
// representable, so it contributes no rounding of its own and every deviation
// the tolerance sees comes from the multiply-add.
constexpr float S = 0.75f;

// saxpy is two flops and one FFMA per element. The reference computes the same
// fused form in double (see ref_saxpy) and rounds once on the way back to
// float, so it is the more accurate of the two rather than a differently-sloppy
// one -- and on this box every one of the 1,500,003 outputs of all three
// kernels comes back BIT-IDENTICAL to it. Measured worst relative deviation,
// using the same |got - want| / max(|want|, 1) that compare() thresholds:
//
//   saxpy_scalar       0
//   saxpy_vec4         0
//   saxpy_misaligned   0
//
// The runner prints all three every run, so this is checkable rather than
// believable. The tolerance is not 0 anyway, and that is deliberate: exact
// agreement here is a consequence of nvcc contracting `s * a[i] + b[i]` into
// one FFMA, which is a compiler default (-fmad=true), not a property of the
// problem. Written as two roundings instead of one the deviation would be about
// one ulp -- 6e-08 on values of this size -- and that kernel would still be a
// correct saxpy. 1e-6 clears an ulp by more than a decade and would still catch
// any real indexing bug, all of which produce O(1) errors here.
constexpr float TOL = 1e-6f;

// The two asserted performance claims. Both are floors on measured effects
// rather than descriptions of them, and both are ratios of PAIRED per-rep
// measurements (see bench_all): each rep times all three kernels a few hundred
// microseconds apart, the ratio is formed inside the rep, and the median across
// the reps is the estimate.
//
// A third relationship -- misaligned against the ALIGNED SCALAR kernel, which
// is the cleanest statement of the alignment effect because the two kernels
// differ by one character and by nothing else (ncu: identical instruction
// counts, identical request counts, 375,002 load sectors against 421,877) -- is
// measured and printed every run but deliberately NOT asserted. It is the least
// stable of the three, and the reason is worth knowing. SaxpyScalar is the one
// kernel sitting on the boundary between sector-bound and latency-bound, and
// which side of it lands on is decided by the rest of the box. Measured over
// the 70 gated-in runs of a 105-run campaign its best time is bimodal, and this
// ratio follows the split exactly:
//
//   SaxpyScalar 0.0108-0.0111 ms (28 runs, at the sector roof)
//       misaligned/scalar 1.053x-1.083x   -- about the 13/12 = 1.083x predicted
//   SaxpyScalar 0.0112-0.0143 ms (42 runs, latency-bound)
//       misaligned/scalar 0.997x-1.045x   -- the extra sector is free
//
// The second mode dips BELOW 1.00, so unlike MARGIN_VEC4 there is not even a
// safe sign to assert here: on a run where SaxpyScalar is latency-bound, the
// misaligned kernel can measure marginally faster than the aligned one. So the
// number is reported instead -- and SaxpyScalar's best ms on the table above
// tells you which mode you got.
//
// MARGIN_VEC4 -- scalar / vec4. The float4 kernel moves the identical bytes in
// a quarter of the requests. It does NOT move fewer sectors: ncu reports
// 375,002 global load sectors for both, to the sector (see prof.mk). So what it
// buys is request and instruction count, and against a kernel this close to the
// sector roof that is worth per cent, not a factor: measured 1.031x-1.355x over
// about 110 gated-in runs, median 1.07x.
//
// The margin is 1.00 rather than something inside that range, and the reason is
// the block-size sweep in the README: at 128 threads per block SaxpyScalar
// reaches the sector roof on its own and this ratio measures exactly 1.000x.
// The size of the gap is a property of the launch geometry, not of
// vectorisation, so a floor placed inside the measured range would be asserting
// something true only of this configuration. What IS a property of
// vectorisation is the sign: the same bytes in a quarter of the requests cannot
// cost more, so vec4 must not be slower. That claim holds at every block size
// measured, and it still catches the failure this assert exists for -- a float4
// kernel whose lanes read a[4*i+k] instead of the float4 view turns a 4-sector
// request into a 16-sector one and measures about 3.4x SLOWER, not 3 % faster.
//
// MARGIN_SPAN -- misaligned / vec4. The full span of the puzzle: the worst way
// to ask for these bytes against the best. It is the most stable of the three,
// because SaxpyVec4 is pinned to the sector path in both of SaxpyScalar's
// modes. Measured 1.102x-1.364x over the same 70 runs, median 1.15x.
//
// Both margins sit several points below the lowest value ever measured on a
// gated-in run, and every gated-in run passed both asserts. All three ratios are
// printed every run with their inter-quartile ranges whether the asserts fire
// or not, and the full distributions are in the README.
constexpr float MARGIN_VEC4 = 1.00f;
constexpr float MARGIN_SPAN = 1.05f;

// The throughput floor both asserts sit behind: half of this runner's
// measurement-validity check, the other half being the competing-process
// detector in common/puzzle_utils.cuh (see QuietCheck there for the design and
// for the three outcomes a timing section has).
//
// Why a validity check at all. This is a shared-memory SoC: one LPDDR5X behind
// both the GPU and the CPUs. When something else on the box takes a large share
// of it, all three kernels here stop being limited by their own sector traffic
// and start being limited by the contention -- and the ratios between them
// collapse toward 1.0. That is physics, not noise: under an external bottleneck
// the extra sector genuinely costs nothing, because the sector path is no longer
// the constraint. No estimator recovers an effect that is not present, and no
// margin below the contended floor would be a claim worth making (the floor is
// 1.00).
//
// So the runner establishes that it got a measurement before it asserts anything
// about one, and a run that did not get one FAILs as measurement_invalid rather
// than passing quietly. It never skips: a busy box is not a wrong kernel, but it
// is not a green suite either.
//
// SaxpyVec4 is the reference kernel because it has the least slack of its own:
// on a settled idle box it moves the 18 MB at 1748-1759 GB/s, run after run,
// with a quiet-window spread under 2 % inside any one run and under 1 % across
// runs. That tightness is what makes a floor possible at all.
//
// Where the floor goes was re-measured on this box for these semantics, because
// a floor that fires now costs a red suite rather than a skipped line. What it
// is really demanding is the settled state above, and the honest description of
// it is a settledness check rather than a contention check:
//
//   this box has an idle-but-unsettled state, and it takes minutes to clear
//   after a heavy memory load. During it, with the GPU verifiably free (the
//   detector below reports zero competing processes) and the CPUs at a load
//   average under 1, fifteen consecutive runs measured 1064.8-1505.0 GB/s.
//
//   under load the rates overlap that: twenty CPU streamer threads left ten runs
//   at 1602-1736 GB/s, over this floor, on a machine under heavy memory load. An
//   absolute rate is a weak proxy for who else is on the machine. Asking the
//   driver, which is what the other detector does, is not a proxy at all.
//
// The floor stays at 1600 anyway, and the reason is MARGIN_SPAN rather than the
// rate. That margin is 1.05 and it has almost no headroom in its low tail: over
// 25 runs on an idle box misaligned/vec4 measured 1.0864-1.2830, but a run at
// 914.1 GB/s came in at 1.0495 and one under 40 CPU streamer threads at 546.9
// GB/s came in at 1.0257, with scalar/vec4 at 1.0020 against MARGIN_VEC4's 1.00
// -- four thousandths from inverting. The margin only reliably holds in the
// settled mode, so the floor demands the settled mode.
//
// That is a deliberate choice about which mistake to make. At 1600 an unsettled
// run FAILs as measurement_invalid: loud, non-zero, and explicitly not a claim
// about the kernels. Lowered to admit the unsettled runs it would turn those
// same runs into FAILs of misalignment_costs_sectors -- blaming SaxpyMisaligned
// for a machine that had not settled. Of the two, the first is the one this file
// is willing to print. It does mean a red suite on a box that has not settled;
// the answer to that is to let it settle, which is what the diagnosis says.
constexpr double QUIET_GBPS = 1600.0;

// What SaxpyVec4 reaches on a settled idle box, quoted in the
// measurement_invalid diagnosis so a reader can see how far off an invalid run
// was. Measured, not asserted: the low end of the settled idle mode above.
constexpr double IDLE_GBPS = 1748.5;

// The same quantity compare() thresholds -- |got - want| / max(|want|, 1) --
// but reported rather than only asserted. The tolerance above was chosen from
// these numbers, so the numbers belong in the output.
static double max_rel_dev(const float* got, const float* want, int n, int* at) {
  double worst = 0.0;
  *at = 0;
  for (int i = 0; i < n; i++) {
    if (std::isnan(got[i])) {
      *at = i;
      return HUGE_VAL;
    }
    const double d = fabs((double)got[i] - (double)want[i]) /
                     fmax(fabs((double)want[i]), 1.0);
    if (d > worst) {
      worst = d;
      *at = i;
    }
  }
  return worst;
}

// A deliberate undefined-behaviour demonstration, and the only place in this
// repository where a CUDA error is expected instead of fatal. It is NOT part of
// the puzzle -- it lives here in the immutable harness, not in either kernel
// tree, so it is identical whichever kernels are being run.
//
// `a` is a cudaMalloc base, so it is at least 256-byte aligned; `a + 1` is that
// base plus four bytes, and casting it to `const float4*` is undefined
// behaviour by the C++ object model regardless of what the hardware does. What
// this hardware does about it is the point.
__global__ void MisalignedFloat4Probe(const float* a, float* out, int nv) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nv) {
    const float4 v = reinterpret_cast<const float4*>(a + 1)[i];
    out[i] = v.x + v.y + v.z + v.w;
  }
}

// Measures the SM clock, in MHz, by spinning a single warp for a known number
// of SM cycles and timing it on the wall clock.
//
// This is not decoration. nvidia-smi on this box reports `clocks.sm` as 208 MHz
// and pstate P8 even while the GPU is at 91 % utilisation drawing 64 W, so the
// only way to know what clock a measurement was taken at is to ask the device
// itself. It matters here because SaxpyScalar sits close enough to the memory
// roof that a low SM clock is the difference between it being memory-bound and
// it being issue-bound, and that shows up in the ratios below. One warp
// spinning on clock64() draws almost no power, so this also serves as the
// clock ramp: the GPU idles at 208 MHz and needs to be woken before anything
// downstream of it is a measurement of the memory system.
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

// One timed kernel: a name, a handle, and its launch shape. All three take the
// same argument pack; only the grid differs, because the float4 kernel covers
// four elements per thread.
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
// one rep of each in turn gives every kernel the same distribution of machine
// states.
//
// Keeping every sample rather than only the minimum is what lets the caller
// form PAIRED ratios -- kernel A and kernel B as measured inside the same rep,
// a few hundred microseconds apart -- and take the median of those. That is a
// deliberate departure from the ratio-of-minima this repository's earlier
// timing puzzles use, and it was chosen by measurement, not taste. The effects
// here are 6-15 %, small enough that two minima drawn from different machine
// states swamp them: measured under a GPU bandwidth hog and eight CPU memory
// streamers, the misaligned/scalar RATIO OF MINIMA scattered over 0.93x-1.48x
// while the median paired ratio stayed inside 1.094x-1.120x on the same runs.
// A ratio of minima is the right estimator when the effect is a large multiple
// (puzzle 32's is 1.6x, puzzle 33's is 4x); it is the wrong one at 8 %.
//
// Launched through cudaLaunchKernel rather than <<<>>> so the loop is written
// once: CUDA does not allow <<<>>> through a function pointer, but the
// host-side stub address is a perfectly good kernel handle.
static void bench_all(const Job* jobs, int nk, int tpb, void** args, int warmup,
                      int iters, int reps, std::vector<float>* samples) {
  for (int k = 0; k < nk; k++) {
    samples[k].clear();
    for (int i = 0; i < warmup; i++) {
      CHECK_CUDA(cudaLaunchKernel(jobs[k].kern, jobs[k].grid, dim3(tpb), args,
                                  0, 0));
    }
  }
  CHECK_LAUNCH();

  for (int r = 0; r < reps; r++) {
    for (int k = 0; k < nk; k++) {
      GpuTimer timer;
      timer.start();
      for (int i = 0; i < iters; i++) {
        CHECK_CUDA(cudaLaunchKernel(jobs[k].kern, jobs[k].grid, dim3(tpb), args,
                                    0, 0));
      }
      samples[k].push_back(timer.stop_ms() / iters);
      CHECK_CUDA(cudaGetLastError());
    }
  }
}

// Ranks the reps by how quiet the machine was during each one -- the sum of the
// three kernels' samples in that rep -- and returns the indices of the quietest
// `keep`, in rep order.
//
// Contention only ever makes a measurement slower, never faster, so the reps
// with the smallest total are the ones least contaminated by whatever else the
// box was doing. This is the same principle the earlier timing puzzles apply
// when they keep each kernel's fastest rep, moved up one level: the unit
// selected is the matched TRIPLE, so the three kernels are still being compared
// against each other inside a single machine state rather than each against its
// own best moment.
//
// It cannot flatter a slow kernel. The ranking is on the rep total, so a kernel
// that is uniformly slow is slow in every rep, quiet or not, and its ratio is
// unchanged. What the ranking removes is the rep in which some other process
// took the memory system away from all three at once -- which on this box, with
// its single LPDDR5X shared between the GPU and the CPUs, is not a rare event
// and is not a fact about alignment.
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

// Median of the per-rep ratio of two kernels over the given reps. `lo` and `hi`
// come back as the quartiles, so the caller can print the spread the median
// came out of.
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

  // n = 1,500,003, and every digit of it is doing something.
  //
  // Working set: a + b + out is 18.0 MB against this box's 25.17 MB L2
  // (printed below), so the whole problem stays resident between reps and what
  // is being timed is the L1<->L2 sector path rather than DRAM. That is the
  // only regime where any of this is visible: past the L2 all three kernels go
  // DRAM-bound at ~250 GB/s and converge, because DRAM sees the same bytes in
  // all three cases. The sweep is in the README.
  //
  // n % 4 == 3, so the float4 view covers 1,500,000 elements and three are left
  // over. A float4 kernel with no tail leaves those three holding the runner's
  // 0xff poison, which reads back NaN.
  //
  // n % 256 == 99 and n % 32 == 3, so the last block of the scalar kernels runs
  // 99 live threads out of 256 and the last warp of it runs 3 out of 32. Guards
  // are exercised at both granularities.
  const int n = 1500003;
  const int TPB = 256;
  const int nv = n / 4;             // float4 elements SaxpyVec4 covers
  const int tail = n - nv * 4;      // scalar elements it does not
  const dim3 grid_scalar(cdiv(n, TPB));
  const dim3 grid_vec4(cdiv(nv, TPB));

  const int WARMUP = 3;
  const int ITERS = 20;
  const int REPS = 201;
  // How many of those reps the ratios are taken over: the quietest KEEP of the
  // REPS, by total time across the three kernels. Odd, so the median is a
  // measured sample rather than the average of two. See quietest_reps().
  const int KEEP = 21;

  // 32-byte sectors, four to a 128-byte line, is the granularity this whole
  // puzzle is arithmetic modulo. cudaDeviceProp does not report either number,
  // so they are constants here rather than a query, and they are printed
  // because every prediction below is made out of them.
  const int SECTOR = 32;
  const int WARP = 32;

  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

  printf("# saxpy: out[i] = %.2f * a[i] + b[i], n = %d\n", S, n);
  printf("# scalar launch: %u blocks of %d threads, one element per thread,"
         " last block %d live\n",
         grid_scalar.x, TPB, n - (int)(grid_scalar.x - 1) * TPB);
  printf("# vec4 launch:   %u blocks of %d threads, one float4 per thread,"
         " %d float4 + %d scalar tail\n",
         grid_vec4.x, TPB, nv, tail);
  printf("# working set: a + b + out = %.2f MB, L2 = %.2f MB\n",
         3.0 * n * sizeof(float) / 1e6, prop.l2CacheSize / 1e6);
  printf("# a warp of %d lanes wants %d contiguous bytes = %d sectors of %d B"
         " when aligned, %d when not\n",
         WARP, WARP * (int)sizeof(float), WARP * (int)sizeof(float) / SECTOR,
         SECTOR, WARP * (int)sizeof(float) / SECTOR + 1);

  // `a` holds n + 1 floats. SaxpyMisaligned reads a[i + 1] for i in [0, n), so
  // the last element it touches is a[n] -- a real element of a real allocation,
  // not an overrun. The misalignment being measured is a property of where the
  // reads START, and nothing here runs off the end; memcheck agrees.
  float* a = new float[n + 1];
  float* b = new float[n];
  float* got = new float[n];
  float* want = new float[n];
  float* want_shift = new float[n];
  // Two seeds, not one: a and b are different lengths but a single seed would
  // still make them share a value prefix, and independent operands are one
  // fewer coincidence for a wrong index expression to hide behind.
  fill_rand(a, n + 1, 35);
  fill_rand(b, n, 36);
  ref_saxpy(a, b, want, S, n);
  ref_saxpy_shift(a, b, want_shift, S, n);

  float *d_a, *d_b, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, (n + 1) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a, (n + 1) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

  // The guarantee the float4 kernel is standing on, checked rather than
  // assumed. cudaMalloc is documented to return memory suitably aligned for any
  // variable type; in practice on this box every allocation comes back on a
  // 256-byte boundary, and 16 is all reinterpret_cast<float4*> needs.
  printf("# cudaMalloc alignment: a %% 256 = %zu, b %% 256 = %zu, out %% 256 ="
         " %zu (float4 needs 16)\n",
         (size_t)((uintptr_t)d_a % 256), (size_t)((uintptr_t)d_b % 256),
         (size_t)((uintptr_t)d_out % 256));

  // Correctness first, and only then timing: a fast wrong kernel is not a
  // result, and a performance relationship between two kernels means nothing
  // unless both compute the same right answer. Every kernel owns every element
  // of `out`, so the buffer is poisoned with 0xff -- a NaN bit pattern --
  // before each launch. An element that is never written is a FAIL, not a lucky
  // zero, which is exactly how a missing float4 tail presents.
  bool ok = true;
  int at = 0;

  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  SaxpyScalar<<<grid_scalar, TPB>>>(d_a, d_b, d_out, S, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  printf("# saxpy_scalar worst relative deviation: %.3g (at index %d, tolerance"
         " %.0e)\n",
         max_rel_dev(got, want, n, &at), at, TOL);
  ok = compare("saxpy_scalar", got, want, n, TOL) && ok;

  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  SaxpyVec4<<<grid_vec4, TPB>>>(d_a, d_b, d_out, S, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  printf("# saxpy_vec4 worst relative deviation: %.3g (at index %d, tolerance"
         " %.0e)\n",
         max_rel_dev(got, want, n, &at), at, TOL);
  ok = compare("saxpy_vec4", got, want, n, TOL) && ok;

  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  SaxpyMisaligned<<<grid_scalar, TPB>>>(d_a, d_b, d_out, S, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  printf("# saxpy_misaligned worst relative deviation: %.3g (at index %d,"
         " tolerance %.0e)\n",
         max_rel_dev(got, want_shift, n, &at), at, TOL);
  ok = compare("saxpy_misaligned", got, want_shift, n, TOL) && ok;

  // The timing section asserts a property of the hardware, so it is only
  // meaningful on an uninstrumented run. Setting P35_SKIP_TIMING (to anything)
  // runs correctness alone; problems/p35_alignment/prof.mk sets it for the
  // three compute-sanitizer runs, where a wall clock measures the sanitizer
  // rather than the memory system.
  const bool skip_timing = getenv("P35_SKIP_TIMING") != nullptr;

  if (!ok) {
    printf("# timing skipped: a kernel is incorrect\n");
  } else if (skip_timing) {
    printf("# timing skipped: P35_SKIP_TIMING is set. All three kernels ran and"
           " were verified above; only the wall-clock comparison is off.\n");
  } else {
    long long* d_cycles;
    CHECK_CUDA(cudaMalloc(&d_cycles, sizeof(long long)));
    // Sampled here rather than immediately before bench_all: NVML resolves on
    // the first watch() and that costs ~14 ms one-off, which must not land
    // between the clock ramp below and the timed section. Subsequent samples
    // cost ~25 us. watch() unions, so the window simply widens to cover the
    // ramp as well.
    QuietCheck qc{"SaxpyVec4", "GB/s", QUIET_GBPS, IDLE_GBPS};
    qc.watch();
    const float mhz_before = measure_sm_mhz(d_cycles);

    // Order matters by one place: SaxpyScalar sits between the other two, so
    // each of the two paired ratios below is formed from adjacent samples.
    const Job jobs[3] = {
        {"SaxpyVec4", (const void*)SaxpyVec4, grid_vec4},
        {"SaxpyScalar", (const void*)SaxpyScalar, grid_scalar},
        {"SaxpyMisaligned", (const void*)SaxpyMisaligned, grid_scalar},
    };
    void* args[] = {(void*)&d_a, (void*)&d_b, (void*)&d_out, (void*)&S,
                    (void*)&n};
    std::vector<float> samples[3];
    bench_all(jobs, 3, TPB, args, WARMUP, ITERS, REPS, samples);
    qc.watch();
    const float mhz_after = measure_sm_mhz(d_cycles);
    CHECK_CUDA(cudaFree(d_cycles));

    // All three kernels read a, read b and write out exactly once, so the
    // traffic is fixed at 3 * n * 4 B no matter which one moves it and GB/s is
    // a fair second axis: same numerator for all three. It is *bytes*, not
    // sectors -- SaxpyMisaligned moves the same 18 MB of user data through 8.3 %
    // more sectors, which is precisely why its GB/s is lower.
    const double bytes = 3.0 * n * sizeof(float);
    printf("# SM clock: %.0f MHz before the timed section, %.0f MHz after"
           " (nvidia-smi reports 208 MHz and P8 throughout; it is wrong)\n",
           mhz_before, mhz_after);
    printf("# timing: %d interleaved reps of (%d warmup + %d timed iterations);"
           " all three move %.2f MB\n",
           REPS, WARMUP, ITERS, bytes / 1e6);
    printf("# %-18s %9s %12s\n", "kernel", "best ms", "eff GB/s");
    for (int k = 0; k < 3; k++) {
      const float best = *std::min_element(samples[k].begin(),
                                           samples[k].end());
      printf("  %-18s %9.5f %12.1f\n", jobs[k].name, best,
             bytes / (best * 1e6));
    }

    float vlo = 0.0f, vhi = 0.0f, mlo = 0.0f, mhi = 0.0f, slo = 0.0f,
          shi = 0.0f;
    const std::vector<int> quiet =
        quietest_reps(samples, 3, REPS, KEEP);
    const float r_vec4 = paired_ratio(samples[1], samples[0], quiet, &vlo,
                                      &vhi);
    const float r_misaligned = paired_ratio(samples[2], samples[1], quiet, &mlo,
                                            &mhi);
    const float r_span = paired_ratio(samples[2], samples[0], quiet, &slo,
                                      &shi);
    printf("# paired per-rep ratios, median over the %d quietest of %d reps:\n",
           KEEP, REPS);
    printf("#   scalar / vec4       = %.4fx  [p25 %.4f, p75 %.4f]  asserted\n",
           r_vec4, vlo, vhi);
    printf("#   misaligned / vec4   = %.4fx  [p25 %.4f, p75 %.4f]  asserted\n",
           r_span, slo, shi);
    printf("#   misaligned / scalar = %.4fx  [p25 %.4f, p75 %.4f]  reported"
           " only: the cleanest statement of the alignment effect, and the"
           " least stable of the three -- 1.053-1.083 when SaxpyScalar lands"
           " at the sector roof (best ms 0.0108-0.0111 above), 0.997-1.045"
           " when it lands latency-bound instead (0.0112-0.0143). See the"
           " README.\n",
           r_misaligned, mlo, mhi);

    // How fast the quiet window actually ran, and half of the answer to whether
    // anything above is a measurement of this GPU rather than of the rest of the
    // box. See QUIET_GBPS, and QuietCheck in common/puzzle_utils.cuh.
    const float quiet_ms = median_over(samples[0], quiet);
    const double quiet_gbps = bytes / (quiet_ms * 1e6);
    printf("# quiet-window SaxpyVec4: %.5f ms = %.1f GB/s (validity floor:"
           " >= %.0f GB/s)\n",
           quiet_ms, quiet_gbps, QUIET_GBPS);
    qc.report();

    const bool measurable = qc.valid(quiet_gbps);

    if (!measurable) {
      qc.fail("misalignment_costs_sectors and vec4_never_slower");
      printf("FAIL   the three ratios above are printed anyway and are real"
             " measurements -- of a saturated machine. Under external load all"
             " three kernels become bound by the contention rather than by"
             " their own sector traffic and the ratios collapse toward 1.0;"
             " that is physics, not an estimator problem, and no amount of"
             " rep selection recovers the effect. The quietest %d of %d reps"
             " were already the best this run had.\n",
             KEEP, REPS);
      ok = false;
    }

    if (measurable && r_span >= MARGIN_SPAN) {
      printf("PASS misalignment_costs_sectors (misaligned / vec4 = %.4fx >="
             " %.2fx, from one extra sector per %d-lane load request)\n",
             r_span, MARGIN_SPAN, WARP);
    } else if (measurable) {
      printf("FAIL misalignment_costs_sectors: ratio %.4fx below margin %.2fx"
             " -- the puzzle's performance claim did not hold on a quiet"
             " machine\n",
             r_span, MARGIN_SPAN);
      printf("FAIL   SaxpyMisaligned reads a[i + 1] where the other two read"
             " a[i]. Same arithmetic, same bytes, same coalescing -- the only"
             " difference is that each warp's %d contiguous bytes now begin 4 B"
             " past a %d-byte boundary and therefore span %d sectors instead of"
             " %d. If it is not the slowest of the three, check that it really"
             " indexes a[i + 1] and not a[i]; and check the misaligned/scalar"
             " line above, which isolates the same effect.\n",
             WARP * (int)sizeof(float), 4 * SECTOR,
             WARP * (int)sizeof(float) / SECTOR + 1,
             WARP * (int)sizeof(float) / SECTOR);
      printf("FAIL   this is the most contention-resistant of the three ratios"
             " -- SaxpyVec4 is pinned to the sector path in both of"
             " SaxpyScalar's modes -- and this run passed the validity check"
             " above: no other process had a CUDA context on this GPU, and"
             " SaxpyVec4 cleared %.1f GB/s. A busy machine is not the"
             " explanation, and it is not what this FAIL is claiming.\n",
             quiet_gbps);
      printf("FAIL   (running under compute-sanitizer or ncu? set"
             " P35_SKIP_TIMING=1 -- instrumented wall-clock time is not a"
             " measurement of this machine)\n");
      ok = false;
    }

    if (measurable && r_vec4 >= MARGIN_VEC4) {
      printf("PASS vec4_never_slower (scalar / vec4 = %.4fx >= %.2fx; same"
             " sectors, a quarter of the requests -- see the README for why the"
             " size of that gap belongs to the block size and not to"
             " vectorisation)\n",
             r_vec4, MARGIN_VEC4);
    } else if (measurable) {
      printf("FAIL vec4_never_slower: ratio %.4fx below margin %.2fx -- the"
             " puzzle's performance claim did not hold on a quiet machine\n",
             r_vec4, MARGIN_VEC4);
      printf("FAIL   SaxpyVec4 issues one 128-bit load per operand per thread"
             " where SaxpyScalar issues four 32-bit ones for the same four"
             " elements. It cannot move fewer sectors -- the bytes are the same"
             " -- but it must not move MORE. If it is slower, the most likely"
             " cause is that the float4 view is not being indexed by float4"
             " element: four separate reads of a[4 * i + k] per thread is a"
             " stride-4 warp access and costs 16 sectors per request, not 4.\n");
      printf("FAIL   moderate contention pushes this ratio UP, not down, and"
             " severe contention is what the validity check above rules out, so"
             " a busy machine does not explain a value below the margin here."
             " What"
             " does is a launch geometry in which SaxpyScalar is not"
             " issue-limited at all: at 128 threads per block instead of %d the"
             " scalar kernel reaches the sector roof on its own and this ratio"
             " measures 1.00x. That is a real result, not a failure -- but it"
             " is not this puzzle's configuration.\n",
             TPB);
      ok = false;
    }
  }

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_out));
  delete[] a;
  delete[] b;
  delete[] got;
  delete[] want;
  delete[] want_shift;

  // The undefined-behaviour demonstration, deliberately last, because it is not
  // survivable: a misaligned global access raises cudaErrorMisalignedAddress,
  // which is a STICKY error -- every subsequent call on this context returns it
  // forever, including cudaFree. So everything above has already run and every
  // buffer above has already been released, this allocates two small buffers of
  // its own, and the process exits immediately afterwards without freeing them.
  // The OS reclaims the context; there is nothing else that can be done with a
  // context in this state, and pretending otherwise would be the real error.
  //
  // Skipped under P35_SKIP_UB_PROBE, which prof.mk sets for the sanitizer runs:
  // memcheck reports this fault as four genuine "Invalid __global__ read of
  // size 16 bytes ... is misaligned" errors, which is exactly right and would
  // otherwise fail `make check` for a fault the harness went out of its way to
  // cause.
  if (getenv("P35_SKIP_UB_PROBE") != nullptr) {
    printf("# UB probe skipped: P35_SKIP_UB_PROBE is set.\n");
    return ok ? 0 : 1;
  }

  const int PROBE_N = 1024;
  float *p_a, *p_out;
  CHECK_CUDA(cudaMalloc(&p_a, (PROBE_N + 4) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&p_out, PROBE_N * sizeof(float)));
  CHECK_CUDA(cudaMemset(p_a, 0, (PROBE_N + 4) * sizeof(float)));
  MisalignedFloat4Probe<<<cdiv(PROBE_N, TPB), TPB>>>(p_a, p_out, PROBE_N);
  // Not CHECK_LAUNCH: this is the one call in the repository whose error is the
  // result. The launch itself succeeds -- the address is only known per lane at
  // execution time -- so the fault surfaces at the synchronise, asynchronously,
  // which is itself part of the lesson.
  const cudaError_t probe_launch = cudaGetLastError();
  const cudaError_t probe_sync = cudaDeviceSynchronize();
  printf("# UB probe: reinterpret_cast<const float4*>(a + 1), where a is a"
         " cudaMalloc base with a %% 256 = %zu\n",
         (size_t)((uintptr_t)p_a % 256));
  printf("#   launch -> %s, synchronise -> %s (\"%s\")\n",
         cudaGetErrorName(probe_launch), cudaGetErrorName(probe_sync),
         cudaGetErrorString(probe_sync));
  if (probe_sync == cudaErrorMisalignedAddress) {
    printf("PASS misaligned_float4_faults (the C++ object model calls this"
           " undefined; this device calls it cudaErrorMisalignedAddress and"
           " kills the context)\n");
  } else {
    printf("FAIL misaligned_float4_faults: expected cudaErrorMisalignedAddress"
           " from a float4 access 4 B past an aligned base, got %s\n",
           cudaGetErrorName(probe_sync));
    printf("FAIL   this is a hardware fact, not a timing one, so contention"
           " cannot explain it. If this device has stopped faulting on a"
           " 16-byte access at a 4-byte-aligned address, the README's central"
           " claim needs re-measuring -- the cast is still undefined behaviour"
           " either way.\n");
    ok = false;
  }
  return ok ? 0 : 1;
}
