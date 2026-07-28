#include "puzzle_utils.cuh"
#include "reference.hpp"

#include <cuda_fp16.h>

extern __global__ void GemmNaiveFp32(const float* a, const float* b, float* c,
                                     int M, int N, int K);
extern __global__ void GemmWmma(const __half* a, const __half* b, float* c,
                                int M, int N, int K);

// Tolerances, both measured on this box rather than guessed, and both far
// tighter than a matmul tolerance usually is -- because the usual reason a
// half-precision matmul needs a loose tolerance has been removed from this
// comparison by construction.
//
// The runner rounds its random inputs to __half and converts them straight
// back to float, and only then hands anything to anyone. So the CPU reference,
// GemmNaiveFp32 and GemmWmma all consume bit-identical values, and input
// quantisation cancels out of the deviation entirely. What is left is the
// accumulation: double for the reference, one serial float chain for the naive
// kernel, and the tensor core's fixed-order fp32 accumulate for the other.
//
// Measured on this box, that round trip is worth three decades: reference the
// unrounded floats instead -- what a half-precision matmul benchmark normally
// does -- and the identical GemmWmma output deviates by 5.09e-03 rather than
// 2.89e-06, while the FP32 kernel run the same way is unchanged at 4.28e-06.
// The difference is __float2half on the way in and nothing else.
//
// Measured worst relative deviation over the whole 320 x 256 output, using the
// same |got - want| / max(|want|, 1) that compare() thresholds. Both kernels
// are deterministic and the numbers did not move a digit across ten
// consecutive runs:
//
//   gemm_naive_fp32   4.35e-06 at index 64350   -> tolerance 1e-04  (23.0x)
//   gemm_wmma         2.89e-06 at index  1374   -> tolerance 1e-04  (34.6x)
//
// Each tolerance is the tightest power of ten clearing its measured deviation
// by at least 10x. They land on the same decade, and the ordering inside it is
// the finding: the tensor-core kernel is the MORE accurate of the two, by 1.5x
// on the worst element. A 16x16x16 MMA with an fp32 accumulator is not
// approximate arithmetic. The half inputs cost nothing here because every
// kernel gets the same half-rounded values, and what is left favours the
// tensor core -- its 192 terms arrive as 12 chained fragment MMAs, each of
// which sums 16 products in one hardware step, against the naive kernel's
// single 192-long serial float chain where every partial sum is rounded.
//
// Every run prints both deviations above the verdict lines, so these two
// constants can be checked rather than believed.
constexpr float TOL_NAIVE = 1e-4f;
constexpr float TOL_WMMA = 1e-4f;

// The performance claim: the tensor-core kernel is at least MARGIN times
// faster than the FP32 kernel doing identical arithmetic on identical values.
//
// This is a floor on the effect, not a description of it. On an idle box the
// measured ratio is remarkably tight -- 3.988x to 4.040x over ten consecutive
// runs. But this is a shared-memory SoC where the GPU and the CPUs sit behind
// one LPDDR5X, and the tensor-core kernel is the one that fills 0.14 of a wave
// of this machine (ncu says so; see prof.mk), so it has the least of its own
// latency to hide and contention hurts it proportionally more. Across 62
// whole-run measurements including two concurrent GPU bandwidth hogs, twenty
// CPU memory streamers, and eight copies of this binary racing each other, the
// range was 3.279x to 4.519x. MARGIN sits below that contended floor rather
// than near the idle value, so the assert can never lie about the machine. The
// measured ratio is printed on the PASS line every run.
//
// Four is also not the size of the hardware effect, and the README says so at
// length: GemmWmma issues 17x fewer warp instructions than GemmNaiveFp32
// (141,760 against 2,432,000) and runs at ~6% of compute throughput against
// the naive kernel's ~50%. At 320 x 256 x 192 there is not enough work to fill
// one wave of 48 SMs, so what is being measured is two small kernels' latency,
// and the tensor core wins by 4x anyway.
constexpr float MARGIN = 3.0f;

// The throughput floor the timing section sits behind, together with the
// competing-process detector in common/puzzle_utils.cuh. QuietCheck there
// documents the design and the three outcomes a timing section has; what belongs
// here are this puzzle's numbers.
//
// GemmWmma is the reference kernel: it fills 0.14 of a wave of this machine, so
// it has the least of its own latency to hide and its throughput moves first when
// something else takes the LPDDR5X this SoC shares between the GPU and the CPUs.
// It is also the denominator of the asserted ratio.
//
// Measured on this box. Fifteen consecutive runs on a settled idle box:
// 5089.5-5161.7 GFLOP/s, a 1.4 % spread; the lowest figure seen on an idle GPU
// across every session was 5005.3.
//
// This floor is a tripwire rather than a discriminator, and the numbers say why.
// The problem is 320 x 256 x 192 and both kernels' working sets are small, so
// external load barely reaches them: under 40 CPU memory streamer threads over
// 40 GB, six runs measured 5079.0-5101.4 GFLOP/s with the ratio at 3.981-4.035.
// Under two GPU bandwidth hogs plus 20 CPU streamers -- a load that halves this
// box's L2-resident puzzles -- eight runs still measured 4542.7-4673.4 GFLOP/s
// with the ratio at 3.777-3.907, which is 26 % clear of MARGIN. No load this box
// could be made to produce brought this assert near its margin.
//
// So the floor is placed as a tripwire: 4000 GFLOP/s is 20 % below the lowest
// idle observation and 12 % below the worst contended one, and it exists to
// refuse a collapse deeper than anything above rather than to separate two
// populations that do not in fact separate here. What it must not be is set so
// high that it fires on a machine that was in fact quiet; the discrimination on
// this puzzle is done by the competing-process detector.
constexpr double QUIET_GFLOPS = 4000.0;

// What GemmWmma reaches on a settled idle box, quoted in the measurement_invalid
// diagnosis so a reader can see how far off an invalid run was. Measured, not
// asserted: the low end of the idle spread above.
constexpr double IDLE_GFLOPS = 5089.5;

// The same quantity compare() thresholds -- |got - want| / max(|want|, 1) --
// but reported rather than only asserted. The tolerances above were chosen
// from these numbers, so the numbers belong in the output.
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

// One timed kernel: its launch shape and its argument pack. The two kernels
// take different pointer types over different buffers and run on different
// grids, so there is nothing to share except the loop.
struct Job {
  const char* name;
  const void* kern;
  dim3 grid;
  dim3 block;
  void* args[6];
};

// Both kernels are timed together: one rep of each in turn, repeated, and each
// keeps its own fastest rep.
//
// Interleaving is not cosmetic. Timing every rep of one kernel and then every
// rep of the next lets a drift in machine state land entirely on one of them;
// one rep of each in turn gives both the same distribution of machine states.
// That matters more here than in any earlier puzzle, because the ratio being
// asserted is large and a drift that inflates only the denominator would
// inflate the claim.
//
// The fastest rep, not the mean, is the estimate: contention only ever makes a
// measurement slower, never faster, so the minimum is the observation least
// contaminated by whatever else the machine was doing.
//
// Launched through cudaLaunchKernel rather than <<<>>> so the loop is written
// once: CUDA does not allow <<<>>> through a function pointer, but the
// host-side stub address is a perfectly good kernel handle.
static void bench_all(Job* jobs, int nk, int warmup, int iters, int reps,
                      float* ms_out) {
  for (int k = 0; k < nk; k++) {
    ms_out[k] = -1.0f;
    for (int i = 0; i < warmup; i++) {
      CHECK_CUDA(cudaLaunchKernel(jobs[k].kern, jobs[k].grid, jobs[k].block,
                                  jobs[k].args, 0, 0));
    }
  }
  CHECK_LAUNCH();

  for (int r = 0; r < reps; r++) {
    for (int k = 0; k < nk; k++) {
      GpuTimer timer;
      timer.start();
      for (int i = 0; i < iters; i++) {
        CHECK_CUDA(cudaLaunchKernel(jobs[k].kern, jobs[k].grid, jobs[k].block,
                                    jobs[k].args, 0, 0));
      }
      const float ms = timer.stop_ms() / iters;
      CHECK_CUDA(cudaGetLastError());
      if (ms_out[k] < 0.0f || ms < ms_out[k]) ms_out[k] = ms;
    }
  }
}

int main() {
  print_device_info();

  // Every dimension is a multiple of 16 and that is not a convenience, it is
  // what the instruction accepts: a wmma fragment is a fixed 16 x 16 window,
  // there is no partial fragment, and no lane mask turns one into a partial
  // fragment. A K that is not a multiple of 16 makes the fragment loop read
  // past the end of every row on its last trip, and the README records the
  // measurement: it launches clean, memcheck reports 0 errors, and the answer
  // is garbage. Real libraries meet a ragged edge by padding the problem to
  // the fragment shape, never by ragging the fragment.
  //
  // Within that constraint the shape is picked to be awkward rather than
  // convenient: 20 x 16 warp tiles is non-square, K is 12 fragment steps, and
  // none of 20, 12 or 320/256/192 is a power of two. A kernel that quietly
  // assumed a square tile grid, or that K equalled one fragment, would produce
  // a wrong answer here rather than a lucky one.
  const int M = 320;
  const int N = 256;
  const int K = 192;
  const int MMA = 16;          // must match the kernels' compile-time MMA_*
  const int WARPS_X = 2;       // must match the kernels' compile-time WARPS_X
  const int WARPS_Y = 2;       // must match the kernels' compile-time WARPS_Y
  const int WARP = 32;
  const int NAIVE_TILE = 16;   // naive kernel: 16 x 16 = 256 threads per block

  const int tiles_m = cdiv(M, MMA);
  const int tiles_n = cdiv(N, MMA);
  const dim3 wmma_block(WARPS_X * WARPS_Y * WARP);
  const dim3 wmma_grid(cdiv(tiles_n, WARPS_X), cdiv(tiles_m, WARPS_Y));
  const dim3 naive_block(NAIVE_TILE, NAIVE_TILE);
  const dim3 naive_grid(cdiv(N, NAIVE_TILE), cdiv(M, NAIVE_TILE));

  const int WARMUP = 3;
  const int ITERS = 20;
  const int REPS = 15;  // see bench_all(): the assert needs the best, not mean

  const int na = M * K, nb = K * N, nc = M * N;

  printf("# gemm: C[%d x %d] = A[%d x %d] * B[%d x %d], all row-major\n", M, N,
         M, K, K, N);
  printf("# A, B: __half for GemmWmma and the exact same values as float for"
         " GemmNaiveFp32; C: float\n");
  printf("# wmma: %d x %d = %d warp tiles of %dx%d, %d fragment steps down K,"
         " %d mma_sync per warp\n",
         tiles_m, tiles_n, tiles_m * tiles_n, MMA, MMA, K / MMA, K / MMA);
  printf("# wmma launch: %u x %u blocks of %u threads (%d warps, %d x %d over"
         " the tile grid)\n",
         wmma_grid.x, wmma_grid.y, wmma_block.x, WARPS_X * WARPS_Y, WARPS_Y,
         WARPS_X);
  printf("# naive launch: %u x %u blocks of %u x %u threads, one thread per"
         " element of C\n",
         naive_grid.x, naive_grid.y, naive_block.x, naive_block.y);

  // Fill in float, round to __half, then convert straight back. `af` and `bf`
  // now hold the exact values of `ah` and `bh`, so the CPU reference, the FP32
  // kernel and the tensor-core kernel are all multiplying the same numbers and
  // the only thing that can separate their answers is how they accumulate.
  // Without this round trip the FP32 kernel would be solving a slightly
  // different problem and the comparison would measure __float2half instead of
  // the tensor core.
  //
  // Two seeds, not one: A and B are different lengths but a single seed would
  // still make them share a value prefix, and independent operands are one
  // fewer coincidence for a wrong index expression to hide behind.
  float* af = new float[na];
  float* bf = new float[nb];
  __half* ah = new __half[na];
  __half* bh = new __half[nb];
  float* got = new float[nc];
  float* want = new float[nc];
  fill_rand(af, na, 33);
  fill_rand(bf, nb, 34);
  for (int i = 0; i < na; i++) {
    ah[i] = __float2half(af[i]);
    af[i] = __half2float(ah[i]);
  }
  for (int i = 0; i < nb; i++) {
    bh[i] = __float2half(bf[i]);
    bf[i] = __half2float(bh[i]);
  }
  ref_gemm(af, bf, want, M, N, K);

  float *d_af, *d_bf, *d_c;
  __half *d_ah, *d_bh;
  CHECK_CUDA(cudaMalloc(&d_af, na * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_bf, nb * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_ah, na * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_bh, nb * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&d_c, nc * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_af, af, na * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_bf, bf, nb * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_ah, ah, na * sizeof(__half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_bh, bh, nb * sizeof(__half), cudaMemcpyHostToDevice));

  // Correctness first, and only then timing: a fast wrong kernel is not a
  // result, and a performance relationship between two kernels means nothing
  // unless both compute the same right answer. Each kernel owns every element
  // of C, so the buffer is poisoned with 0xff -- a NaN bit pattern -- before
  // each launch. A tile that is never written is a FAIL, not a lucky zero, and
  // a warp-tile mapping that leaves a hole shows up as a NaN at the exact
  // index rather than as a plausible-looking number.
  bool ok = true;
  int at = 0;

  CHECK_CUDA(cudaMemset(d_c, 0xff, nc * sizeof(float)));
  GemmNaiveFp32<<<naive_grid, naive_block>>>(d_af, d_bf, d_c, M, N, K);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_c, nc * sizeof(float), cudaMemcpyDeviceToHost));
  printf("# gemm_naive_fp32 worst relative deviation: %.3g (at index %d,"
         " tolerance %.0e)\n",
         max_rel_dev(got, want, nc, &at), at, TOL_NAIVE);
  ok = compare("gemm_naive_fp32", got, want, nc, TOL_NAIVE) && ok;

  CHECK_CUDA(cudaMemset(d_c, 0xff, nc * sizeof(float)));
  GemmWmma<<<wmma_grid, wmma_block>>>(d_ah, d_bh, d_c, M, N, K);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_c, nc * sizeof(float), cudaMemcpyDeviceToHost));
  printf("# gemm_wmma worst relative deviation: %.3g (at index %d, tolerance"
         " %.0e)\n",
         max_rel_dev(got, want, nc, &at), at, TOL_WMMA);
  ok = compare("gemm_wmma", got, want, nc, TOL_WMMA) && ok;

  // The timing section asserts a property of the hardware, so it is only
  // meaningful on an uninstrumented run. Setting P33_SKIP_TIMING (to anything)
  // runs correctness alone; problems/p33_tensor_cores/prof.mk sets it for the
  // three compute-sanitizer runs, where a wall clock measures the sanitizer
  // rather than the tensor pipe.
  const bool skip_timing = getenv("P33_SKIP_TIMING") != nullptr;

  if (!ok) {
    printf("# timing skipped: a kernel is incorrect\n");
  } else if (skip_timing) {
    printf("# timing skipped: P33_SKIP_TIMING is set. Both kernels ran and were"
           " verified above; only the wall-clock comparison is off.\n");
  } else {
    Job jobs[2] = {
        {"GemmNaiveFp32", (const void*)GemmNaiveFp32, naive_grid, naive_block,
         {(void*)&d_af, (void*)&d_bf, (void*)&d_c, (void*)&M, (void*)&N,
          (void*)&K}},
        {"GemmWmma", (const void*)GemmWmma, wmma_grid, wmma_block,
         {(void*)&d_ah, (void*)&d_bh, (void*)&d_c, (void*)&M, (void*)&N,
          (void*)&K}},
    };
    float ms[2];
    QuietCheck qc{"GemmWmma", "GFLOP/s", QUIET_GFLOPS, IDLE_GFLOPS};
    qc.watch();
    bench_all(jobs, 2, WARMUP, ITERS, REPS, ms);
    qc.watch();

    // Both kernels compute the same product, so the flop count is the same for
    // both and GFLOP/s is a fair second axis: 2 * M * N * K counts one multiply
    // and one add per (i, j, k), which is the conventional GEMM count and the
    // one every published number uses.
    const double flops = 2.0 * M * N * K;
    printf("# timing: fastest of %d interleaved reps of (%d warmup + %d timed"
           " iterations); both compute %.2f MFLOP\n",
           REPS, WARMUP, ITERS, flops / 1e6);
    printf("# %-16s %9s %12s\n", "kernel", "best ms", "GFLOP/s");
    for (int k = 0; k < 2; k++) {
      printf("  %-16s %9.4f %12.1f\n", jobs[k].name, ms[k],
             flops / (ms[k] * 1e6));
    }

    const double wmma_gflops = flops / (ms[1] * 1e6);
    qc.report();

    const float ratio = ms[0] / ms[1];
    if (!qc.valid(wmma_gflops)) {
      qc.fail("tensor_cores_beat_fp32");
      printf("FAIL   the ratio above is printed anyway and is a real"
             " measurement -- of a saturated machine. Under external load both"
             " kernels inflate and the smaller one inflates proportionally"
             " more, so the ratio understates the tensor pipe.\n");
      ok = false;
    } else if (ratio >= MARGIN) {
      printf("PASS tensor_cores_beat_fp32 (naive / wmma = %.3fx >= %.2fx, same"
             " values, same product, different pipe)\n",
             ratio, MARGIN);
    } else {
      printf("FAIL tensor_cores_beat_fp32: ratio %.3fx below margin %.2fx --"
             " the puzzle's performance claim did not hold on a quiet"
             " machine\n",
             ratio, MARGIN);
      printf("FAIL   GemmNaiveFp32 issues %d FFMA per thread over %d threads;"
             " GemmWmma issues %d mma_sync per warp over %d warps, and each one"
             " is a whole 16x16x16 product. If the two are close, check that"
             " the wmma kernel really is looping over K and not computing one"
             " fragment, and that every warp got a distinct tile.\n",
             K, M * N, K / MMA, tiles_m * tiles_n);
      printf("FAIL   this run passed the validity check -- no other process had"
             " a CUDA context on this GPU, and GemmWmma cleared %.1f GFLOP/s --"
             " so another process loading this shared-memory SoC is NOT the"
             " explanation, and it is not what this FAIL is claiming.\n",
             wmma_gflops);
      printf("FAIL   (running under compute-sanitizer or ncu? set"
             " P33_SKIP_TIMING=1 -- instrumented wall-clock time is not a"
             " measurement of this machine)\n");
      ok = false;
    }
  }

  CHECK_CUDA(cudaFree(d_af));
  CHECK_CUDA(cudaFree(d_bf));
  CHECK_CUDA(cudaFree(d_ah));
  CHECK_CUDA(cudaFree(d_bh));
  CHECK_CUDA(cudaFree(d_c));
  delete[] af;
  delete[] bf;
  delete[] ah;
  delete[] bh;
  delete[] got;
  delete[] want;
  return ok ? 0 : 1;
}
