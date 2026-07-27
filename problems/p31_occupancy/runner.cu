#include "puzzle_utils.cuh"
#include "reference.hpp"

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
// or not anything is instrumenting it. No margin, no tolerance, no clock.
//
// occupancy_not_predictive is a wall-clock measurement, and this box is a
// shared-memory SoC: the GPU and the CPUs sit behind the same LPDDR5X. On an
// idle machine PolyOne/PolyFour measures 1.28x, and in the cleanest windows
// 1.49x. Let anything else run -- a build, another agent, even a 10 Hz
// nvidia-smi poll -- and both kernels inflate, both go memory-bound, and the
// ratio compresses toward 1.0; the lowest whole-run value in ~80 measurements
// on this box was 1.052. So MARGIN sits below that floor rather than at half
// the effect. The assert's job is to never lie; the size of the effect is
// printed on the line above it either way, and the whole measured distribution
// is in solutions/p31_occupancy/SOLUTION.md.
constexpr float MARGIN = 1.05f;

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

// All three kernels are timed together: one rep of each in turn, repeated, and
// each kernel keeps its own fastest rep.
//
// Interleaving is not cosmetic. Timing every rep of one kernel and then every
// rep of the next lets a drift in machine state land entirely on one of them;
// measured on this box that ordering bias is worth about 0.05 on the ratio,
// always against whichever kernel runs last. One rep of each in turn gives
// every kernel the same distribution of machine states.
//
// The fastest rep, not the mean, is the estimate. These kernels run in 15-25
// us, and contention only ever makes a measurement slower -- never faster --
// so the minimum is the observation least contaminated by whatever else the
// machine was doing. An average folds that contamination straight into the
// number the assert reads.
//
// Launched through cudaLaunchKernel rather than <<<>>> so the loop is written
// once: CUDA does not allow <<<>>> through a function pointer, but the
// host-side stub address is a perfectly good kernel handle.
static void bench_all(const void* const* kerns, const int* blocks, int nk,
                      int tpb, const float* d_a, float* d_out, int n,
                      int warmup, int iters, int reps, float* ms_out) {
  void* args[] = {(void*)&d_a, (void*)&d_out, (void*)&n};
  for (int k = 0; k < nk; k++) {
    ms_out[k] = -1.0f;
    for (int i = 0; i < warmup; i++) {
      CHECK_CUDA(
          cudaLaunchKernel(kerns[k], dim3(blocks[k]), dim3(tpb), args, 0, 0));
    }
  }
  CHECK_LAUNCH();

  for (int r = 0; r < reps; r++) {
    for (int k = 0; k < nk; k++) {
      GpuTimer timer;
      timer.start();
      for (int i = 0; i < iters; i++) {
        CHECK_CUDA(
            cudaLaunchKernel(kerns[k], dim3(blocks[k]), dim3(tpb), args, 0, 0));
      }
      const float ms = timer.stop_ms() / iters;
      CHECK_CUDA(cudaGetLastError());
      if (ms_out[k] < 0.0f || ms < ms_out[k]) ms_out[k] = ms;
    }
  }
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
  const int REPS = 15;  // see bench_all(): the assert needs the best, not the mean
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
  // cannot be perturbed by an instrumented run, and it holds or fails
  // identically under compute-sanitizer and under ncu. So it is checked
  // whenever the kernels are correct.
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
    const void* kerns[3] = {(const void*)PolyOne, (const void*)PolyFour,
                            (const void*)SmemHog};
    const int blocks[3] = {BLOCKS_ONE, BLOCKS_FOUR, BLOCKS_ONE};
    float ms[3];
    bench_all(kerns, blocks, 3, TPB, d_a, d_out, n, WARMUP, ITERS, REPS, ms);
    printf("# timing: fastest of %d interleaved reps of (%d warmup + %d timed"
           " iterations); all three evaluate all %d polynomials\n",
           REPS, WARMUP, ITERS, n);
    printf("# %-10s %9s %12s %11s\n", "kernel", "best ms", "ns/element",
           "occupancy");
    for (int k = 0; k < 3; k++) {
      printf("  %-10s %9.4f %12.5f %10.1f%%\n", names[k], ms[k],
             ms[k] * 1e6 / n, 100.0 * occs[k]->frac);
    }

    const float ratio = ms[0] / ms[1];
    if (ratio >= MARGIN) {
      printf("PASS occupancy_not_predictive (poly_one / poly_four = %.3fx >="
             " %.2fx, at %.1f%% vs %.1f%% occupancy)\n",
             ratio, MARGIN, 100.0 * one.frac, 100.0 * four.frac);
    } else {
      printf("FAIL occupancy_not_predictive: poly_one / poly_four = %.3fx,"
             " expected >= %.2fx\n",
             ratio, MARGIN);
      printf("FAIL   PolyFour does the same total arithmetic in a quarter of"
             " the threads, %d elements per thread instead of one; it should be"
             " decisively faster per element, and it was not\n",
             ELEMS);
      printf("FAIL   a ratio below the margin usually means another process is"
             " loading this shared-memory SoC -- the GPU and the CPUs share one"
             " LPDDR5X, and under contention both kernels go memory-bound and"
             " converge. Re-run on an idle box.\n");
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
