#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void GroupSumRedundant(const float* a, float* out, int n);
extern __global__ void GroupSumStream(const float* a, float* out, int n);

// The relationship this puzzle exists to demonstrate: the kernel that hits in
// L1 almost every time is the SLOWER one. MARGIN is a floor on that effect,
// not a description of it -- it is set to roughly half the ratio measured on
// this box so that run-to-run noise cannot flip the verdict. The real number
// is in solutions/p30_profiling/SOLUTION.md, measured.
constexpr float MARGIN = 1.25f;

// The kernels are launched through cudaLaunchKernel rather than <<<>>> so the
// timing loop is written once: CUDA does not allow <<<>>> through a function
// pointer, but the host-side stub address is a perfectly good kernel handle.
static float bench(const void* kern, int blocks, int tpb, const float* d_a,
                   float* d_out, int n, int warmup, int iters) {
  void* args[] = {(void*)&d_a, (void*)&d_out, (void*)&n};
  for (int i = 0; i < warmup; i++) {
    CHECK_CUDA(cudaLaunchKernel(kern, dim3(blocks), dim3(tpb), args, 0, 0));
  }
  CHECK_LAUNCH();

  GpuTimer timer;
  timer.start();
  for (int i = 0; i < iters; i++) {
    CHECK_CUDA(cudaLaunchKernel(kern, dim3(blocks), dim3(tpb), args, 0, 0));
  }
  const float ms = timer.stop_ms();
  CHECK_CUDA(cudaGetLastError());
  return ms / iters;
}

int main() {
  print_device_info();

  // 1000003 is deliberate on two axes.
  //
  // Guard logic: it is a multiple of neither 32 nor TPB, so the last group has
  // 3 elements, not 32, and the last block has 3 live threads, not 256.
  //
  // Working set: a + out is 8 MB against this box's 25 MB L2 (printed below),
  // so neither kernel is DRAM-bandwidth-bound and the timing table is free to
  // show what the two load paths actually cost. Puzzle 28 measured what
  // happens past the L2 cliff; SOLUTION.md measures it again for these two.
  const int n = 1000003;
  const int TPB = 256;    // must match the kernels' compile-time TPB
  const int GROUP = 32;   // must match the kernels' compile-time GROUP
  const int BLOCKS = cdiv(n, TPB);
  const int WARMUP = 3;
  const int ITERS = 20;
  // Every output is a sum of up to 32 floats, reassociated differently on the
  // GPU (butterfly / left-to-right) than in the reference's double
  // accumulator.
  const float tol = 1e-5f;

  const int groups = cdiv(n, GROUP);
  printf("# group sum: GROUP=%d n=%d, %d groups (last has %d), %d blocks x %d"
         " threads\n",
         GROUP, n, groups, n - (groups - 1) * GROUP, BLOCKS, TPB);

  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  printf("# working set: a + out = %.2f MB, L2 = %.2f MB\n",
         2.0 * n * sizeof(float) / 1e6, prop.l2CacheSize / 1e6);

  float* a = new float[n];
  float* got = new float[n];
  float* want = new float[n];
  fill_rand(a, n, 30);
  ref_group_sum_all(a, want, n);

  float *d_a, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

  // Correctness first, and only then timing: a fast wrong kernel is not a
  // result, and a performance *relationship* between two kernels means nothing
  // unless both of them compute the same right answer. Every kernel owns every
  // output, so poison the buffer with 0xff (a NaN bit pattern) -- an output
  // that is never written is a FAIL, not a lucky zero.
  bool ok = true;

  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  GroupSumRedundant<<<BLOCKS, TPB>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("group_sum_redundant", got, want, n, tol) && ok;

  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  GroupSumStream<<<BLOCKS, TPB>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("group_sum_stream", got, want, n, tol) && ok;

  // The timing section asserts a property of the *hardware*, so it is only
  // meaningful on an uninstrumented run. Setting P30_SKIP_TIMING (to anything)
  // runs correctness alone; problems/p30_profiling/prof.mk sets it for the
  // three compute-sanitizer runs, where a wall clock measures the sanitizer.
  const bool skip_timing = getenv("P30_SKIP_TIMING") != nullptr;

  if (!ok) {
    printf("# timing skipped: a kernel is incorrect\n");
  } else if (skip_timing) {
    printf("# timing skipped: P30_SKIP_TIMING is set. Both kernels ran and"
           " were verified above; only the wall-clock comparison is off.\n");
  } else {
    // Useful traffic only: every input read once, every output written once.
    // What each kernel actually moves through L1 to achieve that is the
    // subject of `make prof P=30`, and it is not this number.
    const double bytes = 2.0 * n * sizeof(float);
    const float ms_red =
        bench((const void*)GroupSumRedundant, BLOCKS, TPB, d_a, d_out, n,
              WARMUP, ITERS);
    const float ms_str = bench((const void*)GroupSumStream, BLOCKS, TPB, d_a,
                               d_out, n, WARMUP, ITERS);
    printf("# timing: %d warmup + %d timed iterations, %.2f MB of useful"
           " traffic per iteration\n",
           WARMUP, ITERS, bytes / 1e6);
    printf("# %-20s %9s %10s\n", "kernel", "avg ms", "eff GB/s");
    printf("  %-20s %9.4f %10.1f\n", "group_sum_redundant", ms_red,
           bytes / (ms_red * 1e6));
    printf("  %-20s %9.4f %10.1f\n", "group_sum_stream", ms_str,
           bytes / (ms_str * 1e6));

    const float ratio = ms_red / ms_str;
    if (ratio >= MARGIN) {
      printf("PASS cache_hit_paradox (ratio %.3fx >= %.2fx)\n", ratio, MARGIN);
    } else {
      printf("FAIL cache_hit_paradox: group_sum_redundant / group_sum_stream ="
             " %.3fx, expected >= %.2fx\n",
             ratio, MARGIN);
      printf("FAIL   the kernel that re-reads its group %d times over should be"
             " decisively slower than the one that reads each element once;"
             " it was not\n",
             GROUP);
      printf("FAIL   (running under compute-sanitizer or ncu? set"
             " P30_SKIP_TIMING=1 -- instrumented wall-clock time is not a"
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
