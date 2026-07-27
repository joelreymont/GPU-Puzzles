#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void WindowSumSync(const float* a, float* out, int n);
extern __global__ void WindowSumAsync(const float* a, float* out, int n);

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

  // Big enough that the kernel is memory-bound rather than launch-bound, and
  // deliberately not a multiple of 32, of TPB, or of the window: the final
  // tile is ragged in both its input count and its output count.
  const int n = 1000003;
  const int WINDOW = 16;  // must match the kernels' compile-time WINDOW
  const int TPB = 256;    // must match the kernels' compile-time TPB
  const int TILE = TPB;   // one output per thread per tile
  // A fixed grid: 4 blocks per SM on this box's 48 SMs. The kernels
  // grid-stride over the tiles, so this is independent of n.
  const int BLOCKS = 192;
  const int WARMUP = 5;
  const int ITERS = 50;
  // Every output is a sum of 16 floats reassociated differently on the GPU
  // than in the reference's double accumulator.
  const float tol = 1e-5f;

  const int in_n = n + WINDOW - 1;  // inputs, including the trailing halo
  const int tiles = cdiv(n, TILE);
  printf("# sliding-window sum: WINDOW=%d n=%d in_n=%d, %d tiles x %d outputs,"
         " %d blocks x %d threads\n",
         WINDOW, n, in_n, tiles, TILE, BLOCKS, TPB);

  float* a = new float[in_n];
  float* got = new float[n];
  float* want = new float[n];
  fill_rand(a, in_n, 28);
  ref_window_sum(a, want, n, WINDOW);

  float *d_a, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, in_n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(d_a, a, in_n * sizeof(float), cudaMemcpyHostToDevice));

  // Correctness first, and only then timing: a fast wrong kernel is not a
  // result. Every kernel owns every output, so poison the buffer with 0xff (a
  // NaN bit pattern) -- an output that is never written is a FAIL, not a
  // lucky zero.
  bool ok = true;

  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  WindowSumSync<<<BLOCKS, TPB>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("window_sum_sync", got, want, n, tol) && ok;

  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  WindowSumAsync<<<BLOCKS, TPB>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("window_sum_async", got, want, n, tol) && ok;

  if (ok) {
    // Global traffic per iteration: every input element is read once and every
    // output written once. The halo overlap between tiles is re-read, but it
    // hits in L2, so the DRAM-facing figure below is the honest one.
    const double bytes = (double)(in_n + n) * sizeof(float);
    const float ms_sync =
        bench((const void*)WindowSumSync, BLOCKS, TPB, d_a, d_out, n, WARMUP,
              ITERS);
    const float ms_async =
        bench((const void*)WindowSumAsync, BLOCKS, TPB, d_a, d_out, n, WARMUP,
              ITERS);
    printf("# timing: %d warmup + %d timed iterations, %.2f MB of global "
           "traffic per iteration\n",
           WARMUP, ITERS, bytes / 1e6);
    printf("# %-18s %9s %10s\n", "kernel", "avg ms", "eff GB/s");
    printf("  %-18s %9.4f %10.1f\n", "window_sum_sync", ms_sync,
           bytes / (ms_sync * 1e6));
    printf("  %-18s %9.4f %10.1f\n", "window_sum_async", ms_async,
           bytes / (ms_async * 1e6));
    printf("# sync/async time ratio: %.3fx\n", ms_sync / ms_async);
  } else {
    printf("# timing skipped: a kernel is incorrect\n");
  }

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_out));
  delete[] a;
  delete[] got;
  delete[] want;
  return ok ? 0 : 1;
}
