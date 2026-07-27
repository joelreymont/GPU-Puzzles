#include "puzzle_utils.cuh"
#include "reference.hpp"

extern __global__ void StencilSyncthreads(const float* a, float* out, int n);
extern __global__ void StencilBarrierArriveWait(const float* a, float* out,
                                                int n);

// Launched through cudaLaunchKernel rather than <<<>>> so the timing loop is
// written once: CUDA does not allow <<<>>> through a function pointer, but the
// host-side stub address is a perfectly good kernel handle.
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

  // Not a multiple of 32, of SEG, or of anything else convenient: the last
  // block owns 163 outputs, not 256, and both array ends land inside a block
  // whose halo runs off the array.
  const int n = 100003;
  const int T = 4;      // must match the kernels' compile-time T
  const int SEG = 256;  // outputs per block; must match the kernels' SEG
  const int TPB = 256;  // must match the kernels' compile-time TPB
  const int BLOCKS = cdiv(n, SEG);
  const int WARMUP = 3;
  const int ITERS = 20;
  // Four float ping-pong steps against a double-buffered CPU reference.
  const float tol = 1e-5f;

  printf("# iterative 3-point stencil: n=%d, T=%d steps, %d blocks x %d "
         "threads, %d outputs per block\n",
         n, T, BLOCKS, TPB, SEG);

  float* a = new float[n];
  float* got = new float[n];
  float* want = new float[n];
  fill_rand(a, n, 29);
  ref_stencil_steps(a, want, n, T);

  float *d_a, *d_out;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

  // Every kernel owns every output, so poison the buffer with 0xff (a NaN bit
  // pattern) before each launch -- an output that is never written is a FAIL,
  // not a lucky zero.
  bool ok = true;

  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  StencilSyncthreads<<<BLOCKS, TPB>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("stencil_syncthreads", got, want, n, tol) && ok;

  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  StencilBarrierArriveWait<<<BLOCKS, TPB>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("stencil_arrive_wait", got, want, n, tol) && ok;

  // This is a correctness puzzle, not a timing one. The two kernels run the
  // same arithmetic through two spellings of the same barrier, so the numbers
  // below exist only to keep any claim about their relative cost honest --
  // they are not a target to optimise against.
  if (ok) {
    const float ms_sync = bench((const void*)StencilSyncthreads, BLOCKS, TPB,
                                d_a, d_out, n, WARMUP, ITERS);
    const float ms_bar = bench((const void*)StencilBarrierArriveWait, BLOCKS,
                               TPB, d_a, d_out, n, WARMUP, ITERS);
    printf("# timing: %d warmup + %d timed iterations\n", WARMUP, ITERS);
    printf("# %-22s %9s\n", "kernel", "avg ms");
    printf("  %-22s %9.4f\n", "stencil_syncthreads", ms_sync);
    printf("  %-22s %9.4f\n", "stencil_arrive_wait", ms_bar);
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
