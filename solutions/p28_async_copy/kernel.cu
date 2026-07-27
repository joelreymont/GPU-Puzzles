#include <cuda_pipeline.h>
#include <cuda_runtime.h>

// Sliding-window sum, out[i] = a[i] + ... + a[i + WINDOW - 1], tiled.
// A tile produces TILE outputs, so it needs TILE + WINDOW - 1 inputs: the
// extra WINDOW - 1 are the halo the tile's last output reaches into. Both
// kernels compute exactly the same thing from exactly the same shared-memory
// layout; the only difference is how the tile gets into shared memory.
constexpr int WINDOW = 16;
constexpr int TPB = 256;   // must match the launch configuration
constexpr int TILE = TPB;  // one output per thread per tile
constexpr int TILE_IN = TILE + WINDOW - 1;

// Baseline. One buffer, one tile in flight: the block issues the tile's loads,
// waits at a barrier for them to land in shared memory, computes, and only
// then may touch the buffer again. Nothing overlaps -- the load latency of
// tile t+1 starts after the last read of tile t.
__global__ void WindowSumSync(const float* a, float* out, int n) {
  __shared__ float buf[TILE_IN];
  const int in_n = n + WINDOW - 1;  // elements a[] actually holds
  const int tiles = (n + TILE - 1) / TILE;

  for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
    /// BEGIN CODE HERE (approx 12 lines) ///
    const int base = t * TILE;
    // The final tile is ragged: clamp so the load stops at the end of a[].
    const int cnt = min(TILE_IN, in_n - base);
    for (int j = threadIdx.x; j < cnt; j += TPB) buf[j] = a[base + j];
    __syncthreads();
    const int i = base + threadIdx.x;
    if (i < n) {
      float s = 0.0f;
      for (int k = 0; k < WINDOW; k++) s += buf[threadIdx.x + k];
      out[i] = s;
    }
    __syncthreads();  // every read of buf[] is done before the next tile lands
    /// END CODE HERE ///
  }
}

// Two-stage double buffer. The copy for tile t+gridDim.x is issued into the
// spare buffer *before* the wait for tile t, so the hardware moves the next
// tile while this block sums the current one. __pipeline_memcpy_async lowers
// to LDGSTS: global -> shared with no register round-trip and no thread
// blocking on the data.
__global__ void WindowSumAsync(const float* a, float* out, int n) {
  __shared__ float buf[2][TILE_IN];  // one to fill, one to read
  const int in_n = n + WINDOW - 1;
  const int tiles = (n + TILE - 1) / TILE;

  /// BEGIN CODE HERE (approx 25 lines) ///
  // Issue tile `tt` into buffer `b` and close the pipeline stage. Every
  // thread commits, including threads with nothing to copy in a ragged tile,
  // so the whole block agrees on how many stages are outstanding.
  auto stage = [&](int tt, int b) {
    const int base = tt * TILE;
    const int cnt = min(TILE_IN, in_n - base);
    for (int j = threadIdx.x; j < cnt; j += TPB)
      __pipeline_memcpy_async(&buf[b][j], &a[base + j], sizeof(float));
    __pipeline_commit();
  };

  int t = blockIdx.x;
  if (t < tiles) stage(t, 0);  // prologue: prime the pipeline
  for (int p = 0; t < tiles; t += gridDim.x, p ^= 1) {
    const int nxt = t + gridDim.x;
    if (nxt < tiles) {
      stage(nxt, p ^ 1);
      __pipeline_wait_prior(1);  // tile t has landed; tile nxt may still fly
    } else {
      __pipeline_wait_prior(0);  // nothing left to prefetch: drain
    }
    __syncthreads();  // wait_prior is per-thread; the buffer is block-wide
    const int i = t * TILE + threadIdx.x;
    if (i < n) {
      float s = 0.0f;
      for (int k = 0; k < WINDOW; k++) s += buf[p][threadIdx.x + k];
      out[i] = s;
    }
    __syncthreads();  // buf[p] is free before it is refilled two tiles later
  }
  /// END CODE HERE ///
}
