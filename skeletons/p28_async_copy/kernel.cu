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
  /// END CODE HERE ///
}
