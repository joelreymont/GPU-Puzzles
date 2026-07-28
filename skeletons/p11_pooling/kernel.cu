#include <cuda_runtime.h>

// Window length: each output is the sum of its element and the two before it,
// truncated at the front of the array. HALO is how far the window reaches to
// the left of a block's own tile, so the tile is HALO elements wider than the
// block and its element for global index i sits at tile[HALO + threadIdx.x].
const int W = 3;
const int HALO = W - 1;

// out[i] = a[i-2] + a[i-1] + a[i], with the missing terms at i = 0 and i = 1
// contributing nothing. Every element of `a` is read by three different
// outputs, so it is staged once into shared memory and read three times from
// there -- including the two elements that belong to the previous block.
//
// `tile` is dynamic shared memory: the runner requests blockDim.x + HALO
// floats, and tile[m] holds a[blockIdx.x * blockDim.x + m - HALO].
__global__ void Pooling(const float* a, float* out, int n) {
  extern __shared__ float tile[];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int local_i = threadIdx.x;

  /// BEGIN CODE HERE (approx 12 lines) ///
  /// END CODE HERE ///
}
