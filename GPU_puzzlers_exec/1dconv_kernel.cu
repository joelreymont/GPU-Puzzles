#include <cuda_runtime.h>

const int TPB = 8;
const int MAX_CONV = 4;
const int TPB_MAX_CONV = TPB + MAX_CONV;

__global__ void Conv1D(float* A, float* B, float* C, int a_size, int b_size) {
  extern __shared__ float sharedMem[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int local_i = threadIdx.x;

  float* shared_a;  // carve from sharedMem
  float* shared_b;  // carve from sharedMem

  /// CODE HERE (approx 25 lines) ///

}
