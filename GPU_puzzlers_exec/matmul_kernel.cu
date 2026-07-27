#include <cuda_runtime.h>

const int TPB = 3;

__global__ void Matmul(float* A, float* B, float* C, int size) {
  extern __shared__ float sharedMem[];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int local_i = threadIdx.x;
  int local_j = threadIdx.y;

  float* a_shared;  // carve from sharedMem
  float* b_shared;  // carve from sharedMem

  /// CODE HERE (approx 20 lines) ///

}
