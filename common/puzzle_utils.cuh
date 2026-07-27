#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err_ = (call);                                               \
    if (err_ != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), \
              __FILE__, __LINE__, cudaGetErrorString(err_));                 \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

// After every kernel launch: catch launch-config errors and execution errors.
#define CHECK_LAUNCH()                       \
  do {                                       \
    CHECK_CUDA(cudaGetLastError());          \
    CHECK_CUDA(cudaDeviceSynchronize());     \
  } while (0)

constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

// Deterministic LCG fill, values in [-1, 1). Never rand() without a seed.
inline void fill_rand(float* a, int n, unsigned seed) {
  unsigned s = seed * 2654435761u + 1u;
  for (int i = 0; i < n; i++) {
    s = s * 1664525u + 1013904223u;
    a[i] = (float)(s >> 8) / (float)(1u << 24) * 2.0f - 1.0f;
  }
}

// Relative tolerance: float reductions reassociate, exact equality lies.
inline bool compare(const char* name, const float* got, const float* want,
                    int n, float tol) {
  for (int i = 0; i < n; i++) {
    float g = got[i], w = want[i];
    if (std::isnan(g) || fabsf(g - w) > tol * fmaxf(fabsf(w), 1.0f)) {
      printf("FAIL %s at index %d: got %g want %g\n", name, i, g, w);
      return false;
    }
  }
  printf("PASS %s\n", name);
  return true;
}

inline bool compare(const char* name, const int* got, const int* want, int n) {
  for (int i = 0; i < n; i++) {
    if (got[i] != want[i]) {
      printf("FAIL %s at index %d: got %d want %d\n", name, i, got[i], want[i]);
      return false;
    }
  }
  printf("PASS %s\n", name);
  return true;
}

struct GpuTimer {
  cudaEvent_t start_, stop_;
  GpuTimer() {
    CHECK_CUDA(cudaEventCreate(&start_));
    CHECK_CUDA(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    // Destructor cannot propagate; events are trivially destroyable here.
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  void start() { CHECK_CUDA(cudaEventRecord(start_)); }
  float stop_ms() {
    CHECK_CUDA(cudaEventRecord(stop_));
    CHECK_CUDA(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }
};

inline void print_device_info() {
  cudaDeviceProp p;
  CHECK_CUDA(cudaGetDeviceProperties(&p, 0));
  printf("# %s, sm_%d%d, %d SMs\n", p.name, p.major, p.minor,
         p.multiProcessorCount);
}
