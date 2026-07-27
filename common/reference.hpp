#pragma once

// CPU references. Double accumulators: the reference must be more accurate
// than the kernel under test, not equally sloppy.

inline float ref_dot(const float* a, const float* b, int n) {
  double acc = 0.0;
  for (int i = 0; i < n; i++) acc += (double)a[i] * (double)b[i];
  return (float)acc;
}

inline int ref_count_gt(const float* a, int n, float thresh) {
  int c = 0;
  for (int i = 0; i < n; i++) c += (a[i] > thresh) ? 1 : 0;
  return c;
}

// Elementwise references: no accumulation, so a double accumulator would buy
// nothing — each output is one IEEE float op the GPU performs identically.

// Forward difference; the last element has no successor and is defined as 0.
inline void ref_neighbor_diff(const float* a, float* out, int n) {
  for (int i = 0; i < n - 1; i++) out[i] = a[i + 1] - a[i];
  if (n > 0) out[n - 1] = 0.0f;
}

// Subtract the value at the start of each aligned 32-element group.
inline void ref_sub_warp_base(const float* a, float* out, int n) {
  for (int i = 0; i < n; i++) out[i] = a[i] - a[(i / 32) * 32];
}

// Shift right by one within each aligned 32-element group; group head gets 0.
inline void ref_shift_up(const float* a, float* out, int n) {
  for (int i = 0; i < n; i++) out[i] = (i % 32 == 0) ? 0.0f : a[i - 1];
}
