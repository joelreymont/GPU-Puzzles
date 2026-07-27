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
