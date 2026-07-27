#pragma once

#include <algorithm>
#include <vector>

// CPU references. Double accumulators: the reference must be more accurate
// than the kernel under test, not equally sloppy.

inline float ref_dot(const float* a, const float* b, int n) {
  double acc = 0.0;
  for (int i = 0; i < n; i++) acc += (double)a[i] * (double)b[i];
  return (float)acc;
}

inline float ref_sum(const float* a, int n) {
  double acc = 0.0;
  for (int i = 0; i < n; i++) acc += (double)a[i];
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

// Group references accumulate, so they use a double accumulator per group.
// A trailing group shorter than 32 sums only the elements that exist.

// Every element of an aligned 32-element group receives that group's total.
inline void ref_group_sum_all(const float* a, float* out, int n) {
  for (int base = 0; base < n; base += 32) {
    int end = (base + 32 < n) ? base + 32 : n;
    double acc = 0.0;
    for (int i = base; i < end; i++) acc += (double)a[i];
    for (int i = base; i < end; i++) out[i] = (float)acc;
  }
}

// Inclusive prefix sum restarted at every aligned `seg`-element boundary.
// seg = 32 is the warp scan (puzzle 26); seg = TPB is the block scan (27).
inline void ref_seg_incl_scan(const float* a, float* out, int n, int seg) {
  for (int base = 0; base < n; base += seg) {
    int end = (base + seg < n) ? base + seg : n;
    double acc = 0.0;
    for (int i = base; i < end; i++) {
      acc += (double)a[i];
      out[i] = (float)acc;
    }
  }
}

// Inclusive prefix sum restarted at every aligned 32-element group boundary.
inline void ref_group_incl_scan(const float* a, float* out, int n) {
  ref_seg_incl_scan(a, out, n, 32);
}

// Sliding-window sum: out[i] = a[i] + a[i+1] + ... + a[i+w-1], for i in
// [0, n). `a` must therefore hold n + w - 1 elements -- the trailing w - 1 are
// the halo the last window reaches into, not outputs of their own. One double
// accumulator per window, so the reference is more accurate than any float
// association the kernel picks.
inline void ref_window_sum(const float* a, float* out, int n, int w) {
  for (int i = 0; i < n; i++) {
    double acc = 0.0;
    for (int k = 0; k < w; k++) acc += (double)a[i + k];
    out[i] = (float)acc;
  }
}

// Iterative 3-point stencil (1D heat diffusion). One step is
//   next[i] = 0.25*cur[i-1] + 0.5*cur[i] + 0.25*cur[i+1]
// with clamped indexing at both ends: an out-of-array neighbour is the element
// itself, so next[0] = 0.75*cur[0] + 0.25*cur[1]. `steps` full-array steps,
// double buffers throughout -- the reference must be more accurate than the
// kernel's float ping-pong, not equally sloppy.
inline void ref_stencil_steps(const float* a, float* out, int n, int steps) {
  std::vector<double> cur(n), nxt(n);
  for (int i = 0; i < n; i++) cur[i] = (double)a[i];
  for (int s = 0; s < steps; s++) {
    for (int i = 0; i < n; i++) {
      double l = cur[i > 0 ? i - 1 : 0];
      double r = cur[i + 1 < n ? i + 1 : n - 1];
      nxt[i] = 0.25 * l + 0.5 * cur[i] + 0.25 * r;
    }
    cur.swap(nxt);
  }
  for (int i = 0; i < n; i++) out[i] = (float)cur[i];
}

// Horner evaluation of the fixed 64-term polynomial p(x) = sum_k x^k / (k+1),
// one output per input. The coefficient expression is character-identical to
// the kernel's, so both round to the same float and the tolerance measures the
// accumulation error alone; only the 63-step accumulation is promoted to
// double, which is what makes this reference more accurate than the kernel
// under test rather than equally sloppy.
inline void ref_poly64(const float* a, float* out, int n) {
  constexpr int TERMS = 64;
  for (int i = 0; i < n; i++) {
    const double x = (double)a[i];
    double acc = (double)(1.0f / (float)TERMS);  // c_{TERMS-1}
    for (int k = TERMS - 2; k >= 0; k--)
      acc = acc * x + (double)(1.0f / (float)(k + 1));
    out[i] = (float)acc;
  }
}

// Out-of-place transpose of a rows x cols row-major matrix into a cols x rows
// row-major matrix. No arithmetic happens: every output is a bit-exact copy of
// exactly one input, so this reference is not an approximation of the kernel's
// result, it IS the kernel's result, and the comparison tolerance is zero.
inline void ref_transpose(const float* a, float* out, int rows, int cols) {
  for (int r = 0; r < rows; r++)
    for (int c = 0; c < cols; c++) out[c * rows + r] = a[r * cols + c];
}

// C = A * B, with A row-major M x K, B row-major K x N, C row-major M x N.
// One double accumulator per output element, so this reference is more
// accurate than any kernel under test rather than equally sloppy.
//
// The products are exact, not merely double-rounded: `a` and `b` hold float
// values that came back through __half2float, so each has at most 11
// significant bits and (double)a * (double)b needs at most 22 -- well inside
// the 53 a double carries. Every kernel under test consumes those same
// numbers. The only thing this reference does not reproduce is the order and
// the width of the K-way accumulation, which is exactly what the runner's
// tolerance is measuring and nothing else.
inline void ref_gemm(const float* a, const float* b, float* c, int M, int N,
                     int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      double acc = 0.0;
      for (int k = 0; k < K; k++)
        acc += (double)a[i * K + k] * (double)b[k * N + j];
      c[i * N + j] = (float)acc;
    }
  }
}

// Histogram of values drawn from [-1, 1) into nbins equal-width bins. The bin
// expression is character-identical to the kernel's so both round the same
// way; the clamp keeps a value on or past the top edge inside the array.
inline void ref_hist(const float* a, int* hist, int nbins, int n) {
  for (int k = 0; k < nbins; k++) hist[k] = 0;
  for (int i = 0; i < n; i++) {
    float v = a[i];
    int b = (int)((v + 1.0f) * 0.5f * nbins);
    b = std::min(std::max(b, 0), nbins - 1);
    hist[b]++;
  }
}
