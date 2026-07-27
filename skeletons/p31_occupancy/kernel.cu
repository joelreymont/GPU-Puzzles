#include <cuda_runtime.h>

// Three kernels, one answer: out[i] is the same 64-term polynomial evaluated
// at a[i] in Horner form. The arithmetic is identical in all three -- the same
// coefficients applied in the same order -- so all three produce the same
// floats. What differs is what each one asks the SM for, and how much work
// each thread does before it exits.
//
// TERMS - 1 = 63 chained fmaf() calls. Every one of them needs the result of
// the previous one, so a thread evaluating a single polynomial has exactly one
// arithmetic instruction it is allowed to issue at a time.
constexpr int TERMS = 64;

// Elements per thread in PolyFour. The runner sizes PolyFour's grid at
// cdiv(n, ELEMS) threads, so PolyOne and PolyFour do the same total work with
// PolyFour using a quarter of the threads.
constexpr int ELEMS = 4;

// c_k = 1/(k+1). After #pragma unroll every k is a compile-time constant, so
// this folds away completely: each fmaf() carries its coefficient as an
// immediate operand, there is no table and no load. common/reference.hpp
// computes the identical expression, so both round to the same float.
__device__ __forceinline__ float coef(int k) { return 1.0f / (float)(k + 1); }

// 24 KiB of static shared memory that SmemHog uses for exactly one float per
// thread. It is here to be a *budget*, not a buffer: large enough that the
// shared-memory budget, rather than the thread budget, is what decides how
// many blocks fit on an SM. Whether it succeeds is a question for the
// occupancy API, and the runner asks it on every run.
constexpr int HOG_FLOATS = 24 * 1024 / (int)sizeof(float);

// One element per thread, one dependent chain per thread. The baseline: the
// smallest register footprint of the three and no shared memory at all.
__global__ void PolyOne(const float* a, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN CODE HERE (approx 5 lines) ///
  /// END CODE HERE ///
}

// ELEMS elements per thread, strided by the whole grid so that each of a
// warp's loads stays exactly as coalesced as PolyOne's. Write it as ELEMS
// chains that are independent of one another -- nothing in element t's
// arithmetic depends on element t-1's -- holding all ELEMS inputs and all
// ELEMS accumulators live across the loop. That costs registers. Whether it
// costs occupancy, and what it buys, are both questions the runner answers
// with numbers.
__global__ void PolyFour(const float* a, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  /// BEGIN CODE HERE (approx 16 lines) ///
  /// END CODE HERE ///
}

// PolyOne's arithmetic, one element per thread, one dependent chain -- but
// staged through a shared array 24 times larger than the 1 KiB of it this
// block actually touches. Nothing about the math changed. Fill your slot,
// barrier, read it back, evaluate.
//
// The launch bounds are a promise and a request: never more than 256 threads
// per block, and please compile so that 6 blocks fit on an SM at once. 6 x 256
// is exactly this SM's 1536-thread budget, which is what makes 6 the largest
// number ptxas will take here -- ask for 7 or 8 and it answers
// `Value of threads per SM ... is out of range. .minnctapersm will be ignored`
// and throws the annotation away, which changes nothing about the binary that
// you would notice without reading the build log. And what ptxas controls in
// return is the register count; it gets no vote at all on shared memory. So
// the request is only ever half the story: run the occupancy API and see which
// budget actually answers.
__global__ void __launch_bounds__(256, 6)
    SmemHog(const float* a, float* out, int n) {
  __shared__ float stage[HOG_FLOATS];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN CODE HERE (approx 7 lines) ///
  /// END CODE HERE ///
}
