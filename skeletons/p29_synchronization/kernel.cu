#include <cuda/barrier>
#include <cuda/std/utility>
#include <cuda_runtime.h>

// A __shared__ object is never dynamically initialised, so cuda::barrier's
// constructor does not run at its declaration -- init() constructs it in place
// instead, which is the documented way to build one. Silence the warning that
// points this out; the pattern below is exactly what it is warning about.
#pragma nv_diag_suppress static_var_with_dynamic_init

// Iterative 1D heat diffusion. One step of the 3-point stencil is
//   next[i] = 0.25*cur[i-1] + 0.5*cur[i] + 0.25*cur[i+1]
// with clamped indexing: a neighbour off the end of the array is the element
// itself. Both kernels apply T steps to the whole array in a single launch,
// entirely in shared memory. The only difference is how the block synchronises
// between steps.
//
// Block b owns outputs [b*SEG, b*SEG + SEG). One step of the stencil consumes
// one element off each end of whatever it can see, so T steps need T elements
// of halo on each side: the block loads SEG + 2*HALO elements and lets its
// visible-and-correct window shrink by one per step, arriving at exactly the
// SEG central outputs after T steps. That trades redundant arithmetic in the
// halo for zero global synchronisation.
constexpr int T = 4;                 // stencil steps applied per launch
constexpr int SEG = 256;             // outputs owned by one block
constexpr int HALO = T;              // one step eats one element per side
constexpr int TPB = 256;             // must match the launch configuration
constexpr int SH = SEG + 2 * HALO;   // shared elements per buffer

// Classic spelling: __syncthreads() between ping-pong steps. It is both an
// execution barrier (no thread runs ahead into the next step) and a
// block-scope memory fence (every shared write before it is visible to every
// thread after it). The ping-pong needs both, and needs them from one call.
__global__ void StencilSyncthreads(const float* a, float* out, int n) {
  __shared__ float sh[2][SH];

  // Shared slot j holds global element base - HALO + j. lo and hi are the
  // shared indices of global elements 0 and n-1; clamping a neighbour into
  // [lo, hi] *is* the boundary condition, re-applied every step. For an
  // interior block they are 0 and SH-1, where the clamp is a no-op on data
  // that has already fallen outside the shrinking window.
  const int base = blockIdx.x * SEG;
  const int lo = max(0, HALO - base);
  const int hi = min(SH - 1, n - 1 - base + HALO);

  /// BEGIN CODE HERE (approx 14 lines) ///
  /// END CODE HERE ///
}

// Same computation, same barrier semantics, split in two. cuda::barrier gives
// arrival and waiting as separate operations: arrive() publishes this thread's
// writes and returns a token, wait(token) blocks until the whole block has
// arrived. Anything between the two runs while the slow threads catch up --
// provided it needs nothing the barrier is there to order.
__global__ void StencilBarrierArriveWait(const float* a, float* out, int n) {
  __shared__ float sh[2][SH];
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;

  // Bootstrap: one thread constructs the barrier with the block's arrival
  // count, and a plain __syncthreads() makes it visible to everyone else.
  // There is no chicken-and-egg escape from that first barrier.
  if (threadIdx.x == 0) init(&bar, blockDim.x);
  __syncthreads();

  const int base = blockIdx.x * SEG;
  const int lo = max(0, HALO - base);
  const int hi = min(SH - 1, n - 1 - base + HALO);

  /// BEGIN CODE HERE (approx 17 lines) ///
  /// END CODE HERE ///
}
