#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// A cluster is one level of the execution hierarchy that did not exist before
// SM90: grid -> cluster -> block -> warp -> thread. The blocks of a cluster are
// co-scheduled on the same GPC, they can barrier with each other
// (cluster.sync()), and -- the part that changes what you can write -- every
// block can address every other block's shared memory. That address space is
// distributed shared memory (DSMEM), and cluster.map_shared_rank() is the only
// way to name it.
//
// CLUSTER is a compile-time constant here because both kernels below carry
// __cluster_dims__(CLUSTER, 1, 1). The geometry is baked into the kernel, so
// the runner launches with an ordinary <<<grid, TPB>>> and the only obligation
// it inherits is that gridDim.x must be a multiple of CLUSTER. The runtime
// alternative -- cudaLaunchKernelEx with a cudaLaunchAttributeClusterDimension
// attribute, which scripts/probe_caps.cu uses -- reaches the same hardware and
// decides the shape at launch instead. Compile time is used here because the
// attribute is half the lesson: it is what makes the following kernels cluster
// kernels rather than ordinary ones.
//
// 4 is measured rather than assumed. cudaOccupancyMaxPotentialClusterSize
// reports 8 for both of these kernels on this box (the runner prints it), and
// the runner then asserts that the hardware really did group 4 blocks -- see
// `meta` in ClusterHistogram. A launch that is silently downgraded to
// one-block clusters is the failure mode this puzzle exists to rule out.
constexpr int CLUSTER = 4;
constexpr int TPB = 256;

// The histogram's bin count. It is a compile-time constant because each block
// holds only its OWN slice of the bins in static shared memory; the runner
// passes the same number as `nbins` and the two must agree.
constexpr int NBINS = 256;
constexpr int SLICE = NBINS / CLUSTER;  // bins owned by one block

// Distributed-shared-memory histogram. Puzzle 27 gave every block a private
// copy of all the bins and merged gridDim.x copies into global memory at the
// end. Here the cluster holds ONE copy of the bins, split across the shared
// memory of its 4 blocks: rank 0 owns bins [0, 64), rank 1 owns [64, 128), and
// so on. A thread that lands in a bin it does not own does not get a private
// copy and does not go to global memory -- it atomically increments the owning
// block's shared memory directly.
//
// So the merge at the end is 4x smaller than puzzle 27's: one global atomic per
// bin per CLUSTER, not per block.
//
// Everything that follows is a consequence of one fact: shared memory belongs
// to a block and dies with it. Three things must therefore be true, and each
// one is a barrier:
//
//   1. Nobody may increment a slice before its owner has zeroed it.
//   2. Nobody may read a slice into global memory before every peer has
//      finished incrementing it.
//   3. No block may exit while a peer might still touch its shared memory --
//      exiting deallocates it.
//
// cluster.sync() is the only barrier that can enforce any of these:
// __syncthreads() synchronises one block and knows nothing about the other 3.
//
// `meta` is not part of the algorithm. Thread 0 of rank 0 writes
// cluster.num_blocks() there and the runner checks it equals CLUSTER, which is
// the one thing no compile-time attribute can promise by itself.
__global__ void __cluster_dims__(CLUSTER, 1, 1)
ClusterHistogram(const float* a, int* hist, int* meta, int nbins, int n) {
  // This block's slice of the cluster's bins. Sized for the largest nbins this
  // kernel is compiled for; `slice` below is what the launch actually uses.
  __shared__ int bins[SLICE];
  cg::cluster_group cluster = cg::this_cluster();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int slice = nbins / CLUSTER;  // bins owned by this block

  // The bin expression must be character-identical to ref_hist()'s in
  // common/reference.hpp, because both have to round the same way:
  //
  //   float v = a[i];
  //   int b = (int)((v + 1.0f) * 0.5f * nbins);
  //   b = min(max(b, 0), nbins - 1);
  //
  // then bin b lives in rank b / slice at local offset b - (b / slice) * slice.
  //
  // The grid is rounded up to a multiple of CLUSTER, so the last cluster
  // contains blocks with no elements at all. They have no work but they are
  // still members: an `if (i >= n) return;` in front of a cluster.sync() leaves
  // three peers waiting on a barrier one participant will never reach, and
  // deallocates shared memory those peers may still be pointing at. Measured on
  // this box that faults -- cudaErrorLaunchFailure, five runs out of five --
  // rather than hanging, but which of the two you get is a scheduling detail
  // you do not control. This is puzzle 24's divergence lesson one level up,
  // with a whole block as the blast radius.
  /// BEGIN CODE HERE (approx 14 lines) ///
  const unsigned rank = cluster.block_rank();
  for (int k = threadIdx.x; k < slice; k += blockDim.x) bins[k] = 0;
  if (rank == 0 && threadIdx.x == 0) meta[0] = cluster.num_blocks();
  cluster.sync();
  if (i < n) {
    float v = a[i];
    int b = (int)((v + 1.0f) * 0.5f * nbins);
    b = min(max(b, 0), nbins - 1);
    const int owner = b / slice;
    int* remote = cluster.map_shared_rank(bins, owner);
    atomicAdd(&remote[b - owner * slice], 1);
  }
  cluster.sync();
  for (int k = threadIdx.x; k < slice; k += blockDim.x)
    atomicAdd(&hist[rank * slice + k], bins[k]);
  /// END CODE HERE ///
}

// The same DSMEM address space, read instead of written, and with the block's
// own data staged first: out[i] = a[i] + (the element TPB further along, if
// that element is staged by a block of the same cluster).
//
// Each block stages its own 256 elements in shared memory, then every block
// except the last rank in its cluster reads its right-hand neighbour's staging
// buffer at the same offset. Nothing crosses a cluster boundary: rank 3 has no
// rank 4 to read, so it adds 0. The elements at the very end of the array add 0
// too, and that costs no code at all -- the out-of-range block staged 0.0f for
// them, which is exactly why the excess blocks must still run the staging
// store.
//
// Both barriers are load-bearing, and they protect opposite things:
//
//   after staging   a peer's tile is not readable until the peer wrote it
//   before exit     a peer must not exit -- deallocating its shared memory --
//                   while this block is still reading it
//
// The second one has no equivalent in any single-block kernel. Inside a block,
// shared memory lives exactly as long as every thread that can see it. Across a
// cluster, one block can retire while another is mid-load out of its smem.
__global__ void __cluster_dims__(CLUSTER, 1, 1)
ClusterShift(const float* a, float* out, int n) {
  __shared__ float tile[TPB];
  cg::cluster_group cluster = cg::this_cluster();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN CODE HERE (approx 10 lines) ///
  tile[threadIdx.x] = (i < n) ? a[i] : 0.0f;
  cluster.sync();
  const unsigned rank = cluster.block_rank();
  float nxt = 0.0f;
  if (rank + 1 < cluster.num_blocks()) {
    const float* peer = cluster.map_shared_rank(tile, rank + 1);
    nxt = peer[threadIdx.x];
  }
  if (i < n) out[i] = a[i] + nxt;
  cluster.sync();
  /// END CODE HERE ///
}
