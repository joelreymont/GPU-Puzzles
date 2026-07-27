#include "puzzle_utils.cuh"
#include "reference.hpp"

#include <cstring>

// Declared without __cluster_dims__ on purpose: the attribute belongs to the
// definition and travels with the kernel in the fatbin, not with this
// declaration. That is precisely why `meta` exists below -- from the host side
// there is nothing to inspect that proves the launch was clustered, so the
// kernel has to say so itself.
extern __global__ void ClusterHistogram(const float* a, int* hist, int* meta,
                                        int nbins, int n);
extern __global__ void ClusterShift(const float* a, float* out, int n);

// The §5 hardware question -- do thread block clusters work on sm_121 at all --
// was not answerable from documentation, and the sources that exist contradict
// each other. It was settled by running scripts/probe_caps.cu on this box,
// whose findings are cached in common/caps.mk and gate this puzzle's build.
// Reproduce them here rather than restating them from memory: a number that
// came out of a file the probe wrote cannot drift away from what the probe
// measured, and a hardcoded one can.
static void print_caps_file() {
  static const char* kPaths[] = {"common/caps.mk", "../common/caps.mk",
                                 "../../common/caps.mk"};
  for (const char* path : kPaths) {
    FILE* f = fopen(path, "r");
    if (!f) continue;
    printf("# %s, measured by scripts/probe_caps.cu on this box:\n", path);
    char line[256];
    while (fgets(line, sizeof line, f)) {
      if (line[0] == '#') continue;
      line[strcspn(line, "\r\n")] = '\0';
      if (line[0] != '\0') printf("#   %s\n", line);
    }
    fclose(f);
    return;
  }
  printf("# common/caps.mk not readable from this working directory; run this"
         " binary from the repo root (make run P=34) to see the cached probe"
         " results.\n");
}

// The occupancy API's answer for a kernel that already carries
// __cluster_dims__: the largest cluster this kernel's register and shared
// memory footprint can still co-schedule. It is a capability question, not a
// launch, so it never runs anything and never proves the hardware groups
// blocks -- meta[0] does that.
static void print_kernel_info(const char* name, const void* kern, dim3 grid,
                              int tpb) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(tpb, 1, 1);
  cfg.dynamicSmemBytes = 0;
  int maxc = -1;
  CHECK_CUDA(cudaOccupancyMaxPotentialClusterSize(&maxc, kern, &cfg));
  cudaFuncAttributes fa;
  CHECK_CUDA(cudaFuncGetAttributes(&fa, kern));
  printf("#   %-16s max potential cluster size %d, %zu B static smem/block,"
         " %d regs\n",
         name, maxc, fa.sharedSizeBytes, fa.numRegs);
}

int main() {
  print_device_info();

  // Not a multiple of TPB, and cdiv(n, TPB) = 3907 is not a multiple of
  // CLUSTER either, so both guards this puzzle cares about are exercised: the
  // partial block at the end of the array, and the entirely empty block that
  // rounding the grid up to a whole number of clusters creates.
  const int n = 1000003;
  const int TPB = 256;      // must match the kernels' compile-time TPB
  const int CLUSTER = 4;    // must match the kernels' __cluster_dims__
  const int NBINS = 256;    // must match the kernels' compile-time NBINS
  // ClusterShift performs one float add per element in the same order as the
  // reference, so the tolerance is a formality rather than a reassociation
  // budget; it is not zero only because compare() takes a tolerance.
  const float tol = 1e-6f;

  // A cluster launch is rejected outright unless the grid divides evenly into
  // clusters, so the block count is rounded UP -- the excess blocks own no
  // elements but are full members of their cluster's barriers.
  const int blocks = cdiv(cdiv(n, TPB), CLUSTER) * CLUSTER;
  const dim3 grid(blocks);

  printf("# n = %d, TPB = %d, CLUSTER = %d, NBINS = %d\n", n, TPB, CLUSTER,
         NBINS);
  printf("# grid: %d blocks needed for %d elements, rounded up to %d = %d"
         " clusters of %d\n",
         cdiv(n, TPB), n, blocks, blocks / CLUSTER, CLUSTER);
  printf("# tail: block %d holds %d live elements, block %d holds none and"
         " exists only to complete its cluster\n",
         cdiv(n, TPB) - 1, n - (cdiv(n, TPB) - 1) * TPB, blocks - 1);
  int cluster_launch = -1;
  CHECK_CUDA(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch,
                                    0));
  printf("# cudaDevAttrClusterLaunch on this device right now: %d\n",
         cluster_launch);
  printf("# cudaOccupancyMaxPotentialClusterSize for the kernels under test:\n");
  print_kernel_info("ClusterHistogram", (const void*)ClusterHistogram, grid,
                    TPB);
  print_kernel_info("ClusterShift", (const void*)ClusterShift, grid, TPB);
  print_caps_file();

  float* a = new float[n];
  float* got = new float[n];
  float* want = new float[n];
  int* got_h = new int[NBINS];
  int* want_h = new int[NBINS];
  fill_rand(a, n, 34);

  float *d_a, *d_out;
  int *d_hist, *d_meta;
  CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_hist, NBINS * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_meta, sizeof(int)));
  CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));

  bool ok = true;

  // The histogram is an atomicAdd accumulator, so zero is its identity and the
  // only correct initial value -- the same reason puzzle 27 zeroes rather than
  // poisons. It still fails loudly against a kernel that writes nothing,
  // because every expected bin count here is in the thousands. `meta` is not an
  // accumulator and is poisoned with 0xff, so an unwritten meta reads -1.
  ref_hist(a, want_h, NBINS, n);
  CHECK_CUDA(cudaMemset(d_hist, 0, NBINS * sizeof(int)));
  CHECK_CUDA(cudaMemset(d_meta, 0xff, sizeof(int)));
  ClusterHistogram<<<grid, TPB>>>(d_a, d_hist, d_meta, NBINS, n);
  CHECK_LAUNCH();
  int meta = 0;
  CHECK_CUDA(cudaMemcpy(&meta, d_meta, sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(got_h, d_hist, NBINS * sizeof(int),
                        cudaMemcpyDeviceToHost));

  // Geometry first: if the blocks did not actually become a cluster, every
  // number below is noise and the reason is worth more than the numbers.
  if (meta == CLUSTER) {
    printf("PASS cluster_geometry (cluster_group::num_blocks() = %d)\n", meta);
  } else {
    ok = false;
    printf("FAIL cluster_geometry: cluster_group::num_blocks() reported %d,"
           " expected %d\n",
           meta, CLUSTER);
    if (meta == -1) {
      printf("FAIL   meta[0] still holds the 0xff poison, so nothing wrote it:"
             " the kernel is unimplemented, or rank 0 never reached the"
             " write.\n");
    } else {
      printf("FAIL   this is the silent-downgrade failure mode the brief's"
             " section 5 warns about: the launch is ACCEPTED, the"
             " __cluster_dims__ attribute is quietly ignored, the blocks never"
             " become a cluster, and a subsequent cluster.sync() has no peers"
             " to wait for. Every distributed-shared-memory access above is"
             " then reading this block's own smem under another rank's name."
             " Re-run scripts/probe_caps.cu: if the hardware story changed,"
             " this puzzle must be replaced, not patched.\n");
    }
  }

  // Integer counts do not reassociate: compare exactly, across all NBINS.
  ok = compare("cluster_histogram", got_h, want_h, NBINS) && ok;

  // ClusterShift owns every element, so its output is poisoned with 0xff -- a
  // NaN bit pattern. An element the kernel forgets to write is a FAIL, not a
  // lucky zero, which matters here because "add 0" is the correct answer for
  // the last rank of every cluster and for the tail of the array.
  ref_cluster_shift(a, want, n, TPB, CLUSTER);
  CHECK_CUDA(cudaMemset(d_out, 0xff, n * sizeof(float)));
  ClusterShift<<<grid, TPB>>>(d_a, d_out, n);
  CHECK_LAUNCH();
  CHECK_CUDA(cudaMemcpy(got, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
  ok = compare("cluster_shift", got, want, n, tol) && ok;

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_out));
  CHECK_CUDA(cudaFree(d_hist));
  CHECK_CUDA(cudaFree(d_meta));
  delete[] a;
  delete[] got;
  delete[] want;
  delete[] got_h;
  delete[] want_h;
  return ok ? 0 : 1;
}
