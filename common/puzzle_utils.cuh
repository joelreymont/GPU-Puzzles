#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <unistd.h>

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

// ---------------------------------------------------------------------------
// Measurement validity: whether a wall-clock number is a measurement of this
// GPU, or of the rest of the box.
//
// Every timing assert in this repository is a claim about two kernels, and it is
// only that if nothing else was using the machine while it was measured. This is
// a shared-memory SoC -- one LPDDR5X behind both the GPU and the twenty CPUs --
// so when something else takes a large share of it, the kernels being compared
// stop being limited by whatever the puzzle is about and start being limited by
// the contention, and every ratio in the file collapses toward 1.0 at once. That
// is physics, not noise, and no estimator recovers an effect that is not
// present.
//
// So a run that did not get the machine cannot assert, and it also cannot pass.
// There are exactly three outcomes, and no silent one:
//
//   valid, ratio >= margin   PASS <assert>            exit 0
//   valid, ratio <  margin   FAIL <assert>            exit 1   the claim is false
//   not valid                FAIL measurement_invalid exit 1   the run is void
//
// The third is a FAIL because a green suite has to mean the asserts were
// evaluated. It is deliberately NOT phrased as the puzzle's claim being false --
// the run does not know whether the claim holds, which is the whole point -- and
// it names the evidence that voided it.
//
// Two detectors, deliberately different in kind, because they fail differently.
//
// (a) Competing CUDA processes, from the driver's own accounting. Causal: it
//     names what else was on the GPU, and it fires whether or not the
//     contention happened to be visible in the timings.
//
//     Measured on this box (GB10, driver 580.159.03) against a scratch
//     bandwidth hog: nvmlDeviceGetComputeRunningProcesses_v3 reported it by
//     pid, by name and by device memory, and reported this process too, which
//     is why getpid() is excluded below. On an idle GPU it returns zero
//     processes -- the desktop's Xorg and gnome-shell come back from the
//     GRAPHICS-side query, not this one, so they do not void every run.
//
//     The alternative, scanning /proc/<pid>/fd for links into /dev/nvidia*,
//     was measured too and found the same hog, but it is strictly weaker here:
//     386 of ~390 /proc/<pid>/fd directories are unreadable to an ordinary
//     user, so it can only ever see the caller's own processes, while NVML
//     crosses user boundaries. It is not implemented, because a second and
//     blinder answer to the same question is not a second detector.
//
//     NVML is resolved with dlopen at runtime rather than linked, so no puzzle
//     needs an extra -l on its compile line. If libnvidia-ml.so.1 cannot be
//     loaded or queried, this detector says so on the output -- loudly, on the
//     line above the verdict -- and the run leans on (b) alone. It is never
//     silently absent.
//
// (b) A throughput floor on the run's reference kernel, which each puzzle sets
//     from its own measured distribution. Symptomatic rather than causal, and it
//     is what catches the load (a) cannot see at all: a CPU process on the other
//     side of the same LPDDR5X holds no CUDA context and appears in no GPU-side
//     accounting, but it is what actually flattens these ratios on this box.
//
// Where (a) is sampled: immediately before and immediately after the timed
// section, and the two sets are unioned. Not inside it -- the query costs 25 us
// typically but 3.2 ms at worst, measured over 200 calls on this box, which is
// more than a whole rep of most of these kernels and would perturb the thing
// being measured. A process that both starts and exits strictly inside that
// window is invisible to (a); if it mattered, it slowed the kernels down, and
// that is (b)'s job.

// NVML's per-process record as of the _v3 entry point. Declared here rather
// than included, because nvml.h is not part of the CUDA toolkit's default
// include path and this file resolves the library at runtime anyway.
struct NvmlProc {
  unsigned pid;
  unsigned long long used_mem;
  unsigned gi;
  unsigned ci;
};

// The three NVML entry points this needs, resolved once per process.
struct Nvml {
  void* dev;
  int (*procs)(void*, unsigned*, NvmlProc*);
  int (*pname)(unsigned, char*, unsigned);
};

// Returns the resolved NVML entry points, or nullptr if the library is not
// usable on this machine. Deliberately never calls nvmlShutdown or dlclose:
// the handles live for the length of the process and there is nothing to order
// their teardown against.
//
// _v3 is the only entry point resolved. _v2 and the unversioned symbol exist on
// this driver, but _v3 has shipped since driver 510 and sm_121 does not run on
// anything older than 570, so falling back to them would be unreachable code
// with a different struct layout -- see AGENTS.md on portability shims.
inline const Nvml* nvml() {
  static Nvml n = [] {
    Nvml r{nullptr, nullptr, nullptr};
    void* lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!lib) return r;
    auto init = (int (*)())dlsym(lib, "nvmlInit_v2");
    auto handle =
        (int (*)(unsigned, void**))dlsym(lib, "nvmlDeviceGetHandleByIndex_v2");
    auto procs = (int (*)(void*, unsigned*, NvmlProc*))dlsym(
        lib, "nvmlDeviceGetComputeRunningProcesses_v3");
    if (!init || !handle || !procs || init() != 0) return r;
    void* dev = nullptr;
    if (handle(0, &dev) != 0 || dev == nullptr) return r;
    r.dev = dev;
    r.procs = procs;
    r.pname = (int (*)(unsigned, char*, unsigned))dlsym(
        lib, "nvmlSystemGetProcessName");
    return r;
  }();
  return n.dev != nullptr ? &n : nullptr;
}

// One other process holding a CUDA context on this device. The name is NVML's,
// which is the full argv[0] path; 192 B is enough for one that is not truncated
// into something misleading.
struct GpuProc {
  unsigned pid;
  unsigned long long mem;
  char name[192];
};

// The validity gate a timing section sits behind. Configure it with the
// reference kernel -- the timed kernel with the least slack of its own, so the
// one whose throughput moves first when the machine is taken away -- its rate
// unit, the floor below which the run is not a measurement, and what it reaches
// on an idle box (quoted in the diagnosis, so a reader can see how far off the
// run was).
//
// Usage, from a runner:
//
//   QuietCheck qc{"PolyFour", "Gelem/s", QUIET_GELEM, IDLE_GELEM};
//   qc.watch();                     // before the timed section
//   bench_all(...);
//   qc.watch();                     // after it
//   qc.report();                    // says what it found, every run
//   if (!qc.valid(rate)) { qc.fail("occupancy_not_predictive"); ok = false; }
//   else if (ratio >= MARGIN) { ...PASS... } else { ...FAIL...; ok = false; }
struct QuietCheck {
  const char* kern;
  const char* unit;
  double floor;
  double idle;

  // Eight is more competing processes than any diagnosis needs to name; past
  // that the count is the evidence, not the list.
  static const int MAX = 8;
  GpuProc other[MAX] = {};
  int n_other = 0;    // distinct competing pids, unioned over the samples
  int dropped = 0;    // competing pids seen past MAX
  int samples = 0;    // watch() calls
  int answered = 0;   // watch() calls the causal detector answered
  double rate = 0.0;  // what valid() was given, kept for the diagnosis

  bool causal_ok() const { return samples > 0 && answered == samples; }

  // Unions this instant's set of OTHER compute processes into `other`.
  void watch() {
    samples++;
    const Nvml* nv = nvml();
    if (nv == nullptr) return;
    NvmlProc buf[64];
    unsigned n = (unsigned)(sizeof buf / sizeof buf[0]);
    const int rc = nv->procs(nv->dev, &n, buf);
    // NVML_ERROR_INSUFFICIENT_SIZE (7) means there are more processes than the
    // buffer holds, and `n` comes back as how many. It is still an answer, but
    // `buf` is NOT filled in that case, so the count is all there is: record it
    // as dropped rather than reading uninitialised records.
    if (rc == 7) {
      answered++;
      dropped += (int)n;
      return;
    }
    if (rc != 0) return;
    answered++;
    const unsigned self = (unsigned)getpid();
    for (unsigned i = 0; i < n; i++) {
      if (buf[i].pid == self) continue;
      bool seen = false;
      for (int j = 0; j < n_other; j++) {
        if (other[j].pid == buf[i].pid) seen = true;
      }
      if (seen) continue;
      if (n_other == MAX) {
        dropped++;
        continue;
      }
      GpuProc& p = other[n_other++];
      p.pid = buf[i].pid;
      p.mem = buf[i].used_mem;
      p.name[0] = '\0';
      if (nv->pname == nullptr ||
          nv->pname(p.pid, p.name, (unsigned)sizeof p.name) != 0) {
        snprintf(p.name, sizeof p.name, "name unavailable");
      }
    }
  }

  // True when this run is a measurement of these kernels and the asserts may be
  // evaluated. `measured` is the reference kernel's achieved rate.
  bool valid(double measured) {
    if (samples < 2) {
      // A harness bug, not a machine state: without a sample on each side of
      // the timed section the causal detector is guarding nothing. Refusing to
      // guess is the whole point of this file.
      fprintf(stderr,
              "harness bug: QuietCheck::valid() with %d watch() calls; it must"
              " be sampled on both sides of the timed section\n",
              samples);
      exit(1);
    }
    rate = measured;
    return n_other == 0 && dropped == 0 && measured >= floor;
  }

  // What the detectors found, printed on every run whatever the verdict, so the
  // gate is checkable rather than believable.
  void report() const {
    if (!causal_ok()) {
      printf("# competing GPU processes: NOT CHECKED -- libnvidia-ml.so.1 could"
             " not be loaded or queried (%d of %d samples answered), so the"
             " %s floor below is the only thing guarding this run\n",
             answered, samples, kern);
      return;
    }
    if (n_other == 0 && dropped == 0) {
      printf("# competing GPU processes: none, sampled before and after the"
             " timed section (NVML compute-process accounting; this process"
             " excluded)\n");
      return;
    }
    printf("# competing GPU processes: %d%s", n_other + dropped,
           n_other > 0 ? " --" : " (too many to name)");
    for (int i = 0; i < n_other; i++) {
      printf("%s pid %u %s, %.0f MiB", i ? "," : "", other[i].pid,
             other[i].name, (double)other[i].mem / 1048576.0);
    }
    if (n_other > 0 && dropped > 0) printf(", and %d more", dropped);
    printf("\n");
  }

  // The verdict for a run that is not a measurement. `asserts` names what was
  // therefore not evaluated.
  void fail(const char* asserts) const {
    char why[512];
    int k = 0;
    const int competing = n_other + dropped;
    if (competing > 0) {
      k += snprintf(why + k, sizeof why - k,
                    "%d other process%s held a CUDA context on this GPU while"
                    " the section was timed%s",
                    competing, competing == 1 ? "" : "es",
                    n_other > 0 ? " --" : " (too many to name)");
      for (int i = 0; i < n_other && k < (int)sizeof why; i++) {
        k += snprintf(why + k, sizeof why - k, "%s pid %u %s, %.0f MiB",
                      i ? "," : "", other[i].pid, other[i].name,
                      (double)other[i].mem / 1048576.0);
      }
      if (n_other > 0 && dropped > 0 && k < (int)sizeof why) {
        k += snprintf(why + k, sizeof why - k, ", and %d more", dropped);
      }
    }
    if (rate < floor && k < (int)sizeof why) {
      k += snprintf(why + k, sizeof why - k,
                    "%s%s ran at %.1f %s, under the %.0f %s floor and well"
                    " under the %.0f %s it reaches on an idle box",
                    k ? "; also " : "", kern, rate, unit, floor, unit, idle,
                    unit);
    }
    printf("FAIL measurement_invalid: %s -- another process had the"
           " GPU/memory system; this run proves nothing about the kernels."
           " Re-run on an idle box.\n",
           why);
    printf("FAIL   %s NOT evaluated. This run can neither confirm nor refute"
           " the puzzle's claim, so it does not get to pass: a green suite has"
           " to mean the asserts were checked. Everything above this line ran"
           " and is unaffected -- correctness is not a measurement of the"
           " machine, and neither is arithmetic on the compiled kernels.\n",
           asserts);
    if (!causal_ok()) {
      printf("FAIL   (the competing-process detector did not run on this"
             " machine, so the throughput floor is the only evidence here; a"
             " CPU process on the other side of this SoC's LPDDR5X would look"
             " exactly like this and would not have shown up in it either)\n");
    }
  }
};
