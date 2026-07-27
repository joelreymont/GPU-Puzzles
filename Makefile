ARCH ?= native
NVCC ?= nvcc
# RmProfilingAdminOnly=1 on this box; /etc/sudoers.d/joel-ncu grants NOPASSWD
# for exactly this binary, so profiling works without interaction.
NCU ?= sudo /usr/local/cuda/bin/ncu
MODE ?= skeleton
NVCCFLAGS = -arch=$(ARCH) -lineinfo -O2 -Xptxas -v -Icommon

# Empirical hardware capabilities, cached from a real run of the probe.
-include common/caps.mk

common/caps.mk: scripts/probe_caps.cu common/puzzle_utils.cuh
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -o build/probe_caps scripts/probe_caps.cu
	./build/probe_caps > $@
	@cat $@

ifdef P

PDIR := $(notdir $(wildcard problems/p$(P)_*))
ifeq ($(PDIR),)
$(error no puzzle matching P=$(P))
endif
ifeq ($(filter $(MODE),skeleton solution),)
$(error MODE must be skeleton or solution, got '$(MODE)')
endif
# Puzzle 34 is the one puzzle whose existence depends on a measured hardware
# fact rather than a documented one, so it is the one puzzle with a build gate.
# scripts/probe_caps.cu launched a real 2-block cluster and a real
# cluster.sync() on this box and cached the result; if that ever stops holding,
# stop here rather than build a puzzle that deadlocks the GPU (brief section 5).
ifeq ($(PDIR),p34_clusters)
ifneq ($(CAPS_CLUSTER_SYNC),1)
$(error p34_clusters needs working thread block clusters, but common/caps.mk \
records CAPS_CLUSTER_LAUNCH=$(CAPS_CLUSTER_LAUNCH) \
CAPS_CLUSTER_MAX=$(CAPS_CLUSTER_MAX) CAPS_CLUSTER_REAL=$(CAPS_CLUSTER_REAL) \
CAPS_CLUSTER_SYNC=$(CAPS_CLUSTER_SYNC). Re-measure with scripts/probe_caps.cu \
(rm common/caps.mk && make common/caps.mk). If cluster.sync() genuinely does \
not complete on this GPU, replace puzzle 34 per brief section 5 -- do not patch it)
endif
endif

KDIR := $(if $(filter solution,$(MODE)),solutions,skeletons)
BIN := build/$(MODE)/$(PDIR)
SAN ?= compute-sanitizer --error-exitcode 1
-include problems/$(PDIR)/prof.mk
NCU_ARGS ?= --section SpeedOfLight --section MemoryWorkloadAnalysis

build: $(BIN)

$(BIN): problems/$(PDIR)/runner.cu $(KDIR)/$(PDIR)/kernel.cu \
        common/puzzle_utils.cuh common/reference.hpp
	@mkdir -p build/$(MODE)
	$(NVCC) $(NVCCFLAGS) -o $@ problems/$(PDIR)/runner.cu $(KDIR)/$(PDIR)/kernel.cu

run: $(BIN)
	./$(BIN)

check: $(BIN)
	$(SAN) --tool memcheck ./$(BIN)
	$(SAN) --tool racecheck ./$(BIN)
	$(SAN) --tool synccheck ./$(BIN)

prof: $(BIN)
	$(NCU) $(NCU_ARGS) ./$(BIN)

else

build run check prof:
	$(error target '$@' needs P=<puzzle number>, e.g. make $@ P=24)

endif

test:
	@fail=0; for d in solutions/p*/; do \
	  n=$$(basename $$d); n=$${n#p}; n=$${n%%_*}; \
	  $(MAKE) --no-print-directory run P=$$n MODE=solution || fail=1; \
	done; exit $$fail

sync:
	python3 scripts/check_sync.py

clean:
	rm -rf build

.PHONY: build run check prof test sync clean
