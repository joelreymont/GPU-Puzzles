ARCH ?= native
NVCC ?= nvcc
NVCCFLAGS = -arch=$(ARCH) -lineinfo -O2

PUZZLES = map zip guards map2d broadcast blocks map2d_block shared \
          pooling dotproduct 1dconv prefixsum axis_sum matmul

BIN = bin
BINS = $(PUZZLES:%=$(BIN)/%)

all: $(BINS)

$(BIN)/%: %_runner.cu %_kernel.cu | $(BIN)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(BIN):
	mkdir -p $(BIN)

test: all
	@fail=0; for p in $(PUZZLES); do \
	  if ./$(BIN)/$$p >/dev/null 2>&1; then echo "PASS $$p"; \
	  else echo "FAIL $$p"; fail=1; fi; \
	done; exit $$fail

clean:
	rm -rf $(BIN)

.PHONY: all test clean
