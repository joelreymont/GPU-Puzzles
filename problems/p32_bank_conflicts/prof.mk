# Puzzle 32 asks the profiler one question -- how many extra cycles did the
# shared-memory data stage spend because two lanes wanted the same bank -- and
# then asks for enough context to reconstruct that number from the source.
#
#   l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
#       THE metric. Extra data-stage cycles caused by conflicting shared LOADS,
#       which is where a column read of a [32][32] tile lands. Large and
#       nonzero for TransposeNaive, exactly 0 for the other two.
#   l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
#       The store-side twin. All three kernels write the tile row-major, so
#       this is 0 everywhere -- which is the point: the conflict is a property
#       of one access, not of the kernel, and the naive kernel's other shared
#       access is already perfect.
#   l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
#   l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
#       What the conflict count is extra *to*. A wavefront is one cycle of L1
#       shared data-path work; a conflict-free warp-wide access is one
#       wavefront, an N-way conflict is N. conflicts = wavefronts - requests,
#       and having both columns is what turns the counter into a ratio you can
#       check against the access pattern by hand.
#   smsp__inst_executed_op_shared_ld.sum
#   smsp__inst_executed_op_shared_st.sum
#       The warp-level instruction counts -- how many LDS and STS were issued
#       at all. Divide the wavefronts by these and you get the average conflict
#       degree per instruction, which should come out at the tile width for
#       TransposeNaive and at 1.00 for the other two.
#
# SpeedOfLight names what each kernel is actually near the roof of, and the two
# conflict-free kernels are a different answer from the naive one: the same
# global traffic, moved at very different effective bandwidth, with nothing
# about the global access pattern changed to explain it.
#
# -c 3 profiles only the first three launches -- the three correctness
# launches, Naive then Padded then Swizzle. Without it ncu would replay every
# launch in the file, including the 1035-launch timing loop, for no extra
# information.
#
# --kill yes then stops the process as soon as those three are done. That is
# not just a speed-up: ncu's injection library adds tens of microseconds to
# every launch and these kernels run in 7-13 us, so the runner's own timing
# table would be measuring the profiler. Counters from ncu, timings from
# `make run`.
NCU_ARGS = -c 3 --kill yes --section SpeedOfLight \
           --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,smsp__inst_executed_op_shared_ld.sum,smsp__inst_executed_op_shared_st.sum

# This runner asserts a *performance relationship*, and compute-sanitizer's
# instrumentation destroys it: under any of the three tools the wall clock is
# measuring the tool, not the shared-memory pipe. So the sanitizer runs get the
# runner's documented opt-out and check correctness only. Nothing is skipped
# except the clock -- all three kernels still run in full under all three
# tools, which is what matters here, because racecheck on the barrier between
# the tile load and the transposed store is a required result for this puzzle
# and not an optional one.
#
# The root Makefile declares SAN with `?=` before it includes this file, which
# is what lets this plain assignment win.
SAN = env P32_SKIP_TIMING=1 compute-sanitizer --error-exitcode 1
