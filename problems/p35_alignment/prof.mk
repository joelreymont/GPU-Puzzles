# Puzzle 35 asks the profiler one question -- how many 32-byte sectors did the
# memory system have to move to satisfy each warp-wide access, and how many
# instructions asked for them -- because the whole puzzle is the gap between
# those two numbers.
#
#   l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
#   l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
#       THE metric. Sectors are the granule the memory system moves; a 32-lane
#       load of consecutive floats wants 128 contiguous bytes, which is 4
#       sectors when it starts on a 32-byte boundary. SaxpyScalar and SaxpyVec4
#       report the SAME load-sector count to the sector -- vectorising does not
#       move fewer bytes. SaxpyMisaligned reports more, and the ratio is the
#       whole story.
#   l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
#   l1tex__t_requests_pipe_lsu_mem_global_op_st.sum
#       What the sectors were asked for BY. One warp-wide access is one request
#       regardless of how wide each lane's piece is, so SaxpyVec4 issues exactly
#       a quarter of SaxpyScalar's, and SaxpyMisaligned issues exactly as many
#       as SaxpyScalar. Requests are instructions; sectors are bytes; the two
#       columns are the two things vectorisation and misalignment each change,
#       and they are not the same thing.
#   l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
#   l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio
#       The ratio, computed for you, and the one number to predict before
#       running this. SaxpyScalar loads two operands at 4 sectors each, so 4.00.
#       SaxpyVec4 moves 16 bytes per lane, so 512 bytes per request, so 16.00.
#       SaxpyMisaligned loads `a` at 5 sectors and `b` at 4, so the average over
#       the two is 4.50 -- and that .50 is the entire measurable difference
#       between it and the baseline.
#   smsp__inst_executed.sum
#       Warp-level instruction count, the coarse version of the request column.
#       SaxpyVec4's advantage is here and nowhere else.
#
# SpeedOfLight names what each kernel is near the roof of. All three are the
# same 18 MB of user data resident in a 25 MB L2, so any difference between them
# is the sector path and nothing else.
#
# -c 3 profiles only the first three launches -- the three correctness launches,
# Scalar then Vec4 then Misaligned. Without it ncu would replay every launch in
# the file, including the ~1900-launch timing loop, for no extra information.
#
# --kill yes then stops the process as soon as those three are done. That is not
# just a speed-up. These kernels run in 10-12 us and ncu's injection library
# adds tens of microseconds to every launch, so the runner's own timing table
# would be measuring the profiler; and the runner's last act is a deliberate
# misaligned float4 access that kills the CUDA context on purpose, which is not
# something to hand a profiler. Counters from ncu, timings from `make run`.
#
# Note the metric names. dram__* metrics are not available on GB10 at all, so
# there is no way to ask this device how many bytes reached memory; every
# number above is measured at the L1TEX/L2 boundary, which is where the effect
# this puzzle is about actually lives. `--query-metrics` is the only reliable
# source for what exists on this chip.
NCU_ARGS = -c 3 --kill yes --section SpeedOfLight \
           --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,smsp__inst_executed.sum

# This runner asserts two performance relationships and then deliberately
# faults, and compute-sanitizer's instrumentation ruins the first and correctly
# objects to the second.
#
# P35_SKIP_TIMING: under any of the three tools the wall clock is measuring the
# tool, not the sector path, so the timing section is off. Nothing else is
# skipped -- all three kernels still run in full under all three tools, which is
# what matters here, because the two places an off-by-one hides in this puzzle
# are the float4 kernel's scalar tail and the misaligned kernel's read of a[n],
# and memcheck is the instrument that finds both.
#
# P35_SKIP_UB_PROBE: the runner's last act is an intentional misaligned float4
# load. memcheck reports it as four real "Invalid __global__ read of size 16
# bytes ... is misaligned" errors, which is precisely correct and would fail
# `make check` for a fault the harness caused on purpose. The demonstration
# belongs to `make run`.
#
# The root Makefile declares SAN with `?=` before it includes this file, which
# is what lets this plain assignment win.
SAN = env P35_SKIP_TIMING=1 P35_SKIP_UB_PROBE=1 \
      compute-sanitizer --error-exitcode 1
