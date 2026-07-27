# Puzzle 30 profiles the two load paths against each other, so every number the
# analysis rests on has to come out of this one command.
#
#   l1tex__t_sector_hit_rate.pct
#       The headline, over all L1 traffic. The paradox is that the kernel with
#       the high value here is the one the timing table calls slower.
#   l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct
#       The same ratio restricted to global loads, so the always-missing
#       stores do not dilute it. This is the number to quote.
#   l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
#   l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
#   l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
#   l1tex__data_pipe_lsu_wavefronts.sum
#       The volume that the hit RATE is a ratio of -- one warp-wide load
#       instruction is one request, a request resolves to one or more 32 B
#       sectors, and a wavefront is one cycle of L1 data-path work. These
#       counters are the actual answer to the puzzle; the hit rate is the
#       distraction. Watch the sectors/request line especially: the usual
#       reading of it ("closer to 1 is better") points the wrong way here.
#   lts__t_sectors.sum
#       Everything downstream of L1: the sectors L1 actually asked L2 for.
#       Compare the two kernels on this line before concluding anything.
#       (dram__* counters read `n/a` on GB10 -- this SoC does not expose the
#       DRAM-side performance counters, so L2 traffic is the deepest level
#       this box can measure. Confirmed by --query-metrics.)
#
# SpeedOfLight names the limiter for each kernel (compute pipe vs memory pipe
# vs neither), and MemoryWorkloadAnalysis breaks the traffic down per level so
# the L1-side amplification and the identical L2-side traffic are visible side
# by side.
#
# -c 2 profiles only the first two launches -- the two correctness launches,
# GroupSumRedundant then GroupSumStream. Without it ncu would replay all 48
# launches in the file, including the timing loop, for no extra information.
#
# --kill yes then stops the process as soon as those two are done, which is not
# just a speed-up: ncu's injection library adds 12-22 us to every launch here,
# so the runner's own timing table, if it were allowed to run here, would
# report 1.06x instead of the real 2.50x and the relationship check would fail
# on an artefact of the profiler. Counters from ncu, timings from `make run` --
# the same rule puzzle 28 arrived at, enforced by the build this time.
NCU_ARGS = -c 2 --kill yes --section SpeedOfLight --section MemoryWorkloadAnalysis \
           --metrics l1tex__t_sector_hit_rate.pct,l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__data_pipe_lsu_wavefronts.sum,lts__t_sectors.sum

# This runner asserts a *performance relationship*, and compute-sanitizer's
# instrumentation destroys it: measured on this box, memcheck inflates the
# 2.50x to 15.4x and racecheck flattens it to 1.03x, because under either tool
# the wall clock is measuring the instrumentation rather than the memory
# system. A timing assert evaluated there is noise, so the sanitizer runs get
# the runner's documented opt-out and check correctness only -- both kernels
# still run, in full, under all three tools.
#
# The root Makefile declares SAN with `?=` before it includes this file, which
# is what lets this plain assignment win.
SAN = env P30_SKIP_TIMING=1 compute-sanitizer --error-exitcode 1
