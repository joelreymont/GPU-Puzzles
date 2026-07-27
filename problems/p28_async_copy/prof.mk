# Puzzle 28 profiles the async-copy path itself.
#
# LDGSTS is the SASS instruction __pipeline_memcpy_async lowers to: one
# instruction that moves global memory straight into shared memory without a
# register round-trip. The two metrics below are the proof that the async
# kernel really took that path and the sync kernel did not --
# smsp__inst_executed_op_ldgsts.sum counts the instructions issued, and
# l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum counts the
# shared-memory-side wavefronts they generated. Both must be zero for
# WindowSumSync and non-zero for WindowSumAsync.
#
# SpeedOfLight is what the puzzle's argument rests on: if the memory pipe is
# already near 100% of its achievable throughput, there is no idle time for a
# prefetch to hide in, and the two kernels will land on the same number.
#
# -c 2 profiles only the first two launches -- the two correctness launches,
# sync then async. Without it ncu would replay all 112 launches in the file,
# including the timing loop, for no extra information.
NCU_ARGS = -c 2 --section SpeedOfLight \
           --metrics smsp__inst_executed_op_ldgsts.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum
