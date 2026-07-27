# Puzzle 31 asks the profiler to confirm a resource calculation and then to
# explain why that calculation does not predict the timing table.
#
#   --section Occupancy
#       The section this puzzle is named after. `Achieved Occupancy` is what
#       the hardware actually sustained; `Theoretical Occupancy` and
#       `Block Limit Shared Mem` / `Block Limit Registers` / `Block Limit
#       Warps` are the same integer divisions the runner prints from
#       cudaOccupancyMaxActiveBlocksPerMultiprocessor, so the two instruments
#       can be checked against each other. `Block Limit SM` names the winning
#       constraint per kernel -- shared memory for SmemHog, warps for the other
#       two.
#   --section SpeedOfLight
#       Which pipe each kernel is actually near the roof of. PolyOne and
#       PolyFour have identical occupancy and very different Compute (SM)
#       Throughput, which is the whole puzzle in two rows.
#   --section WarpStateStats
#       The mechanism. `Stall Long Scoreboard` is a warp waiting on a memory
#       operand and `Stall Wait` is a warp waiting on a fixed-latency
#       arithmetic dependency; the per-instruction stall breakdown is where
#       PolyOne's and PolyFour's identical occupancy stops being the story.
#       Read `Warp Cycles Per Issued Instruction` alongside it.
#
# -c 3 profiles only the first three launches -- the three correctness
# launches, PolyOne then PolyFour then SmemHog. Without it ncu would replay
# every launch in the file, including the 900-launch timing loop, for no extra
# information.
#
# --kill yes then stops the process as soon as those three are done. That is
# not just a speed-up: ncu's injection library adds tens of microseconds to
# every launch, and these kernels run in 15-25 us, so the runner's own timing
# table would be measuring the profiler. Counters from ncu, timings from
# `make run` -- and the occupancy table from cudaOccupancy*, which is exact
# under both because it reads the binary rather than the clock.
NCU_ARGS = -c 3 --kill yes --section Occupancy --section SpeedOfLight \
           --section WarpStateStats

# This runner asserts a *performance relationship*, and compute-sanitizer's
# instrumentation destroys it: under any of the three tools the wall clock is
# measuring the tool, not the memory system. So the sanitizer runs get the
# runner's documented opt-out and check correctness only. Nothing is skipped
# except the clock -- all three kernels still run in full under all three
# tools, and smem_limits_occupancy is still asserted, because it is integer
# arithmetic over the compiled binary and instrumentation cannot move it.
#
# The root Makefile declares SAN with `?=` before it includes this file, which
# is what lets this plain assignment win.
SAN = env P31_SKIP_TIMING=1 compute-sanitizer --error-exitcode 1
