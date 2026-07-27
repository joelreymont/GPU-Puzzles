# Puzzle 33 asks the profiler one question -- did the multiply-accumulate
# actually issue to the tensor cores -- and then asks for enough context to
# reconstruct the count from the source. The check is by inspection: every
# tensor counter is a large exact integer for GemmWmma and exactly 0 for
# GemmNaiveFp32, and the FMA counter says the opposite.
#
#   sm__inst_executed_pipe_tensor_subpipe_hmma_op_hmma.sum
#       THE metric: warp-level HMMA instructions retired. This is the one that
#       proves the hardware ran, not merely that the source said wmma. Note the
#       name -- `sm__inst_executed_pipe_tensor_op_hmma.sum`, which older write-
#       ups quote, does not exist on this box; the tensor pipe here is split
#       into hmma/imma subpipes and the metric name follows.
#   sm__inst_executed_pipe_tensor.sum
#       Every tensor-pipe instruction regardless of subpipe. Equal to the HMMA
#       count here, which is how you know nothing else (IMMA, sparsity) is
#       mixed in.
#   sm__ops_path_tensor_op_hmma_src_fp16_dst_fp32.sum
#       Math ops down the FP16-in/FP32-out tensor path. This is the counter
#       that names the precision recipe: it is the fp16-source fp32-destination
#       variant that is nonzero, and it lands on 2*M*N*K exactly.
#   sm__pipe_tensor_cycles_active.sum
#       Cycles the tensor pipe was busy. Divide by the HMMA count for the
#       issue-to-issue cost of one instruction.
#   sm__inst_executed_pipe_fma.sum
#       The control: the general-purpose FP32 pipe. Huge for GemmNaiveFp32,
#       ~0 for GemmWmma. The two kernels compute the same product from the same
#       values; these two counters are where the difference lives.
#   smsp__inst_executed.sum
#       Total warp instructions, all pipes. The headline ratio between the two
#       kernels, and the one number that says plainly how much work the tensor
#       core removed rather than merely relocated.
#   launch__waves_per_multiprocessor
#       How many full passes over the 48 SMs this grid amounts to. Below 1 the
#       device is not full and every throughput figure has to be read in that
#       light, which is the whole caveat on this puzzle's timing table.
#
# SpeedOfLight is what turns the timing table into an explanation, and the
# explanation is not the flattering one. GemmNaiveFp32 runs at 61% of this
# device's throughput roofline over 1.11 waves of the 48 SMs -- it is a
# reasonable kernel being measured near its limit. GemmWmma runs at 8% over
# 0.14 waves; ncu says outright that the grid is too small to fill the device.
# The 4x on the wall clock is therefore a floor imposed by the problem size,
# not the size of the hardware effect: one of the two kernels is being measured
# near its roof and the other is being measured almost entirely idle.
#
# -c 2 profiles only the first two launches -- the two correctness launches,
# GemmNaiveFp32 then GemmWmma. Without it ncu would replay every launch in the
# file, including the 606-launch timing loop, for no extra information.
#
# --kill yes then stops the process as soon as those two are done. That is not
# just a speed-up: ncu's injection library adds tens of microseconds to every
# launch and these kernels run in 6-25 us, so the runner's own timing table
# would be measuring the profiler. Counters from ncu, timings from `make run`.
NCU_ARGS = -c 2 --kill yes --section SpeedOfLight \
           --metrics sm__inst_executed_pipe_tensor_subpipe_hmma_op_hmma.sum,sm__inst_executed_pipe_tensor.sum,sm__ops_path_tensor_op_hmma_src_fp16_dst_fp32.sum,sm__pipe_tensor_cycles_active.sum,sm__inst_executed_pipe_fma.sum,smsp__inst_executed.sum,launch__waves_per_multiprocessor

# This runner asserts a performance relationship, and compute-sanitizer's
# instrumentation destroys it: under any of the three tools the wall clock is
# measuring the tool, not the tensor pipe. So the sanitizer runs get the
# runner's documented opt-out and check correctness only. Nothing is skipped
# except the clock -- both kernels still run in full under all three tools,
# which is what matters here, because a wmma fragment is warp-collective state
# and synccheck on a kernel full of *_sync calls is a result worth having.
#
# The root Makefile declares SAN with `?=` before it includes this file, which
# is what lets this plain assignment win.
SAN = env P33_SKIP_TIMING=1 compute-sanitizer --error-exitcode 1
