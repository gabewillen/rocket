# Correctness tests. Each one returns 77 to skip when the artifact it needs (a
# checkpoint, a second booster) is not on this node.

# Decode kernels against CPU references. Synthetic inputs, so no checkpoint.
add_executable(test-kernels test_kernels.cu)
target_link_libraries(test-kernels PRIVATE rocket_engine rocket_fuel)
set_target_properties(test-kernels PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
add_test(NAME kernels COMMAND test-kernels)
set_tests_properties(kernels PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)

# The layer registry and the tokenizer, both read off the checkpoint.
add_executable(test-fuel-registry test_fuel_registry.cc)
target_link_libraries(test-fuel-registry PRIVATE rocket_fuel)
set_target_properties(test-fuel-registry PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
add_test(NAME fuel-registry COMMAND test-fuel-registry)
set_tests_properties(fuel-registry PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)

# M=8 concurrent streams vs the same engine's M=1 output, token for token.
# Real weights, real expert cache; the streaming half makes this the slowest
# test here.
add_executable(test-batch-parity test_batch_parity.cu)
target_link_libraries(test-batch-parity PRIVATE rocket_engine rocket_fuel)
set_target_properties(test-batch-parity PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
add_test(NAME batch-parity COMMAND test-batch-parity)
# 3600 covered the original M=8 check alone; the M=32 (4x duplicated prompt)
# check added in blog/posts/runtime/2026-09-07-grouped-gemm-beats-gemv-at-
# every-m/'s Next ("batch-parity must stay green at the larger max_batch")
# adds a second concurrent run at 4x the per-step cost over the same step
# count, so the budget is doubled rather than padded arbitrarily.
set_tests_properties(batch-parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 7200)

if(NOT EXISTS "${ROCKET_CUTLASS_DIR}/include/cutlass/cutlass.h")
  message(STATUS
    "test-loader-swizzle skipped: no CUTLASS at ${ROCKET_CUTLASS_DIR}. "
    "Fetch it with engines/glm5-moe-nvfp4-2b/bench/fetch-cutlass.sh")
  return()
endif()

add_executable(test-loader-swizzle test_loader_swizzle.cu)
target_link_libraries(test-loader-swizzle PRIVATE rocket_fuel)
target_include_directories(test-loader-swizzle PRIVATE
  "${ROCKET_CUTLASS_DIR}/include"
  "${ROCKET_CUTLASS_DIR}/tools/util/include"
)
target_compile_options(test-loader-swizzle PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
# Block-scaled FP4 MMA needs the arch-accelerated variant of this SM.
set_target_properties(test-loader-swizzle PROPERTIES
  CUDA_ARCHITECTURES "${ROCKET_CUDA_ARCH}a"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
)

add_test(NAME loader-swizzle COMMAND test-loader-swizzle)
set_tests_properties(loader-swizzle PROPERTIES
  SKIP_RETURN_CODE 77
  TIMEOUT 600
)

# Stage 2 batching arc: grouped-GEMM vs GEMV logits agreement at M=1, plus the
# M-sweep that finds the crossover model.cu::kGroupedMoeMinBatch encodes.
# Real weights, one engine load; see test_moe_grouped.cu's header comment for
# why it stays a single binary.
add_executable(test-moe-grouped test_moe_grouped.cu)
target_link_libraries(test-moe-grouped PRIVATE rocket_engine rocket_fuel)
set_target_properties(test-moe-grouped PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
add_test(NAME moe-grouped COMMAND test-moe-grouped)
# 1800 covered the M in {1,2,4,8} sweep; extending to {16,32} plus a CUDA
# graph on/off timing pass over the same M set (each M pays a graph-capture
# rebuild) roughly triples the wall time this test needs.
set_tests_properties(moe-grouped PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 5400)

# Synthetic-input properties the grouped path relies on (row invariance,
# fused-w13 concatenation, determinism): no checkpoint, seconds not minutes.
add_executable(test-moe-grouped-invariance test_moe_grouped_invariance.cu)
target_link_libraries(test-moe-grouped-invariance PRIVATE rocket_engine)
set_target_properties(test-moe-grouped-invariance PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
add_test(NAME moe-grouped-invariance COMMAND test-moe-grouped-invariance)
set_tests_properties(moe-grouped-invariance PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
