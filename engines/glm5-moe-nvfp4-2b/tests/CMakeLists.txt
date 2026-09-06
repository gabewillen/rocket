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
set_tests_properties(batch-parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 3600)

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
