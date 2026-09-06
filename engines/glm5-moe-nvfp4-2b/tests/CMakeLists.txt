# Correctness tests. Each one runs under ctest and returns 77 to skip when the
# artifact it needs (a checkpoint, a second booster) is not on this node.

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
