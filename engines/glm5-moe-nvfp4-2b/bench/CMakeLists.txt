# Microbenchmarks for this engine. Each one answers a question that gets a
# verdict in blog/posts/, so a bench is committed with the entry that cites it.
#
# The workload this engine is built for is concurrent agent and subagent
# streams: many parallel sequences, forked prefixes, bursty tool-result
# prefills. Benches measure that regime, not single-stream decode.
#
# Coming, in order:
#   nvfp4-grouped-gemm   NVFP4 grouped GEMM across expert batch M=8..64,
#                        against dequant-to-FP16 GEMM at the same M.
#   paged-kv-read        paged-KV read bandwidth under multi-stream load,
#                        block-gather at 64 KiB granularity
#                        (blog/posts/hardware/2026-09-06-measured-bandwidth/).
#
# Add one with:
#   rocket_add_bench(nvfp4-grouped-gemm nvfp4_grouped_gemm.cu)

function(rocket_add_bench name)
  if(ARGC LESS 2)
    message(FATAL_ERROR "rocket_add_bench(${name}) needs at least one source")
  endif()
  add_executable(bench-${name} ${ARGN})
  target_link_libraries(bench-${name} PRIVATE rocket_engine)
  set_target_properties(bench-${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bench"
  )
endfunction()

# CUTLASS is header-only for the kernel. Pinned checkout and exact commit:
#   third_party/cutlass, ROCKET_PINNED_SHA
# sm_121a is an upstream-supported arch (CUTLASS_NVCC_ARCHS_SUPPORTED, CUDA>=12.9);
# no patch is needed to target this box.
find_package(CUDAToolkit REQUIRED)

set(ROCKET_CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cutlass")

if(EXISTS "${ROCKET_CUTLASS_DIR}/include/cutlass/cutlass.h")
  rocket_add_bench(nvfp4-grouped-gemm nvfp4_grouped_gemm.cu)
  target_include_directories(bench-nvfp4-grouped-gemm PRIVATE
    "${ROCKET_CUTLASS_DIR}/include"
    "${ROCKET_CUTLASS_DIR}/tools/util/include"
  )
  # Block-scaled FP4 MMA needs the arch-accelerated variant of this SM.
  target_compile_options(bench-nvfp4-grouped-gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  set_target_properties(bench-nvfp4-grouped-gemm PROPERTIES
    CUDA_ARCHITECTURES "${ROCKET_CUDA_ARCH}a"
  )
  target_link_libraries(bench-nvfp4-grouped-gemm PRIVATE CUDA::cublas)
else()
  message(STATUS
    "bench-nvfp4-grouped-gemm skipped: no CUTLASS at ${ROCKET_CUTLASS_DIR}. "
    "Fetch it with engines/glm5-moe-nvfp4-2b/bench/fetch-cutlass.sh")
endif()
