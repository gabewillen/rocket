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
