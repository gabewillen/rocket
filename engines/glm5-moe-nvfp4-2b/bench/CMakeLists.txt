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

# Page size for the KV pool: gather locality against sharing granularity, on
# the two kernels that read the cache through the block table every step.
# No CUTLASS, no checkpoint.
rocket_add_bench(kv-page-gather kv_page_gather.cu)

# The agent-workload trace replayed through the real page tables: peak
# resident pages, not a cumulative average. Host only, no CUDA, no checkpoint.
add_executable(bench-kv-capacity-replay kv_capacity_replay.cc)
target_link_libraries(bench-kv-capacity-replay PRIVATE rocket_engine rocket_fuel)
set_target_properties(bench-kv-capacity-replay PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bench")

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

  # One MoE layer step end to end: w13 -> SwiGLU+quantize -> w2, and the
  # fusion/overlap variants of it.
  rocket_add_bench(moe-layer-fusion moe_layer_fusion.cu)
  target_include_directories(bench-moe-layer-fusion PRIVATE
    "${ROCKET_CUTLASS_DIR}/include"
    "${ROCKET_CUTLASS_DIR}/tools/util/include"
  )
  target_compile_options(bench-moe-layer-fusion PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  set_target_properties(bench-moe-layer-fusion PROPERTIES
    CUDA_ARCHITECTURES "${ROCKET_CUDA_ARCH}a"
  )

  # Tile shapes below 128 in M, at per-expert M=2..8.
  rocket_add_bench(nvfp4-tile-sweep nvfp4_tile_sweep.cu)
  target_include_directories(bench-nvfp4-tile-sweep PRIVATE
    "${ROCKET_CUTLASS_DIR}/include"
    "${ROCKET_CUTLASS_DIR}/tools/util/include"
  )
  target_compile_options(bench-nvfp4-tile-sweep PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  set_target_properties(bench-nvfp4-tile-sweep PROPERTIES
    CUDA_ARCHITECTURES "${ROCKET_CUDA_ARCH}a"
  )

  # Does not compile, on purpose: tile M below 128 does not exist for a
  # block-scaled NVFP4 GEMM. Excluded from ALL.
  add_executable(bench-tile-m64-probe EXCLUDE_FROM_ALL tile_m64_probe.cu)
  target_link_libraries(bench-tile-m64-probe PRIVATE rocket_engine)
  target_include_directories(bench-tile-m64-probe PRIVATE
    "${ROCKET_CUTLASS_DIR}/include"
    "${ROCKET_CUTLASS_DIR}/tools/util/include"
  )
  target_compile_options(bench-tile-m64-probe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  set_target_properties(bench-tile-m64-probe PROPERTIES
    CUDA_ARCHITECTURES "${ROCKET_CUDA_ARCH}a"
  )

  # Does not compile, on purpose: it is the record of the epilogue CUTLASS 4.8.0
  # cannot build for a grouped GEMM. Excluded from ALL; build it to see the
  # error quoted in blog/posts/kernels/2026-09-06-moe-layer-fusion/.
  add_executable(bench-evt-swiglu-aux-probe EXCLUDE_FROM_ALL evt_swiglu_aux_probe.cu)
  target_link_libraries(bench-evt-swiglu-aux-probe PRIVATE rocket_engine)
  target_include_directories(bench-evt-swiglu-aux-probe PRIVATE
    "${ROCKET_CUTLASS_DIR}/include"
    "${ROCKET_CUTLASS_DIR}/tools/util/include"
  )
  target_compile_options(bench-evt-swiglu-aux-probe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  set_target_properties(bench-evt-swiglu-aux-probe PROPERTIES
    CUDA_ARCHITECTURES "${ROCKET_CUDA_ARCH}a"
  )
else()
  message(STATUS
    "bench-nvfp4-grouped-gemm skipped: no CUTLASS at ${ROCKET_CUTLASS_DIR}. "
    "Fetch it with engines/glm5-moe-nvfp4-2b/bench/fetch-cutlass.sh")
endif()

# Attention lane. These need no CUTLASS: the questions are state traffic and
# gather traffic, and both are answered with plain CUDA at the checkpoint's
# real dims (fuels/glm-5.3-flash/attention.yaml).
#
#   kda-decode-step   34 KDA layers carry a 64x128x128 recurrent state per
#                     stream; read+write cost per layer per step, M=8..64.
rocket_add_bench(kda-decode-step kda_decode_step.cu)
set_target_properties(bench-kda-decode-step PROPERTIES
  CUDA_ARCHITECTURES "${ROCKET_CUDA_ARCH}"
)

#   sparse-mla-decode gather top-k KV from a 262k paged pool and run the MLA
#                     math in absorbed form, against dense 8k.
rocket_add_bench(sparse-mla-decode sparse_mla_decode.cu)
set_target_properties(bench-sparse-mla-decode PROPERTIES
  CUDA_ARCHITECTURES "${ROCKET_CUDA_ARCH}"
)
target_link_libraries(bench-sparse-mla-decode PRIVATE CUDA::cublas)

#   dsa-indexer-scan  score all 262144 candidates per stream, which is what
#                     feeds the top-2048 selection.
rocket_add_bench(dsa-indexer-scan dsa_indexer_scan.cu)
set_target_properties(bench-dsa-indexer-scan PROPERTIES
  CUDA_ARCHITECTURES "${ROCKET_CUDA_ARCH}"
)
