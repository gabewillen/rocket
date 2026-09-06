# Engine: glm5-moe-nvfp4-2b
#
# Chemistry: glm5-moe-288e-top8-kda-dsa   (blog/fuels/glm-5.3-flash.qmd)
# Weight format: NVFP4, 4-bit experts
# Boosters: 2, split arrangement (blog/rocket.qmd#boosters)
# Target workload: concurrent agent/subagent streams, not single-stream decode.
#
# Hardware is fixed at build time; there is no generic-arch fallback.
#   GB10, compute capability 12.1 (sm_121)
#   nvcc 13.0 V13.0.88
#   64 KiB host pages (6.17.13-rocket64k)
cmake_minimum_required(VERSION 3.28)

project(rocket_glm5_moe_nvfp4_2b LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# One architecture. A build that would run somewhere else is a build for a
# machine this engine does not target.
set(ROCKET_CUDA_ARCH 121 CACHE STRING "CUDA architecture, sm_<value>")
set(CMAKE_CUDA_ARCHITECTURES ${ROCKET_CUDA_ARCH})

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
  message(FATAL_ERROR
    "nvcc ${CMAKE_CUDA_COMPILER_VERSION} is older than 13.0; sm_${ROCKET_CUDA_ARCH} needs CUDA 13.x")
endif()

# 64 KiB is the universal quantum on this platform (blog/rocket.qmd#alignment).
set(ROCKET_PAGE_BYTES 65536 CACHE STRING "Host page size the plan is aligned to")

add_library(rocket_engine STATIC
  src/engine.cu
  src/kernels.cu
  src/weights.cu
  src/model.cu
  src/kv/page_pool.cc
  src/kv/kv_arena.cu
)
target_include_directories(rocket_engine PUBLIC src)
target_compile_definitions(rocket_engine PUBLIC
  ROCKET_PAGE_BYTES=${ROCKET_PAGE_BYTES}
  ROCKET_CUDA_ARCH=${ROCKET_CUDA_ARCH}
)
target_link_libraries(rocket_engine PUBLIC rocket_fuel)
target_compile_options(rocket_engine PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo>
)
set_target_properties(rocket_engine PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# Stage 2's grouped-GEMM launcher (model.cu::run_moe -> moe_grouped.h) needs
# CUTLASS's block-scaled FP4 tensor core op, which needs the sm_121a variant
# of this SM (CUDA_ARCHITECTURES "121a"), while the rest of rocket_engine
# builds at plain sm_121 above. Kept as its own static library so only this
# one translation unit pays the "a" arch and the CUTLASS include path; see
# tests/CMakeLists.txt's test-loader-swizzle for the same split, and
# moe_grouped.h for why no CUTLASS type crosses into model.cu or kernels.cu.
set(ROCKET_CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass")
if(EXISTS "${ROCKET_CUTLASS_DIR}/include/cutlass/cutlass.h")
  add_library(rocket_moe_grouped STATIC src/moe_grouped.cu)
  target_include_directories(rocket_moe_grouped PUBLIC src)
  target_include_directories(rocket_moe_grouped PRIVATE
    "${ROCKET_CUTLASS_DIR}/include"
    "${ROCKET_CUTLASS_DIR}/tools/util/include"
  )
  target_compile_options(rocket_moe_grouped PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  set_target_properties(rocket_moe_grouped PROPERTIES
    CUDA_ARCHITECTURES "${ROCKET_CUDA_ARCH}a"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )
  target_link_libraries(rocket_engine PUBLIC rocket_moe_grouped)
else()
  message(STATUS
    "rocket_moe_grouped skipped: no CUTLASS at ${ROCKET_CUTLASS_DIR}. "
    "run_moe's grouped-GEMM path (model.cu) will fail to link; fetch CUTLASS with "
    "engines/glm5-moe-nvfp4-2b/bench/fetch-cutlass.sh")
endif()

# Fuel loader: reads the NVFP4 checkpoint off disk and stages it for the
# kernel. Pure C++, no CUDA and no Python, because it runs at serving time.
add_library(rocket_fuel STATIC
  src/safetensors.cc
  src/nvfp4.cc
  src/model_config.cc
  src/tokenizer.cc
)
target_include_directories(rocket_fuel PUBLIC src)
# The layer registry is read from the repository's own fuel description rather
# than compiled in, so the loader needs to know where the repository is.
target_compile_definitions(rocket_fuel PUBLIC
  ROCKET_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../..")
set_target_properties(rocket_fuel PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ROCKET_CUTLASS_DIR is set above, before rocket_engine's own moe_grouped
# target; bench/ and tests/ reuse the same variable.

# Sanity target: the "test stand" of the build itself. Runs one trivial kernel
# and reports what the engine plan has to be compiled against.
add_executable(rocket-test-stand src/test_stand.cu)
target_link_libraries(rocket-test-stand PRIVATE rocket_engine)

# The forward pass: load the fuel, run the step, print tokens and per-stage ms.
add_executable(rocket-decode src/decode_main.cu)
target_link_libraries(rocket-decode PRIVATE rocket_engine rocket_fuel)

# Fabric: the RoCE transport between the two boosters (src/fabric/fabric.h).
# Plain C++ against libibverbs, no CUDA, because nothing in it touches the
# device: the staging buffers are cudaHostAlloc'd by the caller and register
# as ordinary pinned host pages.
find_library(ROCKET_IBVERBS ibverbs)
if(NOT ROCKET_IBVERBS)
  message(FATAL_ERROR
    "libibverbs not found. The two-booster arrangement is the engine's only "
    "arrangement (CMakeLists.txt header); install libibverbs-dev.")
endif()

add_library(rocket_fabric STATIC src/fabric/rdma.cc src/fabric/expert_parallel.cu
                                 src/fabric/expert_balance.cc)
target_include_directories(rocket_fabric PUBLIC src)
target_link_libraries(rocket_fabric PUBLIC ${ROCKET_IBVERBS})
set_target_properties(rocket_fabric PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(rocket_engine PUBLIC rocket_fabric)

# Transport measurement at the message sizes the decode step uses. Launch both
# ranks with scripts/fabric/fabric-microbench.sh.
add_executable(rocket-fabric-bench src/fabric/fabric_bench.cu)
target_link_libraries(rocket-fabric-bench PRIVATE rocket_fabric)

# Static fire: both boosters lit, expert-parallel, gated on token parity
# against the single-booster result. Launch with scripts/fabric/static-fire.sh.
# Registered as a test below rather than in tests/CMakeLists.txt so the target
# and its ctest entry stay in one place with the rest of the fabric.
add_executable(rocket-static-fire src/fabric/static_fire.cu)
target_link_libraries(rocket-static-fire PRIVATE rocket_engine rocket_fuel rocket_fabric)

enable_testing()
add_test(NAME test-stand COMMAND rocket-test-stand)

# Needs the peer booster, so it skips (77) wherever the peer is not reachable,
# matching the other real-artifact tests in tests/CMakeLists.txt.
add_test(NAME static-fire
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/fabric/static-fire.sh")
set_tests_properties(static-fire PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 7200)

add_subdirectory(tests)
add_subdirectory(bench)
