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
)
target_include_directories(rocket_engine PUBLIC src)
target_compile_definitions(rocket_engine PUBLIC
  ROCKET_PAGE_BYTES=${ROCKET_PAGE_BYTES}
  ROCKET_CUDA_ARCH=${ROCKET_CUDA_ARCH}
)
target_compile_options(rocket_engine PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo>
)
set_target_properties(rocket_engine PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# Fuel loader: reads the NVFP4 checkpoint off disk and stages it for the
# kernel. Pure C++, no CUDA and no Python, because it runs at serving time.
add_library(rocket_fuel STATIC
  src/safetensors.cc
  src/nvfp4.cc
)
target_include_directories(rocket_fuel PUBLIC src)
set_target_properties(rocket_fuel PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Pinned CUTLASS checkout, shared by bench/ and tests/.
set(ROCKET_CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass")

# Sanity target: the "test stand" of the build itself. Runs one trivial kernel
# and reports what the engine plan has to be compiled against.
add_executable(rocket-test-stand src/test_stand.cu)
target_link_libraries(rocket-test-stand PRIVATE rocket_engine)

enable_testing()
add_test(NAME test-stand COMMAND rocket-test-stand)

add_subdirectory(tests)
add_subdirectory(bench)
