cmake_minimum_required(VERSION 3.28)

project(rocket_qwen38_flash_next_nvfp4_2b LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(ROCKET_QWEN38_CUDA_ARCH 121 CACHE STRING "Fixed Qwen3.8 GB10 CUDA architecture")
set(CMAKE_CUDA_ARCHITECTURES ${ROCKET_QWEN38_CUDA_ARCH})
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
  message(FATAL_ERROR "Qwen3.8 sm_121 requires CUDA 13.x")
endif()

find_library(ROCKET_QWEN38_IBVERBS ibverbs REQUIRED)

add_library(qwen38_pair_reduce STATIC
  src/pair_reduce/pair_reduce.cu
  src/pair_reduce/rdma.cc
)
target_include_directories(qwen38_pair_reduce PUBLIC src)
target_link_libraries(qwen38_pair_reduce PUBLIC ${ROCKET_QWEN38_IBVERBS})
target_compile_options(qwen38_pair_reduce PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>
)
set_target_properties(qwen38_pair_reduce PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

add_executable(qwen38-pair-reduce-bench bench/pair_reduce_bench.cu)
target_link_libraries(qwen38-pair-reduce-bench PRIVATE qwen38_pair_reduce)

enable_testing()
add_executable(qwen38-test-pair-reduce tests/test_pair_reduce.cu)
target_link_libraries(qwen38-test-pair-reduce PRIVATE qwen38_pair_reduce)
target_compile_options(qwen38-test-pair-reduce PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
)
add_test(NAME qwen38-pair-reduce-contract COMMAND qwen38-test-pair-reduce)
