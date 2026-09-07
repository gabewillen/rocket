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

find_library(ROCKET_QWEN38_NVML nvidia-ml REQUIRED)
find_package(CUDAToolkit 13.0 REQUIRED)
find_path(ROCKET_QWEN38_NCCL_INCLUDE_DIR nccl.h REQUIRED)
find_library(ROCKET_QWEN38_NCCL_LIBRARY nccl REQUIRED)

# CUTLASS 4.8 is pinned by the measured GLM engine. Reuse that checkout for
# the Qwen-only c16 Q/K/V path, while keeping its sm_121a code out of the
# ordinary CUDA targets.
set(ROCKET_QWEN38_CUTLASS_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../glm5-moe-nvfp4-2b/third_party/cutlass"
  CACHE PATH "Pinned CUTLASS checkout")
if(NOT EXISTS "${ROCKET_QWEN38_CUTLASS_DIR}/include/cutlass/cutlass.h")
  message(FATAL_ERROR "CUTLASS 4.8 checkout missing at ${ROCKET_QWEN38_CUTLASS_DIR}")
endif()

add_library(qwen38_cutlass_qkv SHARED src/projection/cutlass_qkv.cu)
target_include_directories(qwen38_cutlass_qkv PRIVATE
  src
  "${ROCKET_QWEN38_CUTLASS_DIR}/include"
  "${ROCKET_QWEN38_CUTLASS_DIR}/tools/util/include"
)
target_compile_options(qwen38_cutlass_qkv PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo>
)
set_target_properties(qwen38_cutlass_qkv PROPERTIES
  CUDA_ARCHITECTURES "121a"
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(qwen38_cutlass_qkv PRIVATE CUDA::cudart)

add_executable(qwen38-qsa-expand-smoke bench/qsa_expand_smoke.cu)
target_include_directories(qwen38-qsa-expand-smoke PRIVATE src)
target_link_libraries(qwen38-qsa-expand-smoke PRIVATE qwen38_cutlass_qkv CUDA::cudart)
set_target_properties(qwen38-qsa-expand-smoke PROPERTIES CUDA_ARCHITECTURES "121")

add_library(qwen38_pair_reduce STATIC
  src/pair_reduce/pair_reduce.cu
  src/pair_reduce/nccl_collective.cc
)
target_include_directories(qwen38_pair_reduce PUBLIC
  src ${ROCKET_QWEN38_NCCL_INCLUDE_DIR})
target_link_libraries(qwen38_pair_reduce PUBLIC
  ${ROCKET_QWEN38_NCCL_LIBRARY} CUDA::cudart)
target_compile_options(qwen38_pair_reduce PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>
)
set_target_properties(qwen38_pair_reduce PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON
)

add_library(qwen38_decode_execution STATIC
  src/decode/execution.cc
  src/decode/full_attention_layer.cc
  src/decode/linear_attention_layer.cc
)
target_include_directories(qwen38_decode_execution PUBLIC src)
target_link_libraries(qwen38_decode_execution PUBLIC qwen38_pair_reduce CUDA::cudart)
target_compile_options(qwen38_decode_execution PRIVATE -Wall -Wextra -Werror)

add_library(qwen38_layer_pair_reduce SHARED src/decode/pair_reduce_c_api.cc)
target_include_directories(qwen38_layer_pair_reduce PUBLIC src)
target_link_libraries(qwen38_layer_pair_reduce PRIVATE qwen38_pair_reduce CUDA::cudart)
target_compile_options(qwen38_layer_pair_reduce PRIVATE -Wall -Wextra -Werror)
set_target_properties(qwen38_layer_pair_reduce PROPERTIES
  LINKER_LANGUAGE CUDA
  CUDA_SEPARABLE_COMPILATION ON
)

add_library(qwen38_residual_rmsnorm STATIC src/norm/residual_rmsnorm.cu)
target_include_directories(qwen38_residual_rmsnorm PUBLIC src)
target_link_libraries(qwen38_residual_rmsnorm PUBLIC CUDA::cudart)
target_compile_options(qwen38_residual_rmsnorm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>
)
set_target_properties(qwen38_residual_rmsnorm PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

add_library(qwen38_gdn_core SHARED src/linear_attention/gdn_core.cu)
target_include_directories(qwen38_gdn_core PUBLIC src)
target_link_libraries(qwen38_gdn_core PUBLIC CUDA::cudart)
target_compile_options(qwen38_gdn_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>
)
set_target_properties(qwen38_gdn_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

add_library(qwen38_gdn_graph SHARED src/linear_attention/gdn_cutlass.cu)
target_include_directories(qwen38_gdn_graph PUBLIC
  src
  "${ROCKET_QWEN38_CUTLASS_DIR}/include"
  "${ROCKET_QWEN38_CUTLASS_DIR}/tools/util/include"
)
target_link_libraries(qwen38_gdn_graph PUBLIC
  qwen38_gdn_core qwen38_decode_execution CUDA::cudart
)
target_compile_options(qwen38_gdn_graph PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo>
)
set_target_properties(qwen38_gdn_graph PROPERTIES
  CUDA_ARCHITECTURES "121a"
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

add_executable(qwen38-gdn-graph-smoke bench/gdn_graph_smoke.cu)
target_include_directories(qwen38-gdn-graph-smoke PRIVATE src)
target_link_libraries(qwen38-gdn-graph-smoke PRIVATE qwen38_gdn_graph CUDA::cudart)
set_target_properties(qwen38-gdn-graph-smoke PROPERTIES CUDA_ARCHITECTURES "121")

add_library(qwen38_hyperconnection SHARED src/hyperconnection/hyperconnection.cu)
target_include_directories(qwen38_hyperconnection PUBLIC src)
target_link_libraries(qwen38_hyperconnection PUBLIC CUDA::cudart CUDA::cublas)
target_compile_options(qwen38_hyperconnection PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>
)
set_target_properties(qwen38_hyperconnection PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

add_library(qwen38_token_output STATIC src/output/token_output.cu)
target_include_directories(qwen38_token_output PUBLIC src)
target_link_libraries(qwen38_token_output PUBLIC CUDA::cudart CUDA::cublas)
target_compile_options(qwen38_token_output PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>
)
set_target_properties(qwen38_token_output PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

add_executable(qwen38-pair-reduce-bench bench/pair_reduce_bench.cu)
target_link_libraries(qwen38-pair-reduce-bench PRIVATE qwen38_pair_reduce ${ROCKET_QWEN38_NVML})

add_executable(qwen38-residual-rmsnorm-bench bench/residual_rmsnorm_bench.cu)
target_link_libraries(qwen38-residual-rmsnorm-bench PRIVATE qwen38_residual_rmsnorm)

add_executable(qwen38-token-output-bench bench/token_output_bench.cu)
target_link_libraries(qwen38-token-output-bench PRIVATE qwen38_token_output)

enable_testing()
add_test(NAME qwen38-qsa-expand-smoke COMMAND qwen38-qsa-expand-smoke)
add_executable(qwen38-test-pair-reduce tests/test_pair_reduce.cu)
target_link_libraries(qwen38-test-pair-reduce PRIVATE qwen38_pair_reduce)
target_compile_options(qwen38-test-pair-reduce PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
)
add_test(NAME qwen38-pair-reduce-contract COMMAND qwen38-test-pair-reduce)
add_executable(qwen38-test-decode-execution tests/test_decode_execution.cc)
target_link_libraries(qwen38-test-decode-execution PRIVATE qwen38_decode_execution)
target_compile_options(qwen38-test-decode-execution PRIVATE -Wall -Wextra -Werror)
set_target_properties(qwen38-test-decode-execution PROPERTIES
  LINKER_LANGUAGE CUDA
  CUDA_SEPARABLE_COMPILATION ON
)
add_test(NAME qwen38-decode-pair-reduce-contract COMMAND qwen38-test-decode-execution)

add_executable(qwen38-test-full-attention-layer tests/test_full_attention_layer.cc)
target_link_libraries(qwen38-test-full-attention-layer PRIVATE qwen38_decode_execution)
target_compile_options(qwen38-test-full-attention-layer PRIVATE -Wall -Wextra -Werror)
set_target_properties(qwen38-test-full-attention-layer PROPERTIES LINKER_LANGUAGE CUDA)
add_test(NAME qwen38-full-attention-layer-contract COMMAND qwen38-test-full-attention-layer)

add_executable(qwen38-test-linear-attention-layer tests/test_linear_attention_layer.cc)
target_link_libraries(qwen38-test-linear-attention-layer PRIVATE qwen38_decode_execution)
target_compile_options(qwen38-test-linear-attention-layer PRIVATE -Wall -Wextra -Werror)
set_target_properties(qwen38-test-linear-attention-layer PROPERTIES LINKER_LANGUAGE CUDA)
add_test(NAME qwen38-linear-attention-layer-contract COMMAND qwen38-test-linear-attention-layer)

add_executable(qwen38-test-hyperconnection tests/test_hyperconnection.cu)
target_link_libraries(qwen38-test-hyperconnection PRIVATE qwen38_hyperconnection CUDA::cudart)
target_compile_options(qwen38-test-hyperconnection PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
)
add_test(NAME qwen38-hyperconnection-contract COMMAND qwen38-test-hyperconnection)

add_executable(qwen38-test-residual-rmsnorm tests/test_residual_rmsnorm.cu)
target_link_libraries(qwen38-test-residual-rmsnorm PRIVATE qwen38_residual_rmsnorm)
target_compile_options(qwen38-test-residual-rmsnorm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
)
add_test(NAME qwen38-residual-rmsnorm-contract COMMAND qwen38-test-residual-rmsnorm)

add_executable(qwen38-test-token-output tests/test_token_output.cu)
target_link_libraries(qwen38-test-token-output PRIVATE qwen38_token_output)
target_compile_options(qwen38-test-token-output PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
)
add_test(NAME qwen38-token-output-contract COMMAND qwen38-test-token-output)
