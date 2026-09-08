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
find_library(ROCKET_QWEN38_NVML nvidia-ml REQUIRED)
find_package(CUDAToolkit 13.0 REQUIRED)

# CUTLASS 4.8 is pinned by the measured GLM engine. Reuse that checkout for
# the Qwen-only c16 Q/K/V path, while keeping its sm_121a code out of the
# ordinary CUDA targets.
set(ROCKET_QWEN38_CUTLASS_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../glm5-moe-nvfp4-2b/third_party/cutlass"
  CACHE PATH "Pinned CUTLASS checkout")
if(NOT EXISTS "${ROCKET_QWEN38_CUTLASS_DIR}/include/cutlass/cutlass.h")
  message(FATAL_ERROR "CUTLASS 4.8 checkout missing at ${ROCKET_QWEN38_CUTLASS_DIR}")
endif()

set(ROCKET_QWEN38_FLASHINFER_91BDA04_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/vendor/flashinfer-91bda04"
  CACHE PATH "Pinned FlashInfer 91bda04 CUTLASS runner slice")
set(ROCKET_QWEN38_FLASHINFER_CUTLASS_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cutlass-b46b16d"
  CACHE PATH "Pinned CUTLASS 4.5 b46b16d dependency closure for FlashInfer")
set(_gdn_flashinfer_template
  "${ROCKET_QWEN38_FLASHINFER_91BDA04_DIR}/include/flashinfer/gemm/fp4_gemm_template_sm120.h")
if(NOT EXISTS "${_gdn_flashinfer_template}")
  message(FATAL_ERROR
    "Pinned FlashInfer 91bda04 GDN CUTLASS runner is required at ${_gdn_flashinfer_template}")
endif()
set(_gdn_flashinfer_cutlass_version
  "${ROCKET_QWEN38_FLASHINFER_CUTLASS_DIR}/include/cutlass/version.h")
if(NOT EXISTS "${_gdn_flashinfer_cutlass_version}")
  message(FATAL_ERROR
    "Pinned CUTLASS 4.5 b46b16d closure is required at ${_gdn_flashinfer_cutlass_version}")
endif()
file(READ "${_gdn_flashinfer_cutlass_version}" _gdn_flashinfer_cutlass_version_text)
if(NOT _gdn_flashinfer_cutlass_version_text MATCHES "#define CUTLASS_MAJOR 4" OR
   NOT _gdn_flashinfer_cutlass_version_text MATCHES "#define CUTLASS_MINOR 5")
  message(FATAL_ERROR "FlashInfer GDN runner requires pinned CUTLASS 4.5 b46b16d")
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
# Only this static archive needs host PIC because qwen38_gdn_graph links it into
# a shared library. Keep the relocation fix local to its owner.
set_target_properties(qwen38_decode_execution PROPERTIES POSITION_INDEPENDENT_CODE ON)

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

add_library(qwen38_gdn_prefill_fused STATIC
  src/linear_attention/gdn_prefill_fused.cu)
target_include_directories(qwen38_gdn_prefill_fused PUBLIC src)
target_link_libraries(qwen38_gdn_prefill_fused PUBLIC CUDA::cudart)
target_compile_options(qwen38_gdn_prefill_fused PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>)
set_target_properties(qwen38_gdn_prefill_fused PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

set(ROCKET_QWEN38_GDN_B12X_AOT_DIR "" CACHE PATH
  "Directory containing the four generated GDN B12X headers and objects")
set(ROCKET_QWEN38_CUTE_DSL_RUNTIME "" CACHE FILEPATH
  "Native libcute_dsl_runtime used by generated GDN B12X objects")
add_library(qwen38_gdn_b12x_aot STATIC
  src/linear_attention/gdn_b12x_aot.cc)
target_include_directories(qwen38_gdn_b12x_aot PUBLIC src)
target_link_libraries(qwen38_gdn_b12x_aot PUBLIC CUDA::cudart)
target_compile_options(qwen38_gdn_b12x_aot PRIVATE -Wall -Wextra -Werror)
set_target_properties(qwen38_gdn_b12x_aot PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(ROCKET_QWEN38_GDN_B12X_AOT_DIR)
  set(_gdn_b12x_objects)
  foreach(_shape t300_qkvz t300_ba t8192_qkvz t8192_ba)
    if(NOT EXISTS "${ROCKET_QWEN38_GDN_B12X_AOT_DIR}/${_shape}.h" OR
       NOT EXISTS "${ROCKET_QWEN38_GDN_B12X_AOT_DIR}/${_shape}.o")
      message(FATAL_ERROR "Incomplete GDN B12X AOT shape ${_shape}")
    endif()
    list(APPEND _gdn_b12x_objects
      "${ROCKET_QWEN38_GDN_B12X_AOT_DIR}/${_shape}.o")
  endforeach()
  if(NOT EXISTS "${ROCKET_QWEN38_CUTE_DSL_RUNTIME}")
    message(FATAL_ERROR "ROCKET_QWEN38_CUTE_DSL_RUNTIME is required for B12X AOT")
  endif()
  set_source_files_properties(${_gdn_b12x_objects} PROPERTIES
    EXTERNAL_OBJECT TRUE GENERATED TRUE)
  target_sources(qwen38_gdn_b12x_aot PRIVATE ${_gdn_b12x_objects})
  target_include_directories(qwen38_gdn_b12x_aot SYSTEM PRIVATE
    "${ROCKET_QWEN38_GDN_B12X_AOT_DIR}")
  target_compile_definitions(qwen38_gdn_b12x_aot PRIVATE
    ROCKET_QWEN38_GDN_B12X_AOT=1)
  target_link_libraries(qwen38_gdn_b12x_aot PRIVATE
    "${ROCKET_QWEN38_CUTE_DSL_RUNTIME}")
else()
  target_compile_definitions(qwen38_gdn_b12x_aot PRIVATE
    ROCKET_QWEN38_GDN_B12X_AOT=0)
endif()

add_library(qwen38_gdn_flashinfer_cutlass STATIC
  src/linear_attention/gdn_flashinfer_cutlass.cu)
target_include_directories(qwen38_gdn_flashinfer_cutlass PUBLIC src PRIVATE
  "${ROCKET_QWEN38_FLASHINFER_CUTLASS_DIR}/include"
  "${ROCKET_QWEN38_FLASHINFER_CUTLASS_DIR}/tools/util/include"
  "${ROCKET_QWEN38_FLASHINFER_91BDA04_DIR}/include")
target_link_libraries(qwen38_gdn_flashinfer_cutlass PUBLIC CUDA::cudart)
target_compile_options(qwen38_gdn_flashinfer_cutlass PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-use_fast_math;-Xfatbin=-compress-all;--compress-mode=size;-static-global-template-stub=false;-lineinfo>)
target_compile_definitions(qwen38_gdn_flashinfer_cutlass PRIVATE
  ENABLE_BF16=1
  ENABLE_FP4=1
  FLASHINFER_ENABLE_BF16=1
  FLASHINFER_ENABLE_F16=1
  FLASHINFER_ENABLE_FP4_E2M1=1
  FLASHINFER_ENABLE_FP8_E8M0=1
  CUTLASS_ENABLE_GDC_FOR_SM100=1)
set_target_properties(qwen38_gdn_flashinfer_cutlass PROPERTIES
  # Match the prebuilt FlashInfer 0.6.17 wheel artifact loaded by the
  # standalone vLLM comparator. Its SM12x CUTLASS module contains sm_120f
  # family code even on GB10 (SM121).
  CUDA_ARCHITECTURES "120f"
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON)

add_library(qwen38_gdn_graph SHARED
  src/linear_attention/gdn_cutlass.cu
  src/linear_attention/gdn_flashinfer_wheel.cc
  src/linear_attention/gdn_verifier.cu)
target_include_directories(qwen38_gdn_graph PUBLIC
  src
  "${ROCKET_QWEN38_CUTLASS_DIR}/include"
  "${ROCKET_QWEN38_CUTLASS_DIR}/tools/util/include"
)
target_link_libraries(qwen38_gdn_graph PUBLIC
  qwen38_gdn_core qwen38_gdn_b12x_aot qwen38_gdn_flashinfer_cutlass
  qwen38_decode_execution CUDA::cudart ${CMAKE_DL_LIBS}
)
target_compile_options(qwen38_gdn_graph PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
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

add_executable(qwen38-gdn-prefill-fused-bench
  bench/gdn_prefill_fused_bench.cu)
target_include_directories(qwen38-gdn-prefill-fused-bench PRIVATE src)
target_link_libraries(qwen38-gdn-prefill-fused-bench PRIVATE
  qwen38_gdn_prefill_fused CUDA::cudart)
target_compile_options(qwen38-gdn-prefill-fused-bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>)

add_library(qwen38_decoder_verifier STATIC src/decode/decoder_verifier.cc)
target_include_directories(qwen38_decoder_verifier PUBLIC src)
target_link_libraries(qwen38_decoder_verifier PUBLIC
  qwen38_gdn_graph qwen38_decode_execution CUDA::cudart
)
target_compile_options(qwen38_decoder_verifier PRIVATE -Wall -Wextra -Werror)
set_target_properties(qwen38_decoder_verifier PROPERTIES
  LINKER_LANGUAGE CUDA
  POSITION_INDEPENDENT_CODE ON
)

add_library(qwen38_decoder_verifier_c_api SHARED
  src/decode/decoder_verifier_c_api.cc
)
target_include_directories(qwen38_decoder_verifier_c_api PUBLIC src)
target_link_libraries(qwen38_decoder_verifier_c_api PRIVATE
  qwen38_decoder_verifier CUDA::cudart
)
target_compile_options(qwen38_decoder_verifier_c_api PRIVATE -Wall -Wextra -Werror)
set_target_properties(qwen38_decoder_verifier_c_api PROPERTIES LINKER_LANGUAGE CUDA)

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

add_library(qwen38_mtp_executor STATIC
  src/mtp/graph_runtime.cu
  src/mtp/input_fusion.cu
  src/mtp/nccl_winner_exchange.cu
  src/mtp/native_executor.cu
  src/mtp/state_arena.cu
)
target_include_directories(qwen38_mtp_executor PUBLIC src)
target_link_libraries(qwen38_mtp_executor PUBLIC
  qwen38_hyperconnection qwen38_token_output CUDA::cudart CUDA::cublas
  ${CMAKE_DL_LIBS})
target_compile_options(qwen38_mtp_executor PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>
)

add_library(qwen38_mtp_decoder_step SHARED
  src/mtp/decoder_step.cc
  src/mtp/decoder_step_c_api.cc
)
target_include_directories(qwen38_mtp_decoder_step PUBLIC src)
target_link_libraries(qwen38_mtp_decoder_step PRIVATE
  qwen38_mtp_executor qwen38_decoder_verifier CUDA::cudart)
target_compile_options(qwen38_mtp_decoder_step PRIVATE -Wall -Wextra -Werror)
set_target_properties(qwen38_mtp_decoder_step PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(qwen38_mtp_executor PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

add_executable(qwen38-pair-reduce-bench bench/pair_reduce_bench.cu)
target_link_libraries(qwen38-pair-reduce-bench PRIVATE qwen38_pair_reduce ${ROCKET_QWEN38_NVML})

add_executable(qwen38-decode-pair-reduce-smoke bench/decode_execution_smoke.cu)
target_link_libraries(qwen38-decode-pair-reduce-smoke PRIVATE qwen38_decode_execution)

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
add_test(NAME qwen38-pair-reduce-reject-timeout
  COMMAND qwen38-pair-reduce-bench --rank 0 --timeout-ms 99)
set_tests_properties(qwen38-pair-reduce-reject-timeout PROPERTIES WILL_FAIL TRUE)
add_test(NAME qwen38-pair-reduce-reject-fault-budget
  COMMAND qwen38-pair-reduce-bench --rank 0 --timeout-ms 1000
          --fault-stall-rank 1 --fault-stall-ms 1000)
set_tests_properties(qwen38-pair-reduce-reject-fault-budget PROPERTIES WILL_FAIL TRUE)

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

add_executable(qwen38-test-decoder-verifier tests/test_decoder_verifier.cc)
target_link_libraries(qwen38-test-decoder-verifier PRIVATE
  qwen38_decoder_verifier qwen38_decoder_verifier_c_api
)
target_compile_options(qwen38-test-decoder-verifier PRIVATE -Wall -Wextra -Werror)
set_target_properties(qwen38-test-decoder-verifier PROPERTIES LINKER_LANGUAGE CUDA)
add_test(NAME qwen38-decoder-verifier-contract COMMAND qwen38-test-decoder-verifier)

add_executable(qwen38-test-gdn-verifier tests/test_gdn_verifier_contract.cc)
target_include_directories(qwen38-test-gdn-verifier PRIVATE src)
target_link_libraries(qwen38-test-gdn-verifier PRIVATE CUDA::cudart)
target_compile_options(qwen38-test-gdn-verifier PRIVATE -Wall -Wextra -Werror)
add_test(NAME qwen38-gdn-verifier-contract COMMAND qwen38-test-gdn-verifier)

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

add_executable(qwen38-test-native-mtp-executor tests/test_native_mtp_executor.cu)
target_link_libraries(qwen38-test-native-mtp-executor PRIVATE
  qwen38_mtp_executor CUDA::cudart)
target_compile_options(qwen38-test-native-mtp-executor PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
)
add_test(NAME qwen38-native-mtp-executor-contract
  COMMAND qwen38-test-native-mtp-executor)

add_executable(qwen38-test-qsa-mtp-state-view
  tests/test_qsa_mtp_state_view.cc)
target_include_directories(qwen38-test-qsa-mtp-state-view PRIVATE src)
target_link_libraries(qwen38-test-qsa-mtp-state-view PRIVATE CUDA::cudart)
target_compile_options(qwen38-test-qsa-mtp-state-view PRIVATE
  -Wall -Wextra -Werror)
add_test(NAME qwen38-qsa-mtp-state-view-contract
  COMMAND qwen38-test-qsa-mtp-state-view)

add_executable(qwen38-test-mtp-state-arena tests/test_mtp_state_arena.cu)
target_link_libraries(qwen38-test-mtp-state-arena PRIVATE qwen38_mtp_executor CUDA::cudart)
add_test(NAME qwen38-mtp-state-arena-contract COMMAND qwen38-test-mtp-state-arena)

add_executable(qwen38-test-mtp-input-fusion tests/test_mtp_input_fusion.cu)
target_link_libraries(qwen38-test-mtp-input-fusion PRIVATE
  qwen38_mtp_executor CUDA::cudart)
target_compile_options(qwen38-test-mtp-input-fusion PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
)
add_test(NAME qwen38-mtp-input-fusion-contract
  COMMAND qwen38-test-mtp-input-fusion)

add_executable(qwen38-test-mtp-graph-runtime-native
  tests/test_mtp_graph_runtime_native.cu)
target_link_libraries(qwen38-test-mtp-graph-runtime-native PRIVATE
  qwen38_mtp_executor CUDA::cudart)
target_compile_options(qwen38-test-mtp-graph-runtime-native PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
)
add_test(NAME qwen38-mtp-graph-runtime-native-contract
  COMMAND qwen38-test-mtp-graph-runtime-native)

add_executable(qwen38-test-gdn-prefill-fused tests/test_gdn_prefill_fused.cu)
target_link_libraries(qwen38-test-gdn-prefill-fused PRIVATE
  qwen38_gdn_prefill_fused CUDA::cudart)
target_compile_options(qwen38-test-gdn-prefill-fused PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>)
add_test(NAME qwen38-gdn-prefill-fused-contract
  COMMAND qwen38-test-gdn-prefill-fused)

add_executable(qwen38-test-gdn-prefill-width4
  tests/test_gdn_prefill_width4.cc)
target_include_directories(qwen38-test-gdn-prefill-width4 PRIVATE src)
target_compile_options(qwen38-test-gdn-prefill-width4 PRIVATE
  -Wall -Wextra -Werror)
add_test(NAME qwen38-gdn-prefill-width4-cpu-contract
  COMMAND qwen38-test-gdn-prefill-width4)

add_executable(qwen38-test-gdn-prefill-projection-contract
  tests/test_gdn_prefill_projection_contract.cc)
target_include_directories(qwen38-test-gdn-prefill-projection-contract PRIVATE
  src)
target_link_libraries(qwen38-test-gdn-prefill-projection-contract PRIVATE
  CUDA::cudart)
target_compile_options(qwen38-test-gdn-prefill-projection-contract PRIVATE
  -Wall -Wextra -Werror)
add_test(NAME qwen38-gdn-prefill-projection-cpu-contract
  COMMAND qwen38-test-gdn-prefill-projection-contract)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME qwen38-gdn-b12x-aot-cpu-contract
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gdn_b12x_aot_contract.py)

add_test(NAME qwen38-gdn-flashinfer-cutlass-cpu-contract
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gdn_flashinfer_cutlass_contract.py)

add_executable(qwen38-test-gdn-b12x-aot-contract
  tests/test_gdn_b12x_aot_contract.cc)
target_link_libraries(qwen38-test-gdn-b12x-aot-contract PRIVATE
  qwen38_gdn_b12x_aot CUDA::cudart)
target_compile_options(qwen38-test-gdn-b12x-aot-contract PRIVATE
  -Wall -Wextra -Werror)
add_test(NAME qwen38-gdn-b12x-aot-native-contract
  COMMAND qwen38-test-gdn-b12x-aot-contract)
