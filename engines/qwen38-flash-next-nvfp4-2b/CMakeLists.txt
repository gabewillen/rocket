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
find_library(ROCKET_QWEN38_CRYPTO NAMES crypto libcrypto.so.3 REQUIRED)
find_library(ROCKET_QWEN38_JSONC NAMES libjson-c.so.5 REQUIRED)
if(NOT ROCKET_QWEN38_JSONC MATCHES "libjson-c\\.so\\.5$")
  message(FATAL_ERROR "Qwen native plan parser requires json-c ABI 5")
endif()
file(SHA256 "${ROCKET_QWEN38_JSONC}" _rocket_qwen38_jsonc_sha256)
if(NOT _rocket_qwen38_jsonc_sha256 STREQUAL
   "773990495f98a200f449dc915a14709007ce4b09f41d08c8e46c8d04082df35b")
  message(FATAL_ERROR "Qwen json-c runtime identity changed")
endif()
add_library(rocket_qwen38_jsonc SHARED IMPORTED GLOBAL)
set_target_properties(rocket_qwen38_jsonc PROPERTIES
  IMPORTED_LOCATION "${ROCKET_QWEN38_JSONC}")
if(NOT ROCKET_QWEN38_CRYPTO MATCHES "libcrypto\\.so\\.3$")
  message(FATAL_ERROR "Qwen target MoE materializer requires OpenSSL 3 libcrypto ABI")
endif()
file(SHA256 "${ROCKET_QWEN38_CRYPTO}" _rocket_qwen38_crypto_sha256)
if(NOT _rocket_qwen38_crypto_sha256 STREQUAL
   "60076983c2ea0d6b73cc64fb39cafa0a0f90f220ef000f6effb4aaac47500949")
  message(FATAL_ERROR "Qwen OpenSSL 3 runtime identity changed")
endif()
add_library(rocket_qwen38_openssl3_crypto SHARED IMPORTED GLOBAL)
set_target_properties(rocket_qwen38_openssl3_crypto PROPERTIES
  IMPORTED_LOCATION "${ROCKET_QWEN38_CRYPTO}")
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

add_library(qwen38_qsa_target_preprocess STATIC
  src/attention/qsa_target_preprocess.cu)
target_include_directories(qwen38_qsa_target_preprocess PUBLIC src)
target_link_libraries(qwen38_qsa_target_preprocess PUBLIC CUDA::cudart)
target_compile_options(qwen38_qsa_target_preprocess PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>)
set_target_properties(qwen38_qsa_target_preprocess PROPERTIES
  CUDA_ARCHITECTURES "121"
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON)

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
  src/decode/target_full_layer.cc
  src/decode/target_layer3_prefill.cc
  src/decode/target_layer3_pair_reduce.cc
  src/decode/target_k0_pair_reduce.cc
  src/decode/target_k0_executor.cc
  src/decode/linear_attention_layer.cc
  src/decode/target_gdn_layer.cc
)
target_include_directories(qwen38_decode_execution PUBLIC src)
target_link_libraries(qwen38_decode_execution PUBLIC qwen38_pair_reduce CUDA::cudart)
target_compile_options(qwen38_decode_execution PRIVATE -Wall -Wextra -Werror)
# Only this static archive needs host PIC because qwen38_gdn_graph links it into
# a shared library. Keep the relocation fix local to its owner.
set_target_properties(qwen38_decode_execution PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(qwen38_target_k0_layer_owner_inventory STATIC
  src/decode/target_k0_layer_owner_inventory.cc)
target_include_directories(qwen38_target_k0_layer_owner_inventory PUBLIC src)
target_link_libraries(qwen38_target_k0_layer_owner_inventory PUBLIC
  qwen38_decode_execution)
target_compile_options(qwen38_target_k0_layer_owner_inventory PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38_target_k0_layer_owner_inventory PROPERTIES
  POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE CUDA)

add_library(qwen38_target_k0_startup STATIC
  src/decode/target_k0_startup.cc
  src/decode/target_k0_native_plan_inventory.cc)
target_include_directories(qwen38_target_k0_startup PUBLIC src)
target_link_libraries(qwen38_target_k0_startup PUBLIC
  qwen38_target_k0_layer_owner_inventory
  qwen38_target_k0_oracle_comparator qwen38_native_token_io)
target_compile_options(qwen38_target_k0_startup PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38_target_k0_startup PROPERTIES
  POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE CUDA)

add_executable(qwen38-target-k0-startup
  src/decode/target_k0_startup_cli.cc)
target_link_libraries(qwen38-target-k0-startup PRIVATE
  qwen38_target_k0_startup)
target_compile_options(qwen38-target-k0-startup PRIVATE
  -Wall -Wextra -Werror)

add_executable(qwen38-test-target-k0-startup
  tests/test_target_k0_startup.cc)
target_link_libraries(qwen38-test-target-k0-startup PRIVATE
  qwen38_target_k0_startup)
target_compile_options(qwen38-test-target-k0-startup PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38-test-target-k0-startup PROPERTIES
  LINKER_LANGUAGE CUDA)

add_executable(qwen38-test-target-k0-native-plan-inventory
  tests/test_target_k0_native_plan_inventory.cc)
target_link_libraries(qwen38-test-target-k0-native-plan-inventory PRIVATE
  qwen38_target_k0_startup)
target_compile_options(qwen38-test-target-k0-native-plan-inventory PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38-test-target-k0-native-plan-inventory PROPERTIES
  LINKER_LANGUAGE CUDA)

add_library(qwen38_target_layer3_native_plan STATIC
  src/decode/target_layer3_native_plan.cc)
target_include_directories(qwen38_target_layer3_native_plan PUBLIC src)
target_link_libraries(qwen38_target_layer3_native_plan PUBLIC
  rocket_qwen38_openssl3_crypto rocket_qwen38_jsonc)
target_compile_options(qwen38_target_layer3_native_plan PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38_target_layer3_native_plan PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

add_library(qwen38_target_layer_native_plan STATIC
  src/decode/target_layer_native_plan.cc)
target_include_directories(qwen38_target_layer_native_plan PUBLIC src)
target_link_libraries(qwen38_target_layer_native_plan PUBLIC
  rocket_qwen38_openssl3_crypto rocket_qwen38_jsonc qwen38_decode_execution)
target_compile_options(qwen38_target_layer_native_plan PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38_target_layer_native_plan PROPERTIES
  POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE CUDA)
target_link_libraries(qwen38_target_k0_startup PUBLIC
  qwen38_target_layer_native_plan)

add_library(qwen38_target_layer_native_bindings STATIC
  src/decode/target_layer_native_bindings.cc)
target_include_directories(qwen38_target_layer_native_bindings PUBLIC src)
target_link_libraries(qwen38_target_layer_native_bindings PUBLIC
  qwen38_target_layer_native_plan qwen38_target_moe_n640_device_stage
  qwen38_target_slab_owner)
target_compile_options(qwen38_target_layer_native_bindings PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38_target_layer_native_bindings PROPERTIES
  POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE CUDA)

add_library(qwen38_target_layer3_generation STATIC
  src/decode/target_layer3_generation.cu)
target_include_directories(qwen38_target_layer3_generation PUBLIC src)
target_link_libraries(qwen38_target_layer3_generation PUBLIC
  qwen38_decode_execution CUDA::cudart)
target_compile_options(qwen38_target_layer3_generation PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>)
set_target_properties(qwen38_target_layer3_generation PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

add_library(qwen38_target_layer3_oracle_comparator STATIC
  src/decode/target_layer3_oracle_comparator.cc)
target_include_directories(qwen38_target_layer3_oracle_comparator PUBLIC src)
target_link_libraries(qwen38_target_layer3_oracle_comparator PUBLIC
  qwen38_decode_execution CUDA::cudart rocket_qwen38_openssl3_crypto)
target_compile_options(qwen38_target_layer3_oracle_comparator PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38_target_layer3_oracle_comparator PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

add_library(qwen38_target_k0_oracle_comparator STATIC
  src/decode/target_k0_oracle_comparator.cc)
target_include_directories(qwen38_target_k0_oracle_comparator PUBLIC src)
target_link_libraries(qwen38_target_k0_oracle_comparator PUBLIC
  qwen38_decode_execution qwen38_target_layer3_oracle_comparator
  rocket_qwen38_openssl3_crypto rocket_qwen38_jsonc CUDA::cudart)
target_compile_options(qwen38_target_k0_oracle_comparator PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38_target_k0_oracle_comparator PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

add_executable(qwen38-target-k0-oracle-auth
  tests/target_k0_oracle_auth.cc)
target_link_libraries(qwen38-target-k0-oracle-auth PRIVATE
  qwen38_target_k0_oracle_comparator)
target_compile_options(qwen38-target-k0-oracle-auth PRIVATE
  -Wall -Wextra -Werror)

add_executable(qwen38-target-layer3-oracle-auth
  tests/target_layer3_oracle_auth.cc)
target_link_libraries(qwen38-target-layer3-oracle-auth PRIVATE
  qwen38_target_layer3_oracle_comparator)
target_compile_options(qwen38-target-layer3-oracle-auth PRIVATE
  -Wall -Wextra -Werror)

add_library(qwen38_target_moe_n640_materializer SHARED
  src/moe/target_moe_n640_materializer.cu)
target_include_directories(qwen38_target_moe_n640_materializer PUBLIC src)
target_link_libraries(qwen38_target_moe_n640_materializer PUBLIC
  CUDA::cudart rocket_qwen38_openssl3_crypto)
target_compile_options(qwen38_target_moe_n640_materializer PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>)
set_target_properties(qwen38_target_moe_n640_materializer PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

add_library(qwen38_qsa_sidecar_owner STATIC
  src/attention/qsa_sidecar_owner.cc)
target_include_directories(qwen38_qsa_sidecar_owner PUBLIC src)
target_link_libraries(qwen38_qsa_sidecar_owner PUBLIC
  CUDA::cudart rocket_qwen38_openssl3_crypto)
target_compile_options(qwen38_qsa_sidecar_owner PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38_qsa_sidecar_owner PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

add_library(qwen38_layer3_rope_owner STATIC
  src/attention/layer3_rope_owner.cc)
target_include_directories(qwen38_layer3_rope_owner PUBLIC src)
target_link_libraries(qwen38_layer3_rope_owner PUBLIC CUDA::cudart)
target_compile_options(qwen38_layer3_rope_owner PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38_layer3_rope_owner PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

# The all-layer binder owns both GDN and QSA bindings. Keep its QSA identity
# definitions available even when the final consumer is a GDN-only owner.
target_link_libraries(qwen38_target_layer_native_bindings PUBLIC
  qwen38_qsa_sidecar_owner qwen38_layer3_rope_owner)

add_executable(qwen38-qsa-sidecar-auth tests/qsa_sidecar_auth.cc)
target_link_libraries(qwen38-qsa-sidecar-auth PRIVATE
  qwen38_qsa_sidecar_owner)
target_compile_options(qwen38-qsa-sidecar-auth PRIVATE
  -Wall -Wextra -Werror)

add_executable(qwen38-test-layer3-rope-owner
  tests/test_layer3_rope_owner.cc)
target_link_libraries(qwen38-test-layer3-rope-owner PRIVATE
  qwen38_layer3_rope_owner)
target_compile_options(qwen38-test-layer3-rope-owner PRIVATE
  -Wall -Wextra -Werror)

add_executable(qwen38-test-target-qsa-shared-assets
  tests/test_target_qsa_shared_assets.cc)
target_link_libraries(qwen38-test-target-qsa-shared-assets PRIVATE
  qwen38_qsa_sidecar_owner qwen38_layer3_rope_owner)
target_compile_options(qwen38-test-target-qsa-shared-assets PRIVATE
  -Wall -Wextra -Werror)
add_test(NAME qwen38-target-qsa-shared-assets-cpu-contract
  COMMAND qwen38-test-target-qsa-shared-assets)

add_library(qwen38_target_k0_qsa_state_owner STATIC
  src/attention/target_k0_qsa_state_owner.cu)
target_include_directories(qwen38_target_k0_qsa_state_owner PUBLIC src)
target_link_libraries(qwen38_target_k0_qsa_state_owner PUBLIC CUDA::cudart)
target_compile_options(qwen38_target_k0_qsa_state_owner PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>)
set_target_properties(qwen38_target_k0_qsa_state_owner PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

add_executable(qwen38-test-target-k0-qsa-state-owner
  tests/test_target_k0_qsa_state_owner.cc)
target_link_libraries(qwen38-test-target-k0-qsa-state-owner PRIVATE
  qwen38_target_k0_qsa_state_owner)
target_compile_options(qwen38-test-target-k0-qsa-state-owner PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38-test-target-k0-qsa-state-owner PROPERTIES
  LINKER_LANGUAGE CUDA)
add_test(NAME qwen38-target-k0-qsa-state-owner-cpu-contract
  COMMAND qwen38-test-target-k0-qsa-state-owner)

add_library(qwen38_target_slab_owner SHARED src/model/target_slab_owner.cc)
target_include_directories(qwen38_target_slab_owner PUBLIC src)
target_link_libraries(qwen38_target_slab_owner PUBLIC
  CUDA::cudart CUDA::cuda_driver rocket_qwen38_openssl3_crypto)
target_compile_options(qwen38_target_slab_owner PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38_target_slab_owner PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

add_executable(qwen38-target-slab-auth tests/target_slab_auth.cc)
target_link_libraries(qwen38-target-slab-auth PRIVATE qwen38_target_slab_owner)

add_executable(qwen38-test-target-slab-process-lease
  tests/test_target_slab_process_lease.cc)
target_link_libraries(qwen38-test-target-slab-process-lease PRIVATE
  qwen38_target_slab_owner)
target_compile_options(qwen38-test-target-slab-process-lease PRIVATE
  -Wall -Wextra -Werror)
add_test(NAME qwen38-target-slab-process-lease-cpu-contract
  COMMAND qwen38-test-target-slab-process-lease)
target_compile_options(qwen38-target-slab-auth PRIVATE
  -Wall -Wextra -Werror)

add_executable(qwen38-target-slab-load bench/target_slab_load.cc)
target_link_libraries(qwen38-target-slab-load PRIVATE qwen38_target_slab_owner)
target_compile_options(qwen38-target-slab-load PRIVATE
  -Wall -Wextra -Werror)

add_library(qwen38_target_moe_n640_device_stage STATIC
  src/moe/target_moe_n640_device_stage.cu)
target_include_directories(qwen38_target_moe_n640_device_stage PUBLIC src)
target_link_libraries(qwen38_target_moe_n640_device_stage PUBLIC CUDA::cudart)
target_compile_options(qwen38_target_moe_n640_device_stage PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>)
set_target_properties(qwen38_target_moe_n640_device_stage PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

add_library(qwen38_target_layer3_native_bindings STATIC
  src/decode/target_layer3_native_bindings.cc)
target_include_directories(qwen38_target_layer3_native_bindings PUBLIC src)
target_link_libraries(qwen38_target_layer3_native_bindings PUBLIC
  qwen38_target_layer3_native_plan qwen38_target_slab_owner
  qwen38_qsa_sidecar_owner qwen38_layer3_rope_owner
  qwen38_target_moe_n640_device_stage qwen38_decode_execution CUDA::cudart)
target_compile_options(qwen38_target_layer3_native_bindings PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38_target_layer3_native_bindings PROPERTIES
  POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE CUDA)

add_library(qwen38_native_hyperconnection_adapter STATIC
  src/hyperconnection/native_adapter.cc)
target_include_directories(qwen38_native_hyperconnection_adapter PUBLIC src)
target_link_libraries(qwen38_native_hyperconnection_adapter PUBLIC
  qwen38_decode_execution qwen38_hyperconnection CUDA::cudart)
target_compile_options(qwen38_native_hyperconnection_adapter PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38_native_hyperconnection_adapter PROPERTIES
  POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE CUDA)

add_library(qwen38_native_qsa_graph STATIC
  src/attention/native_qsa_graph.cc)
target_include_directories(qwen38_native_qsa_graph PUBLIC src)
target_link_libraries(qwen38_native_qsa_graph PUBLIC
  qwen38_decode_execution qwen38_qsa_target_preprocess qwen38_cutlass_qkv
  CUDA::cudart)
target_compile_options(qwen38_native_qsa_graph PRIVATE -Wall -Wextra -Werror)
set_target_properties(qwen38_native_qsa_graph PROPERTIES
  POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE CUDA)

add_library(qwen38_native_qsa_c_api SHARED
  src/attention/native_qsa_graph.cc)
target_include_directories(qwen38_native_qsa_c_api PUBLIC src)
target_link_libraries(qwen38_native_qsa_c_api PUBLIC
  qwen38_decode_execution qwen38_qsa_target_preprocess qwen38_cutlass_qkv
  CUDA::cudart)
target_compile_options(qwen38_native_qsa_c_api PRIVATE -Wall -Wextra -Werror)
set_target_properties(qwen38_native_qsa_c_api PROPERTIES LINKER_LANGUAGE CUDA)

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

add_library(qwen38_native_token_io STATIC
  src/output/native_token_io.cu
  src/output/token_io_roots.cc)
target_include_directories(qwen38_native_token_io PUBLIC src)
target_link_libraries(qwen38_native_token_io PUBLIC
  qwen38_token_output qwen38_hyperconnection qwen38_target_slab_owner
  qwen38_pair_reduce rocket_qwen38_openssl3_crypto CUDA::cudart
  CUDA::cuda_driver CUDA::cublas)
target_compile_options(qwen38_native_token_io PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>)
set_target_properties(qwen38_native_token_io PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

add_library(qwen38_native_token_io_preflight SHARED
  src/output/native_token_io_preflight.cc)
target_include_directories(qwen38_native_token_io_preflight PUBLIC src)
target_link_libraries(qwen38_native_token_io_preflight PRIVATE
  qwen38_native_token_io CUDA::cudart)
target_compile_options(qwen38_native_token_io_preflight PRIVATE
  -Wall -Wextra -Werror)
target_link_options(qwen38_native_token_io_preflight PRIVATE
  -Wl,--exclude-libs,ALL)
set_target_properties(qwen38_native_token_io_preflight PROPERTIES
  POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE CUDA
  CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

add_library(qwen38_route_compaction_contract STATIC
  src/moe/route_compaction.cc
  src/moe/fp8_routed_experts.cc)
target_include_directories(qwen38_route_compaction_contract PUBLIC src)
target_link_libraries(qwen38_route_compaction_contract PUBLIC CUDA::cudart)
target_compile_options(qwen38_route_compaction_contract PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38_route_compaction_contract PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

add_library(qwen38_route_compaction STATIC
  src/moe/route_compaction.cu
  src/moe/fp8_routed_experts.cu)
set(QWEN38_MTP_FP8_TRITON_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/vendor/mtp-fp8-triton-sm121")
foreach(_kernel IN ITEMS
    manifest.json
    quantize_hidden.cubin
    silu_mul_quantize.cubin
    rank0/gate_up_silu.cubin
    rank0/down_weighted_reduce.cubin
    rank1/gate_up_silu.cubin
    rank1/down_weighted_reduce.cubin)
  if(NOT EXISTS "${QWEN38_MTP_FP8_TRITON_DIR}/${_kernel}")
    message(FATAL_ERROR "missing AOT Triton kernel: ${_kernel}")
  endif()
endforeach()
target_include_directories(qwen38_route_compaction PUBLIC src)
target_link_libraries(qwen38_route_compaction PUBLIC
  qwen38_route_compaction_contract CUDA::cudart CUDA::cuda_driver)
target_compile_definitions(qwen38_route_compaction PRIVATE
  ROCKET_QWEN38_MTP_FP8_TRITON_DIR="${QWEN38_MTP_FP8_TRITON_DIR}")
target_compile_options(qwen38_route_compaction PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>)
set_target_properties(qwen38_route_compaction PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON)

add_library(qwen38_target_moe_c1_contract STATIC src/moe/target_moe_c1.cc)
target_include_directories(qwen38_target_moe_c1_contract PUBLIC src)
target_link_libraries(qwen38_target_moe_c1_contract PUBLIC CUDA::cudart)
target_compile_options(qwen38_target_moe_c1_contract PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38_target_moe_c1_contract PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

add_library(qwen38_target_moe_c1 STATIC src/moe/target_moe_c1.cu)
target_include_directories(qwen38_target_moe_c1 PUBLIC src)
target_link_libraries(qwen38_target_moe_c1 PUBLIC
  qwen38_target_moe_c1_contract CUDA::cudart)
target_compile_options(qwen38_target_moe_c1 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>)
set_target_properties(qwen38_target_moe_c1 PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON)

set(ROCKET_QWEN38_TARGET_MOE_B12X_AOT_DIR "" CACHE PATH
  "Directory containing the generated fixed-c1 target MoE header and object")
add_library(qwen38_target_moe_b12x_aot STATIC
  src/moe/target_moe_b12x_aot.cc)
target_include_directories(qwen38_target_moe_b12x_aot PUBLIC src)
target_link_libraries(qwen38_target_moe_b12x_aot PUBLIC
  qwen38_target_moe_c1_contract CUDA::cudart)
target_compile_options(qwen38_target_moe_b12x_aot PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38_target_moe_b12x_aot PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
if(ROCKET_QWEN38_TARGET_MOE_B12X_AOT_DIR)
  if(NOT EXISTS "${ROCKET_QWEN38_TARGET_MOE_B12X_AOT_DIR}/target_moe_b12x_c1.h" OR
     NOT EXISTS "${ROCKET_QWEN38_TARGET_MOE_B12X_AOT_DIR}/target_moe_b12x_c1.o" OR
     NOT EXISTS "${ROCKET_QWEN38_TARGET_MOE_B12X_AOT_DIR}/target_moe_artifact_key.h" OR
     NOT EXISTS "${ROCKET_QWEN38_TARGET_MOE_B12X_AOT_DIR}/target_moe_compact_config.h")
    message(FATAL_ERROR "Incomplete target MoE B12X AOT fixed-c1 export")
  endif()
  if(NOT EXISTS "${ROCKET_QWEN38_CUTE_DSL_RUNTIME}")
    message(FATAL_ERROR "ROCKET_QWEN38_CUTE_DSL_RUNTIME is required for target MoE AOT")
  endif()
  set(_target_moe_b12x_object
    "${ROCKET_QWEN38_TARGET_MOE_B12X_AOT_DIR}/target_moe_b12x_c1.o")
  set_source_files_properties(${_target_moe_b12x_object} PROPERTIES
    EXTERNAL_OBJECT TRUE GENERATED TRUE)
  target_sources(qwen38_target_moe_b12x_aot PRIVATE
    ${_target_moe_b12x_object})
  target_include_directories(qwen38_target_moe_b12x_aot SYSTEM PRIVATE
    "${ROCKET_QWEN38_TARGET_MOE_B12X_AOT_DIR}")
  target_compile_definitions(qwen38_target_moe_b12x_aot PRIVATE
    ROCKET_QWEN38_TARGET_MOE_B12X_AOT=1)
  target_link_libraries(qwen38_target_moe_b12x_aot PRIVATE
    "${ROCKET_QWEN38_CUTE_DSL_RUNTIME}")

  add_library(qwen38_target_router_shared_c1 SHARED
    src/moe/target_router_shared_c1.cu)
  target_include_directories(qwen38_target_router_shared_c1 PUBLIC src PRIVATE
    "${ROCKET_QWEN38_TARGET_MOE_B12X_AOT_DIR}")
  target_link_libraries(qwen38_target_router_shared_c1 PUBLIC
    qwen38_target_moe_b12x_aot CUDA::cudart)
  target_compile_options(qwen38_target_router_shared_c1 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>)
  set_target_properties(qwen38_target_router_shared_c1 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

  add_library(qwen38_target_full_moe_c1 SHARED
    src/moe/target_full_moe_c1.cc)
  target_include_directories(qwen38_target_full_moe_c1 PUBLIC src)
  target_link_libraries(qwen38_target_full_moe_c1 PUBLIC CUDA::cudart PRIVATE
    qwen38_target_router_shared_c1 qwen38_target_moe_c1
    qwen38_target_moe_b12x_aot qwen38_target_moe_n640_device_stage)
  target_compile_options(qwen38_target_full_moe_c1 PRIVATE
    -Wall -Wextra -Werror)
  set_target_properties(qwen38_target_full_moe_c1 PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

  add_library(qwen38_native_target_moe_graph STATIC
    src/moe/native_target_moe_graph.cc)
  target_include_directories(qwen38_native_target_moe_graph PUBLIC src)
  target_link_libraries(qwen38_native_target_moe_graph PUBLIC
    qwen38_decode_execution qwen38_target_full_moe_c1 CUDA::cudart)
  target_compile_options(qwen38_native_target_moe_graph PRIVATE
    -Wall -Wextra -Werror)
  set_target_properties(qwen38_native_target_moe_graph PROPERTIES
    POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE CUDA)

  add_library(qwen38_target_layer3_moe_owner STATIC
    src/moe/target_layer3_moe_owner.cu)
  target_include_directories(qwen38_target_layer3_moe_owner PUBLIC src)
  target_link_libraries(qwen38_target_layer3_moe_owner PUBLIC
    qwen38_target_layer3_native_bindings qwen38_native_target_moe_graph
    qwen38_target_moe_n640_device_stage qwen38_target_full_moe_c1
    qwen38_target_moe_b12x_aot qwen38_target_router_shared_c1 CUDA::cudart)
  target_compile_options(qwen38_target_layer3_moe_owner PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>)
  set_target_properties(qwen38_target_layer3_moe_owner PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

  add_library(qwen38_target_layer_moe_owner STATIC
    src/moe/target_layer_moe_owner.cu)
  target_include_directories(qwen38_target_layer_moe_owner PUBLIC src)
  target_link_libraries(qwen38_target_layer_moe_owner PUBLIC
    qwen38_target_layer_native_bindings qwen38_native_target_moe_graph
    qwen38_target_moe_n640_device_stage qwen38_target_full_moe_c1
    qwen38_target_moe_b12x_aot qwen38_target_router_shared_c1 CUDA::cudart)
  target_compile_options(qwen38_target_layer_moe_owner PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>)
  set_target_properties(qwen38_target_layer_moe_owner PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

  add_library(qwen38_target_qsa_k0_layer_owner STATIC
    src/decode/target_qsa_k0_layer_owner.cu)
  target_include_directories(qwen38_target_qsa_k0_layer_owner PUBLIC src)
  target_link_libraries(qwen38_target_qsa_k0_layer_owner PUBLIC
    qwen38_target_layer_moe_owner qwen38_target_layer_native_bindings
    qwen38_target_layer3_generation qwen38_native_qsa_graph
    qwen38_hyperconnection qwen38_native_hyperconnection_adapter
    qwen38_decode_execution CUDA::cudart)
  target_compile_options(qwen38_target_qsa_k0_layer_owner PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>)
  set_target_properties(qwen38_target_qsa_k0_layer_owner PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

  add_library(qwen38_target_gdn_layer_owner STATIC
    src/decode/target_gdn_layer_owner.cu)
  target_include_directories(qwen38_target_gdn_layer_owner PUBLIC src)
  target_link_libraries(qwen38_target_gdn_layer_owner PUBLIC
    qwen38_target_layer_moe_owner qwen38_target_layer_native_plan
    qwen38_target_slab_owner qwen38_gdn_graph
    qwen38_native_hyperconnection_adapter qwen38_decode_execution
    CUDA::cudart)
  target_compile_options(qwen38_target_gdn_layer_owner PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>)
  set_target_properties(qwen38_target_gdn_layer_owner PROPERTIES
    CUDA_ARCHITECTURES "121a" CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

  add_executable(qwen38-test-target-gdn-layer-owner
    tests/test_target_gdn_layer_owner.cc)
  target_link_libraries(qwen38-test-target-gdn-layer-owner PRIVATE
    qwen38_target_gdn_layer_owner)
  target_compile_options(qwen38-test-target-gdn-layer-owner PRIVATE
    -Wall -Wextra -Werror)
  set_target_properties(qwen38-test-target-gdn-layer-owner PROPERTIES
    LINKER_LANGUAGE CUDA)
  add_test(NAME qwen38-target-gdn-layer-owner-cpu-contract
    COMMAND qwen38-test-target-gdn-layer-owner)

  add_library(qwen38_target_k0_physical_layer_owners STATIC
    src/decode/target_k0_physical_layer_owners.cu)
  target_include_directories(qwen38_target_k0_physical_layer_owners PUBLIC src)
  target_link_libraries(qwen38_target_k0_physical_layer_owners PUBLIC
    qwen38_target_gdn_layer_owner qwen38_target_qsa_k0_layer_owner
    qwen38_target_k0_qsa_state_owner qwen38_target_k0_layer_owner_inventory
    qwen38_target_layer_native_plan CUDA::cudart)
  target_compile_options(qwen38_target_k0_physical_layer_owners PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>)
  set_target_properties(qwen38_target_k0_physical_layer_owners PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

  add_library(qwen38_target_k0_physical_startup_contract STATIC
    src/decode/target_k0_physical_startup_contract.cc)
  target_include_directories(qwen38_target_k0_physical_startup_contract PUBLIC src)
  target_link_libraries(qwen38_target_k0_physical_startup_contract PUBLIC
    CUDA::cudart)
  target_compile_options(qwen38_target_k0_physical_startup_contract PRIVATE
    -Wall -Wextra -Werror)

  add_library(qwen38_target_k0_physical_startup STATIC
    src/decode/target_k0_physical_startup.cu)
  target_include_directories(qwen38_target_k0_physical_startup PUBLIC src)
  target_link_libraries(qwen38_target_k0_physical_startup PUBLIC
    qwen38_target_k0_physical_startup_contract
    qwen38_target_k0_physical_layer_owners qwen38_target_k0_startup
    qwen38_native_token_io qwen38_mtp_executor qwen38_pair_reduce
    CUDA::cudart)
  target_compile_options(qwen38_target_k0_physical_startup PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xcompiler=-Wall,-Wextra,-Werror>)
  set_target_properties(qwen38_target_k0_physical_startup PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)

  add_executable(qwen38-test-target-k0-physical-startup
    tests/test_target_k0_physical_startup.cc)
  target_link_libraries(qwen38-test-target-k0-physical-startup PRIVATE
    qwen38_target_k0_physical_startup_contract)
  target_compile_options(qwen38-test-target-k0-physical-startup PRIVATE
    -Wall -Wextra -Werror)
  add_test(NAME qwen38-target-k0-physical-startup-cpu-contract
    COMMAND qwen38-test-target-k0-physical-startup)

  add_library(qwen38_target_layer3_moe_owner_preflight SHARED
    src/moe/target_layer3_moe_owner_preflight.cc)
  target_include_directories(qwen38_target_layer3_moe_owner_preflight PUBLIC src)
  target_link_libraries(qwen38_target_layer3_moe_owner_preflight PRIVATE
    qwen38_target_layer3_moe_owner qwen38_target_slab_owner CUDA::cudart)
  target_compile_options(qwen38_target_layer3_moe_owner_preflight PRIVATE
    -Wall -Wextra -Werror)
  target_link_options(qwen38_target_layer3_moe_owner_preflight PRIVATE
    -Wl,--exclude-libs,ALL)
  set_target_properties(qwen38_target_layer3_moe_owner_preflight PROPERTIES
    POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE CUDA
    CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
else()
  target_compile_definitions(qwen38_target_moe_b12x_aot PRIVATE
    ROCKET_QWEN38_TARGET_MOE_B12X_AOT=0)
endif()

add_library(qwen38_target_moe_b12x_native SHARED
  src/moe/target_moe_b12x_native_anchor.cc)
target_include_directories(qwen38_target_moe_b12x_native PUBLIC src)
target_link_libraries(qwen38_target_moe_b12x_native PRIVATE
  qwen38_target_moe_b12x_aot CUDA::cudart)
target_compile_options(qwen38_target_moe_b12x_native PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38_target_moe_b12x_native PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

add_library(qwen38_mtp_executor STATIC
  src/mtp/graph_runtime.cu
  src/mtp/input_fusion.cu
  src/mtp/nccl_winner_exchange.cu
  src/mtp/native_executor.cu
  src/mtp/state_arena.cu
)
target_include_directories(qwen38_mtp_executor PUBLIC src)
target_link_libraries(qwen38_mtp_executor PUBLIC
  qwen38_hyperconnection qwen38_route_compaction qwen38_token_output
  CUDA::cudart CUDA::cublas
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
add_test(NAME qwen38-target-layer3-oracle-cpu-contract
  COMMAND qwen38-target-layer3-oracle-auth)
add_test(NAME qwen38-target-k0-oracle-cpu-contract
  COMMAND qwen38-target-k0-oracle-auth)
set(ROCKET_QWEN38_LAYER3_ORACLE_CAPTURE
  "/home/glwillen/calibration/qwen38-k0-oracle-a1794d5-01/capture"
  CACHE PATH "Authenticated layer-3 K0 oracle capture")
if(EXISTS "${ROCKET_QWEN38_LAYER3_ORACLE_CAPTURE}/manifest.json")
  add_test(NAME qwen38-target-layer3-oracle-authenticated-contract
    COMMAND qwen38-target-layer3-oracle-auth
      "${ROCKET_QWEN38_LAYER3_ORACLE_CAPTURE}")
  add_test(NAME qwen38-target-k0-oracle-authenticated-contract
    COMMAND qwen38-target-k0-oracle-auth
      "${ROCKET_QWEN38_LAYER3_ORACLE_CAPTURE}")
endif()
add_test(NAME qwen38-qsa-expand-smoke COMMAND qwen38-qsa-expand-smoke)
add_test(NAME qwen38-layer3-rope-owner-cpu-contract
  COMMAND qwen38-test-layer3-rope-owner)
add_executable(qwen38-test-pair-reduce tests/test_pair_reduce.cu)
target_link_libraries(qwen38-test-pair-reduce PRIVATE qwen38_pair_reduce)
target_compile_options(qwen38-test-pair-reduce PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
)
add_test(NAME qwen38-pair-reduce-contract COMMAND qwen38-test-pair-reduce)
add_executable(qwen38-test-rdma-bootstrap-identity
  tests/test_rdma_bootstrap_identity.cc)
target_link_libraries(qwen38-test-rdma-bootstrap-identity PRIVATE
  qwen38_pair_reduce)
target_compile_options(qwen38-test-rdma-bootstrap-identity PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38-test-rdma-bootstrap-identity PROPERTIES
  LINKER_LANGUAGE CUDA)
add_test(NAME qwen38-rdma-bootstrap-identity-contract
  COMMAND qwen38-test-rdma-bootstrap-identity)
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

add_executable(qwen38-test-target-full-layer tests/test_target_full_layer.cc)
target_link_libraries(qwen38-test-target-full-layer PRIVATE qwen38_decode_execution)
target_compile_options(qwen38-test-target-full-layer PRIVATE -Wall -Wextra -Werror)
set_target_properties(qwen38-test-target-full-layer PROPERTIES LINKER_LANGUAGE CUDA)
add_test(NAME qwen38-target-full-layer-contract COMMAND qwen38-test-target-full-layer)

add_executable(qwen38-test-target-k0-layer-owner-inventory
  tests/test_target_k0_layer_owner_inventory.cc)
target_link_libraries(qwen38-test-target-k0-layer-owner-inventory PRIVATE
  qwen38_target_k0_layer_owner_inventory)
target_compile_options(qwen38-test-target-k0-layer-owner-inventory PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38-test-target-k0-layer-owner-inventory PROPERTIES
  LINKER_LANGUAGE CUDA)
add_test(NAME qwen38-target-k0-layer-owner-inventory-cpu-contract
  COMMAND qwen38-test-target-k0-layer-owner-inventory)

add_executable(qwen38-test-target-layer3-prefill
  tests/test_target_layer3_prefill.cc)
target_link_libraries(qwen38-test-target-layer3-prefill PRIVATE
  qwen38_decode_execution)
target_compile_options(qwen38-test-target-layer3-prefill PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38-test-target-layer3-prefill PROPERTIES
  LINKER_LANGUAGE CUDA)
add_test(NAME qwen38-target-layer3-prefill-contract
  COMMAND qwen38-test-target-layer3-prefill)

add_executable(qwen38-test-target-layer3-pair-reduce
  tests/test_target_layer3_pair_reduce.cc)
target_link_libraries(qwen38-test-target-layer3-pair-reduce PRIVATE
  qwen38_decode_execution)
target_compile_options(qwen38-test-target-layer3-pair-reduce PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38-test-target-layer3-pair-reduce PROPERTIES
  LINKER_LANGUAGE CUDA)
add_test(NAME qwen38-target-layer3-pair-reduce-contract
  COMMAND qwen38-test-target-layer3-pair-reduce)

add_executable(qwen38-test-target-k0-pair-reduce
  tests/test_target_k0_pair_reduce.cc)
target_link_libraries(qwen38-test-target-k0-pair-reduce PRIVATE
  qwen38_decode_execution)
target_compile_options(qwen38-test-target-k0-pair-reduce PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38-test-target-k0-pair-reduce PROPERTIES
  LINKER_LANGUAGE CUDA)
add_test(NAME qwen38-target-k0-pair-reduce-contract
  COMMAND qwen38-test-target-k0-pair-reduce)

add_executable(qwen38-test-target-k0-executor
  tests/test_target_k0_executor.cc)
target_link_libraries(qwen38-test-target-k0-executor PRIVATE
  qwen38_decode_execution)
target_compile_options(qwen38-test-target-k0-executor PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38-test-target-k0-executor PROPERTIES
  LINKER_LANGUAGE CUDA)
add_test(NAME qwen38-target-k0-executor-contract
  COMMAND qwen38-test-target-k0-executor)

add_executable(qwen38-test-target-layer3-native-plan
  tests/test_target_layer3_native_plan.cc)
target_link_libraries(qwen38-test-target-layer3-native-plan PRIVATE
  qwen38_target_layer3_native_plan)
target_compile_options(qwen38-test-target-layer3-native-plan PRIVATE
  -Wall -Wextra -Werror)

add_executable(qwen38-test-target-layer-native-plan
  tests/test_target_layer_native_plan.cc)
target_link_libraries(qwen38-test-target-layer-native-plan PRIVATE
  qwen38_target_layer_native_plan)
target_compile_options(qwen38-test-target-layer-native-plan PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38-test-target-layer-native-plan PROPERTIES
  LINKER_LANGUAGE CUDA)

add_executable(qwen38-test-target-layer3-native-bindings
  tests/test_target_layer3_native_bindings.cc)
target_link_libraries(qwen38-test-target-layer3-native-bindings PRIVATE
  qwen38_target_layer3_native_bindings)
target_compile_options(qwen38-test-target-layer3-native-bindings PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38-test-target-layer3-native-bindings PROPERTIES
  LINKER_LANGUAGE CUDA)

add_executable(qwen38-test-target-moe-n640-device-stage
  tests/test_target_moe_n640_device_stage.cc)
target_link_libraries(qwen38-test-target-moe-n640-device-stage PRIVATE
  qwen38_target_moe_n640_device_stage)
target_compile_options(qwen38-test-target-moe-n640-device-stage PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38-test-target-moe-n640-device-stage PROPERTIES
  LINKER_LANGUAGE CUDA)
add_test(NAME qwen38-target-moe-n640-device-stage-cpu-contract
  COMMAND qwen38-test-target-moe-n640-device-stage)

add_executable(qwen38-test-target-layer3-generation
  tests/test_target_layer3_generation.cc)
target_link_libraries(qwen38-test-target-layer3-generation PRIVATE
  qwen38_target_layer3_generation)
target_compile_options(qwen38-test-target-layer3-generation PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38-test-target-layer3-generation PROPERTIES
  LINKER_LANGUAGE CUDA)
add_test(NAME qwen38-target-layer3-generation-contract
  COMMAND qwen38-test-target-layer3-generation)

add_executable(qwen38-test-native-qsa-c-api
  tests/test_native_qsa_graph_c_api.cc)
target_link_libraries(qwen38-test-native-qsa-c-api PRIVATE
  qwen38_native_qsa_c_api)
target_compile_options(qwen38-test-native-qsa-c-api PRIVATE -Wall -Wextra -Werror)
set_target_properties(qwen38-test-native-qsa-c-api PROPERTIES LINKER_LANGUAGE CUDA)
add_test(NAME qwen38-native-qsa-c-api-contract
  COMMAND qwen38-test-native-qsa-c-api)

add_executable(qwen38-test-native-hyperconnection-adapter
  tests/test_native_hyperconnection_adapter.cc)
target_link_libraries(qwen38-test-native-hyperconnection-adapter PRIVATE
  qwen38_native_hyperconnection_adapter)
target_compile_options(qwen38-test-native-hyperconnection-adapter PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38-test-native-hyperconnection-adapter PROPERTIES
  LINKER_LANGUAGE CUDA)
add_test(NAME qwen38-native-hyperconnection-adapter-contract
  COMMAND qwen38-test-native-hyperconnection-adapter)

add_executable(qwen38-test-linear-attention-layer tests/test_linear_attention_layer.cc)
target_link_libraries(qwen38-test-linear-attention-layer PRIVATE qwen38_decode_execution)
target_compile_options(qwen38-test-linear-attention-layer PRIVATE -Wall -Wextra -Werror)
set_target_properties(qwen38-test-linear-attention-layer PROPERTIES LINKER_LANGUAGE CUDA)
add_test(NAME qwen38-linear-attention-layer-contract COMMAND qwen38-test-linear-attention-layer)

add_executable(qwen38-test-target-gdn-layer tests/test_target_gdn_layer.cc)
target_link_libraries(qwen38-test-target-gdn-layer PRIVATE qwen38_decode_execution)
target_compile_options(qwen38-test-target-gdn-layer PRIVATE -Wall -Wextra -Werror)
set_target_properties(qwen38-test-target-gdn-layer PROPERTIES LINKER_LANGUAGE CUDA)
add_test(NAME qwen38-target-gdn-layer-cpu-contract COMMAND qwen38-test-target-gdn-layer)

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
add_executable(qwen38-test-token-io-roots tests/test_token_io_roots.cc)
target_link_libraries(qwen38-test-token-io-roots PRIVATE qwen38_native_token_io)
target_compile_options(qwen38-test-token-io-roots PRIVATE
  -Wall -Wextra -Werror)
set_target_properties(qwen38-test-token-io-roots PROPERTIES LINKER_LANGUAGE CUDA)
add_test(NAME qwen38-token-io-roots-rejection-contract
  COMMAND qwen38-test-token-io-roots)
add_test(NAME qwen38-target-k0-startup-rejection-contract
  COMMAND qwen38-test-target-k0-startup)
set(ROCKET_QWEN38_TOKENIZER_ROOT
  "/home/glwillen/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47"
  CACHE PATH "Pinned Qwen3.8 tokenizer root")
set(ROCKET_QWEN38_K0_ORACLE_ROOT
  "/home/glwillen/calibration/qwen38-k0-oracle-a1794d5-01/capture"
  CACHE PATH "Pinned 35-row K0 oracle capture")
if(EXISTS "${ROCKET_QWEN38_TOKENIZER_ROOT}/tokenizer.json" AND
   EXISTS "${ROCKET_QWEN38_K0_ORACLE_ROOT}/manifest.json")
  add_test(NAME qwen38-token-io-roots-authenticated-contract
    COMMAND qwen38-test-token-io-roots
      "${ROCKET_QWEN38_TOKENIZER_ROOT}" "${ROCKET_QWEN38_K0_ORACLE_ROOT}")
  add_test(NAME qwen38-target-k0-startup-authenticated-contract
    COMMAND qwen38-test-target-k0-startup
      "${ROCKET_QWEN38_TOKENIZER_ROOT}" "${ROCKET_QWEN38_K0_ORACLE_ROOT}")
  add_test(NAME qwen38-target-k0-startup-incomplete-contract
    COMMAND qwen38-target-k0-startup --tokenizer
      "${ROCKET_QWEN38_TOKENIZER_ROOT}" --oracle "${ROCKET_QWEN38_K0_ORACLE_ROOT}")
  set_tests_properties(qwen38-target-k0-startup-incomplete-contract
    PROPERTIES WILL_FAIL TRUE)
endif()
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

add_executable(qwen38-test-route-compaction tests/test_route_compaction.cc)
target_link_libraries(qwen38-test-route-compaction PRIVATE
  qwen38_route_compaction_contract)
target_compile_options(qwen38-test-route-compaction PRIVATE
  -Wall -Wextra -Werror)
add_test(NAME qwen38-route-compaction-cpu-contract
  COMMAND qwen38-test-route-compaction)

add_executable(qwen38-test-target-moe-c1 tests/test_target_moe_c1.cc)
target_link_libraries(qwen38-test-target-moe-c1 PRIVATE
  qwen38_target_moe_c1_contract)
target_compile_options(qwen38-test-target-moe-c1 PRIVATE
  -Wall -Wextra -Werror)
add_test(NAME qwen38-target-moe-c1-cpu-contract
  COMMAND qwen38-test-target-moe-c1)

add_executable(qwen38-test-target-moe-b12x-aot-contract
  tests/test_target_moe_b12x_aot_contract.cc)
target_link_libraries(qwen38-test-target-moe-b12x-aot-contract PRIVATE
  qwen38_target_moe_b12x_aot)
target_compile_options(qwen38-test-target-moe-b12x-aot-contract PRIVATE
  -Wall -Wextra -Werror)
add_test(NAME qwen38-target-moe-b12x-aot-cpu-contract
  COMMAND qwen38-test-target-moe-b12x-aot-contract)

add_executable(qwen38-test-target-moe-n640-materializer
  tests/test_target_moe_n640_materializer.cc)
target_link_libraries(qwen38-test-target-moe-n640-materializer PRIVATE
  qwen38_target_moe_n640_materializer)
target_compile_options(qwen38-test-target-moe-n640-materializer PRIVATE
  -Wall -Wextra -Werror)
add_test(NAME qwen38-target-moe-n640-materializer-cpu-contract
  COMMAND qwen38-test-target-moe-n640-materializer)

if(TARGET qwen38_target_router_shared_c1)
  add_executable(qwen38-test-target-router-shared-c1-contract
    tests/test_target_router_shared_c1_contract.cc)
  target_link_libraries(qwen38-test-target-router-shared-c1-contract PRIVATE
    qwen38_target_router_shared_c1)
  target_compile_options(qwen38-test-target-router-shared-c1-contract PRIVATE
    -Wall -Wextra -Werror)
  add_test(NAME qwen38-target-router-shared-c1-cpu-contract
    COMMAND qwen38-test-target-router-shared-c1-contract)

  add_executable(qwen38-test-target-full-moe-c1-contract
    tests/test_target_full_moe_c1_contract.cc)
  target_link_libraries(qwen38-test-target-full-moe-c1-contract PRIVATE
    qwen38_target_full_moe_c1)
  target_compile_options(qwen38-test-target-full-moe-c1-contract PRIVATE
    -Wall -Wextra -Werror)
  add_test(NAME qwen38-target-full-moe-c1-cpu-contract
    COMMAND qwen38-test-target-full-moe-c1-contract)

  add_executable(qwen38-test-native-target-moe-graph-contract
    tests/test_native_target_moe_graph_contract.cc)
  target_link_libraries(qwen38-test-native-target-moe-graph-contract PRIVATE
    qwen38_native_target_moe_graph)
  target_compile_options(qwen38-test-native-target-moe-graph-contract PRIVATE
    -Wall -Wextra -Werror)
  add_test(NAME qwen38-native-target-moe-graph-cpu-contract
    COMMAND qwen38-test-native-target-moe-graph-contract)

  add_executable(qwen38-test-target-layer3-moe-owner-plan
    tests/test_target_layer3_moe_owner_plan.cc)
  target_link_libraries(qwen38-test-target-layer3-moe-owner-plan PRIVATE
    qwen38_target_layer3_moe_owner)
  target_compile_options(qwen38-test-target-layer3-moe-owner-plan PRIVATE
    -Wall -Wextra -Werror)
  set_target_properties(qwen38-test-target-layer3-moe-owner-plan PROPERTIES
    LINKER_LANGUAGE CUDA)
  add_test(NAME qwen38-target-layer3-moe-owner-plan-cpu-contract
    COMMAND qwen38-test-target-layer3-moe-owner-plan)
endif()

add_executable(qwen38-test-fp8-routed-experts
  tests/test_fp8_routed_experts.cc)
target_link_libraries(qwen38-test-fp8-routed-experts PRIVATE
  qwen38_route_compaction_contract)
target_compile_options(qwen38-test-fp8-routed-experts PRIVATE
  -Wall -Wextra -Werror)
add_test(NAME qwen38-fp8-routed-experts-cpu-contract
  COMMAND qwen38-test-fp8-routed-experts)

add_executable(qwen38-test-fp8-routed-experts-cuda
  tests/test_fp8_routed_experts.cu)
target_link_libraries(qwen38-test-fp8-routed-experts-cuda PRIVATE
  qwen38_route_compaction CUDA::cudart)
target_compile_options(qwen38-test-fp8-routed-experts-cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>)
add_test(NAME qwen38-fp8-routed-experts-cuda-contract
  COMMAND qwen38-test-fp8-routed-experts-cuda)

add_executable(qwen38-mtp-routed-expert-profile
  bench/mtp_routed_expert_profile.cu)
target_link_libraries(qwen38-mtp-routed-expert-profile PRIVATE
  qwen38_route_compaction CUDA::cudart)
target_compile_options(qwen38-mtp-routed-expert-profile PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>)

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
add_test(NAME qwen38-target-k0-physical-layer-owners-source-contract
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_target_k0_physical_layer_owners_source.py)
if(TARGET qwen38_target_layer3_moe_owner_preflight)
  add_test(NAME qwen38-target-layer3-moe-owner-preflight-cpu-contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/runtime/test_qwen38_layer3_moe_owner_preflight.py)
  set_tests_properties(qwen38-target-layer3-moe-owner-preflight-cpu-contract
    PROPERTIES ENVIRONMENT
      "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src;ROCKET_QWEN38_LAYER3_MOE_PREFLIGHT_LIBRARY=$<TARGET_FILE:qwen38_target_layer3_moe_owner_preflight>")
endif()
add_test(NAME qwen38-cuda-slab-loader-cpu-contract
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_slab_loader.py)
set_tests_properties(qwen38-cuda-slab-loader-cpu-contract
  PROPERTIES ENVIRONMENT
    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src:${CMAKE_CURRENT_SOURCE_DIR}/tests")
add_test(NAME qwen38-layer3-factory-accepted-loader-cpu-contract
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_layer3_factory.py)
set_tests_properties(qwen38-layer3-factory-accepted-loader-cpu-contract
  PROPERTIES ENVIRONMENT
    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src;ROCKET_QWEN38_TARGET_SLAB_OWNER_LIBRARY=$<TARGET_FILE:qwen38_target_slab_owner>")
add_test(NAME qwen38-target-layer-descriptor-cpu-contract
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_target_layer_descriptor.py)
set_tests_properties(qwen38-target-layer-descriptor-cpu-contract
  PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src")
add_test(NAME qwen38-target-moe-n640-materializer-source-contract
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_target_moe_n640_materializer_source.py)
add_test(NAME qwen38-qsa-sidecar-owner-source-contract
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_qsa_sidecar_owner_source.py)
add_test(NAME qwen38-layer3-rope-owner-source-contract
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_layer3_rope_owner_source.py)
add_test(NAME qwen38-target-moe-n640-device-stage-source-contract
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_target_moe_n640_device_stage_source.py)
add_test(NAME qwen38-route-compaction-source-contract
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_route_compaction_source.py)
add_test(NAME qwen38-fp8-routed-experts-source-contract
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fp8_routed_experts_source.py)
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
