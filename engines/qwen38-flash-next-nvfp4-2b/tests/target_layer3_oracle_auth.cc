// SPDX-License-Identifier: Apache-2.0
#include "decode/target_layer3_oracle_comparator.h"

#include <filesystem>
#include <cstdlib>
#include <cstring>

namespace decode = rocket::qwen38::decode;
namespace pr = rocket::qwen38::pair_reduce;
namespace {
struct Sink final : pr::OtelStageSink {
  void emit_span_and_log(const pr::SpanRecord& p) noexcept override {
    ++spans; rank = p.rank; outcome = p.outcome;
  }
  void record_duration(const pr::MetricPoint&) noexcept override { ++metrics; }
  int spans=0, metrics=0, rank=-2;
  pr::Outcome outcome=pr::Outcome::kOk;
};
struct FakeCuda final : decode::TargetLayer3OracleCudaApi {
  cudaError_t host_alloc(void** p,std::size_t n) noexcept override {
    if(fail_alloc)return cudaErrorMemoryAllocation;
    alloc_bytes=n; *p=std::malloc(n); return *p?cudaSuccess:cudaErrorMemoryAllocation;
  }
  cudaError_t free_host(void* p) noexcept override {
    std::free(p); ++frees; return fail_free?cudaErrorUnknown:cudaSuccess;
  }
  cudaError_t event_create(cudaEvent_t* e) noexcept override {
    *e=reinterpret_cast<cudaEvent_t>(0x20); ++creates; return cudaSuccess;
  }
  cudaError_t event_destroy(cudaEvent_t) noexcept override {++destroys;return cudaSuccess;}
  cudaError_t copy_d2h(void* d,const void* s,std::size_t n,cudaStream_t) noexcept override {
    copy_bytes=n; if(fail_copy)return cudaErrorUnknown; std::memcpy(d,s,n);return cudaSuccess;
  }
  cudaError_t event_record(cudaEvent_t,cudaStream_t) noexcept override{return cudaSuccess;}
  cudaError_t event_sync(cudaEvent_t) noexcept override{return cudaSuccess;}
  std::size_t alloc_bytes=0,copy_bytes=0;int creates=0,destroys=0,frees=0;
  bool fail_alloc=false,fail_copy=false,fail_free=false;
};
}
int main(int argc, char** argv) {
  if (argc != 1 && argc != 2) return 2;
  try {
    std::array<std::uint16_t, decode::kLayer3OracleWidth> expected{};
    if (argc == 2)
      expected = decode::authenticate_layer3_oracle_row34(
          std::filesystem::path(argv[1]));
    Sink sink;
    try { decode::NativeTargetLayer3OracleComparator invalid(7, ".", sink); return 9; }
    catch (const std::invalid_argument&) {}
    if(sink.spans!=1 || sink.metrics!=0 || sink.rank!=-1 ||
       sink.outcome!=pr::Outcome::kContractError) return 10;
    if(argc==2){
      FakeCuda fake; Sink valid;
      {
        decode::NativeTargetLayer3OracleComparator comparator(0,argv[1],valid,&fake);
        if(!comparator.compare_row34(
             reinterpret_cast<const __nv_bfloat16*>(expected.data()),
             reinterpret_cast<cudaStream_t>(0x10))) return 12;
        if(fake.alloc_bytes!=2*decode::kLayer3OracleRowBytes ||
           fake.copy_bytes!=decode::kLayer3OracleRowBytes ||
           !comparator.evidence().observed_exact_hash) return 13;
      }
      if(fake.creates!=1 || fake.destroys!=1 || fake.frees!=1) return 14;
      FakeCuda rejected_cuda; rejected_cuda.fail_copy=true; Sink rejected_sink;
      {
        decode::NativeTargetLayer3OracleComparator comparator(
            0,argv[1],rejected_sink,&rejected_cuda);
        if(comparator.compare_row34(
             reinterpret_cast<const __nv_bfloat16*>(expected.data()),
             reinterpret_cast<cudaStream_t>(0x10))) return 15;
      }
      if(rejected_sink.outcome!=pr::Outcome::kCudaError) return 16;
      FakeCuda allocation_cuda; allocation_cuda.fail_alloc=true;
      Sink allocation_sink;
      try { decode::NativeTargetLayer3OracleComparator comparator(
            0,argv[1],allocation_sink,&allocation_cuda); return 18; }
      catch (const std::runtime_error&) {}
      if(allocation_sink.outcome!=pr::Outcome::kCudaError) return 19;
      FakeCuda cleanup_cuda; cleanup_cuda.fail_free=true; Sink cleanup_sink;
      { decode::NativeTargetLayer3OracleComparator comparator(
            0,argv[1],cleanup_sink,&cleanup_cuda); }
      if(cleanup_sink.outcome!=pr::Outcome::kCudaError) return 17;
    }
    auto observed = expected;
    auto exact = decode::compare_layer3_oracle_row34(
        expected.data(), observed.data());
    if (!exact.accepted || exact.max_ulp || exact.mismatch_count) return 3;
    std::size_t index = 0;
    while (index < observed.size() &&
           ((observed[index] & 0x7f80) == 0x7f80 ||
            observed[index] == 0x7fff)) ++index;
    if (index == observed.size()) return 4;
    observed[index] = static_cast<std::uint16_t>(expected[index] + 1);
    auto one = decode::compare_layer3_oracle_row34(expected.data(), observed.data());
    if (!one.accepted || one.max_ulp != 1 || one.mismatch_count != 1) return 5;
    observed[index] = static_cast<std::uint16_t>(expected[index] + 2);
    auto two = decode::compare_layer3_oracle_row34(expected.data(), observed.data());
    if (two.accepted || two.max_ulp != 2 || two.mismatch_count != 1) return 6;
    std::array<std::uint16_t, decode::kLayer3OracleWidth> edge_expected{};
    auto edge_observed = edge_expected;
    edge_expected[0] = 0x8000;  // -0 and +0 are the same BF16 value.
    edge_expected[1] = 0xbf80; edge_observed[1] = 0xbf7f;
    auto edges = decode::compare_layer3_oracle_row34(
        edge_expected.data(), edge_observed.data());
    if (!edges.accepted || edges.max_ulp != 1) return 7;
    edge_expected.fill(0); edge_observed.fill(0);
    edge_expected[2] = edge_observed[2] = 0x7fc1;
    auto nan = decode::compare_layer3_oracle_row34(
        edge_expected.data(), edge_observed.data());
    if(nan.accepted || nan.nan_count != 1) return 8;
    edge_expected.fill(0); edge_observed.fill(0);
    edge_expected[2] = edge_observed[2] = 0x7f80;
    auto infinity = decode::compare_layer3_oracle_row34(
        edge_expected.data(), edge_observed.data());
    return infinity.accepted && infinity.rms == 0 ? 0 : 11;
  } catch (...) { return 1; }
}
