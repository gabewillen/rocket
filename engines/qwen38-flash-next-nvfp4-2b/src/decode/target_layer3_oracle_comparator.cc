// SPDX-License-Identifier: Apache-2.0
#include "decode/target_layer3_oracle_comparator.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

extern "C" {
struct evp_md_ctx_st; struct evp_md_st;
evp_md_ctx_st* EVP_MD_CTX_new(); void EVP_MD_CTX_free(evp_md_ctx_st*);
const evp_md_st* EVP_sha256();
int EVP_DigestInit_ex(evp_md_ctx_st*, const evp_md_st*, void*);
int EVP_DigestUpdate(evp_md_ctx_st*, const void*, std::size_t);
int EVP_DigestFinal_ex(evp_md_ctx_st*, unsigned char*, unsigned int*);
}

namespace rocket::qwen38::decode { namespace {
struct Fd { int n=-1; ~Fd(){ if(n>=0) close(n); } };
class CudaFailure final : public std::runtime_error {
 public: using std::runtime_error::runtime_error;
};
std::string hash_fd(int fd, std::uint64_t offset, std::size_t bytes) {
  auto* ctx=EVP_MD_CTX_new(); if(!ctx) throw std::runtime_error("oracle SHA init failed");
  std::array<unsigned char,32> out{}; unsigned int n=0;
  std::array<std::uint8_t,65536> buffer{}; std::size_t done=0;
  bool ok=EVP_DigestInit_ex(ctx,EVP_sha256(),nullptr)==1;
  while(ok && done<bytes) {
    const auto want=std::min(buffer.size(),bytes-done);
    const auto got=pread(fd,buffer.data(),want,offset+done);
    ok=got==static_cast<ssize_t>(want) && EVP_DigestUpdate(ctx,buffer.data(),want)==1;
    done+=want;
  }
  ok=ok && EVP_DigestFinal_ex(ctx,out.data(),&n)==1 && n==out.size();
  EVP_MD_CTX_free(ctx); if(!ok) throw std::runtime_error("oracle hash/read failed");
  constexpr char hex[]="0123456789abcdef"; std::string text(64,'0');
  for(std::size_t i=0;i<out.size();++i){text[2*i]=hex[out[i]>>4];text[2*i+1]=hex[out[i]&15];}
  return text;
}
std::string hash_bytes(const void* data, std::size_t bytes) {
  auto* ctx=EVP_MD_CTX_new(); std::array<unsigned char,32> out{}; unsigned int n=0;
  const bool ok=ctx && EVP_DigestInit_ex(ctx,EVP_sha256(),nullptr)==1 &&
      EVP_DigestUpdate(ctx,data,bytes)==1 &&
      EVP_DigestFinal_ex(ctx,out.data(),&n)==1 && n==out.size();
  if(ctx)EVP_MD_CTX_free(ctx);
  if(!ok)throw std::runtime_error("oracle memory hash failed");
  constexpr char hex[]="0123456789abcdef";std::string text(64,'0');
  for(std::size_t i=0;i<out.size();++i){text[2*i]=hex[out[i]>>4];text[2*i+1]=hex[out[i]&15];}
  return text;
}
float bf16(std::uint16_t bits) { return std::bit_cast<float>(std::uint32_t(bits)<<16); }
bool nan(std::uint16_t b){return (b&0x7f80)==0x7f80 && (b&0x007f)!=0;}
std::uint32_t ordered(std::uint16_t b){
  return (b&0x8000) ? 0x8000u-(b&0x7fffu) : 0x8000u+b;
}
class NativeCudaApi final : public TargetLayer3OracleCudaApi {
 public:
  cudaError_t host_alloc(void** p,std::size_t n) noexcept override{return cudaHostAlloc(p,n,cudaHostAllocDefault);}
  cudaError_t free_host(void* p) noexcept override{return cudaFreeHost(p);}
  cudaError_t event_create(cudaEvent_t* e) noexcept override{return cudaEventCreateWithFlags(e,cudaEventDisableTiming);}
  cudaError_t event_destroy(cudaEvent_t e) noexcept override{return cudaEventDestroy(e);}
  cudaError_t copy_d2h(void* d,const void* s,std::size_t n,cudaStream_t st) noexcept override{return cudaMemcpyAsync(d,s,n,cudaMemcpyDeviceToHost,st);}
  cudaError_t event_record(cudaEvent_t e,cudaStream_t st) noexcept override{return cudaEventRecord(e,st);}
  cudaError_t event_sync(cudaEvent_t e) noexcept override{return cudaEventSynchronize(e);}
};
NativeCudaApi native_cuda;
}  // namespace

std::array<std::uint16_t,kLayer3OracleWidth> authenticate_layer3_oracle_row34(
    const std::filesystem::path& capture) {
  Fd dir{open(capture.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW)};
  if(dir.n<0) throw std::invalid_argument("oracle capture changed");
  Fd manifest{openat(dir.n,"manifest.json",O_RDONLY|O_CLOEXEC|O_NOFOLLOW)};
  struct stat st{};
  if(manifest.n<0 || fstat(manifest.n,&st) || !S_ISREG(st.st_mode) || st.st_size!=19533 ||
     hash_fd(manifest.n,0,st.st_size)!=kLayer3OracleManifestSha256)
    throw std::invalid_argument("oracle manifest identity changed");
  Fd layer{openat(dir.n,"layer-03.bin",O_RDONLY|O_CLOEXEC|O_NOFOLLOW)};
  if(layer.n<0 || fstat(layer.n,&st) || !S_ISREG(st.st_mode) || st.st_size!=716800)
    throw std::invalid_argument("oracle layer03 identity changed");
  const struct stat opened_stat=st;
  std::array<std::uint16_t,kLayer3OracleWidth> row{};
  if(pread(layer.n,row.data(),kLayer3OracleRowBytes,kLayer3OracleRow34Offset)!=
         static_cast<ssize_t>(kLayer3OracleRowBytes) ||
     hash_bytes(row.data(),kLayer3OracleRowBytes)!=kLayer3OracleRow34Sha256 ||
     hash_fd(layer.n,0,opened_stat.st_size)!=kLayer3OracleFileSha256 ||
     fstat(layer.n,&st) || st.st_dev!=opened_stat.st_dev ||
     st.st_ino!=opened_stat.st_ino || st.st_size!=opened_stat.st_size ||
     st.st_mtim.tv_sec!=opened_stat.st_mtim.tv_sec ||
     st.st_mtim.tv_nsec!=opened_stat.st_mtim.tv_nsec ||
     st.st_ctim.tv_sec!=opened_stat.st_ctim.tv_sec ||
     st.st_ctim.tv_nsec!=opened_stat.st_ctim.tv_nsec)
    throw std::invalid_argument("oracle row34 identity changed");
  return row;
}

TargetLayer3OracleEvidence compare_layer3_oracle_row34(
    const std::uint16_t* expected,const std::uint16_t* observed) {
  if(!expected||!observed) throw std::invalid_argument("oracle rows changed");
  TargetLayer3OracleEvidence e{}; double squares=0;
  for(std::size_t i=0;i<kLayer3OracleWidth;++i){
    const bool has_nan=nan(expected[i])||nan(observed[i]);
    const auto ulp=has_nan?65535u:static_cast<std::uint32_t>(
        std::max(ordered(expected[i]),ordered(observed[i]))-
        std::min(ordered(expected[i]),ordered(observed[i])));
    const float delta=has_nan?INFINITY:(expected[i]==observed[i]?0.0f:
        std::abs(bf16(expected[i])-bf16(observed[i])));
    e.max_abs=std::max(e.max_abs,delta); squares+=double(delta)*delta;
    if(ulp){++e.mismatch_count;} if(has_nan){++e.nan_count;}
    if(ulp>e.max_ulp){e.max_ulp=ulp;e.max_ulp_index=i;}
    e.hidden_stream_max_ulp[i/2560]=std::max(e.hidden_stream_max_ulp[i/2560],ulp);
  }
  e.rms=std::sqrt(squares/kLayer3OracleWidth); e.token_max_ulp=e.max_ulp;
  e.accepted=e.nan_count==0 && e.max_ulp<=1;
  return e;
}

NativeTargetLayer3OracleComparator::NativeTargetLayer3OracleComparator(
    int rank,const std::filesystem::path& capture,pair_reduce::OtelStageSink& telemetry,
    TargetLayer3OracleCudaApi* cuda_api)
    :rank_(rank),telemetry_(&telemetry),cuda_api_(cuda_api?cuda_api:&native_cuda){
  try {
    if(rank!=0&&rank!=1) throw std::invalid_argument("oracle rank changed");
    const auto row=authenticate_layer3_oracle_row34(capture);
    if(cuda_api_->host_alloc(&pinned_,2*kLayer3OracleRowBytes)!=cudaSuccess)
      throw CudaFailure("oracle pinned allocation failed");
    expected_=static_cast<std::uint16_t*>(pinned_); observed_=expected_+kLayer3OracleWidth;
    std::memcpy(expected_,row.data(),kLayer3OracleRowBytes);
    if(cuda_api_->event_create(&ready_)!=cudaSuccess)
      throw CudaFailure("oracle event allocation failed");
    authenticated_=true;
  } catch (const CudaFailure&) {
    if(ready_) cuda_api_->event_destroy(ready_);
    if(pinned_) cuda_api_->free_host(pinned_);
    emit(pair_reduce::Outcome::kCudaError,0);
    throw;
  } catch (...) { if(ready_) cuda_api_->event_destroy(ready_); if(pinned_) cuda_api_->free_host(pinned_); emit(pair_reduce::Outcome::kContractError,0); throw; }
}
NativeTargetLayer3OracleComparator::~NativeTargetLayer3OracleComparator(){
  bool failed=false;
  if(ready_&&cuda_api_->event_destroy(ready_)!=cudaSuccess)failed=true;
  if(pinned_&&cuda_api_->free_host(pinned_)!=cudaSuccess)failed=true;
  if(failed)emit(pair_reduce::Outcome::kCudaError,0);
}
void NativeTargetLayer3OracleComparator::emit(pair_reduce::Outcome o,std::uint64_t bytes) noexcept {
  if(!telemetry_) return;
  const int diagnostic_rank=(rank_==0||rank_==1)?rank_:-1;
  telemetry_->emit_span_and_log({"rocket.qwen38.layer3_oracle.lifecycle","layer3-row34-oracle","oracle-05ea3af",diagnostic_rank,1,pair_reduce::kDtype,o,0,bytes});
  if(diagnostic_rank>=0)
    telemetry_->record_duration({diagnostic_rank,1,pair_reduce::kDtype,o,0});
}
bool NativeTargetLayer3OracleComparator::compare_row34(const __nv_bfloat16* device,cudaStream_t stream){
  if(!authenticated_||compared_||!device||!stream){emit(pair_reduce::Outcome::kContractError,0);return false;}
  compared_=true;
  if(cuda_api_->copy_d2h(observed_,device,kLayer3OracleRowBytes,stream)!=cudaSuccess ||
     cuda_api_->event_record(ready_,stream)!=cudaSuccess || cuda_api_->event_sync(ready_)!=cudaSuccess){emit(pair_reduce::Outcome::kCudaError,0);return false;}
  try {
    evidence_=compare_layer3_oracle_row34(expected_,observed_);
    evidence_.expected_hash_authenticated=true;
    evidence_.observed_sha256 = [&]{
    // The pinned observation is fixed-size; use an anonymous temporary digest context.
    auto* c=EVP_MD_CTX_new(); std::array<unsigned char,32> d{}; unsigned int n=0;
    bool ok=c&&EVP_DigestInit_ex(c,EVP_sha256(),nullptr)==1&&EVP_DigestUpdate(c,observed_,kLayer3OracleRowBytes)==1&&EVP_DigestFinal_ex(c,d.data(),&n)==1&&n==32;
    if(c) EVP_MD_CTX_free(c);
    if(!ok) throw std::runtime_error("oracle observed hash failed");
      constexpr char h[]="0123456789abcdef";std::string s(64,'0');for(int i=0;i<32;++i){s[2*i]=h[d[i]>>4];s[2*i+1]=h[d[i]&15];}return s;}();
    evidence_.observed_exact_hash=evidence_.observed_sha256==kLayer3OracleRow34Sha256;
  } catch (...) {
    emit(pair_reduce::Outcome::kContractError,kLayer3OracleRowBytes);
    return false;
  }
  emit(evidence_.accepted?pair_reduce::Outcome::kOk:pair_reduce::Outcome::kContractError,kLayer3OracleRowBytes);
  return evidence_.accepted;
}
}  // namespace rocket::qwen38::decode
