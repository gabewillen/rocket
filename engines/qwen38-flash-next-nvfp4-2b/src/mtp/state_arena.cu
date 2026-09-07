// SPDX-License-Identifier: Apache-2.0
#include "mtp/state_arena.h"
#include <cuda_runtime.h>
#include <string>

namespace rocket::qwen38::mtp { namespace {
void check(cudaError_t s, const char* op) { if (s != cudaSuccess) throw StateArenaError(std::string(op)+": "+cudaGetErrorString(s)); }
template<class T> T* take(std::byte*& p, std::size_t n) { T* out=reinterpret_cast<T*>(p); p += n*sizeof(T); return out; }
__global__ void select_prefixes(PrefixStateView src, PrefixStateView dst,
 const std::int32_t* widths, int sequences, int depth, bool mrope) {
  int sequence=blockIdx.x; int step=min(max(widths[sequence],1),depth)-1;
  for(int c=threadIdx.x;c<kMtpMultiHidden;c+=blockDim.x) dst.multi_hidden[sequence*kMtpMultiHidden+c]=src.multi_hidden[(step*sequences+sequence)*kMtpMultiHidden+c];
  for(int c=threadIdx.x;c<kQsaMainKvWidth;c+=blockDim.x){int f=(step*sequences+sequence)*kQsaMainKvWidth+c,t=sequence*kQsaMainKvWidth+c;dst.main_key[t]=src.main_key[f];dst.main_value[t]=src.main_value[f];}
  for(int c=threadIdx.x;c<kQsaIndexerWidth;c+=blockDim.x){int f=(step*sequences+sequence)*kQsaIndexerWidth+c,t=sequence*kQsaIndexerWidth+c;dst.raw_key[t]=src.raw_key[f];dst.compressed_key[t]=src.compressed_key[f];}
  if(threadIdx.x==0){int f=step*sequences+sequence;dst.main_slots[sequence]=src.main_slots[f];dst.raw_slots[sequence]=src.raw_slots[f];dst.compressed_slots[sequence]=src.compressed_slots[f];dst.compressed_valid[sequence]=src.compressed_valid[f];if(mrope)for(int a=0;a<kMropeAxes;++a)dst.rope_positions[sequence*kMropeAxes+a]=src.rope_positions[f*kMropeAxes+a];}
}
}  // namespace
StateArena::StateArena(int s,int d,bool mr):sequences_(s),depth_(d),uses_mrope_(mr){
 if(!(s==1||s==2||s==4||s==8||s==16)||d<1||d>7||(d>4&&s>4))throw StateArenaError("MTP state arena shape is outside graph policy");
 auto size_for=[mr](std::size_t r){return r*(sizeof(__nv_bfloat16)*(kMtpMultiHidden+2*kQsaMainKvWidth+2*kQsaIndexerWidth)+sizeof(std::int32_t)*4+(mr?sizeof(std::int64_t)*kMropeAxes:0));};
 std::size_t pr=std::size_t(s)*d,sr=s;bytes_=size_for(pr)+size_for(sr);check(cudaMalloc(&allocation_,bytes_),"allocate complete MTP state arena");std::byte* p=static_cast<std::byte*>(allocation_);
 auto bind=[&](PrefixStateView& v,std::size_t r){v.multi_hidden=take<__nv_bfloat16>(p,r*kMtpMultiHidden);v.main_key=take<__nv_bfloat16>(p,r*kQsaMainKvWidth);v.main_value=take<__nv_bfloat16>(p,r*kQsaMainKvWidth);v.raw_key=take<__nv_bfloat16>(p,r*kQsaIndexerWidth);v.compressed_key=take<__nv_bfloat16>(p,r*kQsaIndexerWidth);v.rope_positions=mr?take<std::int64_t>(p,r*kMropeAxes):nullptr;v.main_slots=take<std::int32_t>(p,r);v.raw_slots=take<std::int32_t>(p,r);v.compressed_slots=take<std::int32_t>(p,r);v.compressed_valid=take<std::int32_t>(p,r);};bind(prefixes_,pr);bind(selected_,sr);check(cudaMemset(allocation_,0,bytes_),"clear complete MTP state arena");}
StateArena::~StateArena(){if(allocation_)cudaFree(allocation_);}
PrefixStateView StateArena::prefix(int step)const{if(step<0||step>=depth_)throw StateArenaError("MTP prefix step is invalid");auto v=prefixes_;std::size_t r=std::size_t(step)*sequences_;v.multi_hidden+=r*kMtpMultiHidden;v.main_key+=r*kQsaMainKvWidth;v.main_value+=r*kQsaMainKvWidth;v.raw_key+=r*kQsaIndexerWidth;v.compressed_key+=r*kQsaIndexerWidth;if(v.rope_positions)v.rope_positions+=r*kMropeAxes;v.main_slots+=r;v.raw_slots+=r;v.compressed_slots+=r;v.compressed_valid+=r;return v;}
InactiveStateView StateArena::inactive(std::uint64_t g)const{if(!g||g!=pending_generation_)throw StateArenaError("MTP inactive generation is unavailable");return{selected_,sequences_,g};}
void StateArena::select(const std::int32_t* w,std::uint64_t g,cudaStream_t st){if(!w||!st||!g||pending_generation_)throw StateArenaError("MTP state selection contract changed");select_prefixes<<<sequences_,256,0,st>>>(prefixes_,selected_,w,sequences_,depth_,uses_mrope_);check(cudaGetLastError(),"select complete accepted MTP prefix");pending_generation_=g;}
void StateArena::commit(std::uint64_t g){if(!g||g!=pending_generation_)throw StateArenaError("MTP state commit generation changed");pending_generation_=0;}
void StateArena::discard(std::uint64_t g)noexcept{if(g&&g==pending_generation_)pending_generation_=0;}
}  // namespace rocket::qwen38::mtp
