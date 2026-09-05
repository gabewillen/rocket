// Remaining software levers on the read path: cache-hint variants.
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#define CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){printf("err %s %d\n",cudaGetErrorString(e),__LINE__);exit(1);} }while(0)

// plain
__global__ void k_plain(const float4* __restrict__ a, size_t n4, float* out){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, s=(size_t)gridDim.x*blockDim.x; float acc=0;
  for(;i<n4;i+=s){float4 v=a[i];acc+=v.x+v.y+v.z+v.w;}
  if(acc==1.2345e-30f)out[0]=acc;
}
// __ldcs: streaming, evict-first
__global__ void k_ldcs(const float4* __restrict__ a, size_t n4, float* out){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, s=(size_t)gridDim.x*blockDim.x; float acc=0;
  for(;i<n4;i+=s){float4 v=__ldcs(&a[i]);acc+=v.x+v.y+v.z+v.w;}
  if(acc==1.2345e-30f)out[0]=acc;
}
// __ldg: non-coherent read-only path
__global__ void k_ldg(const float4* __restrict__ a, size_t n4, float* out){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, s=(size_t)gridDim.x*blockDim.x; float acc=0;
  for(;i<n4;i+=s){float4 v=__ldg(&a[i]);acc+=v.x+v.y+v.z+v.w;}
  if(acc==1.2345e-30f)out[0]=acc;
}
// __ldlu: last-use
__global__ void k_ldlu(const float4* __restrict__ a, size_t n4, float* out){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, s=(size_t)gridDim.x*blockDim.x; float acc=0;
  for(;i<n4;i+=s){float4 v; asm volatile("ld.global.lu.v4.f32 {%0,%1,%2,%3}, [%4];"
      : "=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w) : "l"(&a[i])); acc+=v.x+v.y+v.z+v.w;}
  if(acc==1.2345e-30f)out[0]=acc;
}
__global__ void kfill(float4* a,size_t n4,float v){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x,s=(size_t)gridDim.x*blockDim.x;
  for(;i<n4;i+=s)a[i]=make_float4(v,v,v,v);
}
int main(){
  size_t bytes=(size_t)8<<30, n4=bytes/sizeof(float4);
  cudaDeviceProp p;CHECK(cudaGetDeviceProperties(&p,0));
  int blocks=p.multiProcessorCount*32, th=1024;
  float4*a;float*out;CHECK(cudaMalloc(&a,bytes));CHECK(cudaMalloc(&out,4));
  kfill<<<1024,256>>>(a,n4,1.f);CHECK(cudaDeviceSynchronize());
  cudaEvent_t e0,e1;CHECK(cudaEventCreate(&e0));CHECK(cudaEventCreate(&e1));
  auto go=[&](const char*nm,void(*k)(const float4*,size_t,float*)){
    std::vector<double>ms;
    for(int r=0;r<10;r++){CHECK(cudaEventRecord(e0));k<<<blocks,th>>>(a,n4,out);
      CHECK(cudaEventRecord(e1));CHECK(cudaEventSynchronize(e1));
      float t;CHECK(cudaEventElapsedTime(&t,e0,e1));ms.push_back(t);}
    CHECK(cudaGetLastError());std::sort(ms.begin(),ms.end());
    printf("%-8s best %.1f GB/s  median %.1f GB/s\n",nm,bytes/(ms.front()*1e6),bytes/(ms[5]*1e6));
  };
  go("plain",k_plain); go("ldcs",k_ldcs); go("ldg",k_ldg); go("ldlu",k_ldlu);
  return 0;
}
