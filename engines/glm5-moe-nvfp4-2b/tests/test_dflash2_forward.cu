#include "dflash2.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

int main() {
  const char* dir = std::getenv("ROCKET_DFLASH2_DIR");
  if (!dir) { std::fprintf(stderr, "set ROCKET_DFLASH2_DIR\n"); return 77; }
  __nv_bfloat16 *table=nullptr, *aux=nullptr;
  const std::size_t table_elems=static_cast<std::size_t>(154880)*4096;
  if (cudaMalloc(&table,table_elems*sizeof(*table)) != cudaSuccess ||
      cudaMalloc(&aux,static_cast<std::size_t>(5)*16*4096*sizeof(*aux)) != cudaSuccess) return 2;
  cudaMemset(table,0,table_elems*sizeof(*table));
  cudaMemset(aux,0,static_cast<std::size_t>(5)*16*4096*sizeof(*aux));
  try {
    rocket::engine::DFlash2DraftEngine draft(dir,table,table,2,128,7);
    draft.append_context(aux,16,1,2,{0,0},{1,1},nullptr);
    std::vector<int> out;
    draft.propose({1},{1},1,7,out,nullptr);
    if(out.size()!=7) return 3;
    for(int id:out) if(id<0 || id>=154880) return 3;
    std::vector<int> compact;
    draft.propose_slots({1},{1},{1},7,compact,nullptr);
    if(compact.size()!=7) return 3;
    for(int id:compact) if(id<0 || id>=154880) return 3;
    std::printf("PASS: full and compact-slot DFlash2 forward tokens=%d,%d\n",out[0],compact[0]);
  } catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
  cudaFree(aux); cudaFree(table);
  return 0;
}
