// DFlash2 draft model for the CUDA engine: 5 dense qwen3-style layers with
// learned grouped conv (taps=2, group 16, block 8) and a codebook candidate
// selector. Loaded from the DFlash2 checkpoint (2.18 GiB BF16, single
// safetensors). Non-autoregressive: all k draft positions are masked
// (mask_token_id 154856) and proposed in one forward pass of width B*k.
//
// The draft has no embed_tokens and no lm_head: both come from the target.
#include "dflash2.h"

#include <fstream>
#include <stdexcept>

namespace rocket::engine {

void DFlash2Weights::load(const std::filesystem::path& ckpt_file) {
  // Parse the safetensors header and upload all tensors
  std::ifstream in(ckpt_file, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + ckpt_file.string());
  const std::size_t fsize = static_cast<std::size_t>(in.tellg());
  in.seekg(0);
  std::vector<char> raw(fsize);
  if (!in.read(raw.data(), fsize)) throw std::runtime_error("short read");
  in.close();

  std::uint64_t hl = 0;
  std::memcpy(&hl, raw.data(), 8);
  // Simple header parse: find "name":{"dtype":"BF16","shape":[...],"data_offsets":[s,e]}
  // For the DFlash2 checkpoint all tensors are BF16 and contiguous.
  // We upload them into a single flat device buffer with named offsets.
  std::string hdr(raw.data() + 8, hl);
  // ... (safetensors header parsing - reuse the fuel::Checkpoint approach)

  // For now: load via mmap like the fuel checkpoint
  // The DFlash2 weights are small (2.18 GiB) so a full read is acceptable.
  blob_ = std::move(raw);
  data_ = blob_.data() + 8 + hl;

  // Parse tensor names/shapes from the header JSON
  // (simplified: scan for known tensor names)
  // The 5 layers have identical structure; fc/hidden_norm/candidate_selector
  // are shared heads.
  // Full implementation: parse the safetensors JSON header, register each
  // tensor's offset and shape, and create named device pointers.
  // This follows the same pattern as the fuel Checkpoint class.
}

}  // namespace rocket::engine
