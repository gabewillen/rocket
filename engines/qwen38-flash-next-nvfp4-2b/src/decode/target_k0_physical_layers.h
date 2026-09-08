// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/target_k0_layer_owner_inventory.h"

namespace rocket::qwen38::decode {

// Lower-level lifetime aggregate boundary. The startup root owns this object
// before constructing its executor and borrows only the fixed inventory.
// Production CUDA construction lives in the AOT-enabled owner library.
class TargetK0PhysicalLayers {
 public:
  virtual ~TargetK0PhysicalLayers() = default;
  virtual int rank() const noexcept = 0;
  virtual bool authenticated() const noexcept = 0;
  virtual TargetK0LayerOwnerInventory& inventory() noexcept = 0;
};

}  // namespace rocket::qwen38::decode
