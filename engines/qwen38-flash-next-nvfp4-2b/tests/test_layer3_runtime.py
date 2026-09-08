# SPDX-License-Identifier: Apache-2.0
import unittest

from qwen38_slab.k0_composition import K0_DOMAIN
from qwen38_slab.layer3_runtime import (
    ARENA_SPECS,
    REQUIRED_LAYER3_DEPENDENCIES,
    Layer3RuntimeError,
    TwoRankLayer3Factory,
    allocate_layer3_qsa_storage,
)


class Tensor:
    def __init__(self, shape, dtype, device, fill=None):
        self.shape = shape; self.dtype = dtype; self.device = device
        self.fill_value = fill; self.values = {}
    def fill_(self, value): self.fill_value = value; return self
    def __setitem__(self, key, value): self.values[key] = value


class Torch:
    uint8 = "uint8"; bfloat16 = "bfloat16"; float32 = "float32"
    int32 = "int32"; int64 = "int64"
    @staticmethod
    def empty(shape, *, dtype, device): return Tensor(shape, dtype, device)
    @staticmethod
    def full(shape, fill, *, dtype, device): return Tensor(shape, dtype, device, fill)


class Port:
    execution_domain = K0_DOMAIN
    production_eligible = True
    graph_safe = True
    def __init__(self, rank=None): self.rank = rank; self.layer = 3
    def compare(self): pass
    def data_ptr(self): return 1
    def launch(self): pass
    def mix(self): pass
    def combine(self): pass
    def reduce(self): pass
    def enqueue(self): pass


class Slab(Port):
    dtype = "torch.uint8"
    device = "cuda:0"


class Layer3RuntimeTests(unittest.TestCase):
    def test_owns_exact_arena_and_advances_one_causal_row(self):
        storage = allocate_layer3_qsa_storage(Torch, device="cuda:0", max_tokens=31)
        self.assertEqual(set(storage.arena), set(ARENA_SPECS))
        self.assertEqual(storage.state["main_key_cache"].shape, (1, 1600, 256))
        self.assertEqual(storage.state["compressed_key_cache"].shape, (1, 400, 128))
        self.assertEqual(storage.advance(0), 1)
        self.assertEqual(storage.state["sequence_lengths"].fill_value, 1)
        with self.assertRaisesRegex(Layer3RuntimeError, "position/generation"):
            storage.advance(2)

    def test_two_rank_factory_names_every_missing_production_port(self):
        with self.assertRaises(Layer3RuntimeError) as caught:
            TwoRankLayer3Factory().bind({})
        self.assertEqual(caught.exception.missing, REQUIRED_LAYER3_DEPENDENCIES)
        dependencies = {
            name: Port(int(name[4]) if name.startswith("rank") else None)
            for name in REQUIRED_LAYER3_DEPENDENCIES
        }
        for name in dependencies:
            if name.endswith(("target_slab", "indexer_sidecar_slab")):
                dependencies[name] = Slab()
        binding = TwoRankLayer3Factory().bind(dependencies)
        self.assertEqual(tuple(binding.dependencies), REQUIRED_LAYER3_DEPENDENCIES)
        dependencies["rank1.shared_expert"] = object()
        with self.assertRaisesRegex(Layer3RuntimeError, "rank1.shared_expert"):
            TwoRankLayer3Factory().bind(dependencies)


if __name__ == "__main__":
    unittest.main()
