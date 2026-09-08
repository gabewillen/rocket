# SPDX-License-Identifier: Apache-2.0
import unittest
from pathlib import Path

from qwen38_slab.native_qsa import (
    ARENA_FIELDS,
    STATE_POINTER_FIELDS,
    STATE_DTYPES,
    NativeQsaBindingError,
    bind_slab_pointers,
    bind_state_view,
)
from qwen38_slab.qsa_weights import load_qsa_weights

REAL_SLAB = Path(
    "/home/glwillen/calibration/qwen38-rank-slabs-fc694/"
    "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4"
)
SIDECAR = Path(
    "/home/glwillen/calibration/qwen38-rank-slabs-fc694/qsa-indexer-sidecars/"
    "bdbebd4f45c398f090a41ab98cd3881b969d958d8ae0bc42f3411844d3262edd"
)


class Tensor:
    def __init__(self, pointer, elements, dtype="torch.uint8"):
        self._pointer = pointer
        self._elements = elements
        self.dtype = dtype
        self.device = "cuda:0"

    def data_ptr(self):
        return self._pointer

    def numel(self):
        return self._elements


class NativeQsaBindingTests(unittest.TestCase):
    def test_binds_exact_c1_state_generation(self):
        tensors = {
            name: Tensor(
                0x500000000 + index * 0x10000,
                1,
                f"torch.{STATE_DTYPES[name]}",
            )
            for index, name in enumerate(STATE_POINTER_FIELDS)
        }
        state = bind_state_view(
            tensors, rank=1, layer=47, generation=9, main_blocks=2,
            compressed_blocks=1, compression_work_items=1,
        )
        self.assertEqual((state.rows, state.rank, state.layer), (1, 1, 47))
        self.assertEqual((state.generation, state.expected_generation), (9, 9))
        with self.assertRaisesRegex(NativeQsaBindingError, "inventory"):
            bind_state_view(
                {}, rank=1, layer=47, generation=9, main_blocks=2,
                compressed_blocks=1, compression_work_items=1,
            )

    @unittest.skipUnless(REAL_SLAB.is_dir() and SIDECAR.is_dir(), "QSA slabs absent")
    def test_resolves_authenticated_base_and_sidecar_offsets(self):
        descriptor = load_qsa_weights(REAL_SLAB, SIDECAR, 0, 3)
        base, sidecar = 0x100000000, 0x200000000
        arena = {
            name: Tensor(0x300000000 + index * 0x10000, 1, "torch.bfloat16")
            for index, name in enumerate(ARENA_FIELDS)
        }
        pointers = bind_slab_pointers(
            descriptor, Tensor(base, descriptor.slab_bytes),
            Tensor(sidecar, descriptor.indexer_sidecar_path.stat().st_size),
            Tensor(0x400000000, 262144 * 64, "torch.bfloat16"), arena,
        )
        components = {
            item.name.split(".self_attn.", 1)[1]: item
            for item in descriptor.components
        }
        self.assertEqual(
            pointers.projection.q_weight,
            base + components["q_proj.weight"].offset,
        )
        self.assertEqual(
            pointers.preprocess.index_qk,
            sidecar + components["indexer.index_qk_proj.weight"].offset,
        )
        self.assertEqual(pointers.arena.projected_output, arena["projected_output"].data_ptr())

    @unittest.skipUnless(REAL_SLAB.is_dir() and SIDECAR.is_dir(), "QSA slabs absent")
    def test_rejects_incomplete_caller_owned_arena(self):
        descriptor = load_qsa_weights(REAL_SLAB, SIDECAR, 0, 3)
        with self.assertRaisesRegex(NativeQsaBindingError, "arena inventory"):
            bind_slab_pointers(
                descriptor, Tensor(1, descriptor.slab_bytes),
                Tensor(2, descriptor.indexer_sidecar_path.stat().st_size),
                Tensor(3, 1, "torch.bfloat16"), {},
            )


if __name__ == "__main__":
    unittest.main()
