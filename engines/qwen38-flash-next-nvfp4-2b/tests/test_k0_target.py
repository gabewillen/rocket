from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from qwen38_slab.k0_target import (
    K0_TARGET_SCHEMA,
    TARGET_BINDING_BYTES,
    TARGET_OUTPUT_BYTES,
    TARGET_ROWS,
    K0TargetBinding,
    K0TargetError,
    K0TargetKernel,
)


class K0TargetContractTests(unittest.TestCase):
    def test_model_interface_is_fixed_for_tp2_k0(self):
        self.assertEqual(
            K0_TARGET_SCHEMA,
            "qwen3.8-flash-next:tp2:k0-target-prologue:v1",
        )
        self.assertEqual(TARGET_BINDING_BYTES, 16)
        self.assertEqual(TARGET_ROWS, 16)
        self.assertEqual(TARGET_OUTPUT_BYTES, 512)

    def test_unbound_slab_interface_is_immutable(self):
        binding = K0TargetBinding()
        self.assertEqual(
            (binding.slab_weight_table, binding.slab_weight_generation), (0, 0)
        )
        with self.assertRaises(FrozenInstanceError):
            binding.slab_weight_table = 1

    def test_slab_binding_accepts_only_unsigned_64_bit_words(self):
        for value in (-1, 1 << 64, True, "1"):
            with self.subTest(value=value), self.assertRaises(K0TargetError):
                K0TargetBinding(value, 0)
            with self.subTest(value=value), self.assertRaises(K0TargetError):
                K0TargetBinding(0, value)

    def test_v1_rejects_weights_until_a_kernel_consumes_them(self):
        for values in ((1, 1), (1, 0), (0, 1)):
            with self.subTest(values=values), self.assertRaisesRegex(
                K0TargetError, "reserved"
            ):
                K0TargetBinding(*values)

    def test_kernel_library_paths_are_explicit(self):
        with self.assertRaisesRegex(K0TargetError, "explicit Path"):
            K0TargetKernel(nvrtc="libnvrtc.so", driver=Path("libcuda.so"))
        with self.assertRaisesRegex(K0TargetError, "explicit Path"):
            K0TargetKernel(nvrtc=Path("libnvrtc.so"), driver="libcuda.so")


if __name__ == "__main__":
    unittest.main()
