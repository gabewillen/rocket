#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "attention" / "export-qwen38-gdn-b12x-aot.py"
SPEC = importlib.util.spec_from_file_location("gdn_b12x_aot", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class GdnB12xAotContract(unittest.TestCase):
    def test_fixed_workload_shapes_select_pinned_plan(self) -> None:
        plans = tuple(MODULE.select_plan(shape, 20) for shape in MODULE.SHAPES)
        self.assertEqual(
            tuple((plan.name, plan.tokens, plan.output_width) for plan in plans),
            (
                ("t300_qkvz", 300, 8192),
                ("t300_ba", 300, 48),
                ("t8192_qkvz", 8192, 8192),
                ("t8192_ba", 8192, 48),
            ),
        )
        self.assertTrue(
            all(
                (plan.input_width, plan.tile_m, plan.tile_n, plan.tile_k,
                 plan.swap_ab, plan.use_prefetch)
                == (2560, 128, 128, 128, False, False)
                for plan in plans
            )
        )

    def test_source_and_serving_abi_are_pinned(self) -> None:
        self.assertEqual(
            MODULE.PINNED_FLASHINFER_COMMIT,
            "91bda04c66f7cb851e1ab3b78b9fecea644b9844",
        )
        self.assertEqual(
            MODULE.PINNED_KERNEL_SHA256,
            "6739702e27afad21767b71024678b35c6892e3f9c74aaff375fb3f6734399f86",
        )


if __name__ == "__main__":
    unittest.main()
