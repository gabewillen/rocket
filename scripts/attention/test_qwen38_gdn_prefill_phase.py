#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import ast
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).with_name("qwen38-gdn-prefill-phase.py")


def load_functions(*names):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    return tuple(namespace[name] for name in names)


class GdnPrefillPhaseContract(unittest.TestCase):
    def test_projection_contract_names_scope_and_counts(self) -> None:
        (contract,) = load_functions("projection_contract")
        result = contract(300, 5, 50)
        self.assertEqual(result["engine"], "vllm")
        self.assertEqual(result["physical_mnk"], [[300, 8192, 2560], [300, 64, 2560]])
        self.assertEqual(result["activation_quantizations"], 2)
        self.assertEqual(result["gemms"], 2)
        self.assertEqual(result["timer"], "cuda_events_around_graph_replay")
        self.assertEqual(result["gpu_clocks"], "unlocked")

    def test_sm120_tactic_table_decodes_fallback_and_tactic_22(self) -> None:
        (decode,) = load_functions("sm120_cutlass_tactic")
        self.assertEqual(decode(-1)["tile_mnk"], [128, 128, 256])
        self.assertEqual(decode(-1)["scheduler"], "dp_static_persistent")
        self.assertFalse(decode(-1)["swap_ab"])
        self.assertEqual(decode(22)["tile_mnk"], [128, 128, 256])
        self.assertEqual(decode(22)["scheduler"], "stream_k")
        self.assertTrue(decode(22)["swap_ab"])

    def test_unknown_tactic_fails_closed(self) -> None:
        (decode,) = load_functions("sm120_cutlass_tactic")
        with self.assertRaises(RuntimeError):
            decode(32)

    def test_both_projection_measurements_are_attributed(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "attribute_projection_backend"
        ]
        attributed_results = {
            call.args[0].id
            for call in calls
            if call.args and isinstance(call.args[0], ast.Name)
        }
        self.assertEqual(attributed_results, {"input_result", "output_result"})

    def test_results_preserve_samples_and_power_snapshot(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"samples_us": samples', source)
        self.assertIn('"gpu_clock_power": gpu_clock_power_snapshot()', source)
        self.assertIn('input_result["data_identity"]', source)
        self.assertIn('"profiler_sha256": sha256_file(script_path)', source)
        self.assertNotIn('result["nvfp4_kernel"]', source)

    def test_fixture_contract_preserves_distinct_linear_inputs(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        for name in (
            "hidden.bin", "qkvz_a.bin", "qkvz_sfa.bin", "ba_a.bin",
            "ba_sfa.bin", "qkvz_b.bin", "qkvz_sfb.bin", "ba_b.bin",
            "ba_sfb.bin", "alpha.bin",
        ):
            self.assertIn(f'"{name}"', source)
        self.assertIn('"format": "rocket-gdn-fp4-fixture-v1"', source)
        self.assertIn('"provenance": "python-synthetic-seed-7"', source)


if __name__ == "__main__":
    unittest.main()
