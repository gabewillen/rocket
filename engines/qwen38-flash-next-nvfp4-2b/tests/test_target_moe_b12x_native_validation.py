#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "moe" / "validate-qwen38-target-moe-b12x-native.py"


class TargetMoeB12xNativeValidationContract(unittest.TestCase):
    def test_validation_harness_is_parseable_and_fixed_c1(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("graph = torch.cuda.CUDAGraph()", source)
        self.assertIn("with torch.cuda.graph(graph, stream=stream):", source)
        self.assertIn("physical_n != 768", source)
        self.assertIn("MoeShape(1)", source)
        self.assertIn("route_slots\": 10", source)

    def test_validation_path_uses_real_authenticated_weights_and_oracle(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("load_owner_local_moe", source)
        self.assertIn("materialize_flashinfer_weights", source)
        self.assertIn("FlashInferRoutedMoeBackend", source)
        self.assertIn("oracle_backend.routed_only", source)
        self.assertNotIn("torch.zeros(1, 2560", source)
        self.assertNotIn("synthetic", source.lower())

    def test_result_dimensions_are_bounded(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        for forbidden in ("global_ids.tolist", "routing_weights.tolist", "data_ptr()}"):
            if forbidden == "data_ptr()":
                continue
            self.assertNotIn(forbidden, source)
        self.assertIn('"rank": args.rank', source)
        self.assertIn('"layer": args.layer', source)


if __name__ == "__main__":
    unittest.main()
