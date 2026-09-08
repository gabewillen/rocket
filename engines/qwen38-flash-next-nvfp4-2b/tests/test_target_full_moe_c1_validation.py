#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import ast
import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/moe/validate-qwen38-target-full-moe-c1.py"
SPEC = importlib.util.spec_from_file_location("target_full_moe_validation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TargetFullMoeC1ValidationContract(unittest.TestCase):
    def test_harness_uses_real_router_routed_and_shared_oracles(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        ast.parse(source)
        for required in (
            "load_owner_local_moe", "materialize_flashinfer_weights",
            "reconstruct_router", "backend.routed_only", "backend.shared_partial",
            "torch.cuda.CUDAGraph", "rocket_qwen38_target_full_moe_c1_enqueue",
        ):
            self.assertIn(required, source)
        self.assertNotIn("torch.zeros", source)
        self.assertNotIn("synthetic", source.lower())

    def test_failure_telemetry_has_only_bounded_dimensions(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        begin = source.index("def emit_failure")
        end = source.index("def read_router_tensors", begin)
        failure = source[begin:end]
        for forbidden in ("artifact", "sha256", "pointer", "request"):
            self.assertNotIn(forbidden, failure)
        for required in ("phase", "status", "rank", "layer", "failure_class"):
            self.assertIn(required, failure)

    def test_ctypes_embeds_all_caller_owned_workspaces(self) -> None:
        names = [name for name, _ in MODULE.FullWorkspace._fields_]
        self.assertEqual(len(names), 12)
        self.assertEqual(names[0], "router_logits_f32")
        self.assertIn("routed", names)
        self.assertEqual(names[-1], "shared_gate_scalar_f32")
        self.assertEqual(MODULE.ctypes.sizeof(MODULE.FullWeights), 120)
        self.assertEqual(MODULE.ctypes.sizeof(MODULE.FullWorkspace), 184)
        self.assertEqual(MODULE.ctypes.sizeof(MODULE.FullLaunch), 208)


if __name__ == "__main__":
    unittest.main()
