#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import ast
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "moe" / "validate-qwen38-target-moe-b12x-native.py"
SPEC = importlib.util.spec_from_file_location("target_moe_native_validation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


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

    def test_static_native_alpha_matches_pinned_wrapper_fold(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("folded_w1_alpha", source)
        self.assertIn("views.w1_alpha * source_weights.input_scale", source)
        self.assertIn("pointer(folded_w1_alpha)", source)
        self.assertNotIn("pointer(views.w1_alpha)", source)

    def test_result_dimensions_are_bounded(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        for forbidden in ("global_ids.tolist", "routing_weights.tolist", "data_ptr()}"):
            if forbidden == "data_ptr()":
                continue
            self.assertNotIn(forbidden, source)
        self.assertIn('"rank": args.rank', source)
        self.assertIn('"layer": args.layer', source)

    def test_create_failure_telemetry_names_first_invalid_field(self) -> None:
        record = MODULE.create_failure_record(
            status=1, diagnostic=2, rank=0, layer=0,
        )
        self.assertEqual(record["first_invalid_field"], "artifact_sha256")
        self.assertEqual(record["failure_class"], "contract")
        self.assertFalse(record["accepted"])
        self.assertEqual(
            set(record),
            {"abi", "accepted", "failure_class", "first_invalid_field",
             "layer", "phase", "rank", "status"},
        )
        self.assertEqual(len(MODULE.CREATE_FAILURE_FIELDS), 14)

    def test_create_diagnostic_precedes_create_and_failure_raise(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        diagnose = source.index(
            "rocket_qwen38_target_moe_b12x_diagnose_create(",
            source.index("def main"),
        )
        create = source.index(
            "rocket_qwen38_target_moe_b12x_create(", diagnose,
        )
        emit = source.index("print(json.dumps(create_failure_record(", create)
        failure = source.index("raise RuntimeError", emit)
        self.assertLess(diagnose, create)
        self.assertLess(create, emit)
        self.assertLess(emit, failure)

    def test_content_addressed_mount_basename_is_mandatory(self) -> None:
        payload = {"schema": "test"}
        claimed = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        payload["artifact_key"] = claimed
        with tempfile.TemporaryDirectory() as directory:
            wrong = pathlib.Path(directory) / "artifact"
            wrong.mkdir()
            (wrong / "manifest.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "mount basename"):
                MODULE.validate_artifact_mount_identity(wrong, claimed)

    def test_red_mount_name_is_rejected_before_torch_import(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        identity = source.index("artifact_key = validate_artifact_mount_identity")
        torch_import = source.index("import torch", identity)
        self.assertLess(identity, torch_import)
        self.assertIn("--preflight-only", source)


if __name__ == "__main__":
    unittest.main()
