from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

SCRIPT = Path(__file__).with_name("qwen38-layer3-moe-owner-preflight.py")
SPEC = importlib.util.spec_from_file_location("layer3_moe_owner_preflight", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class PreflightTests(unittest.TestCase):
    def descriptor(self, rank=0):
        return {
            "schema": "rocket.qwen38.layer3-native-plan.v1", "rank": rank,
            "artifact_key": module.ARTIFACT_KEY,
            "slab_key": f"rank{rank}-target",
            "slab_publication_layout_sha256": "a" * 64,
        }

    def test_descriptor_rejects_rank_artifact_and_layout_drift(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "plan.json"
            for field, value in (("rank", 1), ("artifact_key", "0" * 64),
                                 ("slab_publication_layout_sha256", "bad")):
                item = self.descriptor()
                item[field] = value
                path.write_text(json.dumps(item), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "identity"):
                    module._descriptor(path, 0)

    def test_supervisor_timeout_is_bounded_and_fail_closed(self):
        args = type("Args", (), {"rank": 0, "artifact": Path("a"),
                                  "native_plan": Path("p"), "library": Path("l"),
                                  "device_index": 0,
                                  "timeout_seconds": 30})()
        original = subprocess.run
        subprocess.run = lambda *a, **k: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(a[0], k["timeout"]))
        try:
            with mock.patch("builtins.print") as output:
                self.assertEqual(module.supervise(args), 124)
            record = json.loads(output.call_args.args[0])
            self.assertFalse(record["valid"])
            self.assertEqual(record["phase"], "timeout")
            self.assertEqual(record["kernel_launches"], 0)
        finally:
            subprocess.run = original

    def test_native_source_has_no_enqueue_or_source_wait(self):
        source = (SCRIPT.parents[2] / "engines/qwen38-flash-next-nvfp4-2b/src/moe/target_layer3_moe_owner_preflight.cc").read_text()
        self.assertNotIn("wait_source(", source)
        self.assertNotIn(".launch(", source)
        self.assertNotIn(".enqueue(", source)

    def test_preflight_dso_exports_only_bounded_control_abi(self):
        library = os.environ.get("ROCKET_QWEN38_LAYER3_MOE_PREFLIGHT_LIBRARY")
        if not library:
            self.skipTest("configured AOT preflight library was not supplied")
        symbols = subprocess.run(
            ["nm", "-D", "--defined-only", library], check=True,
            text=True, capture_output=True,
        ).stdout
        self.assertIn(" qwen38_target_layer3_moe_owner_preflight\n", symbols)
        self.assertIn(" qwen38_target_layer3_moe_owner_preflight_last_error\n", symbols)
        self.assertNotIn("enqueue", symbols)
        self.assertNotIn("launch", symbols)
        self.assertNotIn("wait_source", symbols)

    def test_supervisor_rejects_unbounded_child_output(self):
        args = type("Args", (), {"rank": 1, "artifact": Path("a"),
                                  "native_plan": Path("p"), "library": Path("l"),
                                  "device_index": 0, "timeout_seconds": 30})()
        result = subprocess.CompletedProcess([], 0, "noise\n", "secret stderr")
        with mock.patch.object(subprocess, "run", return_value=result), \
             mock.patch("builtins.print") as output:
            self.assertEqual(module.supervise(args), 1)
        record = json.loads(output.call_args.args[0])
        self.assertEqual(record["phase"], "child_result")
        self.assertNotIn("secret", output.call_args.args[0])

    def test_nested_loader_cause_is_bounded_and_survives_wrapper(self):
        inner = module.Layer3FactoryError("secret native detail /private/path")
        outer = module.CudaSlabLoadError("generic outer")
        outer.__cause__ = inner
        chain = module._typed_cause_chain(outer, "load")
        self.assertEqual(chain, (
            {"class": "slab_load", "stage": "accepted_loader"},
            {"class": "layer3_factory", "stage": "native_finalize"},
        ))
        self.assertNotIn("secret", json.dumps(chain))
        self.assertNotIn("private", json.dumps(chain))
        class Counter:
            def __init__(self): self.records = []
            def add(self, value, attributes): self.records.append((value, attributes))
        counter = Counter()
        module._emit_failure(counter, 0, "load", chain[-1])
        self.assertEqual(counter.records, [(1, {
            "rank": 0, "phase": "load", "outcome": "failure",
            "failure.class": "layer3_factory",
            "failure.stage": "native_finalize",
        })])


if __name__ == "__main__":
    unittest.main()
