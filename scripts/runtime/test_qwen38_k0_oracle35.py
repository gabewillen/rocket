from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).with_name("qwen38-k0-oracle35.py")
SPEC = importlib.util.spec_from_file_location("qwen38_k0_oracle35", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class Oracle35LauncherTests(unittest.TestCase):
    def _descriptor_inventory(self, root: Path) -> None:
        for rank in (0, 1):
            for layer in range(48):
                descriptor = {
                    "schema": "test", "rank": rank, "layer": layer,
                    "attention_kind": "qsa" if layer % 4 == 3 else "gdn",
                    "native_binding_inventory_sha256": "",
                    "slab_publication_layout_sha256": "a" * 64,
                    "moe_layout_sha256": "b" * 64,
                    "extents": [],
                }
                descriptor["native_binding_inventory_sha256"] = module.hashlib.sha256(
                    module.canonical_bytes(descriptor["extents"])).hexdigest()
                descriptor["descriptor_sha256"] = module.hashlib.sha256(
                    module.canonical_bytes(descriptor)).hexdigest()
                (root / f"rank{rank}-layer{layer}.json").write_bytes(
                    module.canonical_bytes(descriptor) + b"\n")

    def test_complete_descriptor_inventory_is_consumed_without_regeneration(self):
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            self._descriptor_inventory(root)
            with mock.patch.object(module, "load_descriptor_allowlist",
                                   return_value=()), \
                 mock.patch.object(module, "authenticate_descriptor_identity",
                                   return_value=True) as authenticate:
                self.assertIsNone(module._descriptors(root))
            self.assertEqual(authenticate.call_count, 96)
            self.assertFalse(hasattr(module, "target_layer_descriptor"))

    def test_descriptor_inventory_rejects_missing_or_noncanonical_file(self):
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            self._descriptor_inventory(root)
            (root / "rank1-layer47.json").unlink()
            with self.assertRaisesRegex(ValueError, "inventory changed"):
                module._descriptors(root)
            self._descriptor_inventory(root)
            path = root / "rank1-layer47.json"
            path.write_bytes(path.read_bytes() + b"\n")
            with mock.patch.object(module, "load_descriptor_allowlist",
                                   return_value=()), \
                 mock.patch.object(module, "authenticate_descriptor_identity",
                                   return_value=True):
                with self.assertRaisesRegex(ValueError, "canonical encoding"):
                    module._descriptors(root)

    @unittest.skipUnless(
        os.environ.get("ROCKET_QWEN38_K0_RUNTIME_LIBRARY"),
        "configured K0 runtime library was not supplied",
    )
    def test_runtime_rejects_capability_not_minted_by_its_wrapper(self):
        args = argparse.Namespace(
            library=Path(os.environ["ROCKET_QWEN38_K0_RUNTIME_LIBRARY"]),
            device_index=0, rank=0, descriptor_directory=Path("d"),
            sidecar_payload=Path("s"), tokenizer=Path("t"),
            oracle_capture=Path("o"), bootstrap_host="192.0.2.1",
            layer_port=18838, embedding_port=18839, nccl_port=18840,
            timeout_ms=120000,
        )
        with self.assertRaises(module.NativeRunStatusError) as raised:
            module._native_run(args, ctypes.c_void_p(0x1234),
                               (b"a" * 32, b"b" * 32,
                                b"c" * 32, b"d" * 32))
        self.assertEqual(raised.exception.stage, "validation")

    def test_native_status_mapping_is_closed_and_bounded(self):
        self.assertEqual(set(module.NATIVE_STATUS_STAGES), {
            10, 20, 21, 22, 30, 31, 32, 40, 41, 42, 43, 50, 51, 255,
        })
        for status, stage in module.NATIVE_STATUS_STAGES.items():
            error = module.NativeRunStatusError(status)
            self.assertEqual(error.stage, stage)
            self.assertEqual(module._typed_cause_chain(error, "native"), (
                {"class": "native_run", "stage": stage},
            ))
        unknown = module.NativeRunStatusError(999)
        self.assertEqual(unknown.stage, "unknown")

    def test_physical_layer_substage_and_layer_mapping_is_closed(self):
        self.assertEqual(set(module.PHYSICAL_LAYER_SUBSTAGES), set(range(8)))
        for code, name in module.PHYSICAL_LAYER_SUBSTAGES.items():
            error = module.NativeRunStatusError(31, code, 47)
            self.assertEqual(error.physical_substage, name)
            expected_layer = 47 if name in ("gdn_owner", "qsa_owner") else -1
            self.assertEqual(error.physical_layer, expected_layer)
        self.assertEqual(
            module.NativeRunStatusError(31, 999, 47).physical_substage,
            "unknown",
        )
        self.assertEqual(module.NativeRunStatusError(31, 5, 48).physical_layer,
                         -1)

    def test_secret_requires_exact_32_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "secret"
            path.write_bytes(b"x" * 31)
            with self.assertRaisesRegex(ValueError, "extent"):
                module._secret(path)
            path.write_bytes(b"x" * 32)
            self.assertEqual(module._secret(path), b"x" * 32)

    def test_nested_slab_cause_is_bounded_and_published(self):
        inner = module.SlabError("private artifact path")
        outer = module.CudaSlabLoadError("outer loader wrapper")
        outer.__cause__ = inner
        chain = module._typed_cause_chain(outer, "load")
        self.assertEqual(chain, (
            {"class": "slab_load", "stage": "accepted_loader"},
            {"class": "slab_contract", "stage": "accepted_loader_contract"},
        ))
        self.assertNotIn("private", json.dumps(chain))

        class Counter:
            def __init__(self): self.records = []
            def add(self, value, attributes):
                self.records.append((value, attributes))

        counter = Counter()
        module._emit_failure(counter, 1, "load", chain[-1], outer)
        self.assertEqual(counter.records, [(1, {
            "rank": 1, "phase": "load", "outcome": "failure",
            "failure.class": "slab_contract",
            "failure.stage": "accepted_loader_contract",
        })])

    def test_supervisor_timeout_is_bounded(self):
        args = argparse.Namespace(
            worker=False, timeout_seconds=60, rank=0, device_index=0,
            artifact=Path("a"), sidecar=Path("s"), sidecar_payload=Path("p"),
            descriptor_directory=Path("d"), tokenizer=Path("t"),
            oracle_capture=Path("o"), library=Path("l"),
            bootstrap_host="192.0.2.1", layer_port=1, embedding_port=2,
            nccl_port=3, timeout_ms=120000, layer_session_file=Path("ls"),
            embedding_session_file=Path("es"), nccl_session_file=Path("ns"),
            nccl_authentication_key_file=Path("nk"),
        )
        with mock.patch.object(subprocess, "run", side_effect=
                               subprocess.TimeoutExpired([], 60)), \
             mock.patch("builtins.print") as output:
            self.assertEqual(module.supervise(args), 124)
        record = json.loads(output.call_args.args[0])
        self.assertEqual(record["phase"], "timeout")
        self.assertNotIn("session", record)

    def test_native_result_snapshot_has_only_bounded_fields(self):
        result = module._NativeResult(token=module.EXPECTED_TOKEN, rows=35,
                                      final_generation=35)
        self.assertEqual(set(module._snapshot(result)), {
            "token", "rows", "final_generation", "lifecycle_outcomes",
            "moe_components", "stage_counters", "state_outcomes",
            "nccl_stages", "nccl_outcomes",
            "duration_samples", "total_bytes",
            "physical_layer_substage",
        })


if __name__ == "__main__":
    unittest.main()
