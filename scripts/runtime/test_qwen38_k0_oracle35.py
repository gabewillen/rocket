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
        with self.assertRaisesRegex(RuntimeError, "native_status_1"):
            module._native_run(args, ctypes.c_void_p(0x1234),
                               (b"a" * 32, b"b" * 32,
                                b"c" * 32, b"d" * 32))

    def test_secret_requires_exact_32_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "secret"
            path.write_bytes(b"x" * 31)
            with self.assertRaisesRegex(ValueError, "extent"):
                module._secret(path)
            path.write_bytes(b"x" * 32)
            self.assertEqual(module._secret(path), b"x" * 32)

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
        })


if __name__ == "__main__":
    unittest.main()
