import hashlib
import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/runtime/qwen38-layer3-comparator.py"
REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"


class Layer3ComparatorTests(unittest.TestCase):
    def fixture(self, root: Path) -> Path:
        capture = root / "capture"
        capture.mkdir()
        names = [
            "embedding",
            *[f"layer.{i:02d}" for i in range(48)],
            "final_norm",
            "logits",
        ]
        artifacts = []
        for index, name in enumerate(names):
            shape = (
                [1, 248320]
                if name == "logits"
                else ([1, 2560] if name in ("embedding", "final_norm")
                      else [1, 10240])
            )
            value = 0.25 if name == "layer.03" else index / 128.0
            bits = struct.unpack("<I", struct.pack("<f", value))[0] >> 16
            payload = struct.pack("<H", bits) * (shape[0] * shape[1])
            file = name.replace(".", "-") + ".bin"
            (capture / file).write_bytes(payload)
            artifacts.append({"name": name, "file": file, "dtype": "bfloat16",
                              "shape": shape, "strides": [shape[1], 1],
                              "numel": shape[0] * shape[1], "bytes": len(payload),
                              "sha256": hashlib.sha256(payload).hexdigest()})
        manifest = {"schema": "rocket.qwen38.k0-target-oracle.v1", "valid": True,
                    "complete": True, "identity": {
                        "model_revision": REVISION,
                        "model": "nvidia/Qwen3.8-Flash-Next-NVFP4",
                        "tensor_parallel_size": 2,
                        "node_count": 2,
                        "speculation": "disabled",
                    }, "input_token_ids": [7], "greedy_token_id": 7,
                    "artifacts": artifacts}
        (capture / "manifest.json").write_text(json.dumps(manifest))
        return capture

    def prepare_command(self, capture, contract):
        digest = hashlib.sha256((capture / "manifest.json").read_bytes()).hexdigest()
        return [
            "python3", str(SCRIPT), "prepare", "--capture-dir", str(capture),
            "--output", str(contract), "--expected-manifest-sha256", digest,
            "--expected-greedy-token", "7", "--expected-tokens", "1",
        ]

    def test_prepares_exact_layer_pair_and_compares_bf16(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); capture = self.fixture(root); contract = root / "slice.json"
            prepared = subprocess.run(
                self.prepare_command(capture, contract), capture_output=True
            )
            self.assertEqual(prepared.returncode, 0, prepared.stderr.decode())
            record = json.loads(contract.read_text())
            self.assertEqual(record["entry_state"], "materialized_post_layer_02")
            self.assertEqual(record["oracle_artifact_count"], 51)
            self.assertEqual(record["oracle_extents"]["logits"], [1, 248320])
            observed = root / "observed.bin"
            observed.write_bytes((capture / "layer-03.bin").read_bytes())
            compared = subprocess.run(["python3", str(SCRIPT), "compare", "--contract",
                                       str(contract), "--observed", str(observed)], capture_output=True)
            self.assertEqual(compared.returncode, 0, compared.stdout.decode())
            self.assertTrue(json.loads(compared.stdout)["valid"])

    def test_rejects_missing_oracle_and_numeric_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); missing = root / "missing"; missing.mkdir()
            rejected = subprocess.run(["python3", str(SCRIPT), "prepare", "--capture-dir",
                                       str(missing), "--output", str(root / "slice.json")], capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("51-artifact oracle manifest is absent", rejected.stdout.decode())
            capture = self.fixture(root); contract = root / "slice.json"
            subprocess.run(self.prepare_command(capture, contract), check=True)
            observed = root / "observed.bin"
            observed.write_bytes(b"\x00\x00" * 10240)
            mismatch = subprocess.run(["python3", str(SCRIPT), "compare", "--contract",
                                       str(contract), "--observed", str(observed),
                                       "--atol", "0", "--rtol", "0"], capture_output=True)
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertGreater(json.loads(mismatch.stdout)["mismatches"], 0)

    def test_rejects_non_layer_extent_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = self.fixture(root)
            manifest_path = capture / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"][0]["shape"] = [1, 2559]
            manifest_path.write_text(json.dumps(manifest))
            result = subprocess.run(
                self.prepare_command(capture, root / "slice.json"),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("oracle artifact extent changed: embedding", result.stdout)


if __name__ == "__main__":
    unittest.main()
