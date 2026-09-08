# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/runtime/qwen38-layer3-pairreduce-preflight.py"
ORACLE = Path("/home/glwillen/calibration/qwen38-k0-oracle-a1794d5-01/capture/manifest.json")


class PairReducePreflightTests(unittest.TestCase):
    def test_fake_two_rank_topology_and_identity(self):
        if not ORACLE.is_file():
            self.skipTest("authenticated oracle unavailable")
        with tempfile.TemporaryDirectory() as directory:
            sysfs = Path(directory)
            for rail in ("rocep1s0f1", "roceP2p1s0f1"):
                port = sysfs / rail / "ports" / "1"
                (port / "gids").mkdir(parents=True)
                (port / "state").write_text("4: ACTIVE\n")
                (port / "gids" / "3").write_text("fe80::1\n")
            for rank in (0, 1):
                result = subprocess.run([
                    "python3", str(SCRIPT), "--rank", str(rank),
                    "--peer-rank", str(1 - rank), "--bootstrap-host",
                    "192.168.100.10", "--bootstrap-port", "18839",
                    "--oracle-manifest", str(ORACLE),
                    "--infiniband-sysfs", str(sysfs),
                ], check=False, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout)
                record = json.loads(result.stdout)
                self.assertEqual(record["rank"], rank)
                self.assertEqual(record["calls"], 70)
                self.assertEqual(record["cuda_launches"], 0)
            (sysfs / "rocep1s0f1/ports/1/state").write_text("1: DOWN\n")
            red = subprocess.run([
                "python3", str(SCRIPT), "--rank", "0", "--peer-rank", "1",
                "--bootstrap-host", "192.168.100.10", "--bootstrap-port", "18839",
                "--oracle-manifest", str(ORACLE), "--infiniband-sysfs", str(sysfs),
            ], check=False, capture_output=True, text=True)
            self.assertEqual(red.returncode, 1)
            self.assertFalse(json.loads(red.stdout)["valid"])


if __name__ == "__main__":
    unittest.main()
