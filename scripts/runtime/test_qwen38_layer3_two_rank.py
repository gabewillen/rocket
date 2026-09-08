import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/runtime/qwen38-layer3-two-rank.py"


class TwoRankLayer3ExecutableTests(unittest.TestCase):
    def test_names_every_missing_concrete_dependency(self):
        result = subprocess.run(
            ["python3", str(SCRIPT)], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 1)
        record = json.loads(result.stdout)
        self.assertEqual(record["phase"], "bind")
        self.assertEqual(len(record["missing"]), 19)
        self.assertIn("rank0.target_router", record["missing"])
        self.assertIn("rank1.shared_expert", record["missing"])
        self.assertIn("oracle_comparator", record["missing"])


if __name__ == "__main__":
    unittest.main()
