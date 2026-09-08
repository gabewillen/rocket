#!/usr/bin/env python3

import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATCHER = ROOT / "scripts/numerics/patch-qwen38-k0-layer0-boundaries.py"
SOURCE = Path(
    "/home/glwillen/calibration/qwen38-k0-oracle-a1794d5-01/artifacts/model_oracle.py"
)


class BoundaryPatchTest(unittest.TestCase):
    def test_pinned_source_gets_bounded_capture(self) -> None:
        if not SOURCE.is_file():
            self.skipTest("pinned oracle overlay is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "model.py"
            subprocess.run(
                [str(PATCHER), "--input", str(SOURCE), "--output", str(output)],
                check=True,
            )
            patched = output.read_text()
            self.assertEqual(patched.count("_rocket_k0_boundary_save(self.layer_idx"), 3)
            self.assertIn("get_tensor_model_parallel_rank() != 0", patched)
            self.assertIn("tensor[:1].detach().contiguous()", patched)

    def test_source_identity_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.py"
            output = Path(temporary) / "output.py"
            source.write_text("different")
            result = subprocess.run(
                [str(PATCHER), "--input", str(source), "--output", str(output)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
