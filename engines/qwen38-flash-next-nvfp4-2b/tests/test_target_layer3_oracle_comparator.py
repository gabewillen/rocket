import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAPTURE = Path("/home/glwillen/calibration/qwen38-k0-oracle-a1794d5-01/capture")
BUILD = ROOT / "engines/qwen38-flash-next-nvfp4-2b/build-layer3-oracle"
AUTH = BUILD / "qwen38-target-layer3-oracle-auth"


@unittest.skipUnless(CAPTURE.is_dir() and AUTH.is_file(), "oracle/build unavailable")
class TargetLayer3OracleComparatorTests(unittest.TestCase):
    def test_real_oracle_and_ulp_boundary(self):
        subprocess.run([AUTH, CAPTURE], check=True)

    def test_manifest_layer_and_symlink_mutations_fail_closed(self):
        for mutation in ("manifest", "layer", "layer_symlink", "manifest_symlink", "capture_symlink"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                capture = Path(tmp) / "capture"
                if mutation == "capture_symlink":
                    capture.symlink_to(CAPTURE, target_is_directory=True)
                    self.assertEqual(subprocess.run([AUTH, capture], check=False).returncode, 1)
                    continue
                capture.mkdir()
                if mutation == "manifest_symlink":
                    (capture / "manifest.json").symlink_to(CAPTURE / "manifest.json")
                else:
                    shutil.copyfile(CAPTURE / "manifest.json", capture / "manifest.json")
                if mutation == "layer_symlink":
                    (capture / "layer-03.bin").symlink_to(CAPTURE / "layer-03.bin")
                else:
                    shutil.copyfile(CAPTURE / "layer-03.bin", capture / "layer-03.bin")
                if mutation in ("manifest", "layer"):
                    target = capture / ("manifest.json" if mutation == "manifest" else "layer-03.bin")
                    data = bytearray(target.read_bytes()); data[-1] ^= 1; target.write_bytes(data)
                result = subprocess.run([AUTH, capture], check=False)
                self.assertEqual(result.returncode, 1)

    def test_source_keeps_exact_bounded_validation_surface(self):
        source = (ROOT / "engines/qwen38-flash-next-nvfp4-2b/src/decode/target_layer3_oracle_comparator.cc").read_text()
        header = (ROOT / "engines/qwen38-flash-next-nvfp4-2b/src/decode/target_layer3_oracle_comparator.h").read_text()
        self.assertIn("O_NOFOLLOW", source)
        self.assertIn("hash_bytes(row.data(),kLayer3OracleRowBytes)", source.replace(" ", ""))
        self.assertIn("copy_d2h(observed_,device,kLayer3OracleRowBytes", source)
        self.assertIn("2*kLayer3OracleRowBytes", source)
        self.assertIn("e.max_ulp<=1", source.replace(" ", ""))
        self.assertIn("Validation-only", header)
        self.assertNotIn("cudaGraph", source + header)


if __name__ == "__main__":
    unittest.main()
