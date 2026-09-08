#!/usr/bin/env python3
from pathlib import Path
import unittest


class TestGdnPrefillBundlePatcher(unittest.TestCase):
    def test_capture_is_real_prefill_only_and_content_addressed(self):
        source = Path(__file__).with_name(
            "patch-qwen38-gdn-prefill-bundle.py"
        ).read_text()
        self.assertIn('"q": ((35, 8, 128), torch.bfloat16)', source)
        self.assertIn('"initial_state": ((1, 24, 128, 128), torch.float32)', source)
        self.assertIn("q, k, v, fi_g, fi_beta, fi_state", source)
        self.assertIn("ROCKET_QWEN38_K0_GDN_CAPTURE_ACTIVE", source)
        self.assertIn("get_tensor_model_parallel_rank", source)
        self.assertIn("hashlib.sha256(canonical).hexdigest()", source)
        self.assertNotIn("repeat(", source)


if __name__ == "__main__":
    unittest.main()
