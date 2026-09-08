#!/usr/bin/env python3
from pathlib import Path
import unittest


class TestGdnBoundaryPatcher(unittest.TestCase):
    def test_capture_is_fixed_first_row_and_authenticated_request_gated(self):
        source = Path(__file__).with_name("patch-qwen38-k0-gdn-boundaries.py").read_text()
        self.assertIn('"qkvz": 8192', source)
        self.assertIn('"ba": 48', source)
        self.assertEqual(source.count('"core": 3072'), 1)
        self.assertEqual(source.count('"normalized": 3072'), 1)
        self.assertIn("GDN_CAPTURE_ACTIVE", source)
        self.assertIn("tensor[:1].detach().contiguous()", source)
        self.assertIn("name in _ROCKET_K0_GDN_SEEN", source)


if __name__ == "__main__":
    unittest.main()
