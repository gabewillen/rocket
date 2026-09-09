#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("patch-qwen38-qsa-c1-oracle.py")


class TestQsaC1OraclePatcher(unittest.TestCase):
    def test_c1_continuation_is_bound_to_sampled_token_and_exact_extent(self):
        source = SCRIPT.read_text()
        self.assertIn("actual != [self.c1_token]", source)
        self.assertIn("tuple(embedding.shape) != (1, 2560)", source)
        self.assertIn("if self.c1_active:", source)
        self.assertIn("self.complete = True", source)
        self.assertIn("PINNED_SOURCE_SHA256", source)
        self.assertNotIn("synthetic", source)


if __name__ == "__main__":
    unittest.main()
