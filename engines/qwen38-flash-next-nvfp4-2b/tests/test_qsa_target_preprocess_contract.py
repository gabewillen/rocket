# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import unittest

ENGINE = Path(__file__).parents[1]


class TargetQsaPreprocessSourceTests(unittest.TestCase):
    def test_replay_path_is_c1_external_state_only(self):
        source = (ENGINE / "src/attention/qsa_target_preprocess.cu").read_text()
        header = (ENGINE / "src/attention/qsa_target_preprocess.h").read_text()
        launch = source.split("void launch_target_qsa_preprocess_c1", 1)[1]
        self.assertIn("state.main_key_cache", launch)
        self.assertIn("state.raw_key_cache", launch)
        self.assertIn("index_qk_gemv<<<kIndexOutputs", launch)
        self.assertNotIn("cudaMalloc", launch)
        self.assertNotIn("cudaMemcpy", launch)
        self.assertNotIn("cudaDeviceSynchronize", launch)
        self.assertNotIn("cudaStreamSynchronize", launch)
        self.assertIn("Caller-owned graph arena", header)

    def test_pinned_and_current_upstream_provenance_is_named(self):
        source = (ENGINE / "src/attention/qsa_target_preprocess.cu").read_text()
        self.assertIn("8e685d198", source)
        self.assertIn("9ea8f3ffc354901b740f0b31988900897b7221d7", source)


if __name__ == "__main__":
    unittest.main()
