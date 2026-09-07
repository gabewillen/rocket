# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
COMPOSITION = (ROOT / "src" / "mtp" / "decoder_step.cc").read_text(encoding="utf-8")
C_API = (ROOT / "src" / "mtp" / "decoder_step_c_api.cc").read_text(encoding="utf-8")
VERIFIER = (ROOT / "src" / "decode" / "decoder_verifier.cc").read_text(encoding="utf-8")


class MtpDecoderStepSourceTests(unittest.TestCase):
    def test_composition_is_one_native_draft_verify_sequence(self):
        self.assertEqual(COMPOSITION.count("drafter_.draft("), 1)
        self.assertEqual(COMPOSITION.count("verifier_.step("), 1)
        self.assertNotRegex(COMPOSITION, re.compile(r"\b(for|while)\s*\("))
        self.assertNotIn("Synchronize", COMPOSITION)

    def test_mtp_accept_is_inside_verifier_fence_and_atomic_publish(self):
        accept = VERIFIER.index("accepted_state_->stage_accept")
        fence = VERIFIER.index("runtime_.synchronize", accept)
        publish = VERIFIER.index("state_.publish", fence)
        commit = VERIFIER.index("accepted_state_->commit", publish)
        self.assertLess(accept, fence)
        self.assertLess(fence, publish)
        self.assertLess(publish, commit)

    def test_c_abi_exposes_one_engine_step_without_model_callbacks(self):
        self.assertIn('extern "C" int qwen38_mtp_decoder_step(', C_API)
        self.assertNotIn("callback", C_API.lower())
        self.assertEqual(C_API.count("step.step("), 1)


if __name__ == "__main__":
    unittest.main()
