# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import unittest
from pathlib import Path


SOURCE = (
    Path(__file__).parents[1] / "src" / "mtp" / "native_executor.cu"
).read_text(encoding="utf-8")


def body(signature: str, next_signature: str) -> str:
    match = re.search(
        re.escape(signature) + r"(?P<body>.*?)" + re.escape(next_signature),
        SOURCE,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing native method: {signature}")
    return match.group("body")


class NativeMtpHotPathTests(unittest.TestCase):
    def test_draft_has_no_host_copy_or_device_fence(self):
        draft = body("DeviceDraftView NativeExecutor::draft", "void NativeExecutor::stage_accept")
        self.assertNotIn("cudaMemcpy", draft)
        self.assertNotIn("Synchronize", draft)

    def test_publish_uses_device_widths_without_fence(self):
        publish = body(
            "void NativeExecutor::stage_accept", "void NativeExecutor::commit"
        )
        self.assertNotIn("cudaMemcpy", publish)
        self.assertNotIn("Synchronize", publish)
        self.assertIn("accepted_widths_device", publish)


if __name__ == "__main__":
    unittest.main()
