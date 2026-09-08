# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src/moe/route_compaction.h"
CUDA = ROOT / "src/moe/route_compaction.cu"


class RouteCompactionSourceTest(unittest.TestCase):
    def test_replay_path_has_fixed_storage_and_no_host_round_trip(self) -> None:
        header = HEADER.read_text()
        cuda = CUDA.read_text()
        self.assertIn("RouteCompactionBuffers", header)
        self.assertIn("source_generation", header)
        self.assertIn("requested_generation", header)
        self.assertIn("<<<1, kThreads, 0, launch.stream>>>", cuda)
        for forbidden in (
            "cudaMalloc",
            "cudaFree",
            "cudaMemcpy",
            "cudaStreamSynchronize",
            "cudaDeviceSynchronize",
        ):
            self.assertNotIn(forbidden, cuda)

    def test_active_union_controls_work_and_bounded_otel_values(self) -> None:
        header = HEADER.read_text()
        cuda = CUDA.read_text()
        self.assertIn("active_experts", cuda)
        self.assertIn("expert_row_counts", cuda)
        self.assertIn("active_weight_bytes", cuda)
        self.assertIn("kActiveExperts,", header)
        self.assertIn("kActiveRows,", header)
        self.assertIn("kActiveWeightBytes,", header)
        self.assertNotIn("generation;\n  int rank", header)

    def test_reference_revisions_are_named(self) -> None:
        cuda = CUDA.read_text()
        self.assertIn("vLLM 8e685d198", cuda)
        self.assertIn("6b5a12c0", cuda)
        self.assertIn("FlashInfer 91bda04c", cuda)


if __name__ == "__main__":
    unittest.main()
