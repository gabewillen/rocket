# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src/moe/fp8_routed_experts.h"
ROUTE_HEADER = ROOT / "src/moe/route_compaction.h"
CUDA = ROOT / "src/moe/fp8_routed_experts.cu"
EXECUTOR_HEADER = ROOT / "src/mtp/native_executor.h"
EXECUTOR_CUDA = ROOT / "src/mtp/native_executor.cu"


class Fp8RoutedExpertsSourceTest(unittest.TestCase):
    def test_authenticated_source_and_serving_layout_stay_distinct(self) -> None:
        header = HEADER.read_text()
        route_header = ROUTE_HEADER.read_text()
        self.assertIn("kMtpExpertSourceAbi", header)
        self.assertIn("kMtpExpertServingAbi", header)
        self.assertIn('"fp8_e4m3_block_128x128"', header)
        self.assertIn("kLogicalIntermediate", route_header)
        self.assertIn("kPhysicalIntermediate", route_header)
        self.assertNotIn("nvfp4", header.lower())

    def test_replay_path_is_caller_owned_and_device_bounded(self) -> None:
        cuda = CUDA.read_text()
        for forbidden in (
            "cudaMalloc",
            "cudaFree",
            "cudaMemcpy",
            "cudaStreamSynchronize",
            "cudaDeviceSynchronize",
        ):
            self.assertNotIn(forbidden, cuda)
        for required in (
            "active_experts",
            "active_routes",
            "active_global_expert_ids",
            "expert_row_counts",
            "expert_route_offsets",
            "expert_route_indices",
            "owner_route_weights",
            "owner_route_rows",
        ):
            self.assertIn(required, cuda)
        self.assertIn("valid_generation", cuda)

    def test_selected_vllm_fallback_and_license_are_attributed(self) -> None:
        cuda = CUDA.read_text()
        self.assertIn("SPDX-License-Identifier: Apache-2.0", cuda)
        self.assertIn("vLLM 8e685d198", cuda)
        self.assertIn("6b5a12c0", cuda)
        self.assertIn("Triton", cuda)
        self.assertIn("DeepGEMM", cuda)

    def test_executor_requires_post_fence_consumer_publication(self) -> None:
        header = EXECUTOR_HEADER.read_text()
        cuda = EXECUTOR_CUDA.read_text()
        self.assertIn("RoutedExpertDeviceSummary", header)
        self.assertIn("expert_summaries_device_", header)
        self.assertIn("require_expert_enqueue", cuda)
        self.assertIn("clear routed expert publication", cuda)
        self.assertIn("validate_routed_expert_summary", cuda)
        self.assertIn("export_routed_expert_otel_after_fence", cuda)
        self.assertLess(cuda.index("enqueue_route_compaction"),
                        cuda.index("middle_.stage_moe"))


if __name__ == "__main__":
    unittest.main()
