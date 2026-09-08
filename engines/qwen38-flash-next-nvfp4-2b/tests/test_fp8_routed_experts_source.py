# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import hashlib
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src/moe/fp8_routed_experts.h"
ROUTE_HEADER = ROOT / "src/moe/route_compaction.h"
CUDA = ROOT / "src/moe/fp8_routed_experts.cu"
EXECUTOR_HEADER = ROOT / "src/mtp/native_executor.h"
EXECUTOR_CUDA = ROOT / "src/mtp/native_executor.cu"
GENERATOR = ROOT.parents[1] / "scripts/kernels/qwen38-mtp-fp8-triton-aot.py"
VENDOR = ROOT / "vendor/mtp-fp8-triton-sm121"


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
            "cuModuleLoad",
            "cuLaunchKernel",
            "prepare_consumer",
            "ROCKET_QWEN38_MTP_FP8_TRITON_DIR",
            "global_scratch",
            "profile_scratch",
        ):
            self.assertIn(required, cuda)
        for scalar_production_path in (
            "for (int input",
            "fp8_value(",
            "gate_up_silu<<<",
            "down_and_reduce<<<",
        ):
            self.assertNotIn(scalar_production_path, cuda)
        header = HEADER.read_text()
        self.assertIn("RoutedExpertStageEvents", header)
        self.assertIn("stage_events = nullptr", header)
        self.assertIn("if (launch.stage_events", cuda)

    def test_aot_kernels_use_tensor_core_dot_and_active_prefix(self) -> None:
        generator = GENERATOR.read_text()
        for required in (
            "tl.dot",
            "tl.float8e4nv",
            "consumer_active_routes",
            "active_global_ids",
            "local_to_active",
            "grouped_permutation",
            'GPUTarget("cuda", 121, 32)',
        ):
            self.assertIn(required, generator)
        for relative in (
            "quantize_hidden.cubin",
            "silu_mul_quantize.cubin",
            "rank0/gate_up_silu.cubin",
            "rank0/down_weighted_reduce.cubin",
            "rank1/gate_up_silu.cubin",
            "rank1/down_weighted_reduce.cubin",
        ):
            self.assertTrue((VENDOR / relative).is_file(), relative)
        manifest = json.loads((VENDOR / "manifest.json").read_text())
        self.assertEqual(manifest["target"], "cuda-sm121-warp32")
        self.assertEqual(manifest["triton"], "3.7.1")
        self.assertEqual(
            manifest["raw_cubin_abi_suffix"],
            ["global_scratch", "profile_scratch"],
        )
        for relative, expected in manifest["artifacts"].items():
            actual = hashlib.sha256((VENDOR / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)

    def test_selected_vllm_fallback_and_license_are_attributed(self) -> None:
        cuda = CUDA.read_text()
        self.assertIn("SPDX-License-Identifier: Apache-2.0", cuda)
        self.assertIn("vLLM 8e685d198", cuda)
        self.assertIn("869f78732b64454293d2ba42ae0008386fdaa6a6", cuda)
        self.assertIn("Triton", cuda)
        self.assertIn("DeepGEMM", cuda)

    def test_executor_requires_post_fence_consumer_publication(self) -> None:
        header = EXECUTOR_HEADER.read_text()
        cuda = EXECUTOR_CUDA.read_text()
        self.assertIn("RoutedExpertDeviceSummary", header)
        self.assertIn("expert_summaries_device_", header)
        self.assertIn("require_expert_enqueue", cuda)
        self.assertIn("clear routed expert publication", cuda)
        self.assertIn("clear_routed_expert_publication<<<", cuda)
        self.assertNotIn("cudaMemsetAsync(expert_summaries_device_", cuda)
        self.assertIn("validate_routed_expert_summary", cuda)
        self.assertIn("export_routed_expert_otel_after_fence", cuda)
        self.assertLess(cuda.index("enqueue_route_compaction"),
                        cuda.index("middle_.stage_moe"))


if __name__ == "__main__":
    unittest.main()
