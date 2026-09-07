# SPDX-License-Identifier: Apache-2.0
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/attention/qsa_prefill.cu"
HEADER = ROOT / "src/attention/qsa_prefill.h"


class QsaPrefillSourceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text()
        cls.header = HEADER.read_text()

    def test_pins_reference_provenance_and_qwen_geometry(self):
        self.assertIn("FlashInfer 91bda04c", self.source)
        self.assertIn("vLLM 8e685d198", self.source)
        for declaration in (
            "constexpr int kHeads = 12;",
            "constexpr int kDim = 256;",
            "constexpr int kTopk = 2051;",
        ):
            self.assertIn(declaration, self.source)

    def test_only_measured_prompt_and_concurrency_shapes_are_admitted(self):
        shape = re.search(
            r"bool shape_ok\(.*?\n\}", self.source, flags=re.DOTALL
        )
        self.assertIsNotNone(shape)
        contract = shape.group(0)
        for bucket in (1, 2, 4, 8, 16):
            self.assertIn(f"sequences == {bucket}", contract)
        self.assertIn("query_tokens == 300 && context_tokens == 8492", contract)
        self.assertIn("query_tokens == 8192 && context_tokens == 8192", contract)

    def test_hot_launches_are_allocation_and_fence_free(self):
        hot = self.source[self.source.index('extern "C" int qwen38_qsa_prefill_union2') :]
        for forbidden in ("cudaMalloc", "cudaFree", "cudaDeviceSynchronize", "cudaStreamSynchronize"):
            self.assertNotIn(forbidden, hot)
        self.assertIn("enqueues exclusively on stream", self.header)
        self.assertIn("strictly ascending, unique", self.header)
        self.assertIn("treats K/V as immutable", self.header)
        self.assertIn("const __nv_bfloat16* key", self.header)

    def test_scalar_oracle_precedes_timing_in_smoke(self):
        smoke = (ROOT / "bench/qsa_prefill_smoke.cu").read_text()
        self.assertLess(smoke.index("qwen38_qsa_prefill_control("), smoke.index("std::array<float, 7> samples"))
        self.assertIn("scalar_prefill<<<dim3(sequences * query_tokens, kHeads), 32", self.source)
        self.assertIn('pattern=" << (disjoint ? "disjoint" : "causal-overlap")', smoke)
        self.assertNotRegex(smoke, r"% (127|113|109) -")
        self.assertIn("state_unchanged=yes", smoke)
        timed = smoke[smoke.index("std::array<float, 7> samples") :]
        self.assertGreaterEqual(timed.count("nullptr, stream"), 2)
        self.assertIn("optional diagnostic telemetry", self.header)

    def test_reference_runner_fails_closed_on_vllm_identity(self):
        runner = (
            ROOT.parents[1] / "scripts/kernels/qwen38-qsa-prefill-vllm-reference.py"
        ).read_text()
        self.assertIn('VLLM_COMMIT = "8e685d198"', runner)
        self.assertIn("if VLLM_COMMIT not in vllm_version", runner)
        self.assertIn("qsa_sparse_paged_attention", runner)

    def test_merge_progress_is_cta_shared(self):
        self.assertIn("std::int32_t* merge;", self.source)
        self.assertIn("++s.merge[0]", self.source)
        self.assertIn("++s.merge[1]", self.source)
        self.assertNotIn("while (left < kTopk", self.source)
        self.assertRegex(
            self.source,
            r"(?s)while \(true\).*?__syncthreads\(\);\n    if \(s\.mask\[0\] == 0\) break;",
        )


if __name__ == "__main__":
    unittest.main()
