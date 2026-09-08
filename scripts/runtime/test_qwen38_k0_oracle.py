#!/usr/bin/env python3
"""Contract tests for the pinned Qwen3.8 K0 oracle capture."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


HERE = Path(__file__).parent
PATCHER = HERE / "patch-qwen38-k0-oracle.py"
CLIENT = HERE / "qwen38-k0-oracle.py"


class K0OracleTest(unittest.TestCase):
    def test_patcher_requires_exact_model_anchors_and_adds_all_boundaries(self):
        source = '''from itertools import islice

import torch

@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
class Qwen3_8FlashNextModel(nn.Module):
    def forward(self, input_ids):
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.repeat(1, self.config.hc_count)
            if deepstack_input_embeds is not None and layer_idx < len(
        return sample_hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)
'''
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.py"
            model.write_text(source)
            result = subprocess.run(
                ["python3", str(PATCHER), str(model)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            patched = model.read_text()
        self.assertIn('oracle.begin(input_ids, hidden_states)', patched)
        self.assertIn('f"layer.{layer_idx:02d}"', patched)
        self.assertIn('oracle.save("final_norm", sample_hidden_states)', patched)
        self.assertIn('oracle.finish(logits)', patched)
        self.assertIn('get_tensor_model_parallel_rank', patched)
        self.assertIn('output.parent / "ARMED"', patched)
        self.assertIn('"valid": False', patched)

    def test_validator_accepts_only_complete_lossless_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture"
            capture.mkdir()
            request = {
                "schema": "rocket.qwen38.k0-target-oracle-request.v1",
                "prompt": "prompt",
                "input_token_ids": [1, 2],
                "tokenizer_class": "FixtureTokenizer",
                "tokenizer_files": {},
            }
            (root / "request.json").write_text(json.dumps(request))
            artifacts = []
            names = ["embedding", *[f"layer.{i:02d}" for i in range(48)], "final_norm", "logits"]
            for index, name in enumerate(names):
                shape = [1, 8] if name == "logits" else ([2, 8] if name in ("embedding", "final_norm") else [2, 16])
                payload = bytes([index % 256]) * (shape[0] * shape[1] * 2)
                filename = name.replace(".", "-") + ".bin"
                (capture / filename).write_bytes(payload)
                artifacts.append({
                    "name": name,
                    "file": filename,
                    "dtype": "bfloat16",
                    "shape": shape,
                    "strides": [shape[1], 1],
                    "numel": shape[0] * shape[1],
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                })
            manifest = {
                "schema": "rocket.qwen38.k0-target-oracle.v1",
                "valid": True,
                "complete": True,
                "identity": {"model_revision": "fc694"},
                "input_token_ids": [1, 2],
                "artifacts": artifacts,
                "top_k": [{"token_id": 3, "logit": 1.0}],
                "greedy_token_id": 3,
            }
            (capture / "manifest.json").write_text(json.dumps(manifest))
            (root / "response.json").write_text(json.dumps({"choices": [{"text": "x"}]}))
            command = [
                "python3", str(CLIENT), "validate",
                "--request", str(root / "request.json"),
                "--capture-dir", str(capture),
                "--response", str(root / "response.json"),
                "--output", str(root / "result.json"),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            artifacts[1]["shape"] = [2, 8]
            (capture / "manifest.json").write_text(json.dumps(manifest))
            rejected = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(rejected.returncode, 0)

    def test_fixed_prompt_is_real_coding_request(self):
        source = CLIENT.read_text()
        self.assertIn("is_prime(n: int)", source)
        self.assertIn('"temperature": 0', source)
        self.assertIn('"max_tokens": 1', source)


if __name__ == "__main__":
    unittest.main()
