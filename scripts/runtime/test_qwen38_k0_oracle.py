#!/usr/bin/env python3
"""Contract tests for the pinned Qwen3.8 K0 oracle capture."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


HERE = Path(__file__).parent
PATCHER = HERE / "patch-qwen38-k0-oracle.py"
CLIENT = HERE / "qwen38-k0-oracle.py"


class K0OracleTest(unittest.TestCase):
    @staticmethod
    def load_patcher():
        spec = importlib.util.spec_from_file_location("k0_oracle_patcher", PATCHER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

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

    def embed_input_ids(self, input_ids, multimodal_embeddings=None):
        inputs_embeds = self._embed_text_input_ids(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
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
        self.assertIn('oracle.begin(input_ids, inputs_embeds)', patched)
        self.assertNotIn('oracle.begin(input_ids, hidden_states)', patched)
        self.assertIn('f"layer.{layer_idx:02d}"', patched)
        self.assertIn('oracle.save("final_norm", sample_hidden_states)', patched)
        self.assertIn('oracle.finish(logits)', patched)
        self.assertIn('get_tensor_model_parallel_rank', patched)
        self.assertIn('output.parent / "ARMED"', patched)
        self.assertIn('"valid": False', patched)
        self.assertIn('oracle request identity differs', patched)
        self.assertIn('previous authenticated forward is missing logits', patched)
        self.assertIn('after exactly-once consumption', patched)

    def test_oracle_state_spans_external_embedding_forward_and_logits_lifecycle(self):
        patcher = self.load_patcher()

        class FakeTensor:
            def __init__(self, values, shape):
                self.values = values
                self.shape = shape
                self.ndim = len(shape)

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self.values

            def __getitem__(self, index):
                return self.values[index] if self.ndim == 1 else self

            def float(self):
                return self

            def numel(self):
                return self.shape[-1]

        class FakeTorch:
            @staticmethod
            def topk(_tensor, count, sorted=True):
                del sorted
                return FakeTensor([1.0] * count, [count]), FakeTensor(list(range(count)), [count])

        namespace = {
            "Path": Path,
            "hashlib": hashlib,
            "json": json,
            "os": __import__("os"),
            "torch": FakeTorch,
        }
        exec(patcher.HELPER, namespace)
        oracle_class = namespace["_RocketK0Oracle"]
        with tempfile.TemporaryDirectory() as directory:
            oracle = oracle_class.__new__(oracle_class)
            oracle.output = Path(directory)
            oracle.expected_ids = [1, 2]
            oracle.identity = {"request_sha256": "fixture"}
            oracle.request_sha256 = "fixture"
            oracle.generation_index = 0
            oracle.artifacts = []
            oracle.artifact_by_name = {}
            oracle.expected_forward_names = ["embedding", *[f"layer.{i:02d}" for i in range(48)], "final_norm"]
            oracle.consumed_tokens = 0
            oracle.active_forward = False
            oracle.forward_names = []
            oracle.complete = False
            oracle.append = lambda name, _tensor: oracle.artifacts.append({"name": name}) if not any(item["name"] == name for item in oracle.artifacts) else None
            for token_ids in ([1], [2]):
                oracle.begin(FakeTensor(token_ids, [1]), FakeTensor([], [1, 8]))
                for index in range(48):
                    oracle.save(f"layer.{index:02d}", FakeTensor([], [1, 16]))
                oracle.save("final_norm", FakeTensor([], [1, 8]))
                oracle.finish(FakeTensor([], [1, 32]))
            self.assertTrue(oracle.complete)
            self.assertEqual(json.loads((Path(directory) / "manifest.json").read_text())["generation_index"], 0)
            with self.assertRaisesRegex(RuntimeError, "second generation"):
                oracle.begin(FakeTensor([1, 2], [2]), FakeTensor([], [2, 8]))

            stale = oracle_class.__new__(oracle_class)
            stale.expected_ids = [1, 2]
            stale.consumed_tokens = 1
            stale.active_forward = True
            stale.complete = False
            with self.assertRaisesRegex(RuntimeError, "missing logits"):
                stale.begin(FakeTensor([1, 2], [2]), FakeTensor([], [2, 8]))

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
                "identity": {
                    "model_revision": "fc694",
                    "request_sha256": hashlib.sha256((root / "request.json").read_bytes()).hexdigest(),
                    "generation_index": 0,
                },
                "request_sha256": hashlib.sha256((root / "request.json").read_bytes()).hexdigest(),
                "generation_index": 0,
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
        self.assertIn('"request_sha256": sha256(request_path)', source)
        self.assertIn('"generation_index": 0', source)


if __name__ == "__main__":
    unittest.main()
