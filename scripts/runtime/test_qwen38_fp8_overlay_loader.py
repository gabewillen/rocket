#!/usr/bin/env python3
"""Deterministic source and behavior tests for the FP8 overlay patch."""

import importlib.util
import pathlib
import subprocess
import tempfile
import unittest


PATCHER = pathlib.Path(__file__).with_name("patch-qwen38-fp8-overlay-loader.py")
SPEC = importlib.util.spec_from_file_location("overlay_patcher", PATCHER)
patcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patcher)
RUN = pathlib.Path("/home/glwillen/calibration/qwen38-expanded-mtp3-20260907-03")
SNAPSHOT = pathlib.Path(
    "/home/glwillen/.cache/huggingface/hub/"
    "models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/"
    "fc694b54fb0174e0913e6adf86691ef85a4ead47"
)


SOURCE = '''def safetensors_weights_iterator(
    hf_weights_files,
    use_tqdm_on_load,
    safetensors_load_strategy=None,
):
    sorted_files = sorted(hf_weights_files, key=_natural_sort_key)
    for st_file in tqdm(sorted_files):
        if safetensors_load_strategy == "eager":
            pass
        else:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():
                    if should_skip_weight(name, None):
                        continue
                    # Bound staging to one tensor and avoid CUDA copying directly
                    # from a 64 KiB-page safetensors mmap.
                    param = f.get_tensor(name).clone()
                    yield name, param
'''


class OverlayPatchTest(unittest.TestCase):
    def test_absent_opt_in_keeps_original_yield_path(self):
        result = patcher.patched(SOURCE)
        self.assertIn("if not manifest_name:\n        return None", result)
        self.assertIn("param = f.get_tensor(name).clone()\n                    yield name, param", result)
        self.assertNotIn("mmap(", result)
        self.assertNotIn("cuFile", result)
        self.assertNotIn("nvidia-fs", result)

    def test_injects_exact_modelopt_source_names(self):
        result = patcher.patched(SOURCE)
        self.assertIn('prefix + ".weight_scale"', result)
        self.assertIn('prefix + ".input_scale"', result)
        self.assertIn('("F8_E4M3", entry["shape"])', result)
        self.assertIn('("F32", [1])', result)

    def test_preflight_is_before_first_yield_and_checks_full_family(self):
        result = patcher.patched(SOURCE)
        self.assertLess(result.index("_rocket_qwen38_fp8_overlay_preflight("), result.index("yield name, param"))
        self.assertIn("len(entries) != 180", result)
        self.assertIn("len(layers) != 36", result)
        self.assertIn("source hash mismatch", result)
        self.assertIn("artifact key mismatch", result)
        self.assertIn("extra/missing tensors", result)
        self.assertIn("quant method is not FP8", result)
        self.assertIn("selected tensor remains excluded", result)
        self.assertIn('"header_sha256": hashlib.sha256(prefix + raw).hexdigest()', result)
        self.assertIn('"size": target.stat().st_size', result)
        self.assertNotIn('_rocket_qwen38_file_sha256(target)', result)

    def test_preserves_64k_clone_for_base_and_overlay(self):
        result = patcher.patched(SOURCE)
        self.assertEqual(result.count(".clone()"), 2)
        self.assertIn("Preserve bounded staging on 64 KiB hosts", result)

    def test_source_drift_and_repatch_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "64 KiB clone"):
            patcher.patched(SOURCE.replace("param = f.get_tensor(name).clone()", "param = f.get_tensor(name)"))
        with self.assertRaisesRegex(ValueError, "already patched"):
            patcher.patched(patcher.patched(SOURCE))

    def test_patch_writes_only_explicit_generated_target(self):
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "weight_utils.py"
            target.write_text(SOURCE)
            patcher.patch(target)
            self.assertIn(patcher.MARKER, target.read_text())

    def test_real_hf_symlink_paths_keep_lexical_revision(self):
        paths = sorted(SNAPSHOT.glob("*.safetensors"))
        self.assertEqual(len(paths), 11)
        self.assertTrue(patcher.snapshot_paths_match_revision(paths, SNAPSHOT.name))
        self.assertTrue(all(path.resolve().parent.name == "blobs" for path in paths))

    def test_actual_pinned_sources_prove_selection_and_stacked_scale_path(self):
        modelopt = (RUN / "artifacts/modelopt_patched.py").read_text()
        model = (RUN / "artifacts/model_telemetry.py").read_text()
        patched_loader = patcher.patched(SOURCE)
        self.assertIn('if quant_algo == "FP8":', modelopt)
        self.assertIn("return ModelOptFp8LinearMethod(self.fp8_config)", modelopt)
        self.assertIn("for shard_name in self.packed_modules_mapping[proj_name]", modelopt)
        self.assertIn("hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper", model)
        self.assertIn('orig_to_new_prefix={"model.language_model.": "model."}', model)
        self.assertIn('"in_proj_qkvz": ["in_proj_qkv", "in_proj_z"]', model)
        self.assertIn('"in_proj_ba": ["in_proj_b", "in_proj_a"]', model)
        self.assertIn("loader.load_weights(\n            weights,\n            mapper=self.hf_to_vllm_mapper,", model)
        self.assertIn('r"^model\\.language_model\\.layers\\.(\\d+)\\.linear_attn\\."', patched_loader)
        self.assertIn('prefix + ".weight_scale"', patched_loader)
        self.assertIn('prefix + ".input_scale"', patched_loader)
        image_source = subprocess.run(
            [
                "docker", "run", "--rm", "--entrypoint", "sh",
                "vllm/vllm-openai:qwen38-flash-next", "-c",
                "sed -n '205,240p' /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py; "
                "sed -n '46,145p' /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py; "
                "sed -n '255,288p' /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/utils.py; "
                "sed -n '260,305p' /usr/local/lib/python3.12/dist-packages/vllm/model_executor/parameter.py",
            ],
            check=True, capture_output=True, text=True,
        ).stdout
        self.assertIn('".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2))', image_source)
        self.assertIn('".in_proj_z": (".in_proj_qkvz", 3)', image_source)
        self.assertIn('".in_proj_b": (".in_proj_ba", 0)', image_source)
        self.assertIn('".in_proj_a": (".in_proj_ba", 1)', image_source)
        self.assertIn("for substr, (new_key, new_shard_id) in self.orig_to_new_stacked.items()", image_source)
        self.assertIn("key = key.replace(substr, new_key, 1)", image_source)
        self.assertIn("data.shard_id = shard_id", image_source)
        self.assertIn("quant_config.apply_vllm_mapper", image_source)
        self.assertIn("hf_to_vllm_mapper.get_unstacked_mapper()", image_source)
        self.assertIn("class PerTensorScaleParameter", image_source)
        self.assertIn("self._load_into_shard_id", image_source)


if __name__ == "__main__":
    unittest.main()
