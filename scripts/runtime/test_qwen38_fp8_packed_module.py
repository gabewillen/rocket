#!/usr/bin/env python3
"""Container regression for Qwen3.8 packed linear-attention FP8 loading."""

from __future__ import annotations

import pathlib
import subprocess
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).parents[2]
PATCHER = ROOT / "scripts/numerics/patch-qwen38-modelopt-fp8-block-moe.py"
IMAGE = "vllm/vllm-openai:qwen38-flash-next"
ARTIFACT = pathlib.Path(
    "/home/glwillen/calibration/qwen38-linear-fp8-artifacts/"
    "dbefeae04f00118080ce821909786b0c84941b3ac39f1854941a2d2bf4cd516d"
)
MODEL_OPT = (
    "/usr/local/lib/python3.12/dist-packages/vllm/"
    "model_executor/layers/quantization/modelopt.py"
)


REPRO = textwrap.dedent(
    """
    import json
    from types import SimpleNamespace

    import torch
    import vllm.model_executor.parameter as parameter
    import vllm.model_executor.layers.quantization.modelopt as modelopt
    from vllm.config.vllm import set_current_vllm_config
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear
    from vllm.model_executor.models.utils import WeightsMapper

    parameter.get_tensor_model_parallel_rank = lambda: 0
    parameter.get_tensor_model_parallel_world_size = lambda: 1
    modelopt.init_fp8_linear_kernel = lambda **kwargs: object()

    config = modelopt.ModelOptMixedPrecisionConfig.from_config(
        json.load(open("/work/hf_quant_config.json"))
    )
    config.packed_modules_mapping = {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
    mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "language_model.model."},
        orig_to_new_stacked={
            ".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2)),
            ".in_proj_z": (".in_proj_qkvz", 3),
        },
    )
    prefix = "model.language_model.model.layers.0.linear_attn.in_proj_qkvz"
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        kernel_config=SimpleNamespace(linear_backend=None),
        compilation_config=SimpleNamespace(),
    )
    with set_current_vllm_config(runtime):
        layer = MergedColumnParallelLinear(
            8,
            [8, 8, 8, 8],
            bias=False,
            quant_config=config,
            prefix=prefix,
            disable_tp=True,
        )

    source = "model.language_model.layers.0.linear_attn.in_proj_qkv"
    tensors = [
        (source + ".weight", torch.empty(24, 8, dtype=torch.float8_e4m3fn)),
        (source + ".weight_scale", torch.ones(1, dtype=torch.float32)),
        (source + ".input_scale", torch.ones(1, dtype=torch.float32)),
    ]
    mapped = list(mapper.apply(tensors))
    relative = [(name.rsplit("in_proj_qkvz.", 1)[1], value) for name, value in mapped]
    loaded = list(layer.load_weights(relative))
    assert type(layer.quant_method).__name__ == "ModelOptFp8LinearMethod"
    assert loaded == ["weight", "weight_scale", "input_scale"]
    print("packed_fp8_first_weight=ok")
    """
)


class PackedFp8ModuleTest(unittest.TestCase):
    def test_actual_container_constructs_and_loads_first_packed_projection(self):
        self.assertTrue(ARTIFACT.is_dir(), f"missing artifact: {ARTIFACT}")
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            work = pathlib.Path(directory)
            modelopt_path = work / "modelopt.py"
            container = subprocess.run(
                ["docker", "create", IMAGE],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            try:
                subprocess.run(
                    ["docker", "cp", f"{container}:{MODEL_OPT}", str(modelopt_path)],
                    check=True,
                )
            finally:
                subprocess.run(
                    ["docker", "rm", container], capture_output=True, check=True
                )
            subprocess.run(
                ["python3", str(PATCHER), str(modelopt_path)], check=True
            )
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{modelopt_path}:{MODEL_OPT}:ro",
                    "-v",
                    f"{ARTIFACT / 'hf_quant_config.json'}:/work/hf_quant_config.json:ro",
                    "--entrypoint",
                    "python3",
                    IMAGE,
                    "-c",
                    REPRO,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("packed_fp8_first_weight=ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
