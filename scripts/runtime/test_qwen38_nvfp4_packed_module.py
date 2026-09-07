#!/usr/bin/env python3
"""Pinned-image regressions for actual packed Qwen NVFP4 module loading."""

from __future__ import annotations

import pathlib
import subprocess
import textwrap
import unittest


ROOT = pathlib.Path(__file__).parents[2]
IMAGE = "vllm/vllm-openai:qwen38-flash-next"
ARTIFACT = pathlib.Path(
    "/home/glwillen/calibration/qwen38-linear-nvfp4-artifacts/"
    "64539e4a5a4b533aa143d20e055ae73ebd10b21f05dbd422847a9be30e56c4e5"
)
MODEL_OPT = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py"
PATCHED_MODEL_OPT = pathlib.Path(
    "/home/glwillen/calibration/qwen38-linear-fp8-production-20260907-10/artifacts/modelopt_patched.py"
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
    modelopt.init_nvfp4_linear_kernel = lambda **kwargs: SimpleNamespace(input_quant_key=lambda: None)
    config = modelopt.ModelOptMixedPrecisionConfig.from_config(json.load(open('/work/hf_quant_config.json')))
    config.packed_modules_mapping = {'in_proj_qkvz': ['in_proj_qkv', 'in_proj_z']}
    mapper = WeightsMapper(
        orig_to_new_prefix={'model.language_model.': 'language_model.model.'},
        orig_to_new_stacked={
            '.in_proj_qkv': ('.in_proj_qkvz', (0, 1, 2)),
            '.in_proj_z': ('.in_proj_qkvz', 3),
        },
    )
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        kernel_config=SimpleNamespace(linear_backend=None),
        compilation_config=SimpleNamespace(),
    )
    prefix = 'model.language_model.model.layers.0.linear_attn.in_proj_qkvz'
    with set_current_vllm_config(runtime):
        layer = MergedColumnParallelLinear(32, [16, 16, 16, 16], bias=False, quant_config=config, prefix=prefix, disable_tp=True)
    source = 'model.language_model.layers.0.linear_attn.in_proj_qkv'
    tensors = [
        (source + '.weight', torch.empty(48, 16, dtype=torch.uint8)),
        (source + '.weight_scale', torch.ones(48, 2, dtype=torch.float8_e4m3fn)),
        (source + '.weight_scale_2', torch.ones(1, dtype=torch.float32)),
        (source + '.input_scale', torch.ones(1, dtype=torch.float32)),
    ]
    mapped = list(mapper.apply(tensors))
    relative = [(name.rsplit('in_proj_qkvz.', 1)[1], value) for name, value in mapped]
    print('registered', {name: tuple(value.shape) for name, value in layer.named_parameters()})
    print('mapped', [(name, tuple(value.shape), getattr(value, 'shard_id', None)) for name, value in relative])
    loaded = list(layer.load_weights(relative))
    assert type(layer.quant_method).__name__ == 'ModelOptNvFp4LinearMethod'
    assert loaded == ['weight', 'weight_scale', 'weight_scale_2', 'input_scale'], loaded
    print('packed_nvfp4_first_weight=ok')
    """
)

FULL_ATTENTION_REPRO = textwrap.dedent(
    """
    import json
    from types import SimpleNamespace
    import torch
    import vllm.model_executor.parameter as parameter
    import vllm.model_executor.layers.quantization.modelopt as modelopt
    from vllm.config.vllm import set_current_vllm_config
    from vllm.model_executor.layers.linear import QKVParallelLinear
    from vllm.model_executor.models.utils import WeightsMapper

    parameter.get_tensor_model_parallel_rank = lambda: 0
    parameter.get_tensor_model_parallel_world_size = lambda: 1
    modelopt.init_nvfp4_linear_kernel = lambda **kwargs: SimpleNamespace(input_quant_key=lambda: None)
    raw = json.load(open('/work/hf_quant_config.json'))
    quant = raw['quantization']
    quant['exclude_modules'].remove('model.language_model.layers.3.self_attn*')
    for projection in ('q_proj', 'k_proj', 'v_proj'):
        quant['quantized_layers'][f'model.language_model.layers.3.self_attn.{projection}'] = {'quant_algo': 'NVFP4'}
    config = modelopt.ModelOptMixedPrecisionConfig.from_config(raw)
    config.packed_modules_mapping = {'qkv_proj': ['q_proj', 'k_proj', 'v_proj']}
    mapper = WeightsMapper(
        orig_to_new_prefix={'model.language_model.': 'language_model.model.'},
        orig_to_new_stacked={
            '.q_proj': ('.qkv_proj', 'q'),
            '.k_proj': ('.qkv_proj', 'k'),
            '.v_proj': ('.qkv_proj', 'v'),
        },
    )
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        kernel_config=SimpleNamespace(linear_backend=None),
        compilation_config=SimpleNamespace(),
    )
    prefix = 'model.language_model.model.layers.3.self_attn.qkv_proj'
    with set_current_vllm_config(runtime):
        layer = QKVParallelLinear(
            32, 8, 2, 1, bias=False, quant_config=config, prefix=prefix, disable_tp=True
        )
    tensors = []
    for projection, rows in (('q_proj', 16), ('k_proj', 8), ('v_proj', 8)):
        source = f'model.language_model.layers.3.self_attn.{projection}'
        tensors.extend([
            (source + '.weight', torch.empty(rows, 16, dtype=torch.uint8)),
            (source + '.weight_scale', torch.ones(rows, 2, dtype=torch.float8_e4m3fn)),
            (source + '.weight_scale_2', torch.ones(1, dtype=torch.float32)),
            (source + '.input_scale', torch.ones(1, dtype=torch.float32)),
        ])
    mapped = list(mapper.apply(tensors))
    relative = [(name.rsplit('qkv_proj.', 1)[1], value) for name, value in mapped]
    shard_ids = {getattr(value, 'shard_id', None) for _, value in relative}
    loaded = list(layer.load_weights(relative))
    assert type(layer.quant_method).__name__ == 'ModelOptNvFp4LinearMethod'
    assert shard_ids == {'q', 'k', 'v'}, shard_ids
    assert loaded, loaded
    print('full_attention_qkv_nvfp4=ok')
    """
)


class PackedNvfp4ModuleTest(unittest.TestCase):
    def run_repro(self, source: str):
        return subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{PATCHED_MODEL_OPT}:{MODEL_OPT}:ro",
                "-v", f"{ARTIFACT / 'hf_quant_config.json'}:/work/hf_quant_config.json:ro",
                "--entrypoint", "python3", IMAGE, "-c", source,
            ],
            capture_output=True, text=True, check=False,
        )

    def test_actual_container_constructs_and_loads_first_packed_projection(self):
        self.assertTrue(ARTIFACT.is_dir(), f"missing artifact: {ARTIFACT}")
        result = self.run_repro(REPRO)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("packed_nvfp4_first_weight=ok", result.stdout)

    def test_actual_modelopt_constructs_and_loads_fused_full_attention_qkv(self):
        self.assertTrue(ARTIFACT.is_dir(), f"missing artifact: {ARTIFACT}")
        result = self.run_repro(FULL_ATTENTION_REPRO)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("full_attention_qkv_nvfp4=ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
