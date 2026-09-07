#!/usr/bin/env python3
"""CPU/static regression tests for Qwen3.8 block-FP8 MoE dispatch."""

import contextlib
import hashlib
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


PATCHER = pathlib.Path(__file__).with_name("patch-qwen38-modelopt-fp8-block-moe.py")


SOURCE = '''class ModelOptMxFp8FusedMoE:
    def __init__(self, **kwargs): pass

class RoutedExperts: pass

class Config:
    def __init__(self, algo, group_size):
        self.algo = algo
        self.quantized_layers = {
            "mtp.layers.48.mlp.experts": {
                "quant_algo": algo,
                "group_size": group_size,
            }
        }
        self.mxfp8_config = object()

    @staticmethod
    def _quantized_layer_prefix_candidates(prefix: str) -> tuple[str, ...]:
        return (prefix,)

    def get_quant_method(self, layer, prefix):
        quant_algo = self.algo
        if isinstance(layer, RoutedExperts):
            if quant_algo == "MXFP8":
                return ModelOptMxFp8FusedMoE(
                    quant_config=self.mxfp8_config,
                    moe_config=layer.moe_config,
                )
            return None

    def _resolve_quant_algo(self, prefix):
        proj_name = prefix.rsplit(".", 1)[-1]
        fused_projection_shards = {
            "qkv_proj": ("q_proj", "k_proj", "v_proj"),
            "gate_up_proj": ("gate_proj", "up_proj"),
        }
        shard_names = fused_projection_shards.get(proj_name)
        if shard_names is not None:
            parent_dot = prefix.rsplit(".", 1)[0] + "."
            shard_algos = {
                self.quantized_layers[parent_dot + name]["quant_algo"].upper()
                for name in shard_names
                if parent_dot + name in self.quantized_layers
            }
            if len(shard_algos) == 1:
                return shard_algos.pop()
            if len(shard_algos) > 1:
                raise ValueError("Mixed quant_algo within fused layer")
        return None
'''


class FakeFp8Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeFp8MoEMethod:
    def __init__(self, quant_config, layer):
        self.quant_config = quant_config
        self.layer = layer


class FakeLogger:
    def info_once(self, *_args):
        pass


class BlockFp8DispatchTest(unittest.TestCase):
    @contextlib.contextmanager
    def patched_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            target = pathlib.Path(directory) / "modelopt_patched.py"
            target.write_text(SOURCE)
            result = subprocess.run(
                [sys.executable, str(PATCHER), str(target)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            patched = target.read_text()

        fp8_module = types.ModuleType(
            "vllm.model_executor.layers.quantization.fp8"
        )
        fp8_module.Fp8Config = FakeFp8Config
        fp8_module.Fp8MoEMethod = FakeFp8MoEMethod
        modules = {
            "vllm": types.ModuleType("vllm"),
            "vllm.model_executor": types.ModuleType("vllm.model_executor"),
            "vllm.model_executor.layers": types.ModuleType(
                "vllm.model_executor.layers"
            ),
            "vllm.model_executor.layers.quantization": types.ModuleType(
                "vllm.model_executor.layers.quantization"
            ),
            "vllm.model_executor.layers.quantization.fp8": fp8_module,
        }
        with mock.patch.dict(sys.modules, modules):
            namespace = {"logger": FakeLogger()}
            exec(compile(patched, "modelopt_patched.py", "exec"), namespace)
            yield namespace

    def test_both_checkpoint_spellings_dispatch_equivalently(self):
        with self.patched_namespace() as namespace:
            layer = namespace["RoutedExperts"]()
            methods = []
            for spelling in ("FP8_BLOCK_SCALES", "FP8_PB_WO"):
                config = namespace["Config"](spelling, 128)
                method = config.get_quant_method(
                    layer, "mtp.layers.48.mlp.experts"
                )
                self.assertIsInstance(method, FakeFp8MoEMethod)
                self.assertEqual(method.quant_config.weight_block_size, [128, 128])
                methods.append(type(method))
            self.assertEqual(methods[0], methods[1])

    def test_group_size_must_be_exactly_128(self):
        with self.patched_namespace() as namespace:
            layer = namespace["RoutedExperts"]()
            for invalid in (None, True, 0, 64, 256):
                with self.subTest(group_size=invalid), self.assertRaisesRegex(
                    ValueError, "group_size=128"
                ):
                    namespace["Config"]("FP8_PB_WO", invalid).get_quant_method(
                        layer, "mtp.layers.48.mlp.experts"
                    )

    def test_generator_output_is_deterministic(self):
        outputs = []
        with tempfile.TemporaryDirectory() as directory:
            for index in range(2):
                target = pathlib.Path(directory) / f"modelopt-{index}.py"
                target.write_text(SOURCE)
                subprocess.run(
                    [sys.executable, str(PATCHER), str(target)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                outputs.append(hashlib.sha256(target.read_bytes()).hexdigest())
        self.assertEqual(outputs[0], outputs[1])

    def test_qwen38_packed_linear_attention_resolves_fp8(self):
        with self.patched_namespace() as namespace:
            for fused, shards in {
                "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
                "in_proj_ba": ("in_proj_b", "in_proj_a"),
            }.items():
                config = namespace["Config"]("FP8_PB_WO", 128)
                config.quantized_layers = {
                    f"language_model.model.layers.0.linear_attn.{shard}": {
                        "quant_algo": "FP8"
                    }
                    for shard in shards
                }
                self.assertEqual(
                    config._resolve_quant_algo(
                        f"language_model.model.layers.0.linear_attn.{fused}"
                    ),
                    "FP8",
                )

    def test_qwen38_packed_linear_attention_rejects_mixed_algorithms(self):
        with self.patched_namespace() as namespace:
            config = namespace["Config"]("FP8_PB_WO", 128)
            config.quantized_layers = {
                "language_model.model.layers.0.linear_attn.in_proj_qkv": {
                    "quant_algo": "FP8"
                },
                "language_model.model.layers.0.linear_attn.in_proj_z": {
                    "quant_algo": "NVFP4"
                },
            }
            with self.assertRaisesRegex(ValueError, "Mixed quant_algo"):
                config._resolve_quant_algo(
                    "language_model.model.layers.0.linear_attn.in_proj_qkvz"
                )


if __name__ == "__main__":
    unittest.main()
