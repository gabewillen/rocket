#!/usr/bin/env python3
"""Add Qwen3.8 block-FP8 RoutedExperts dispatch to patched modelopt.py."""

import argparse
import ast
from pathlib import Path


HELPER_ANCHOR = """    @staticmethod
    def _quantized_layer_prefix_candidates(prefix: str) -> tuple[str, ...]:
"""
HELPER = '''    def _fp8_block_scales_config(self, prefix: str):
        """Build the checkpoint's exact 128x128 block-FP8 configuration."""
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        info = None
        for candidate in self._quantized_layer_prefix_candidates(prefix):
            info = self.quantized_layers.get(candidate)
            if info is None:
                prefix_dot = candidate + "."
                for key, value in self.quantized_layers.items():
                    if key.startswith(prefix_dot):
                        info = value
                        break
            if info is not None:
                break

        group_size = (info or {}).get("group_size")
        if type(group_size) is not int or group_size != 128:
            raise ValueError(
                f"block-FP8 layer {prefix} requires exact group_size=128 "
                f"in quantized_layers (got {group_size!r})"
            )
        return Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[group_size, group_size],
        )

'''

DISPATCH_ANCHOR = '''            if quant_algo == "MXFP8":
                return ModelOptMxFp8FusedMoE(
                    quant_config=self.mxfp8_config,
                    moe_config=layer.moe_config,
                )
            return None
'''
DISPATCH = '''            if quant_algo == "MXFP8":
                return ModelOptMxFp8FusedMoE(
                    quant_config=self.mxfp8_config,
                    moe_config=layer.moe_config,
                )
            if quant_algo in ("FP8_BLOCK_SCALES", "FP8_PB_WO"):
                from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

                logger.info_once(
                    "Routed experts %s use %s; building 128x128 block-FP8 "
                    "weights with Fp8MoEMethod.",
                    prefix,
                    quant_algo,
                )
                return Fp8MoEMethod(
                    quant_config=self._fp8_block_scales_config(prefix),
                    layer=layer,
                )
            return None
'''
PATCH_MARKER = 'quant_algo in ("FP8_BLOCK_SCALES", "FP8_PB_WO")'

FUSED_SHARDS_ANCHOR = '''        fused_projection_shards = {
            "qkv_proj": ("q_proj", "k_proj", "v_proj"),
            "gate_up_proj": ("gate_proj", "up_proj"),
        }
'''
FUSED_SHARDS = '''        fused_projection_shards = {
            "qkv_proj": ("q_proj", "k_proj", "v_proj"),
            "gate_up_proj": ("gate_proj", "up_proj"),
            # Qwen3.8 stores these projections separately but constructs two
            # MergedColumnParallelLinear modules at runtime.
            "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
            "in_proj_ba": ("in_proj_b", "in_proj_a"),
        }
'''

PREFIX_CANDIDATE_ANCHOR = '''        if prefix.startswith("language_model.model."):
            candidates.append(
                "model.language_model." + prefix[len("language_model.model.") :]
            )
        elif prefix.startswith("model.language_model."):
            candidates.append(
                "language_model.model." + prefix[len("model.language_model.") :]
            )
'''
PREFIX_CANDIDATE_PATCH = '''        if prefix.startswith("model.language_model.model."):
            suffix = prefix[len("model.language_model.model.") :]
            candidates.append(
                "language_model.model."
                + suffix
            )
            candidates.append(
                "model.language_model."
                + suffix
            )
            candidates.append("model." + suffix)
        elif prefix.startswith("language_model.model."):
            suffix = prefix[len("language_model.model.") :]
            candidates.append("model.language_model." + suffix)
            candidates.append("model." + suffix)
        elif prefix.startswith("model.language_model."):
            candidates.append(
                "language_model.model." + prefix[len("model.language_model.") :]
            )
'''


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"modelopt source drift: {label} anchor count={count}")
    return source.replace(old, new)


def patch(target: Path) -> None:
    source = target.read_text()
    if PATCH_MARKER in source:
        raise SystemExit("already patched")
    source = replace_once(source, HELPER_ANCHOR, HELPER + HELPER_ANCHOR, "helper")
    source = replace_once(
        source, FUSED_SHARDS_ANCHOR, FUSED_SHARDS, "Qwen3.8 fused projections"
    )
    source = replace_once(
        source,
        PREFIX_CANDIDATE_ANCHOR,
        PREFIX_CANDIDATE_PATCH,
        "Qwen3.8 multimodal prefix",
    )
    source = replace_once(source, DISPATCH_ANCHOR, DISPATCH, "dispatch")
    ast.parse(source, filename=str(target))
    target.write_text(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "modelopt_py",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("modelopt_patched.py"),
    )
    args = parser.parse_args()
    patch(args.modelopt_py)


if __name__ == "__main__":
    main()
