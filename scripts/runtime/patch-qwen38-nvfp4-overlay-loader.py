#!/usr/bin/env python3
"""Patch pinned vLLM weight_utils.py for the verified Qwen NVFP4 overlay."""

from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path


BASE_PATH = Path(__file__).with_name("patch-qwen38-fp8-overlay-loader.py")
SPEC = importlib.util.spec_from_file_location("qwen38_fp8_loader", BASE_PATH)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)
MARKER = "ROCKET_QWEN38_NVFP4_OVERLAY_V2"


def replace_exact(source: str, old: str, new: str, count: int, label: str) -> str:
    actual = source.count(old)
    if actual != count:
        raise ValueError(f"FP8 loader template drift: {label} count={actual}/{count}")
    return source.replace(old, new)


def nvfp4_helper() -> str:
    result = base.HELPER.replace("FP8", "NVFP4").replace("fp8", "nvfp4")
    result = replace_exact(
        result,
        '# ROCKET_QWEN38_NVFP4_OVERLAY_V1',
        '# ROCKET_QWEN38_NVFP4_OVERLAY_V2',
        1,
        "marker",
    )
    result = replace_exact(
        result,
        '_ROCKET_QWEN38_NVFP4_SCHEMA = "rocket.qwen38.linear-nvfp4-overlay.v1"',
        '_ROCKET_QWEN38_NVFP4_SCHEMAS = {\n'
        '    "rocket.qwen38.linear-nvfp4-overlay.v1",\n'
        '    "rocket.qwen38.nvfp4-overlay.v2",\n'
        '}',
        1,
        "manifest schemas",
    )
    result = replace_exact(
        result,
        '    r"^model\\.language_model\\.layers\\.(\\d+)\\.linear_attn\\."\n'
        '    r"(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)\\.weight$"',
        '    r"^model\\.language_model\\.layers\\.(\\d+)\\."\n'
        '    r"(linear_attn|self_attn|mlp|ple)\\."\n'
        '    r"(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj|q_proj|k_proj|v_proj|o_proj|gate|key_proj|value_proj)\\.weight$"',
        1,
        "selected family regex",
    )
    result = replace_exact(
        result,
        '    if manifest.get("schema") != _ROCKET_QWEN38_NVFP4_SCHEMA:\n'
        '        raise ValueError("Qwen3.8 NVFP4 overlay schema mismatch")',
        '    schema = manifest.get("schema")\n'
        '    if schema not in _ROCKET_QWEN38_NVFP4_SCHEMAS:\n'
        '        raise ValueError("Qwen3.8 NVFP4 overlay schema mismatch")',
        1,
        "schema validation",
    )
    result = replace_exact(
        result,
        '    if source.get("revision") != _ROCKET_QWEN38_REVISION:\n'
        '        raise ValueError("Qwen3.8 NVFP4 overlay revision mismatch")',
        '    if source.get("revision") != _ROCKET_QWEN38_REVISION:\n'
        '        raise ValueError("Qwen3.8 NVFP4 overlay revision mismatch")\n'
        '    families = source.get("families") if schema == "rocket.qwen38.nvfp4-overlay.v2" else ["linear_attention"]\n'
        '    valid_families = {"base_ple", "base_routers", "full_attention", "linear_attention"}\n'
        '    if (not isinstance(families, list) or not families or families != sorted(set(families))\n'
        '            or set(families) - valid_families):\n'
        '        raise ValueError("Qwen3.8 NVFP4 overlay family selection is invalid")',
        1,
        "family selection",
    )
    result = replace_exact(
        result,
        '    if not isinstance(entries, list) or len(entries) != 180:\n'
        '        raise ValueError("Qwen3.8 NVFP4 overlay requires exactly 180 source tensors")',
        '    expected_counts = {"linear_attention": 180, "full_attention": 48, "base_routers": 48, "base_ple": 2}\n'
        '    expected_count = sum(expected_counts[family] for family in families)\n'
        '    if not isinstance(entries, list) or len(entries) != expected_count:\n'
        '        raise ValueError(f"Qwen3.8 NVFP4 overlay requires exactly {expected_count} source tensors")',
        1,
        "dynamic tensor count",
    )
    result = replace_exact(
        result,
        '        selected.add(name)\n'
        '        layers.setdefault(int(match.group(1)), set()).add(match.group(2))',
        '        family = {"linear_attn": "linear_attention", "self_attn": "full_attention", "mlp": "base_routers", "ple": "base_ple"}[match.group(2)]\n'
        '        if family not in families:\n'
        '            raise ValueError(f"Qwen3.8 NVFP4 overlay tensor outside selected families: {name}")\n'
        '        selected.add(name)\n'
        '        layers.setdefault((family, int(match.group(1))), set()).add(match.group(3))',
        1,
        "family membership",
    )
    result = replace_exact(
        result,
        '    projections = {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}\n'
        '    if len(layers) != 36 or any(value != projections for value in layers.values()):\n'
        '        raise ValueError("Qwen3.8 NVFP4 overlay is a partial 36-layer family")',
        '    family_contracts = {\n'
        '        "linear_attention": (set(range(48)) - set(range(3, 48, 4)), {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}),\n'
        '        "full_attention": (set(range(3, 48, 4)), {"q_proj", "k_proj", "v_proj", "o_proj"}),\n'
        '        "base_routers": (set(range(48)), {"gate"}),\n'
        '        "base_ple": ({1}, {"key_proj", "value_proj"}),\n'
        '    }\n'
        '    for family in families:\n'
        '        expected_layers, projections = family_contracts[family]\n'
        '        actual_layers = {layer for candidate, layer in layers if candidate == family}\n'
        '        if actual_layers != expected_layers or any(layers[(family, layer)] != projections for layer in actual_layers):\n'
        '            raise ValueError(f"Qwen3.8 NVFP4 overlay is a partial {family} family")',
        1,
        "family completeness",
    )
    result = replace_exact(
        result,
        '    selected_prefixes = {name.removesuffix(".weight") for name in selected}\n'
        '    if any(quantized_layers.get(prefix, {}).get("quant_algo") != "NVFP4" for prefix in selected_prefixes):',
        '    selected_prefixes = {name.removesuffix(".weight") for name in selected}\n'
        '    selected_family_policies = {prefix for prefix in quantized_layers if _ROCKET_QWEN38_SELECTED.fullmatch(prefix + ".weight")}\n'
        '    if selected_family_policies != selected_prefixes:\n'
        '        raise ValueError("Qwen3.8 NVFP4 overlay family policy does not match selected tensors")\n'
        '    for family, (expected_layers, _) in family_contracts.items():\n'
        '        if family in families:\n'
        '            continue\n'
        '        module = {"linear_attention": "linear_attn", "full_attention": "self_attn", "base_routers": "mlp.gate", "base_ple": "ple"}[family]\n'
        '        if family == "base_ple":\n'
        '            continue\n'
        '        for layer in expected_layers:\n'
        '            prefix = f"model.language_model.layers.{layer}.{module}"\n'
        '            if not any(fnmatch.fnmatch(prefix, pattern) for pattern in excludes):\n'
        '                raise ValueError(f"Qwen3.8 NVFP4 overlay unselected {family} is not excluded")\n'
        '    if any(quantized_layers.get(prefix, {}).get("quant_algo") != "NVFP4" for prefix in selected_prefixes):',
        1,
        "exact policy selection",
    )
    result = replace_exact(
        result,
        'expected.update((name, prefix + ".weight_scale", prefix + ".input_scale"))',
        'expected.update((name, prefix + ".weight_scale", prefix + ".weight_scale_2", prefix + ".input_scale"))',
        1,
        "expected tensor set",
    )
    result = replace_exact(
        result,
        'contracts = {\n            name: ("F8_E4M3", entry["shape"]),\n            prefix + ".weight_scale": ("F32", [1]),\n            prefix + ".input_scale": ("F32", [1]),\n        }',
        'n, k = entry["shape"]\n        if k % 16:\n            raise ValueError(f"Qwen3.8 NVFP4 overlay K is not block-16: {name}")\n        contracts = {\n            name: ("U8", [n, k // 2]),\n            prefix + ".weight_scale": ("F8_E4M3", [n, k // 16]),\n            prefix + ".weight_scale_2": ("F32", [1]),\n            prefix + ".input_scale": ("F32", [1]),\n        }',
        1,
        "tensor ABI",
    )
    return result


def nvfp4_lazy_patch() -> str:
    result = base.LAZY_PATCH.replace("FP8", "NVFP4").replace("fp8", "nvfp4")
    return replace_exact(
        result,
        'name, prefix + ".weight_scale", prefix + ".input_scale"',
        'name, prefix + ".weight_scale", prefix + ".weight_scale_2", prefix + ".input_scale"',
        1,
        "auxiliary yields",
    )


def patched(source: str) -> str:
    if MARKER in source or "ROCKET_QWEN38_NVFP4_OVERLAY_V1" in source:
        raise ValueError("weight_utils source is already patched")
    if source.count(base.FUNCTION_ANCHOR) != 1:
        raise ValueError("pinned weight_utils source drift: iterator definition")
    helper = nvfp4_helper()
    result = source.replace(base.FUNCTION_ANCHOR, helper + base.FUNCTION_ANCHOR)
    sort_patch = base.SORT_PATCH.replace("FP8", "NVFP4").replace("fp8", "nvfp4")
    result = base.replace_once(result, base.SORT_ANCHOR, sort_patch, "sorted files")
    result = base.replace_once(result, base.LAZY_ANCHOR, nvfp4_lazy_patch(), "64 KiB clone")
    ast.parse(result)
    return result


def patch(path: Path) -> None:
    path.write_text(patched(path.read_text()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("weight_utils_py", type=Path)
    args = parser.parse_args()
    try:
        patch(args.weight_utils_py)
    except (OSError, ValueError) as error:
        print(f"error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
