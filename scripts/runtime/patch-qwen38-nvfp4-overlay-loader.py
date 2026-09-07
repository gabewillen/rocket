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
MARKER = "ROCKET_QWEN38_NVFP4_OVERLAY_V1"


def replace_exact(source: str, old: str, new: str, count: int, label: str) -> str:
    actual = source.count(old)
    if actual != count:
        raise ValueError(f"FP8 loader template drift: {label} count={actual}/{count}")
    return source.replace(old, new)


def nvfp4_helper() -> str:
    result = base.HELPER.replace("FP8", "NVFP4").replace("fp8", "nvfp4")
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
    if MARKER in source:
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
