#!/usr/bin/env python3
"""Embed Rocket's verified Qwen3.8 FP8 policy in the effective model config."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SELECTED = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.linear_attn\."
    r"(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)$"
)


def embed(base: dict, sidecar: dict) -> dict:
    current = base.get("quantization_config")
    overlay = sidecar.get("quantization")
    if not isinstance(current, dict) or current.get("quant_algo") != "MIXED_PRECISION":
        raise ValueError("base config lacks MIXED_PRECISION quantization_config")
    if not isinstance(overlay, dict) or overlay.get("quant_algo") != "MIXED_PRECISION":
        raise ValueError("overlay sidecar lacks MIXED_PRECISION quantization")
    layers = overlay.get("quantized_layers")
    excludes = overlay.get("exclude_modules")
    if not isinstance(layers, dict) or not isinstance(excludes, list):
        raise ValueError("overlay policy lacks quantized_layers or exclude_modules")
    selected = {key for key in layers if SELECTED.fullmatch(key)}
    if len(selected) != 180:
        raise ValueError(f"overlay selected projection coverage is {len(selected)}/180")
    if any(layers[key].get("quant_algo") != "FP8" for key in selected):
        raise ValueError("overlay selected projection is not FP8")
    if any("linear_attn" in value for value in excludes):
        raise ValueError("overlay still excludes linear attention")
    method = current.get("quant_method")
    if method != "modelopt":
        raise ValueError(f"base quant_method is {method!r}, expected 'modelopt'")
    result = json.loads(json.dumps(base))
    effective = json.loads(json.dumps(overlay))
    effective["quant_method"] = method
    result["quantization_config"] = effective
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_config", type=Path)
    parser.add_argument("overlay_sidecar", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        value = embed(
            json.loads(args.base_config.read_text()),
            json.loads(args.overlay_sidecar.read_text()),
        )
        args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
