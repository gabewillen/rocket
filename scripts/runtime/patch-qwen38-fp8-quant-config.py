#!/usr/bin/env python3
"""Create the minimal read-only ModelOpt config for Qwen3.8 linear FP8."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
LINEAR_EXCLUDE = re.compile(r"^model\.language_model\.layers\.(\d+)\.linear_attn\*$")
PROJECTIONS = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")


def patched(config: dict) -> dict:
    result = json.loads(json.dumps(config))
    quant = result.get("quantization")
    if not isinstance(quant, dict) or quant.get("quant_algo") != "MIXED_PRECISION":
        raise ValueError("pinned config is not ModelOpt MIXED_PRECISION")
    excludes = quant.get("exclude_modules")
    layers = quant.get("quantized_layers")
    if not isinstance(excludes, list) or not isinstance(layers, dict):
        raise ValueError("pinned config lacks excludes or quantized_layers")
    selected_layers = []
    kept = []
    for value in excludes:
        match = LINEAR_EXCLUDE.fullmatch(value) if isinstance(value, str) else None
        if match:
            selected_layers.append(int(match.group(1)))
        else:
            kept.append(value)
    if len(selected_layers) != 36 or len(set(selected_layers)) != 36:
        raise ValueError(f"linear-attention exclude coverage is {len(set(selected_layers))}/36")
    for layer in sorted(selected_layers):
        for projection in PROJECTIONS:
            prefix = f"model.language_model.layers.{layer}.linear_attn.{projection}"
            if prefix in layers:
                raise ValueError(f"quantized layer already exists: {prefix}")
            layers[prefix] = {"quant_algo": "FP8"}
    quant["exclude_modules"] = kept
    return result


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        value = patched(json.loads(args.source.read_text()))
        encoded = canonical_bytes(value)
        with args.output.open("xb") as stream:
            stream.write(encoded)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(hashlib.sha256(encoded).hexdigest(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
