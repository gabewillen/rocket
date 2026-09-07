#!/usr/bin/env python3
"""Create a complete mixed-precision ModelOpt policy for Qwen attention NVFP4."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


LINEAR_EXCLUDE = re.compile(r"^model\.language_model\.layers\.(\d+)\.linear_attn\*$")
FULL_EXCLUDE = re.compile(r"^model\.language_model\.layers\.(\d+)\.self_attn\*$")
FAMILY_PROJECTIONS = {
    "linear_attention": ("linear_attn", ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")),
    "full_attention": ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
}
FAMILY_EXCLUDES = {
    "linear_attention": (LINEAR_EXCLUDE, set(range(48)) - set(range(3, 48, 4))),
    "full_attention": (FULL_EXCLUDE, set(range(3, 48, 4))),
}


def normalize_families(families) -> tuple[str, ...]:
    """Return the canonical non-empty family selection or raise ValueError."""
    selected = tuple(sorted(set(families)))
    unknown = set(selected) - set(FAMILY_PROJECTIONS)
    if not selected or unknown:
        raise ValueError(f"invalid NVFP4 families: {sorted(unknown) if unknown else selected}")
    return selected


def patched(config: dict, families=("linear_attention",)) -> dict:
    """Copy and extend a pinned config; input is borrowed and never mutated."""
    families = normalize_families(families)
    result = json.loads(json.dumps(config))
    quant = result.get("quantization")
    if not isinstance(quant, dict) or quant.get("quant_algo") != "MIXED_PRECISION":
        raise ValueError("pinned config is not ModelOpt MIXED_PRECISION")
    if quant.get("group_size") != 16:
        raise ValueError("pinned config is not ModelOpt group-size-16")
    excludes, layers = quant.get("exclude_modules"), quant.get("quantized_layers")
    if not isinstance(excludes, list) or not isinstance(layers, dict):
        raise ValueError("pinned config lacks excludes or quantized_layers")
    selected = {family: [] for family in families}
    kept = []
    for value in excludes:
        matched_family = None
        if isinstance(value, str):
            for family in families:
                match = FAMILY_EXCLUDES[family][0].fullmatch(value)
                if match:
                    selected[family].append(int(match.group(1)))
                    matched_family = family
                    break
        if matched_family is None:
            kept.append(value)
    for family in families:
        actual = selected[family]
        expected = FAMILY_EXCLUDES[family][1]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError(
                f"{family} exclude coverage is {len(set(actual))}/{len(expected)}"
            )
        module, projections = FAMILY_PROJECTIONS[family]
        for layer in sorted(actual):
            for projection in projections:
                prefix = f"model.language_model.layers.{layer}.{module}.{projection}"
                if prefix in layers:
                    raise ValueError(f"quantized layer already exists: {prefix}")
                layers[prefix] = {"quant_algo": "NVFP4"}
    quant["exclude_modules"] = kept
    return result


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--family",
        action="append",
        choices=tuple(FAMILY_PROJECTIONS),
        default=None,
        help="attention family to enable; repeat to create a combined policy",
    )
    args = parser.parse_args()
    try:
        encoded = canonical_bytes(
            patched(json.loads(args.source.read_text()), args.family or ("linear_attention",))
        )
        with args.output.open("xb") as stream:
            stream.write(encoded)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(hashlib.sha256(encoded).hexdigest(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
