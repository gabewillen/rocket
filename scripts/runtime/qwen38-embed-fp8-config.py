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
FULL_SELECTED = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.self_attn\."
    r"(q_proj|k_proj|v_proj|o_proj)$"
)
FAMILY_PATTERNS = {
    "linear_attention": (SELECTED, 180, "linear_attn"),
    "full_attention": (FULL_SELECTED, 48, "self_attn"),
}
FAMILY_LAYERS = {
    "linear_attention": set(range(48)) - set(range(3, 48, 4)),
    "full_attention": set(range(3, 48, 4)),
}


def embed(
    base: dict,
    sidecar: dict,
    expected_algo: str = "FP8",
    families=("linear_attention",),
) -> dict:
    """Return a copied effective config after complete selected-family validation."""
    families = tuple(sorted(set(families)))
    if not families or set(families) - set(FAMILY_PATTERNS):
        raise ValueError(f"invalid overlay families: {families}")
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
    selected = set()
    for family in families:
        pattern, expected, module = FAMILY_PATTERNS[family]
        family_selected = {key for key in layers if pattern.fullmatch(key)}
        if len(family_selected) != expected:
            raise ValueError(
                f"overlay {family} projection coverage is {len(family_selected)}/{expected}"
            )
        if any(module in value for value in excludes):
            raise ValueError(f"overlay still excludes {family}")
        selected.update(family_selected)
    attention_selected = {
        key
        for key in layers
        if any(pattern.fullmatch(key) for pattern, _, _ in FAMILY_PATTERNS.values())
    }
    if attention_selected != selected:
        raise ValueError("overlay attention policy does not match selected families")
    for family in set(FAMILY_PATTERNS) - set(families):
        _, _, module = FAMILY_PATTERNS[family]
        for layer in FAMILY_LAYERS[family]:
            prefix = f"model.language_model.layers.{layer}.{module}"
            if not any(re.fullmatch(re.escape(prefix) + r"\*", value) for value in excludes):
                raise ValueError(f"overlay unselected {family} is not fully excluded")
    if expected_algo not in ("FP8", "NVFP4"):
        raise ValueError(f"unsupported overlay quantization algorithm: {expected_algo}")
    if any(layers[key].get("quant_algo") != expected_algo for key in selected):
        raise ValueError(f"overlay selected projection is not {expected_algo}")
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
    parser.add_argument("--quant-algo", choices=("FP8", "NVFP4"), default="FP8")
    parser.add_argument(
        "--family",
        action="append",
        choices=tuple(FAMILY_PATTERNS),
        default=None,
        help="overlay family to require; repeat for a combined policy",
    )
    args = parser.parse_args()
    try:
        value = embed(
            json.loads(args.base_config.read_text()),
            json.loads(args.overlay_sidecar.read_text()),
            args.quant_algo,
            args.family or ("linear_attention",),
        )
        args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
