#!/usr/bin/env python3
"""Reduce Rocket Qwen3.8 calibration log lines into a checked JSON map."""

import argparse
import json
import re
import sys

LINE = re.compile(r"ROCKET_NVFP4_CALIBRATION\t(layer\.(\d+)\.linear_attn\.(in_proj_qkvz|in_proj_ba|out_proj))\t([0-9.eE+-]+)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=36)
    args = parser.parse_args()
    maxima = {}
    for line in sys.stdin:
        match = LINE.search(line)
        if not match:
            continue
        name, _, _, raw_value = match.groups()
        value = float(raw_value)
        maxima[name] = max(value, maxima.get(name, 0.0))
    by_layer = {}
    for name, value in sorted(maxima.items()):
        _, layer, _, projection = name.split(".")
        by_layer.setdefault(layer, {})[projection] = value
    complete = [layer for layer, values in by_layer.items() if len(values) == 3]
    if len(complete) != args.layers or len(maxima) != args.layers * 3:
        print(f"incomplete calibration: {len(complete)}/{args.layers} layers, "
              f"{len(maxima)}/{args.layers * 3} channels", file=sys.stderr)
        raise SystemExit(1)
    json.dump({"schema": "rocket.qwen38.activation-maxima.v1",
               "linear_attention_layers": len(complete), "channels": maxima},
              sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
